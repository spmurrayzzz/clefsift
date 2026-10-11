import argparse
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import stat
import subprocess
import sys
import threading
import time
import uuid


REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "./corpus.json"
RUNTIME_FILES = ("control_corpus.py", "corpus.py", "common.py", "pi_client.py",
                 "pi-generation.ts", "general-recipe-compatibility.json")
TERMINAL_STATES = {"completed", "completed-with-shortfalls", "stopped", "failure"}
IDENTITY_KEYS = ("pid", "birth_ticks", "boot_id", "uid", "pgid", "sid", "argv", "cwd")
COUNTERS = {"accepted", "reused", "invalid", "errors", "pending", "exhausted", "calls",
            "attempts", "completed", "active", "in_flight", "workers", "submitted", "selected_tasks",
            "requests", "requests_used", "limit", "plan_attempts_used", "plan_attempts_limit", "attempts_used"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def epoch(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0, "Use a UTC timestamp with Z or +00:00")
    return parsed.timestamp()


def strict_json(text):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate JSON key")
            value[key] = item
        return value

    def invalid(value):
        raise ValueError("Nonfinite JSON number")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read_json(path, limit=1024 * 1024):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), f"Not a regular file: {path}")
        data = stream.read(limit + 1)
    require(len(data) <= limit, f"JSON file too large: {path}")
    value = strict_json(data)
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def write_json(path, value, exclusive=False):
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def work_identity(work):
    info = work.lstat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o022,
            f"Directory must be owned by this user and not writable by other users: {work}")
    return info.st_dev, info.st_ino, info.st_uid


def configuration(path, validate=True):
    path = Path(path).expanduser().absolute()
    try:
        supplied = read_json(path)
        path = path.resolve()
        value = supplied.get("work_dir")
        require(isinstance(value, str) and value.strip() and "\0" not in value, "Set work_dir to a local path")
        work = (path.parent / Path(value).expanduser()).resolve()
        identity = work_identity(work)
        config = {"work_dir": str(work)}
        workers = supplied.get("concurrency")
        if not validate:
            config["concurrency"] = {route: count for route, count in workers.items() if isinstance(route, str) and type(count) is int and count > 0} if isinstance(workers, dict) else {}
            return path, config, work, identity
        allowed = {"work_dir", "pi_bin_dir", "concurrency", "request_timeout", "generation_hours", "max_attempts", "deadline_utc",
                   "import_dir", "python", "controller_python", "model", "concurrency_approval", "concurrency_approval_sha256"}
        require(not set(supplied) - allowed, "Unknown config fields: " + ", ".join(sorted(set(supplied) - allowed)))
        if "pi_bin_dir" in supplied:
            value = supplied["pi_bin_dir"]
            require(isinstance(value, str) and value.strip() and "\0" not in value, "Set pi_bin_dir to a local path")
            config["pi_bin_dir"] = str((path.parent / Path(value).expanduser()).resolve())
        require(isinstance(workers, dict) and workers
                and all(isinstance(route, str) and re.fullmatch(r"[^/\s]+/\S+", route)
                        and not any(ord(c) < 32 or ord(c) == 127 for c in route)
                        and all(not part.startswith("-") for part in route.split("/", 1))
                        and type(count) is int and count > 0 for route, count in workers.items()),
                "Set concurrency to a map of full Pi provider/model IDs and positive integer worker counts")
        config["concurrency"] = workers
        for key, default in (("request_timeout", 120), ("generation_hours", None)):
            value = supplied.get(key, default)
            if value is None and key not in supplied:
                continue
            require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                    f"{key} must be a finite positive number")
            require(key != "request_timeout" or value <= 600, "request_timeout must be at most 600 seconds")
            config[key] = value
        if "max_attempts" in supplied:
            require(type(supplied["max_attempts"]) is int and supplied["max_attempts"] > 0,
                    "max_attempts must be a positive integer cumulative limit")
            config["max_attempts"] = supplied["max_attempts"]
        if "deadline_utc" in supplied:
            value = supplied["deadline_utc"]
            require(isinstance(value, str), "deadline_utc must be a UTC timestamp")
            epoch(value)
            config["deadline_utc"] = value
        return path, config, work, identity
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Config {path}: {exc}") from None


@contextmanager
def file_lock(path, create=True):
    flags = os.O_NOFOLLOW | os.O_NONBLOCK | (os.O_RDWR | os.O_CREAT if create else os.O_RDONLY)
    fd = os.open(path, flags, 0o600)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1 and not info.st_mode & 0o022,
                f"Unsafe lock file: {path}")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def build_busy(work):
    try:
        with file_lock(work / ".build-corpus.lock", create=False):
            return False
    except FileNotFoundError:
        return False
    except BlockingIOError:
        return True


def plan_metadata(work):
    path = work / "generation/plan.json"
    try:
        plan = read_json(path)
        require(plan.get("workflow") == "clefsift-general-v1" and plan.get("stage") == "generation-plan"
                and plan.get("version") == 1, "Expected an existing general corpus generation plan")
        require(type(plan.get("max_attempts")) is int and plan["max_attempts"] > 0,
                "Plan max_attempts must be a positive integer")
        return plan, digest(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Generation plan {path}: {exc}. Check work_dir in the config") from None


def deadline(work, config, plan_hash, create=False):
    path = work / "generation-clock.json"
    existing = path.exists() or path.is_symlink()
    try:
        if existing:
            clock = read_json(path)
            require(all(isinstance(clock.get(key), str) for key in ("first_start_utc", "deadline_utc", "plan_sha256")),
                    "Clock requires first_start_utc, deadline_utc, and plan_sha256")
            first, end = epoch(clock["first_start_utc"]), epoch(clock["deadline_utc"])
            require(first <= time.time() and end > first, "Invalid generation clock timestamps")
            require(clock.get("plan_sha256") == plan_hash, "Clock plan_sha256 does not match generation/plan.json")
        else:
            require("generation_hours" in config, "Set generation_hours in the config to create the first generation clock")
            require(not (work / "generation/attempts").exists() and not any((work / "generation").glob("run.*.json")),
                    "Generation records exist without a clock; restore the original clock instead of resetting its deadline")
            first = time.time()
            end = first + config["generation_hours"] * 3600
            clock = {"first_start_utc": datetime.fromtimestamp(first, timezone.utc).isoformat(),
                     "deadline_utc": datetime.fromtimestamp(end, timezone.utc).isoformat(), "plan_sha256": plan_hash}
        if "deadline_utc" in config:
            requested = epoch(config["deadline_utc"])
            require(requested <= end, "Config deadline_utc cannot extend the generation clock")
            end = requested
        if create:
            require(end > time.time(), "Generation deadline has expired; the existing clock cannot be reset")
            if not existing:
                clock["deadline_utc"] = datetime.fromtimestamp(end, timezone.utc).isoformat()
                write_json(path, clock, exclusive=True)
        return {"deadline_utc": datetime.fromtimestamp(end, timezone.utc).isoformat(),
                "remaining_seconds": max(0, end - time.time()), "clock": "generation" if existing or create else "not-started"}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError(f"Generation clock {path}: {exc}") from None


def launch_environment(config):
    require(sys.platform == "linux" and Path("/proc/self/stat").exists(), "Start requires Linux /proc process identities")
    require(sys.version_info >= (3, 11), f"Use Python 3.11 or newer for tools/corpus; current Python is {sys.executable}")
    env = {key: value for key, value in os.environ.items() if key not in {"VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONHOME", "PYTHONPATH"}}
    if "pi_bin_dir" in config:
        directory = Path(config["pi_bin_dir"])
        require(directory.is_dir(), f"Config pi_bin_dir does not exist: {directory}")
        require(all((directory / name).is_file() and os.access(directory / name, os.X_OK) for name in ("node", "pi")),
                f"Config pi_bin_dir must contain executable node and pi files: {directory}")
        env["PATH"] = str(directory) + os.pathsep + env.get("PATH", os.defpath)
    for name in ("uv", "node", "pi"):
        require(shutil.which(name, path=env.get("PATH")) is not None,
                f"Cannot find {name}; check PATH" + (" or pi_bin_dir in the config" if name != "uv" else ""))
    work = Path(config["work_dir"])
    for directory in (work / "tmp", work / "cache", work / "cache/uv"):
        directory.mkdir(mode=0o700, exist_ok=True)
        work_identity(directory)
    env.update(TMPDIR=str(work / "tmp"), UV_CACHE_DIR=str(work / "cache/uv"),
               PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", UV_OFFLINE="1",
               UV_PYTHON_DOWNLOADS="never", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.setdefault("CLEFSIFT_LOCAL_UNAUTHENTICATED", "1")
    env.setdefault("CLEFSIFT_STRUCTURED_GENERATION", "1")
    return env


def proc_stat(pid):
    value = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"state": value[0], "pgid": int(value[2]), "sid": int(value[3]), "birth_ticks": int(value[19])}


def process_identity(pid):
    require(sys.platform == "linux", "Process identity requires Linux /proc")
    before = proc_stat(pid)
    if before["state"] in {"Z", "X", "x"}:
        return None
    root = Path(f"/proc/{pid}")
    uid_line = next(line for line in (root / "status").read_text().splitlines() if line.startswith("Uid:"))
    uids = [int(value) for value in uid_line.split()[1:]]
    require(len(set(uids)) == 1, "Process user identity changed")
    with (root / "cmdline").open("rb") as stream:
        raw = stream.read(131073)
    require(len(raw) <= 131072 and raw.endswith(b"\0"), "Invalid process command line")
    identity = {"pid": pid, "birth_ticks": before["birth_ticks"], "uid": uids[0],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "pgid": before["pgid"], "sid": before["sid"], "argv": [os.fsdecode(v) for v in raw[:-1].split(b"\0")],
                "cwd": os.readlink(root / "cwd")}
    after = proc_stat(pid)
    require(all(before[key] == after[key] for key in ("birth_ticks", "pgid", "sid")), "Process identity changed during inspection")
    return None if after["state"] in {"Z", "X", "x"} else identity


def startup_identity(process):
    until = time.monotonic() + 5
    while True:
        require(process.poll() is None, "Controller exited before its identity could be recorded")
        try:
            identity = process_identity(process.pid)
            if identity is not None:
                return identity
        except FileNotFoundError:
            pass
        except ValueError as exc:
            if str(exc) != "Invalid process command line":
                raise
        require(time.monotonic() < until, "Timed out reading new controller identity")
        time.sleep(0.05)


def process_state(record):
    identity = record.get("identity")
    if identity is None:
        return "exited" if record.get("state") == "launch-failed" or (record.get("state") == "exited" and type(record.get("exit_code")) is int) else "unverified"
    if sys.platform != "linux":
        return "unverified"
    try:
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != identity["boot_id"]:
            return "stale"
        current = proc_stat(identity["pid"])
        if current["birth_ticks"] != identity["birth_ticks"]:
            return "stale"
        if current["state"] in {"Z", "X", "x"}:
            return "exited"
        actual = process_identity(identity["pid"])
        if actual is None:
            return "exited"
        return "alive" if all(actual[key] == identity[key] for key in IDENTITY_KEYS) else "identity-mismatch"
    except FileNotFoundError:
        return "exited" if not Path(f"/proc/{identity['pid']}").exists() else "unverified"
    except (OSError, ValueError, KeyError, IndexError, StopIteration):
        return "unverified"


def active_record(work):
    path = work / "control-active.json"
    if not path.exists() and not path.is_symlink():
        return None
    record = read_json(path)
    name = record.get("run_id", "")
    require(isinstance(name, str) and re.fullmatch(r"build-\d{8}T\d{12}Z-[0-9a-f]{12}", name), f"Invalid run_id in {path}")
    require(record.get("work_dir") == str(work), f"work_dir differs in {path}")
    repo = record.get("repo")
    require(isinstance(repo, str) and Path(repo).is_absolute() and ".." not in Path(repo).parts, f"Invalid repo in {path}")
    for key, expected in (("out_dir", work / name), ("log_path", work / (name + ".log")),
                          ("stop_file", work / name / "stop.request.json")):
        require(record.get(key) == str(expected) and not expected.is_symlink(), f"Invalid {key} in {path}")
    require(record.get("version") in (None, 2), f"Unknown active record version in {path}")
    runtime = work / name / "code" if record.get("version") == 2 else Path(repo)
    argv = record.get("argv")
    require(isinstance(argv, list) and argv and all(isinstance(item, str) for item in argv), f"Invalid argv in {path}")
    if record.get("version") == 2:
        require(record.get("runtime_repo") == str(runtime) and not runtime.is_symlink(), f"Invalid snapshot path in {path}")
        require(argv[-4:] == [str(runtime / "finetune/control_corpus.py"), "_run", "--run-dir", str(work / name)],
                f"Invalid snapshot command in {path}")
    else:
        require(str(runtime / "finetune/build_corpus.py") in argv, f"Invalid legacy controller command in {path}")
        for flag, value in (("--work-dir", str(work)), ("--out-dir", str(work / name)), ("--repo", repo)):
            require(flag in argv and argv[argv.index(flag) + 1:argv.index(flag) + 2] == [value], f"Invalid legacy {flag} in {path}")
    identity = record.get("identity")
    if identity is not None:
        require(isinstance(identity, dict) and set(IDENTITY_KEYS) <= set(identity), f"Incomplete process identity in {path}")
        require(all(type(identity[key]) is int and identity[key] > 0 for key in ("pid", "birth_ticks", "pgid", "sid")),
                f"Invalid process identity in {path}")
        require(identity["uid"] == os.getuid() and identity["cwd"] == str(runtime) and identity["argv"] == argv
                and identity["pid"] == identity["pgid"] == identity["sid"] and record.get("start_new_session") is True,
                f"Invalid controller owner, command, or session in {path}")
    return record


def persist(work, record, state):
    record["state"] = state
    record["updated_at_utc"] = stamp()
    write_json(work / "control-active.json", record)


def phase_status(record):
    path = Path(record["out_dir"]) / "status.json"
    try:
        return read_json(path)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        return {"status_error": f"Cannot read {path}: {exc}"}


def numeric_counts(value):
    result = {key: count for key, count in value.items() if key in COUNTERS and type(count) is int and count >= 0}
    if "errors" not in result and type(value.get("error")) is int and value["error"] >= 0:
        result["errors"] = value["error"]
    return result


def summary_state(value):
    if value.get("state") in TERMINAL_STATES:
        return value["state"]
    reasons = set(value.get("stop_reasons", []))
    if reasons - {"drained", "cancelled", "stopped"}:
        return "failure"
    if value.get("pending") == 0 and value.get("exhausted", 0) > 0:
        return "completed-with-shortfalls"
    if reasons or value.get("drained"):
        return "stopped"
    return "completed" if value.get("pending") == 0 else "failure"


def progress_item(value, path, updated, final=False):
    per_route = {}
    for key in ("route_counts", "per_route", "routes", "route_stats"):
        if isinstance(value.get(key), dict):
            per_route.update({route: numeric_counts(counts) for route, counts in value[key].items() if isinstance(counts, dict)})
    return {"stage": value.get("stage"), "loaded": value.get("loaded"), "tasks": value.get("tasks"),
            "counts": numeric_counts(value), "per_route": per_route, "disabled_routes": value.get("disabled_routes", {}),
            "stop_reasons": value.get("stop_reasons", []), "path": str(path), "final": final,
            "state": summary_state(value) if final else None, "age_seconds": max(0, round(time.time() - updated))}


def progress(record):
    run = Path(record["out_dir"])
    latest = {}
    paths = list(run.glob("*.development.stdout")) + list(run.glob("*.final_holdout.stdout"))
    for path in sorted(paths, reverse=True):
        split = "final_holdout" if path.name.endswith(".final_holdout.stdout") else "development"
        if split in latest:
            continue
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode):
                    continue
                offset = max(0, info.st_size - 131072)
                stream.seek(offset)
                lines = stream.read(131072).splitlines()
            if offset and lines:
                lines = lines[1:]
            for index in range(len(lines) - 1, -1, -1):
                try:
                    value = strict_json(b"\n".join(lines[index:]) if lines[index] == b"{" else lines[index])
                except (ValueError, UnicodeError):
                    continue
                if not isinstance(value, dict) or value.get("stage") not in {
                    "resume-loading-plan", "resume-loading-results", "generation-progress", "generation-run"
                }:
                    continue
                latest[split] = progress_item(value, path, info.st_mtime, value["stage"] == "generation-run")
                break
        except OSError:
            continue
    return latest


def status(config_path, config, work, record=None):
    record = active_record(work) if record is None else record
    busy = build_busy(work)
    result = {"config": str(config_path), "work_dir": str(work), "concurrency": config["concurrency"],
              "configured_workers": sum(config["concurrency"].values()), "build_lock_held": busy}
    if record is None:
        result.update(state="unmanaged" if busy else "stopped", process_state="not-recorded", pid=None)
    else:
        state = process_state(record)
        phase = phase_status(record)
        summary = None
        if record.get("version") == 2:
            path = Path(record["out_dir"]) / "summary.json"
            try:
                summary = read_json(path)
            except FileNotFoundError:
                pass
            except (OSError, ValueError) as exc:
                result["summary_error"] = f"Cannot read {path}: {exc}"
            value = summary if summary is not None else phase
            updated = epoch(value["updated_at"]) if value.get("updated_at") else time.time()
            items = {record["split"]: progress_item(value, path if summary is not None else path.with_name("status.json"),
                                                  updated, summary is not None)}
        else:
            items = progress(record)
        reason = phase.get("reason")
        if summary is not None:
            label = summary_state(summary)
            reason = summary.get("reason") or ", ".join(summary.get("stop_reasons", []))
            result["summary"] = summary
        elif phase.get("stage") in {"paused", "stopped", "ready-for-training"} and record.get("version") != 2:
            label = {"paused": "stopped", "stopped": "failure", "ready-for-training": "completed"}[phase["stage"]]
            latest = next(iter(items.values()), {})
            if latest.get("final") and latest.get("state") == "completed-with-shortfalls":
                label = "completed-with-shortfalls"
        elif state == "alive":
            loading = not phase or phase.get("stage") == "loading" or any(str(item["stage"]).startswith("resume-loading") for item in items.values())
            label = "draining" if Path(record["stop_file"]).exists() else "loading" if loading else "generating"
        else:
            label = "failure"
            reason = f"Controller exited without a terminal summary; inspect {record['log_path']}"
        if state not in {"alive", "exited", "stale"}:
            label = state
        elif state != "alive" and busy:
            label = "unmanaged"
        result.update(state=label, reason=reason, process_state=state, pid=record["identity"]["pid"] if record.get("identity") else None,
                      registry_state=record["state"], run_dir=record["out_dir"], log_path=record["log_path"], stop_file=record["stop_file"],
                      launch_concurrency=record["route_workers"], launched_workers=sum(record["route_workers"].values()),
                      phase_status=phase, progress=items)
    try:
        _, plan_hash = plan_metadata(work)
        safety_config = config
        if record and record.get("deadline_utc"):
            safety_config = {**config, "deadline_utc": record["deadline_utc"]}
        result["safety"] = deadline(work, safety_config, plan_hash)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["safety"] = {"error": str(exc)}
    return result


def snapshot(run, config_path, config, request):
    directory = run / "code/finetune"
    directory.mkdir(parents=True, mode=0o700)
    hashes = {}
    for name in RUNTIME_FILES:
        source, destination = REPO / "finetune" / name, directory / name
        with source.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        hashes["finetune/" + name] = digest(destination)
    write_json(run / "config.json", config, exclusive=True)
    write_json(run / "request.json", {**request, "config": str(config_path), "repo": str(REPO),
               "config_sha256": digest(run / "config.json"), "code_sha256": hashes,
               "python": sys.executable, "created_at_utc": stamp()}, exclusive=True)


def start(config_path, config, work, owner, construct_final=False):
    with file_lock(work / ".corpus-control.lock"):
        require(work_identity(work) == owner, "Work directory identity changed")
        record = active_record(work)
        if record:
            state = process_state(record)
            require(state != "alive", f"Controller already active; inspect {record['log_path']}")
            require(state in {"exited", "stale"}, f"Cannot verify previous launch; inspect {work / 'control-active.json'}")
        env = launch_environment(config)
        name = "build-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:12]
        run = work / name
        runtime = run / "code"
        split = "final_holdout" if construct_final else "development"
        argv = [str(Path(shutil.which("uv", path=env.get("PATH"))).absolute()), "run", "--offline", "--no-project",
                "--python", sys.executable, "python", str(runtime / "finetune/control_corpus.py"), "_run", "--run-dir", str(run)]
        with file_lock(work / ".build-corpus.lock"):
            require(work_identity(work) == owner, "Work directory identity changed")
            require(configuration(config_path)[1] == config, f"Config changed during launch; retry with {config_path}")
            plan, plan_hash = plan_metadata(work)
            safety = deadline(work, config, plan_hash)
            require(safety["remaining_seconds"] > 0, f"Generation deadline in {work / 'generation-clock.json'} has expired")
            effective = min(config.get("max_attempts", plan["max_attempts"]), plan["max_attempts"])
            run.mkdir(mode=0o700)
            safety = deadline(work, config, plan_hash, create=True)
            snapshot(run, config_path, {**config, "max_attempts": effective},
                     {"plan_sha256": plan_hash, "split": split, "construct_final": construct_final,
                      "deadline_utc": safety["deadline_utc"]})
            record = {"version": 2, "run_id": name, "work_dir": str(work), "repo": str(REPO), "runtime_repo": str(runtime),
                      "out_dir": str(run), "stop_file": str(run / "stop.request.json"), "log_path": str(work / (name + ".log")),
                      "config": str(config_path), "route_workers": config["concurrency"], "split": split,
                      "deadline_utc": safety["deadline_utc"], "argv": argv, "identity": None,
                      "start_new_session": True, "created_at_utc": stamp()}
            log_fd = os.open(record["log_path"], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                persist(work, record, "launching")
            except BaseException:
                os.close(log_fd)
                raise
        process = None
        try:
            with os.fdopen(log_fd, "wb") as log:
                process = subprocess.Popen(argv, cwd=runtime, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            identity = startup_identity(process)
            require(identity is not None and identity["uid"] == os.getuid() and identity["argv"] == argv
                    and identity["cwd"] == str(runtime) and identity["pid"] == identity["pgid"] == identity["sid"],
                    "Cannot verify new controller identity")
            record["identity"] = identity
            persist(work, record, "starting")
            until = time.monotonic() + 15
            while time.monotonic() < until:
                if process.poll() is not None:
                    record["exit_code"] = process.returncode
                    persist(work, record, "exited")
                    result = status(config_path, config, work, record)
                    require("summary" in result, f"Controller exited during startup; inspect {record['log_path']}")
                    result["action"] = "generation finished"
                    return result
                state = process_state(record)
                if state == "unverified":
                    time.sleep(0.1)
                    continue
                require(state == "alive", "Controller identity changed during startup")
                phase = phase_status(record)
                if phase.get("stage") in {"loading", "generating", "draining"} | TERMINAL_STATES:
                    persist(work, record, "running")
                    result = status(config_path, config, work, record)
                    result["action"] = "started"
                    return result
                time.sleep(0.1)
            persist(work, record, "starting-unconfirmed")
            raise ValueError(f"Controller is alive but startup is not confirmed; inspect {record['log_path']}. Do not launch another controller")
        except BaseException:
            if process is None:
                persist(work, record, "launch-failed")
            elif process.poll() is not None and record["state"] != "exited":
                record["exit_code"] = process.returncode
                persist(work, record, "exited")
            elif record.get("identity") is None:
                record["unverified_pid"] = process.pid
                persist(work, record, "unverified")
            raise


class GenerationOutput:
    def __init__(self, run, split):
        self.run, self.stream, self.buffer = run, sys.stdout, ""
        self.mutex = threading.Lock()
        self.value = {"stage": "loading", "split": split}

    def publish(self):
        self.value["updated_at"] = stamp()
        write_json(self.run / "status.json", self.value)

    def write(self, text):
        with self.mutex:
            self.stream.write(text)
            self.buffer += text
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                try:
                    event = strict_json(line)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                stage = event.get("stage", "")
                if stage in {"resume-loading-plan", "resume-loading-results", "generation-progress"}:
                    self.value.update(event)
                    loading = stage.startswith("resume-loading") and ("tasks" not in event or event.get("loaded") != event["tasks"])
                    self.value["stage"] = "loading" if loading else "generating"
                elif stage == "generation-route-disabled":
                    self.value.setdefault("disabled_routes", {})[event["route"]] = {
                        key: event[key] for key in ("reason", "error", "error_detail") if key in event}
                elif stage == "generation-request-failure":
                    self.value["last_error"] = event
                else:
                    continue
                if (self.run / "stop.request.json").exists():
                    self.value["stage"] = "draining"
                self.publish()
        return len(text)

    def flush(self):
        self.stream.flush()


def run_generation(run):
    run = run.expanduser().resolve()
    require(Path(__file__).resolve() == run / "code/finetune/control_corpus.py",
            "_run is private; use tools/corpus start to create a runtime snapshot")
    output = GenerationOutput(run, "development")
    claimed = False
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def interrupted(signum, frame):
        raise InterruptedError(f"Generation stopped by {signal.Signals(signum).name}")

    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        _, config, work, owner = configuration(run / "config.json")
        request = read_json(run / "request.json")
        final = request.get("construct_final")
        require(type(final) is bool and request.get("split") == ("final_holdout" if final else "development"),
                f"Invalid split or explicit final-construction flag in {run / 'request.json'}")
        output.value["split"] = request["split"]
        with file_lock(work / ".build-corpus.lock"):
            require(work_identity(work) == owner and run.parent == work, "Run or work directory identity changed")
            require(not any((run / name).exists() or (run / name).is_symlink() for name in ("status.json", "summary.json")),
                    f"Run already has child state: {run}; use tools/corpus start for a new launch")
            claimed = True
            output.publish()
            plan, plan_hash = plan_metadata(work)
            require(request.get("plan_sha256") == plan_hash, f"Generation plan changed after launch: {work / 'generation/plan.json'}")
            import corpus
            safety = deadline(work, {**config, "deadline_utc": request["deadline_utc"]}, plan_hash)
            minutes = safety["remaining_seconds"] / 60
            require(minutes > 0, f"Generation deadline has expired: {work / 'generation-clock.json'}")
            cap = min(config["max_attempts"], plan["max_attempts"])
            args = argparse.Namespace(plan_dir=work / "generation", split=request["split"], construct_final=final,
                                      route_workers=config["concurrency"], workers=sum(config["concurrency"].values()),
                                      timeout=config["request_timeout"], max_calls=cap, total_attempts=cap,
                                      minutes=minutes, stop_file=run / "stop.request.json")
            with redirect_stdout(output):
                summary = corpus.run(args)
    except (KeyboardInterrupt, InterruptedError) as exc:
        if not claimed:
            raise
        summary = {"state": "stopped", "reason": str(exc) or "Generation interrupted", "counts_available": False}
    except Exception as exc:
        if not claimed:
            raise
        summary = {"state": "failure", "reason": f"{type(exc).__name__}: {exc}", "counts_available": False}
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    summary = {"disabled_routes": output.value.get("disabled_routes", {}), **summary,
               "state": summary_state(summary), "split": output.value["split"], "updated_at": stamp()}
    write_json(run / "summary.json", summary, exclusive=True)
    write_json(run / "status.json", {**summary, "stage": summary["state"]})
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return 0 if summary["state"] in {"completed", "stopped"} else 1


def pidfd_call(number, *arguments):
    import ctypes
    require(os.uname().machine in {"x86_64", "aarch64"}, "Use normal stop: pidfd fallback is unavailable on this architecture")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *arguments)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def open_pidfd(pid):
    return os.pidfd_open(pid) if hasattr(os, "pidfd_open") else pidfd_call(434, pid, 0)


def terminate_pidfd(fd):
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    else:
        import ctypes
        pidfd_call(424, fd, signal.SIGTERM, ctypes.c_void_p(), 0)


def force_stop(record):
    identity = record["identity"]
    require(process_state(record) == "alive", "Refusing SIGTERM: stored process identity is no longer valid")
    require(identity["pid"] == identity["pgid"] == identity["sid"] and record["start_new_session"] is True,
            "Refusing SIGTERM: controller is not the recorded session leader")
    handles = []
    try:
        handles.append((open_pidfd(identity["pid"]), identity))
        require(process_state(record) == "alive", "Refusing SIGTERM: process identity changed")
        for path in Path("/proc").iterdir():
            if not path.name.isdigit() or int(path.name) == identity["pid"]:
                continue
            try:
                member = proc_stat(int(path.name))
                if member["pgid"] != identity["pgid"] or member["sid"] != identity["sid"]:
                    continue
                member = process_identity(int(path.name))
                if member is None:
                    continue
                require(member["uid"] == identity["uid"] and member["pgid"] == identity["pgid"] and member["sid"] == identity["sid"],
                        "Refusing SIGTERM: group membership or ownership changed")
                handles.append((open_pidfd(member["pid"]), member))
                actual = process_identity(member["pid"])
                require(actual is None or actual == member, "Refusing SIGTERM: group member identity changed")
            except (FileNotFoundError, ProcessLookupError):
                continue
        require(process_state(record) == "alive", "Refusing SIGTERM: controller identity changed")
        for fd, member in reversed(handles):
            try:
                actual = process_identity(member["pid"])
                if actual is None:
                    continue
                require(actual == member, "Refusing SIGTERM: group member identity changed")
                terminate_pidfd(fd)
            except (FileNotFoundError, ProcessLookupError):
                continue
    finally:
        for fd, _ in handles:
            os.close(fd)


def stop(config_path, config, work, owner, wait_seconds=0, force=False):
    with file_lock(work / ".corpus-control.lock"):
        require(work_identity(work) == owner, "Work directory identity changed")
        record = active_record(work)
        state = process_state(record) if record else "exited"
        if state in {"exited", "stale"}:
            require(not build_busy(work), "Build lock is held without a verified active controller; refusing adoption or signals")
            result = status(config_path, config, work, record)
            result["action"] = "already stopped; no signal sent"
            return result
        require(state == "alive", f"Cannot verify controller identity in {work / 'control-active.json'}; refusing stop or signals")
        require(Path(record["out_dir"]).is_dir(), "Controller output directory is not ready; retry stop shortly")
        stop_path = Path(record["stop_file"])
        try:
            write_json(stop_path, {"requested_at_utc": stamp(), "run_id": record["run_id"], "reason": "operator_stop"}, exclusive=True)
        except FileExistsError:
            require(stop_path.is_file() and not stop_path.is_symlink(), f"Unsafe existing stop file: {stop_path}")
        persist(work, record, "draining")
        if force:
            force_stop(record)
            persist(work, record, "term-sent")
    until = float("inf") if wait_seconds is None else time.monotonic() + wait_seconds
    next_notice = 0
    while time.monotonic() < until:
        state = process_state(record)
        if state in {"exited", "stale"}:
            break
        require(state == "alive", "Controller identity changed while draining; no further signals sent")
        if time.monotonic() >= next_notice:
            phase = phase_status(record)
            counts = numeric_counts(phase)
            print(f"Waiting for PID {record['identity']['pid']} to finish: {phase.get('stage', 'draining')} "
                  f"{json.dumps(counts, sort_keys=True)}. Ctrl-C stops waiting only.", file=sys.stderr, flush=True)
            next_notice = time.monotonic() + 5
        time.sleep(min(0.25, max(0, until - time.monotonic())))
    result = status(config_path, config, work, record)
    result["action"] = "SIGTERM requested" if force else "stop requested; no signal sent"
    return result


def display(result, as_json):
    if as_json:
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return
    print(f"State: {result['state']} | PID: {result.get('pid') or '-'} | process: {result['process_state']}")
    if result.get("action"):
        print(result["action"])
    if result.get("reason"):
        print(f"Reason: {result['reason']}")
    print(f"Config: {result['config']}")
    print(f"Configured workers ({result['configured_workers']}): " + json.dumps(result["concurrency"], sort_keys=True))
    if "launch_concurrency" in result:
        print(f"Launched workers ({result['launched_workers']}): " + json.dumps(result["launch_concurrency"], sort_keys=True))
    for key in ("run_dir", "log_path", "stop_file"):
        if key in result:
            print(f"{key}: {result[key]}")
    for error in (result.get("summary_error"), result.get("phase_status", {}).get("status_error")):
        if error:
            print(error)
    for split, item in result.get("progress", {}).items():
        if item.get("stage") in {"loading", "resume-loading-plan", "resume-loading-results"}:
            detail = f"loaded {item['loaded']}/{item['tasks']} tasks" if item.get("tasks") is not None else "reading saved plan"
            print(f"{split}: {detail}")
        else:
            counts = " ".join(f"{key}={count}" for key, count in item["counts"].items()) or "counts unavailable"
            basis = "terminal summary" if item["final"] else f"last progress sample, age ~{item['age_seconds']}s"
            print(f"{split}: {counts} ({basis})")
        if item["per_route"]:
            print("  Per route: " + json.dumps(item["per_route"], sort_keys=True))
        if item.get("disabled_routes"):
            print("  Disabled routes: " + json.dumps(item["disabled_routes"], sort_keys=True))
        if item.get("stop_reasons"):
            print("  Stop reasons: " + ", ".join(item["stop_reasons"]))
    safety = result.get("safety", {})
    if "deadline_utc" in safety:
        print(f"Safety deadline ({safety['clock']}): {safety['deadline_utc']} | remaining {safety['remaining_seconds']:.0f}s")
    else:
        print(safety.get("error", "Safety deadline unavailable"))


def main():
    if sys.argv[1:2] == ["_run"]:
        parser = argparse.ArgumentParser(description="Private generation child. Use tools/corpus start.")
        parser.add_argument("--run-dir", type=Path, required=True)
        parser.set_defaults(command="_run", json=False)
        args = parser.parse_args(sys.argv[2:])
    else:
        parser = argparse.ArgumentParser(description="Start, stop, or inspect corpus generation on Linux. Never prepare, assemble, train, or score.",
                                         epilog="Config requires work_dir and concurrency. Set generation_hours for a new clock. Relative paths use the config directory.")
        parser.add_argument("--config", default=os.environ.get("CLEFSIFT_CORPUS_CONFIG", DEFAULT_CONFIG), type=Path,
                            help="Config file (default: CLEFSIFT_CORPUS_CONFIG or ./corpus.json)")
        parser.add_argument("--json", action="store_true")
        commands = parser.add_subparsers(dest="command", required=True)
        for name in ("start", "stop", "status"):
            command = commands.add_parser(name)
            command.add_argument("--config", type=Path, default=argparse.SUPPRESS)
            command.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
            if name == "start":
                command.add_argument("--construct-final", action="store_true", help="Generate final_holdout only; default is development")
            if name == "stop":
                command.add_argument("--wait", action="store_true", help="Wait for exit; Ctrl-C stops waiting without cancellation")
                command.add_argument("--wait-seconds", type=float, help="Wait at most this many seconds; 0 returns immediately (default)")
                command.add_argument("--force", action="store_true", help="Send SIGTERM to verified controller-group members, never SIGKILL")
        args = parser.parse_args()
    try:
        if args.command == "_run":
            return run_generation(args.run_dir)
        path, config, work, owner = configuration(args.config, validate=args.command == "start")
        if args.command == "start":
            result = start(path, config, work, owner, args.construct_final)
        elif args.command == "stop":
            require(args.wait_seconds is None or math.isfinite(args.wait_seconds) and args.wait_seconds >= 0,
                    "--wait-seconds must be finite and nonnegative")
            seconds = args.wait_seconds if args.wait_seconds is not None else None if args.wait else 0
            result = stop(path, config, work, owner, seconds, args.force)
        else:
            result = status(path, config, work)
        display(result, args.json)
        return int(args.command == "start" and result["state"] in {"failure", "completed-with-shortfalls"})
    except (OSError, ValueError, KeyError, TypeError, OverflowError, subprocess.SubprocessError) as exc:
        message = "Another controller or control operation holds the lock; use status before retrying" if isinstance(exc, BlockingIOError) else str(exc)
        if args.json:
            print(json.dumps({"error": message}))
        else:
            print(f"corpus: {message}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("corpus: Interrupted; no automatic signal was sent to the controller", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
