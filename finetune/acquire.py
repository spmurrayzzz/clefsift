import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


GIB = 1024 ** 3
CHUNK = 1024 ** 2


class Stop(Exception):
    pass


def save(path, value):
    data = value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def load(path):
    return json.loads(path.read_bytes())


def progress(**value):
    print(json.dumps(value, sort_keys=True), flush=True)


def relative(value):
    if (not isinstance(value, str) or not value or value.startswith("/")
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or re.search(r"[\\%?#\x00-\x1f\x7f]", value)
            or re.search(r"(^|[/_.-])(test|tests|testing|heldout|held-out)([/_.-]|$)", value, re.I)):
        raise Stop("invalid_catalog_path")
    return value


def url(value, redirect=False):
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.fragment or (parsed.query and not redirect)
            or re.search(r"[\\\s\x00-\x1f\x7f]", value)
            or not re.fullmatch(r"[A-Za-z0-9.-]+", parsed.hostname)
            or parsed.port not in {None, 443}):
        raise Stop("invalid_https_url")
    return value


def catalog_files(catalog):
    if (catalog.get("status") != "approved-exploratory-acquisition"
            or catalog.get("workflow") != "clefsift-general-v1"):
        raise Stop("unapproved_catalog")
    files, seen, sources = [], set(), set()
    for source in catalog["sources"]:
        sid = source["id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", sid) or sid in sources:
            raise Stop("invalid_source_id")
        sources.add(sid)
        for entry in source["files"]:
            if "url" in entry:
                address = url(entry["url"])
                path = relative(urllib.parse.unquote(urllib.parse.urlsplit(address).path.lstrip("/")))
                original = entry["url"]
            else:
                original = path = relative(entry["path"])
                if "dataset" in source:
                    dataset = relative(source["dataset"])
                    revision = source["revision"]
                    if len(dataset.split("/")) != 2 or not re.fullmatch(r"[0-9a-f]{40}", revision):
                        raise Stop("unpinned_dataset")
                    address = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{urllib.parse.quote(path)}"
                else:
                    archive = relative(source["archive"])
                    if "/" in archive:
                        raise Stop("invalid_archive")
                    address = f"https://archive.org/download/{urllib.parse.quote(archive)}/{urllib.parse.quote(path)}"
            local = f"raw/{sid}/{path.rsplit('/', 1)[-1]}"
            if local.casefold() in seen or any(
                    record.get(key, "unsplit") not in {"train", "valid", "validation", "unsplit"}
                    for record in (source, entry) for key in ("source_split", "split")):
                raise Stop("ambiguous_or_test_file")
            seen.add(local.casefold())
            hashes = {key: entry[key].lower() for key in ("sha256", "sha1") if key in entry}
            if any(not re.fullmatch(r"[0-9a-f]{" + str(64 if key == "sha256" else 40) + "}", value)
                   for key, value in hashes.items()):
                raise Stop("invalid_catalog_hash")
            size = entry.get("bytes")
            if size is not None and (type(size) is not int or size < 0):
                raise Stop("invalid_catalog_length")
            files.append(dict(source_id=sid, catalog_path=original, local_path=local,
                              source_url=url(address), expected_bytes=size, expected_hashes=hashes))
    if not files:
        raise Stop("empty_catalog")
    return files


def disk(root, reserve=0):
    if any(shutil.disk_usage(path).free < 30 * GIB + reserve for path in (Path("/"), Path.home(), root)):
        raise Stop("disk_floor")


def interrupted(signum, frame):
    raise Stop("deadline" if signum == signal.SIGALRM else "interrupted")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def response(opener, address):
    for _ in range(11):
        try:
            return opener.open(urllib.request.Request(url(address, redirect=True),
                               headers={"Accept-Encoding": "identity", "User-Agent": "ClefSift-acquire/1"}), timeout=30)
        except urllib.error.HTTPError as exc:
            location = exc.headers.get("Location")
            exc.close()
            if exc.code not in {301, 302, 303, 307, 308} or not location:
                raise
            address = url(urllib.parse.urljoin(address, location), redirect=True)
    raise Stop("redirect_limit")


def receipt(directory):
    journal = directory / "ledger.jsonl"
    reserved = 0
    if journal.exists():
        for line in journal.read_text().splitlines():
            value = json.loads(line)["reserved_bytes"]
            if type(value) is not int or not reserved <= value <= reserved + CHUNK:
                raise Stop("invalid_ledger")
            reserved = value
    result = load(directory / "result.json") if (directory / "result.json").exists() else {}
    charged = result.get("charged_bytes", reserved)
    payload = directory / "payload.part"
    size = payload.stat().st_size if payload.exists() else 0
    if type(charged) is not int or not size <= charged <= reserved:
        raise Stop("invalid_receipt")
    return charged, result


def download(root, directory, spec, remaining, opener):
    size, charged, status, code = 0, 0, "failed", None
    hashes = {name: hashlib.new(name) for name in {"sha256", *spec["expected_hashes"]}}
    try:
        with (directory / "payload.part").open("xb") as target, (directory / "ledger.jsonl").open("xb") as ledger:
            with response(opener, spec["source_url"]) as source:
                code = source.status
                if code != 200 or source.headers.get("Content-Encoding", "identity") != "identity":
                    raise Stop("invalid_response")
                length = source.headers.get("Content-Length")
                length = int(length) if length is not None else None
                if length is not None and (length < 0 or length > remaining):
                    raise Stop("byte_limit")
                expected = spec["expected_bytes"]
                if expected is not None and length is not None and length != expected:
                    raise ValueError("length")
                while True:
                    if length is not None and size == length:
                        break
                    amount = min(CHUNK, remaining - size)
                    if length is not None:
                        amount = min(amount, length - size)
                    if amount <= 0:
                        raise Stop("byte_limit")
                    disk(root, amount)
                    charged = size + amount
                    ledger.write((json.dumps({"reserved_bytes": charged}) + "\n").encode())
                    ledger.flush()
                    os.fsync(ledger.fileno())
                    chunk = source.read(amount)
                    target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                    for digest in hashes.values():
                        digest.update(chunk)
                    size += len(chunk)
                    charged = size
                    progress(event="payload", attempt=directory.name, source_id=spec["source_id"],
                             local_path=spec["local_path"], bytes=size)
                    if not chunk:
                        break
                if ((length is not None and size != length) or (expected is not None and size != expected)
                        or any(hashes[key].hexdigest() != value for key, value in spec["expected_hashes"].items())):
                    raise ValueError("integrity")
                status = "complete"
    except urllib.error.HTTPError as exc:
        code = exc.code
        if code in {401, 403, 429}:
            raise Stop(f"http_{code}") from None
    except (urllib.error.URLError, OSError, ValueError):
        pass
    finally:
        payload = directory / "payload.part"
        stored = payload.stat().st_size if payload.exists() else 0
        save(directory / "result.json", dict(status=status, bytes=stored, charged_bytes=charged,
                                             sha256=hashes["sha256"].hexdigest() if stored == size else None,
                                             http_status=code))
    return receipt(directory)


def completed(root, directory, spec, result):
    payload = directory / "payload.part"
    hashes = {name: hashlib.new(name) for name in {"sha256", *spec["expected_hashes"]}}
    with payload.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            for digest in hashes.values():
                digest.update(chunk)
    size = payload.stat().st_size
    if (size != result["bytes"] or hashes["sha256"].hexdigest() != result["sha256"]
            or (spec["expected_bytes"] is not None and size != spec["expected_bytes"])
            or any(hashes[key].hexdigest() != value for key, value in spec["expected_hashes"].items())):
        raise Stop("completed_file_corrupt")
    local = root / spec["local_path"]
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        if not local.samefile(payload):
            raise Stop("output_collision")
    else:
        os.link(payload, local)
    return {key: spec[key] for key in ("source_id", "catalog_path", "local_path", "source_url")} | {
        "bytes": size, "sha256": result["sha256"], "verified_hash": bool(spec["expected_hashes"]),
        "integrity_status": "verified" if spec["expected_hashes"] else "unverified",
        "upstream_hashes": spec["expected_hashes"]}


def acquire(args):
    started = time.time()
    if not 0 < args.max_bytes <= 10 * GIB or not math.isfinite(args.minutes) or not 0 < args.minutes <= 120:
        raise Stop("invalid_limits")
    deadline = started + args.minutes * 60
    if args.deadline_utc:
        shared = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
        if shared.utcoffset() is None:
            raise Stop("deadline_requires_timezone")
        deadline = min(deadline, shared.timestamp())
    original = args.catalog.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    files = catalog_files(json.loads(original))
    root = args.out_dir.absolute()
    if root.is_symlink():
        raise Stop("unsafe_root")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o022:
        raise Stop("unowned_root")
    for parent, dirs, names in os.walk(root):
        if any((Path(parent) / name).is_symlink() for name in dirs + names):
            raise Stop("symlink_in_root")
    marker = root / "acquisition.json"
    if not marker.exists() and any(path.name != ".lock" for path in root.iterdir()):
        raise Stop("unowned_root")
    with os.fdopen(os.open(root / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), "r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Stop("locked") from None
        if marker.exists():
            owned = load(marker)
            if (owned["catalog_sha256"] != digest or owned["owner_uid"] != os.getuid()
                    or (root / "catalog.json").read_bytes() != original):
                raise Stop("catalog_or_owner_mismatch")
            deadline = min(deadline, owned["deadline_epoch"])
            args.max_bytes = min(args.max_bytes, owned["max_bytes"])
        else:
            if set(path.name for path in root.iterdir()) != {".lock"}:
                raise Stop("unowned_root")
            disk(root)
            save(root / "catalog.json", original)
            save(marker, dict(catalog_sha256=digest, owner_uid=os.getuid(), max_bytes=args.max_bytes,
                              deadline_epoch=deadline))
        run = root / "runs" / uuid.uuid4().hex
        run.mkdir(parents=True)
        save(run / "request.json", dict(catalog_sha256=digest, max_bytes=args.max_bytes, minutes=args.minutes,
                                        deadline_utc=datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
                                        requested_deadline_utc=args.deadline_utc, started_epoch=started))
        used, acquired, error = 0, [], None
        try:
            for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, interrupted)
            if deadline <= time.time():
                raise Stop("deadline")
            signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
            disk(root)
            histories = []
            for index, spec in enumerate(files):
                history = []
                for number in (1, 2):
                    directory = root / "attempts" / f"{index:04d}-{number}"
                    if directory.exists():
                        if (directory / "request.json").exists():
                            request = load(directory / "request.json")
                            if request["catalog_sha256"] != digest or request["file"] != spec:
                                raise Stop("attempt_catalog_mismatch")
                        elif any(directory.iterdir()):
                            raise Stop("missing_attempt_request")
                        charged, result = receipt(directory)
                        used += charged
                        history.append((directory, result))
                histories.append(history)
            if set((root / "attempts").glob("*")) != {path for history in histories for path, _ in history}:
                raise Stop("unknown_attempt")
            if used > args.max_bytes:
                raise Stop("byte_limit")
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
            for index, (spec, history) in enumerate(zip(files, histories)):
                while not history or history[-1][1].get("status") != "complete":
                    if any(result.get("http_status") in {401, 403, 429} for _, result in history):
                        raise Stop("http_retry_prohibited")
                    if len(history) >= 2:
                        raise Stop("attempt_limit")
                    if used >= args.max_bytes or (spec["expected_bytes"] or 0) > args.max_bytes - used:
                        raise Stop("byte_limit")
                    disk(root)
                    directory = root / "attempts" / f"{index:04d}-{len(history) + 1}"
                    directory.mkdir(parents=True)
                    save(directory / "request.json", dict(catalog_sha256=digest, file=spec, run_id=run.name))
                    charged, result = download(root, directory, spec, args.max_bytes - used, opener)
                    used += charged
                    history.append((directory, result))
                acquired.append(completed(root, history[-1][0], spec, history[-1][1]))
            manifest = dict(stage="acquired", catalog_sha256=digest, files=acquired)
            if (root / "manifest.json").exists():
                if load(root / "manifest.json") != manifest:
                    raise Stop("manifest_mismatch")
            else:
                save(root / "manifest.json", manifest)
        except BaseException as exc:
            error = str(exc) if isinstance(exc, Stop) else type(exc).__name__
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                used = sum(receipt(path)[0] for path in (root / "attempts").glob("*"))
            except (OSError, ValueError, KeyError, TypeError, Stop):
                used = None
            summary = dict(stage="acquired" if error is None else "stopped", catalog_sha256=digest,
                           charged_bytes=used, completed_files=len(acquired), error=error,
                           elapsed_seconds=time.time() - started, run_id=run.name)
            save(run / "summary.json", summary)
            progress(**summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--deadline-utc")
    args = parser.parse_args()
    try:
        acquire(args)
    except (Exception, KeyboardInterrupt) as exc:
        progress(stage="stopped", error=str(exc) if isinstance(exc, Stop) else type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
