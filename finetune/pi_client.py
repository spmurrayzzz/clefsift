from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import time


MAX_LOG_BYTES = 8 * 1024 * 1024
FLAGS = (
    "--offline", "--no-session", "--no-context-files", "--no-tools",
    "--no-extensions", "--no-skills", "--no-prompt-templates", "--no-themes",
    "--approve",
)
SETTINGS = {
    "retry": {"enabled": False, "maxRetries": 0, "provider": {"maxRetries": 0}},
    "compaction": {"enabled": False},
}


class PiError(Exception):
    def __init__(self, kind, message=None, *, retryable=None):
        self.kind = kind
        self.retryable = kind in {"transport", "timeout", "output_limit", "empty_response"} if retryable is None else retryable
        super().__init__((message or f"Pi request failed: {kind}.") +
                         " Raw process logs remain in the attempt directory; size limits can truncate them.")


def _check(deadline, stopped):
    if stopped.is_set():
        raise PiError("stopped")
    if time.monotonic() >= deadline:
        raise PiError("timeout")


def _exclusive(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")


def _write(path, value):
    with _exclusive(path) as target:
        target.write(value)


def _write_json(path, value):
    _write(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _failure(value, fallback="transport"):
    if re.search(r"\b(401|402|403|429)\b|unauthori[sz]ed|forbidden|authenticat|credentials?|"
                 r"api[ _-]?key|access[ _-]?token|rate[ _-]?limit|quota|usage[ _-]?limit|"
                 r"too many requests|billing|insufficient.*(credit|balance)|token.*expir", value, re.I):
        return PiError("auth_or_rate_limit", value)
    if re.search(r"unknown[ _-]?(provider|model)\b|\b(model|provider)\b[^\n]{0,120}\b(not found|not available|not supported|does not exist)\b|no models", value, re.I):
        return PiError("provider_config", value)
    if re.search(r"\b400\b", value) and re.search(r"privacy|data policy|data retention", value, re.I):
        return PiError("provider_config", value)
    if re.search(r"unknown option|unrecognized|not found|invalid argument", value, re.I):
        return PiError("invalid_response", value)
    return PiError(fallback, value)


def _run(args, directory, stem, stdin_path, deadline, stopped, env):
    stdout_path = directory / ("events.jsonl" if stem == "completion" else f"{stem}.stdout")
    stderr_path = directory / ("stderr.log" if stem == "completion" else f"{stem}.stderr")
    process = None
    finished = False
    sizes = {"stdout": 0, "stderr": 0}
    truncated = []
    started = time.monotonic()
    error = None
    try:
        with ExitStack() as stack:
            targets = {"stdout": stack.enter_context(_exclusive(stdout_path)),
                       "stderr": stack.enter_context(_exclusive(stderr_path))}
            source = stack.enter_context(stdin_path.open("rb")) if stdin_path else subprocess.DEVNULL
            selector = stack.enter_context(selectors.DefaultSelector())
            _check(deadline, stopped)
            process = subprocess.Popen(args, cwd=directory, env=env, stdin=source,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=True)
            try:
                for name in targets:
                    stream = getattr(process, name)
                    stack.enter_context(stream)
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, selectors.EVENT_READ, name)
                while selector.get_map():
                    _check(deadline, stopped)
                    for key, _ in selector.select(min(0.05, max(0, deadline - time.monotonic()))):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        name = key.data
                        remaining = MAX_LOG_BYTES - sizes[name]
                        targets[name].write(chunk[:remaining])
                        sizes[name] += min(len(chunk), remaining)
                        if len(chunk) > remaining:
                            truncated.append(name)
                            kind = "output_limit" if stem == "completion" and name == "stdout" else "invalid_response"
                            raise PiError(kind, "Pi process output exceeded the size limit.", retryable=kind == "output_limit")
                while True:
                    _check(deadline, stopped)
                    try:
                        process.wait(timeout=min(0.05, max(0, deadline - time.monotonic())))
                        break
                    except subprocess.TimeoutExpired:
                        pass
                finished = True
            finally:
                if not finished:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=5)
                    except ProcessLookupError:
                        pass
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    finally:
                        process.wait()
    except KeyboardInterrupt:
        error = "stopped"
        raise PiError("stopped") from None
    except PiError as exc:
        error = exc.kind
        raise
    except OSError:
        error = "transport"
        raise PiError("transport", "Pi process or audit file could not be accessed.", retryable=False) from None
    finally:
        _write_json(directory / f"{stem}.process.json", {
            "exit_code": process.returncode if process else None,
            "elapsed_seconds": time.monotonic() - started,
            "captured_bytes": sizes, "truncated": truncated, "error": error,
            "stdout": stdout_path.name, "stderr": stderr_path.name,
        })
    stdout, stderr = stdout_path.read_bytes(), stderr_path.read_bytes()
    if process.returncode != 0:
        raise _failure((stderr + b"\n" + stdout).decode("utf-8", errors="replace"))
    return stdout, stderr


def _unique(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError
    return value


def _constant(value):
    raise ValueError


def _number(value):
    return type(value) in (int, float) and value >= 0 and math.isfinite(value)


def _message(message, provider, model, *, partial=False):
    if not isinstance(message, dict) or message.get("role") not in {"assistant", "user", "system"}:
        raise PiError("invalid_response")
    if message["role"] == "assistant" and (message.get("provider") != provider or message.get("model") != model):
        raise PiError("invalid_response", "Pi returned an unexpected provider or model.")
    if message.get("aborted") or message.get("stopReason") == "aborted":
        raise PiError("stopped")
    if message.get("errorMessage") or message.get("isError"):
        raise _failure(str(message.get("errorMessage", "")))
    role = message["role"]
    content = message.get("content")
    if role == "assistant":
        reason = message.get("stopReason")
        if reason == "error":
            raise _failure(str(message.get("errorMessage", "")))
        if reason == "length":
            raise PiError("output_limit", "Pi response reached its output limit.", retryable=True)
        if reason not in ({"pending", "stop"} if partial else {"stop"}):
            raise PiError("invalid_response", "Pi did not return a complete text response.")
        if not partial:
            usage = message.get("usage")
            if not isinstance(usage, dict) or any(not _number(usage.get(key)) for key in
                                                 ("input", "output", "cacheRead", "cacheWrite", "totalTokens")):
                raise PiError("invalid_response", "Pi returned invalid usage.")
            cost = usage.get("cost")
            if not isinstance(cost, dict) or any(not _number(cost.get(key)) for key in
                                                ("input", "output", "cacheRead", "cacheWrite", "total")):
                raise PiError("invalid_response", "Pi returned invalid usage costs.")
            if any(not _number(usage[key]) for key in ("reasoning", "cacheWrite1h") if key in usage):
                raise PiError("invalid_response", "Pi returned invalid usage.")
            if "responseModel" in message and (not isinstance(message["responseModel"], str) or not message["responseModel"].strip()):
                raise PiError("invalid_response", "Pi returned an invalid response model.")
    elif isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list) or any(not isinstance(block, dict) or block.get("type") not in {"text", "thinking"}
                                          or not isinstance(block.get("text" if block["type"] == "text" else "thinking"), str) for block in content):
        raise PiError("invalid_response", "Pi returned unsupported content or a tool call.")
    if message.get("toolsAdded") or message.get("toolsRemoved"):
        raise PiError("invalid_response", "Pi changed the tool configuration.")


def _parse(raw, provider, model):
    if not raw or not raw.endswith(b"\n"):
        raise PiError("invalid_response", "Pi JSONL output is missing or truncated.")
    allowed = {"session", "agent_start", "turn_start", "message_start", "message_update", "message_end",
               "turn_end", "agent_end", "agent_settled", "thinking_level_changed"}
    counts = {name: 0 for name in allowed}
    answer = None
    assistant_started = False
    ended = False
    settled = False
    try:
        for line in raw.split(b"\n")[:-1]:
            event = json.loads(line.removesuffix(b"\r").decode("utf-8"), object_pairs_hook=_unique,
                               parse_constant=_constant)
            if not isinstance(event, dict):
                raise PiError("invalid_response")
            kind = event.get("type")
            if kind not in allowed or settled or event.get("willRetry") or event.get("aborted") or event.get("errorMessage"):
                raise PiError("invalid_response", "Pi emitted an unexpected event.")
            if not counts["session"] and kind != "session":
                raise PiError("invalid_response", "Pi JSONL output has no session header.")
            if ended and kind != "agent_settled":
                raise PiError("invalid_response", "Pi emitted events after the run ended.")
            if kind == "turn_start" and not counts["agent_start"]:
                raise PiError("invalid_response")
            if kind.startswith("message_") and (not counts["turn_start"] or counts["turn_end"]):
                raise PiError("invalid_response")
            counts[kind] += 1
            if kind in {"session", "agent_start", "turn_start", "turn_end", "agent_end", "agent_settled"} and counts[kind] != 1:
                raise PiError("invalid_response", "Pi emitted multiple runs or turns.")
            if kind in {"message_start", "message_end"}:
                message = event.get("message")
                _message(message, provider, model, partial=kind == "message_start")
                if message["role"] == "assistant":
                    if kind == "message_start":
                        if assistant_started or answer is not None:
                            raise PiError("invalid_response")
                        assistant_started = True
                    else:
                        if not assistant_started or answer is not None:
                            raise PiError("invalid_response", "Pi must return exactly one assistant message.")
                        answer = message
            elif kind == "message_update":
                update = event.get("assistantMessageEvent")
                if not assistant_started or answer is not None or not isinstance(update, dict):
                    raise PiError("invalid_response")
                update_kind = update.get("type")
                if update_kind == "error":
                    if update.get("reason") == "aborted":
                        raise PiError("stopped")
                    raise _failure(json.dumps(update))
                if update_kind not in {"start", "text_start", "text_delta", "text_end", "thinking_start", "thinking_delta", "thinking_end", "done"}:
                    raise PiError("invalid_response", "Pi emitted an unsupported update or tool call.")
                if update_kind in {"text_start", "text_delta", "text_end", "thinking_start", "thinking_delta", "thinking_end"}:
                    if type(update.get("contentIndex")) is not int or update["contentIndex"] < 0:
                        raise PiError("invalid_response")
                    field = {"text_delta": "delta", "text_end": "content", "thinking_delta": "delta", "thinking_end": "content"}.get(update_kind)
                    if field and not isinstance(update.get(field), str):
                        raise PiError("invalid_response")
                if update_kind == "done":
                    _message(update.get("message"), provider, model)
                    if update.get("reason") != "stop":
                        raise PiError("invalid_response")
                if "message" in event:
                    _message(event["message"], provider, model, partial=True)
                if "partial" in update:
                    _message(update["partial"], provider, model, partial=True)
            elif kind == "turn_end":
                if answer is None or event.get("message") != answer or event.get("toolResults") != []:
                    raise PiError("invalid_response")
            elif kind == "agent_end":
                messages = event.get("messages")
                if answer is None or not counts["turn_end"] or not isinstance(messages, list):
                    raise PiError("invalid_response")
                for message in messages:
                    _message(message, provider, model)
                if [message for message in messages if message["role"] == "assistant"] != [answer]:
                    raise PiError("invalid_response")
                ended = True
            elif kind == "agent_settled":
                if not ended:
                    raise PiError("invalid_response")
                settled = True
            elif kind == "thinking_level_changed" and event.get("level") not in {"off", "minimal", "low", "medium", "high", "xhigh", "max"}:
                raise PiError("invalid_response", "Pi returned an unknown thinking level.")
        if not settled or answer is None or any(counts[kind] != 1 for kind in
                                               ("session", "agent_start", "turn_start", "turn_end", "agent_end")):
            raise PiError("invalid_response", "Pi did not finish one complete run.")
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise PiError("invalid_response", "Pi returned malformed JSONL.") from None
    if not "".join(block["text"] for block in answer["content"] if block["type"] == "text").strip():
        raise PiError("empty_response", "Pi returned empty final assistant text.")
    result = {"provider": provider, "model": model, "text": "".join(block["text"] for block in answer["content"] if block["type"] == "text"),
              "usage": answer["usage"], "requested_thinking": "off"}
    if "providerThinkingLevel" in answer:
        result["provider_thinking_level"] = answer["providerThinkingLevel"]
    if "responseModel" in answer:
        result["response_model"] = answer["responseModel"]
    return result


def complete(route, system, prompt, directory, *, timeout, deadline, stopped, executable="pi") -> dict:
    if os.name != "posix":
        raise PiError("invalid_response", "Pi client requires POSIX process groups.")
    if not isinstance(route, str) or "/" not in route or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in route):
        raise PiError("invalid_response", "Route must be an exact provider/model string.")
    provider, model = route.split("/", 1)
    if not provider or not model or provider.startswith("-") or model.startswith("-"):
        raise PiError("invalid_response", "Route must contain a provider and model.")
    if any(not isinstance(value, str) or not value.strip() or "\x00" in value for value in (system, prompt)) or system.startswith("\ufeff"):
        raise PiError("invalid_response", "System and user prompts must be nonempty text.")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in (timeout, deadline)) or timeout <= 0:
        raise PiError("invalid_response", "Timeout and monotonic deadline must be finite; timeout must be positive.")
    call_deadline = min(deadline, time.monotonic() + timeout)
    _check(call_deadline, stopped)
    try:
        directory = Path(directory).absolute()
        directory.mkdir(mode=0o700, exist_ok=False)
        directory = directory.resolve(strict=True)
    except OSError:
        raise PiError("invalid_response", "Attempt directory must be new with an existing parent.") from None
    try:
        owned = directory / ".pi"
        owned.mkdir(mode=0o700)
        _write_json(owned / "settings.json", SETTINGS)
        _write(owned / "APPEND_SYSTEM.md", b"")
        system_bytes, prompt_bytes = system.encode("utf-8"), prompt.encode("utf-8")
        _write(directory / "system.txt", system_bytes)
        _write(directory / "prompt.txt", prompt_bytes)
        resolved = shutil.which(os.fspath(executable))
        if not resolved:
            raise PiError("invalid_response", "Pi executable was not found.")
        resolved = os.path.abspath(resolved)
        env = os.environ.copy()
        overrides = {"PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0", "NO_COLOR": "1"}
        env.update(overrides)
        base = [resolved, *FLAGS]
        args = [*base, "--print", "--mode", "json", "--provider", provider, "--model", model,
                "--thinking", "off", "--system-prompt", str(directory / "system.txt")]
        local_placeholder = provider == "local" and os.environ.get("CLEFSIFT_LOCAL_UNAUTHENTICATED") == "1"
        if local_placeholder:
            args.extend(["--api-key", "local"])
        request_extension = (Path(__file__).with_name("pi-generation.ts")
                             if os.environ.get("CLEFSIFT_STRUCTURED_GENERATION") == "1"
                             and route in {"local/qwen3.8-27b", "openai-codex/gpt-6-luna"} else None)
        extension_hash = _hash(request_extension.read_bytes()) if request_extension else None
        if request_extension:
            args.extend(["--extension", str(request_extension.resolve())])
        _write_json(directory / "config.json", {
            "route": route, "argv": args, "cwd": str(directory), "project_settings": SETTINGS,
            "environment_overrides": overrides, "timeout_seconds": timeout, "deadline_monotonic": deadline,
            "call_deadline_monotonic": call_deadline, "max_bytes_per_log": MAX_LOG_BYTES,
            "system_sha256": _hash(system_bytes), "prompt_sha256": _hash(prompt_bytes),
            "append_system": {"path": ".pi/APPEND_SYSTEM.md", "sha256": _hash(b""),
                              "policy": "Empty trusted project file shadows the user append prompt."},
            "system_prompt_policy": "Explicit system.txt; Pi also adds the working directory in its version-specific format.",
            "local_unauthenticated_placeholder": local_placeholder,
            "request_extension_sha256": extension_hash,
            "configuration_scope": "Enforced CLI and project overrides; other user provider settings remain inherited. Credentials are managed by Pi. The explicit local unauthenticated flag supplies only a nonsecret placeholder.",
        })
        version_raw, _ = _run([*base, "--version"], directory, "version", None, call_deadline, stopped, env)
        version = version_raw.decode("utf-8").strip()
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.+-]+)?", version):
            raise PiError("invalid_response", "Pi did not report a valid version.")
        help_raw, _ = _run([*base, "--help"], directory, "help", None, call_deadline, stopped, env)
        help_text = help_raw.decode("utf-8")
        if any(flag not in help_text for flag in (*FLAGS, "--print", "--mode", "--provider", "--model", "--thinking", "--system-prompt")):
            raise PiError("invalid_response", "Pi does not advertise the required CLI options.")
        stdout, stderr = _run(args, directory, "completion", directory / "prompt.txt", call_deadline, stopped, env)
        _check(call_deadline, stopped)
        if request_extension:
            if _hash(request_extension.read_bytes()) != extension_hash:
                raise PiError("invalid_response", "Generation request extension changed.")
            options = json.loads((directory / "request-options.json").read_bytes())
            if options.get("route") != route:
                raise PiError("invalid_response", "Missing generation request options.")
        result = {**_parse(stdout, provider, model), "pi_version": version,
                  "events_sha256": _hash(stdout), "stderr_sha256": _hash(stderr)}
        _check(call_deadline, stopped)
        return result
    except PiError:
        raise
    except KeyboardInterrupt:
        raise PiError("stopped") from None
    except (OSError, ValueError, UnicodeError):
        raise PiError("invalid_response", "Pi configuration or audit data could not be processed.") from None
