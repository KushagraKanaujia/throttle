"""``throttle coldstart``: how long does your server take to come up?

Throttle launches your server command, then measures two times from the
moment it started the command:

* **ready**: until ``/health`` or ``/v1/models`` answers 200;
* **first token**: until the first streamed token of a tiny chat request.

Then it stops the server (SIGTERM to the whole process group, SIGKILL after
``--stop-timeout``), waits until the port is quiet, and does it again,
``--runs`` times. Give ``--cmd`` more than once (each with a ``--label``) to
compare engines or settings; runs alternate between the commands so a slow
disk cache or a busy machine hits all of them alike.

The server always gets killed: on success, on error, on Ctrl-C, on SIGTERM
and on SIGHUP. Nothing is sent anywhere except the tiny request to ``--url``.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx

from . import __version__
from . import style as style_module
from .benchmark import normalize_chat_completions_url

DEFAULT_RUNS = 3
DEFAULT_STARTUP_TIMEOUT = 900.0
DEFAULT_STOP_TIMEOUT = 30.0
POLL_SECONDS = 0.2
PROBE_TIMEOUT = 2.0
PROMPT = "Say hi."

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

# Set by tests; None means real sockets.
_TRANSPORT: httpx.BaseTransport | None = None

COLDSTART_DESCRIPTION = __doc__.split("\n\n", 1)[1] if __doc__ else ""
COLDSTART_EPILOG = """\
examples:
  throttle coldstart --cmd "vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000" \\
      --url http://localhost:8000 --model Qwen/Qwen2.5-7B-Instruct --runs 3

  # compare two engines on the same port
  throttle coldstart --url http://localhost:8000 --model Qwen/Qwen2.5-7B-Instruct \\
      --cmd "vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000" --label vllm \\
      --cmd "python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 8000" --label sglang

  Commands run with /bin/sh. Stop any server already on --url first.
"""


class ColdstartError(Exception):
    pass


def add_coldstart_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cmd", action="append", required=True, metavar="COMMAND",
                        help="the command that launches your server; repeat to compare "
                        "several (each with a --label)")
    parser.add_argument("--label", action="append", metavar="NAME",
                        help="a name for each --cmd, in the same order (default cmd1, cmd2, ...)")
    parser.add_argument("--url", required=True, metavar="URL",
                        help="where the server will answer (OpenAI-compatible base URL)")
    parser.add_argument("--model", required=True, metavar="MODEL",
                        help="model name for the tiny first-token request")
    parser.add_argument("--runs", type=_positive_int, default=DEFAULT_RUNS, metavar="K",
                        help=f"cold starts per command (default {DEFAULT_RUNS})")
    parser.add_argument("--startup-timeout", type=_positive_float, default=DEFAULT_STARTUP_TIMEOUT,
                        metavar="SECONDS", help="give up on a start after this long "
                        f"(default {DEFAULT_STARTUP_TIMEOUT:g})")
    parser.add_argument("--stop-timeout", type=_positive_float, default=DEFAULT_STOP_TIMEOUT,
                        metavar="SECONDS", help="SIGKILL the process group if SIGTERM has not "
                        f"stopped it after this long (default {DEFAULT_STOP_TIMEOUT:g})")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", metavar="NAME",
                        help="environment variable holding the bearer key (default OPENAI_API_KEY)")
    parser.add_argument("--allow-insecure-http", action="store_true",
                        help="allow plain http to a non-loopback host")
    parser.add_argument("--log-dir", type=Path, metavar="DIR",
                        help="where to keep each run's server log (default: a temp directory)")
    parser.add_argument("--json", dest="json_output", type=Path, metavar="PATH",
                        help="write every run and the summary as JSON")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not parsed > 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


@dataclass
class Run:
    label: str
    run: int
    ready_s: float | None = None
    ready_via: str | None = None
    first_token_s: float | None = None
    error: str | None = None
    log: str | None = None


class Server:
    """One launched server command, in its own process group."""

    def __init__(self, command: str, log_path: Path, stop_timeout: float) -> None:
        self.command = command
        self.log_path = log_path
        self.stop_timeout = stop_timeout
        self.process: subprocess.Popen[bytes] | None = None
        self._term_sent = False

    def start(self) -> None:
        with open(self.log_path, "wb") as log:
            log.write(f"$ {self.command}\n".encode())
            log.flush()
            self.process = subprocess.Popen(
                ["/bin/sh", "-c", self.command],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True,
            )

    def exited(self) -> int | None:
        return None if self.process is None else self.process.poll()

    def stop(self) -> None:
        """SIGTERM the whole process group, SIGKILL after the timeout.

        ``self.process`` is only cleared once the group is gone, so a second
        Ctrl-C during the wait cannot leak the server: the next call (the
        outer handler, ``finally`` or atexit) goes straight to SIGKILL."""

        process = self.process
        if process is None:
            return
        if self._term_sent:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
            self.process = None
            return
        self._term_sent = True
        deadline = time.monotonic() + self.stop_timeout
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=self.stop_timeout)
        # Wait for the rest of the group (children outlive the shell), then
        # SIGKILL whatever ignored SIGTERM. killpg(pid, 0) raises once it is gone.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            while time.monotonic() < deadline:
                os.killpg(process.pid, 0)
                time.sleep(POLL_SECONDS)
            os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)
        self.process = None


def _origin(chat_url: str) -> str:
    parts = urlsplit(chat_url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def probe_urls(chat_url: str) -> list[tuple[str, str]]:
    models = chat_url[: -len("/chat/completions")] + "/models"
    return [("/health", _origin(chat_url) + "/health"), ("/v1/models", models)]


def answering(client: httpx.Client, chat_url: str, headers: Mapping[str, str]) -> str | None:
    for name, url in probe_urls(chat_url):
        try:
            if client.get(url, headers=headers, timeout=PROBE_TIMEOUT).status_code == 200:
                return name
        except httpx.HTTPError:
            continue
    return None


def first_token(
    client: httpx.Client, chat_url: str, headers: Mapping[str, str], model: str
) -> tuple[bool, str | None]:
    """(True, None) once a streamed token arrives; (False, why) otherwise."""

    body = {
        "model": model, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 8, "temperature": 0, "stream": True,
    }
    try:
        with client.stream("POST", chat_url, headers=headers, json=body, timeout=60.0) as response:
            if response.status_code in (401, 403, 404):
                # Up (it answered /health or /v1/models) but this request can
                # never work: wrong --model, path or key. Retrying is pointless.
                detail = response.read().decode(errors="replace")[:200]
                raise ColdstartError(f"chat request rejected with HTTP {response.status_code} "
                                     f"(check --model, --url and the API key): {detail}")
            if response.status_code != 200:
                return False, f"HTTP {response.status_code}"
            for line in response.iter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                choices = event.get("choices") if isinstance(event, dict) else None
                for choice in choices if isinstance(choices, list) else []:
                    delta = choice.get("delta") if isinstance(choice, dict) else None
                    if not isinstance(delta, dict):
                        continue
                    if (delta.get("content") or delta.get("reasoning_content")
                            or delta.get("reasoning") or delta.get("tool_calls")):
                        return True, None
            return False, "the stream ended without a token"
    except httpx.HTTPError as exc:
        return False, type(exc).__name__


def _log_tail(path: Path, lines: int = 10) -> str:
    try:
        text = path.read_text(errors="replace").splitlines()[-lines:]
    except OSError:
        return ""
    return "\n".join("      " + line for line in text)


def measure(
    server: Server, client: httpx.Client, chat_url: str, headers: Mapping[str, str],
    model: str, startup_timeout: float, run: Run,
) -> None:
    start = time.monotonic()
    server.start()
    deadline = start + startup_timeout
    while run.ready_s is None:
        code = server.exited()
        if code is not None:
            raise ColdstartError(f"the server exited with code {code} before it answered")
        via = answering(client, chat_url, headers)
        if via:
            run.ready_s = time.monotonic() - start
            run.ready_via = via
            break
        if time.monotonic() >= deadline:
            raise ColdstartError(f"not ready within {startup_timeout:g}s (--startup-timeout)")
        time.sleep(POLL_SECONDS)
    last = None
    while True:
        ok, why = first_token(client, chat_url, headers, model)
        if ok:
            run.first_token_s = time.monotonic() - start
            return
        last = why
        code = server.exited()
        if code is not None:
            raise ColdstartError(f"the server exited with code {code} before its first token")
        if time.monotonic() >= deadline:
            raise ColdstartError(f"no first token within {startup_timeout:g}s ({last})")
        time.sleep(POLL_SECONDS)


def wait_quiet(client: httpx.Client, chat_url: str, headers: Mapping[str, str],
               timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while answering(client, chat_url, headers):
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_SECONDS)
    return True


def spread(values: Sequence[float | None]) -> dict[str, Any]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {"median": statistics.median(clean), "min": min(clean), "max": max(clean),
            "n": len(clean)}


def _secs(s: Mapping[str, Any]) -> str:
    if s["median"] is None:
        return "-"
    low, high = f"{s['min']:.1f}", f"{s['max']:.1f}"
    if s["n"] < 2 or low == high:
        return f"{s['median']:.1f}s"
    return f"{s['median']:.1f}s ({low}..{high})"


@contextlib.contextmanager
def _sigterm_as_interrupt() -> Iterator[None]:
    def raise_interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt
    # SIGHUP too: closing the terminal must not orphan the server group.
    signums = [signal.SIGTERM] + ([signal.SIGHUP] if hasattr(signal, "SIGHUP") else [])
    previous: dict[int, Any] = {}
    try:
        for signum in signums:
            previous[signum] = signal.signal(signum, raise_interrupt)
    except ValueError:  # not the main thread
        yield
        return
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def handle_coldstart(args: argparse.Namespace) -> int:
    st = style_module.Style()
    commands: list[str] = list(args.cmd)
    labels: list[str] = list(args.label or [])
    if labels and len(labels) != len(commands):
        print(f"Error: {len(commands)} --cmd but {len(labels)} --label; give one --label per "
              "--cmd, in the same order (or none)", file=sys.stderr)
        return EXIT_USAGE
    if not labels:
        labels = [f"cmd{i + 1}" for i in range(len(commands))] if len(commands) > 1 else ["server"]
    if len(set(labels)) != len(labels):
        print("Error: --label values must be different", file=sys.stderr)
        return EXIT_USAGE
    if any(not c.strip() for c in commands):
        print("Error: --cmd must not be empty", file=sys.stderr)
        return EXIT_USAGE
    try:
        chat_url = normalize_chat_completions_url(
            args.url, allow_insecure_http=args.allow_insecure_http)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    log_dir = args.log_dir or Path(tempfile.mkdtemp(prefix="throttle-coldstart-"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Error: cannot create --log-dir {log_dir}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    runs: list[Run] = []
    current: list[Server] = []

    def cleanup() -> None:
        # Stop before forgetting the server: if a second Ctrl-C lands while we
        # wait for it to exit, the next cleanup() call still finds it.
        while current:
            current[-1].stop()
            current.pop()

    atexit.register(cleanup)
    interrupted = False
    with httpx.Client(transport=_TRANSPORT) as client, _sigterm_as_interrupt():
        try:
            if answering(client, chat_url, headers):
                print(f"Error: something is already answering at {args.url}. Stop it first: "
                      "coldstart starts the server itself, so it would time the wrong one.",
                      file=sys.stderr)
                return EXIT_USAGE
            print(f"{st.brand()} coldstart {st.dot} {args.model} {st.dot} {args.runs} runs "
                  f"x {len(commands)} command{'s' if len(commands) > 1 else ''} {st.dot} "
                  f"logs in {log_dir}")
            for number in range(1, args.runs + 1):
                for label, command in zip(labels, commands):
                    run = Run(label=label, run=number)
                    log_path = log_dir / f"{label}-run{number}.log"
                    run.log = str(log_path)
                    server = Server(command, log_path, args.stop_timeout)
                    current.append(server)
                    try:
                        measure(server, client, chat_url, headers, args.model,
                                args.startup_timeout, run)
                    except ColdstartError as exc:
                        run.error = str(exc)
                    finally:
                        cleanup()
                    runs.append(run)
                    if run.error:
                        print(st.bad(f"  {label} run {number}: FAILED, {run.error} (log: {log_path})"))
                        tail = _log_tail(log_path)
                        if tail:
                            print(st.dim(tail))
                    else:
                        print(f"  {label} run {number}: ready {run.ready_s:.1f}s "
                              f"(via {run.ready_via}), first token {run.first_token_s:.1f}s")
                    if not wait_quiet(client, chat_url, headers, args.stop_timeout):
                        print(f"Error: something still answers at {args.url} after the server "
                              "was stopped; stop it by hand. Results so far are below.",
                              file=sys.stderr)
                        raise ColdstartError("port still busy")
        except KeyboardInterrupt:
            interrupted = True
            cleanup()
            print("\nInterrupted: the server was stopped.", file=sys.stderr)
        except ColdstartError:
            pass
        finally:
            cleanup()
            atexit.unregister(cleanup)

    summary = []
    print()
    print(f"  {'command':<16}{'ok':<8}{'ready':<26}first token")
    for label, command in zip(labels, commands):
        mine = [r for r in runs if r.label == label]
        ok = [r for r in mine if r.error is None]
        ready = spread([r.ready_s for r in ok])
        token = spread([r.first_token_s for r in ok])
        summary.append({"label": label, "cmd": command, "runs": len(mine), "ok": len(ok),
                        "ready_s": ready, "first_token_s": token})
        print(f"  {label[:15]:<16}{f'{len(ok)}/{len(mine)}':<8}{_secs(ready):<26}{_secs(token):<26}"
              .rstrip())
    print(st.dim("  seconds from launching the command; median (min..max) over the runs"))
    if len(summary) > 1:
        timed = [s for s in summary if s["first_token_s"]["median"] is not None]
        if len(timed) > 1:
            timed.sort(key=lambda s: s["first_token_s"]["median"])
            best, second = timed[0], timed[1]
            gap = second["first_token_s"]["median"] - best["first_token_s"]["median"]
            noise = (best["first_token_s"]["max"] >= second["first_token_s"]["min"])
            if noise:
                print(f"  {best['label']} and {second['label']}: first-token times overlap, within noise")
            else:
                print(f"  {best['label']} reaches its first token {gap:.1f}s sooner than "
                      f"{second['label']}")

    if args.json_output:
        record = {
            "type": "throttle_coldstart", "version": 1, "throttle_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(), "url": chat_url,
            "model": args.model, "runs_per_command": args.runs,
            "summary": summary, "runs": [asdict(r) for r in runs],
            "interrupted": interrupted,
        }
        try:
            args.json_output.write_text(json.dumps(record, indent=2) + "\n")
        except OSError as exc:
            print(f"Error: could not write {args.json_output}: {exc}", file=sys.stderr)
    if interrupted:
        return EXIT_INTERRUPTED
    return EXIT_OK if runs and all(r.error is None for r in runs) and \
        len(runs) == args.runs * len(commands) else EXIT_FAILED
