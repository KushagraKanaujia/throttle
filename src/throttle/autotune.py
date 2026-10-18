"""``throttle autotune``: search serving configs and keep the cheapest one
that is verifiably cheaper and passes the OUTPUT CHANGED check.

You give a launch-command template with placeholders, a search space and your
GPU price. For each trial Throttle (re)starts the server with one config,
waits until it answers, and runs the same measurement as ``throttle check``
against a calibrated baseline:

1. Trial 0 is the baseline config, measured ``--baseline-runs`` times (3 by
   default) on one server start, so the run-to-run noise bound is calibrated.
2. Every candidate is measured once and judged against that baseline with the
   ``throttle check`` rules: CHEAPER only when the change is larger than the
   noise bound and the 95% intervals do not overlap; OUTPUT CHANGED when the
   answers' length or max-token-cap hits moved (the same prompts at
   temperature 0 should get the same answers). Only CHEAPER counts.
3. The search is greedy coordinate descent: one setting at a time, walking
   away from the best config so far while each step is a verified CHEAPER
   that also beats the best $/M; a direction stops at the first step that is
   not. It stops when a full pass finds nothing better, or after
   ``--max-trials`` candidates (default 8).

GPU count is a real cost. ``--gpu-hourly-rate`` is the price of ONE GPU here,
multiplied by the GPU count of each trial (the ``tp`` / ``tensor_parallel_size``
setting, or ``--gpus-param``): 2 GPUs that are 1.56x faster are still 28%
more per token.

During early access, without ``throttle login`` Autotune runs fully locally and
free (no cloud metering). Logged in, every trial is metered ($1.00 of credits,
POST /autotune/trials before it runs; 402 stops the run cleanly) and the
result is saved to your Throttle Cloud account; if the cloud cannot be
reached, Autotune warns and carries on locally. ``--dry-run`` simulates the
server, needs no GPU and no login, charges nothing, and labels every number
SIMULATED.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import secrets
import signal
import string
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import httpx

from . import __version__
from . import check as check_module
from . import cloud
from . import style as style_module

RECORD_TYPE = "throttle_autotune"
RECORD_VERSION = 1
DEFAULT_MAX_TRIALS = 8
DEFAULT_BASELINE_RUNS = 3
DEFAULT_STARTUP_TIMEOUT = 900.0
HEALTH_POLL_SECONDS = 2.0
STOP_TIMEOUT_SECONDS = 60.0
GPU_PARAM_NAMES = ("tp", "tensor_parallel_size", "tensor-parallel-size", "gpus", "num_gpus")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

EARLY_ACCESS_LINE = (
    "Autotune is free during early access. `throttle login` saves runs to Throttle Cloud."
)

DEMO_MODEL = "demo-model"
DEMO_URL = "http://127.0.0.1:8000"
DEMO_RATE = 2.00
DEMO_LAUNCH = (
    "vllm serve {model} --tensor-parallel-size {tp} --max-num-seqs {seqs} "
    "--gpu-memory-utilization {mem} {quant}"
)
DEMO_SPACE: dict[str, list[str]] = {
    "tp": ["1", "2"],
    "seqs": ["64", "128", "256", "512"],
    "mem": ["0.85", "0.90", "0.95"],
    "quant": ["", "--quantization fp8"],
}

AUTOTUNE_DESCRIPTION = __doc__.split("\n\n", 1)[1] if __doc__ else ""
AUTOTUNE_EPILOG = """\
examples:
  # simulated, no GPU, no login, nothing charged:
  throttle autotune --dry-run --monthly-tokens 2B

  # real: Throttle starts and stops the server for every trial; free
  # during early access (after `throttle login` runs are saved to Throttle Cloud)
  throttle autotune --url http://localhost:8000 --model Qwen/Qwen2.5-7B-Instruct \\
      --gpu-hourly-rate 2.49 --concurrency 16 --monthly-tokens 2B \\
      --launch "vllm serve {model} --tensor-parallel-size {tp} --max-num-seqs {seqs} --gpu-memory-utilization {mem}" \\
      --space tp=1,2 seqs=64,128,256 mem=0.85,0.9

  --space also takes a YAML or JSON file mapping each placeholder to its values.
  Values are inserted verbatim into the command, which runs with /bin/sh.
"""

# check options that mean nothing to autotune: hidden from its --help.
_HIDDEN_CHECK_OPTIONS = (
    "history", "limit", "against", "no_save", "no_calibrate", "fail_if_costlier",
    "share", "share_id", "push", "project", "label",
)
# ...and refused if given, instead of being silently ignored.
_REFUSED_CHECK_OPTIONS = {
    "history": "--history", "against": "--against", "no_save": "--no-save",
    "fail_if_costlier": "--fail-if-costlier", "share": "--share", "share_id": "--share-id",
}


class AutotuneError(Exception):
    """A usage problem found before any trial runs."""


class TrialError(Exception):
    """One trial could not be measured (server did not start, check failed)."""


class ServerStuck(Exception):
    """A server could not be stopped (or something else answers on its URL):
    any further trial would measure the wrong server, so the run ends."""


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------


def add_autotune_arguments(parser: argparse.ArgumentParser) -> None:
    """``throttle check``'s measuring options plus the search options."""

    check_module.add_check_arguments(parser)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    for action in parser._actions:
        if action.dest in _HIDDEN_CHECK_OPTIONS:
            action.help = argparse.SUPPRESS
        elif action.dest == "gpu_hourly_rate":
            action.help = (
                "what ONE GPU costs you per hour, in dollars (ASSUMED). Each trial "
                "is priced at this x its GPU count (see --gpus-param), so a config "
                "on more GPUs pays for them"
            )
        elif action.dest == "gpus":
            action.help = (
                "GPU count of every trial when no --space setting is the GPU count "
                "(see --gpus-param; default 1)"
            )
        elif action.dest == "json_output":
            action.help = "also write the autotune result (every trial, the best config) as JSON"
        elif action.dest == "monthly_tokens":
            action.help = (
                "your monthly volume of the headline tokens (e.g. 500M, 2B) to project "
                "the monthly $ saved by the best config"
            )
        elif action.dest == "history_dir":
            action.help = (
                "where trial checks are stored, one folder per run "
                "(default: ~/.throttle/autotune/RUN_ID)"
            )
    group = parser.add_argument_group("autotune")
    group.add_argument(
        "--launch", metavar="TEMPLATE",
        help="command that starts the server, with {placeholders} for every --space "
        "setting and optionally {model}, e.g. \"vllm serve {model} "
        "--tensor-parallel-size {tp} --max-num-seqs {seqs}\". Runs with /bin/sh; "
        "Throttle stops it (its whole process group) after each trial",
    )
    group.add_argument(
        "--space", nargs="+", metavar="KEY=V1,V2",
        help="search space: one KEY=V1,V2,... per setting (values in order, so the "
        "search can walk up or down), or one YAML/JSON file mapping KEY to a list",
    )
    group.add_argument(
        "--baseline", nargs="+", metavar="KEY=VALUE", dest="baseline_pairs",
        help="the config you run today (default: the first value of every setting)",
    )
    group.add_argument(
        "--max-trials", type=check_module._positive_int, default=DEFAULT_MAX_TRIALS,
        metavar="N",
        help=f"candidate configs to measure at most, after the baseline (default: "
        f"{DEFAULT_MAX_TRIALS}). Free during early access; logged in, each trial, the "
        "baseline included, uses $1.00 of Throttle Cloud credits",
    )
    group.add_argument(
        "--baseline-runs", type=check_module._positive_int, default=DEFAULT_BASELINE_RUNS,
        metavar="N",
        help="checks of the baseline config, to calibrate run-to-run noise (default: "
        f"{DEFAULT_BASELINE_RUNS}, the minimum for a verdict)",
    )
    group.add_argument(
        "--gpus-param", metavar="KEY",
        help="the setting that is the GPU count of a trial (default: tp, "
        "tensor_parallel_size, gpus or num_gpus when in --space; 'none' to turn off "
        "and use --gpus for every trial)",
    )
    group.add_argument(
        "--health-url", metavar="URL",
        help="URL that answers HTTP 200 once the server is ready (default: <url>/v1/models)",
    )
    group.add_argument(
        "--startup-timeout", type=check_module._positive_float,
        default=DEFAULT_STARTUP_TIMEOUT, metavar="SECONDS",
        help=f"give up on a trial whose server is not ready in this long (default: "
        f"{DEFAULT_STARTUP_TIMEOUT:g})",
    )
    group.add_argument(
        "--stop-command", metavar="TEMPLATE",
        help="extra command run after each trial (same placeholders), e.g. "
        "\"docker rm -f vllm\", for servers that outlive the launch command",
    )
    group.add_argument(
        "--dry-run", action="store_true",
        help="simulate the server and every trial: no GPU, no launch, no login, "
        "nothing charged; every number is SIMULATED. Without --launch/--space it "
        "runs a built-in demo space",
    )


# --------------------------------------------------------------------------
# Search space
# --------------------------------------------------------------------------


def _value_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def parse_space(items: Sequence[str]) -> dict[str, list[str]]:
    """``KEY=V1,V2`` items, or one YAML/JSON file, into ordered value lists."""

    if len(items) == 1 and "=" not in items[0]:
        path = Path(items[0]).expanduser()
        try:
            text = path.read_text()
        except OSError as exc:
            raise AutotuneError(f"--space {items[0]}: {exc}") from None
        try:
            data = json.loads(text)
        except ValueError:
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError:
                raise AutotuneError(
                    f"--space {items[0]} is not JSON, and reading YAML needs PyYAML "
                    "(pip install 'throttle-pro[config]')"
                ) from None
            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError as exc:
                raise AutotuneError(f"--space {items[0]}: invalid YAML ({exc})") from None
        if not isinstance(data, dict) or not data:
            raise AutotuneError(f"--space {items[0]} must map each setting to a list of values")
        raw = {str(k): v if isinstance(v, list) else [v] for k, v in data.items()}
    else:
        raw = {}
        for item in items:
            key, sep, values = item.partition("=")
            if not sep or not key:
                raise AutotuneError(f"--space {item!r}: expected KEY=V1,V2,...")
            if key in raw:
                raise AutotuneError(f"--space: {key} is given twice")
            raw[key] = values.split(",")
    space: dict[str, list[str]] = {}
    for key, values in raw.items():
        if not check_module._CONFIG_KEY.fullmatch(key):
            raise AutotuneError(
                f"--space: {key!r} is not a valid setting name (letters, digits, _ . -)"
            )
        texts = [_value_text(v).strip() for v in values]
        if len(set(texts)) != len(texts):
            raise AutotuneError(f"--space: {key} lists a value twice")
        if not texts:
            raise AutotuneError(f"--space: {key} has no values")
        space[key] = texts
    return space


def parse_baseline(pairs: Sequence[str] | None, space: Mapping[str, list[str]]) -> dict[str, str]:
    baseline = {key: values[0] for key, values in space.items()}
    for pair in pairs or ():
        key, sep, value = pair.partition("=")
        if not sep or key not in space:
            raise AutotuneError(
                f"--baseline {pair!r}: expected KEY=VALUE with KEY one of {', '.join(space)}"
            )
        if value not in space[key]:
            raise AutotuneError(
                f"--baseline {pair!r}: {value!r} is not in --space {key} "
                f"({', '.join(space[key])})"
            )
        baseline[key] = value
    return baseline


def template_fields(template: str) -> set[str]:
    try:
        return {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}
    except ValueError as exc:
        raise AutotuneError(f"--launch: {exc}") from None


def render(template: str, config: Mapping[str, str], model: str | None) -> str:
    values = {"model": model or "", **config}
    return " ".join(template.format_map(values).split())


def config_text(config: Mapping[str, str]) -> str:
    return " ".join(f"{k}={v if v != '' else chr(39) * 2}" for k, v in config.items())


def _freeze(config: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(config.items()))


# --------------------------------------------------------------------------
# Server backends
# --------------------------------------------------------------------------


class ServerLauncher:
    """Starts the real server for one trial and stops it afterwards."""

    simulated = False

    def __init__(
        self,
        launch: str,
        *,
        model: str | None,
        health_url: str,
        startup_timeout: float,
        stop_command: str | None,
        log_dir: Path,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.launch = launch
        self.model = model
        self.health_url = health_url
        self.startup_timeout = startup_timeout
        self.stop_command = stop_command
        self.log_dir = log_dir
        self.headers = dict(headers or {})
        self.process: subprocess.Popen[bytes] | None = None
        self.config: dict[str, str] | None = None

    def _healthy(self) -> bool:
        try:
            response = httpx.get(self.health_url, headers=self.headers, timeout=5.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def preflight(self) -> None:
        if self._healthy():
            raise AutotuneError(
                f"something is already answering at {self.health_url}. Stop it first: "
                "autotune starts and stops the server itself for every trial, so it "
                "would otherwise measure the wrong server"
            )

    def _log_tail(self, path: Path, lines: int = 15) -> str:
        try:
            text = path.read_text(errors="replace").splitlines()[-lines:]
        except OSError:
            return ""
        return "\n".join("      " + line for line in text)

    def start(self, config: Mapping[str, str], name: str) -> None:
        if self._healthy():
            raise ServerStuck(
                f"something is already answering at {self.health_url} before trial "
                f"{name} started; stop it, or the trial would measure the wrong server"
            )
        command = render(self.launch, config, self.model)
        log_path = self.log_dir / f"{name}.server.log"
        self.config = dict(config)
        with open(log_path, "wb") as log:
            log.write(f"$ {command}\n".encode())
            log.flush()
            self.process = subprocess.Popen(
                ["/bin/sh", "-c", command],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                code = self.process.returncode
                self.process = None
                raise TrialError(
                    f"the server exited with code {code} before it was ready "
                    f"(log: {log_path})\n{self._log_tail(log_path)}"
                )
            if self._healthy():
                return
            time.sleep(HEALTH_POLL_SECONDS)
        self.stop()
        raise TrialError(
            f"the server was not ready within {self.startup_timeout:g}s (--startup-timeout; "
            f"log: {log_path})"
        )

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                process.wait(timeout=STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        if self.stop_command and self.config is not None:
            subprocess.run(
                ["/bin/sh", "-c", render(self.stop_command, self.config, self.model)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
        # The next trial must not measure a server that is still shutting down.
        deadline = time.monotonic() + STOP_TIMEOUT_SECONDS
        while self._healthy():
            if time.monotonic() >= deadline:
                raise ServerStuck(
                    f"the server still answers at {self.health_url} {STOP_TIMEOUT_SECONDS:g}s "
                    "after it was stopped; stop it by hand (or pass --stop-command)"
                )
            time.sleep(HEALTH_POLL_SECONDS)

    @contextlib.contextmanager
    def measuring(self) -> Iterator[None]:
        yield


class _VirtualClock:
    """perf_counter for check.py during a dry run: advanced by the simulator,
    so simulated trials are instant and their timing is exactly the model's."""

    def __init__(self) -> None:
        self.now = 1000.0

    def perf_counter(self) -> float:
        return self.now


class SimulatedServer:
    """A fake OpenAI-compatible server whose speed and answers depend on the
    trial config. Every number it produces is SIMULATED.

    The model is illustrative, not a benchmark: tensor parallel 2 is 1.56x
    faster on 2 GPUs (so 28% more per token; the ratio of Throttle's own
    MI300X run), max-num-seqs helps up to a point, more GPU memory helps a
    little, and fp8 quantization is much faster but every answer runs to the
    token cap (so OUTPUT CHANGED rejects it, as with on-the-fly FP8 on 72B).
    Settings it does not know get a fixed pseudo-random effect of up to 10%.
    """

    simulated = True
    SECONDS_PER_OUTPUT_TOKEN = 0.01
    SECONDS_PER_PROMPT_TOKEN = 0.0002
    JITTER = 0.03

    def __init__(self, model: str, concurrency: int, *, gpus_param: str | None = None) -> None:
        self.model = model
        self.concurrency = max(1, concurrency)
        self.gpus_param = gpus_param
        self.config: dict[str, str] = {}
        self.clock = _VirtualClock()
        self.rng = random.Random(0)
        self.trials = 0

    # -- the illustrative speed model --
    def speedup(self, config: Mapping[str, str]) -> tuple[float, bool]:
        factor, broken = 1.0, False
        for key, value in config.items():
            name = key.lower().replace("-", "_")
            text = value.lower()
            number: float | None
            try:
                number = float(value)
            except ValueError:
                number = None
            if name in ("tp", "tensor_parallel_size") and number:
                factor *= 1.56 ** math.log2(number)
            elif name in ("seqs", "max_num_seqs") and number:
                k = math.log2(max(number, 1.0) / 64.0)
                factor *= max(0.5, 1 + 0.2 * k - 0.04 * k * k)
            elif name in ("mem", "gpu_memory_utilization") and number:
                factor *= 1 + 0.8 * (number - 0.85)
            elif "quant" in name or "dtype" in name:
                if "fp8" in text or "int4" in text or "int8" in text:
                    factor *= 1.9
                    broken = True
            elif text:
                digest = hashlib.sha256(f"{name}={text}".encode()).digest()
                factor *= 0.9 + 0.2 * digest[0] / 255
        return factor, broken

    def start(self, config: Mapping[str, str], name: str) -> None:
        self.config = dict(config)
        self.trials += 1
        seed = hashlib.sha256(f"{name}|{config_text(config)}".encode()).digest()
        self.rng = random.Random(int.from_bytes(seed[:8], "big"))

    def stop(self) -> None:
        pass

    def preflight(self) -> None:
        pass

    async def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/models"):
            return httpx.Response(200, json={
                "object": "list",
                "data": [{"id": self.model, "object": "model", "owned_by": "vllm",
                          "max_model_len": 32768}],
            })
        if path.endswith("/chat/completions"):
            body = json.loads(request.content)
            words = sum(len(str(m.get("content", "")).split()) for m in body.get("messages", []))
            prompt_tokens = max(1, int(words * 1.3))
            max_tokens = int(body.get("max_tokens") or 64)
            speed, broken = self.speedup(self.config)
            if max_tokens <= check_module.TAG_PROBE_MAX_TOKENS:
                completion, finish = max_tokens, "length"
            elif broken:
                completion, finish = max_tokens, "length"
            else:
                completion = max(1, int(max_tokens * (0.6 + 0.3 * self.rng.random())))
                finish = "stop"
            latency = (
                completion * self.SECONDS_PER_OUTPUT_TOKEN
                + prompt_tokens * self.SECONDS_PER_PROMPT_TOKEN
            ) / speed
            latency *= math.exp(self.rng.gauss(0.0, self.JITTER))
            self.clock.now += latency / self.concurrency
            return httpx.Response(200, json={
                "id": "sim", "object": "chat.completion", "model": self.model,
                "choices": [{"index": 0, "finish_reason": finish,
                             "message": {"role": "assistant",
                                         "content": "!!!!" if broken else "simulated answer"}}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion,
                          "total_tokens": prompt_tokens + completion},
            })
        return httpx.Response(404, text="not found")

    @contextlib.contextmanager
    def measuring(self) -> Iterator[None]:
        """Route check.py's HTTP to this simulator and its clock to ours."""

        shim = types.SimpleNamespace(**{
            name: getattr(time, name) for name in dir(time) if not name.startswith("__")
        })
        shim.perf_counter = self.clock.perf_counter
        old_transport, old_time = check_module._TRANSPORT, check_module.time
        check_module._TRANSPORT = httpx.MockTransport(self.handle)
        check_module.time = shim  # type: ignore[assignment]
        try:
            yield
        finally:
            check_module._TRANSPORT = old_transport
            check_module.time = old_time


# --------------------------------------------------------------------------
# Trials
# --------------------------------------------------------------------------


@dataclass
class Trial:
    index: int
    config: dict[str, str]
    role: str  # "baseline" or "candidate"
    status: str = "pending"  # measured, failed, not run
    check_ids: list[str] = field(default_factory=list)
    mean: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    gpus: int = 1
    verdict: str | None = None
    reason: str | None = None
    error: str | None = None
    log: str | None = None

    @property
    def eligible(self) -> bool:
        return self.role == "candidate" and self.verdict == check_module.VERDICT_CHEAPER

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_index": self.index,
            "role": self.role,
            "config": dict(self.config),
            "status": self.status,
            "gpus": self.gpus,
            "check_ids": list(self.check_ids),
            "usd_per_m": self.mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "verdict": self.verdict,
            "reason": self.reason,
            "eligible": self.eligible,
            "error": self.error,
            "log": self.log,
        }


class StopSearch(Exception):
    """Cloud metering refused a trial (402/403/401/offline): stop cleanly."""


class Runner:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        space: dict[str, list[str]],
        baseline: dict[str, str],
        gpus_param: str | None,
        backend: Any,
        client: cloud.CloudClient | None,
        run_dir: Path,
        run_id: str,
        out: Callable[[str], None],
    ) -> None:
        self.args = args
        self.space = space
        self.baseline_config = baseline
        self.gpus_param = gpus_param
        self.backend = backend
        self.client = client
        self.run_dir = run_dir
        self.run_id = run_id
        self.out = out
        self.trials: list[Trial] = []
        self.tried: dict[tuple[tuple[str, str], ...], Trial] = {}
        self.baseline_id: str | None = None
        self.noise_bound: float | None = None
        self.stopped: str | None = None
        self.current: Trial | None = None
        self.st = style_module.Style()

    # -- one trial --
    def gpus_for(self, config: Mapping[str, str]) -> int:
        if self.gpus_param:
            return int(config[self.gpus_param])
        return int(self.args.gpus or 1)

    def _check_namespace(
        self, config: Mapping[str, str], label: str, json_path: Path, against: str | None
    ) -> argparse.Namespace:
        ns = argparse.Namespace(**vars(self.args))
        gpus = self.gpus_for(config)
        static = list(self.args.config_pairs or [])
        ns.config_pairs = static + [(k, v) for k, v in config.items()]
        ns.label = label
        ns.gpus = gpus
        ns.gpu_hourly_rate = self.args.gpu_hourly_rate * gpus
        ns.json_output = json_path
        ns.history_dir = self.run_dir
        ns.against = against
        ns.no_calibrate = True
        ns.no_save = False
        ns.history = False
        ns.share = False
        ns.share_id = None
        ns.fail_if_costlier = None
        ns.push = False  # trials are metered as trials, never as $0.25 checks
        return ns

    def _meter(self, trial: Trial) -> None:
        if self.client is None:
            return
        try:
            body = self.client.autotune_trial(
                self.run_id, trial.index,
                {k: check_module.share_value(k, v) for k, v in trial.config.items()},
            )
        except cloud.CloudError as exc:
            if exc.kind in ("offline", "server"):
                # An unreachable cloud never blocks a run.
                self.out(self.st.warn("      warning: ") + f"{exc} Continuing locally, not metered.")
                self.client = None
                return
            raise StopSearch(str(exc)) from None
        if body.get("allowed") is False:
            raise StopSearch(
                f"Throttle Cloud did not allow trial {trial.index}. See {cloud.PRICING_URL}"
            )
        extra = cloud.credits_text(body)
        if extra:
            self.out(self.st.dim(f"      metered: {extra}"))

    def _run_check(self, ns: argparse.Namespace, log_path: Path) -> dict[str, Any]:
        with open(log_path, "a") as log, contextlib.redirect_stdout(log), \
                contextlib.redirect_stderr(log):
            print(f"$ throttle check (autotune trial, label {ns.label})")
            code = check_module.handle_check(ns)
        if code != check_module.EXIT_OK:
            raise TrialError(f"the check failed (exit {code}); see {log_path}")
        try:
            return json.loads(Path(ns.json_output).read_text())
        except (OSError, ValueError) as exc:
            raise TrialError(f"the check wrote no readable result ({exc}); see {log_path}") from None

    def run_trial(self, config: dict[str, str], role: str) -> Trial:
        index = len(self.trials)
        trial = Trial(index=index, config=dict(config), role=role, gpus=self.gpus_for(config))
        self.trials.append(trial)
        self.tried[_freeze(config)] = trial
        name = f"trial-{index:02d}"
        log_path = self.run_dir / f"{name}.log"
        trial.log = str(log_path)
        self.out(f"  trial {index}  {config_text(config)}"
                 + self.st.dim(f"  ({'baseline' if role == 'baseline' else 'candidate'}, "
                               f"{trial.gpus} GPU{'s' if trial.gpus != 1 else ''})"))
        try:
            self._meter(trial)
        except StopSearch:
            trial.status = "not run"
            raise
        runs = self.args.baseline_runs if role == "baseline" else 1
        try:
            self.backend.start(config, name)
            try:
                with self.backend.measuring():
                    record: dict[str, Any] = {}
                    for run in range(runs):
                        ns = self._check_namespace(
                            config,
                            "autotune-baseline" if role == "baseline" else f"autotune-{index}",
                            self.run_dir / f"{name}.run{run + 1}.json",
                            None if role == "baseline" else self.baseline_id,
                        )
                        record = self._run_check(ns, log_path)
                        trial.check_ids.append(record["id"])
                        if role == "baseline" and runs > 1:
                            mean = record["result"][record["metric"]]["mean"]
                            self.out(self.st.dim(
                                f"      baseline run {run + 1} of {runs}: "
                                f"{check_module._money(mean)}/M [{self._label()}]"
                            ))
            finally:
                self.backend.stop()
        except TrialError as exc:
            trial.status = "failed"
            trial.error = str(exc)
            self.out(f"      {self.st.warn('FAILED')}  {exc}")
            return trial
        headline = record["result"][record["metric"]]
        trial.mean, trial.ci_low, trial.ci_high = (
            headline.get("mean"), headline.get("ci_low"), headline.get("ci_high")
        )
        trial.status = "measured"
        comparison = record.get("comparison") or {}
        if role == "baseline":
            self.baseline_id = record["id"]
        else:
            trial.verdict = comparison.get("verdict")
            trial.reason = comparison.get("reason")
            noise = comparison.get("noise") or {}
            if noise.get("calibrated"):
                self.noise_bound = noise.get("floor_percent")
        line = (
            f"      {self.st.bold(check_module._money(trial.mean))}/M "
            f"{self.st.dim(check_module._ci_text(headline) + ' [' + self._label() + ']')}"
        )
        if role == "candidate":
            pct = comparison.get("measured_delta_percent")
            pct_text = f" {pct:+.1f}% vs baseline" if pct is not None else ""
            line += f"  {self.st.verdict(str(trial.verdict))}{pct_text}"
        self.out(line)
        return trial

    def _label(self) -> str:
        return "SIMULATED" if self.backend.simulated else "MEASURED"

    # -- the search --
    @staticmethod
    def beats(trial: Trial, current: Trial) -> bool:
        """A step is taken only to a verified CHEAPER (vs the baseline) whose
        95% interval also lies entirely below the current best's: a config
        that is only noise-cheaper than the best so far is not a move."""

        if not trial.eligible or trial.mean is None or current.mean is None:
            return False
        if current.role == "baseline":
            return True  # CHEAPER already means beyond noise and intervals apart
        if trial.ci_high is None or current.ci_low is None:
            return trial.mean < current.mean
        return trial.ci_high < current.ci_low

    @staticmethod
    def ties(trial: Trial, current: Trial) -> bool:
        """Verified CHEAPER than the baseline and not clearly worse than the best."""

        return (
            trial.eligible and trial.mean is not None and current.ci_high is not None
            and trial.mean <= current.ci_high
        )

    def best(self) -> Trial | None:
        """The cheapest verified CHEAPER config the search moved to."""

        return self.current if self.current is not None and self.current.role == "candidate" else None

    def search(self) -> None:
        baseline = self.run_trial(self.baseline_config, "baseline")
        if baseline.status != "measured":
            raise TrialError(f"the baseline could not be measured: {baseline.error}")
        current = dict(self.baseline_config)
        current_trial = baseline
        budget = self.args.max_trials
        improved = True
        while improved and budget > 0:
            improved = False
            # GPU count last: a bigger restart, and more GPUs rarely cut $/M.
            keys = sorted(self.space, key=lambda k: k == self.gpus_param)
            for key in keys:
                values = self.space[key]
                if budget <= 0:
                    break
                start = values.index(current[key])
                for direction in (1, -1):
                    moved = False
                    position = start + direction
                    ties = 0
                    while 0 <= position < len(values) and budget > 0:
                        candidate = {**current, key: values[position]}
                        trial = self.tried.get(_freeze(candidate))
                        if trial is None:
                            budget -= 1
                            trial = self.run_trial(candidate, "candidate")
                        if self.beats(trial, current_trial):
                            current, current_trial = candidate, trial
                            self.current = trial
                            moved = improved = True
                            ties = 0
                            position += direction
                            continue
                        if ties < 1 and self.ties(trial, current_trial):
                            # Within noise of the best: look one step further
                            # before giving up on this direction.
                            ties += 1
                            position += direction
                            continue
                        break  # early stop: this direction got no cheaper
                    if moved:
                        break  # found the better side; skip the other direction
        if budget <= 0:
            self.stopped = f"reached --max-trials {self.args.max_trials}"


# --------------------------------------------------------------------------
# Command
# --------------------------------------------------------------------------


def _new_run_id() -> str:
    return "at_" + secrets.token_hex(6)


def _conservative_monthly(
    run_dir: Path, baseline_id: str, best_id: str, tokens: int
) -> float | None:
    """The `throttle savings` conservative figure for baseline -> best."""

    from .savings import SOURCE_OPERATOR, SavingsRefused, build_statement

    records, _ = check_module.load_history(run_dir)
    try:
        statement = build_statement(
            records, baseline_id, best_id, {"count": tokens, "source": SOURCE_OPERATOR},
            run_dir / check_module.HISTORY_FILENAME,
        )
    except SavingsRefused:
        return None
    return statement["savings"]["conservative_usd"]


def _validate(args: argparse.Namespace) -> None:
    for dest, flag in _REFUSED_CHECK_OPTIONS.items():
        if getattr(args, dest, None) not in (None, False):
            raise AutotuneError(f"{flag} is a `throttle check` option; autotune does not take it")


def handle_autotune(
    args: argparse.Namespace,
    *,
    backend: Any = None,
    client: cloud.CloudClient | None = None,
) -> int:
    """``backend`` and ``client`` are injectable for tests; by default they
    come from the arguments (a real launcher, or the simulator with
    --dry-run) and the saved login."""

    st = style_module.Style()
    try:
        _validate(args)
        if args.dry_run and not args.launch and not args.space:
            args.launch, space = DEMO_LAUNCH, {k: list(v) for k, v in DEMO_SPACE.items()}
        elif not args.launch or not args.space:
            raise AutotuneError("autotune needs --launch TEMPLATE and --space KEY=V1,V2 ...")
        else:
            space = parse_space(args.space)
        if args.dry_run:
            args.url = args.url or DEMO_URL
            args.model = args.model or DEMO_MODEL
            args.gpu_hourly_rate = args.gpu_hourly_rate or DEMO_RATE
        missing = [f for f, v in (("--url", args.url), ("--gpu-hourly-rate", args.gpu_hourly_rate)) if not v]
        if missing:
            raise AutotuneError("autotune needs " + ", ".join(missing))
        fields = template_fields(args.launch)
        unknown = sorted(fields - set(space) - {"model"})
        if unknown:
            raise AutotuneError(
                "--launch uses {" + "}, {".join(unknown) + "}, which --space does not define"
            )
        unused = [k for k in space if k not in fields]
        if unused:
            raise AutotuneError(
                f"--space {', '.join(unused)} never appears in --launch, so it would not "
                "change the server; add {" + "}, {".join(unused) + "} to the template"
            )
        if "model" in fields and not args.model:
            raise AutotuneError("--launch uses {model}: pass --model")
        if args.blocks < 3:
            raise AutotuneError("--blocks must be at least 3")
        if args.baseline_runs < check_module.CALIBRATION_RUNS:
            raise AutotuneError(
                f"--baseline-runs must be at least {check_module.CALIBRATION_RUNS}: fewer "
                "repeats cannot calibrate the noise bound, so no trial could win"
            )
        baseline = parse_baseline(args.baseline_pairs, space)
        gpus_param: str | None
        if args.gpus_param and args.gpus_param.lower() == "none":
            gpus_param = None
        elif args.gpus_param:
            if args.gpus_param not in space:
                raise AutotuneError(f"--gpus-param {args.gpus_param} is not a --space setting")
            gpus_param = args.gpus_param
        else:
            gpus_param = next((k for k in space if k.lower() in GPU_PARAM_NAMES), None)
        if gpus_param is not None:
            if args.gpus:
                raise AutotuneError(
                    f"--gpus conflicts with the GPU count from {gpus_param}; drop one "
                    "(or pass --gpus-param none)"
                )
            bad = [v for v in space[gpus_param] if not v.isdigit() or int(v) < 1]
            if bad:
                raise AutotuneError(
                    f"{gpus_param} is used as the GPU count, so its values must be whole "
                    f"numbers (got {', '.join(bad)}); pass --gpus-param none if it is not"
                )
        chat_url = check_module.normalize_chat_completions_url(
            args.url, allow_insecure_http=args.allow_insecure_http
        )
        max_gpus = max((int(v) for v in space[gpus_param]), default=1) if gpus_param else int(args.gpus or 1)
        _preflight_limits(args, max_gpus)
    except (AutotuneError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if client is None and not args.dry_run:
        try:
            creds = cloud.load_credentials()
            client = cloud.CloudClient(creds.key) if creds else None
        except cloud.CloudError as exc:
            print(f"warning: {exc} Running locally.", file=sys.stderr)
            client = None

    run_id = _new_run_id()
    run_dir = (
        Path(args.history_dir).expanduser() if args.history_dir
        else Path.home() / ".throttle" / "autotune"
    ) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if backend is None:
        if args.dry_run:
            backend = SimulatedServer(args.model, args.concurrency, gpus_param=gpus_param)
        else:
            api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
            backend = ServerLauncher(
                args.launch, model=args.model,
                health_url=args.health_url or check_module._models_url(chat_url),
                startup_timeout=args.startup_timeout, stop_command=args.stop_command,
                log_dir=run_dir,
                headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
            )

    simulated = bool(getattr(backend, "simulated", False))
    label = "SIMULATED" if simulated else "MEASURED"
    sep = st.dim(f" {st.dot} ")
    print(f"{st.brand()} {st.dim(__version__)}{sep}autotune{sep}{run_id}"
          + (f"{sep}{st.warn('DRY RUN: simulated server, nothing charged')}" if simulated else ""))
    print(f"  launch     {args.launch}")
    for key, values in space.items():
        print(f"  space      {key}: {', '.join(v if v else chr(39) * 2 for v in values)}")
    print(f"  baseline   {config_text(baseline)}  ({args.baseline_runs} runs to calibrate noise)")
    gpu_text = (
        f"x GPU count from {gpus_param}" if gpus_param
        else f"x {int(args.gpus or 1)} GPU(s)"
    )
    print(f"  GPU rate   {check_module._money(args.gpu_hourly_rate)}/hr per GPU [ASSUMED] {gpu_text}")
    print(f"  budget     up to {args.max_trials} candidate trials after the baseline")
    if client is not None:
        try:
            me = client.me()
            print(f"  cloud      {cloud._account_line(me)}; each trial uses $1.00 of credits")
        except cloud.CloudError as exc:
            # Logged in but the cloud is unreachable (or rejects the key):
            # warn and run locally, free, instead of blocking.
            print(f"warning: {exc} Running locally, not metered.", file=sys.stderr)
            client = None
    elif not args.dry_run:
        print(f"  {EARLY_ACCESS_LINE}")
    print(f"  trials     logs and checks in {run_dir}")
    print()

    runner = Runner(
        args, space=space, baseline=baseline, gpus_param=gpus_param, backend=backend,
        client=client, run_dir=run_dir, run_id=run_id, out=print,
    )
    try:
        backend.preflight()
    except AutotuneError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        runner.search()
    except StopSearch as exc:
        runner.stopped = str(exc)
        print()
        print(f"Stopped: {exc}")
    except (TrialError, ServerStuck) as exc:
        print()
        print(f"Autotune stopped: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:
        with contextlib.suppress(Exception):
            backend.stop()
        print("\nInterrupted; the server was stopped.")
        return EXIT_FAILED

    baseline_trial = runner.trials[0] if runner.trials else None
    if baseline_trial is None or baseline_trial.status != "measured":
        return EXIT_FAILED
    best = runner.best()
    result = _result(args, runner, space, baseline_trial, best, simulated, chat_url)
    _print_result(st, args, runner, result, label)

    client = runner.client  # None if the cloud became unreachable mid-run
    if client is not None:
        try:
            response = client.autotune_result(run_id, _cloud_body(result))
            result["cloud_url"] = response.get("url")
            if response.get("url"):
                print(f"Saved to Throttle Cloud: {response['url']}")
        except cloud.CloudError as exc:
            print(f"warning: the result was not saved to Throttle Cloud: {exc}", file=sys.stderr)
    out_path = run_dir / "autotune.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.json_output:
        try:
            Path(args.json_output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(f"JSON written to {args.json_output}")
        except OSError as exc:
            print(f"Error: could not write {args.json_output}: {exc}", file=sys.stderr)
            return EXIT_FAILED
    return EXIT_OK


def _preflight_limits(args: argparse.Namespace, max_gpus: int) -> None:
    """Refuse before any trial (or charge) when the priciest trial would
    break a check safety limit."""

    try:
        prompts = check_module.load_prompts(args.prompts)
    except (OSError, ValueError) as exc:
        raise AutotuneError(f"could not load prompts: {exc}") from None
    mode = check_module.CACHE_MODE_WARM if args.warm_cache else check_module.CACHE_MODE_COLD
    probes = (
        check_module.tag_probe_count(prompts, args.requests_per_block, mode)
        if args.workload != "agent" else 0
    )
    ns = argparse.Namespace(**vars(args))
    ns.gpu_hourly_rate = args.gpu_hourly_rate * max_gpus
    problem, _ = check_module._limit_problem(ns, probes)
    if problem:
        raise AutotuneError(f"the {max_gpus}-GPU trial would be refused: {problem}")


def _result(
    args: argparse.Namespace,
    runner: Runner,
    space: Mapping[str, list[str]],
    baseline: Trial,
    best: Trial | None,
    simulated: bool,
    chat_url: str,
) -> dict[str, Any]:
    winner = best or baseline
    delta = None
    if best is not None and baseline.mean:
        delta = (best.mean - baseline.mean) / baseline.mean * 100  # type: ignore[operator]
    monthly = conservative = None
    if best is not None and args.monthly_tokens and baseline.mean is not None:
        monthly = (baseline.mean - best.mean) * args.monthly_tokens / 1e6  # type: ignore[operator]
        if runner.baseline_id and best.check_ids:
            conservative = _conservative_monthly(
                runner.run_dir, runner.baseline_id, best.check_ids[-1], args.monthly_tokens
            )
    return {
        "record_type": RECORD_TYPE,
        "record_version": RECORD_VERSION,
        "run_id": runner.run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "throttle_version": __version__,
        "simulated": simulated,
        "metric": args.metric,
        "endpoint": chat_url,
        "model": args.model,
        "launch_template": args.launch,
        "space": {k: list(v) for k, v in space.items()},
        "gpu_hourly_rate_per_gpu_usd": args.gpu_hourly_rate,
        "gpus_param": runner.gpus_param,
        "baseline_config": dict(baseline.config),
        "best_config": dict(winner.config),
        "found_cheaper": best is not None,
        "baseline": {"check_id": runner.baseline_id, "usd_per_m": baseline.mean,
                     "ci_low": baseline.ci_low, "ci_high": baseline.ci_high},
        "best": {"check_id": winner.check_ids[-1] if winner.check_ids else None,
                 "usd_per_m": winner.mean, "ci_low": winner.ci_low, "ci_high": winner.ci_high},
        "delta_percent": delta,
        "noise_bound_percent": runner.noise_bound,
        "monthly_tokens": args.monthly_tokens,
        "projected_monthly_savings_usd": monthly,
        "conservative_monthly_savings_usd": conservative,
        "launch_command": render(args.launch, winner.config, args.model),
        "trials": [t.to_dict() for t in runner.trials],
        "stopped": runner.stopped,
        "cloud_url": None,
    }


def _cloud_body(result: Mapping[str, Any]) -> dict[str, Any]:
    """POST /autotune/runs/{id}/result: configs and numbers only (no URL,
    no launch command, no paths)."""

    def clean(config: Mapping[str, str]) -> dict[str, str]:
        return {k: check_module.share_value(k, v) for k, v in config.items()}

    return {
        "best_config": clean(result["best_config"]),
        "baseline_usd_per_m": result["baseline"]["usd_per_m"],
        "best_usd_per_m": result["best"]["usd_per_m"],
        "found_cheaper": result["found_cheaper"],
        "metric": result["metric"],
        "noise_bound_percent": result["noise_bound_percent"],
        "monthly_tokens": result["monthly_tokens"],
        "projected_monthly_savings_usd": result["projected_monthly_savings_usd"],
        "conservative_monthly_savings_usd": result["conservative_monthly_savings_usd"],
        "trials": [
            {
                "trial_index": t["trial_index"], "role": t["role"], "config": clean(t["config"]),
                "status": t["status"], "gpus": t["gpus"], "usd_per_m": t["usd_per_m"],
                "ci_low": t["ci_low"], "ci_high": t["ci_high"], "verdict": t["verdict"],
                "eligible": t["eligible"],
            }
            for t in result["trials"]
        ],
    }


def _print_result(
    st: style_module.Style,
    args: argparse.Namespace,
    runner: Runner,
    result: Mapping[str, Any],
    label: str,
) -> None:
    metric_label = check_module.METRIC_LABELS[args.metric]
    money = check_module._money
    print()
    print(st.bold("Trials") + st.dim(f"  ($/M {metric_label}, 95% CI) [{label}]"))
    width = max(len(config_text(t.config)) for t in runner.trials) + 2
    for trial in runner.trials:
        if trial.status == "measured":
            figure = f"{money(trial.mean)} ({money(trial.ci_low)} to {money(trial.ci_high)})"
        else:
            figure = trial.status.upper()
        verdict = "BASELINE" if trial.role == "baseline" else (trial.verdict or "-")
        mark = (" * best" if trial is runner.current else " *") if trial.eligible else ""
        print(f"  {trial.index:>2}  {config_text(trial.config):<{width}}{figure:<28} "
              f"{st.verdict(verdict)}{mark}")
    print(st.dim("  * verified CHEAPER than the baseline: beyond the noise bound, "
                 "intervals apart, no OUTPUT CHANGED"))
    rejected = [t for t in runner.trials if t.verdict == check_module.VERDICT_OUTPUT_CHANGED]
    if rejected:
        print(st.dim(f"  {len(rejected)} config(s) rejected as OUTPUT CHANGED: cheaper "
                     "per token, but the answers changed length or hit the token cap"))
    if result["stopped"]:
        print(st.dim(f"  stopped early: {result['stopped']}"))
    rows: list[Any] = []
    base, best = result["baseline"], result["best"]
    if result["found_cheaper"]:
        rows.append(f"{st.verdict(check_module.VERDICT_CHEAPER)}  "
                    + st.bold(config_text(result["best_config"])))
        rows.append((
            f"$/M       {money(base['usd_per_m'])} -> {money(best['usd_per_m'])} "
            f"({result['delta_percent']:+.1f}%) {metric_label} [{label}]", (),
            "          ",
        ))
        if result["noise_bound_percent"] is not None:
            rows.append((f"noise     run-to-run bound {result['noise_bound_percent']:.1f}%", ()))
        if result["projected_monthly_savings_usd"] is not None:
            tokens = check_module._tokens_text(args.monthly_tokens)
            conservative = result["conservative_monthly_savings_usd"]
            rows.append((
                f"monthly   saves {money(result['projected_monthly_savings_usd'])} per month at "
                f"{tokens} {metric_label}"
                + (f" (conservative {money(conservative)})" if conservative is not None else "")
                + f" [PROJECTED: {label} $/M x ASSUMED volume]",
                (), "          ",
            ))
        elif not args.monthly_tokens:
            rows.append(("monthly   add --monthly-tokens N to project the monthly $ saved", ("2",)))
    else:
        rows.append(f"{st.verdict(check_module.VERDICT_NO_WINNER)}  "
                    + "no config was verifiably cheaper than the baseline")
        rows.append((f"$/M       baseline {money(base['usd_per_m'])} {metric_label} [{label}]; "
                     "keep your current config", ()))
    print()
    print(st.panel(rows, title="Autotune"))
    print(st.bold("Launch command") + st.dim(" (ready to paste):"))
    print(f"  {st.paint(result['launch_command'], '36')}")
    if label == "SIMULATED":
        print(st.dim("Every number above is SIMULATED (--dry-run): a made-up speed model, "
                     "not your server. Run without --dry-run to measure your own."))
