"""HTTP API and static file server for `throttle ui`.

Every figure the console shows comes from the check history and from
throttle.check (compare_checks / judge_recorded): the console never
recomputes a statistic of its own.
"""

from __future__ import annotations

import ipaddress
import re
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..check import (
    HISTORY_FILENAME,
    compare_checks,
    comparison_changes,
    flatten_fingerprint,
    judge_recorded,
    load_history,
)

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"
DOCS_DIR = HERE / "docs"
DEFAULT_PORT = 8787

CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "font-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)
LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1", "[::1]"}
SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*(/[a-z0-9][a-z0-9-]*)*$")

# Sidebar order for the docs, grouped the way the console shows them.
DOCS_NAV: list[dict[str, Any]] = [
    {"tab": "docs", "group": "Getting Started", "pages": [
        ("index", "Introduction"),
        ("getting-started/quickstart", "Quickstart"),
        ("getting-started/installation", "Installation"),
        ("getting-started/first-check", "Your first cost check"),
        ("getting-started/free-gpu", "Try on a free GPU"),
    ]},
    {"tab": "docs", "group": "Concepts", "pages": [
        ("concepts/cost-per-million-tokens", "Cost per million tokens"),
        ("concepts/noise-and-calibration", "Noise and calibration"),
        ("concepts/verdicts", "Verdicts"),
        ("concepts/cold-vs-warm-cache", "Cold vs warm cache"),
        ("concepts/config-fingerprint", "What changed"),
    ]},
    {"tab": "docs", "group": "Guides", "pages": [
        ("guides/vllm", "vLLM"),
        ("guides/sglang", "SGLang"),
        ("guides/ollama", "Ollama"),
        ("guides/other-engines", "Other servers"),
        ("guides/compare-gpus", "Compare two GPUs"),
        ("guides/test-quantization", "Test a quantization"),
        ("guides/ci-github-action", "Gate costs in CI"),
        ("guides/share-results", "Share your results"),
    ]},
    {"tab": "docs", "group": "More", "pages": [
        ("more/cost-audit", "Cost Audit"),
        ("more/pro", "Pro"),
        ("more/privacy", "Privacy"),
        ("more/faq", "FAQ"),
    ]},
    {"tab": "cli", "group": "CLI Reference", "pages": [
        ("cli/overview", "Overview"),
        ("cli/check", "throttle check"),
        ("cli/cost", "throttle cost"),
        ("cli/watch", "throttle watch"),
        ("cli/demo", "throttle demo"),
        ("cli/evidence-pipeline", "plan · smoke · benchmark · golden"),
        ("cli/measure-compare", "measure · compare · report"),
        ("cli/proxy-sessions", "proxy · sessions"),
    ]},
    {"tab": "docs", "group": "Results", "pages": [
        ("results/overview", "Results"),
        ("results/a100-max-num-seqs", "One vLLM flag on an A100"),
        ("results/nothing-changed", "Nothing changed"),
        ("results/model-swap", "A real change on a laptop"),
        ("results/community", "Community results"),
    ]},
    {"tab": "docs", "group": "Changelog", "pages": [("changelog", "Changelog")]},
]


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------


def is_loopback(host: str) -> bool:
    host = host.strip("[]")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def mask_endpoint(url: str) -> str:
    """Host:port for display. Loopback hosts are shown; others are masked."""

    parts = urlsplit(url or "")
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    if not host:
        return "unknown"
    if is_loopback(host):
        return f"{host}{port}"
    labels = host.split(".")
    if len(labels) >= 2:
        return f"{labels[0][:1]}•••.{labels[-1]}{port}"
    return f"{host[:1]}•••{port}"


def _summary(record: Mapping[str, Any]) -> dict[str, Any]:
    metric = record.get("metric", "output")
    return dict((record.get("result") or {}).get(metric) or {})


def _card(record: Mapping[str, Any], comparison: Mapping[str, Any] | None) -> dict[str, Any]:
    fp = record.get("fingerprint") or {}
    noise = (comparison or {}).get("noise") or {}
    return {
        "id": record.get("id"),
        "created_at": record.get("created_at"),
        "label": fp.get("label"),
        "model": fp.get("model"),
        "endpoint_masked": mask_endpoint(record.get("endpoint", "")),
        "metric": record.get("metric", "output"),
        "summary": _summary(record),
        "gpu_hourly_rate_usd": fp.get("gpu_hourly_rate_usd"),
        "rate_source": fp.get("gpu_hourly_rate_source"),
        "throttle_version": record.get("throttle_version"),
        "baseline_id": (comparison or {}).get("previous_id"),
        "verdict": (comparison or {}).get("verdict"),
        "delta_percent": (comparison or {}).get("delta_percent"),
        "noise_floor_percent": noise.get("floor_percent"),
        "changes": comparison_changes(comparison) if comparison else [],
    }


class History:
    """The check history, re-read on every request so new checks show up."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def load(self) -> tuple[list[dict[str, Any]], int]:
        return load_history(self.directory)

    def cards(self) -> list[dict[str, Any]]:
        records, _ = self.load()
        return [_card(r, judge_recorded(records, i)[1]) for i, r in enumerate(records)]

    def find(self, records: Sequence[Mapping[str, Any]], check_id: str) -> int:
        for index, record in enumerate(records):
            if record.get("id") == check_id:
                return index
        raise HTTPException(status_code=404, detail=f"no check with id {check_id!r}")


def _doc_titles() -> dict[str, str]:
    return {slug: title for group in DOCS_NAV for slug, title in group["pages"]}


def _doc_path(slug: str) -> Path:
    if not SLUG_PATTERN.fullmatch(slug):
        raise HTTPException(status_code=404, detail="no such page")
    path = (DOCS_DIR / f"{slug}.md").resolve()
    if DOCS_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="no such page")
    return path


def _snippet(text: str, needle: str, width: int = 90) -> str:
    at = text.lower().find(needle)
    if at < 0:
        return text[:width].strip()
    start = max(0, at - width // 3)
    return ("…" if start else "") + text[start:start + width].replace("\n", " ").strip() + "…"


def quickstart_status(records: Sequence[Mapping[str, Any]], cwd: Path) -> list[dict[str, Any]]:
    judged = [judge_recorded(records, i)[1] for i in range(len(records))]
    calibrated = any(
        c and c.get("verdict") in {"CHEAPER", "MORE EXPENSIVE", "NO WINNER"} for c in judged
    )
    workflows = cwd / ".github" / "workflows"
    ci = False
    if workflows.is_dir():
        for path in workflows.glob("*.y*ml"):
            try:
                if "throttle" in path.read_text(encoding="utf-8", errors="replace").lower():
                    ci = True
                    break
            except OSError:
                continue
    return [
        {"key": "connect", "done": bool(records)},
        {"key": "first", "done": bool(records)},
        {"key": "calibrate", "done": calibrated},
        {"key": "ci", "done": ci},
    ]


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


def create_app(history_dir: Path, *, cwd: Path | None = None, allow_any_host: bool = False) -> FastAPI:
    history = History(history_dir)
    workdir = cwd or Path.cwd()
    app = FastAPI(title="Throttle Console", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def guard(request: Request, call_next):  # type: ignore[no-untyped-def]
        # DNS-rebinding guard: a loopback-bound console only answers to
        # loopback host names.
        if not allow_any_host:
            host = (request.headers.get("host") or "").rsplit(":", 1)[0]
            if host not in LOOPBACK_NAMES and not is_loopback(host):
                return PlainTextResponse("forbidden host", status_code=403)
        response: Response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/summary")
    def summary() -> dict[str, Any]:
        records, skipped = history.load()
        return {
            "version": __version__,
            "history_file": str(history_dir / HISTORY_FILENAME),
            "checks": len(records),
            "skipped_lines": skipped,
            "endpoints": len({r.get("endpoint") for r in records}),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/checks")
    def checks() -> list[dict[str, Any]]:
        return list(reversed(history.cards()))

    @app.get("/api/checks/{check_id}")
    def check_detail(check_id: str) -> dict[str, Any]:
        records, _ = history.load()
        index = history.find(records, check_id)
        record = records[index]
        baseline, comparison = judge_recorded(records, index)
        card = _card(record, comparison)
        card.update({
            "endpoint": record.get("endpoint"),
            "blocks": [
                {"dollars_per_million": (b or {}).get("dollars_per_million")}
                for b in record.get("blocks") or []
                if isinstance(b, dict)
            ],
            "result": record.get("result"),
            "workload": {
                k: v for k, v in (record.get("workload") or {}).items() if k != "prompt_nonce"
            },
            "fingerprint": flatten_fingerprint(record.get("fingerprint") or {}),
            "comparison": comparison,
            "baseline": _card(baseline, None) if baseline else None,
            "share_command": f"throttle check --share-id {record.get('id')}",
        })
        return card

    @app.get("/api/compare")
    def compare(a: str = Query(...), b: str = Query(...)) -> dict[str, Any]:
        records, _ = history.load()
        ia, ib = history.find(records, a), history.find(records, b)
        if ia == ib:
            raise HTTPException(status_code=400, detail="pick two different checks")
        previous, current = records[ia], records[ib]
        # Judge with only the history that existed when the later one ran.
        comparison = compare_checks(previous, current, records[: max(ia, ib) + 1])
        return {"a": _card(previous, None), "b": _card(current, None), "comparison": comparison}

    @app.get("/api/endpoints")
    def endpoints() -> list[dict[str, Any]]:
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for card in history.cards():
            key = (card["endpoint_masked"], card["model"] or "?")
            group = groups.setdefault(key, {
                "endpoint_masked": key[0], "model": key[1], "labels": [], "series": [],
            })
            if card["label"] and card["label"] not in group["labels"]:
                group["labels"].append(card["label"])
            group["series"].append({
                "id": card["id"], "created_at": card["created_at"], "label": card["label"],
                "mean": card["summary"].get("mean"), "ci_low": card["summary"].get("ci_low"),
                "ci_high": card["summary"].get("ci_high"), "verdict": card["verdict"],
            })
        return list(groups.values())

    @app.get("/api/quickstart-status")
    def quickstart() -> list[dict[str, Any]]:
        records, _ = history.load()
        return quickstart_status(records, workdir)

    @app.get("/api/docs-nav")
    def docs_nav() -> list[dict[str, Any]]:
        return [
            {"tab": g["tab"], "group": g["group"],
             "pages": [{"slug": s, "title": t} for s, t in g["pages"]]}
            for g in DOCS_NAV
        ]

    @app.get("/api/docs/{slug:path}")
    def doc(slug: str) -> JSONResponse:
        path = _doc_path(slug)
        return JSONResponse({
            "slug": slug,
            "title": _doc_titles().get(slug, slug),
            "markdown": path.read_text(encoding="utf-8"),
        })

    @app.get("/api/search")
    def search(q: str = Query("", max_length=100)) -> list[dict[str, Any]]:
        needle = q.strip().lower()
        if len(needle) < 2:
            return []
        hits: list[dict[str, Any]] = []
        for slug, title in _doc_titles().items():
            try:
                text = _doc_path(slug).read_text(encoding="utf-8")
            except HTTPException:
                continue
            score = (3 if needle in title.lower() else 0) + text.lower().count(needle)
            if score:
                hits.append({"kind": "doc", "slug": slug, "title": title,
                             "snippet": _snippet(text, needle), "score": score})
        for card in history.cards():
            hay = " ".join(str(card.get(k) or "") for k in ("id", "label", "model", "verdict"))
            if needle in hay.lower():
                mean = card["summary"].get("mean")
                hits.append({
                    "kind": "check", "id": card["id"], "title": card["label"] or card["id"],
                    "snippet": f"{card['model']} · ${mean:.2f}/M · {card['verdict'] or 'first check'}"
                    if isinstance(mean, (int, float)) else card["model"],
                    "score": 5,
                })
        hits.sort(key=lambda h: -h["score"])
        return hits[:20]

    return app


def run_ui(history_dir: Path, host: str = "127.0.0.1", port: int = DEFAULT_PORT, open_browser: bool = True) -> int:
    import uvicorn

    loopback = is_loopback(host)
    if not loopback:
        print(
            f"WARNING: binding the Throttle Console to {host}, not a loopback address. "
            "Anyone who can reach this port can read your check history, including "
            "endpoint URLs. Use --host 127.0.0.1 unless you mean it.",
            file=sys.stderr,
        )
    shown = f"[{host}]" if ":" in host else host
    url = f"http://{'127.0.0.1' if host in {'0.0.0.0', '::'} else shown}:{port}/"
    records, skipped = load_history(history_dir)
    print(f"Throttle Console {__version__}")
    print(f"  History: {(history_dir / HISTORY_FILENAME).expanduser().resolve()} ({len(records)} checks"
          + (f", {skipped} unreadable lines skipped" if skipped else "") + ")")
    print(f"  Open:    {url}")
    if loopback:
        print("  Local only: the console makes no network requests. Ctrl+C to stop.")
    else:
        print("  Ctrl+C to stop.")
    if open_browser:
        try:
            webbrowser.open(url)
        except webbrowser.Error:
            pass
    app = create_app(history_dir, allow_any_host=not loopback)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0
