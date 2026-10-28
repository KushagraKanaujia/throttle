"""AI jobs: each place in the code that calls an AI API, tracked over time.

``throttle jobs`` keeps ``<root>/jobs/registry.json`` in step with the code::

    {"schema": "throttle.own.jobs/1",
     "jobs": {"<job id>": {
        "status": "testing" | "moved" | "api" | "retired",
        "site": {...},                 # the call site as last scanned (scan.CallSite)
        "route": {...} | null,         # set only while "moved": candidate, url, model,
                                       # the eval run and the prompt fingerprint it passed on
        "suspended": {...} | null,     # why a move was taken back, and the old route
        "first_seen", "updated_at", "ref",
        "history": [{"ts", "event", "detail"}]   # every decision, newest last
     }}}

Statuses, in plain words:

* ``testing``: calls go to the API while Throttle collects proof.
* ``moved``: your own model answers it, with fallback to the API. Only
  ``throttle jobs approve`` sets this, and only on a passing job-level eval
  of exactly the job's current version (:attr:`CallSite.approval_key`:
  prompt, model, tools, JSON mode and call settings). A job whose prompt
  cannot be read is never moved.
* ``api``: the team pinned it to the API. Throttle never moves it.
* ``retired``: the call is gone from the code; its route is removed.

``throttle jobs sync --repo .`` (run it when code lands on your main branch)
applies the code to the registry:

* a new call site becomes a ``testing`` job;
* a removed call site is ``retired`` and its route is removed;
* a call site whose prompt, model, tools or JSON mode changed loses its
  move: status goes back to ``testing``, the old route is kept under
  ``suspended`` for the record, and calls go to the API until a new eval of
  the new prompt passes and the job is approved again;
* a call site that moved to another file or function with the same prompt
  keeps its state under its new id.

Even before a sync, :class:`Router` serves a moved job from your model only
when the request still matches the version that was approved: every
readable prompt template, the model, tools and JSON mode
(:func:`request_fits_site`).

Once a job is in the registry, only its own state decides where it runs;
workload-type routes (``throttle route approve --type``) apply only to
traffic that is not a tracked job.

Matching traffic to jobs: the ``X-Throttle-Job`` header when the app sends
one (see :mod:`throttle.own.tag`), else the prompt template: the job's system
prompt (or first user message) with its ``{slots}`` as wildcards must match
the request's text, whitespace collapsed. At least 12 literal characters are
needed; the most specific template wins.

Lessons (stub for a later fine-tune loop): every fallback of a tracked job
and every judged loss in ``throttle jobs eval`` is appended to ``<root>/jobs/lessons/<job>.jsonl`` with
the conversation (already redacted at capture), the API's answer and your
model's answer. Nothing trains on them yet.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import threading
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import httpx

from . import evaluate as ev
from . import judge as judge_mod
from . import records
from . import scan as scan_mod

SCHEMA = "throttle.own.jobs/1"
EVAL_SCHEMA = "throttle.own.job-eval/1"
LESSON_SCHEMA = "throttle.own.lesson/1"

TESTING = "testing"
MOVED = "moved"
PINNED = "api"
RETIRED = "retired"
STATUSES = (TESTING, MOVED, PINNED, RETIRED)
STATUS_WORDS = {
    TESTING: "on the API, being tested",
    MOVED: "on your model (API fallback on)",
    PINNED: "pinned to the API",
    RETIRED: "removed from the code",
}
PASSING = (ev.BETTER, ev.NO_DIFF)
EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


def jobs_dir(root: Path) -> Path:
    return Path(root) / "jobs"


def registry_path(root: Path) -> Path:
    return jobs_dir(root) / "registry.json"


# ---------------------------------------------------------------------------
# Matching traffic to jobs
# ---------------------------------------------------------------------------


def first_user_text(norm: records.Normalized) -> str:
    for message in norm.messages:
        if message.role == "user":
            return message.text
    return ""


class JobMatcher:
    """Which job a request belongs to, from a set of call sites."""

    def __init__(self, sites: Iterable[scan_mod.CallSite]) -> None:
        self.by_id: dict[str, scan_mod.CallSite] = {}
        for site in sites:
            self.by_id.setdefault(site.job, site)
        self._templated = sorted(
            ((site, *site.match_template) for site in self.by_id.values() if site.match_template is not None),
            key=lambda item: -item[2].literal_chars,
        )

    def __contains__(self, job: str) -> bool:
        return job in self.by_id

    @staticmethod
    def _text(norm: records.Normalized, role: str) -> str:
        return norm.system_text if role == "system" else first_user_text(norm)

    def fits(self, job: str, norm: records.Normalized) -> bool:
        """Does the request match this job's current prompt template? True
        when the job has no readable template (nothing to compare)."""
        site = self.by_id.get(job)
        if site is None:
            return False
        found = site.match_template
        if found is None:
            return True
        role, template = found
        return template.matches(self._text(norm, role))

    def fits_exactly(self, job: str, model: str | None, norm: records.Normalized | None) -> bool:
        """Stricter than :meth:`fits`, for serving and judging a job: the
        request must fit every prompt template the code has (system and first
        user message), and use the same model, tools and JSON mode."""
        site = self.by_id.get(job)
        return site is not None and request_fits_site(site, model, norm)

    def match(self, exchange: records.Exchange | None, norm: records.Normalized | None, *,
              header: str | None = None) -> str | None:
        """The job id, or None. A header naming a known job wins (when the
        request still fits its template); otherwise the most specific
        template that matches."""
        tagged = header or (exchange.job if exchange is not None else None)
        if tagged and tagged in self.by_id:
            if norm is None or self.fits(tagged, norm):
                return tagged
            return None
        if norm is None:
            return None
        best: scan_mod.CallSite | None = None
        best_chars = -1
        model = exchange.model if exchange is not None else None
        for site, role, template in self._templated:
            if template.literal_chars < best_chars:
                break
            if not template.matches(self._text(norm, role)):
                continue
            if best is None or (template.literal_chars == best_chars and site.model and site.model == model):
                best, best_chars = site, template.literal_chars
        return best.job if best is not None else None


def bare_model(name: str | None) -> str:
    return (name or "").strip().lower().rsplit("/", 1)[-1]


def request_fits_site(site: scan_mod.CallSite, model: str | None, norm: records.Normalized | None) -> bool:
    """Is this request what the code at ``site`` sends? Model (when the code
    names it), tools, JSON mode and every readable prompt template must agree.
    Anything Throttle cannot confirm counts as not fitting."""
    if site.model and bare_model(model) != bare_model(site.model):
        return False
    if norm is None:
        return False
    if bool(norm.tools) != site.has_tools:
        return False
    if (norm.response_format in ("json", "json_schema")) != site.wants_json:
        return False
    if site.system is not None and not site.system.matches(norm.system_text, min_chars=0):
        return False
    if site.user is not None and not site.user.matches(first_user_text(norm), min_chars=0):
        return False
    return True


# ---------------------------------------------------------------------------
# Comparing two versions of a job
# ---------------------------------------------------------------------------


def changes(before: scan_mod.CallSite, after: scan_mod.CallSite) -> list[str]:
    """What changed between two versions of a call site, in plain words.
    Empty when nothing that affects the answers changed."""
    out: list[str] = []
    if before.prompt_fingerprint != after.prompt_fingerprint:
        if before.prompt_fingerprint is None:
            out.append("prompt text is now readable")
        elif after.prompt_fingerprint is None:
            out.append("prompt is now built at run time")
        else:
            out.append("prompt edited")
    b_model, a_model = before.model or before.model_expr, after.model or after.model_expr
    if b_model != a_model:
        out.append(f"model changed from {b_model or 'unknown'} to {a_model or 'unknown'}")
    if before.has_tools != after.has_tools:
        out.append("tools added" if after.has_tools else "tools removed")
    if before.wants_json != after.wants_json:
        out.append("JSON output turned on" if after.wants_json else "JSON output turned off")
    if before.provider != after.provider and after.provider != scan_mod.PROVIDER_UNKNOWN:
        out.append(f"provider changed from {before.provider} to {after.provider}")
    if not out and before.call_digest and after.call_digest and before.call_digest != after.call_digest:
        out.append("call settings edited (messages, tools, schema or options)")
    return out


def pair_moves(removed: Sequence[scan_mod.CallSite], added: Sequence[scan_mod.CallSite]
               ) -> list[tuple[scan_mod.CallSite, scan_mod.CallSite]]:
    """Removed and added call sites that are the same job moved to a new
    place: same readable prompt and the same model, tools and JSON mode."""
    pairs: list[tuple[scan_mod.CallSite, scan_mod.CallSite]] = []
    used: set[int] = set()
    for old in removed:
        if old.prompt_fingerprint is None:
            continue
        for index, new in enumerate(added):
            if index in used or new.prompt_fingerprint != old.prompt_fingerprint:
                continue
            if changes(old, new):
                continue
            pairs.append((old, new))
            used.add(index)
            break
    return pairs


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class JobEntry:
    job: str
    site: dict[str, Any]
    status: str = TESTING
    route: dict[str, Any] | None = None
    suspended: dict[str, Any] | None = None
    first_seen: str = ""
    updated_at: str = ""
    ref: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def call_site(self) -> scan_mod.CallSite:
        return scan_mod.CallSite.from_json(self.site)

    def log(self, event: str, detail: str, ts: str | None = None) -> None:
        stamp = ts or records.utc_now_iso()
        self.history.append({"ts": stamp, "event": event, "detail": detail})
        self.updated_at = stamp

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status, "site": self.site, "route": self.route, "suspended": self.suspended,
            "first_seen": self.first_seen, "updated_at": self.updated_at, "ref": self.ref, "history": self.history,
        }

    @classmethod
    def from_json(cls, job: str, data: Mapping[str, Any]) -> "JobEntry":
        status = data.get("status") if data.get("status") in STATUSES else TESTING
        return cls(
            job=job, site=dict(data.get("site") or {}), status=status,
            route=dict(data["route"]) if isinstance(data.get("route"), Mapping) else None,
            suspended=dict(data["suspended"]) if isinstance(data.get("suspended"), Mapping) else None,
            first_seen=str(data.get("first_seen") or ""), updated_at=str(data.get("updated_at") or ""),
            ref=data.get("ref"), history=list(data.get("history") or []),
        )


def load_registry(root: Path) -> dict[str, JobEntry]:
    path = registry_path(root)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    jobs = data.get("jobs") if isinstance(data, Mapping) else None
    if not isinstance(jobs, Mapping):
        raise ValueError(f"{path}: expected {{'jobs': {{...}}}}")
    out: dict[str, JobEntry] = {}
    for job, entry in jobs.items():
        if isinstance(entry, Mapping) and isinstance(entry.get("site"), Mapping) and entry["site"].get("path"):
            out[job] = JobEntry.from_json(job, entry)
    return out


def save_registry(root: Path, registry: Mapping[str, JobEntry]) -> Path:
    path = registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
    doc = {"schema": SCHEMA, "jobs": {k: v.to_json() for k, v in sorted(registry.items())}}
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


@dataclass
class SyncEvent:
    job: str
    event: str      # added | changed | suspended | retired | moved | back | updated
    detail: str

    def to_json(self) -> dict[str, str]:
        return {"job": self.job, "event": self.event, "detail": self.detail}


def plan_sync(registry: Mapping[str, JobEntry], sites: Sequence[scan_mod.CallSite], *, ref: str | None = None,
              ts: str | None = None) -> tuple[dict[str, JobEntry], list[SyncEvent]]:
    """The registry after applying the scanned code, and what changed. Pure:
    the input registry is not modified."""
    ts = ts or records.utc_now_iso()
    out = {k: JobEntry.from_json(k, json.loads(json.dumps(v.to_json()))) for k, v in registry.items()}
    events: list[SyncEvent] = []
    current = {site.job: site for site in sites}
    live = {k for k, e in out.items() if e.status != RETIRED}
    missing = [out[k].call_site for k in sorted(live - set(current))]
    fresh = [current[k] for k in sorted(set(current) - set(out))]
    moves = pair_moves(missing, fresh)
    for old, new in moves:
        entry = out.pop(old.job)
        entry.job = new.job
        entry.site = new.to_json()
        entry.ref = ref
        entry.log("moved", f"moved from {old.location} ({old.job}) to {new.location}; same prompt, state kept", ts)
        out[new.job] = entry
        events.append(SyncEvent(new.job, "moved", f"moved from {old.job}; same prompt, kept as {STATUS_WORDS[entry.status]}"))
    moved_new = {new.job for _, new in moves}
    moved_old = {old.job for old, _ in moves}
    for job, site in current.items():
        if job in moved_new:
            continue
        entry = out.get(job)
        if entry is None:
            entry = JobEntry(job=job, site=site.to_json(), first_seen=ts, ref=ref)
            entry.log("added", f"new AI call at {site.location}; on the API until tests pass", ts)
            out[job] = entry
            events.append(SyncEvent(job, "added", f"new AI call at {site.location}"))
            continue
        before = entry.call_site
        diff = changes(before, site)
        entry.site = site.to_json()
        entry.ref = ref
        if entry.status == RETIRED:
            entry.status = TESTING
            entry.log("back", f"back in the code at {site.location}; on the API until tests pass", ts)
            events.append(SyncEvent(job, "back", "back in the code; on the API until tests pass"))
            continue
        if not diff:
            if before.location != site.location:
                entry.updated_at = ts
            continue
        what = ", ".join(diff)
        if entry.status == MOVED:
            entry.suspended = {"at": ts, "reason": what, "route": entry.route}
            entry.route = None
            entry.status = TESTING
            entry.log("suspended", f"{what}: moved back to testing; calls go to the API until the new version passes", ts)
            events.append(SyncEvent(job, "suspended", f"{what}; was on your model, now back on the API until it passes again"))
        else:
            entry.log("changed", what, ts)
            events.append(SyncEvent(job, "changed", what))
    for job in sorted(live - set(current) - moved_old):
        entry = out[job]
        had_route = entry.status == MOVED
        entry.status = RETIRED
        entry.route = None
        entry.ref = ref
        entry.log("retired", "removed from the code" + ("; its route to your model is removed" if had_route else ""), ts)
        events.append(SyncEvent(job, "retired", "removed from the code" + ("; route removed" if had_route else "")))
    return out, events


def sync(root: Path, sites: Sequence[scan_mod.CallSite], *, ref: str | None = None) -> list[SyncEvent]:
    registry, events = plan_sync(load_registry(root), sites, ref=ref)
    save_registry(root, registry)
    return events


# ---------------------------------------------------------------------------
# Approvals and control
# ---------------------------------------------------------------------------


def job_eval_dir(root: Path) -> Path:
    return jobs_dir(root) / "evals"


def iter_job_evals(root: Path, candidate: str | None = None) -> list[dict[str, Any]]:
    out = []
    directory = job_eval_dir(root)
    if not directory.is_dir():
        return out
    for path in directory.glob("*.json"):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict) or doc.get("schema") != EVAL_SCHEMA:
            continue
        if candidate is not None and doc.get("candidate") != candidate:
            continue
        out.append(doc)
    return sorted(out, key=lambda d: (str(d.get("ts", "")), str(d.get("run_id", ""))))


_ANY = object()


def latest_job_result(root: Path, job: str, candidate: str | None = None,
                      site_key: Any = _ANY) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """(eval doc, the job's result in it) from the newest job eval that has
    this job. With ``site_key`` (a :attr:`CallSite.approval_key`), only an
    eval of exactly that version of the call counts: same prompt, model,
    tools, JSON mode and call settings. ``None`` matches nothing."""
    for doc in reversed(iter_job_evals(root, candidate)):
        result = (doc.get("jobs") or {}).get(job)
        if not isinstance(result, Mapping):
            continue
        if site_key is not _ANY and (site_key is None or result.get("site_key") != site_key):
            continue
        return doc, dict(result)
    return None


def approve_job(root: Path, job: str, label: str, url: str, model: str, *, api_key_env: str | None = None,
                canary: float | None = None) -> tuple[JobEntry | None, str]:
    """Move a job to your model. Refused unless the newest job eval for
    ``label`` on the job's current prompt said BETTER or NO MATERIAL DIFFERENCE."""
    registry = load_registry(root)
    entry = registry.get(job)
    if entry is None:
        return None, f"no job {job!r} in {registry_path(root)}. Run: throttle jobs sync --repo ."
    if entry.status == RETIRED:
        return None, f"{job} is no longer in the code"
    if entry.status == PINNED:
        return None, f"{job} is pinned to the API. Unpin it first: throttle jobs unpin {job}"
    site = entry.call_site
    fingerprint = site.prompt_fingerprint
    if fingerprint is None:
        return None, (f"Throttle cannot read the prompt of {job} in the code (it is built at run time or imported), "
                      "so it could not tell when the prompt changes. Jobs move only when their prompt is readable")
    site_key = site.approval_key
    found = latest_job_result(root, job, label, site_key)
    if found is None:
        any_run = latest_job_result(root, job, label)
        if any_run is not None:
            return None, (f"the newest eval of {job} for {label} tested an older version of the call (prompt, model, "
                          f"tools, JSON mode or settings). Test the current one: "
                          f"throttle jobs eval --repo . --candidate {label} ...")
        return None, f"no job eval of {job} for {label}. Run: throttle jobs eval --repo . --candidate {label} ..."
    doc, result = found
    verdict = result.get("verdict")
    if verdict not in PASSING:
        return None, f"eval run {doc.get('run_id')} gave {job} the verdict {verdict}; moving needs {ev.BETTER} or {ev.NO_DIFF}"
    if doc.get("candidate_model") and doc["candidate_model"] != model:
        return None, f"eval run {doc.get('run_id')} tested model {doc['candidate_model']}, not {model}"
    entry.route = {
        "candidate_label": label, "url": url, "model": model, "api_key_env": api_key_env,
        "approved_at": records.utc_now_iso(), "approved_eval_run_id": doc.get("run_id"), "eval_verdict": verdict,
        "mode": "canary" if canary is not None else "live", "canary_rate": canary, "forced": False,
        "prompt_fingerprint": fingerprint, "site_key": site_key,
    }
    entry.status = MOVED
    entry.suspended = None
    entry.log("approved", f"moved to {label} ({model}); eval run {doc.get('run_id')} said {verdict}")
    save_registry(root, registry)
    return entry, f"eval run {doc.get('run_id')} gave {job} {verdict}"


def set_status(root: Path, job: str, status: str, detail: str) -> JobEntry | None:
    registry = load_registry(root)
    entry = registry.get(job)
    if entry is None:
        return None
    entry.status = status
    entry.route = None
    entry.log(status if status != TESTING else "undone", detail)
    save_registry(root, registry)
    return entry


class RegistryCache:
    """The registry, read again whenever the file changes (for the router)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._key: tuple[int, int] | None = None
        self._entries: dict[str, JobEntry] = {}
        self._matcher = JobMatcher(())
        self._lock = threading.Lock()

    def refresh(self) -> None:
        path = registry_path(self.root)
        try:
            stat = path.stat()
            key = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            self._key, self._entries, self._matcher = None, {}, JobMatcher(())
            return
        if key == self._key:
            return
        with self._lock:
            try:
                entries = load_registry(self.root)
            except (OSError, ValueError):
                return
            live = [e.call_site for e in entries.values() if e.status != RETIRED]
            self._entries, self._matcher, self._key = entries, JobMatcher(live), key

    def entries(self) -> dict[str, JobEntry]:
        self.refresh()
        return self._entries

    def match(self, exchange: records.Exchange | None, norm: records.Normalized | None, header: str | None = None) -> str | None:
        self.refresh()
        if not self._entries:
            return header
        found = self._matcher.match(exchange, norm, header=header)
        return found or header

    def tracked(self, job: str | None) -> JobEntry | None:
        if not job:
            return None
        entry = self.entries().get(job)
        return entry if entry is not None and entry.status != RETIRED else None

    def approved_route(self, entry: JobEntry, norm: records.Normalized, model: str | None = None) -> dict[str, Any] | None:
        """The job's route, only while it is moved, it was approved on this
        exact version of the call, and the request still fits it: the prompt
        templates, model, tools and JSON mode it was approved on."""
        if entry.status != MOVED or not entry.route or entry.route.get("forced"):
            return None
        site = entry.call_site
        if site.prompt_fingerprint is None or entry.route.get("prompt_fingerprint") != site.prompt_fingerprint:
            return None
        if entry.route.get("site_key") != site.approval_key:
            return None
        if not request_fits_site(site, model, norm):
            return None
        return entry.route


# ---------------------------------------------------------------------------
# Lessons (stub: recorded for a future fine-tune loop, not used yet)
# ---------------------------------------------------------------------------


def lessons_path(root: Path, job: str) -> Path:
    digest = hashlib.sha1(job.encode()).hexdigest()[:8]
    return jobs_dir(root) / "lessons" / f"{records.slug(job)[:60]}-{digest}.jsonl"


def _messages_json(norm: records.Normalized | None) -> list[dict[str, Any]]:
    if norm is None:
        return []
    return [{"role": m.role, "text": m.text} for m in norm.messages]


def add_lesson(root: Path, job: str, *, source: str, reason: str | None, exchange_id: str | None,
               norm: records.Normalized | None, preferred: str | None, rejected: str | None,
               candidate: str | None = None, workload_type: str | None = None, redaction: str | None = None) -> Path:
    """Append one candidate training example. ``preferred`` is the answer to
    learn from (the API's); ``rejected`` is what your model said, when known."""
    record = {
        "schema": LESSON_SCHEMA, "ts": records.utc_now_iso(), "job": job, "source": source, "reason": reason,
        "exchange_id": exchange_id, "workload_type": workload_type, "candidate": candidate,
        "redaction": redaction, "messages": _messages_json(norm), "preferred": preferred, "rejected": rejected,
    }
    path = lessons_path(root, job)
    records.append_jsonl(path, record)
    return path


def lesson_from_fallback(root: Path, exchange: records.Exchange) -> Path | None:
    """A fallback of a tracked job: the API answered what your model could not."""
    if not exchange.fallback or not exchange.job:
        return None
    norm = records.normalize(exchange) if exchange.has_bodies else None
    return add_lesson(
        root, exchange.job, source="fallback", reason=exchange.route_reason, exchange_id=exchange.id,
        norm=norm, preferred=norm.output_text if norm else None, rejected=None,
        workload_type=exchange.workload_type, redaction=exchange.redaction,
    )


def lesson_counts(root: Path) -> dict[str, Counter[str]]:
    out: dict[str, Counter[str]] = {}
    directory = jobs_dir(root) / "lessons"
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.jsonl")):
        for record in records.read_jsonl(path):
            out.setdefault(str(record.get("job")), Counter())[str(record.get("source"))] += 1
    return out


# ---------------------------------------------------------------------------
# Job-level eval
# ---------------------------------------------------------------------------


def run_job_eval(
    root: Path,
    candidate: str,
    sites: Sequence[scan_mod.CallSite],
    *,
    judge_config: judge_mod.JudgeConfig | None,
    budget: judge_mod.JudgeBudget | None = None,
    margin: float = 0.05,
    min_n: int = 30,
    judge_concurrency: int = 4,
    transport: httpx.AsyncBaseTransport | None = None,
    run_id: str | None = None,
    classify: Callable[[records.Exchange, records.Normalized], str] | None = None,
    max_per_job: int | None = None,
) -> dict[str, Any]:
    """Evaluate the candidate per job, on traffic that fits each job's
    current prompt. Same rules as ``throttle eval``. Writes
    ``<root>/jobs/evals/<run_id>.json``; judged losses become lessons."""
    matcher = JobMatcher(sites)
    by_id = {s.job: s for s in sites}

    def group(exchange: records.Exchange, norm: records.Normalized) -> str | None:
        # only calls that are exactly this version of the job count as proof
        job = matcher.match(exchange, norm)
        if job is None or not matcher.fits_exactly(job, exchange.model, norm):
            return None
        return job

    excluded: dict[str, int] = {}
    sets = ev.build_eval_sets(root, candidate, classify=classify, excluded=excluded, group=group)
    for key in excluded:
        sets.setdefault(key, [])
    sets = dict(sorted(sets.items()))
    if max_per_job is not None:
        sets = {k: v[:max(0, max_per_job)] for k, v in sets.items()}
    outcomes: dict[str, ev.TypeOutcome] | None = None
    stopped = None
    if judge_config is not None and sets:
        budget = budget or judge_mod.JudgeBudget(max_calls=600)
        outcomes, stopped = asyncio.run(ev._judge_all(
            sets, config=judge_config, budget=budget, concurrency=judge_concurrency, transport=transport,
        ))
    models = Counter(p.shadow.candidate_model for pairs in sets.values() for p in pairs)
    ts = records.utc_now_iso()
    run_id = run_id or f"jobeval-{ts[:19].replace(':', '').replace('-', '')}-{records.slug(candidate)}-{uuid.uuid4().hex[:6]}"
    set_dir = job_eval_dir(root) / "sets"
    results: dict[str, Any] = {}
    for job, pairs in sets.items():
        outcome = (outcomes or {}).get(job) if outcomes is not None else None
        summary = ev.summarize_type(job, pairs, candidate=candidate, root=root, outcome=outcome, margin=margin,
                                    min_n=min_n, transport_excluded=excluded.get(job, 0), evalset_dir=set_dir)
        site = by_id.get(job)
        summary["prompt_fingerprint"] = site.prompt_fingerprint if site else None
        summary["site_key"] = site.approval_key if site else None
        kinds = Counter(p.workload for p in pairs)
        summary["workload_type"] = kinds.most_common(1)[0][0] if kinds else None
        results[job] = summary
        if outcome is not None:
            for pair in pairs:
                if outcome.by_exchange.get(pair.exchange.id) == judge_mod.LOSS:
                    add_lesson(root, job, source="judged_loss", reason="the judge preferred the API's answer in both orders"
                               if pair.shadow.ok else (pair.shadow.error or "your model failed"),
                               exchange_id=pair.exchange.id, norm=pair.norm, preferred=pair.norm.output_text,
                               rejected=pair.shadow.output_text, candidate=candidate, workload_type=pair.workload,
                               redaction=pair.exchange.redaction)
    document = {
        "schema": EVAL_SCHEMA, "run_id": run_id, "ts": ts, "candidate": candidate,
        "candidate_model": models.most_common(1)[0][0] if models else None,
        "judge": {"provider": judge_config.provider, "model": judge_config.model} if judge_config else None,
        "margin": margin, "min_n": min_n,
        "judge_calls": budget.calls if (budget and judge_config) else 0,
        "judge_usd": round(budget.usd, 6) if (budget and judge_config and budget.price) else None,
        "judge_stopped": stopped,
        "jobs": results,
    }
    path = job_eval_dir(root) / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    document["path"] = str(path)
    return document


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _root(args: argparse.Namespace) -> Path:
    return Path(args.dir).expanduser() if getattr(args, "dir", None) else records.default_root()


def _sites(args: argparse.Namespace) -> tuple[list[scan_mod.CallSite], str | None]:
    repo = Path(args.repo).expanduser()
    ref = getattr(args, "ref", None)
    if ref:
        return scan_mod.scan_ref(repo, ref), scan_mod.resolve_ref(repo, ref)
    return scan_mod.scan_dir(repo), None


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "jobs",
        help="find the AI calls in your code and track each one: tested, moved, pinned or retired",
        description=(
            "Each place in your code that calls OpenAI or Anthropic is a job. Throttle tracks where each "
            "job runs and keeps that in step with the code: a removed call loses its route, and a call "
            "whose prompt changed goes back to the API until it passes again."
        ),
    )
    dir_help = "data folder (default: ~/.throttle/own or $THROTTLE_OWN_DIR)"
    parser.add_argument("--dir", type=Path, default=None, help=dir_help)
    actions = parser.add_subparsers(dest="jobs_action",
                                    metavar="{scan,sync,list,log,eval,approve,undo,pin,unpin,lessons}")
    actions.required = True

    def repo_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--repo", default=".", help="the code to scan (default: current folder)")
        p.add_argument("--ref", default=None, help="scan this git ref instead of the files on disk")

    s = actions.add_parser("scan", help="list the AI calls in the code (reads nothing else)")
    repo_args(s)
    s.add_argument("--json", action="store_true")

    y = actions.add_parser("sync", help="apply the code to the job list (run when code lands on main)")
    repo_args(y)
    y.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    y.add_argument("--json", action="store_true")

    actions.add_parser("list", help="every job and where it runs")
    g = actions.add_parser("log", help="every decision for one job")
    g.add_argument("job")

    e = actions.add_parser("eval", help="judge your model per job, on captured traffic that fits each job's prompt")
    repo_args(e)
    e.add_argument("--candidate", required=True, help="candidate label used with `throttle shadow`")
    e.add_argument("--judge-url", default=None)
    e.add_argument("--judge-model", default=None)
    e.add_argument("--judge-provider", choices=judge_mod.PROVIDERS, default=records.PROVIDER_OPENAI)
    e.add_argument("--judge-key-env", default=None, metavar="VAR")
    e.add_argument("--no-judge", action="store_true")
    e.add_argument("--max-judge-calls", type=int, default=600)
    e.add_argument("--judge-timeout", type=float, default=120.0, help="seconds per judge call (default 120)")
    e.add_argument("--judge-concurrency", type=int, default=4, help="judge calls in flight (default 4)")
    e.add_argument("--max-per-job", type=int, default=None, metavar="N", help="judge at most N calls per job")
    e.add_argument("--margin", type=float, default=0.05)
    e.add_argument("--min-n", type=int, default=30)
    e.add_argument("--json", action="store_true")

    a = actions.add_parser("approve", help="move one job to your model (needs a passing job eval of its current prompt)")
    a.add_argument("job")
    a.add_argument("--candidate", required=True)
    a.add_argument("--url", required=True, help="OpenAI-compatible server, e.g. http://gpu-box:8000/v1")
    a.add_argument("--model", required=True)
    a.add_argument("--api-key-env", default=None, metavar="VAR")
    a.add_argument("--canary", type=float, default=None, metavar="RATE")

    for name, text in (("undo", "send a moved job back to the API (it stays in testing)"),
                       ("pin", "keep a job on the API for good; Throttle will not move it"),
                       ("unpin", "let Throttle test a pinned job again")):
        p = actions.add_parser(name, help=text)
        p.add_argument("job")
    actions.add_parser("lessons", help="misses saved per job for a future fine-tune (not used yet)")
    for name, sub in actions.choices.items():
        sub.add_argument("--dir", type=Path, default=argparse.SUPPRESS, help=dir_help)


def handle(args: argparse.Namespace, *, transport: httpx.AsyncBaseTransport | None = None) -> int:
    root = _root(args)
    action = args.jobs_action
    try:
        if action == "scan":
            sites, _ = _sites(args)
            if args.json:
                print(json.dumps([s.to_json() for s in sites], indent=2))
            else:
                print(render_scan(sites))
            return EXIT_OK
        if action == "sync":
            sites, ref = _sites(args)
            registry, events = plan_sync(load_registry(root), sites, ref=ref)
            if not args.dry_run:
                save_registry(root, registry)
            if args.json:
                print(json.dumps([e.to_json() for e in events], indent=2))
            else:
                print(render_events(events, dry_run=args.dry_run))
            return EXIT_OK
        if action == "list":
            print(render_registry(load_registry(root)))
            return EXIT_OK
        if action == "log":
            entry = load_registry(root).get(args.job)
            if entry is None:
                print(f"throttle jobs log: no job {args.job!r}", file=sys.stderr)
                return EXIT_FAILED
            for item in entry.history:
                print(f"{item.get('ts')}  {item.get('event'):<10} {item.get('detail')}")
            return EXIT_OK
        if action == "eval":
            return _eval(args, root, transport)
        if action == "approve":
            canary = args.canary
            if canary is not None and not 0.0 < canary <= 1.0:
                raise ValueError("--canary must be in (0, 1]")
            entry, reason = approve_job(root, args.job, args.candidate, args.url, args.model,
                                        api_key_env=args.api_key_env, canary=canary)
            if entry is None:
                print(f"throttle jobs approve: refused. {reason}.", file=sys.stderr)
                return EXIT_FAILED
            print(f"Moved: {args.job} now runs on {args.candidate} ({args.model}). {reason}.")
            print("  Anything doubtful still goes to the API. Undo: throttle jobs undo " + args.job)
            return EXIT_OK
        if action in ("undo", "pin", "unpin"):
            status, detail = {
                "undo": (TESTING, "move undone by the team; on the API"),
                "pin": (PINNED, "pinned to the API by the team"),
                "unpin": (TESTING, "unpinned by the team; on the API until tests pass"),
            }[action]
            entry = set_status(root, args.job, status, detail)
            if entry is None:
                print(f"throttle jobs {action}: no job {args.job!r}", file=sys.stderr)
                return EXIT_FAILED
            print(f"{args.job}: {STATUS_WORDS[entry.status]}.")
            return EXIT_OK
        if action == "lessons":
            counts = lesson_counts(root)
            if not counts:
                print("No lessons yet. Fallbacks and judged losses of tracked jobs are saved here.")
            for job, counter in sorted(counts.items()):
                print(f"{job}: " + ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in sorted(counter.items())))
            return EXIT_OK
    except (ValueError, scan_mod.GitError, OSError) as exc:
        print(f"throttle jobs {action}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_USAGE


def _eval(args: argparse.Namespace, root: Path, transport: httpx.AsyncBaseTransport | None) -> int:
    from .shadow import api_key_from_env, chat_url

    config = None
    budget = None
    if not args.no_judge:
        if not args.judge_model:
            print("throttle jobs eval: pass --judge-model, or --no-judge for automatic checks only", file=sys.stderr)
            return EXIT_USAGE
        api_key = api_key_from_env(args.judge_key_env)
        url = args.judge_url or judge_mod.DEFAULT_URLS[args.judge_provider]
        if args.judge_provider == records.PROVIDER_OPENAI and args.judge_url:
            url = chat_url(args.judge_url)
        config = judge_mod.JudgeConfig(provider=args.judge_provider, model=args.judge_model, url=url, api_key=api_key,
                                       timeout=args.judge_timeout)
        budget = judge_mod.JudgeBudget(max_calls=max(0, args.max_judge_calls))
    sites, _ = _sites(args)
    print(f"Judging your model per job on captured calls that fit each job's current prompt "
          f"({len(sites)} jobs in the code{', judge asked twice per call' if config else ''})...", file=sys.stderr)
    doc = run_job_eval(root, args.candidate, sites, judge_config=config, budget=budget, margin=args.margin,
                       min_n=args.min_n, transport=transport, judge_concurrency=args.judge_concurrency,
                       max_per_job=args.max_per_job)
    if args.json:
        print(json.dumps(doc, indent=2, sort_keys=True))
        return EXIT_OK
    if not doc["jobs"]:
        print("No shadow answers matched a job's current prompt. Run `throttle shadow` on captured traffic first.")
    for job, result in doc["jobs"].items():
        judged = result.get("judged") or 0
        print(f"{job}: {result['verdict']} ({result['wins']} better, {result['ties'] + result.get('order_disagree', 0)} "
              f"same, {result['losses']} worse of {judged} judged)")
    print(f"Saved {doc['path']}")
    return EXIT_OK


def render_scan(sites: Sequence[scan_mod.CallSite]) -> str:
    if not sites:
        return "No OpenAI or Anthropic calls found."
    lines = [f"{len(sites)} AI call{'s' if len(sites) != 1 else ''} found:"]
    for site in sites:
        model = site.model or (f"set by {site.model_expr}" if site.model_expr else "model not set here")
        prompt = "prompt read" if site.prompt_fingerprint else "prompt built at run time"
        lines.append(f"  {site.job}  ({site.location}, {site.provider} {site.api}, {model}, {prompt})")
    return "\n".join(lines)


def render_events(events: Sequence[SyncEvent], *, dry_run: bool = False) -> str:
    if not events:
        return "Jobs already match the code. Nothing changed."
    head = "Would change:" if dry_run else "Changed:"
    return "\n".join([head] + [f"  {e.job}: {e.event}, {e.detail}" for e in events])


def render_registry(registry: Mapping[str, JobEntry]) -> str:
    if not registry:
        return "No jobs yet. Run: throttle jobs sync --repo ."
    lines = []
    for job, entry in sorted(registry.items()):
        where = STATUS_WORDS[entry.status]
        if entry.status == MOVED and entry.route:
            where += f": {entry.route.get('candidate_label')} ({entry.route.get('model')})"
        lines.append(f"{job}  {where}  [{entry.call_site.location}]")
        if entry.suspended:
            lines.append(f"    move suspended {entry.suspended.get('at')}: {entry.suspended.get('reason')}")
    return "\n".join(lines)
