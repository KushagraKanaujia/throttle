"""``throttle workloads``: what kinds of work the captured API traffic is, and
what each kind costs at list price.

Each captured exchange gets one workload type from a fixed, ordered rule
table (:data:`RULES`); the first rule that fires wins. Types are computed when
the capture is read. Capture files are append-only and are never rewritten,
so later stages call :func:`classify` themselves.

Exchanges captured without bodies (metadata-only redaction) are typed
``unknown``: their tokens and spend still count, under their own row.

Spend is measured usage tokens times the published list price (see
:mod:`throttle.own.pricing`). A request with no usage, an unknown price or an
error is excluded and counted, never estimated. The monthly figure is spend
over the capture window scaled to 30 days ("at captured rate") and is refused
when the window is shorter than a day.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import pricing as pricing_mod
from . import records
from .records import Exchange, Normalized

UNKNOWN = "unknown"
MIN_PROJECTION_DAYS = 1.0
SHORT_OUTPUT_WORDS = 8
SUMMARY_MIN_INPUT_WORDS = 400
SUMMARY_MIN_RATIO = 5.0
JSON_SCHEMA = "throttle.own.workloads/1"
EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

_WORD = re.compile(r"\S+")
_FENCE = re.compile(r"```")
_ASKS_JSON = re.compile(
    r"\b(respond|reply|answer|return|output|format|formatted|give|provide|produce|emit)\b[^.\n]{0,40}\bjson\b"
    r"|\b(as|in|valid|only|strict)\s+json\b|\bjson\s+(only|object|format|schema)\b",
    re.I,
)
_ASKS_FIELDS = re.compile(
    r"\bextract\b|\bfollowing fields\b|\b(fill in|populate)\b[^.\n]{0,30}\bfields?\b|^\s*fields?\s*:",
    re.I | re.M,
)
_ASKS_CLASSIFY = re.compile(r"\b(classify|classification|categori[sz]e|label (this|the|each)|which (category|label|class))\b", re.I)
_LISTS_LABELS = re.compile(
    r"\b(labels?|categories|category|classes|options|choices)\s*(are|is|:)"
    r"|\bone of\b|\b(one|single)\s+(word|label|category|letter)\b"
    r"|\b(yes|true)\s*(/|or)\s*(no|false)\b",
    re.I,
)
_ASKS_CODE = re.compile(
    r"\b(write|implement|fix|refactor|debug|generate|complete|port|convert)\b[^.\n]{0,40}"
    r"\b(code|function|script|class|method|program|query|regex|sql|unit tests?|snippet)\b",
    re.I,
)
_CODE_LINE = re.compile(
    r"^\s*(def |class |import |from \S+ import |function |const |let |var |public |private |#include|package |"
    r"SELECT |INSERT |UPDATE |CREATE TABLE|fn |func |return |if \(|for \()",
    re.M,
)
_ASKS_SUMMARY = re.compile(r"\b(summari[sz]e|summari[sz]ation|summary of|tl;?dr|condense|recap)\b", re.I)


def _words(text: str | None) -> int:
    return len(_WORD.findall(text or ""))


def _prompt(norm: Normalized) -> str:
    return f"{norm.system_text}\n{norm.last_user_text}"


def _forced_tool(exchange: Exchange) -> str | None:
    """Name of the one tool the request forces, if it forces one (OpenAI or
    Anthropic shape). Forcing a single tool is how structured output is often
    requested, so it is extraction, not an agent loop."""
    choice = (exchange.request or {}).get("tool_choice")
    if not isinstance(choice, dict):
        return None
    if choice.get("type") == "tool":
        return choice.get("name")
    if choice.get("type") == "function" and isinstance(choice.get("function"), dict):
        return choice["function"].get("name")
    return None


def _has_tool_history(norm: Normalized) -> bool:
    return any(m.role == "tool" or m.tool_calls for m in norm.messages)


def _rule_agent(exchange: Exchange, norm: Normalized) -> str | None:
    if _has_tool_history(norm):
        return "tool calls or tool results in the conversation"
    if norm.tools:
        if len(norm.tools) == 1 and _forced_tool(exchange):
            return None  # structured output through one forced tool: extraction
        return f"request offers {len(norm.tools)} tool(s)"
    if norm.output_tool_calls:
        return "response calls a tool"
    return None


def _rule_extraction(exchange: Exchange, norm: Normalized) -> str | None:
    if norm.response_format in ("json", "json_schema"):
        return f"response_format {norm.response_format}"
    request = exchange.request or {}
    output_config = request.get("output_config") or request.get("output_format")
    if isinstance(output_config, dict):
        fmt = output_config.get("format", output_config)
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            return "structured output (json_schema)"
    if norm.tools and _forced_tool(exchange):
        return "one forced tool used as an output schema"
    prompt = _prompt(norm)
    if _ASKS_JSON.search(prompt):
        return "prompt asks for JSON"
    if _ASKS_FIELDS.search(prompt):
        return "prompt asks to extract fields"
    return None


def _short_output(exchange: Exchange, norm: Normalized) -> bool:
    if norm.output_text is not None:
        return 0 < _words(norm.output_text) <= SHORT_OUTPUT_WORDS
    # No response body (e.g. a failed call): a tiny output cap says the same.
    return norm.max_tokens is not None and norm.max_tokens <= SHORT_OUTPUT_WORDS * 2


def _rule_classification(exchange: Exchange, norm: Normalized) -> str | None:
    if not _short_output(exchange, norm):
        return None
    prompt = _prompt(norm)
    if _ASKS_CLASSIFY.search(prompt):
        return "short answer to a classify/categorize/label prompt"
    if _LISTS_LABELS.search(prompt):
        return "short answer picked from listed labels or options"
    return None


def _rule_code(exchange: Exchange, norm: Normalized) -> str | None:
    output = norm.output_text or ""
    if _FENCE.search(output):
        return "code fence in the response"
    if len(_CODE_LINE.findall(output)) >= 2:
        return "response looks like code"
    if _FENCE.search(norm.last_user_text):
        return "code fence in the prompt"
    if _ASKS_CODE.search(_prompt(norm)):
        return "prompt asks for code"
    return None


def _rule_summarization(exchange: Exchange, norm: Normalized) -> str | None:
    if _ASKS_SUMMARY.search(_prompt(norm)):
        return "prompt asks to summarize"
    input_words = _words(norm.last_user_text)
    output_words = _words(norm.output_text) if norm.output_text is not None else None
    if output_words is None or input_words <= SUMMARY_MIN_INPUT_WORDS:
        return None
    if output_words == 0 or input_words / output_words > SUMMARY_MIN_RATIO:
        return f"long input ({input_words} words) with short output ({output_words} words)"
    return None


def _rule_chat(exchange: Exchange, norm: Normalized) -> str | None:
    return "no other rule matched"


# Checked top to bottom; the first rule that returns a reason wins.
RULES: tuple[tuple[str, Callable[[Exchange, Normalized], str | None]], ...] = (
    ("agent", _rule_agent),
    ("extraction", _rule_extraction),
    ("classification", _rule_classification),
    ("code", _rule_code),
    ("summarization", _rule_summarization),
    ("chat", _rule_chat),
)

def classify_reason(exchange: Exchange, norm: Normalized | None = None) -> tuple[str, str]:
    """``(workload_type, why)``. Metadata-only exchanges are ``unknown``."""
    if not exchange.has_bodies:
        return UNKNOWN, "captured without bodies (metadata only)"
    norm = norm if norm is not None else records.normalize(exchange)
    for name, rule in RULES:
        reason = rule(exchange, norm)
        if reason:
            return name, reason
    return "chat", "no other rule matched"


def classify(exchange: Exchange, norm: Normalized | None = None) -> str:
    return classify_reason(exchange, norm)[0]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


@dataclass
class Excluded:
    no_usage: int = 0
    unknown_price: int = 0
    errors: int = 0
    metadata_only: int = 0


@dataclass
class WorkloadSummary:
    workload_type: str
    requests: int = 0
    priced_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    spend_usd: float | None = None
    first_ts: str | None = None
    last_ts: str | None = None
    monthly_projection_usd: float | None = None
    projection_note: str = ""
    top_models: list[tuple[str, int]] = field(default_factory=list)
    excluded: Excluded = field(default_factory=Excluded)
    price_labels: list[str] = field(default_factory=list)
    clusters: list[dict[str, Any]] | None = None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = data.pop("workload_type")
        data["top_models"] = [{"model": m, "requests": n} for m, n in self.top_models]
        if self.clusters is None:
            data.pop("clusters")
        return data


@dataclass
class CaptureWindow:
    first_ts: str | None
    last_ts: str | None

    @property
    def days(self) -> float:
        if not self.first_ts or not self.last_ts:
            return 0.0
        delta = records.parse_iso(self.last_ts) - records.parse_iso(self.first_ts)
        return max(0.0, delta.total_seconds() / 86400)

    @property
    def projectable(self) -> bool:
        return self.days >= MIN_PROJECTION_DAYS


def capture_window(exchanges: Sequence[Exchange]) -> CaptureWindow:
    stamps = sorted(e.ts for e in exchanges)
    return CaptureWindow(stamps[0], stamps[-1]) if stamps else CaptureWindow(None, None)


def _window_text(window: CaptureWindow) -> str:
    hours = window.days * 24
    return f"{hours:.1f} hours" if hours < 48 else f"{window.days:.1f} days"


def summarize(
    exchanges: Iterable[Exchange],
    prices: pricing_mod.PriceTable,
    *,
    classifier: Callable[[Exchange], str] | None = None,
) -> list[WorkloadSummary]:
    """One row per workload type, sorted by spend (unpriced rows last).

    The monthly projection divides by the whole capture window, not the
    type's own first-to-last span: the capture ran the whole time, so a type
    seen only once in a week is a once-a-week type.
    """
    exchanges = list(exchanges)
    classifier = classifier or classify
    window = capture_window(exchanges)
    rows: dict[str, WorkloadSummary] = {}
    models: dict[str, Counter[str]] = {}
    labels: dict[str, set[str]] = {}
    for exchange in exchanges:
        kind = classifier(exchange)
        row = rows.setdefault(kind, WorkloadSummary(workload_type=kind))
        row.requests += 1
        models.setdefault(kind, Counter())[exchange.model or "?"] += 1
        if row.first_ts is None or exchange.ts < row.first_ts:
            row.first_ts = exchange.ts
        if row.last_ts is None or exchange.ts > row.last_ts:
            row.last_ts = exchange.ts
        if not exchange.has_bodies:
            row.excluded.metadata_only += 1
        if not exchange.ok:
            row.excluded.errors += 1
            continue
        usage = exchange.usage
        if not usage.complete:
            row.excluded.no_usage += 1
            continue
        row.input_tokens += usage.input_tokens or 0
        row.output_tokens += usage.output_tokens or 0
        row.cached_input_tokens += usage.cached_input_tokens or 0
        price = prices.lookup(exchange.provider, exchange.model)
        cost = pricing_mod.cost_usd(price, usage)
        if cost is None:
            row.excluded.unknown_price += 1
            continue
        row.priced_requests += 1
        row.spend_usd = (row.spend_usd or 0.0) + cost
        labels.setdefault(kind, set()).add(price.label)  # type: ignore[union-attr]
    for kind, row in rows.items():
        row.top_models = models[kind].most_common(3)
        row.price_labels = sorted(labels.get(kind, ()))
        if row.spend_usd is None:
            row.projection_note = "no priced requests"
        elif not window.projectable:
            row.projection_note = (
                f"capture window is {_window_text(window)}; need at least 1 day to project a month"
            )
        else:
            row.monthly_projection_usd = row.spend_usd / window.days * 30
            row.projection_note = f"at captured rate over {_window_text(window)}"
    return sorted(rows.values(), key=lambda r: (r.spend_usd is None, -(r.spend_usd or 0.0), r.workload_type))


# ---------------------------------------------------------------------------
# Optional: embedding sub-clusters
# ---------------------------------------------------------------------------


class EmbeddingsUnavailable(RuntimeError):
    pass


def _embed(texts: list[str]) -> list[list[float]]:
    from .. import embeddings

    if not embeddings.EMBEDDINGS_AVAILABLE:
        raise EmbeddingsUnavailable(embeddings.get_load_error() or "embedding dependencies not installed")
    if not embeddings.is_model_available():
        raise EmbeddingsUnavailable(embeddings.get_load_error() or "embedding model could not be loaded")
    return [list(map(float, row)) for row in embeddings.get_embeddings(texts)]


def _unit(vector: Sequence[float]) -> list[float] | None:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0 or not math.isfinite(norm):
        return None
    return [v / norm for v in vector]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def kmeans(vectors: list[list[float]], k: int, *, iterations: int = 25) -> list[int]:
    """Spherical k-means on unit vectors with farthest-point seeding, so the
    result is deterministic."""
    if not vectors:
        return []
    k = max(1, min(k, len(vectors)))
    centers = [vectors[0]]
    while len(centers) < k:
        far = min(range(len(vectors)), key=lambda i: max(_dot(vectors[i], c) for c in centers))
        centers.append(vectors[far])
    assignment = [0] * len(vectors)
    for step in range(iterations):
        new = [max(range(len(centers)), key=lambda c: _dot(v, centers[c])) for v in vectors]
        if new == assignment and step > 0:
            break
        assignment = new
        for c in range(len(centers)):
            members = [vectors[i] for i, a in enumerate(assignment) if a == c]
            if members:
                mean = [sum(col) / len(members) for col in zip(*members)]
                centers[c] = _unit(mean) or centers[c]
    return assignment


def _cluster_text(norm: Normalized) -> str:
    return f"{norm.system_text}\n{norm.last_user_text}".strip()[:2000]


def subclusters(
    exchanges: Sequence[Exchange],
    *,
    classifier: Callable[[Exchange], str] | None = None,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Split each workload type into prompt-similarity sub-clusters. Raises
    :class:`EmbeddingsUnavailable` rather than clustering zero vectors."""
    classifier = classifier or classify
    embed = embed or _embed
    by_type: dict[str, list[tuple[Exchange, str]]] = {}
    for exchange in exchanges:
        if not exchange.has_bodies:
            continue
        by_type.setdefault(classifier(exchange), []).append((exchange, _cluster_text(records.normalize(exchange))))
    result: dict[str, list[dict[str, Any]]] = {}
    for kind, items in by_type.items():
        vectors = embed([text for _, text in items])
        if len(vectors) != len(items):
            raise EmbeddingsUnavailable("embedder returned the wrong number of vectors")
        units = [_unit(v) for v in vectors]
        if any(u is None for u in units):
            raise EmbeddingsUnavailable("embedder returned zero vectors; refusing to cluster on them")
        k = max(1, min(6, round(math.sqrt(len(items) / 2))))
        assignment = kmeans(units, k)  # type: ignore[arg-type]
        groups: dict[int, list[int]] = {}
        for index, cluster in enumerate(assignment):
            groups.setdefault(cluster, []).append(index)
        clusters = []
        for members in sorted(groups.values(), key=len, reverse=True):
            example = " ".join(items[members[0]][1].split())[:60]
            clusters.append({"requests": len(members), "example": example})
        result[kind] = clusters
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_SINCE = re.compile(r"^(\d+)([mhdw])$")


def parse_since(value: str, *, now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    match = _SINCE.match(value.strip().lower())
    if match:
        amount, unit = int(match.group(1)), match.group(2)
        delta = {"m": timedelta(minutes=amount), "h": timedelta(hours=amount),
                 "d": timedelta(days=amount), "w": timedelta(weeks=amount)}[unit]
        return now - delta
    try:
        parsed = records.parse_iso(value.strip())
    except ValueError as exc:
        raise ValueError(f"--since takes 30m, 24h, 7d, 2w or an ISO date, not {value!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "workloads",
        help="group captured API traffic by kind of work and show what each kind costs at list price",
        description=(
            "Group captured API calls by kind of work (classification, extraction, summarization, "
            "agent, code, chat) and show requests, tokens and API spend per kind. Spend is measured "
            "tokens times the published list price; requests without usage or a known price are "
            "excluded and counted."
        ),
    )
    parser.add_argument("--dir", type=Path, default=None, help="data directory (default ~/.throttle/own or $THROTTLE_OWN_DIR)")
    parser.add_argument("--prices", type=Path, default=None, metavar="FILE",
                        help="JSON price file that replaces or adds entries; its prices are labelled ASSUMED")
    parser.add_argument("--since", default=None, help="only calls newer than this: 30m, 24h, 7d, 2w or an ISO date")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--embed", action="store_true",
                        help="also split each kind into sub-clusters by prompt similarity (needs throttle-pro[embeddings])")


def _fmt_tokens(value: int) -> str:
    for size, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if value >= size:
            return f"{value / size:.1f}{suffix}"
    return str(value)


def _fmt_usd(value: float | None) -> str:
    if value is None:
        return "-"
    return f"${value:,.2f}" if value >= 1 or value == 0 else f"${value:.4f}"


def build_report(
    exchanges: Sequence[Exchange],
    prices: pricing_mod.PriceTable,
    *,
    root: Path,
    override: Path | None = None,
    clusters: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    rows = summarize(exchanges, prices)
    if clusters is not None:
        for row in rows:
            row.clusters = clusters.get(row.workload_type, [])
    window = capture_window(exchanges)
    total_spend = sum(r.spend_usd for r in rows if r.spend_usd is not None)
    priced = any(r.spend_usd is not None for r in rows)
    excluded = Excluded()
    for row in rows:
        for key in asdict(excluded):
            setattr(excluded, key, getattr(excluded, key) + getattr(row.excluded, key))
    labels = sorted({label for r in rows for label in r.price_labels})
    sources = sorted({p.source for p in prices if p.origin == pricing_mod.ORIGIN_BUNDLED})
    return {
        "schema": JSON_SCHEMA,
        "dir": str(root),
        "window": {"first_ts": window.first_ts, "last_ts": window.last_ts, "days": round(window.days, 4),
                   "projectable": window.projectable},
        "spend_basis": "MEASURED tokens x list price; batch and priority tiers not covered",
        "projection_basis": "at captured rate (spend / window days x 30)",
        "prices": {"labels": labels, "sources": sources, "override": str(override) if override else None},
        "totals": {
            "requests": sum(r.requests for r in rows),
            "priced_requests": sum(r.priced_requests for r in rows),
            "spend_usd": total_spend if priced else None,
            "monthly_projection_usd": (total_spend / window.days * 30) if priced and window.projectable else None,
            "excluded": asdict(excluded),
        },
        "workloads": [r.to_json() for r in rows],
    }


def render(report: dict[str, Any]) -> str:
    from .. import style as style_mod

    st = style_mod.get()
    window = report["window"]
    totals = report["totals"]
    lines: list[str] = []
    head = f"{'TYPE':<15}{'REQS':>7}{'IN TOK':>9}{'OUT TOK':>9}{'API SPEND':>12}{'PER MONTH':>13}"
    lines.append(st.bold(head))
    for row in report["workloads"]:
        lines.append(
            f"{row['type']:<15}{row['requests']:>7}{_fmt_tokens(row['input_tokens']):>9}"
            f"{_fmt_tokens(row['output_tokens']):>9}{_fmt_usd(row['spend_usd']):>12}"
            f"{_fmt_usd(row['monthly_projection_usd']):>13}"
        )
        for cluster in row.get("clusters") or []:
            lines.append(st.dim(f"  {cluster['requests']:>4} x {cluster['example']}"))
    lines.append("")
    lines.append(
        f"{'total':<15}{totals['requests']:>7}{'':>18}{_fmt_usd(totals['spend_usd']):>12}"
        f"{_fmt_usd(totals['monthly_projection_usd']):>13}"
    )
    lines.append("")
    if window["first_ts"]:
        lines.append(f"Captured {window['first_ts']} to {window['last_ts']}.")
    if not window["projectable"] and window["first_ts"]:
        hours = window["days"] * 24
        lines.append(st.warn(f"No monthly projection: the capture covers {hours:.1f} hours, under 1 day."))
    else:
        lines.append("Per month is at the captured rate (spend / window days x 30).")
    labels = report["prices"]["labels"]
    lines.append("API spend = measured tokens x " + (", ".join(labels) if labels else "list price") + ".")
    if report["prices"]["override"]:
        lines.append(f"Operator prices from {report['prices']['override']} are ASSUMED.")
    ex = totals["excluded"]
    lines.append(
        f"Excluded from spend: {ex['no_usage']} without usage, {ex['unknown_price']} with no known price, "
        f"{ex['errors']} errors."
    )
    if ex["metadata_only"]:
        lines.append(f"{ex['metadata_only']} captured as metadata only: typed 'unknown', still priced.")
    for source in report["prices"]["sources"]:
        lines.append(st.dim(f"Prices: {source}"))
    return st.panel(lines, title="Workloads")


def handle(args: argparse.Namespace) -> int:
    root = args.dir.expanduser() if args.dir else records.default_root()
    try:
        prices = pricing_mod.load_prices(args.prices)
    except ValueError as exc:
        print(f"throttle workloads: {exc}", file=sys.stderr)
        return EXIT_USAGE
    since = None
    if args.since:
        try:
            since = parse_since(args.since)
        except ValueError as exc:
            print(f"throttle workloads: {exc}", file=sys.stderr)
            return EXIT_USAGE
    exchanges = list(records.iter_exchanges(root, since=since))
    if not exchanges:
        print(
            f"throttle workloads: no captured calls in {records.capture_dir(root)}"
            + (f" since {args.since}" if args.since else "")
            + ". Run 'throttle capture' in front of your API traffic first.",
            file=sys.stderr,
        )
        return EXIT_FAILED
    clusters = None
    if args.embed:
        try:
            clusters = subclusters(exchanges)
        except EmbeddingsUnavailable as exc:
            print(
                f"throttle workloads --embed: cannot cluster by prompt similarity: {exc}. "
                "Install throttle-pro[embeddings] or run without --embed.",
                file=sys.stderr,
            )
            return EXIT_FAILED
    report = build_report(exchanges, prices, root=root, override=args.prices, clusters=clusters)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report))
    return EXIT_OK
