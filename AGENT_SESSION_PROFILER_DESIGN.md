# Agent Session Profiler - Design Document

## Motivation

Modern LLM workloads are multi-turn, tool-interrupted agent sessions (Claude Code, LangGraph, OpenHands), not single "prompt → response" requests. Published data shows GPUs running at 30-40% utilization on agent traffic. **Nobody measures why.**

Current Throttle benchmarks single-shot traffic. This is no longer representative of production usage. We need continuous profiling of real agent sessions to understand where wall clock time actually goes.

## Goal

Transform Throttle from a one-shot benchmark tool into an **always-running agent session profiler** that:
- Groups requests into sessions
- Captures complete session traces
- Reports where wall clock time is spent
- Provides actionable findings tied to specific config changes

Users leave it running for days/weeks and get more value each week as historical data accumulates.

## Design Overview

### 1. Session Grouping Heuristic

Requests arriving at `/v1/chat/completions` are grouped into sessions using a three-tier priority system:

**Priority 1: Explicit Header**
- Check for `X-Throttle-Session: <session_id>` header
- If present, use that session_id directly
- This allows clients to explicitly control session boundaries

**Priority 2: Message Prefix Hash**
- Compute stable hash of longest common message prefix across recent requests from same client
- Agent frameworks typically build conversation history incrementally
- Requests with `messages: [A, B, C]` and `messages: [A, B, C, D]` share prefix `[A, B, C]`
- Hash the prefix to create session identifier
- Window: last 10 minutes of requests per client IP

**Priority 3: IP + Idle Timeout**
- Fall back to client IP address
- Session ends after 10 minutes of inactivity
- Start new session if gap > 10 minutes

**Session ID Format**: `sess_<timestamp>_<hash[:8]>` for traceability

### 2. Per-Turn Trace Schema

SQLite database at `~/.throttle/sessions.db` with schema designed for forward compatibility:

```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    first_seen_at REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    client_ip TEXT,
    explicit_header INTEGER DEFAULT 0,  -- 1 if used X-Throttle-Session
    prefix_hash TEXT,
    turn_count INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE turns (
    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,  -- 0-indexed turn within session

    -- Timing
    arrival_timestamp REAL NOT NULL,
    ttft_ms REAL,
    total_latency_ms REAL NOT NULL,
    gap_since_previous_turn_seconds REAL,  -- NULL for turn 0

    -- Token metrics
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    prefix_overlap_tokens INTEGER,  -- tokens shared with previous turn
    prefix_overlap_percent REAL,    -- 0.0-1.0

    -- Response characteristics
    contains_tool_calls INTEGER DEFAULT 0,  -- 1 if tool_calls present
    finish_reason TEXT,

    -- Privacy: NEVER store prompt/completion text
    -- Only hashes for deduplication
    prompt_hash TEXT NOT NULL,
    completion_hash TEXT,

    -- Metadata (JSON blob for future extensibility)
    metadata TEXT,

    created_at REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX idx_turns_session ON turns(session_id, turn_index);
CREATE INDEX idx_turns_timestamp ON turns(arrival_timestamp);
CREATE INDEX idx_sessions_last_seen ON sessions(last_seen_at);
```

**Key Design Decisions:**
- `metadata TEXT` (JSON blob) allows adding fields without schema migration
- All timestamps are Unix epoch floats for consistency
- Privacy-first: NO prompt/completion text, only hashes
- Indexed for fast session lookups and time-range queries

### 3. Wall Clock Accounting

For each session, compute:

**Total Wall Clock Span**
```
wall_clock_seconds = last_turn_end - first_turn_start
```

**Time Spent Generating** (GPU busy)
```
generation_seconds = sum(turn.total_latency_ms) / 1000
generation_percent = (generation_seconds / wall_clock_seconds) * 100
```

**Time Waiting on Client** (idle between turns)
```
idle_seconds = sum(turn.gap_since_previous_turn_seconds for all turns > 0)
idle_percent = (idle_seconds / wall_clock_seconds) * 100
```

**Redundant Prefill Tokens** (reprocessing without prefix cache)
```
redundant_tokens = sum(turn.prefix_overlap_tokens where no cache hit)
redundant_percent = (redundant_tokens / total_prompt_tokens) * 100
```

**Metrics Output Format:**
```json
{
  "session_id": "sess_1694...",
  "wall_clock_seconds": 425.3,
  "generation_seconds": 87.2,
  "generation_percent": 20.5,
  "idle_seconds": 312.1,
  "idle_percent": 73.4,
  "redundant_prefill_tokens": 8432,
  "redundant_percent": 34.2
}
```

### 4. CLI Interface

**Default View: Recent Sessions Table**
```bash
$ throttle sessions

Recent Sessions (last 24h):
┌──────────────────────┬───────┬──────────┬────────────┬────────────────┐
│ Session ID           │ Turns │ Duration │ Idle %     │ Redundant Pfx  │
├──────────────────────┼───────┼──────────┼────────────┼────────────────┤
│ sess_169...3a2       │ 47    │ 12m 34s  │ 68.2%      │ 2,341 (23.1%)  │
│ sess_169...7f1       │ 23    │ 6m 12s   │ 71.5%      │ 892 (12.3%)    │
│ sess_169...9bc       │ 156   │ 42m 18s  │ 82.1%      │ 12,432 (42.8%) │
└──────────────────────┴───────┴──────────┴────────────┴────────────────┘
```

**Session Detail View**
```bash
$ throttle sessions sess_169...3a2

Session: sess_169...3a2
Duration: 12m 34s (754 seconds)
Turns: 47
Client: 192.168.1.100

Wall Clock Breakdown:
  Generation:  163.2s (21.7%)  ████████
  Client Wait: 590.8s (78.3%)  ███████████████████████████████

Turn-by-Turn Timeline:
┌───────┬────────────┬──────────┬────────────┬──────────┬───────────┐
│ Turn  │ Gap (s)    │ TTFT (ms)│ Total (ms) │ Tokens   │ Tool Call │
├───────┼────────────┼──────────┼────────────┼──────────┼───────────┤
│ 0     │ -          │ 234      │ 1,823      │ 45 → 128 │           │
│ 1     │ 8.3        │ 189      │ 2,134      │ 173 → 89 │ ✓         │
│ 2     │ 15.7       │ 201      │ 1,956      │ 262 → 76 │           │
│ ...   │            │          │            │          │           │
└───────┴────────────┴──────────┴────────────┴──────────┴───────────┘

Prefix Overlap:
  Average: 78.3% (181 tokens/turn)
  Total redundant: 2,341 tokens (23.1% of all prompt tokens)
```

**Time Window Aggregates with Trend Detection**
```bash
$ throttle sessions --since 7d

Aggregate Metrics (last 7 days):
  Sessions: 234
  Total Turns: 8,923
  Median Turn Count: 38

Wall Clock Distribution:
  Median Generation %: 22.3%
  Median Idle %: 73.2%

Performance Trends:
  TTFT (median): 203ms → 287ms  📈 +41% (p < 0.05)  ⚠️  REGRESSION
  Idle % (median): 71.2% → 73.8%  (not significant)

Findings (see below for details):
  [HIGH] TTFT increasing with turn_index → KV cache pressure
  [HIGH] 73% idle time at low concurrency → GPU underutilized
  [MED]  34% redundant prefill → enable prefix caching
```

### 5. Findings System

Each session and aggregate view includes **concrete findings** tied to actionable fixes.

**Findings Module** (`src/throttle/session_findings.py`):

```python
@dataclass
class Finding:
    severity: Literal["high", "medium", "low"]
    category: str
    observation: str
    recommendation: str
    config_change: str | None  # Exact config to change

def analyze_session(session_data: SessionMetrics) -> list[Finding]:
    """Generate findings for a single session."""
    findings = []

    # Rule 1: High redundant prefill
    if session_data.redundant_percent > 30:
        findings.append(Finding(
            severity="high",
            category="prefix_caching",
            observation=f"{session_data.redundant_percent:.1f}% of prompt tokens are redundant prefill",
            recommendation="Enable prefix caching on backend to reuse KV cache across turns",
            config_change="--enable-prefix-caching (vLLM) or prefix_cache_ttl=300 (SGLang)"
        ))

    # Rule 2: High idle time at low concurrency
    if session_data.idle_percent > 70 and session_data.concurrent_sessions < 3:
        findings.append(Finding(
            severity="high",
            category="utilization",
            observation=f"{session_data.idle_percent:.1f}% idle time with <3 concurrent sessions",
            recommendation="GPU is underutilized. Scale down to save costs or serve more traffic",
            config_change="Reduce instance size or enable request batching"
        ))

    # Rule 3: TTFT grows with turn index
    if has_ttft_growth_trend(session_data.turns):
        findings.append(Finding(
            severity="high",
            category="kv_cache_pressure",
            observation="TTFT increases significantly after turn ~15",
            recommendation="KV cache is filling up, causing evictions and recomputation",
            config_change="Increase --kv-cache-free-gpu-mem-fraction or reduce --max-model-len"
        ))

    # Rule 4: Frequent tool calls with long gaps
    tool_call_turns = [t for t in session_data.turns if t.contains_tool_calls]
    if len(tool_call_turns) > 5 and avg_gap_after_tool > 10:
        findings.append(Finding(
            severity="medium",
            category="agent_orchestration",
            observation=f"Average {avg_gap_after_tool:.1f}s gap after tool calls",
            recommendation="Tool execution is slow. Profile agent framework or parallelize tools",
            config_change=None  # Client-side issue
        ))

    return findings
```

**Output Format:**
```
FINDINGS:

[HIGH] Prefix Caching
  Observation: 42.8% of prompt tokens are redundant prefill
  Recommendation: Enable prefix caching on backend to reuse KV cache across turns
  Config: --enable-prefix-caching (vLLM) or prefix_cache_ttl=300 (SGLang)

[HIGH] GPU Utilization
  Observation: 78.3% idle time with <3 concurrent sessions
  Recommendation: GPU is underutilized. Scale down to save costs or serve more traffic
  Config: Reduce instance size or enable request batching

[MED] Agent Orchestration
  Observation: Average 12.3s gap after tool calls
  Recommendation: Tool execution is slow. Profile agent framework or parallelize tools
  Config: (client-side optimization)
```

### 6. Proxy Integration (Minimal Overhead)

**Key Constraint: <5ms overhead per request**

**Session Tracking in proxy.py:**

```python
# Add to ProxyServer.__init__
self._session_tracker: Optional[SessionTracker] = None
if enable_session_tracking:
    from .sessions import SessionTracker
    self._session_tracker = SessionTracker()

# In chat_completions() method, AFTER response received:
async def chat_completions(self, request: Request):
    start_time = time.perf_counter()
    request_body = await request.json()

    # ... existing cache/dedup logic ...

    # Get response (cached or from backend)
    response_data = await ...

    total_latency_ms = (time.perf_counter() - start_time) * 1000

    # Track session asynchronously (non-blocking)
    if self._session_tracker:
        asyncio.create_task(
            self._session_tracker.record_turn(
                request=request,
                request_body=request_body,
                response=response_data,
                latency_ms=total_latency_ms,
                ttft_ms=ttft_ms  # if available
            )
        )

    return response
```

**Design Notes:**
- Session tracking is **async/non-blocking** via `create_task()`
- Database writes happen in background
- Use batch inserts (buffer 100 turns, flush every 5 seconds)
- SQLite WAL mode for concurrent reads during writes
- Overhead target: <1ms for session ID assignment, rest is async

### 7. Statistical Rigor (Existing Throttle Standards)

Reuse existing `statistics.py` helpers:

**Confidence Intervals:**
- Week-over-week comparisons use `t_interval_95()` for medians
- Require ≥20 sessions per window for statistical claims
- Mark insufficient data as "exploratory" (existing pattern)

**Decision-Grade vs Exploratory:**
- Single session view: always exploratory (one sample)
- Aggregate view with ≥20 sessions: decision-grade if CI excludes zero
- Explicit `decision_eligible: false` in JSON output (existing field)

**Example Output:**
```json
{
  "time_window": "7d",
  "sample_size": 234,
  "decision_eligible": true,
  "ttft_median_week1_ms": 203,
  "ttft_median_week2_ms": 287,
  "ttft_delta_percent": 41.4,
  "confidence_interval_95": [28.3, 54.5],
  "p_value": 0.0023,
  "verdict": "statistically_significant_regression"
}
```

## Implementation Plan

### Phase 1: Core Session Tracking (Days 1-2)
1. Create `src/throttle/sessions.py` with:
   - `SessionTracker` class
   - SQLite schema initialization
   - Session grouping logic (header → prefix → IP+timeout)
   - Turn recording with async batch writes

2. Integrate into `proxy.py`:
   - Add `--enable-session-tracking` flag
   - Hook into request/response flow
   - Async task for recording

3. Tests:
   - Session grouping heuristics
   - Database writes
   - Overhead benchmark (<5ms)

### Phase 2: Accounting & Findings (Days 3-4)
1. Add wall clock accounting functions
2. Create `src/throttle/session_findings.py` with rule engine
3. Tests for each finding rule

### Phase 3: CLI Commands (Day 5)
1. Add `sessions` subcommand to `cli.py`
2. Implement table rendering
3. Add `--since` time window support
4. Tests for CLI output

### Phase 4: Polish & Documentation (Day 6)
1. Performance profiling
2. Integration tests
3. Update README with agent profiling section
4. Example workflows

## Testing Strategy

**Unit Tests:**
- `test_sessions.py`: Session grouping, database ops
- `test_session_findings.py`: Each finding rule
- `test_session_cli.py`: CLI output formatting

**Integration Tests:**
- `test_proxy_session_tracking.py`: End-to-end request → database
- `test_session_overhead.py`: Assert <5ms overhead

**Performance Test:**
```python
def test_proxy_overhead_with_session_tracking():
    """Verify session tracking adds <5ms overhead."""
    proxy_without_tracking = ProxyServer(..., enable_session_tracking=False)
    proxy_with_tracking = ProxyServer(..., enable_session_tracking=True)

    for _ in range(100):
        t0 = time.perf_counter()
        await proxy_without_tracking.chat_completions(request)
        baseline_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        await proxy_with_tracking.chat_completions(request)
        tracked_ms = (time.perf_counter() - t0) * 1000

        overhead_ms = tracked_ms - baseline_ms
        assert overhead_ms < 5.0, f"Overhead {overhead_ms:.2f}ms exceeds 5ms limit"
```

## Migration & Compatibility

**Backward Compatibility:**
- Session tracking is opt-in via `--enable-session-tracking`
- Proxy continues to work exactly as before when disabled
- No changes to existing benchmarking commands

**Schema Evolution:**
- `metadata TEXT` (JSON) column allows adding fields without migration
- Future: add `ALTER TABLE` migration system if needed

## Open Questions & Future Work

**Q: How to handle distributed proxies?**
A: Phase 1 is single-instance only. Future: session_id in Redis, turn data in PostgreSQL.

**Q: Should we expose session metrics via `/health` endpoint?**
A: Yes, add `"active_sessions": N, "total_turns_today": M` to health response.

**Q: Integration with existing tracing.py?**
A: Keep separate for now. `tracing.py` is for client-side agent instrumentation, `sessions.py` is server-side proxy observation. Could merge later.

**Future Enhancements:**
- Export to OpenTelemetry format
- Grafana dashboard integration
- Real-time session alerting (e.g., Slack when TTFT > threshold)
- Session replay (privacy-preserving, hash-based)

## Success Criteria

1. ✅ Proxy overhead < 5ms with session tracking enabled
2. ✅ All tests pass (unit + integration)
3. ✅ Can run `throttle sessions` and see meaningful output after 1 day of traffic
4. ✅ Findings correctly identify common issues (prefix caching, idle time, KV pressure)
5. ✅ No prompt/completion text stored (privacy audit pass)
6. ✅ Schema supports adding fields without breaking changes

## Files to Create/Modify

**New Files:**
- `src/throttle/sessions.py` (core tracker)
- `src/throttle/session_findings.py` (findings engine)
- `tests/test_sessions.py`
- `tests/test_session_findings.py`
- `tests/test_session_overhead.py`

**Modified Files:**
- `src/throttle/proxy.py` (add session tracking hook)
- `src/throttle/cli.py` (add `sessions` subcommand)
- `README.md` (document agent profiling)
- `pyproject.toml` (no new dependencies needed!)

---

**This design provides a clear path to transform Throttle into a YC-fundable always-running agent profiler while maintaining the rigorous measurement standards that make Throttle unique.**
