"""Agent session tracking and profiling.

Records multi-turn agent sessions from proxy traffic to understand where
wall clock time is spent. Designed for always-on profiling of production
agent workloads (Claude Code, LangGraph, OpenHands, etc.).

Privacy: NEVER stores prompt/completion text, only hashes and token counts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

from fastapi import Request


@dataclass
class Turn:
    """A single turn within a session."""

    turn_id: int | None
    session_id: str
    turn_index: int

    # Timing
    arrival_timestamp: float
    ttft_ms: float | None
    total_latency_ms: float
    gap_since_previous_turn_seconds: float | None

    # Tokens
    prompt_tokens: int
    completion_tokens: int
    prefix_overlap_tokens: int | None
    prefix_overlap_percent: float | None

    # Response characteristics
    contains_tool_calls: bool
    finish_reason: str | None

    # Privacy: only hashes
    prompt_hash: str
    completion_hash: str | None

    # Extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionMetrics:
    """Computed metrics for a session."""

    session_id: str
    turn_count: int
    wall_clock_seconds: float
    generation_seconds: float
    generation_percent: float
    idle_seconds: float
    idle_percent: float
    redundant_prefill_tokens: int
    redundant_percent: float
    total_prompt_tokens: int
    total_completion_tokens: int
    avg_ttft_ms: float | None
    median_gap_seconds: float | None


class SessionTracker:
    """Tracks agent sessions from proxy requests.

    Groups requests into sessions using:
    1. Explicit X-Throttle-Session header
    2. Conversation continuation: the new messages array extends the previous
       messages array of a recent session from the same client
    3. Client + idle timeout (only when the request has no messages)

    Session IDs never contain the raw client IP; the client address is reduced
    to a short one-way hash.

    Writes turns to SQLite with batching; the per-request path is in-memory only
    (except one lookup when an explicit session ID is first seen, to resume
    turn numbering after a proxy restart).
    """

    SESSION_HEADER = "X-Throttle-Session"
    MAX_SESSION_ID_LENGTH = 128

    def __init__(
        self,
        db_path: Path | None = None,
        batch_size: int = 100,
        flush_interval_seconds: float = 5.0,
        idle_timeout_seconds: float = 600.0,  # 10 minutes
    ):
        self.db_path = Path(db_path) if db_path else Path.home() / ".throttle" / "sessions.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.idle_timeout_seconds = idle_timeout_seconds

        # Connection guarded by a (non-reentrant) thread lock. Never await
        # while holding it.
        self._conn: sqlite3.Connection | None = None
        self._lock = Lock()

        # Batching
        self._pending_turns: List[Turn] = []
        self._last_flush_time = time.time()

        # In-memory session state (bounded by idle-timeout expiry)
        # session_id -> wall-clock end time of the last turn
        self._active_sessions: Dict[str, float] = {}
        # session_id -> next turn index
        self._session_turn_counts: Dict[str, int] = {}
        # session_id -> messages of the last turn (memory only, never persisted)
        self._session_last_messages: Dict[str, List[Dict[str, Any]]] = {}
        # session_id -> hashed client key (for continuation matching)
        self._session_client: Dict[str, str] = {}

        self._initialize_db()

        # Background flush task
        self._flush_task: asyncio.Task | None = None
        self._shutdown = False

    def _initialize_db(self) -> None:
        """Initialize SQLite schema with WAL mode for concurrent access."""
        with self._lock:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row

            # Enable WAL mode for concurrent reads during writes
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")

            # Sessions table
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    first_seen_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    client_ip TEXT,
                    explicit_header INTEGER DEFAULT 0,
                    prefix_hash TEXT,
                    turn_count INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                )
            """)

            # Turns table
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,

                    arrival_timestamp REAL NOT NULL,
                    ttft_ms REAL,
                    total_latency_ms REAL NOT NULL,
                    gap_since_previous_turn_seconds REAL,

                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    prefix_overlap_tokens INTEGER,
                    prefix_overlap_percent REAL,

                    contains_tool_calls INTEGER DEFAULT 0,
                    finish_reason TEXT,

                    prompt_hash TEXT NOT NULL,
                    completion_hash TEXT,

                    metadata TEXT,
                    created_at REAL NOT NULL,

                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Indexes
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_turns_session
                ON turns(session_id, turn_index)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_turns_timestamp
                ON turns(arrival_timestamp)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_last_seen
                ON sessions(last_seen_at)
            """)

            self._conn.commit()

    def _compute_messages_hash(self, messages: List[Dict[str, Any]]) -> str:
        """Compute stable hash of messages array for session grouping."""
        # Serialize messages to stable JSON (sorted keys)
        serialized = json.dumps(messages, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _find_common_prefix(
        self, messages1: List[Dict[str, Any]], messages2: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find longest common prefix of two message arrays."""
        prefix = []
        for msg1, msg2 in zip(messages1, messages2):
            if msg1 == msg2:
                prefix.append(msg1)
            else:
                break
        return prefix

    @staticmethod
    def _client_key(request: Request) -> str:
        """One-way short hash of the client address (raw IP is never stored)."""
        host = request.client.host if getattr(request, "client", None) else "unknown"
        return hashlib.sha256(str(host).encode()).hexdigest()[:8]

    @staticmethod
    def _serialized_len(messages: Sequence[Dict[str, Any]]) -> int:
        return sum(
            len(json.dumps(m, sort_keys=True, separators=(",", ":"), default=str))
            for m in messages
        )

    def _compute_prefix_overlap(
        self,
        current_messages: List[Dict[str, Any]],
        previous_messages: List[Dict[str, Any]] | None,
        prompt_tokens: int,
    ) -> tuple[int, float]:
        """Estimate how much of this prompt was already sent in the previous turn.

        Returns (overlap_tokens, overlap_fraction). The fraction is the share of
        the serialized messages that is an exact message-level prefix of the
        previous turn's messages. overlap_tokens applies that fraction to the
        backend-reported prompt_tokens (0 when the backend reported no usage).
        This is an estimate: it does not tokenize and ignores the chat template.
        """
        if not previous_messages or not current_messages:
            return 0, 0.0

        prefix = self._find_common_prefix(previous_messages, current_messages)
        if not prefix:
            return 0, 0.0

        total_len = self._serialized_len(current_messages)
        if total_len <= 0:
            return 0, 0.0
        fraction = min(1.0, self._serialized_len(prefix) / total_len)
        overlap_tokens = int(round(max(0, prompt_tokens) * fraction))
        return overlap_tokens, fraction

    def _expire_idle_sessions(self, now: float) -> None:
        for session_id, last_seen in list(self._active_sessions.items()):
            if now - last_seen > self.idle_timeout_seconds:
                self._active_sessions.pop(session_id, None)
                self._session_turn_counts.pop(session_id, None)
                self._session_last_messages.pop(session_id, None)
                self._session_client.pop(session_id, None)

    def _assign_session_id(
        self,
        request: Request,
        request_body: Dict[str, Any],
        arrival_time: float,
    ) -> tuple[str, bool]:
        """Assign session ID using priority: header -> continuation -> client+timeout.

        Returns (session_id, is_explicit).
        """
        client_key = self._client_key(request)
        self._expire_idle_sessions(arrival_time)

        # Priority 1: Explicit header
        headers = getattr(request, "headers", None) or {}
        session_header = headers.get(self.SESSION_HEADER)
        if session_header:
            session_header = str(session_header).strip()[: self.MAX_SESSION_ID_LENGTH]
            if session_header:
                return session_header, True

        messages = request_body.get("messages") or []
        if messages:
            # Priority 2: continuation of a recent session from the same client.
            # Most recently active session wins.
            candidates = sorted(
                (
                    (last_seen, sid)
                    for sid, last_seen in self._active_sessions.items()
                    if self._session_client.get(sid) == client_key
                ),
                reverse=True,
            )
            for _, sid in candidates:
                prev = self._session_last_messages.get(sid)
                if not prev:
                    continue
                prefix = self._find_common_prefix(prev, messages)
                # Same conversation if the new request carries >=80% of the
                # previous turn's messages as its prefix.
                if len(prefix) >= max(1, len(prev) * 0.8):
                    return sid, False

            prefix_hash = self._compute_messages_hash(messages)[:8]
            return f"sess_{client_key}_{int(arrival_time)}_{prefix_hash}", False

        # Priority 3: client + idle-timeout fallback (no messages to compare)
        for sid in self._active_sessions:
            if self._session_client.get(sid) == client_key and sid.endswith("_ip"):
                return sid, False
        return f"sess_{client_key}_{int(arrival_time)}_ip", False

    def _load_persisted_session_state(self, session_id: str) -> tuple[int, float | None]:
        """For a session not in memory, return (next_turn_index, last_end_time) from DB."""
        try:
            with self._lock:
                if not self._conn:
                    return 0, None
                row = self._conn.execute(
                    """
                    SELECT turn_index, arrival_timestamp, total_latency_ms
                    FROM turns WHERE session_id = ?
                    ORDER BY turn_index DESC LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
        except sqlite3.Error:
            return 0, None
        if row is None:
            return 0, None
        last_end = row["arrival_timestamp"] + (row["total_latency_ms"] or 0.0) / 1000.0
        return int(row["turn_index"]) + 1, last_end

    async def record_turn(
        self,
        request: Request,
        request_body: Dict[str, Any],
        response: Dict[str, Any],
        latency_ms: float,
        ttft_ms: float | None = None,
        started_at: float | None = None,
    ) -> None:
        """Record one completed request/response turn.

        Call after the response is complete. ``started_at`` is the wall-clock
        (time.time()) moment the request arrived; if omitted it is inferred as
        now - latency. The gap for a turn is its start minus the previous
        turn's end, i.e. time the client spent outside the model.
        """
        now = time.time()
        latency_ms = max(0.0, float(latency_ms))
        arrival_time = started_at if started_at is not None else now - latency_ms / 1000.0
        end_time = arrival_time + latency_ms / 1000.0
        client_key = self._client_key(request)

        session_id, is_explicit = self._assign_session_id(request, request_body, arrival_time)

        # Turn index and previous end time
        if session_id in self._session_turn_counts:
            turn_index = self._session_turn_counts[session_id]
            prev_end = self._active_sessions.get(session_id)
        elif is_explicit:
            turn_index, prev_end = self._load_persisted_session_state(session_id)
        else:
            turn_index, prev_end = 0, None

        gap_seconds = None
        if turn_index > 0 and prev_end is not None:
            gap_seconds = max(0.0, arrival_time - prev_end)

        messages = request_body.get("messages") or []
        prompt_hash = self._compute_messages_hash(messages)

        usage = response.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)

        overlap_tokens, overlap_fraction = self._compute_prefix_overlap(
            messages,
            self._session_last_messages.get(session_id) if turn_index > 0 else None,
            prompt_tokens,
        )

        choices = response.get("choices") or []
        contains_tool_calls = False
        finish_reason = None
        completion_content: Any = None
        if choices:
            message = choices[0].get("message") or {}
            contains_tool_calls = bool(message.get("tool_calls"))
            finish_reason = choices[0].get("finish_reason")
            completion_content = message.get("content")
        if completion_content and not isinstance(completion_content, str):
            completion_content = json.dumps(completion_content, sort_keys=True, default=str)
        completion_hash = (
            hashlib.sha256(completion_content.encode()).hexdigest() if completion_content else None
        )

        # Update in-memory session state
        self._active_sessions[session_id] = end_time
        self._session_turn_counts[session_id] = turn_index + 1
        self._session_last_messages[session_id] = messages
        self._session_client[session_id] = client_key

        turn = Turn(
            turn_id=None,  # Assigned by database
            session_id=session_id,
            turn_index=turn_index,
            arrival_timestamp=arrival_time,
            ttft_ms=ttft_ms,
            total_latency_ms=latency_ms,
            gap_since_previous_turn_seconds=gap_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prefix_overlap_tokens=overlap_tokens,
            prefix_overlap_percent=overlap_fraction,
            contains_tool_calls=contains_tool_calls,
            finish_reason=finish_reason,
            prompt_hash=prompt_hash,
            completion_hash=completion_hash,
            metadata={
                "model": request_body.get("model", "unknown"),
                "explicit_session": is_explicit,
            },
        )

        with self._lock:
            self._pending_turns.append(turn)
            should_flush = len(self._pending_turns) >= self.batch_size

        # Flush outside the lock: _flush_batch takes the same non-reentrant lock.
        if should_flush:
            await self._flush_batch()

    async def _flush_batch(self) -> None:
        """Flush pending turns to the database (synchronous SQLite, batched)."""
        with self._lock:
            if not self._pending_turns or not self._conn:
                return
            turns_to_write = self._pending_turns[:]
            self._pending_turns.clear()
            self._last_flush_time = time.time()

            try:
                for turn in turns_to_write:
                    explicit = 1 if turn.metadata.get("explicit_session") else 0
                    self._conn.execute(
                        """
                        INSERT INTO sessions (
                            session_id, first_seen_at, last_seen_at, client_ip,
                            explicit_header, prefix_hash, turn_count, created_at
                        ) VALUES (?, ?, ?, NULL, ?, ?, 1, ?)
                        ON CONFLICT(session_id) DO UPDATE SET
                            first_seen_at = MIN(first_seen_at, excluded.first_seen_at),
                            last_seen_at = MAX(last_seen_at, excluded.last_seen_at),
                            turn_count = turn_count + 1
                        """,
                        (
                            turn.session_id,
                            turn.arrival_timestamp,
                            turn.arrival_timestamp,
                            explicit,
                            turn.prompt_hash[:8],
                            time.time(),
                        ),
                    )
                    self._conn.execute(
                        """
                        INSERT INTO turns (
                            session_id, turn_index, arrival_timestamp, ttft_ms,
                            total_latency_ms, gap_since_previous_turn_seconds,
                            prompt_tokens, completion_tokens, prefix_overlap_tokens,
                            prefix_overlap_percent, contains_tool_calls, finish_reason,
                            prompt_hash, completion_hash, metadata, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            turn.session_id,
                            turn.turn_index,
                            turn.arrival_timestamp,
                            turn.ttft_ms,
                            turn.total_latency_ms,
                            turn.gap_since_previous_turn_seconds,
                            turn.prompt_tokens,
                            turn.completion_tokens,
                            turn.prefix_overlap_tokens,
                            turn.prefix_overlap_percent,
                            1 if turn.contains_tool_calls else 0,
                            turn.finish_reason,
                            turn.prompt_hash,
                            turn.completion_hash,
                            json.dumps(turn.metadata),
                            time.time(),
                        ),
                    )
                self._conn.commit()
            except sqlite3.Error as e:
                # Never crash the proxy over profiling data.
                self._conn.rollback()
                print(f"Session tracking error: {e}", file=sys.stderr)

    async def start_background_flush(self) -> None:
        """Start background task to flush pending turns periodically."""
        async def flush_loop():
            while not self._shutdown:
                await asyncio.sleep(self.flush_interval_seconds)

                # Check if we should flush
                with self._lock:
                    elapsed = time.time() - self._last_flush_time
                    should_flush = (
                        len(self._pending_turns) > 0
                        and elapsed >= self.flush_interval_seconds
                    )

                if should_flush:
                    await self._flush_batch()

        self._flush_task = asyncio.create_task(flush_loop())

    async def shutdown(self) -> None:
        """Shutdown tracker and flush remaining turns."""
        self._shutdown = True

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush_batch()

        # Close connection
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    def close(self) -> None:
        """Close the database connection (synchronous; for read-only callers like the CLI)."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    def find_session_ids(self, prefix: str, limit: int = 10) -> List[str]:
        """Return session IDs that start with ``prefix`` (most recent first)."""
        with self._lock:
            if not self._conn:
                return []
            rows = self._conn.execute(
                "SELECT session_id FROM sessions "
                "WHERE substr(session_id, 1, length(?)) = ? "
                "ORDER BY last_seen_at DESC LIMIT ?",
                (prefix, prefix, limit),
            ).fetchall()
            return [r["session_id"] for r in rows]

    def get_recent_sessions(
        self, since_seconds: float = 86400, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent sessions (last 24h by default)."""
        cutoff = time.time() - since_seconds

        with self._lock:
            if not self._conn:
                return []

            cursor = self._conn.execute("""
                SELECT
                    session_id,
                    first_seen_at,
                    last_seen_at,
                    turn_count
                FROM sessions
                WHERE last_seen_at >= ?
                ORDER BY last_seen_at DESC
                LIMIT ?
            """, (cutoff, limit))

            sessions = []
            for row in cursor.fetchall():
                sessions.append(dict(row))

            return sessions

    def get_session_turns(self, session_id: str) -> List[Turn]:
        """Get all turns for a session."""
        with self._lock:
            if not self._conn:
                return []

            cursor = self._conn.execute("""
                SELECT *
                FROM turns
                WHERE session_id = ?
                ORDER BY turn_index ASC
            """, (session_id,))

            turns = []
            for row in cursor.fetchall():
                metadata_json = row["metadata"]
                metadata = json.loads(metadata_json) if metadata_json else {}

                turn = Turn(
                    turn_id=row["turn_id"],
                    session_id=row["session_id"],
                    turn_index=row["turn_index"],
                    arrival_timestamp=row["arrival_timestamp"],
                    ttft_ms=row["ttft_ms"],
                    total_latency_ms=row["total_latency_ms"],
                    gap_since_previous_turn_seconds=row["gap_since_previous_turn_seconds"],
                    prompt_tokens=row["prompt_tokens"],
                    completion_tokens=row["completion_tokens"],
                    prefix_overlap_tokens=row["prefix_overlap_tokens"],
                    prefix_overlap_percent=row["prefix_overlap_percent"],
                    contains_tool_calls=bool(row["contains_tool_calls"]),
                    finish_reason=row["finish_reason"],
                    prompt_hash=row["prompt_hash"],
                    completion_hash=row["completion_hash"],
                    metadata=metadata,
                )
                turns.append(turn)

            return turns

    def compute_session_metrics(self, session_id: str) -> SessionMetrics | None:
        """Compute aggregated metrics for a session."""
        turns = self.get_session_turns(session_id)

        if not turns:
            return None

        # Wall clock span
        first_arrival = min(t.arrival_timestamp for t in turns)
        last_end = max(t.arrival_timestamp + t.total_latency_ms / 1000.0 for t in turns)
        wall_clock_seconds = last_end - first_arrival

        # Generation time
        generation_seconds = sum(t.total_latency_ms for t in turns) / 1000.0
        generation_percent = (generation_seconds / wall_clock_seconds * 100) if wall_clock_seconds > 0 else 0.0

        # Idle time
        idle_seconds = sum(
            t.gap_since_previous_turn_seconds
            for t in turns
            if t.gap_since_previous_turn_seconds is not None
        )
        idle_percent = (idle_seconds / wall_clock_seconds * 100) if wall_clock_seconds > 0 else 0.0

        # Redundant prefill
        total_prompt_tokens = sum(t.prompt_tokens for t in turns)
        redundant_tokens = sum(
            t.prefix_overlap_tokens
            for t in turns
            if t.prefix_overlap_tokens is not None
        )
        redundant_percent = (redundant_tokens / total_prompt_tokens * 100) if total_prompt_tokens > 0 else 0.0

        # Averages
        ttft_values = [t.ttft_ms for t in turns if t.ttft_ms is not None]
        avg_ttft_ms = sum(ttft_values) / len(ttft_values) if ttft_values else None

        gap_values = [
            t.gap_since_previous_turn_seconds
            for t in turns
            if t.gap_since_previous_turn_seconds is not None
        ]
        median_gap_seconds = statistics.median(gap_values) if gap_values else None

        return SessionMetrics(
            session_id=session_id,
            turn_count=len(turns),
            wall_clock_seconds=wall_clock_seconds,
            generation_seconds=generation_seconds,
            generation_percent=generation_percent,
            idle_seconds=idle_seconds,
            idle_percent=idle_percent,
            redundant_prefill_tokens=redundant_tokens,
            redundant_percent=redundant_percent,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=sum(t.completion_tokens for t in turns),
            avg_ttft_ms=avg_ttft_ms,
            median_gap_seconds=median_gap_seconds,
        )
