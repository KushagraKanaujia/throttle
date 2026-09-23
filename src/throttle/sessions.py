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
import time
from collections import defaultdict, deque
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
    2. Stable hash of longest common message prefix
    3. Client IP + 10min idle timeout

    Writes turns to SQLite asynchronously with batching for performance.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        batch_size: int = 100,
        flush_interval_seconds: float = 5.0,
        idle_timeout_seconds: float = 600.0,  # 10 minutes
    ):
        self.db_path = db_path or Path.home() / ".throttle" / "sessions.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.idle_timeout_seconds = idle_timeout_seconds

        # Thread-safe connection management
        self._conn: sqlite3.Connection | None = None
        self._lock = Lock()

        # Batching
        self._pending_turns: List[Turn] = []
        self._last_flush_time = time.time()

        # Session state tracking
        # Maps client_ip -> deque of (timestamp, messages_hash)
        self._recent_requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))

        # Maps session_id -> last_seen_timestamp
        self._active_sessions: Dict[str, float] = {}

        # Maps session_id -> turn_index
        self._session_turn_counts: Dict[str, int] = {}

        # Maps session_id -> previous_messages_hash (for prefix overlap)
        self._session_last_messages: Dict[str, str] = {}

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

    def _compute_prefix_overlap(
        self,
        current_messages: List[Dict[str, Any]],
        previous_messages_hash: str | None,
        client_ip: str,
    ) -> tuple[int, float]:
        """Compute prefix overlap tokens and percentage.

        Returns (overlap_tokens, overlap_percent).
        For simplicity, we estimate 1 message ≈ 50 tokens (conservative).
        Actual token counts would require tiktoken, but we avoid new dependencies.
        """
        if not previous_messages_hash:
            return 0, 0.0

        # Try to find previous messages in recent requests
        recent = self._recent_requests.get(client_ip, deque())
        previous_messages = None

        for timestamp, msg_hash, messages in recent:
            if msg_hash == previous_messages_hash:
                previous_messages = messages
                break

        if not previous_messages:
            return 0, 0.0

        # Find common prefix
        prefix = self._find_common_prefix(previous_messages, current_messages)

        # Estimate tokens: ~50 per message (conservative)
        TOKENS_PER_MESSAGE = 50
        overlap_tokens = len(prefix) * TOKENS_PER_MESSAGE
        total_tokens = len(current_messages) * TOKENS_PER_MESSAGE
        overlap_percent = overlap_tokens / total_tokens if total_tokens > 0 else 0.0

        return overlap_tokens, overlap_percent

    def _assign_session_id(
        self,
        request: Request,
        request_body: Dict[str, Any],
        arrival_time: float,
    ) -> tuple[str, bool, str | None]:
        """Assign session ID using priority: header → prefix → IP+timeout.

        Returns (session_id, is_explicit, prefix_hash).
        """
        client_ip = request.client.host if request.client else "unknown"

        # Priority 1: Explicit header
        session_header = request.headers.get("X-Throttle-Session")
        if session_header:
            return session_header, True, None

        # Priority 2: Message prefix hash
        messages = request_body.get("messages", [])
        if messages:
            messages_hash = self._compute_messages_hash(messages)

            # Store recent request
            self._recent_requests[client_ip].append((arrival_time, messages_hash, messages))

            # Check if this hash matches any recent session
            for session_id, last_seen in list(self._active_sessions.items()):
                # Timeout check
                if arrival_time - last_seen > self.idle_timeout_seconds:
                    # Session expired
                    del self._active_sessions[session_id]
                    if session_id in self._session_turn_counts:
                        del self._session_turn_counts[session_id]
                    if session_id in self._session_last_messages:
                        del self._session_last_messages[session_id]
                    continue

                # Check if messages share prefix with this session
                if session_id.startswith(f"sess_{client_ip}_"):
                    prev_hash = self._session_last_messages.get(session_id)
                    if prev_hash:
                        # Check if current messages extend previous messages
                        for _, prev_msg_hash, prev_messages in self._recent_requests[client_ip]:
                            if prev_msg_hash == prev_hash:
                                prefix = self._find_common_prefix(prev_messages, messages)
                                # If >80% overlap, consider it same session
                                if len(prefix) >= len(prev_messages) * 0.8:
                                    self._session_last_messages[session_id] = messages_hash
                                    return session_id, False, messages_hash[:8]

            # New session from prefix
            prefix_hash = messages_hash[:8]
            session_id = f"sess_{client_ip}_{int(arrival_time)}_{prefix_hash}"
            self._session_last_messages[session_id] = messages_hash
            return session_id, False, prefix_hash

        # Priority 3: IP + timeout fallback
        # Check for existing IP-based session
        for session_id, last_seen in list(self._active_sessions.items()):
            if (
                session_id.startswith(f"sess_{client_ip}_")
                and arrival_time - last_seen <= self.idle_timeout_seconds
            ):
                return session_id, False, None

        # New IP-based session
        session_id = f"sess_{client_ip}_{int(arrival_time)}_ip"
        return session_id, False, None

    async def record_turn(
        self,
        request: Request,
        request_body: Dict[str, Any],
        response: Dict[str, Any],
        latency_ms: float,
        ttft_ms: float | None = None,
    ) -> None:
        """Record a turn asynchronously (non-blocking for proxy).

        This is called from proxy.chat_completions() via asyncio.create_task().
        """
        arrival_time = time.time()
        client_ip = request.client.host if request.client else "unknown"

        # Assign session
        session_id, is_explicit, prefix_hash = self._assign_session_id(
            request, request_body, arrival_time
        )

        # Update session state
        self._active_sessions[session_id] = arrival_time
        turn_index = self._session_turn_counts.get(session_id, 0)
        self._session_turn_counts[session_id] = turn_index + 1

        # Extract metrics
        messages = request_body.get("messages", [])
        prompt_hash = self._compute_messages_hash(messages)

        # Get usage from response
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Compute prefix overlap
        previous_hash = self._session_last_messages.get(session_id) if turn_index > 0 else None
        overlap_tokens, overlap_percent = self._compute_prefix_overlap(
            messages, previous_hash, client_ip
        )

        # Check for tool calls
        choices = response.get("choices", [])
        contains_tool_calls = False
        finish_reason = None
        if choices:
            message = choices[0].get("message", {})
            contains_tool_calls = "tool_calls" in message and bool(message["tool_calls"])
            finish_reason = choices[0].get("finish_reason")

        # Compute completion hash (hash the content, not the content itself)
        completion_content = ""
        if choices:
            completion_content = choices[0].get("message", {}).get("content", "")
        completion_hash = hashlib.sha256(completion_content.encode()).hexdigest() if completion_content else None

        # Compute gap from previous turn
        gap_seconds = None
        if turn_index > 0:
            # Query previous turn's timestamp
            gap_seconds = await self._get_gap_from_previous_turn(session_id, turn_index)

        # Create turn
        turn = Turn(
            turn_id=None,  # Will be assigned by database
            session_id=session_id,
            turn_index=turn_index,
            arrival_timestamp=arrival_time,
            ttft_ms=ttft_ms,
            total_latency_ms=latency_ms,
            gap_since_previous_turn_seconds=gap_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prefix_overlap_tokens=overlap_tokens,
            prefix_overlap_percent=overlap_percent,
            contains_tool_calls=contains_tool_calls,
            finish_reason=finish_reason,
            prompt_hash=prompt_hash,
            completion_hash=completion_hash,
            metadata={"model": request_body.get("model", "unknown")},
        )

        # Add to batch
        with self._lock:
            self._pending_turns.append(turn)

            # Flush if batch is full
            if len(self._pending_turns) >= self.batch_size:
                await self._flush_batch()

    async def _get_gap_from_previous_turn(self, session_id: str, current_turn_index: int) -> float | None:
        """Get gap from previous turn by querying database."""
        try:
            with self._lock:
                if not self._conn:
                    return None

                cursor = self._conn.execute("""
                    SELECT arrival_timestamp, total_latency_ms
                    FROM turns
                    WHERE session_id = ? AND turn_index = ?
                    ORDER BY turn_index DESC
                    LIMIT 1
                """, (session_id, current_turn_index - 1))

                row = cursor.fetchone()
                if row:
                    prev_arrival = row["arrival_timestamp"]
                    prev_latency_ms = row["total_latency_ms"]
                    prev_end = prev_arrival + (prev_latency_ms / 1000.0)
                    gap = time.time() - prev_end
                    return max(0.0, gap)  # Ensure non-negative

                return None
        except sqlite3.Error:
            return None

    async def _flush_batch(self) -> None:
        """Flush pending turns to database."""
        with self._lock:
            if not self._pending_turns or not self._conn:
                return

            turns_to_write = self._pending_turns[:]
            self._pending_turns.clear()
            self._last_flush_time = time.time()

        # Write to database (outside lock for better concurrency)
        try:
            with self._lock:
                for turn in turns_to_write:
                    # Upsert session
                    self._conn.execute("""
                        INSERT INTO sessions (
                            session_id, first_seen_at, last_seen_at, client_ip,
                            explicit_header, prefix_hash, turn_count, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                        ON CONFLICT(session_id) DO UPDATE SET
                            last_seen_at = ?,
                            turn_count = turn_count + 1
                    """, (
                        turn.session_id,
                        turn.arrival_timestamp,
                        turn.arrival_timestamp,
                        "unknown",  # We don't store client_ip for privacy
                        0,  # TODO: track explicit_header
                        turn.prompt_hash[:8],
                        turn.arrival_timestamp,
                        turn.arrival_timestamp,
                    ))

                    # Insert turn
                    self._conn.execute("""
                        INSERT INTO turns (
                            session_id, turn_index, arrival_timestamp, ttft_ms,
                            total_latency_ms, gap_since_previous_turn_seconds,
                            prompt_tokens, completion_tokens, prefix_overlap_tokens,
                            prefix_overlap_percent, contains_tool_calls, finish_reason,
                            prompt_hash, completion_hash, metadata, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
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
                    ))

                self._conn.commit()
        except sqlite3.Error as e:
            # Log but don't crash proxy
            print(f"Session tracking error: {e}")

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
        first_arrival = turns[0].arrival_timestamp
        last_turn = turns[-1]
        last_end = last_arrival + (last_turn.total_latency_ms / 1000.0)
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
        median_gap_seconds = sorted(gap_values)[len(gap_values) // 2] if gap_values else None

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
