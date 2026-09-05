"""Checkpoint and resume for golden protocol runs.

Provides atomic per-position checkpointing, manifest validation,
and automatic resume with outlier detection. Designed for unattended
execution over SSH on rented GPU pods.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CHECKPOINT_DIR = Path(".throttle/checkpoints")
MAX_STALENESS_HOURS = 24
SCHEMA_VERSION = "checkpoint.v1"


@dataclass(frozen=True)
class ManifestSnapshot:
    """Immutable snapshot of the run configuration.

    GPU fingerprint uses class-level attributes (model, VRAM, driver,
    CUDA) not device UUID, because rented pods assign different UUIDs
    on every allocation.
    """
    model_revision: str
    gpu_model: str
    gpu_vram_gb: int
    driver_version: str
    cuda_version: str
    engine_version: str
    workload_hash: str
    treatment_pair: Tuple[str, str]
    safety_limits: Dict[str, Any]

    def hash(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(raw.encode()).hexdigest()}"


@dataclass(frozen=True)
class PositionReport:
    """Metrics from a completed position."""
    requests_completed: int
    tokens_generated: int
    wall_clock_seconds: float
    spend_usd: float
    errors: int
    output_throughput_toks_per_s: float
    raw_report_path: str


@dataclass
class PositionCheckpoint:
    """Atomic checkpoint for a single golden position."""
    schema_version: str
    session_id: str
    position: str
    position_index: int
    total_positions: int
    completed_at_utc: str
    manifest_hash: str
    manifest_snapshot: Dict[str, Any]
    position_report: Dict[str, Any]


class CheckpointError(Exception):
    """Raised when checkpoint validation fails."""
    def __init__(self, message: str, exit_code: int = 4):
        super().__init__(message)
        self.exit_code = exit_code


class CheckpointManager:
    """Manages checkpoint lifecycle for a golden protocol session."""

    def __init__(self, session_id: Optional[str] = None):
        if session_id is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            rand = hashlib.sha256(str(time.time()).encode()).hexdigest()[:4]
            session_id = f"golden-{ts}-{rand}"
        self.session_id = session_id
        self.session_dir = CHECKPOINT_DIR / session_id

    def init_session(self, manifest: ManifestSnapshot) -> None:
        """Create session directory and write manifest atomically."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.session_dir / "manifest.json"
        data = {
            "session_id": self.session_id,
            "manifest_hash": manifest.hash(),
            "manifest_snapshot": asdict(manifest),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._atomic_write(manifest_path, data)

    def acquire_lock(self) -> None:
        """Write PID lock file. Refuse if another process is active."""
        lock_path = self.session_dir / "session.lock"
        if lock_path.exists():
            try:
                pid = int(lock_path.read_text().strip())
                os.kill(pid, 0)
                raise CheckpointError(
                    f"Session already active (PID {pid}). "
                    f"Wait for it to finish or delete {lock_path}.",
                    exit_code=5,
                )
            except (ProcessLookupError, ValueError, PermissionError):
                lock_path.unlink(missing_ok=True)

        lock_path.write_text(str(os.getpid()))

        def _cleanup(signum=None, frame=None):
            lock_path.unlink(missing_ok=True)
            if signum is not None:
                sys.exit(128 + signum)

        signal.signal(signal.SIGTERM, _cleanup)
        signal.signal(signal.SIGINT, _cleanup)

    def release_lock(self) -> None:
        lock_path = self.session_dir / "session.lock"
        lock_path.unlink(missing_ok=True)

    def write_position(
        self,
        position: str,
        position_index: int,
        total_positions: int,
        manifest: ManifestSnapshot,
        report: PositionReport,
    ) -> None:
        """Write checkpoint for a completed position atomically."""
        cp = PositionCheckpoint(
            schema_version=SCHEMA_VERSION,
            session_id=self.session_id,
            position=position,
            position_index=position_index,
            total_positions=total_positions,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            manifest_hash=manifest.hash(),
            manifest_snapshot=asdict(manifest),
            position_report=asdict(report),
        )
        filename = f"position-{position_index:02d}-{position}.json"
        path = self.session_dir / filename
        self._atomic_write(path, asdict(cp))

    def discover_incomplete_session(self) -> Optional[Path]:
        """Find the most recent incomplete session directory."""
        if not CHECKPOINT_DIR.exists():
            return None

        sessions = sorted(CHECKPOINT_DIR.iterdir(), reverse=True)
        for session_dir in sessions:
            if not session_dir.is_dir():
                continue
            manifest_path = session_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            checkpoints = list(session_dir.glob("position-*.json"))
            golden_report = session_dir / "golden.json"
            if not golden_report.exists():
                return session_dir
        return None

    def validate_resume(
        self,
        current_manifest: ManifestSnapshot,
    ) -> Tuple[int, List[str]]:
        """Validate that resume is safe.

        Returns (first_incomplete_index, list_of_completed_positions).
        Raises CheckpointError if resume is unsafe.
        """
        manifest_path = self.session_dir / "manifest.json"
        if not manifest_path.exists():
            raise CheckpointError("No manifest.json found in session directory")

        try:
            manifest_data = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as e:
            raise CheckpointError(f"Corrupt manifest.json: {e}")

        saved_hash = manifest_data.get("manifest_hash", "")
        current_hash = current_manifest.hash()

        if saved_hash != current_hash:
            saved_snap = manifest_data.get("manifest_snapshot", {})
            current_snap = asdict(current_manifest)
            diffs = []
            for key in sorted(set(list(saved_snap.keys()) + list(current_snap.keys()))):
                saved_val = saved_snap.get(key)
                current_val = current_snap.get(key)
                if saved_val != current_val:
                    diffs.append(f"  {key}: {saved_val!r} → {current_val!r}")
            diff_text = "\n".join(diffs) if diffs else "  (unknown difference)"
            raise CheckpointError(
                f"Manifest mismatch — refusing to resume.\n"
                f"Changes since original run:\n{diff_text}\n"
                f"A resumed run with different config produces worse data "
                f"than a failed run."
            )

        checkpoints = sorted(self.session_dir.glob("position-*.json"))
        if checkpoints:
            last_cp = json.loads(checkpoints[-1].read_text())
            last_time = datetime.fromisoformat(last_cp["completed_at_utc"])
            now = datetime.now(timezone.utc)
            hours_since = (now - last_time).total_seconds() / 3600
            if hours_since > MAX_STALENESS_HOURS:
                raise CheckpointError(
                    f"Checkpoint is stale ({hours_since:.1f}h old, max {MAX_STALENESS_HOURS}h). "
                    f"Thermal state has likely drifted. "
                    f"Refusing to resume to prevent corrupted comparison. "
                    f"Use --no-resume for a clean run."
                )

        completed_positions = []
        completed_indices = set()
        for cp_path in checkpoints:
            cp_data = json.loads(cp_path.read_text())
            idx = cp_data["position_index"]
            completed_indices.add(idx)
            completed_positions.append(cp_data["position"])

            raw_path = self.session_dir / cp_data["position_report"]["raw_report_path"]
            if not raw_path.exists():
                raise CheckpointError(
                    f"Checkpoint for {cp_data['position']} exists but "
                    f"raw report {raw_path} is missing."
                )

        # Golden positions are 1-indexed (position-01-B1.json)
        first_incomplete = 1
        while first_incomplete in completed_indices:
            first_incomplete += 1

        return first_incomplete, completed_positions

    def detect_outliers(
        self,
        treatment_groups: Dict[str, List[float]],
    ) -> List[str]:
        """Detect anomalous throughput using robust Modified Z-score (Iglewicz & Hoaglin)."""
        suspects = []
        
        def _median(vals: List[float]) -> float:
            s = sorted(vals)
            n = len(s)
            if n % 2 == 1:
                return s[n // 2]
            return (s[n // 2 - 1] + s[n // 2]) / 2.0

        for position, throughputs in treatment_groups.items():
            if len(throughputs) < 3:
                continue  # Need at least 3 points to establish a median reference
            
            med = _median(throughputs)
            abs_devs = [abs(x - med) for x in throughputs]
            mad = _median(abs_devs)
            
            # Avoid division by zero for identical runs by defining a lower bound
            if mad < 1e-9:
                mad = 0.05 * med if med > 0 else 1e-9

            for val in throughputs:
                # Standard Modified Z-score formula (0.6745 matches normal distribution scale)
                mod_z = 0.6745 * abs(val - med) / mad
                # 3.5 is the standard statistical threshold for Modified Z-score outliers
                if mod_z > 3.5:
                    suspects.append(position)
                    break
        return suspects

    @staticmethod
    def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
        """Write JSON data atomically via temp file + os.replace()."""
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(tmp_path).replace(".tmp", ""))
            os.replace(str(tmp_path).replace(".tmp", ""), str(path))
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise
