"""Tests for checkpoint atomicity, manifest validation, and resume logic."""

import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from throttle.checkpoint import (
    CheckpointManager,
    CheckpointError,
    ManifestSnapshot,
    PositionReport,
    CHECKPOINT_DIR,
    MAX_STALENESS_HOURS,
)


@pytest.fixture
def tmp_checkpoint_dir(tmp_path, monkeypatch):
    """Redirect CHECKPOINT_DIR to a temp directory."""
    cp_dir = tmp_path / "checkpoints"
    monkeypatch.setattr("throttle.checkpoint.CHECKPOINT_DIR", cp_dir)
    return cp_dir


@pytest.fixture
def sample_manifest():
    return ManifestSnapshot(
        model_revision="meta-llama/Llama-3.2-1B@main:abc123",
        gpu_model="NVIDIA A100-SXM4-80GB",
        gpu_vram_gb=80,
        driver_version="550.90.07",
        cuda_version="12.4",
        engine_version="vllm==0.6.2",
        workload_hash="sha256:e2b1f89",
        treatment_pair=("baseline:max_num_seqs=32", "candidate:max_num_seqs=64"),
        safety_limits={"max_requests": 200, "max_errors": 3},
    )


@pytest.fixture
def sample_report():
    return PositionReport(
        requests_completed=201,
        tokens_generated=48231,
        wall_clock_seconds=287.4,
        spend_usd=0.83,
        errors=0,
        output_throughput_toks_per_s=167.8,
        raw_report_path="B1.json",
    )


class TestAtomicWrite:
    def test_atomic_write_creates_valid_json(self, tmp_checkpoint_dir):
        mgr = CheckpointManager("test-session")
        mgr.session_dir.mkdir(parents=True, exist_ok=True)
        path = mgr.session_dir / "test.json"
        mgr._atomic_write(path, {"key": "value"})
        assert json.loads(path.read_text()) == {"key": "value"}

    def test_atomic_write_no_tmp_leftover(self, tmp_checkpoint_dir):
        mgr = CheckpointManager("test-session")
        mgr.session_dir.mkdir(parents=True, exist_ok=True)
        path = mgr.session_dir / "test.json"
        mgr._atomic_write(path, {"key": "value"})
        tmp_files = list(mgr.session_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_atomic_write_overwrites_existing(self, tmp_checkpoint_dir):
        mgr = CheckpointManager("test-session")
        mgr.session_dir.mkdir(parents=True, exist_ok=True)
        path = mgr.session_dir / "test.json"
        mgr._atomic_write(path, {"v": 1})
        mgr._atomic_write(path, {"v": 2})
        assert json.loads(path.read_text()) == {"v": 2}


class TestManifestValidation:
    def test_matching_manifest_passes(
        self, tmp_checkpoint_dir, sample_manifest, sample_report
    ):
        mgr = CheckpointManager("test-session")
        mgr.session_dir.mkdir(parents=True, exist_ok=True)
        mgr.init_session(sample_manifest)
        
        # Create dummy raw report
        raw_path = mgr.session_dir / "B1.json"
        raw_path.write_text("{}")
        
        mgr.write_position("B1", 1, 6, sample_manifest, sample_report)

        first_incomplete, completed, suspects = mgr.validate_resume(sample_manifest)
        assert first_incomplete == 2
        assert "B1" in completed
        assert len(suspects) == 0

    def test_divergent_model_refuses(
        self, tmp_checkpoint_dir, sample_manifest, sample_report
    ):
        mgr = CheckpointManager("test-session")
        mgr.session_dir.mkdir(parents=True, exist_ok=True)
        mgr.init_session(sample_manifest)
        
        # Create dummy raw report
        raw_path = mgr.session_dir / "B1.json"
        raw_path.write_text("{}")
        
        mgr.write_position("B1", 1, 6, sample_manifest, sample_report)

        changed = ManifestSnapshot(
            model_revision="different-model@main:xyz789",
            gpu_model=sample_manifest.gpu_model,
            gpu_vram_gb=sample_manifest.gpu_vram_gb,
            driver_version=sample_manifest.driver_version,
            cuda_version=sample_manifest.cuda_version,
            engine_version=sample_manifest.engine_version,
            workload_hash=sample_manifest.workload_hash,
            treatment_pair=sample_manifest.treatment_pair,
            safety_limits=sample_manifest.safety_limits,
        )

        with pytest.raises(CheckpointError, match="Manifest mismatch"):
            mgr.validate_resume(changed)

    def test_divergent_gpu_refuses(
        self, tmp_checkpoint_dir, sample_manifest, sample_report
    ):
        mgr = CheckpointManager("test-session")
        mgr.session_dir.mkdir(parents=True, exist_ok=True)
        mgr.init_session(sample_manifest)

        changed = ManifestSnapshot(
            model_revision=sample_manifest.model_revision,
            gpu_model="NVIDIA RTX 4090",
            gpu_vram_gb=24,
            driver_version=sample_manifest.driver_version,
            cuda_version=sample_manifest.cuda_version,
            engine_version=sample_manifest.engine_version,
            workload_hash=sample_manifest.workload_hash,
            treatment_pair=sample_manifest.treatment_pair,
            safety_limits=sample_manifest.safety_limits,
        )

        with pytest.raises(CheckpointError, match="Manifest mismatch"):
            mgr.validate_resume(changed)


class TestStaleness:
    def test_stale_checkpoint_refuses(
        self, tmp_checkpoint_dir, sample_manifest, sample_report
    ):
        mgr = CheckpointManager("test-session")
        mgr.session_dir.mkdir(parents=True, exist_ok=True)
        mgr.init_session(sample_manifest)
        
        # Create dummy raw report
        raw_path = mgr.session_dir / "B1.json"
        raw_path.write_text("{}")
        
        mgr.write_position("B1", 1, 6, sample_manifest, sample_report)

        cp_path = mgr.session_dir / "position-01-B1.json"
        cp_data = json.loads(cp_path.read_text())
        old_time = (
            datetime.now(timezone.utc) - timedelta(hours=25)
        ).isoformat()
        cp_data["completed_at_utc"] = old_time
        cp_path.write_text(json.dumps(cp_data))

        with pytest.raises(CheckpointError, match="stale"):
            mgr.validate_resume(sample_manifest)


class TestOutlierDetection:
    def test_normal_throughput_not_flagged(self, tmp_checkpoint_dir):
        mgr = CheckpointManager("test-session")
        groups = {
            "B": [165.0, 167.0, 168.0],
            "C": [170.0, 172.0, 171.0],
        }
        suspects = mgr.detect_outliers(groups)
        assert len(suspects) == 0

    def test_anomalous_throughput_flagged(self, tmp_checkpoint_dir):
        mgr = CheckpointManager("test-session")
        groups = {
            "B": [80.0, 165.0, 167.0, 168.0], # 80.0 is outlier
        }
        suspects = mgr.detect_outliers(groups)
        assert "B1" in suspects

    def test_insufficient_data_skips_check(self, tmp_checkpoint_dir):
        mgr = CheckpointManager("test-session")
        groups = {"B": [165.0, 167.0]}
        suspects = mgr.detect_outliers(groups)
        assert len(suspects) == 0


class TestSessionLock:
    def test_acquire_and_release(self, tmp_checkpoint_dir):
        mgr = CheckpointManager("test-session")
        mgr.session_dir.mkdir(parents=True, exist_ok=True)
        mgr.acquire_lock()
        lock_path = mgr.session_dir / "session.lock"
        assert lock_path.exists()
        assert int(lock_path.read_text().strip()) == os.getpid()
        mgr.release_lock()
        assert not lock_path.exists()

    def test_stale_lock_cleaned_up(self, tmp_checkpoint_dir):
        mgr = CheckpointManager("test-session")
        mgr.session_dir.mkdir(parents=True, exist_ok=True)
        lock_path = mgr.session_dir / "session.lock"
        lock_path.write_text("99999999")
        mgr.acquire_lock()
        assert int(lock_path.read_text().strip()) == os.getpid()
        mgr.release_lock()
