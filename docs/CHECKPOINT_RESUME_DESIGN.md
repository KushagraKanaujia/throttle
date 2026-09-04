# Checkpoint and Resume for Golden Protocol

## Problem
One failed block invalidates an entire golden run. At 67 requests per
position with even a 0.5% failure rate, roughly one run in three dies.
At 6 positions this is painful. At 16 (N=4 Williams) it makes the
protocol unusable regardless of statistical merit.

## Non-Negotiable Contracts

- Atomicity: Checkpoint written via os.replace() from a temporary file to prevent partial writes.
- Completeness: Written only after position completes 100%, never during execution.
- Manifest Lock: Resume validates manifest hash against original configuration.
- Explicit Refuse: Divergent manifest exits with code 4 and a detailed diff.
- Auto-Discovery: Resume detects incomplete directories automatically (opt-out via --no-resume).
- No Interactive Prompt: Resume is automatic when manifest matches (prevents SSH hangs).
- 24h Hard Refuse: If last checkpoint is older than 24 hours, exit with code 4 to prevent thermal drift corruption.
- Outlier Detection: Re-runs completed positions if their throughput deviates more than 2 sigma from treatment mean.

## Manifest Fingerprint

GPU fingerprint uses class-level attributes, not device identity:
- GPU model name (e.g., NVIDIA A100-SXM4-80GB)
- VRAM size (e.g., 80GB)
- Driver version (e.g., 550.90.07)
- CUDA version (e.g., 12.4)

Device UUID is deliberately excluded because rented pods assign different UUIDs on every allocation even for the same physical GPU.

## Checkpoint Schema (per position)

    {
      "schema_version": "checkpoint.v1",
      "session_id": "golden-20260903-142315-a7f3",
      "position": "B2",
      "position_index": 3,
      "total_positions": 6,
      "completed_at_utc": "2026-09-03T14:31:22.481Z",
      "manifest_hash": "sha256:8f7c...9d4a",
      "manifest_snapshot": {
        "model_revision": "meta-llama/Llama-3.2-1B-Instruct@main:abc123",
        "gpu_model": "NVIDIA A100-SXM4-80GB",
        "gpu_vram_gb": 80,
        "driver_version": "550.90.07",
        "cuda_version": "12.4",
        "engine_version": "vllm==0.6.2",
        "safety_limits": {
          "max_requests": 200,
          "max_tokens": 50000,
          "max_elapsed_seconds": 300,
          "max_spend_usd": 1.0,
          "max_errors": 3
        },
        "workload_hash": "sha256:e2b1...4f89",
        "treatment_pair": ["baseline:max_num_seqs=32", "candidate:max_num_seqs=64"]
      },
      "position_report": {
        "requests_completed": 201,
        "tokens_generated": 48231,
        "wall_clock_seconds": 287.4,
        "spend_usd": 0.83,
        "output_throughput_toks_per_s": 167.8,
        "raw_report_path": "B2.json"
      }
    }

## Directory Layout

- .throttle/checkpoints/golden-20260903-142315-a7f3/
  - manifest.json: Hash + configuration snapshot
  - position-01-B1.json: Checkpoint metadata for B1
  - position-02-C1.json: Checkpoint metadata for C1
  - B1.json: Raw report metrics (referenced by checkpoint)
  - C1.json: Raw report metrics (referenced by checkpoint)
  - session.lock: PID of active running process

## Resume State Machine (Logic Flow)

1. Is the --no-resume flag active?
   - Yes: Start a clean fresh run.
   - No: Search .throttle/checkpoints/ for incomplete session directories.

2. Was an incomplete session directory discovered?
   - No: Start a clean fresh run.
   - Yes: Check if session.lock contains an active running PID.

3. Is the lock active?
   - Yes: Exit with code 5 ("Already running").
   - No: Load manifest.json and validate hash against current run parameters.

4. Does the manifest match?
   - No: Exit with code 4 and print configuration diff.
   - Yes: Calculate hours elapsed since the last completed position metadata.

5. Is the last checkpoint older than 24 hours?
   - Yes: Exit with code 4 ("Stale session, thermal drift detected").
   - No: Run outlier detection on completed positions.

6. Did any completed position deviate more than 2 sigma from its group mean?
   - Yes: Flag position as SUSPECT and schedule for re-run.
   - No: Accept completed positions and resume execution from first incomplete index.

## Refuse Conditions (Exit 4)

- Model revision changed: Different model or revision hash
- GPU class changed: Different model, VRAM, driver, or CUDA version
- Engine version changed: Different vLLM/SGLang version
- Safety limits changed: Limits became less restrictive
- Workload hash changed: Different prompt set or traffic shape
- Stale checkpoint: Older than 24 hours since last completed position
- Corrupt manifest: Unparseable or missing required fields
- Missing raw report: Checkpoint metadata exists but B1.json raw report is gone

## Outlier Detection

On resume, for each completed position:
1. Group positions by treatment (baseline vs candidate)
2. Compute mean and std of output_throughput_toks_per_s per group
3. If any position deviates more than 2 sigma from its group mean, flag as SUSPECT
4. Re-run suspect positions instead of trusting them
5. If fewer than 2 positions per group (not enough for sigma), skip check

This catches transient thermal throttling, noisy neighbors, and background processes that completed without errors but produced anomalous throughput.

## CLI Integration

No new subcommand. Auto-discovery via checkpoint directory.
One new flag: --no-resume to force a clean run.

    # Normal run (auto-resumes if checkpoint exists)
    throttle golden --config baseline.yaml --config candidate.yaml

    # Force clean run (ignores existing checkpoints)
    throttle golden --config baseline.yaml --config candidate.yaml --no-resume

## Scope

This PR touches:
- docs/CHECKPOINT_RESUME_DESIGN.md (this document)
- src/throttle/checkpoint.py (new module)
- src/throttle/golden.py (hook: write checkpoint after position)
- src/throttle/cli.py (add --no-resume flag to golden subparser)
- tests/test_checkpoint.py (atomicity, refuse, manifest, outlier)

Does NOT touch: benchmark.py, compare.py, N-condition logic.
