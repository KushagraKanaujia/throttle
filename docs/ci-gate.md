# CI cost gate (`throttle gate`)

`throttle gate` compares a candidate check with a baseline check and sets the
exit code:

| Verdict | Exit |
| --- | --- |
| MORE EXPENSIVE (beyond the run-to-run noise bound, 95% intervals apart) | **1** |
| OUTPUT CHANGED (answers changed length, or many more hit the token cap) | **1** |
| CHEAPER, NO WINNER | 0 |
| NOT CALIBRATED (no recent repeat checks of the baseline) | 0, with a warning: it cannot prove a regression |
| Usage error (unreadable file, unknown id) | 2 |

```sh
throttle check --url "$URL" --model "$MODEL" --gpu-hourly-rate 2.49 --json candidate.json
throttle gate --baseline baseline.json --candidate candidate.json
```

`--baseline` and `--candidate` take files written by `throttle check --json`
or check ids from the local history (`--history-dir`). The verdict is computed
locally by the same code as `throttle check`; the noise bound comes from the
baseline's repeat checks in the local history, so keep the history between
runs (the workflow below caches it). `--max-increase PCT` only fails increases
of at least PCT percent.

When `THROTTLE_API_KEY` is set (or you are logged in), the gate is also
recorded in Throttle Cloud (`POST /api/v1/ci/gate`, \$0.10 of credits) so it
shows in your team's history. `--no-cloud` keeps it local. A `--baseline`
that is a Throttle Cloud check id (not a local file or id) is judged by
Throttle Cloud. What is sent is listed in [cloud.md](cloud.md); it never
includes prompts, responses, endpoint URLs or keys. If the cloud call fails
(out of credits, unreachable), the gate still judges locally and only warns:
an unreachable cloud never fails the gate.

OUTPUT CHANGED compares answer length and token-cap hits between the two
checks; it does not check that answers are correct.

## GitHub Actions: copy this

Two jobs share the check history through the Actions cache:

- **baseline** (on `main`, and nightly so the baseline stays inside the 24 h
  calibration window) measures the deployed config 3 times, which calibrates
  the noise bound, and keeps the last check as `baseline.json`.
- **gate** (on pull requests that change the serving config) deploys the
  candidate, measures it once and fails the PR on MORE EXPENSIVE or OUTPUT
  CHANGED.

The runner must reach your inference server: usually a self-hosted runner on
the same network as the GPUs.

```yaml
# .github/workflows/inference-cost-gate.yml
name: Inference cost gate

on:
  push:
    branches: [main]
  schedule:
    - cron: "0 6 * * *"   # nightly: keep the baseline calibrated (24 h window)
  pull_request:
    paths: ["deploy/inference/**"]

permissions:
  contents: read

concurrency:
  group: inference-cost-gate-${{ github.ref }}
  cancel-in-progress: false

env:
  THROTTLE_CHECK_HISTORY_DIR: ${{ github.workspace }}/.throttle-checks
  URL: https://vllm.internal.example:8000
  MODEL: meta-llama/Llama-3.1-8B-Instruct
  GPU_RATE: "2.49"            # what one GPU costs you per hour
  WORKLOAD: --concurrency 8 --max-tokens 128 --blocks 5

jobs:
  baseline:
    if: github.event_name != 'pull_request'
    runs-on: [self-hosted, linux, gpu-network]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python -m pip install "throttle-pro==0.7.0"
      - name: Deploy main's config
        run: ./deploy/inference/deploy.sh   # your deploy; wait until /v1/models answers
      - name: Measure the baseline 3 times (calibrates the noise bound)
        env:
          VLLM_API_KEY: ${{ secrets.VLLM_API_KEY }}
        run: |
          for run in 1 2 3; do
            throttle check --url "$URL" --model "$MODEL" --gpu-hourly-rate "$GPU_RATE" \
              $WORKLOAD --api-key-env VLLM_API_KEY --label "main-${GITHUB_SHA::7}" \
              --json "$THROTTLE_CHECK_HISTORY_DIR/baseline.json"
          done
      - uses: actions/cache/save@v4
        with:
          path: ${{ env.THROTTLE_CHECK_HISTORY_DIR }}
          key: throttle-baseline-${{ github.run_id }}

  gate:
    if: github.event_name == 'pull_request'
    runs-on: [self-hosted, linux, gpu-network]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python -m pip install "throttle-pro==0.7.0"
      - uses: actions/cache/restore@v4
        with:
          path: ${{ env.THROTTLE_CHECK_HISTORY_DIR }}
          key: throttle-baseline-none
          restore-keys: throttle-baseline-
      - name: Deploy the PR's config
        run: ./deploy/inference/deploy.sh
      - name: Measure the candidate
        env:
          VLLM_API_KEY: ${{ secrets.VLLM_API_KEY }}
        run: |
          throttle check --url "$URL" --model "$MODEL" --gpu-hourly-rate "$GPU_RATE" \
            $WORKLOAD --api-key-env VLLM_API_KEY --label "pr-${{ github.event.number }}" \
            --json candidate.json
      - name: Cost gate
        env:
          THROTTLE_API_KEY: ${{ secrets.THROTTLE_API_KEY }}   # optional: records the gate in Throttle Cloud
        run: |
          throttle gate --baseline "$THROTTLE_CHECK_HISTORY_DIR/baseline.json" \
            --candidate candidate.json --json gate.json
      - if: always()
        run: cat gate.json >> "$GITHUB_STEP_SUMMARY" || true
```

Notes:

- Use the same workload options (`$WORKLOAD`) in both jobs: checks of
  different workloads are never compared (the gate passes with a "workload
  differs" reason).
- Without `THROTTLE_API_KEY` the gate runs fully locally and is free.
- `throttle check --fail-if-costlier PCT` is the older single-command gate
  (exit 4 / 5); see [github-action.md](github-action.md).
