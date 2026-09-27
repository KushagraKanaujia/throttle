# Gate costs in CI (GitHub Action)

*Fail a deploy that makes every token more expensive.*

![](../../illustrations/ci.svg)

The **Throttle LLM cost check** action runs `throttle check` against your endpoint on every deploy, writes the verdict to the job summary, and can fail the job on a calibrated cost regression.

```yaml
- uses: actions/setup-python@v5
  with:
    python-version: "3.12"

- name: Throttle cost check
  id: cost
  uses: KushagraKanaujia/throttle@v0.4.2
  env:
    VLLM_API_KEY: ${{ secrets.VLLM_API_KEY }}
  with:
    url: https://vllm.internal.example:8000
    model: meta-llama/Llama-3.1-8B-Instruct
    gpu-hourly-rate: "2.49"        # what 1x H100 costs you per hour
    api-key-env: VLLM_API_KEY      # the NAME of the env var, never the key
    label: ${{ github.sha }}
    config: |
      gpu=H100
      quantization=fp8
      max_num_seqs=256
    concurrency: "8"               # match production load
    fail-if-costlier: "5"          # fail on a calibrated +5% or worse
    not-calibrated: warn
```

The runner must be able to reach the endpoint (usually a self-hosted runner in the same network as the GPU servers). The action needs Python 3.11+ on PATH.

## The first three runs are NOT CALIBRATED

| Run | Compared with | Verdict |
| --- | --- | --- |
| 1 | nothing (first check) | NOT CALIBRATED |
| 2 | run 1; 0 degrees of freedom | NOT CALIBRATED |
| 3 | run 2; 1 degree of freedom | NOT CALIBRATED |
| 4+ | the latest check; noise from runs 1–3 | CHEAPER, MORE EXPENSIVE or NO WINNER |

Checks older than 24 hours don't calibrate. If you deploy rarely, add a scheduled workflow that re-checks the running config every 6 hours so a fresh calibration is always waiting.

## Exit behavior

- **4**: calibrated MORE EXPENSIVE by at least `fail-if-costlier` percent.
- **5**: the change couldn't be judged (NOT CALIBRATED or a different workload). The `not-calibrated` input decides whether that warns or fails.

> **Tip:** Pin the action to a release tag like `v0.4.2`; it then installs the matching `throttle-pro==0.4.2`. The full reference is in the repo's `docs/github-action.md`.
