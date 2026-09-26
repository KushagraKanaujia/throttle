# Throttle cost check in GitHub Actions

The `Throttle cost check` action runs `throttle check` against your
OpenAI-compatible endpoint (vLLM, SGLang, TGI, Ollama) on every deploy. It
measures dollars per million tokens, compares the result with the previous
check of the same endpoint, and writes the verdict to the job summary. It can
also fail the job when a serving change made inference measurably more
expensive.

What it does, step by step:

1. Installs `throttle-pro` with pip into a venv under `$RUNNER_TEMP`.
2. Restores the check history from the Actions cache (see
   [History and the Actions cache](#history-and-the-actions-cache)).
3. Runs `throttle check ... --json` with your inputs.
4. Writes a Markdown job summary: before and now $/M with 95% CIs, the
   verdict, the noise-floor status and what changed in the config.
5. Saves the history back to the cache, even when the check failed the gate.
6. Sets the step's exit status from Throttle's exit code (see
   [Exit behavior](#exit-behavior)).

## Example: check a vLLM deploy

```yaml
name: Deploy inference

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  # One deploy (and one history update) at a time; see "Concurrent runs".
  group: deploy-inference
  cancel-in-progress: false

jobs:
  deploy:
    # The runner must be able to reach the endpoint: usually a self-hosted
    # runner inside the same network as the GPU servers.
    runs-on: [self-hosted, linux, gpu-network]
    steps:
      - uses: actions/checkout@v4

      - name: Deploy vLLM
        run: ./deploy/vllm.sh   # your deploy; wait until /v1/models answers

      # The action needs Python 3.11+ on PATH. setup-python is your job.
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

      - name: Use the result
        if: always()
        env:
          VERDICT: ${{ steps.cost.outputs.verdict }}
          COST: ${{ steps.cost.outputs.cost-per-million }}
        run: echo "verdict $VERDICT, \$${COST}/M output tokens"
```

Pin the action to a release tag (as above) or a full commit SHA. With a
release tag like `v0.4.2` the action installs the matching `throttle-pro==0.4.2`
from PyPI. For any other ref it installs the latest release, unless you set
`throttle-version`.

## Calibration in CI: the first 3 runs are NOT CALIBRATED

A single check's 95% CI is computed within one run, so it cannot see
run-to-run drift: other tenants, thermal state, a noisy neighbor on the node.
Throttle only says CHEAPER or MORE EXPENSIVE when it has also measured that
drift. That takes repeat checks of an unchanged config within 24 hours:
3 earlier checks of one config (2 degrees of freedom), or 2 each of the
before and after configs. Until then the verdict is **NOT CALIBRATED**, and
this is by design:

| run | what it compares with | verdict |
|---|---|---|
| 1 | nothing (first check for this endpoint) | NOT CALIBRATED |
| 2 | run 1; noise has 0 degrees of freedom | NOT CALIBRATED |
| 3 | run 2; noise has 1 degree of freedom | NOT CALIBRATED |
| 4+ | the latest check; noise measured from runs 1-3 | CHEAPER, MORE EXPENSIVE or NO WINNER |

Two more rules matter in CI:

- **24-hour window.** Checks older than 24 h neither calibrate nor serve as a
  judged baseline. If you deploy once a week, every deploy check is NOT
  CALIBRATED unless something re-checks the running config in between.
- **The label is part of the config.** The default label is the short commit
  SHA, so two commits are two configs even if nothing about serving changed.
  Repeats of one commit calibrate each other; the next commit is then judged
  against them. If your deploys often don't touch serving, set `label` to
  something that only changes with the serving config (an image tag or a
  config file hash) so unchanged deploys pool their calibration.

The reliable pattern is a scheduled re-check of whatever is currently
deployed, so a fresh calibration is always waiting for the next deploy:

```yaml
name: Throttle calibration

on:
  schedule:
    - cron: "17 */6 * * *"   # every 6 h: at least 3 checks inside any 24 h
  workflow_dispatch:

jobs:
  recheck:
    runs-on: [self-hosted, linux, gpu-network]
    concurrency: { group: deploy-inference, cancel-in-progress: false }
    steps:
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - uses: KushagraKanaujia/throttle@v0.4.2
        env:
          VLLM_API_KEY: ${{ secrets.VLLM_API_KEY }}
        with:
          # Same url, model, rate, label, config and workload inputs as the
          # deploy job. The label must match what is deployed right now.
          url: https://vllm.internal.example:8000
          model: meta-llama/Llama-3.1-8B-Instruct
          gpu-hourly-rate: "2.49"
          api-key-env: VLLM_API_KEY
          label: ${{ vars.DEPLOYED_SHA }}
          config: |
            gpu=H100
            quantization=fp8
            max_num_seqs=256
          concurrency: "8"
```

For a one-off, run the deploy workflow's check 3 times (for example with
`workflow_dispatch`) before the change you want judged.

Keep the workload inputs (`blocks`, `requests-per-block`, `concurrency`,
`max-tokens`, prompts) identical across runs: a different workload is never
compared. The Throttle version is part of the config too, so pin the action
(or `throttle-version`) and expect a fresh calibration after an upgrade.

## Exit behavior

The step passes through Throttle's exit codes, with one policy input for
NOT CALIBRATED:

| `throttle check` exit | meaning | the action |
|---|---|---|
| 0 | done: cheaper, no winner, or not calibrated without the gate | passes |
| 1 | measurement failed, nothing recorded | fails (1), summary says ERROR |
| 2 | usage error | fails (2) |
| 4 | calibrated MORE EXPENSIVE by at least `fail-if-costlier` % | **fails (4)** |
| 5 | `fail-if-costlier` is set but the change could not be judged: NOT CALIBRATED, or NO WINNER because the baseline ran a different workload (e.g. another prompt cache mode, the first run after upgrading from 0.4.0) | `not-calibrated: warn` (default): `::warning::`, passes. `fail`: fails (5) |

The very first check of an endpoint has no baseline, so Throttle exits 0.
With `fail-if-costlier` set, the action treats it like exit 5 (it is just as
unjudged) and applies `not-calibrated`: a warning by default, or a failure
with `fail`. The `exit-code` output always holds Throttle's raw code.

`not-calibrated` does nothing without `fail-if-costlier`; the action warns if
you set `fail` alone.

## Inputs

| input | required | default | description |
|---|---|---|---|
| `url` | yes | | OpenAI-compatible base URL, e.g. `https://vllm.internal:8000` or `.../v1`. Plain HTTP only to loopback unless `extra-args: --allow-insecure-http`. |
| `gpu-hourly-rate` | yes | | What the GPU(s) behind the endpoint cost you per hour, in dollars. ASSUMED: Throttle cannot see your bill. Include every GPU the model uses. |
| `model` | no | the only model the server lists | Model name to send requests to. |
| `label` | no | short commit SHA | Short name for this config. Part of the config fingerprint (see calibration). |
| `config` | no | | Serving settings the endpoint cannot report, one `KEY=VALUE` per line. `#` lines are ignored. Values appear in the summary; never put secrets here (Throttle refuses secret-looking keys). |
| `fail-if-costlier` | no | | Percent. Fail (exit 4) on a calibrated MORE EXPENSIVE by at least this much. Empty: report only. |
| `not-calibrated` | no | `warn` | `warn` or `fail`: what NOT CALIBRATED does while `fail-if-costlier` is set. |
| `blocks` | no | Throttle's (5) | Measurement blocks; the 95% CI is across blocks (minimum 3). |
| `requests-per-block` | no | Throttle's (4) | Requests per block. |
| `concurrency` | no | Throttle's (2) | Requests in flight. Cost depends heavily on it: match production. |
| `max-tokens` | no | Throttle's (64) | Max output tokens per request. |
| `api-key-env` | no | Throttle's (`OPENAI_API_KEY`) | Name of the env var holding the bearer key. Set that variable on the step from a secret. |
| `extra-args` | no | | More `throttle check` arguments, shell-quoted, e.g. `--monthly-tokens 500M --max-estimated-spend 10`. `--url`, `--json`, `--history` and `--history-dir` are set by the action and refused here. |
| `throttle-version` | no | `auto` | `auto` (the release matching this action's tag, else latest), `latest`, a version like `0.4.1`, or a local path (used by the self-test). |
| `history-key` | no | hash of `url` + `model` | Cache key for the history. Letters, digits, `.`, `_`, `-`. Change it to start a fresh history. |

Spend guard: Throttle refuses a check whose worst case (`--max-elapsed-seconds`,
300 s by default, times the GPU rate) could cost more than $3. For a rate
above $36/hr, raise it with `extra-args: --max-estimated-spend 10`.

## Outputs

| output | description |
|---|---|
| `verdict` | `CHEAPER`, `MORE EXPENSIVE`, `NO WINNER`, `NOT CALIBRATED` (includes the first check), or `ERROR` (Throttle wrote no result). |
| `cost-per-million` | Measured $ per million tokens of the headline metric (output tokens unless `--metric` is passed), unrounded. |
| `change-percent` | Measured change vs the baseline, in percent, at the baseline's GPU rate (what the verdict judges). Empty when there is no comparable baseline. |
| `exit-code` | Throttle's raw exit code: 0, 1, 2, 4 or 5. |
| `check-id` | Id of this check in the history. |

## History and the Actions cache

Throttle keeps checks in a `checks.ndjson` file. On a fresh runner that file
would be empty every time, so the action carries it in the Actions cache:

- It restores with `restore-keys: throttle-check-v1-<length of history-key>-<history-key>-`
  (the length keeps `prod` from matching caches saved for `prod-eu`), which
  picks the newest saved history for that endpoint and model, and saves under
  a new unique key after every run (cache entries cannot be overwritten). The
  history therefore grows run after run.
- Restored lines are merged with any history already on the runner (earlier
  invocations in the same job), deduplicated, and ordered by time, because
  Throttle compares against the newest check of the endpoint.

Tradeoffs to know:

- **Branch scope.** A run can restore caches saved on its own branch and on
  the default branch, not on other branches. Deploys from `main` share one
  history. A PR branch starts from `main`'s history, and its checks stay on
  that branch: they never become `main`'s baseline. That is usually what you
  want: a PR experiment does not become production's baseline.
- **Concurrent runs.** Two runs that overlap both restore the same history and
  each save their own; the next run restores only the newer one, so the other
  run's check is lost from the history (not from its own summary). Put
  deploys and scheduled re-checks in one `concurrency` group.
- **Eviction.** GitHub evicts cache entries unused for 7 days and trims
  repositories over their cache quota, oldest first. The history file is
  small (a few KB per check). Losing it costs you calibration: the next 3
  runs are NOT CALIBRATED again. Checks older than 24 h do not calibrate
  anyway, so a weekly gap loses little.
- **Entries pile up.** Each run saves one small entry. Old ones age out; you
  can delete them in the repository's Actions > Caches page.
- **Fresh start.** Change `history-key` (for example after moving to new
  hardware you do not want compared with the old) to begin a new history.

## Privacy and security

- **API keys.** The key is never an input. Put it in a secret, expose it to
  the step with `env:`, and name that variable in `api-key-env`. The action
  never prints it, and Throttle only sends it as the bearer header.
  `config` keys that look like secrets are refused by Throttle.
- **Endpoint.** The action registers the URL, host and host:port as log
  masks before anything runs, so Throttle's own log lines show `***`. The job
  summary never contains the URL or host: URLs in any value (for example a
  `config` entry) are replaced with `[url hidden]`, the endpoint host with
  `[endpoint]`, and an endpoint change is shown as `(hidden)`. The default
  cache key uses a hash of the URL, not the URL.
- **What the summary does show.** Model name, label, the GPU rate you typed,
  $/M figures and CIs, `config` keys and values, and fingerprint changes read
  from `/v1/models` and vLLM's `/metrics` info labels. Anyone who can read the
  workflow run can read it.
- **What the cache holds.** The history file stores the endpoint URL, model,
  config fingerprint and results. Actions caches are readable by workflows in
  the repository, including pull requests from forks, which can restore caches
  saved on the default branch: a fork PR can add a workflow step that
  restores `throttle-check-v1-*` and read the file. Fork runs cannot write to
  the default branch's cache, so they cannot plant a fake baseline. In a
  public repository, treat the endpoint URL as exposed to anyone who opens a
  pull request, or do not use this action there.
- **Traffic and cost.** A check sends `blocks x requests-per-block` requests
  plus one warm-up (21 by default) with at most `max-tokens` output tokens
  each. On a per-token API that is your bill ceiling; see Throttle's safety
  limits in `throttle check --help`.

## Runner requirements

- Linux or macOS runner with bash. Python 3.11+ on `PATH`
  (`actions/setup-python`). pip must be able to reach PyPI (or your mirror).
- Network access from the runner to the endpoint. For a private cluster this
  usually means a self-hosted runner.

## Self-test

`.github/workflows/action-selftest.yml` runs this repository's `action.yml`
(`uses: ./`) on every pull request against a tiny stdlib mock server started
in the job: three same-config runs (NOT CALIBRATED, exit codes 0, 5, 5, with
`not-calibrated` warn and then fail) and a fourth after making the mock 3x
slower (calibrated MORE EXPENSIVE, exit 4). It asserts the verdicts, exit
codes, step outcomes, and that no summary contains the endpoint, a URL or the
key.
