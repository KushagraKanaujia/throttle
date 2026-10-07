# How Throttle decides

This page answers the questions a technical team asks before running Throttle
against a real inference server. Every mechanism below names the file and
function that implements it, and every number comes from the code's defaults,
a saved artifact in [`validation/`](../validation/), or a recorded run quoted
in the [README](../README.md). If this page and the code disagree, the code
is right; please open an issue.

Contents:

1. [Does the noise bound hold on my traffic?](#1-does-the-noise-bound-hold-on-my-traffic)
2. [How much GPU time does a check cost?](#2-how-much-gpu-time-does-a-check-cost)
3. [What exactly does OUTPUT CHANGED compare?](#3-what-exactly-does-output-changed-compare)
4. [Cheaper than what?](#4-cheaper-than-what)
5. [How is a savings figure computed?](#5-how-is-a-savings-figure-computed)
6. [How does it fit what I already have?](#6-how-does-it-fit-what-i-already-have)

---

## 1. Does the noise bound hold on my traffic?

**Short answer:** the bound is not a constant shipped with Throttle. It is
measured from repeat checks you run on your own server, for one endpoint, one
config and one workload, within the last 24 hours. Until those repeats exist,
Throttle refuses to call a winner.

### Two kinds of noise

- **Within one check.** A check runs `--blocks` independent blocks (default 5,
  minimum 3, enforced in `check.handle_check`). The $/M figure is the mean
  across blocks, with a Student t 95% interval across blocks
  (`statistics.t_interval_95`, called from `check.summarize`).
- **Between checks.** A within-run interval cannot see drift between runs
  (other tenants, thermal state, a background job). That is the noise that
  makes an unchanged server look 10% cheaper or dearer. Throttle measures it
  separately in `check.run_to_run_noise`.

### How the run-to-run bound is computed (`check.run_to_run_noise`)

1. **The pool.** All recorded checks in your local history
   (`~/.throttle/checks/checks.ndjson`), except the check being judged, that
   were created within **24 hours** of it (`CALIBRATION_WINDOW_HOURS = 24.0`).
   Older or undated checks are counted as ignored, not used.
2. **The groups.** A check belongs to a group by `check._config_identity`:
   the same endpoint, the same flattened config fingerprint (model, `--label`,
   GPU rate, settings read from the server, `--config` pairs), the same
   workload identity (`WORKLOAD_IDENTITY_FIELDS`, see section 4) and the same
   Throttle version. Only two groups count: the baseline's config and this
   check's config (one group if they are the same).
3. **The SD.** For each group, the relative deviation of each check's $/M
   mean from the group mean, squared and pooled across both groups. Degrees
   of freedom are `sum(n - 1)` over the groups.
4. **The minimum.** `df >= 2` (`MIN_CALIBRATION_DF = 2`). That is 3 earlier
   checks of one unchanged config, or 2 each of the before and after configs.
5. **The bound.** `t(0.975, df) x sqrt(2) x SD`. The `sqrt(2)` is because the
   verdict compares two single checks, each carrying its own noise. With 2 df,
   t is 4.303, so the bound is about 6.1 x SD; more repeats shrink both t and
   the uncertainty in SD.

`check.compare_checks` then calls CHEAPER or MORE EXPENSIVE only if **both**
hold: the measured change is larger than the bound, **and** the two checks'
95% intervals do not overlap. Otherwise it is NO WINNER, with every failed
condition named.

Recorded examples (README, MacBook, local Ollama `llama3.2:3b`, rate ASSUMED
at \$1.50/hr):

- Three unchanged checks disagreed by up to 9.8%, giving a 5.0% run-to-run SD
  and a **30.4%** bound (2 df). A fourth check measured 11.5% cheaper with
  nothing changed on the server, and the verdict was NO WINNER.
- Four checks in a row of the same server: \$8.77, \$8.86, \$11.26, \$10.02 per
  million output tokens. SD 14.6% at 2 df gives a bound of **89.1%**, so the
  -11.0% between the last two was NO WINNER. A laptop is a noisy machine; the
  point is that the bound reflects the machine it was measured on.

### Why it is per endpoint and per config, on your server

Noise depends on the hardware, the engine, its settings, the load you send
and whatever else shares the box. A bound measured on someone else's GPU says
nothing about yours. So the bound is computed only from your own history, and
only from checks with the same identity as the two being compared. The
`--label` is part of the identity: rerunning an unchanged server under a new
label starts a new group, so it cannot calibrate itself.

When two checks of the same config and workload have non-overlapping 95%
intervals, `run_to_run_noise` also prints a hint naming both checks: your
run-to-run noise is larger than the within-run noise, so a within-run CI
alone understates the uncertainty.

### What NOT CALIBRATED means

`compare_checks` returns NOT CALIBRATED when:

- the noise pool has fewer than 2 degrees of freedom, or
- the baseline check is more than 24 hours older than the check being judged
  (drift over a gap longer than any calibration spanned is unmeasured).

It is a refusal, not a soft pass. The before and after figures, both
intervals and the change are still printed, but no direction and no monthly
figure. With `--fail-if-costlier`, it exits 5 (`EXIT_NOT_CALIBRATED`) so CI
can treat it as a warning or a failure. The fix is to run the same check
again without changing anything.

### How `--workload agent` makes the traffic agent-shaped

The default check sends single-turn built-in prompts, each tagged so that a
prefix cache cannot serve repeats. `--workload agent` sends synthetic
multi-turn agent sessions instead. The generator is
[`src/throttle/agent_workload.py`](../src/throttle/agent_workload.py); the
runner is `check._run_agent_block` and `check._measure_agent`.

- **Shared prefix.** Every session of a run starts with the same system
  message: about `--system-prompt-tokens` (default 1500) of instructions plus
  a JSON tool schema (`agent_workload.system_prompt`). Sessions share that
  prefix and nothing after it, because each session's first user message is a
  task seeded by its index (`agent_workload.session_task`).
- **Growing history.** Turn k resends turn k-1's messages plus a scripted
  assistant turn (about `--assistant-turn-tokens`, default 150) and a tool
  result (about `--tool-output-tokens`, default 300)
  (`agent_workload.turn_messages`). So each turn's prompt starts with the
  previous turn's prompt, and a prefix cache can reuse it within the session,
  as it would for a real agent.
- **Scripted history.** The model's live reply is requested, measured and
  counted, but never appended to the conversation. Every run of one shape
  therefore sends exactly the same prompts apart from the run tag, and one
  reply that drifts cannot change the later prompts of its session. README
  recording (local Ollama `qwen2.5:0.5b`): prompt tokens per turn were 1586,
  2063, 2540, 3017, 3494, 3971 in every run; across three repeat runs mean
  output length moved -0.2% and +1.1%. An earlier version that fed live
  replies back moved 12% to 23% between runs.
- **Run tag.** The system message starts with `[run NNNNNN] `
  (`agent_workload.NONCE_FORMAT`), six random digits per check
  (`check.new_run_id`). Reuse within a run is intended; reuse across runs is
  blocked, so one run's cache cannot make the next run look cheaper. One
  unmeasured warm-up request (`agent_workload.warmup_messages`) puts the run's
  system prefix in the cache before block 1.
- **Shape.** A block is `--sessions-concurrency` concurrent sessions (default:
  `--concurrency`, which defaults to 2) of `--turns` sequential turns
  (default 6). The shape parameters, seed and generator version are hashed
  into the workload identity (`check._agent_workload_record`), so agent checks
  are only ever compared with agent checks of the same shape.
- **Recorded measurements.** Mean prompt and completion tokens per turn (and
  by turn) from the server's `usage`, and, when the server reports
  `usage.prompt_tokens_details.cached_tokens` (vLLM, SGLang), the mean cached
  prompt tokens per turn. Cached tokens are recorded only; they are not part
  of the identity or the verdict.

### The honest limit

The agent workload is a **synthetic shape**, not a replay of your logs. It
reproduces the structure that drives cost on agent traffic (a long shared
prefix, a growing context, sequential turns, concurrent sessions), with sizes
you set to match yours. It does not reproduce your prompt text, your tool
outputs, your reply lengths, your arrival pattern or your production
concurrency. The token sizes are estimates without a tokenizer (about one
token per plain word, one per 4 characters of JSON); the tokens actually sent
are measured from the server and recorded. Cost depends heavily on
concurrency, so a check at concurrency 2 measures cost at concurrency 2; set
`--concurrency` / `--sessions-concurrency` to match production.

The noise bound holds for the workload you measured it on. If your
production traffic differs from that workload, the bound tells you whether a
config change moved cost on that workload, not on your exact mix.

---

## 2. How much GPU time does a check cost?

**Short answer:** a check is capped at 300 seconds of wall-clock time, and
before sending anything it prints the exact request count, token ceiling and
worst-case GPU time at your rate. At \$2.49/hr the worst case is \$0.21.

### The limits line

Before any traffic, `check.handle_check` prints a line of this form, and
`check._limit_problem` refuses the check (exit 2, nothing sent) if any limit
is exceeded:

```text
  limits     25 requests, at most 1,348 output tokens requested, stops after 300s; worst-case GPU time $0.21 [ASSUMED rate] (ceiling $3.00)
```

(That example is the default workload at `--gpu-hourly-rate 2.49`, computed
from the formulas below.)

- **Requests** = blocks x requests per block + warm-ups + tag probes.
- **Output tokens requested** = (blocks x requests per block + warm-ups) x
  `--max-tokens` + probes x 1. This is a ceiling; the server usually returns
  fewer.
- **Worst-case GPU time** = `--gpu-hourly-rate` x `--max-elapsed-seconds` /
  3600. The run is wrapped in `asyncio.wait_for(..., timeout=max_elapsed_seconds)`;
  a check that hits the ceiling is stopped and nothing is recorded.
- **Ceiling** = `--max-estimated-spend`, default \$3.00
  (`models.SafetyLimits`). With the 300 s default, any rate above \$36/hr is
  refused until you lower `--max-elapsed-seconds` or raise the ceiling.

### The defaults, computed

| | Default workload | `--workload agent` (defaults) |
| --- | --- | --- |
| Blocks | 5 | 5 |
| Requests per block | 4 | 2 sessions x 6 turns = 12 |
| Concurrency | 2 requests in flight | 2 sessions at once, turns sequential |
| Warm-up requests | 1 | 1 |
| Tag probes (unmeasured, 1 output token) | 4 | 0 |
| **Total requests** | **25** | **61** |
| `--max-tokens` per request | 64 | 64 |
| **Output tokens requested (ceiling)** | **1,348** | **3,904** |
| Prompt tokens sent | short built-in prompts | about 167,000 measured on `qwen2.5:0.5b` (10 sessions x 16,671; see section 1), plus the warm-up |
| Wall-clock ceiling | 300 s | 300 s |
| Worst-case GPU time at \$2.49/hr | \$0.21 | \$0.21 |
| Worst-case GPU time at \$1.39/hr | \$0.12 | \$0.12 |

The worst-case GPU time is the same for both because it is the time ceiling
times your rate, not a prediction. Real checks usually finish well inside it;
the agent workload sends far more prompt tokens, so it takes longer on the
same server.

### What a first verdict costs

A directional verdict needs calibration: at least 3 checks of the baseline
config plus 1 of the candidate (or 2 + 2), all within 24 hours. That is at
least 4 checks, so at most 4 x 300 s = 20 minutes of endpoint time in the
worst case, at the concurrency you chose.

### Where to run it

`check` only sends ordinary chat-completion requests (plus a read of
`/v1/models` and `/metrics`). It can run against a staging replica with the
same GPU, engine and flags as production, or against production off-peak.
Running it on a shared production server at peak adds your production load
to the measurement, which widens the noise bound; that is what the bound is
for, but a quieter window gives a tighter one.

---

## 3. What exactly does OUTPUT CHANGED compare?

**Short answer:** the length of the answers and how often they hit
`max_tokens`. Not their content.

### The mechanism (`check.output_shape`, `check.output_shift`)

Every check runs at temperature 0 (`check._send_with_reply` sends
`"temperature": 0`). For each check, `output_shape` computes from the
recorded blocks:

- **Output tokens per request**: total `usage.completion_tokens` across all
  blocks / (blocks x requests per block).
- **Max-tokens hit rate**: the share of measured requests whose
  `finish_reason` was `"length"` (counted per block as `max_tokens_hits`).

`output_shift` compares the baseline and the candidate and marks the output
as changed if either:

- output tokens per request moved by **more than 10%** in either direction
  (`OUTPUT_SHIFT_PERCENT = 10.0`), or
- the max-tokens hit rate **rose** by **20 percentage points or more**
  (`MAX_TOKENS_HIT_SHIFT_POINTS = 20.0`). A fall does not trigger it.

In `compare_checks`, this test runs after the workload-identity test and
before the calibration test. So OUTPUT CHANGED overrides CHEAPER, MORE
EXPENSIVE, NO WINNER and NOT CALIBRATED: no direction and no monthly figure
are reported, and `--fail-if-costlier` exits 5. Checks recorded before
Throttle tracked `max_tokens_hits` are compared by output length only; checks
with no usable block data skip the test.

### Why it exists

A cheaper $/M can come from broken output. The case it was built for, as
saved in [`validation/hotaisle-mi300x-20261001/`](../validation/hotaisle-mi300x-20261001/)
(check records plus answer samples): on an MI300X, vLLM's on-the-fly FP8 on Qwen2.5-32B looked
41% cheaper while one answer was 256 tokens of `"!!!!"` and every request ran
to `max_tokens` (8,192 output tokens per block against 6,883 for BF16).

### What it does NOT do

- It does **not** compare answer text. The reply text is read in
  `_send_with_reply` and then discarded; it is not stored, hashed or diffed.
- It does **not** judge quality, correctness or meaning. There is no
  semantic similarity, no evaluation set, no LLM judge.
- It does **not** catch a change that keeps answers about the same length.
  A quantized model that gives wrong answers of normal length passes.
- It does **not** look at single requests. It compares averages over the
  whole check, so a few broken answers among many normal ones can stay under
  10%.

OUTPUT CHANGED is a tripwire for the obvious failure, not a quality gate. If
answer quality matters for a change (quantization, a new model revision, a
new engine version), run your own evaluation as well.

---

## 4. Cheaper than what?

**Short answer:** cheaper than a baseline check of the same endpoint, on the
same workload, priced at the same per-GPU rate, with any change in GPU count
counted as real cost.

### The baseline

By default the baseline is the most recent recorded check of the same
endpoint URL (`check.handle_check`). `--against CHECK_ID` picks another one.
The comparison prints every fingerprint difference Throttle can see
(`check.diff_fingerprints`): model, label, GPU rate, the vLLM `*_info` labels
from `/metrics`, whether the model is listed at `/v1/models`, and your
`--config KEY=VALUE` pairs. A setting that could be read on only one side is
dropped from both and noted (`check._comparable_flat`), so a failed read does
not look like a config change.

### The 1:1 rule

A verdict is only meaningful as a one-change comparison:

- **One change at a time.** Throttle lists every change it can see, but it
  does **not** refuse a comparison with several changes, or with a different
  model, in `check`. If you change two things, a CHEAPER verdict tells you the
  pair was cheaper, not which change did it. Making one change at a time, and
  declaring settings the server cannot report with `--config`, is your part.
  (`throttle savings` does refuse a model mismatch; see section 5.)
- **The same workload identity.** Enforced. See below.
- **GPU count is real cost.** Enforced. See below.

### How mismatched workloads are refused (`check.compare_checks`)

Before computing any delta, `compare_checks` compares
`WORKLOAD_IDENTITY_FIELDS` between the two checks:

- `prompts_sha256` (the prompt file, or for agent checks the hash of the
  generator, seed and shape)
- `requests_per_block`
- `concurrency`
- `max_tokens`
- `prompt_cache_mode` (cold or `--warm-cache`; checks from 0.4.0 and earlier
  read as warm)
- `workload_shape` (the agent shape parameters; `None` for the default
  workload, so an agent check never matches a default one)

If any differ, the verdict is NO WINNER, **no delta is computed or printed**
("vs base not computed (different workloads)"), and the reason names the
fields. With `--fail-if-costlier` this exits 5. `blocks` and `warmup` are not
part of the identity.

### How the GPU rate is handled

The GPU \$/hr is something you type, recorded as ASSUMED. $/M scales linearly
with it, so a typed price cut alone must never be a saving. `compare_checks`
scales the candidate by `factor = max(1, per_gpu_rate(baseline) /
per_gpu_rate(candidate))` before judging (`check.rate_factor`), where
`per_gpu_rate = rate / gpu_count`. A lower per-GPU rate is put back on the
baseline's; a higher per-GPU rate (a pricier GPU type, or a total rate for
more GPUs typed without `--gpus`) is never scaled down, so it counts as real
cost.

The GPU count (`--gpus N`, or a `gpus` `--config` value; default 1, see
`check.gpu_count`) stays in the verdict, because serving on 2 GPUs instead of
1 really doubles the hourly cost. The case saved in
[`validation/hotaisle-mi300x-20261001/`](../validation/hotaisle-mi300x-20261001/): Qwen2.5-72B on 2x MI300X was 1.56x faster but \$2.14 vs \$1.67/M,
28% more per token. Judged at the 1-GPU rate it had been called 36% CHEAPER.

README recording of the rate rule (made with 0.5.x): the same unchanged
server checked at \$3.00/hr against a \$1.50/hr baseline went up 94.2% in $/M
at the typed rates, but -2.9% at the baseline's rate, inside a 30.4% bound:
NO WINNER, exit 0. Since 0.6.0 a rate rise is not scaled away, so that pair
is MORE EXPENSIVE (+94.2%, CIs apart). The GPU count is part of the
flattened fingerprint (`gpu_count`, listed only when it is not 1), so a GPU
count change shows as a config change and its checks form their own
calibration group.

### Decision-grade comparisons

`check` compares two runs taken at different times. It is a guard rail, not
a controlled experiment. For a decision-grade baseline vs candidate answer,
`throttle golden` runs a six-position counterbalanced protocol (B C B C B C
in time order). The one decision-eligible result in this repo
([`validation/golden-live-20260817/`](../validation/golden-live-20260817/RUN_AUDIT.md)):
vLLM 0.16.0, A100 80GB PCIe, Qwen2.5-0.5B-Instruct, `max_num_seqs` 1 -> 8,
1,206 valid measured requests, order-balanced throughput **+217.85%**, 95%
interval **+189.47% to +246.22%**; README reports \$0.746 -> \$0.234/M output
tokens (-68.6%) at \$1.39/hr. The baseline was deliberately bad; it shows
the measurement, not a saving to expect.

---

## 5. How is a savings figure computed?

**Short answer:** `throttle savings` re-judges two recorded checks, refuses
unless the verdict is a calibrated CHEAPER, and multiplies the **smallest
plausible** $/M reduction, allowing for both within-run and run-to-run noise,
by your production token count. Implemented in
[`src/throttle/savings.py`](../src/throttle/savings.py), `build_statement`.

### The formula

```text
CI term ($/M)            = baseline CI low - candidate CI high
noise term ($/M)         = (1 - noise bound) x baseline mean - candidate mean
conservative savings ($) = min(CI term, noise term) x tokens / 1e6
point estimate ($)       = (baseline mean - candidate mean) x tokens / 1e6
```

- Both sides are in the checks' primary metric ($/M output tokens by
  default).
- The candidate's mean and interval are scaled by the same `factor` as the
  verdict, `max(1, baseline per-GPU rate / candidate per-GPU rate)`: a typed
  per-GPU price cut never counts as savings, and a per-GPU price rise (a
  pricier GPU type) always counts as real cost. A change in GPU count counts
  too.
- The noise bound is the run-to-run bound the re-judged verdict used (the
  `t x sqrt(2) x pooled SD` of section 1), as a fraction.
- The conservative figure is the headline. Both terms are printed (and are
  in `--json` as `ci_term_dollars_per_million`, `noise_term_dollars_per_million`
  and `noise_bound_percent`, record_version 2). The point estimate is shown
  and labelled "not the verified figure".

The CI term is the reduction that holds even if the baseline was as cheap as
its interval allows and the candidate as expensive as its interval allows.
That only covers within-run noise; two checks taken at different times also
drift. The noise term takes the run-to-run bound off the baseline before
subtracting, so drift that the verdict treats as noise is never billed.

### Refusals (exit 1, one-line reason)

`build_statement` refuses when:

- `--baseline` and `--candidate` are the same check, or either id is not in
  the history;
- the two checks used different models;
- they used different $/M metrics;
- any `WORKLOAD_IDENTITY_FIELDS` differ;
- the token source is MEASURED but the metric is not output tokens (the vLLM
  counter counts output tokens only);
- the re-judged verdict is not CHEAPER: NO WINNER, NOT CALIBRATED, OUTPUT
  CHANGED and MORE EXPENSIVE are all refused. The re-judgment uses only the
  history that existed when the later of the two checks was recorded;
- either check has no 95% interval;
- the conservative reduction (the smaller term) is zero or negative;
- a MEASURED token window starts before the candidate check was taken
  ("the window starts before the candidate config was measured; take a new
  snapshot after deploying it."): tokens served before the candidate config
  existed are not its savings.

### Token sources

Exactly one of:

- **REPORTED BY OPERATOR**: `--tokens N` (e.g. `1.2B`). Throttle records it
  as reported, not measured.
- **MEASURED**: `throttle savings snapshot --metrics-url ... --out start.json`
  at the period start reads every series of vLLM's
  `vllm:generation_tokens_total` counter. At the period end,
  `throttle savings --metrics-url ... --window-start start.json` reads it
  again (`savings.measured_tokens`). Per series, it keeps only the checks'
  `model_name` (when the server reports that label), takes end minus start,
  and sums. It refuses if the snapshot came from a different URL, if a series
  went down (a counter reset), or if a series present at the snapshot has
  disappeared. A series that is new since the snapshot counts from 0 and is
  noted.

### What the statement assumes

Each statement lists its assumptions: the GPU \$/hr on both sides is
ASSUMED; $/M was measured on the check's workload (for agent checks, the
named synthetic shape) and production traffic is assumed to cost per token
what that workload did; and the token count's source. The statement contains
no fees and no pricing. With `--json` it is one `savings_statement` record.

---

## 6. How does it fit what I already have?

**Short answer:** it is a client of your existing OpenAI-compatible endpoint.
It installs nothing on the server and changes nothing on it.

- **OpenAI-compatible endpoint.** `check` sends `POST /v1/chat/completions`
  with `temperature 0`, `stream false` and your `--max-tokens`, and reads
  `GET /v1/models`. It needs the server to return `usage.prompt_tokens` and
  `usage.completion_tokens`; if it does not, the check fails rather than
  guessing token counts (`check._send_with_reply`). Engines measured end to
  end: vLLM, SGLang, Ollama and LMDeploy
  ([`validation/runpod-five-stack-20260819/`](../validation/runpod-five-stack-20260819/),
  descriptive only, not decision-grade). Auth is a bearer key read from an
  environment variable you name (`--api-key-env`); the key is never an
  argument.
- **`/metrics` if exposed.** `check` reads `<host>/metrics` (or
  `--metrics-url`) once, with a 3 s timeout and a 4 MB cap
  (`check._fetch_metrics`), and records the labels of vLLM `vllm:*_info`
  lines as part of the config fingerprint (`check.parse_info_labels`). If it
  is not reachable, the check continues without it; `--no-metrics` skips it.
  Only vLLM's `*_info` lines are parsed, so on other engines the fingerprint
  is what `/v1/models` and your `--config` pairs provide. `throttle savings`
  reads `vllm:generation_tokens_total` only when you pass `--metrics-url`.
- **GitHub Action.** [`action.yml`](../action.yml) runs `throttle check ...
  --json` on each deploy, keeps the check history in the Actions cache,
  writes the verdict to the job summary, and can fail the job on a calibrated
  MORE EXPENSIVE (`fail-if-costlier`). NOT CALIBRATED warns or fails
  according to `not-calibrated`. Setup: [docs/github-action.md](github-action.md).
- **`--json`.** `check --json PATH` writes the full record (fingerprint,
  workload, per-block tokens and timing, intervals) and its comparison,
  including the noise groups, to a file. `savings --json` prints the savings
  statement as one record. Exit codes (0, 1, 2, 4, 5) are listed in the README.
- **Local history.** Checks are appended to `~/.throttle/checks/checks.ndjson`
  (or `--history-dir`, or `$THROTTLE_CHECK_HISTORY_DIR`). Nothing is uploaded;
  `--share` prints a sanitized summary and a GitHub issue link you submit
  yourself.

**Throttle never changes your server.** It provisions nothing, restarts
nothing and sets no flags. `--config KEY=VALUE` only records what you
changed. Making the change, and rolling it back, is yours.
