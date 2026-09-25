# Throttle

**Know what your LLM inference costs in dollars per million tokens, and whether your last serving-config change made it cheaper or more expensive, with statistics that won't call a winner from overlapping or uncalibrated evidence.**

Throttle is an open-source CLI for any OpenAI-compatible endpoint: vLLM,
SGLang, Ollama, LMDeploy and similar servers. It is for teams that self-host
models, pay for GPU time, and want to know their real $/M tokens instead of
guessing.

```text
change a serving setting ──► throttle check ──► $/M tokens (95% CI) + what changed + verdict
        ▲                                                                   │
        └──────────── keep the change, or roll it back ◄────────────────────┘
```

## The problem

- **You pay for GPU hours but budget in $/M tokens.** The conversion between
  the two is throughput, and throughput depends on the serving config: batch
  size, `max_num_seqs`, quantization, engine version, GPU type.
- **Config changes ship blind.** Someone changes a flag, sends a few curl
  requests, eyeballs the latency and deploys. Nobody re-measures $/M tokens, so
  a change that makes every token more expensive shows up weeks later on the
  bill.
- **Noise looks like a win.** Two measurements taken at different times differ
  from machine load alone. On a MacBook, five checks of the same unchanged
  local Ollama server, all started within three minutes, measured between $8.26
  and $8.59 per million output tokens: a 4.0% spread, although a single check's
  95% interval was as narrow as about ±1%. A tool that calls the lowest one "4% cheaper"
  is reporting noise.

## What Throttle does

| | Command | What you get |
| --- | --- | --- |
| **Price it** | `throttle cost`, `throttle watch` | Dollars per million tokens, measured against a live endpoint (`cost`) or read from vLLM `/metrics` (`watch`). The GPU $/hr is yours and always labelled ASSUMED; tokens and time are MEASURED. |
| **Re-check it after every config change** | `throttle check` | $/M tokens with a 95% confidence interval, what changed in the config, and a verdict against the last check. CHEAPER or MORE EXPENSIVE only when the measured change is larger than a run-to-run noise bound estimated from at least 3 recent repeat checks **and** the intervals don't overlap. Otherwise NO WINNER, or NOT CALIBRATED when there is no recent noise measurement yet. `--fail-if-costlier` makes it a CI gate. |
| **Prove a config change** | `throttle plan` → `smoke` → `benchmark` → `golden` | Traffic that is planned and capped before any request goes out, repeated measurement blocks, 95% intervals, and a six-position counterbalanced protocol for a decision-grade baseline-vs-candidate answer. |
| **Cache (one lever, off by default)** | `throttle proxy --enable-cache` | A proxy in front of your model server that answers exact and near-exact repeated prompts from memory. Semantic matching is a separate opt-in with a known false-match risk (see [Caching proxy](#caching-proxy)). |
| **Profile agents** | `throttle proxy --enable-session-tracking` + `throttle sessions` | A per-session breakdown of time spent generating versus waiting, an estimate of redundant prefill, and suggested backend settings to check. |

Throttle provisions nothing and never changes your server. It measures and
reports; `--config KEY=VALUE` only records what you changed.

## Receipts

Every number below comes from a saved artifact in this repo or from a
recorded local run. Nothing here is a projection.

| What | Result | Where it came from |
| --- | --- | --- |
| Config decision on a real GPU | `max_num_seqs` 1 → 8 on vLLM 0.16.0: **+189.5% to +246.2% throughput** (95% CI, point estimate +217.8%) | A100 80GB, Qwen2.5-0.5B-Instruct, six-position golden protocol, `decision_eligible: true`. [`validation/golden-live-20260817/`](validation/golden-live-20260817/) |
| Cross-engine compatibility | vLLM, SGLang, Ollama and LMDeploy all measured end to end | RunPod GPUs, descriptive only (not decision-grade). [`validation/runpod-five-stack-20260819/`](validation/runpod-five-stack-20260819/) |
| Agent session profile | A simulated 4-turn agent: 41.9% of wall-clock time spent waiting on the client, an estimated 54.3% of prompt tokens redundant prefill | Local Ollama `llama3.2:3b` on a MacBook (Apple M3 Pro, Metal), no datacenter GPU. The tool pauses are scripted at 1.5 s each. |
| Proxy cache hit | Cold call **1.75 s**, then the same prompt from cache in **1.2 ms**, then a reworded prompt from cache in **9.5 ms** | Local Ollama `llama3.2:3b` on a MacBook (Apple M3 Pro, Metal), no datacenter GPU. 4 requests, 2 backend calls. |
| Benchmark-harness cache on realistic traffic | 27.0% hit rate, total runtime 39.46 s → 28.28 s | Local Ollama `llama3.2:1b`, 73-prompt traffic sample. [`validation/CACHE_VALIDATION_SUMMARY.md`](validation/CACHE_VALIDATION_SUMMARY.md) |

The golden result is the **only** decision-eligible result in this repo. It
applies to that exact model, engine, GPU and workload, and it is not a savings
projection. The cache and profiler numbers were measured on a laptop and show
how the mechanism works. They do not predict hit rates or savings on your
traffic. (The proxy cache row used the opt-in semantic tier,
`--enable-embeddings`; the default proxy has no semantic tier.) See
[RESULTS.md](RESULTS.md) for the full evidence and its limitations.

## Install

Throttle needs Python 3.11+. Install it from PyPI with pipx:

```sh
pipx install throttle-pro
throttle --version   # 0.4.1
```

Add the `embeddings` extra (`pipx install 'throttle-pro[embeddings]'`) only if
you want the proxy's opt-in semantic cache tier. To hack on Throttle itself,
see [Install from source](#install-from-source).

## Quick demo (5 minutes, no GPU)

**1. The simulator (no server, no network, about 1 second):**

```sh
throttle demo
```

```text
Metric                                       Baseline        Tuned        Delta
--------------------------------------------------------------------------------
Wall clock time (seconds)                       77.87        63.14       -14.73
GPU hours                                    0.021630     0.017538    -0.004092
Blended cost ($/M tokens, in + out)              0.08         0.07        -0.02
Input cost ($/M tokens)                          0.35         0.28        -0.07
Output cost ($/M tokens)                         0.11         0.09        -0.02
Total cost ($)                                 0.0324       0.0263      -0.0061
...
All values above are [SIMULATED] - they depend entirely on assumed
throughput parameters, not real hardware measurements.
```

Steps 2 to 5 need [Ollama](https://ollama.com/download) running locally with
`ollama pull llama3.2:3b`. That is enough on a laptop. Run `throttle` with no
arguments for the same getting-started steps in the terminal.

**2. Price your endpoint.** The hourly rate is one you supply. A laptop has no
real GPU price, so the examples here assume $1.50/hour unless they say
otherwise:

```sh
throttle cost --url http://localhost:11434 --model llama3.2:3b \
  --gpu-hourly-rate 1.50 --num-requests 5
```

Recorded on a MacBook (Apple M3 Pro, local Ollama 0.32.5, `llama3.2:3b`;
GPU rate ASSUMED):

```text
Measured Cost [MEASURED time x ASSUMED $1.50/hr]:
  GPU hours: 0.001494
  Total cost: $0.0022
  Blended cost: $2.87 per million tokens (input + output, 780 tokens)
  Input cost: $3.78 per million tokens
  Output cost: $11.98 per million tokens
  Note: input and output $/M each charge the whole run to one token type;
  they are not additive. Compare the blended figure with a per-token bill.
```

The input and output figures each charge the full run cost to that one token
type, so don't add them together; the blended line is the one to hold next to
a per-token bill. The prompt is synthetic and requests run one at a time.

`--url` and `--endpoint-url` are the same flag, and both `http://host:port`
and `http://host:port/v1` work here and in `measure`, `check` and
`plan`/`smoke`/`benchmark`. (`proxy` takes `--backend-url`.) `--model` can be
left out of `cost`, `measure` and `check` only: if the server lists exactly
one model at `/v1/models`, that model is used and named in the output; if it
lists several, the error names them.

**3. Re-check cost after a config change.** `throttle check` is the command to
rerun. It measures $/M tokens in repeated blocks, fingerprints the serving
config it can see (`/v1/models`, vLLM `/metrics` `*_info` labels, plus
anything you declare with `--config KEY=VALUE`), saves the result to
`~/.throttle/checks/`, and compares it with the previous check of the same
endpoint. The workflow:

```sh
# 1. record today's config, three times, changing nothing: the repeats measure run-to-run noise
for i in 1 2 3; do
  throttle check --url http://localhost:11434 --model llama3.2:3b \
    --gpu-hourly-rate 1.50 --label before
done
# 2. change one server setting (for Ollama, e.g. restart it with OLLAMA_NUM_PARALLEL=4), then:
throttle check --url http://localhost:11434 --model llama3.2:3b \
  --gpu-hourly-rate 1.50 --label after --config OLLAMA_NUM_PARALLEL=4 \
  --monthly-tokens 500M
```

What those commands printed on a MacBook (Apple M3 Pro, local Ollama
0.32.5, `llama3.2:3b`, default check workload of 5 blocks x 4 requests; GPU
rate ASSUMED). The first check has nothing to compare with:

```text
Result
  $8.64 per million output tokens   [MEASURED]
  95% CI $8.08 to $9.20, across 5 blocks (Student t)
  same GPU spend per other tokens (not additive): $12.93/M input tokens; $5.18/M tokens (input + output)
  = $1.50/hr [ASSUMED] x measured wall-clock / measured tokens, at concurrency 2. Cost at a different production concurrency will differ.

Compared with: nothing yet. This is the first check for this endpoint;
the next check will be compared against it.
```

The second and third `before` checks are compared with the one before them,
but there is not yet enough repeat evidence for a verdict. The second:

```text
Compared with check 20260924T054047Z-aae3a6b5 (2026-09-24T05:40:47Z, 1 min ago, label before)
  What changed in the config: nothing Throttle can see (add --config KEY=VALUE for settings the endpoint does not report)
  before  $8.64/M output tokens  (95% CI $8.08 to $9.20)
  now     $9.36/M output tokens  (95% CI $8.62 to $10.10)
  change  +$0.7198 per million output tokens (+8.3%)
  noise   run-to-run noise bound: not measured (needs 3 earlier checks of one config and workload within 24 h, not counting this one; or 2 each of the before and after configs)
          baseline and this check (same config): label before, 1 earlier check(s), no repeat yet
  why     run-to-run noise unknown: 0 degree(s) of freedom from repeat checks within 24 h, 2 needed (e.g. 3 checks of the baseline config, not counting this one).
  Verdict: NOT CALIBRATED — run-to-run noise unknown. Two checks at different times can differ from load alone. Run 'throttle check' at least 3 times without changing anything (within 24 h) to measure your noise, or use 'throttle golden' for a counterbalanced decision.
```

The third, with 2 earlier checks (1 degree of freedom), is still not enough:
the range of one pair is not an estimate of noise.

```text
Compared with check 20260924T054117Z-c4d9627f (2026-09-24T05:41:17Z, 1 min ago, label before)
  What changed in the config: nothing Throttle can see (add --config KEY=VALUE for settings the endpoint does not report)
  before  $9.36/M output tokens  (95% CI $8.62 to $10.10)
  now     $9.49/M output tokens  (95% CI $8.62 to $10.36)
  change  +$0.1294 per million output tokens (+1.4%)
  noise   run-to-run noise bound: not measured (needs 3 earlier checks of one config and workload within 24 h, not counting this one; or 2 each of the before and after configs)
          baseline and this check (same config): label before, 2 earlier check(s), spread 8.3%
  why     run-to-run noise unknown: 1 degree(s) of freedom from repeat checks within 24 h, 2 needed (e.g. 3 checks of the baseline config, not counting this one).
  Verdict: NOT CALIBRATED — run-to-run noise unknown. Two checks at different times can differ from load alone. Run 'throttle check' at least 3 times without changing anything (within 24 h) to measure your noise, or use 'throttle golden' for a counterbalanced decision.
```

For this recording Ollama was **not** actually restarted, so the server was
unchanged and the right answer to step 2 is "no winner". It measured 11.5%
cheaper anyway, with a tighter CI than the baseline's, and Throttle did not
call it a win or project a monthly saving, because the three unchanged
`before` checks already disagreed by up to 9.8%:

```text
Compared with check 20260924T054147Z-fa7d9cda (2026-09-24T05:41:47Z, 2 min ago, label before)
  What changed in the config:
    config.OLLAMA_NUM_PARALLEL: (not set) -> 4
    label: before -> after
  before  $9.49/M output tokens  (95% CI $8.62 to $10.36)
  now     $8.40/M output tokens  (95% CI $8.14 to $8.67)
  change  -$1.09 per million output tokens (-11.5%)
  noise   run-to-run noise bound 30.4% [MEASURED from 3 earlier checks of unchanged configs within 24 h: t x sqrt(2) x 5.0% SD, 2 df]
          baseline config: label before, 3 earlier check(s), spread 9.8%
          this check's config: label after, 0 earlier check(s), no repeat yet
  monthly not projected: with NO WINNER the difference is not distinguishable from noise
  Verdict: NO WINNER, the change (-11.5%) is not larger than the run-to-run noise bound (30.4% = t x sqrt(2) x 5.0% run-to-run SD, 2 df); and the 95% confidence intervals overlap, so the difference is within measurement noise.
```

See
[`throttle check`: verdicts and exit codes](#throttle-check-verdicts-and-exit-codes)
for the exact rules and what a changed GPU rate does.

**4. Optional: the caching proxy.** Caching is one cost lever. It is off by
default, and it only helps when your traffic repeats itself. Start the proxy
in one terminal:

```sh
throttle proxy --backend-url http://localhost:11434 --port 8090 --enable-cache
```

In a second terminal, send a prompt, the same prompt again, a reworded
version and an unrelated one:

```sh
ask() { curl -s -o /dev/null -w "HTTP %{http_code}  %{time_total}s\n" \
  localhost:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"llama3.2:3b\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":60,\"temperature\":0}"; }
ask "Give me three tips for writing a good README."        # cold: goes to the model
ask "Give me three tips for writing a good README."        # exact repeat
ask "give me three tips for writing a great README"        # reworded
ask "Give me three tips for writing a good cover letter."  # new topic: goes to the model
curl -s localhost:8090/health
```

Recorded on a MacBook (Apple M3 Pro, local Ollama 0.32.5):

```text
HTTP 200  1.386167s
HTTP 200  0.001539s
HTTP 200  1.356098s
HTTP 200  1.406699s
{"status":"ok","cache_enabled":true,"cache_stats":{"hits":1,"misses":3,"evictions":0,"exact_hits":1,"lexical_hits":0,"embedding_hits":0,"embedding_scans_attempted":0,"embedding_comparisons_performed":0,"backend_calls":3}}
```

The exact repeat came from the cache in 1.5 ms. The reworded prompt shares
too few words with the first for the default lexical tier, so it went to the
model. The opt-in semantic tier (`--enable-embeddings`) would catch it, at a
cost described under [Caching proxy](#caching-proxy).

**5. Profile an agent.** Start the proxy with session tracking:

```sh
throttle proxy --backend-url http://localhost:11434 --port 8091 --enable-session-tracking
```

Point your agent's OpenAI base URL at `http://127.0.0.1:8091/v1` and run it.
Adding an `X-Throttle-Session: <id>` header to each request is optional;
without it, Throttle groups a growing conversation into one session by
itself. The proxy writes turns to disk in batches (every 5 to 10 seconds, and
when it stops), so wait about 10 seconds after the last request. Then look at
the results:

```sh
throttle sessions                  # list recent sessions
throttle sessions <SESSION_ID>     # breakdown for one session (ID or unique prefix)
```

Here is what it recorded for a simulated 4-turn coding agent that paused
1.5 s per "tool call":

```text
Wall Clock Breakdown:
  Generation:  6.3s (58.1%)  ███████████
  Client Wait: 4.5s (41.9%)  ████████

Prefix Overlap:
  Average: 38.3%
  Total redundant (estimated): 424 tokens (54.3% of all prompt tokens)

[HIGH] Prefix Caching
  Observation: 54.3% of prompt tokens are redundant prefill (424 tokens across 4 turns)
  Config: vLLM: --enable-prefix-caching (on by default in recent V1 releases) | SGLang: RadixAttention prefix cache is on unless --disable-radix-cache is set
```

## `throttle check`: verdicts and exit codes

A 95% CI is computed across blocks inside one run, so it cannot see
run-to-run drift (machine load, thermal state, other tenants). Two checks
taken at different times are not a controlled comparison. So `check` also
estimates a **run-to-run noise bound** from its own recent history:

- A *same-config group* is every stored check with the same endpoint, the
  same config fingerprint (model, `--label`, GPU rate, server-reported
  settings, `--config` pairs), the same workload (prompts, requests per block,
  concurrency, max tokens, prompt cache mode) and the same Throttle version.
- Only two groups count: the baseline's config and this check's config (one
  group if they are the same). The check being judged is never part of its own
  noise estimate, and only checks created within 24 hours of it count.
- The run-to-run SD is the pooled relative standard deviation of the groups'
  $/M point estimates, with `df = sum(n - 1)` over the groups. It needs
  `df >= 2`: for example 3 earlier checks of the baseline config, or 2 each of
  the before and after configs.
- The bound is `t(0.975, df) x sqrt(2) x SD`, a 95% bound on the difference
  between two single checks of one unchanged config. With 2 df, t is 4.303, so
  3 repeats give a wide bound; more repeats tighten it.
- A baseline more than 24 hours older than the check is NOT CALIBRATED:
  drift over that gap was never measured. Re-check the baseline config first.

The label is part of the config: rerunning an unchanged server under a new
label starts a new group, so it does not calibrate itself.

| Verdict | When | Monthly figure |
| --- | --- | --- |
| **CHEAPER** / **MORE EXPENSIVE** | Same workload, \|measured change\| is larger than the run-to-run bound, **and** the 95% CIs do not overlap | With `--monthly-tokens`, the change per month is printed, labelled PROJECTED |
| **NO WINNER** | The change is not larger than the bound or the CIs overlap (every failed condition is named), or the two checks used different workloads | Not projected |
| **NOT CALIBRATED** | Fewer than 2 df of recent repeat checks, or the baseline is more than 24 h old | Not projected; before/now, both CIs, the change and the reason are still printed |

When two checks of the same config and workload have non-overlapping CIs,
`check` also prints a `hint` line naming both check IDs and saying that
"your machine's run-to-run noise is larger than within-run noise, so a
within-run CI alone understates the uncertainty".

**The GPU rate is ASSUMED, so it never decides a verdict.** $/M scales
linearly with the rate you type. When it differs from the baseline's, the
check is put on the baseline's rate and only that measured part (throughput)
is judged; the rate-driven part is printed separately as arithmetic. The same
laptop, the same unchanged server, `--gpu-hourly-rate 3.00` against the third
`before` check above:

```sh
throttle check --url http://localhost:11434 --model llama3.2:3b \
  --gpu-hourly-rate 3.00 --label rate-3.00 --against <CHECK_ID> \
  --no-save --fail-if-costlier 5
```

```text
Compared with check 20260924T054147Z-fa7d9cda (2026-09-24T05:41:47Z, 1 min ago, label before)
  What changed in the config:
    gpu_hourly_rate_usd (ASSUMED): 1.5000 -> 3.0000
    label: before -> rate-3.00
  before  $9.49/M output tokens  (95% CI $8.62 to $10.36)
  now     $18.42/M output tokens  (95% CI $17.12 to $19.72)
  change  +$8.94 per million output tokens (+94.2%)
  note    the GPU rate changed ($1.50/hr -> $3.00/hr) [ASSUMED]: that alone moves $/M by +100.0%, arithmetic on rates you typed, not a measurement.
          At the old rate this check measures $9.21/M (-2.9% vs before) [MEASURED throughput]; the verdict judges only that part.
  noise   run-to-run noise bound 30.4% [MEASURED from 3 earlier checks of unchanged configs within 24 h: t x sqrt(2) x 5.0% SD, 2 df]
          baseline config: label before, 3 earlier check(s), spread 9.8%
          this check's config: label rate-3.00, 0 earlier check(s), no repeat yet
  Verdict: NO WINNER, the measured change at the baseline's GPU rate (-2.9%) is not larger than the run-to-run noise bound (30.4% = t x sqrt(2) x 5.0% run-to-run SD, 2 df); and the 95% confidence intervals overlap, so the difference is within measurement noise.
```

It exited with code 0. $/M at the typed rates went up 94.2%, but nothing
measured got worse (-2.9% at the old rate, inside the 30.4% bound), so it is
not MORE EXPENSIVE and the CI gate does not fail. `<CHECK_ID>` is an ID from
`throttle check --history`. With `--json`, the check, its comparison and the
noise groups are also written to a file.

None of these recordings produced CHEAPER or MORE EXPENSIVE: the server never
actually changed. The test suite (`tests/test_check.py`) covers the directional
verdicts, the CI gate's exit 4, and the cases above.

**Exit codes:**

| Code | Meaning |
| --- | --- |
| `0` | Check done: CHEAPER, NO WINNER, first check, or NOT CALIBRATED without `--fail-if-costlier` |
| `1` | Measurement failed; nothing recorded |
| `2` | Usage error |
| `4` | `--fail-if-costlier PCT` tripped: a calibrated MORE EXPENSIVE whose measured change (at the baseline's GPU rate) is at least PCT percent |
| `5` | `--fail-if-costlier` given and the change could not be judged: NOT CALIBRATED, or the baseline ran a different workload (NO WINNER, e.g. a cold check against a 0.4.0 or `--warm-cache` baseline); treat it as a warning or a failure |

Other options:

```sh
throttle check --history                                # every past check, all endpoints
throttle check --history --url http://localhost:11434   # only this endpoint
throttle check --history --limit 5                      # only the newest 5
```

`--history` sends no traffic. On a measuring check, `--against CHECK_ID`
compares with a chosen check instead of the latest one, `--json PATH` also
writes the check, its comparison and the noise groups as JSON, and
`--no-save` compares without adding the check to history.

`check` sends its own fixed workload (built-in prompts, 5 blocks x 4
requests, concurrency 2, max 64 output tokens by default), so its $/M is not
comparable with `throttle cost`; compare checks with checks. Keep the
workload flags identical between checks, or the comparison is NO WINNER. It
is capped before any traffic like `smoke`: at most 10,000 requests, 1,024
output tokens per request, 2,000,000 requested output tokens, concurrency 64,
300 seconds, and a worst-case GPU time (rate x time ceiling) of $3.00; each
has a `--max-*` flag to raise it. `check` is still not a controlled
experiment; for a decision-grade answer use the golden protocol below.

### Cold vs warm cache

Servers with prefix caching (vLLM V1, where `--enable-prefix-caching` is on
by default; SGLang RadixAttention; Ollama's KV-cache reuse) skip the prefill
of a prompt prefix they have seen before. A benchmark that resends the same
prompts every block and every run is then served warm, and its $/M looks
cheaper than traffic whose prompts differ.

Since 0.4.1, `throttle check` is **cold by default**: the first user message
of every measured request starts with a unique tag such as
`[run 482915 req 0012] `, so no two requests share a prefix beyond the chat
template (and any system message you put in front, as real traffic would).
The run id is 6 random decimal digits per check, stored in the record
(`workload.prompt_nonce`) with a sha256 of the exact prompts sent, so the
traffic can be rebuilt and audited. Digits tokenize the same way every run,
so the tag costs a constant number of prompt tokens (9 with `llama3.2:3b`),
and those tokens are not counted: a cold check also sends each distinct base
prompt once untagged (unmeasured, 1 output token), and input and total $/M
use those untagged prompt token counts. The block lines show the tag tokens
left out (`110 in (+36 tag, not counted)`), and the record keeps them
(`blocks[].tag_input_tokens`, `workload.prompt_nonce.untagged_prompt_tokens`). `--warm-cache` resends identical prompts
on purpose, to measure cache-friendly traffic. The header prints which mode
ran, and the mode is part of the workload: a cold and a warm check are never
compared (NO WINNER, "the prompt cache mode changed"). Checks from 0.4.0 and
earlier count as warm, because that is what they sent.

What it looked like on a MacBook (Apple M3 Pro, Ollama 0.32.5,
`llama3.2:3b`, GPU rate ASSUMED at $1.50/hr). With the default workload
(short built-in prompts, 64 output tokens) the difference is small and not
stable: one pair of runs gave $8.55/M output tokens cold (95% CI $8.40 to
$8.70) vs $8.61/M warm (95% CI $8.30 to $8.92), another gave $9.09/M cold
($8.84 to $9.35) vs $8.39/M warm ($8.07 to $8.71). (Those runs predate netting out
the tag, which then added 10 to 12 prompt tokens per request and made cold
input $/M look cheaper; output $/M, shown here, was not affected.) With one long prompt (~2,100 tokens, 2 requests
per block, 8 output tokens, `--metric input`) the cache was obvious:

```text
cold   block 2/3: 4226 in / 8 out tokens in 6.04s -> $0.5955/M input tokens
cold   block 3/3: 4226 in / 8 out tokens in 5.98s -> $0.5900/M input tokens
warm   block 1/3: 4204 in / 4 out tokens in 3.05s -> $0.3024/M input tokens
warm   block 2/3: 4204 in / 4 out tokens in 0.22s -> $0.0213/M input tokens
warm   block 3/3: 4204 in / 4 out tokens in 0.23s -> $0.0230/M input tokens
```

Ollama reported the full prompt token count even when it served the prompt
from cache, so once the prefix was cached the warm run looked about 25x
cheaper per input token. Use cold (the default) unless your production
traffic really repeats prefixes.

`cost` and `measure` tag their prompts the same way by default and take
`--warm-cache` too; `measure` records the mode in its JSON and `throttle
compare` does not rank a cold file against a warm one. `smoke`, `benchmark`
and `golden` still resend their fixed prompt set, because the recorded
workload sha256 (and a golden decision's comparability) is defined over those
exact prompts; declare what the server does with `--cache-policy` (`warm` if
prefix caching is on, `disabled` only if it is off). `check` keeps recording
the same `prompts_sha256` of the prompt file, so it still matches the hash
those reports record.

### Share your results

```sh
throttle check ... --share             # measure, then print a shareable summary
throttle check --history --share       # share the latest recorded check, no traffic
throttle check --share-id CHECK_ID     # share a chosen check, no traffic
```

`--share` prints a markdown summary: engine, model, GPU (from `--config
gpu=...`), the GPU hourly rate (ASSUMED), the workload and cache mode, $/M
before and after with their CIs, the verdict and noise-floor status, the
Throttle version, and what changed in the config. It leaves out the endpoint
URL, hostnames, IPs, local paths and keys, and masks config values that look
like secrets (`sk-...`, `hf_...`, long random tokens). It also prints a
link to a pre-filled GitHub issue (the "Share your results" form). Nothing is
uploaded: you open the link, read it, and submit it yourself. Links are kept
under 7,000 characters; a longer summary is cut with a note, and the full
text is in your terminal to paste.

## Run it in CI

`throttle check --fail-if-costlier PCT` exits 4 on a calibrated MORE
EXPENSIVE and 5 on NOT CALIBRATED, so it can gate a serving-config change.
See [docs/github-action.md](docs/github-action.md) for a GitHub Actions
setup.

## Caching proxy

`throttle proxy` is an OpenAI-compatible server (`/v1/chat/completions`,
`/health`). Change your client's base URL to point at it and nothing else
changes. The cache is off unless you pass `--enable-cache`. Lookups go
through these tiers, in this order:

1. **Exact match:** the same messages under the same model and sampling parameters.
2. **Lexical match:** Jaccard token overlap of at least 0.85. On whenever the cache is on.
3. **Semantic match (opt-in):** `--enable-embeddings` (needs the
   `embeddings` extra) uses MiniLM embeddings via ONNX Runtime with a cosine
   threshold of 0.95. It catches some reworded prompts, not every paraphrase:
   in our local test, "What are three tips for writing a good README?" scored
   0.9494 and missed.

**The semantic tier can return an answer to a different question.** Cosine
similarity from this model encodes topic, not polarity: "Is it safe to use
eval in Python?" and "Is it dangerous to use eval in Python?" scored
**0.9874**, well above the 0.95 threshold
([`data/negation_pairs.json`](data/negation_pairs.json)). A guard rejects
embedding matches whose meaning flips on known negations, antonyms or version
conflicts, but it cannot catch every opposite phrasing. Turn the semantic
tier on only for traffic where a wrong cached answer is acceptable.

A cache hit is always under the same model and identical sampling
parameters. Savings depend entirely on how often your traffic repeats itself.
The proxy has been verified against Ollama. vLLM, SGLang and LMDeploy are
expected to work but have not yet been verified through the proxy on a GPU.
See [Proxy mode](#proxy-mode) below and [docs/PROXY_DEMO.md](docs/PROXY_DEMO.md).

## Agent session profiler

Agents don't send one request. They send a loop of calls in which the prompt
keeps growing. Start the proxy with `--enable-session-tracking` and Throttle
records every turn's timing and token counts to `~/.throttle/sessions.db`.
`throttle sessions` then shows:

- **Wall-clock breakdown:** time spent generating versus time the model sat
  idle waiting for the client, such as tool calls.
- **Redundant prefill:** how much of each prompt repeated the previous turn,
  applied to the prompt tokens the backend reports. This is an estimate, and
  the output labels it as one.
- **Findings:** rule-based suggestions naming the real backend flags to check,
  for example `--enable-prefix-caching` on vLLM.

Privacy: prompt and completion text are never stored, only content hashes,
timings and token counts. Session IDs contain a short one-way hash of the
client address, never the raw IP. Known limits: TTFT shows as `-` because
the proxy buffers backend responses, and findings have no minimum sample
size yet, so a short session can still produce a `[HIGH]` finding.

## Built to be believed

- `throttle plan` shows the destination, request count, token ceiling, time
  limit and cost model **before any traffic is sent**.
- Benchmark-family outputs (demo, smoke, benchmark, golden) label their
  evidence kind: `[SIMULATED]`, smoke (`NON-DECISION-GRADE`), exploratory
  sweep, or golden `decision_eligible`.
- A failed, truncated or malformed response invalidates its block. Throttle
  never reports an optimum it didn't test.
- `throttle check` calls nothing CHEAPER or MORE EXPENSIVE unless the
  measured change beats a run-to-run noise bound estimated from at least 3
  recent repeat checks (a t bound, not the range of one pair) and the 95%
  intervals are disjoint. A change in the ASSUMED GPU rate alone is never a
  verdict. In a simulation with identical true cost and 5% run-to-run noise,
  it declared a false winner in about 3-4% of comparisons. A monthly figure appears only when you pass
  `--monthly-tokens`, and is labelled PROJECTED (MEASURED $/M x your ASSUMED
  volume); a monthly *change* appears only with a calibrated verdict.

Throttle complements vLLM's
[auto_tune](https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md):
auto_tune searches for a candidate config, and Throttle's `golden` protocol
checks whether that candidate actually beat your baseline under controlled,
counterbalanced conditions.

## Choose the right path first

Use a sweep to learn the shape of one server, and use the golden protocol to
make a configuration decision. They answer different questions:

| Goal | Command | Can reach `decision_eligible: true`? |
| --- | --- | --- |
| See the cost model with no hardware | `throttle demo` | No (simulated) |
| Price a live endpoint | `throttle cost` / `throttle watch` | No |
| Re-check $/M after each config change | `throttle check` | No (a calibrated CHEAPER / MORE EXPENSIVE is a guard rail, not a controlled experiment) |
| Cache production traffic / profile agents | `throttle proxy` / `throttle sessions` | Not applicable |
| Check connectivity and response validity | `throttle smoke` | No |
| Explore concurrency or request-rate levels | `throttle benchmark --concurrency 1 2 4 8 ...` | No |
| Generate one safety-audited candidate test value | `throttle experimental-tuning ...` | No |
| Decide between one controlled baseline and candidate | `throttle golden ...` | Yes, if every protocol and evidence gate passes |

A concurrency sweep is intentionally descriptive. It is useful for finding a
region worth testing, but its load levels run in condition-major order and do
not counterbalance time drift. Do not spend money on a sweep expecting its
single-run report to become decision-eligible. Use `throttle golden --help`
when the question is whether one verified server configuration beat another.

## Installation

Throttle requires Python 3.11+. The package on PyPI is `throttle-pro`:

```sh
pipx install throttle-pro
throttle --version
```

### Install from source

```sh
git clone https://github.com/KushagraKanaujia/throttle.git
cd throttle
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
throttle --version
```

Optional extras: `python -m pip install -e '.[embeddings]'` for the proxy's
semantic cache tier, and `python -m pip install pyyaml` for the config file
below.

**vLLM with Prometheus metrics exposed** (default port 8000): read live $/M
tokens without sending any requests:

```sh
throttle watch --gpu-rate-per-hour 1.50
```

This reads `/metrics` only. Without a vLLM server on port 8000 it exits with
an error that says it cannot reach the metrics endpoint.

### Configuration File (Optional)

Throttle supports loading default values from `~/.throttle/config.yaml` to avoid repeating CLI flags. All config values are optional, and CLI flags always override config file settings.

**Setup:**
```sh
# Install PyYAML (optional dependency), inside the clone's virtualenv
python -m pip install pyyaml

# Create config directory and copy the example (-n: never overwrite an existing config)
mkdir -p ~/.throttle
cp -n .throttle.yaml.example ~/.throttle/config.yaml
```

Then edit `~/.throttle/config.yaml` and uncomment the defaults you want (every
line in the example starts commented out).

**Example config:**
```yaml
# Endpoint defaults
model: "meta-llama/Llama-2-7b-chat-hf"
url: "http://localhost:8000/v1"
api-key-env: "OPENAI_API_KEY"

# Workload defaults
max-tokens: 128
concurrency: [1, 2, 4, 8]

# Proxy defaults
port: 8080
enable-cache: true
```

See [.throttle.yaml.example](.throttle.yaml.example) for all available options. If PyYAML is not installed, Throttle runs normally without config file support.

### Quick Start (Local Testing)

The fastest way to try Throttle is against a local Ollama server:

1. **Install Ollama** from [ollama.com](https://ollama.com/download)

2. **Pull and start a small model:**
   ```sh
   ollama pull llama3.2:3b
   ollama serve  # if not already running
   ```

3. **Run a smoke test:**
   ```sh
   # No key needed: for a localhost URL with OPENAI_API_KEY unset and no
   # --api-key-env given, throttle sends no Authorization header and says so.
   # $1.50/hr is an ASSUMED rate (a laptop has no GPU bill); use your own.
   throttle smoke \
     --model llama3.2:3b \
     --url http://localhost:11434/v1 \
     --cost-model dedicated-hourly --gpus 1 --total-hourly-price 1.50 \
     --output smoke.json
   ```

4. **Test the cache feature:**
   ```sh
   throttle smoke \
     --model llama3.2:3b \
     --url http://localhost:11434/v1 \
     --cost-model dedicated-hourly --gpus 1 --total-hourly-price 1.50 \
     --enable-cache \
     --output smoke-with-cache.json
   ```

The smoke run sends 27 requests total (24 measured + 3 warm-ups) and stops at a 120-second ceiling; `throttle plan` with the same options shows the estimated cost upper bound ($0.05 at $1.50/hr) before any traffic. With `--cost-model unknown --allow-unknown-cost` it runs without a price but ends with no dollar figure. With `--enable-cache`, repeated prompts are answered from Throttle's in-process cache. Cache hits are reported separately and excluded from latency percentiles (see [Similarity cache](#similarity-cache)).

### Real staging endpoint: plan, then smoke

This is the exact successful flow used against a real Qwen/vLLM GPU endpoint,
with the private hostname and credential replaced. The `$0.53` rate is only an
example—replace it with the operator's actual whole-instance hourly price.

```bash
# Zero traffic; the key does not need to exist yet.
throttle plan \
  --run-mode smoke \
  --model Qwen/Qwen3-8B \
  --url https://YOUR_APPROVED_STAGING_HOST/v1 \
  --api-key-env VLLM_API_KEY \
  --cost-model dedicated-hourly \
  --gpus 1 \
  --total-hourly-price 0.53

# Populate the key without putting its value in shell history.
read -rsp "Endpoint API key: " VLLM_API_KEY
export VLLM_API_KEY
printf '\n'

throttle smoke \
  --model Qwen/Qwen3-8B \
  --url https://YOUR_APPROVED_STAGING_HOST/v1 \
  --api-key-env VLLM_API_KEY \
  --cost-model dedicated-hourly \
  --gpus 1 \
  --total-hourly-price 0.53 \
  --output smoke.json

unset VLLM_API_KEY
```

Smoke defaults to a 120-second whole-run ceiling; benchmark remains 900
seconds. An explicit `--max-elapsed-seconds` always overrides the mode default.

`plan` does not read `VLLM_API_KEY`, resolve DNS, construct an HTTP client, or
send traffic. Review the destination, request/token/time limits, cost model,
and privacy warning before proceeding. With unknown billing it deliberately
blocks traffic until the operator explicitly acknowledges that the spend
calculation is unavailable.

GuideLLM is an optional, out-of-process cross-check backend. The pinned release
is exactly 0.7.3 and its official Python support is 3.10–3.13, so use a 3.13
environment for that extra. Throttle enables GuideLLM traffic only on POSIX
platforms (Linux/macOS), where it can terminate the entire isolated subprocess
group; Windows fails closed before traffic:

```sh
python3.13 -m venv .guidellm-venv
. .guidellm-venv/bin/activate
python -m pip install -e '.[guidellm]'
guidellm --version
```

## Start with a zero-traffic plan

`plan` does not read the API-key environment variable, resolve DNS, construct
an HTTP client, or invoke GuideLLM.

```sh
throttle plan \
  --model Qwen/Qwen3-8B \
  --url https://inference.example/v1 \
  --cost-model dedicated-hourly \
  --gpus 1 \
  --per-gpu-hourly-price 0.50
```

The destination is intentionally visible in the terminal plan. Saved run and
comparison JSON never contain the URL or hostname.

## Smoke mode

```sh
# Bash example: populate the key without placing it in shell history.
read -rsp "Endpoint API key: " VLLM_API_KEY && export VLLM_API_KEY && printf '\n'

throttle smoke \
  --model Qwen/Qwen3-8B \
  --url https://inference.example/v1 \
  --api-key-env VLLM_API_KEY \
  --cost-model dedicated-hourly \
  --gpus 1 \
  --per-gpu-hourly-price 0.50 \
  --output smoke.json
```

The default smoke run sends 24 measured calls and three warm-ups. Its report
uses `mode: smoke`, `decision_eligible: false`, and a short-sample warning. A
descriptive best-tested observation is explicitly marked
`not_applicable_smoke`; it is not a deployment decision.

Plain HTTP is accepted only for exact loopback hosts (`localhost`, `127/8`,
`::1`). Non-loopback traffic requires HTTPS unless
`--allow-insecure-http` is explicitly supplied and recorded. Native requests
use `trust_env=False` and do not follow redirects, so inherited proxy variables
cannot silently receive a bearer key.

## Sustained benchmark evidence

`throttle benchmark --concurrency 1 2 4 8 ...` collects strong repeated-block
evidence at each load, but the sweep itself is exploratory and cannot reach
`decision_eligible: true` because its condition order is not counterbalanced.
Use its results to choose a treatment/load for `throttle golden`, not as the
final configuration decision.

The count-bounded defaults use three blocks of 67 valid requests per condition
(201 measured requests), plus three separate warm-ups. A condition becomes
decision-grade only if:

- at least three blocks are present;
- every measured completion is valid;
- no block is partial or removed;
- every closed-loop block actually reaches its declared concurrency;
- at least 200 valid requests or 60 measured seconds were achieved; and
- no safety limit or cancellation ended the run.

A statistically supported run is still decision-ineligible unless it uses the
strict native streaming path, live-inference evidence, an explicit cache
policy, immutable model and software-environment pins, a supplied accelerator
fingerprint, complete runtime versions, and runtime-verified engine flags.
CUDA keeps the additional immutable container-image, CUDA, and driver
requirements. Those fixed reasons are written under
`decision_ineligible_reasons` instead of being hidden.

Example pinned exploratory sweep (CUDA/vLLM):

```sh
throttle benchmark \
  --model Qwen/Qwen3-8B \
  --url https://inference.example/v1 \
  --api-key-env VLLM_API_KEY \
  --concurrency 1 2 4 8 \
  --blocks 3 \
  --requests-per-block 67 \
  --warmup-requests 3 \
  --max-tokens 128 \
  --cache-policy disabled \
  --model-revision 0123456789abcdef0123456789abcdef01234567 \
  --accelerator-backend cuda \
  --image-digest 'registry.example/vllm@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' \
  --gpu 'NVIDIA A100 80GB PCIe' \
  --gpu-fingerprint 'NVIDIA-A100-80GB-driver550.127.05-cuda12.8' \
  --cuda-version 12.8 \
  --driver-version 550.127.05 \
  --server-name vllm \
  --server-version 0.16.0 \
  --engine-flag max_num_seqs=1 \
  --engine-flag enable_chunked_prefill=true \
  --engine-flags-provenance runtime_verified \
  --evidence-source live_inference \
  --cost-model dedicated-hourly \
  --gpus 1 \
  --total-hourly-price 1.39 \
  --output exploratory-sweep.json
```

This exploratory sweep is `decision_eligible: false` (non-counterbalanced condition order). Use `throttle golden` for decision-grade comparisons.

**Note:** Metal, ROCm, and CPU backends are supported in code but have zero GPU validation artifacts. The A100 80GB (CUDA) is the only GPU type with a decision-eligible golden protocol result.

Use `--block-seconds 20` instead of `--requests-per-block` for duration-bounded
blocks. The achieved duration—not merely the configured value—controls the
60-second qualification floor.

For constant-rate open-loop load, replace `--concurrency` with:

```sh
--request-rate 1 2 4 8 --open-loop-max-in-flight 32
```

Throttle reports the target offered rate, achieved launch rate, scheduler lag,
and backpressure separately. Every decision-grade block must keep achieved
launch rate within 5% of target and p95 scheduler lag within one target
inter-arrival interval. It never turns backpressure into hidden closed-loop
traffic.

## Completion and metric validity

Native non-streaming responses require a correctly shaped assistant choice,
non-empty output, a non-empty finish reason, and positive integer usage.
Streaming additionally requires an assistant-role event, an output-bearing
delta, a terminal finish reason, final usage, and `[DONE]`. A role-only chunk
does not count as TTFT. Malformed HTTP-200 responses are errors, not successes.

For valid native conditions, the report includes:

- end-to-end latency and streaming TTFT;
- TPOT based on first-to-last output events and completion-token gaps (it is
  unavailable when the server batches all output into one SSE event);
- p50/p90/p95/p99 distributions and 95% intervals;
- request and output-token throughput;
- error rate and per-request SLO goodput; and
- client SSE inter-chunk latency.

Native SSE chunks are not guaranteed to be token boundaries. Throttle therefore
does not relabel chunk gaps as ITL: native `itl_ms` is explicitly unavailable,
while `inter_chunk_latency_ms` is separate. The GuideLLM cross-check exposes its
own ITL aggregate with its source identified, but cannot pass Throttle's strict
completion gate.

Any failed, malformed, oversized, incomplete, or truncated response invalidates
the entire block and condition for decisions. Diagnostic counts remain, while
decision metrics are suppressed.

## Similarity cache

Throttle supports an opt-in in-memory similarity cache for bypassing inference
when prompts are semantically similar. Enable with `--enable-cache`:

```bash
throttle smoke --model llama3.2:3b --url http://localhost:11434 \
  --gpu-hourly-rate 1.50 \
  --enable-cache \
  --cache-ttl-seconds 3600 \
  --cache-max-size 1000 \
  --cache-similarity-threshold 0.85 \
  --output smoke-cache.json
```

The same cache flags work on `throttle benchmark`. Against local Ollama this
smoke run answered 17 of its 27 requests from the cache (`cache_hit_rate`
0.63 in `smoke-cache.json`), because the smoke workload repeats its prompts.

Cache hits are excluded from GPU latency percentiles to preserve decision-grade
measurements: a 1ms cache lookup must not pollute a p95 computed from 50-500ms
GPU requests. Run totals report `cache_enabled`, `cache_hits`, `cache_misses`,
and `cache_hit_rate` separately. The cache uses Jaccard similarity on tokenized
prompts and is thread-safe for concurrent requests.

Cache telemetry flows through experimental tuning validation and saved-run
comparison. This is a local optimization tool; cache behavior does not transfer
to production deployments unless the production server implements equivalent
semantic caching.

## Proxy mode

`throttle proxy` runs a standalone OpenAI-compatible HTTP server that caches
responses for external HTTP clients. Unlike the benchmark cache (which only
accelerates Throttle's own load generator), the proxy serves production
traffic from curl, OpenAI SDKs, and other HTTP clients.

**Quick start** (start Ollama first with `ollama serve` and `ollama pull llama3.2:3b`):

```bash
# Start proxy - backend URL does NOT include /v1 (proxy appends it automatically)
throttle proxy \
  --backend-url http://localhost:11434 \
  --enable-cache \
  --port 8080
```

```bash
# First request - cache miss
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'

# Check cache stats - should show "misses": 1, "hits": 0
curl http://localhost:8080/health

# Second IDENTICAL request - cache hit (MUST match model, max_tokens, messages exactly)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'

# Verify cache hit - should show "hits": 1
curl http://localhost:8080/health
```

**Cache scope**: model, temperature, max_tokens, and all other sampling parameters must match exactly for a cache hit. Changing any parameter creates a different cache scope.

**Matching tiers**: the cache checks three tiers in order: exact match (O(1)), then lexical Jaccard token-overlap (threshold 0.85, always on), then an optional semantic embeddings tier.

By default (lexical-only), **paraphrases will miss** despite identical meaning. For example, `"optimize PostgreSQL queries"` vs `"optimize database queries in PostgreSQL"` has Jaccard similarity 0.60, below the 0.85 threshold, so the second request hits the backend. Exact or near-exact token matches work well without any extra setup.

**Semantic embeddings (opt-in)**: enable with `--enable-embeddings` to catch some paraphrases the lexical tier misses. Uses `sentence-transformers/all-MiniLM-L6-v2` via ONNX Runtime, threshold 0.95. Requires the `embeddings` extra, installed inside the clone:
```bash
python -m pip install -e '.[embeddings]'
throttle proxy --backend-url http://localhost:11434 --enable-cache --enable-embeddings --port 8080
```
If `--enable-embeddings` is passed without the extra installed, the proxy starts with embeddings marked `REQUESTED BUT UNAVAILABLE` and falls back to lexical-only matching rather than failing.

**Threshold behavior**: cosine similarity from this model encodes topic, not polarity. At threshold 0.95, `"Is it safe to use eval in Python?"` vs `"Is it dangerous to use eval in Python?"` scores 0.9874 ([`data/negation_pairs.json`](data/negation_pairs.json)), above the threshold on similarity alone. This is a structural property of the embedding model, not something a higher threshold fixes, so the cache runs an explicit negation/antonym/version-conflict guard before accepting an embeddings-tier hit and skips the match if one is detected. The guard only knows the patterns it lists; an opposite question phrased another way can still be served the wrong cached answer, which is why this tier is opt-in.

For detailed configuration, streaming behavior, error handling, and production deployment
considerations, see [docs/PROXY_DEMO.md](docs/PROXY_DEMO.md).

## Boundary and uncertainty rules

The report field is `best_tested_concurrency` or `best_tested_request_rate`.
It never says an optimum was found. If the highest tested load wins,
`boundary_reached` is true and the decision is inconclusive: expand the tested
range safely. Overlapping block-level 95% throughput intervals also produce an
explicit inconclusive state. Request-level bootstrap intervals are bounded,
deterministic diagnostics;
repeated blocks are the independent units for comparisons.
Condition reports retain pooled `completion_tokens / measured_wall` throughput
as a descriptive utilization value, but selection and its CI consistently use
the arithmetic mean of repeated-block throughputs. This avoids mixing two
estimands when count-bounded blocks have unequal durations.
Across tested conditions, mean completion tokens per valid response must remain
within 5%; otherwise the best-tested observation is explicitly inconclusive.
For duration-bounded levels this comparison is normalized per response rather
than comparing unequal request totals.
Native load levels currently run condition-major, so a multi-load best-tested
value is deliberately descriptive/inconclusive even when its repeated-block
intervals do not overlap: those blocks do not counterbalance time/order drift
between conditions. Decision claims come from saved-run comparison or the
six-position counterbalanced golden protocol; a future native scheduler can
remove this gate by using a recorded block-major counterbalanced order and the
same per-block prompt schedule across conditions.
Declared E2E/TTFT SLOs are gated with a Student-t interval over per-block p95s;
the request-bootstrap p95 interval remains diagnostic and cannot support a
decision by itself.

## Cost models

Choose exactly one model; Throttle never combines them:

- `unknown`: no cost calculation. Traffic requires explicit
  `--allow-unknown-cost`, and the spend ceiling is reported as unenforceable.
- `dedicated-hourly`: total hourly price, or per-GPU price multiplied exactly
  once by `--gpus`. Client measured wall time is the accounting basis.
- `serverless-active-seconds`: `--active-second-price` and an explicit
  `--max-active-workers` billing ceiling. Exact final cost requires provider
  `--billed-active-seconds`; queue and cold-start time are not silently treated
  as GPU wall time.
- `user-supplied`: `--user-supplied-total` is attributed to the whole run and
  labeled as supplied, not observed.

Saved runs with different cost-model tags can compare performance, but their
cost delta is unavailable.

## Hard safety limits and cancellation

Every traffic run has hard ceilings for:

- inference requests, including warm-ups;
- output tokens per request and total reserved requested output tokens;
- global monotonic elapsed time;
- errors and concurrency/in-flight work;
- response bytes while reading/streaming; and
- estimated spend when the selected billing model makes it enforceable.

See `throttle smoke --help` or `throttle benchmark --help` for the flags. The
engine reserves request/token budget before launch and stops new scheduling at
a limit. In-flight work is cancelled where the client controls it. SIGINT writes
an atomic, mode-0600, sanitized partial JSON report and exits 130.

## Pinned GuideLLM backend

Throttle pins [GuideLLM 0.7.3](https://github.com/vllm-project/guidellm/releases/tag/v0.7.3)
and invokes `guidellm run` without a shell. It verifies the exact version first,
removes inherited proxy and GuideLLM variables, passes the API key only through
`GUIDELLM__SPEC__BACKEND__API_KEY`, disables redirects, enables TLS verification,
disables GuideLLM's unconstrained backend health probe, forces tokenizer loading
from the local cache with Hugging Face offline mode, removes inherited Hugging
Face credential variables and ambient TLS trust/key-log overrides, captures no
child console output, parses a versioned numeric allow-list, and deletes its
mode-0700 temporary directory.

GuideLLM 0.7.3 cannot prove `finish_reason` or response token provenance and
does not enforce response-byte size. Its aggregate may also synthesize missing
usage from the requested token shape. Therefore this backend is deliberately
cross-check-only, requires an explicit acknowledgement, uses GuideLLM
`synthetic_text` rather than claiming parity with supplied JSONL, and always
sets `decision_eligible: false`:

```sh
throttle benchmark \
  --backend guidellm \
  --guidellm-prompt-tokens 256 \
  --allow-guidellm-validation-gaps \
  ...
```

The adapter accepts only endpoint forms whose route is exactly equivalent to
GuideLLM's `/v1/chat/completions` route (root, `/v1`, or that full path), and
fails closed on custom base paths. If a child is killed or its report cannot be
validated, exact traffic totals become unavailable and conservative bounds are
persisted; declared concurrency is never relabeled as an observed peak.

The official [GuideLLM benchmark guide](https://github.com/vllm-project/guidellm/blob/v0.7.3/docs/getting-started/benchmark.md)
documents its concurrent and constant profiles. vLLM itself recommends
GuideLLM for production-oriented server benchmarking in its
[benchmarking guide](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md).

## Compare saved runs

No endpoint or key is needed:

```sh
throttle compare baseline.json candidate.json --output comparison.json
```

Comparison fails closed on legacy/unknown schemas, smoke or partial artifacts,
missing manifests, invalid blocks, fewer than three blocks, insufficient
requests/duration, malformed responses, mismatched workloads/configuration,
non-disjoint warm-ups, inconsistent run totals/timestamps/cost math, and
completion-token totals outside 5%. Confidence intervals use matched repeated
blocks. Engine-flag differences are listed by safe name only.

A `max_num_seqs=256` versus `2048` change at concurrency 8 is explicitly
unexercised and receives no attribution. A chunked-prefill-only difference also
receives none: current vLLM V1 enables chunked prefill by default whenever
possible, as documented in the
[vLLM optimization guide](https://docs.vllm.ai/en/stable/configuration/optimization/).

## Golden live protocol

The controlled treatment implemented by the protocol is any two distinct,
canonical positive integer values of `max_num_seqs` (ASCII decimal digits,
without a sign, whitespace, or leading zero, in the range 1 through
2,147,483,647). Each golden position contains exactly one closed-loop
condition. Its client concurrency must be at
least the larger treatment value; use the analyzer's original offered
concurrency when it is higher. For example, an `8` versus `10` treatment can
run at `--concurrency 16`. If `--concurrency` is omitted, Golden defaults to the
larger treatment value. Lower exploratory load levels belong in separate
reports. Everything else must remain pinned.

Reaching the declared client concurrency proves that Throttle offered enough
simultaneous demand to exercise the configured limit. It does **not** prove
that the server scheduler held that many sequences simultaneously or that the
server was saturated.

`throttle golden` owns the complete B1/C1/B2/C2/B3/C3 measurement session and
the final validation. It does **not** change server configuration: before each
position it pauses, tells the operator which verified configuration is needed,
and requires an exact confirmation. The operator changes/restarts the staging
server in a separate terminal, verifies the effective runtime flag, and then
lets Throttle continue.

First inspect the complete six-run request/token/time/spend envelope without a
key, DNS lookup, HTTP client, output directory, or traffic:

```sh
throttle golden --dry-run \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --url https://inference.example/v1 \
  --api-key-env VLLM_API_KEY \
  --baseline-config max_num_seqs=1 \
  --candidate-config max_num_seqs=8 \
  --concurrency 8 \
  --cost-model dedicated-hourly \
  --gpus 1 \
  --total-hourly-price 1.39 \
  --cache-policy disabled \
  --model-revision 0123456789abcdef0123456789abcdef01234567 \
  --image-digest 'registry.example/vllm@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' \
  --gpu 'NVIDIA A100 80GB PCIe' \
  --gpu-fingerprint 'NVIDIA-A100-80GB-driver550.127.05-cuda12.8' \
  --cuda-version 12.8 \
  --driver-version 550.127.05 \
  --server-name vllm \
  --server-version 0.16.0 \
  --engine-flag enable_chunked_prefill=true \
  --engine-flags-provenance runtime_verified \
  --evidence-source live_inference \
  --output-dir golden-run-001
```

This example mirrors the treatment, GPU, and engine of the validated golden run
in `validation/golden-live-20260817/` (placeholders replace the pinned
revision/digest).

Add `--p95-slo-ms` / `--ttft-slo-ms` with the operator's actual thresholds if
latency matters. A
throughput-only golden decision is permitted when no latency SLO is declared,
but the artifact says so explicitly and makes no latency claim.

Remove `--dry-run` after reviewing the plan and setting the key locally. With
the default 3 × 67 measured requests and three warm-ups, the command plans 204
calls per position, 1,224 calls total, and 156,672 requested output tokens. The
cumulative request, token, elapsed, error, and spend limits apply to the whole
session; per-request size/token limits and the in-flight ceiling apply at every
position. The default elapsed ceiling is 5,400 seconds. Choose a new output
directory for every attempt—Throttle refuses to overwrite prior evidence.

The v0.3 live orchestrator is deliberately count-bounded: it requires at least
three blocks and 200 measured requests per position (the default 3 × 67 gives
201). Duration-bounded golden evidence can still be produced with six manual
`benchmark --block-seconds ...` reports and validated offline with `compare`.
For session billing, use dedicated hourly, acknowledged unknown billing, or
serverless rate limits without a pre-filled `--billed-active-seconds` total.
`user-supplied` run totals and pre-filled serverless billed seconds are rejected
because one value cannot truthfully describe all six position reports; attach
the final provider total to the external audit record after the session.

The command runs six non-overlapping native benchmarks in this exact order:

```text
B1 → C1 → B2, then C2 → B3 → C3
```

Each position itself uses at least three blocks and meets the 200-valid-request
floor. It writes `B1.json` through `C3.json`, then validates and writes
`golden.json` automatically. The older offline form remains available for
already-saved artifacts:

```sh
throttle compare B1.json C1.json B2.json C2.json B3.json C3.json \
  --output golden.json
```

The gate requires live inference, exact ordering/non-overlap, one hashed
accelerator fingerprint, a pinned software environment and full model commit,
runtime-verified engine flags, the same workload/SLO/cache policy, zero invalid
responses, and one positive, distinct `max_num_seqs` pair with the same
declared client load reached in every position. CUDA positions additionally
require a pinned image digest and CUDA/driver versions. It evaluates
order-balanced phase contrasts and retains the 5% completion-token guard across
every position; a declared SLO must also hold in all six runs. See
[the full protocol](docs/GOLDEN_PROTOCOL.md).

Only when that complete gate passes and the order-balanced 95% interval
excludes zero, the golden artifact and terminal add one clearly labelled,
workload-scoped recommendation line naming the winning configuration and its
candidate-relative throughput delta. The aggregate's sanitized `treatment`
block records the inferred baseline value, candidate value, and common client
concurrency even when the statistical result is inconclusive. The summary says
which declared E2E/TTFT SLO gates passed; it never upgrades SLO compliance into
a “latency parity” or server-saturation claim. An ineligible or inconclusive run
has `decision_summary: null` and prints no recommendation.

Throttle never provisions or reconfigures the accelerator/server. In this repository no
server credentials or endpoint identifiers are retained. A sanitized completed
six-position 1-versus-8 run is included under
[`validation/golden-live-20260817`](validation/golden-live-20260817) as protocol
evidence. It measures only its pinned model, accelerator, workload, and test window; it
is not a universal performance, savings, or production recommendation.



## Pre-flight bottleneck diagnosis
`throttle diagnose` is a lightweight, non-destructive probe that runs prior to any formal benchmarking or `golden` protocol. It classifies the dominant serving bottleneck regime so you do not waste resources running sweeps on config dimensions that do not address your actual constraint.

```bash
throttle diagnose \
  --model Qwen/Qwen3-8B \
  --url https://inference.example/v1 \
  --api-key-env VLLM_API_KEY \
  --concurrency 1 4 8 \
  --requests-per-block 20 \
  --cost-model dedicated-hourly \
  --gpus 1 \
  --total-hourly-price 0.50 \
  --output diagnose.json
```

Like `smoke`, `diagnose` refuses to send traffic without a price; use
`--cost-model unknown --allow-unknown-cost` to run it with no dollar figure.

The command runs 1 block of 20 requests per concurrency level with 3 warm-ups (maximum 200 total requests) under a strict 60-second execution ceiling.

Based on client-side timing heuristics, it classifies the server into one of five regimes:
- **dispatch-bound** (launch overhead): CPU overhead dominates; recommended tuning: `cuda_graph_capture`, `batch_size`.
- **orchestration-bound** (host jitter): high inter-request latency; recommended tuning: `scheduler_config`, `request_batching_strategy`, `python_vs_cpp_runtime`.
- **compute-bound** (GPU arithmetic): throughput scales linearly; recommended tuning: `max_num_seqs`, `max_num_batched_tokens`, quantization.
- **memory-bound** (VRAM limits): TTFT degrades sharply; recommended tuning: `kv_cache_block_size`, `prefix_caching`, `max_model_len`.
- **mixed**: multiple competing bottlenecks; run exploratory sweeps to isolate.

If the error rate exceeds 50%, or samples are insufficient, it returns `classification: inconclusive`. Both `inconclusive` and `mixed` exit with code `3`. It always sets `decision_eligible: false` and cannot be used with `throttle compare`.

## Experimental suggestion-only tuning

`throttle experimental-tuning` is a separate, explicitly opt-in path for one
vLLM deployment. Existing `plan`, `smoke`, `benchmark`, `compare`, and `golden`
behavior is unchanged. The command runs one ordinary native smoke workload
while reading a separately supplied Prometheus URL, then passes the bounded
metrics window and exploratory analysis through the independent safety
boundary. It never changes the server. Throttle has no automatic telemetry or
phone-home behavior; only this command reads the exporter, and only after the
operator supplies `--metrics-url`.

```sh
throttle experimental-tuning \
  --model Qwen/Qwen3-8B \
  --url https://inference.example/v1 \
  --metrics-url https://inference.example/metrics \
  --api-key-env VLLM_API_KEY \
  --concurrency 16 \
  --engine-flag max_num_seqs=8 \
  --engine-flag max_num_batched_tokens=2048 \
  --engine-flags-provenance runtime_verified \
  --attest-same-deployment-exclusive-metrics \
  --cost-model dedicated-hourly \
  --gpus 1 \
  --total-hourly-price 0.50 \
  --output experimental-smoke.json \
  --experimental-output experimental-tuning.json
```

The command requires exactly one closed-loop concurrency and the two effective
runtime flags shown above. It defaults to 201 measured requests plus three
warm-ups so a normal run can clear the analyzer's minimum evidence count, but
passing that floor does not make the evidence decision-grade. The default
900-second limit applies to the traffic run; bounded exporter scrapes, safety
analysis, and artifact writes add processing overhead outside that limit. Use
`--attest-same-deployment-exclusive-metrics` only when the exporter belongs to
the inference deployment under test and no unrelated inference traffic reaches
it during the sampled window. If either fact is unknown, omit the flag; the
result will fail closed as insufficient evidence instead of guessing. Exporter
metrics are process-wide, and neither part of the attestation is independently
proven.

The metrics URL is never inferred from the inference URL. It must be explicit;
the collector sends no authorization header or cookies, ignores ambient
proxies, follows no redirects, permits plaintext only on exact loopback hosts,
and retains neither the URL nor raw metric labels/body. This means an exporter
that requires credentials is intentionally unsupported by this experimental
path.

Two mode-0600 artifacts remain separate:

- `experimental-smoke.json` is an ordinary schema-2.0 `mode: smoke` report. Its
  `decision_eligible` and condition `decision_grade` fields are false, and it
  contains no experimental tuning fields.
- `experimental-tuning.json` is a fixed experimental envelope containing the
  detached safety-validated projection and a canonical SHA-256 binding to the
  ordinary report. Inside that projection, decision eligibility, auto-apply,
  configuration changes, Golden execution, Golden eligibility, and every
  gate-bypass field are hard-locked false.

Both parent directories must already exist and both output files must be new;
the experimental command refuses to overwrite prior evidence. The report hash
is an equality/linkability check, not a signature. It covers only the already
sanitized ordinary artifact and does not add raw prompts, responses, endpoint
details, or exporter labels.

If collection or validation fails before a complete ordinary report exists,
Throttle writes only a fixed sanitized failure artifact and no experimental
envelope. It deliberately does not preserve unvalidated stage-owned partial
fields.

An emitted value is labelled as a candidate for another test, never a
recommendation or guaranteed outcome. It changes only `max_num_seqs` by one
bounded 25% search step. Before any configuration decision, the operator must
run the separate six-position counterbalanced Golden protocol at the recorded
offered concurrency. Reaching that client concurrency proves sufficient
offered demand; it does not prove direct server-scheduler saturation.

The deterministic evidence under
[`validation/experimental-tuning-vllm-docs`](validation/experimental-tuning-vllm-docs)
checks the full offline request/exporter/collector/analyzer/safety/serialization
path against metric names and labels pinned to an official vLLM release. It is
software compatibility evidence, not a live GPU benchmark, measured savings,
or proof that the suggested value improves a deployment.

## Report privacy and exit codes

Reports contain hashes and aggregate numeric evidence, not endpoint URLs,
hostnames, keys, authorization headers, prompts, responses, raw exception text,
or GuideLLM raw output. Engine flag names/values are validated before they can
enter a manifest; accelerator fingerprints are stored only as SHA-256.
(`throttle check` history is different: it is a local file that keeps each
check's endpoint URL so it can compare checks of the same endpoint. It holds
no keys, prompts or responses.)

These exit codes are for plan/smoke/benchmark/compare/golden. `throttle check`
has its own (0, 1, 2, 4, 5), listed under
[`throttle check`: verdicts and exit codes](#throttle-check-verdicts-and-exit-codes).

- `0`: complete smoke, or a supported benchmark/comparison result.
- `1`: stopped/invalid/operational failure; a sanitized artifact is written
  whenever execution started.
- `2`: CLI usage error or incompatible saved reports.
- `3`: valid but statistically/qualification-inconclusive benchmark or compare.
- `130`: user cancellation with sanitized partial report.

For `experimental-tuning`, `0` means a safety-audited candidate test value was
available; `3` means the run was valid but evidence was insufficient or no
clear signal existed. Neither exit code means a configuration decision. Stage
failure returns `1`, usage/preflight failure returns `2`, and cancellation
returns `130`.

## Test

```sh
.venv/bin/python -m pytest -q
```

Every test should pass. Some tests skip when their prerequisites are missing:
the live-proxy integration tests need Ollama at `localhost:11434` with
`llama3.2:1b` and `llama3.2:3b` pulled, and the embedding tests need the
`embeddings` extra plus a downloadable or cached
`sentence-transformers/all-MiniLM-L6-v2`. In CI without network access, set
`HF_HUB_OFFLINE=1` so a missing model skips quickly instead of retrying the
download.

The suite blocks non-loopback DNS/socket use via an offline guard in CI. It covers modes, URL/proxy safety, response validation, streaming termination, hard stops, partial reports, cost separation, open/closed-loop scheduling, confidence and boundary logic, manifest tampering, saved comparisons, the GuideLLM subprocess boundary, the six-run golden gate, the caching proxy, agent session tracking, and the opt-in collector/analyzer/safety chain. Default commands are tested with collector bombs so they cannot accidentally start experimental metric collection.

## Explicitly deferred

Throttle does not build or perform automatic vLLM/TensorRT-LLM
reconfiguration, GPU/pod provisioning, replica autoscaling, GPU/instance
selection, spot orchestration, async job queues, non-OpenAI backends,
distributed multi-host tests, accounts/teams, a hosted dashboard, remote
telemetry, production-log load discovery, monthly-savings claims (the only
monthly figure is `throttle check --monthly-tokens`, labelled PROJECTED), or a
polished UI. The proxy's response cache is in-memory and per-process.
Throttle persists only local files: the `throttle check` history
(`~/.throttle/checks/`, or `--history-dir` / `$THROTTLE_CHECK_HISTORY_DIR`),
the opt-in session database (`~/.throttle/sessions.db`) and, for decision-eligible golden runs, the local
result store (`~/.throttle/results`, disable with `--no-result-store`). The proxy does not yet stream backend tokens
through: it buffers each response, so it cannot measure TTFT.

Remaining limitations and the current evidence boundary are listed in
[Known gaps](docs/KNOWN_GAPS.md).

## License

Throttle is released under the [MIT License](LICENSE).
