# Throttle Quickstart

This takes you from nothing to a measured dollars-per-million-tokens figure,
and then to a cost check you rerun after every serving-config change, in
about five minutes. You don't need a GPU.

## Prerequisites

- Python 3.11 or later
- Step 1 needs nothing else
- Steps 2 to 4 need [Ollama](https://ollama.com/download) running locally
  with `ollama pull llama3.2:3b`

## Install

```bash
pipx install throttle-pro
throttle --version   # 0.4.1
```

No pipx? `python3 -m pip install --user pipx && python3 -m pipx ensurepath`,
or install into a virtualenv with `pip install throttle-pro`.

Run `throttle` with no arguments at any point to print these steps in the
terminal.

## 1. Run the simulator demo

```bash
throttle demo
```

This command:
- generates a sample workload of 300 requests,
- simulates vLLM-style continuous batching at `max_num_seqs` 128 (baseline)
  and 256 (tuned),
- prints a cost table with confidence intervals and a sensitivity analysis,
- finishes in about a second and makes no network calls.

Output excerpt:

```text
Configuration being compared:
  Parameter: max_num_seqs
  Baseline: 128 concurrent sequences
  Tuned:    256 concurrent sequences
...
Metric                                       Baseline        Tuned        Delta
--------------------------------------------------------------------------------
Wall clock time (seconds)                       77.87        63.14       -14.73
GPU hours                                    0.021630     0.017538    -0.004092
Blended cost ($/M tokens, in + out)              0.08         0.07        -0.02
Input cost ($/M tokens)                          0.35         0.28        -0.07
Output cost ($/M tokens)                         0.11         0.09        -0.02
Total cost ($)                                 0.0324       0.0263      -0.0061

Note: input and output $/M each charge the whole run to one token type,
so they are not additive. This workload has 92,903 input and 304,063
output tokens; the type with fewer tokens shows the higher $/M.
Compare the blended figure with a per-token bill.
...
All values above are [SIMULATED] - they depend entirely on assumed
throughput parameters, not real hardware measurements.
```

Every number in this output is simulated from assumed throughput. It shows
how the cost model works. It is not a measurement.

## 2. Measure real cost per million tokens

```bash
throttle cost \
  --url http://localhost:11434 \
  --model llama3.2:3b \
  --gpu-hourly-rate 1.50 \
  --num-requests 5
```

This sends 5 real requests, measures time and tokens, and converts them to
dollars per million tokens at the hourly rate you supply. A laptop has no
real hourly price, so $1.50 is an assumption. Recorded on a MacBook (Apple
M3 Pro, local Ollama 0.32.5; GPU rate ASSUMED):

```text
Measured Cost [MEASURED time x ASSUMED $1.50/hr]:
  GPU hours: 0.001192
  Total cost: $0.0018
  Blended cost: $2.40 per million tokens (input + output, 745 tokens)
  Input cost: $3.02 per million tokens
  Output cost: $11.77 per million tokens
  Note: input and output $/M each charge the whole run to one token type;
  they are not additive. Compare the blended figure with a per-token bill.
```

The input and output $/M figures each charge the full run cost to that token
type, so don't add them together; the blended line (total cost / all tokens)
is the one to hold next to a per-token bill. The URL works with or without
`/v1`.

## 3. Re-check cost after every config change

`throttle cost` is a one-off number. `throttle check` is the one to rerun: it
measures $/M tokens in repeated blocks with a 95% confidence interval, records
the serving config it can see, saves the result, and compares it with the last
check of the same endpoint. Each check takes about 30 seconds on a laptop
(21 requests).

Record today's config **three times without changing anything**. The
repeats measure how much this machine drifts between runs (the run-to-run
noise):

```bash
for i in 1 2 3; do
  throttle check --url http://localhost:11434 --model llama3.2:3b \
    --gpu-hourly-rate 1.50 --label before
done
```

Then change one server setting (for Ollama, for example, restart it with
`OLLAMA_NUM_PARALLEL=4`) and check again with a new label. `--config` only
records what you changed; Throttle never changes your server:

```bash
throttle check --url http://localhost:11434 --model llama3.2:3b \
  --gpu-hourly-rate 1.50 --label after --config OLLAMA_NUM_PARALLEL=4 \
  --monthly-tokens 500M
```

Recorded on a MacBook (Apple M3 Pro, local Ollama 0.32.5, `llama3.2:3b`;
GPU rate ASSUMED). For this recording Ollama was **not** restarted, so the
server never changed and "no winner" is the right answer. The second and
third `before` checks are NOT CALIBRATED: they have only 1 and 2 earlier
checks to learn the noise from. The second ended with:

```text
  before  $8.64/M output tokens  (95% CI $8.08 to $9.20)
  now     $9.36/M output tokens  (95% CI $8.62 to $10.10)
  change  +$0.7198 per million output tokens (+8.3%)
  noise   run-to-run noise bound: not measured (needs 3 earlier checks of one config and workload within 24 h, not counting this one; or 2 each of the before and after configs)
          baseline and this check (same config): label before, 1 earlier check(s), no repeat yet
  why     run-to-run noise unknown: 0 degree(s) of freedom from repeat checks within 24 h, 2 needed (e.g. 3 checks of the baseline config, not counting this one).
  Verdict: NOT CALIBRATED — run-to-run noise unknown. Two checks at different times can differ from load alone. Run 'throttle check' at least 3 times without changing anything (within 24 h) to measure your noise, or use 'throttle golden' for a counterbalanced decision.
```

The `after` check, on the same unchanged server, measured 11.5% cheaper:

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

That 11.5% is drift, not the config: three checks of the unchanged `before`
config already spread 9.8%. Throttle said NO WINNER and projected no monthly
change. Your numbers will differ.

### How the verdict works

A 95% CI is computed across blocks inside one run, so it cannot see
run-to-run drift (machine load, thermal state, other tenants). CHEAPER or
MORE EXPENSIVE therefore needs all of:

- the same workload options in both checks;
- a **run-to-run noise bound** from recent repeat checks, and a measured
  change larger than it. A same-config group is every check with the same
  endpoint, config fingerprint (model, label, GPU rate, server-reported
  settings, `--config` pairs), workload and Throttle version. From the
  baseline's and this check's groups (not counting this check, and only
  checks from the last 24 hours) Throttle pools the relative SD of their
  $/M, and the bound is `t(0.975, df) x sqrt(2) x SD`. It needs `df >= 2`:
  for example 3 earlier checks of the baseline config;
- non-overlapping 95% CIs.

Otherwise the verdict is NO WINNER, naming each condition that failed. With
too few recent repeats, or a baseline more than 24 hours old, it is
**NOT CALIBRATED**: before, now, both CIs, the change and the reason are
printed, but no winner and no monthly figure. The label is part of the
config, so rerunning under a new label also starts uncalibrated.

The GPU rate is ASSUMED, so it never decides a verdict: when it differs from
the baseline's, only the measured part (the change at the baseline's rate) is
judged, and the rate-driven part is printed separately. When two checks of
one config disagree by more than their CIs, the check prints a hint that your
machine's run-to-run noise is larger than within-run noise.

### History and CI

```bash
throttle check --history
throttle check --url http://localhost:11434 --model llama3.2:3b \
  --gpu-hourly-rate 1.50 --label after --config OLLAMA_NUM_PARALLEL=4 \
  --fail-if-costlier 5
```

The first lists past checks and sends no traffic. The second is the `after`
check as a CI gate; its exit code tells the pipeline what happened:

| Exit code | Meaning |
| --- | --- |
| `0` | CHEAPER, NO WINNER, first check, or NOT CALIBRATED without `--fail-if-costlier` |
| `1` | Measurement failed; nothing recorded |
| `2` | Usage error |
| `4` | Calibrated MORE EXPENSIVE by at least the `--fail-if-costlier` percentage (measured change, at the baseline's GPU rate) |
| `5` | `--fail-if-costlier` given and the change could not be judged: NOT CALIBRATED, or the baseline ran a different workload (e.g. another prompt cache mode) (a warning or a failure, your choice) |

The recorded checks above ran without `--fail-if-costlier`, so they exited
with 0; with it, the NOT CALIBRATED ones would exit with 5. Until an
unchanged config has 3 recent checks, the gate can only return 5, never 4.

`check` uses its own fixed workload, so its $/M is not comparable with the
`throttle cost` figure from step 2. Compare checks with checks.

### Cold vs warm cache

Ollama, vLLM and SGLang can reuse the work for a prompt prefix they have seen
before. So by default every request `check` measures starts with a unique tag
like `[run 482915 req 0012] ` (the header says `cache COLD`), and a repeat run
cannot be served from cache. Add `--warm-cache` to resend identical prompts
when your real traffic repeats prefixes. A cold check and a warm check are
never compared with each other (NO WINNER). `cost` and `measure` tag their
prompts the same way and take `--warm-cache` too.

On the laptop above, short built-in prompts showed a small, unstable
difference (one pair of runs overlapped, another had warm about 8% cheaper,
partly because untagged prompts are about 10 tokens shorter). One
~2,100-token prompt repeated with `--warm-cache` ran about 25x faster once
cached, while Ollama still reported every prompt token, so $/M looked about
25x cheaper than fresh traffic.

After upgrading from 0.4.0, your first check is NO WINNER against the old
ones (they were warm). Record three cold checks of the unchanged config
again to recalibrate.

### Share your results

```bash
throttle check --history --share
```

prints a markdown summary of your latest check (engine, model, GPU rate,
workload, $/M before and after with CIs, verdict, what changed) with no URLs,
hostnames, IPs or keys, plus a link to a pre-filled GitHub issue. Nothing is
uploaded; you read it and submit it yourself. Add `--share` to a measuring
check, or use `--share-id CHECK_ID` for an older one. Add `--config gpu=H100`
(or your GPU) to your checks so the summary can say what hardware it ran on.
We read every one, and they tell us where Throttle helps and where it does not.

## 4. Optional: a caching proxy in front of Ollama

Caching is one cost lever, and it is off by default. It only helps when your
traffic repeats itself. With `--enable-cache`, the proxy's tiers are exact
match and lexical (word-overlap) match.

Terminal 1:

```bash
throttle proxy --backend-url http://localhost:11434 --port 8090 --enable-cache
```

Terminal 2:

```bash
ask() { curl -s -o /dev/null -w "HTTP %{http_code}  %{time_total}s\n" \
  localhost:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"llama3.2:3b\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":60,\"temperature\":0}"; }
ask "Give me three tips for writing a good README."
ask "Give me three tips for writing a good README."
ask "give me three tips for writing a great README"
ask "Give me three tips for writing a good cover letter."
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

The exact repeat is answered from the cache in about 1.5 ms; the unrelated
prompt goes to the model. The reworded prompt shares too few words with the
first one for the lexical tier, so it goes to the model too: 3 backend calls
for 4 requests. Your timings will differ.

Semantic matching (`--enable-embeddings`, needs
`pipx install --force 'throttle-pro[embeddings]'`) also catches some reworded prompts, but it is opt-in for a
reason: opposite questions such as "Is it safe to use eval in Python?" and
"Is it dangerous to use eval in Python?" scored 0.9874 similarity, above the
0.95 threshold. A guard rejects known negation and antonym flips, but a
semantic match can still return an answer to a different question. Turn it
on only for traffic where that is acceptable.

## Next steps

- Profile an agent: `throttle proxy ... --enable-session-tracking`, then
  `throttle sessions` (see the README's "Agent session profiler" section)
- Before sending traffic to a real endpoint, preview it: `throttle plan --help`
- For decision-grade config comparisons, see `throttle golden --help` and
  [docs/GOLDEN_PROTOCOL.md](docs/GOLDEN_PROTOCOL.md)

## Getting help

```bash
throttle --help
throttle demo --help
throttle cost --help
throttle check --help
throttle proxy --help
```

## Troubleshooting

### "Failed to connect to endpoint"

Make sure your inference server is running:

```bash
ollama serve
curl http://localhost:11434/api/tags
```

### The proxy says embeddings are unavailable

The embedding tier needs the `embeddings` extra (`pipx install --force 'throttle-pro[embeddings]'`)
and a one-time download of `sentence-transformers/all-MiniLM-L6-v2`. Without
them, the proxy falls back to exact and lexical matching, so reworded prompts
will miss.

### Simulator costs differ from real measurements

That is expected. The simulator uses assumed throughput values. Use
`throttle cost` for measured numbers.
