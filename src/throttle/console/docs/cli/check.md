# throttle check

*Measure $/M tokens after a serving change and compare with the last check.*

```bash
throttle check --url URL --model MODEL --gpu-hourly-rate USD [options]
```

Run it after every serving change. It measures what the endpoint costs per million tokens, remembers the config it was measured on, and compares with the last check of the same endpoint.

## Common options

| Option | Default | Meaning |
| --- | --- | --- |
| `--url URL` | | OpenAI-compatible base URL, with or without `/v1` |
| `--model MODEL` | the only listed model | Model to request |
| `--gpu-hourly-rate USD` | | What the GPU(s) behind the endpoint cost per hour (ASSUMED) |
| `--label LABEL` | | Short name for this config |
| `--config KEY=VALUE` | | A setting the endpoint can't report (repeatable) |
| `--concurrency N` | 2 | Requests in flight; match production |
| `--blocks N` | 5 (min 3) | Measurement blocks; the 95% CI is across blocks |
| `--requests-per-block N` | 4 | Requests per block |
| `--max-tokens N` | 64 | Max output tokens per request |
| `--prompts FILE` | built-in | JSONL prompt file |
| `--warm-cache` | off | Resend identical prompts (see [cold vs warm](../concepts/cold-vs-warm-cache.md)) |
| `--metric` | output | Headline tokens: `output`, `input` or `total` |
| `--monthly-tokens N` | | Your monthly volume (e.g. `500M`) for a projected monthly figure |

## History, CI and sharing

| Option | Meaning |
| --- | --- |
| `--history` | List past checks and exit (no traffic) |
| `--against CHECK_ID` | Compare with this check instead of the latest for the endpoint |
| `--no-save` | Compare but don't store this check |
| `--fail-if-costlier PCT` | Exit 4 on a calibrated MORE EXPENSIVE ≥ PCT; exit 5 if not judgeable |
| `--json PATH` | Also write the check and comparison as JSON |
| `--share` / `--share-id ID` | Print a sanitized summary and a pre-filled issue link |
| `--history-dir DIR` | Where checks are stored (default `$THROTTLE_CHECK_HISTORY_DIR` or `~/.throttle/checks`) |

## Safety limits

Checked before any traffic is sent: `--max-requests` (10000), `--max-tokens-per-request` (1024), `--max-total-requested-tokens` (2,000,000), `--max-concurrency` (64), `--max-elapsed-seconds` (300), `--max-estimated-spend` ($3.00).

## Examples

```bash
throttle check --url http://localhost:8000 --model my-model \
    --gpu-hourly-rate 2.49 --config max_num_seqs=256 --label before
throttle check --url http://localhost:8000 --model my-model \
    --gpu-hourly-rate 2.49 --config max_num_seqs=512 --label after \
    --monthly-tokens 500M --fail-if-costlier 5
throttle check --history
throttle check --history --share     # share the latest check (no traffic)
throttle check --share-id CHECK_ID   # share a chosen check
```

See [Verdicts](../concepts/verdicts.md) for exit codes and exactly when each verdict is used.

<details><summary>Full `throttle check --help` output (0.7.0)</summary>

```text
usage: throttle check [-h] [--url URL] [--model MODEL] [--api-key-env NAME]
                      [--allow-insecure-http] [--metrics-url METRICS_URL]
                      [--no-metrics] [--gpu-hourly-rate USD] [--gpus N]
                      [--metric {output,input,total}] [--monthly-tokens N]
                      [--label LABEL] [--config KEY=VALUE] [--blocks BLOCKS]
                      [--requests-per-block REQUESTS_PER_BLOCK]
                      [--concurrency CONCURRENCY] [--max-tokens MAX_TOKENS]
                      [--warmup WARMUP] [--prompts PROMPTS] [--warm-cache]
                      [--workload {default,agent}] [--turns N]
                      [--sessions-concurrency C] [--tool-output-tokens T]
                      [--assistant-turn-tokens A] [--system-prompt-tokens S]
                      [--request-timeout SECONDS]
                      [--max-requests MAX_REQUESTS]
                      [--max-tokens-per-request MAX_TOKENS_PER_REQUEST]
                      [--max-total-requested-tokens MAX_TOTAL_REQUESTED_TOKENS]
                      [--max-concurrency MAX_CONCURRENCY]
                      [--max-elapsed-seconds MAX_ELAPSED_SECONDS]
                      [--max-estimated-spend USD] [--history] [--limit LIMIT]
                      [--history-dir HISTORY_DIR] [--against CHECK_ID]
                      [--no-save] [--no-calibrate] [--fail-if-costlier PCT]
                      [--json PATH] [--share] [--share-id CHECK_ID]

Measure what this endpoint costs in dollars per million tokens, remember
the serving config it was measured on, and compare with the last check of
the same endpoint.

Run it after every serving change (engine flags, batch size, model,
quantization, GPU type, engine version) so a change that quietly makes
inference more expensive is caught before the bill shows it.

The GPU $/hr is yours (ASSUMED); tokens and time are MEASURED. Overlapping
95% confidence intervals are reported as NO WINNER, never as a winner.

A 95% CI is computed within one run, so it cannot see run-to-run drift
(machine load, thermal state, other tenants). CHEAPER or MORE EXPENSIVE
also needs a run-to-run noise bound (t x sqrt(2) x SD) estimated from
repeat checks of an unchanged config in the last 24 h, and a measured
change larger than it. Until there are 3 such earlier checks (2 df), or
when the baseline is more than 24 h old, the verdict is NOT CALIBRATED.
So a first check calibrates itself: with no recent repeats of this config
it runs up to 3 times back to back (each recorded as its own check, all
within --max-elapsed-seconds). --no-calibrate runs once. A change in the
ASSUMED GPU rate alone is never a verdict.

The path: 1. run it once (it calibrates)  2. change one setting
3. run the same command again for a verdict.

options:
  -h, --help            show this help message and exit
  --gpus N              how many GPUs serve this endpoint (default 1, or a
                        'gpus' --config value). The per-GPU price stays
                        ASSUMED, but a change in GPU count is a real config
                        change, so its cost counts in the verdict

endpoint:
  --url, --endpoint-url URL
                        OpenAI-compatible base URL, e.g. http://localhost:8000
                        or .../v1 (--endpoint-url is accepted too)
  --model MODEL         model name to send requests to (default: the only
                        model the server lists, if it lists exactly one)
  --api-key-env NAME    environment variable holding the bearer key (default:
                        OPENAI_API_KEY)
  --allow-insecure-http
                        allow plain HTTP to a non-loopback host (your key
                        travels unencrypted)
  --metrics-url METRICS_URL
                        vLLM Prometheus /metrics URL to read config labels
                        from (default: <host>/metrics; skipped quietly if not
                        reachable)
  --no-metrics          do not read /metrics for the config fingerprint

cost:
  --gpu-hourly-rate, --gpu-rate-per-hour, --total-hourly-price USD
                        what the GPU(s) behind this endpoint cost you per
                        hour, in dollars (recorded as ASSUMED; include every
                        GPU the model uses)
  --metric {output,input,total}
                        which tokens the headline $/M figure is per (default:
                        output)
  --monthly-tokens N    your monthly volume of the headline tokens (e.g. 500M,
                        2B) to turn $/M into a monthly dollar figure

config fingerprint:
  --label LABEL         short name for this config, e.g. 'fp8-batch256'
  --config KEY=VALUE    a serving setting the endpoint cannot report itself
                        (repeatable), e.g. --config gpu=H100 --config
                        quantization=fp8

workload (keep identical between checks):
  --blocks BLOCKS       independent measurement blocks; the 95% CI is across
                        blocks (default: 5, minimum 3)
  --requests-per-block REQUESTS_PER_BLOCK
                        requests sent in each block (default: 4)
  --concurrency CONCURRENCY
                        requests in flight at once; cost depends heavily on
                        this, so match your production load (default: 2)
  --max-tokens MAX_TOKENS
                        max output tokens per request (default: 64)
  --warmup WARMUP       unmeasured warm-up requests before measuring (default:
                        1)
  --prompts PROMPTS     JSONL prompt file (default: Throttle's built-in
                        prompts)
  --warm-cache          resend identical prompts every block, to measure
                        cache-friendly traffic. By default each measured
                        request's first user message starts with a unique tag
                        such as '[run 482915 req 0012] ' so a server with
                        prefix caching (vLLM, SGLang, Ollama) cannot serve
                        repeats from cache. Cold and warm checks are never
                        compared with each other
  --request-timeout SECONDS
                        per-request timeout (default: 120)

agent workload (--workload agent; keep identical between checks):
  --workload {default,agent}
                        'default': single-turn prompts, each with a unique
                        tag. 'agent': multi-turn synthetic agent sessions.
                        Each session shares one system prompt plus tool
                        schema, and every turn resends the conversation so far
                        plus a scripted assistant turn and a tool result, so a
                        prefix cache can reuse it within the session (a per-
                        run tag in the system prompt stops one run's cache
                        helping the next). A block is --sessions-concurrency
                        concurrent sessions of --turns sequential turns;
                        --requests-per-block and --prompts do not apply. Agent
                        and default checks are never compared
  --turns N             agent: turns per session (default: 6)
  --sessions-concurrency C
                        agent: sessions run at once, and sessions per block
                        (default: --concurrency)
  --tool-output-tokens T
                        agent: approximate size in tokens of the synthetic
                        tool result appended each turn (default: 300)
  --assistant-turn-tokens A
                        agent: approximate size in tokens of the scripted
                        assistant turn appended to the history each turn. The
                        live reply is measured but never fed back, so every
                        run sends the same prompts (default: 150)
  --system-prompt-tokens S
                        agent: approximate size in tokens of the shared system
                        prompt plus tool schema (default: 1500)

safety limits (checked before any traffic; same request, token, concurrency and spend defaults as smoke/benchmark):
  --max-requests MAX_REQUESTS
                        refuse a check that would send more requests than
                        this, warm-ups included (default: 10000)
  --max-tokens-per-request MAX_TOKENS_PER_REQUEST
                        refuse --max-tokens above this (default: 1024)
  --max-total-requested-tokens MAX_TOTAL_REQUESTED_TOKENS
                        refuse a check whose requests x --max-tokens exceeds
                        this; on a per-token API this is your bill ceiling
                        (default: 2,000,000)
  --max-concurrency MAX_CONCURRENCY
                        refuse --concurrency above this (default: 64)
  --max-elapsed-seconds MAX_ELAPSED_SECONDS
                        stop the check (nothing recorded) once it has run this
                        long (default: 300)
  --max-estimated-spend USD
                        refuse a check whose worst-case GPU time (--max-
                        elapsed-seconds x --gpu-hourly-rate) could cost more
                        than this (default: $3.00)

history and CI:
  --history             list past checks (all endpoints, or only --url) and
                        exit; sends no traffic
  --limit LIMIT         rows shown by --history (default: 20)
  --history-dir HISTORY_DIR
                        where checks are stored (default:
                        $THROTTLE_CHECK_HISTORY_DIR or ~/.throttle/checks)
  --against CHECK_ID    compare with this check id instead of the latest one
                        for the endpoint
  --no-save             compare but do not add this check to history
  --no-calibrate        run one check only. By default, when there is no
                        recent noise measurement for this endpoint, config and
                        workload, the first check runs up to 3 times back to
                        back (each recorded as its own check, all within
                        --max-elapsed-seconds) so run-to-run noise is measured
                        in one command, within --max-requests and --max-total-
                        requested-tokens in total. Not done with --no-save,
                        --fail-if-costlier or --against, after a config
                        change, or when CI or GITHUB_ACTIONS is set to a true
                        value (CI runs one check). With --json, a calibrating
                        command writes its last check plus a 'calibration'
                        object (ids, noise, first_comparison: run 1 against
                        the real baseline)
  --fail-if-costlier PCT
                        exit with code 4 if this check is a calibrated MORE
                        EXPENSIVE than the previous one by at least PCT
                        percent; exit 5 if the change could not be judged (NOT
                        CALIBRATED, or the baseline ran a different workload,
                        e.g. another prompt cache mode)
  --json PATH           also write this check and its comparison as JSON

share your results (nothing is uploaded):
  --share               print a sanitized markdown summary (no URLs, hosts,
                        IPs or keys; secret-looking config values masked) and
                        a pre-filled GitHub issue link you can review and
                        submit yourself. With a measuring check it shares that
                        check; with --history it shares the latest recorded
                        check (of --url, if given) without sending traffic
  --share-id CHECK_ID   share this recorded check (see --history) instead;
                        sends no traffic

Examples:
  throttle check --url http://localhost:8000 --model my-model \
      --gpu-hourly-rate 2.49 --config max_num_seqs=256 --label before
  throttle check --url http://localhost:8000 --model my-model \
      --gpu-hourly-rate 2.49 --config max_num_seqs=512 --label after \
      --monthly-tokens 500M --fail-if-costlier 5
  throttle check --history
  throttle check --history --share     # share the latest check (no traffic)
  throttle check --share-id CHECK_ID   # share a chosen check

Prompt cache: by default every measured request's first user message
starts with a unique tag ('[run 482915 req 0012] ', run id recorded) so a
prefix cache cannot serve repeats; --warm-cache resends identical prompts.
Cold and warm checks are never compared (NO WINNER).

OUTPUT CHANGED: the same prompts at temperature 0 got answers of a
different length (more than 10%), or many more answers stopped at
--max-tokens. A $/M change is then not trusted: read a few answers first.

Exit codes:
  0  check done (cheaper, no winner, first check, or not calibrated
     without --fail-if-costlier)
  1  measurement failed; nothing recorded
  2  usage error
  4  --fail-if-costlier tripped: calibrated MORE EXPENSIVE by >= PCT
     (measured change, at the baseline's GPU rate)
  5  --fail-if-costlier given but the verdict is NOT CALIBRATED (run-to-run
     noise not measured yet) or OUTPUT CHANGED; treat it as a warning or a failure
```

</details>
