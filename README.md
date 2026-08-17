# Throttle

Throttle is a local, bring-your-own-endpoint CLI for measuring an existing
OpenAI-compatible chat-completions server. It provisions nothing, changes
nothing on the server, and never claims universal optimization or projected
savings.

Version 0.2 has four explicit modes:

- `throttle plan` sends zero traffic and shows the destination, request/token
  ceilings, duration, cost bound, and privacy implications.
- `throttle smoke` is a short connectivity/load-shape check. Its default is 27
  calls: concurrency 1/4/8 × (8 measured + 1 separate warm-up). It is always
  non-decision-grade.
- `throttle benchmark` runs sustained, repeated blocks using closed-loop
  concurrency or open-loop request rates. It reports a best *tested* condition,
  never an optimum.
- `throttle compare` compares saved reports offline. Two inputs perform a
  normal saved-run comparison; six ordered inputs validate the golden
  B1/C1/B2/C2/B3/C3 protocol.

Results describe only the declared workload and manifest.

## Try it safely

Throttle requires Python 3.11+. Clone the public repository and install the
reviewed v0.2.0 wheel:

```sh
git clone https://github.com/KushagraKanaujia/throttle.git
cd throttle
python3 -m venv .venv
. .venv/bin/activate
python -m pip install ./dist/throttle_bench-0.2.0-py3-none-any.whl
throttle --version
```

Or install the same wheel directly from the GitHub release:

```sh
python -m pip install https://github.com/KushagraKanaujia/throttle/releases/download/v0.2.0/throttle_bench-0.2.0-py3-none-any.whl
```

Run a zero-traffic plan before setting an API key or sending any load. Replace
the model and HTTPS URL with the operator-approved staging destination:

```sh
throttle plan \
  --run-mode smoke \
  --model YOUR_MODEL_ID \
  --url https://inference.example/v1 \
  --api-key-env VLLM_API_KEY \
  --cost-model unknown
```

`plan` does not read `VLLM_API_KEY`, resolve DNS, construct an HTTP client, or
send traffic. With unknown billing it deliberately reports that traffic is
blocked until the operator explicitly acknowledges the unavailable spend
calculation. Review the destination, request/token/time limits, cost model, and
privacy warning before proceeding.

For source development instead of the pre-built wheel, use
`python -m pip install -e .` inside the clone.

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

## Decision-grade benchmark mode

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
policy, immutable image/model pins, a supplied GPU fingerprint, complete
runtime versions, and runtime-verified engine flags. Those fixed reasons are
written under `decision_ineligible_reasons` instead of being hidden.

Example pinned native run:

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
  --p95-slo-ms 5000 \
  --ttft-slo-ms 1000 \
  --cache-policy disabled \
  --model-revision 0123456789abcdef0123456789abcdef01234567 \
  --image-digest 'registry.example/vllm@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' \
  --gpu 'NVIDIA A100 80GB PCIe' \
  --gpu-fingerprint 'operator-private-stable-device-id' \
  --cuda-version 13.0 \
  --driver-version 580.42 \
  --server-version 0.27.1 \
  --engine-flag max_num_seqs=1 \
  --engine-flag enable_chunked_prefill=true \
  --engine-flags-provenance runtime_verified \
  --evidence-source live_inference \
  --cost-model dedicated-hourly \
  --gpus 1 \
  --total-hourly-price 1.39 \
  --output B1.json
```

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

The controlled treatment implemented by the protocol is
`max_num_seqs=1` versus `8`. Each golden position contains exactly one
condition, `--concurrency 8`, so the workload actually exercises the treatment;
lower exploratory load levels belong in separate reports. Everything else must
remain pinned. On one GPU, run six non-overlapping native benchmarks in this
exact order:

```text
B1 → C1 → B2, then C2 → B3 → C3
```

Each position itself uses at least three blocks and meets the 200-valid-request
or 60-second floor per condition. Then validate and aggregate offline:

```sh
throttle compare B1.json C1.json B2.json C2.json B3.json C3.json \
  --output golden.json
```

The gate requires live inference, exact ordering/non-overlap, one hashed GPU
fingerprint, pinned image digest and full model commit, runtime-verified engine
flags, the same workload/SLO/cache policy, zero invalid responses, and the
1-versus-8 treatment. It evaluates order-balanced phase contrasts and retains
the 5% completion-token guard across every position; a declared SLO must also
hold in all six runs. See [the full protocol](docs/GOLDEN_PROTOCOL.md).

Throttle never provisions or reconfigures the GPU/server. In this repository no
server credentials or endpoint identifiers are retained. A sanitized completed
six-position run is included under
[`validation/golden-live-20260817`](validation/golden-live-20260817) as protocol
evidence. It measures only its pinned model, GPU, workload, and test window; it
is not a universal performance, savings, or production recommendation.

## Report privacy and exit codes

Reports contain hashes and aggregate numeric evidence, not endpoint URLs,
hostnames, keys, authorization headers, prompts, responses, raw exception text,
or GuideLLM raw output. Engine flag names/values are validated before they can
enter a manifest; GPU fingerprints are stored only as SHA-256.

- `0`: complete smoke, or a supported benchmark/comparison result.
- `1`: stopped/invalid/operational failure; a sanitized artifact is written
  whenever execution started.
- `2`: CLI usage error or incompatible saved reports.
- `3`: valid but statistically/qualification-inconclusive benchmark or compare.
- `130`: user cancellation with sanitized partial report.

## Test

The suite is offline-only and blocks non-loopback DNS/socket use:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

It covers modes, URL/proxy safety, response validation, streaming termination,
hard stops, partial reports, cost separation, open/closed-loop scheduling,
confidence and boundary logic, manifest tampering, saved comparisons, the
GuideLLM subprocess boundary, and the six-run golden gate.

## Explicitly deferred

Throttle still does not build or perform automatic vLLM/TensorRT-LLM
reconfiguration, GPU/pod provisioning, replica autoscaling, GPU/instance
selection, spot orchestration, semantic/prefix caching, production traffic
proxying, async job queues, non-OpenAI backends, distributed multi-host tests,
accounts/teams, a hosted dashboard, a database/history/telemetry system,
production-log load discovery, monthly-savings claims, or a polished UI.

Remaining limitations and the current evidence boundary are listed in
[Known gaps](docs/KNOWN_GAPS.md).
