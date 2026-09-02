# Throttle

**A local CLI for benchmarking LLM inference endpoints**

Throttle is a bring-your-own-endpoint CLI for measuring existing
OpenAI-compatible chat-completions servers. It provisions nothing, changes
nothing on the server, and never claims universal optimization or projected
savings.

Throttle has eight explicit workflows:

- `throttle plan` sends zero traffic and shows the destination, request/token
  ceilings, duration, cost bound, and privacy implications.
- `throttle smoke` is a short connectivity/load-shape check. Its default is 27
  calls: concurrency 1/4/8 × (8 measured + 1 separate warm-up). It is always
  non-decision-grade.
- `throttle benchmark` runs sustained, repeated blocks using closed-loop
  concurrency or open-loop request rates. A multi-load sweep is exploratory:
  its current condition-major order is not counterbalanced, so it cannot reach
  `decision_eligible: true` even when every request succeeds.
- `throttle diagnose` is a lightweight pre-flight bottleneck classifier that
  runs before formal benchmarking. It identifies the dominant serving regime
  (dispatch-bound, orchestration-bound, compute-bound, memory-bound, or mixed)
  to guide configuration tuning.
- `throttle experimental-tuning` is an explicit, suggestion-only smoke run
  that samples a configured vLLM Prometheus exporter. It never applies a
  setting and cannot change decision or Golden eligibility.
- `throttle golden` orchestrates the counterbalanced
  B1/C1/B2/C2/B3/C3 protocol for a controlled baseline/candidate comparison.
  This is the path for a decision-eligible configuration result.
- `throttle compare` compares saved reports offline. Two inputs perform a
  normal saved-run comparison; six ordered inputs validate the golden
  B1/C1/B2/C2/B3/C3 protocol.
- `throttle proxy` runs a standalone OpenAI-compatible HTTP server that sits
  in front of real inference backends and caches responses using semantic
  similarity matching. Unlike the benchmark cache, this serves external HTTP
  clients (curl, OpenAI SDKs, etc.) for production traffic caching.

Results describe only the declared workload and manifest.

## Proven Results

Throttle has one decision-eligible result: a six-position counterbalanced Golden protocol run on Qwen2.5-0.5B-Instruct with vLLM 0.16.0 on an A100 80GB GPU. Changing `max_num_seqs` from 1 to 8 produced a measured **+189.5% to +246.2%** throughput increase (95% CI) at closed-loop concurrency 8. This result passed all protocol gates and is `decision_eligible: true`.

The caching proxy has been verified compatible with **Ollama** in CI integration tests. Compatibility with **vLLM, SGLang, and LMDeploy** is expected (they implement the OpenAI-compatible `/v1/chat/completions` API) but requires GPU verification. See `validation/gpu_backend_verification.sh` for a runnable verification script on Linux with CUDA.

**See [RESULTS.md](RESULTS.md) for the full validated evidence**, including exact numbers, hardware details, protocol audit, and limitations. All claims trace to specific JSON artifacts in `validation/`.

## Choose the right path first

Use a sweep to learn the shape of one server, and use the golden protocol to
make a configuration decision. They answer different questions:

| Goal | Command | Can reach `decision_eligible: true`? |
| --- | --- | --- |
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

Throttle requires Python 3.11+ and is available on PyPI. Install with pipx (recommended for CLI tools):

```sh
pipx install throttle-pro
throttle --version
```

If you don't have pipx, install it first:
```sh
# macOS
brew install pipx

# Linux/WSL
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

**Alternative:** If you're already inside a virtualenv, use pip:
```sh
pip install throttle-pro
```

**Quickstart:** If you have a vLLM server with Prometheus metrics exposed (default port 8000), get live cost-per-million-tokens instantly:

```sh
throttle watch --gpu-rate-per-hour 1.50
```

This reads `/metrics` without sending requests — one command to see real-time $/MTok.

### Install from source

To install the development version:

```sh
git clone https://github.com/KushagraKanaujia/throttle.git
cd throttle
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
throttle --version
```

### Configuration File (Optional)

Throttle supports loading default values from `~/.throttle/config.yaml` to avoid repeating CLI flags. All config values are optional, and CLI flags always override config file settings.

**Setup:**
```sh
# Install PyYAML (optional dependency)
pip install pyyaml

# Create config directory and copy example
mkdir -p ~/.throttle
cp .throttle.yaml.example ~/.throttle/config.yaml

# Edit with your preferred defaults
nano ~/.throttle/config.yaml
```

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

**Validation:** each config value is checked against the same rules its CLI flag would enforce (type, allowed choices, and whether it expects a single value or a list) before being applied. A mismatch, e.g. `max-tokens: -5` where the flag requires a positive number, or `concurrency: 4` where `--concurrency` expects a list, fails immediately with an error naming the offending config key, rather than crashing later or silently reaching the program with a bad value. Arguments that require an exact number of values (for example `report`'s two positional report files) need a list of exactly that length; `concurrency`-style arguments that accept any number of values need a list of one or more.

### Quick Start (Local Testing)

The fastest way to try Throttle is against a local Ollama server:

1. **Install Ollama** from [ollama.com](https://ollama.com/download)

2. **Pull and start a small model:**
   ```sh
   ollama pull llama3.2:1b
   ollama serve  # if not already running
   ```

3. **Run a smoke test:**
   ```sh
   # Set a dummy API key (Ollama doesn't need one, but throttle requires the variable)
   export OLLAMA_API_KEY="ollama"

   throttle smoke \
     --model llama3.2:1b \
     --url http://localhost:11434/v1 \
     --api-key-env OLLAMA_API_KEY \
     --cost-model unknown \
     --allow-unknown-cost \
     --output smoke.json
   ```

4. **Test the cache feature:**
   ```sh
   throttle smoke \
     --model llama3.2:1b \
     --url http://localhost:11434/v1 \
     --api-key-env OLLAMA_API_KEY \
     --cost-model unknown \
     --allow-unknown-cost \
     --enable-cache \
     --output smoke-with-cache.json
   ```

The smoke run completes in under 2 minutes and sends 27 requests total (24 measured + 3 warmups). With `--enable-cache`, you'll see dramatically higher throughput for cached requests at higher concurrency levels.

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

Example pinned exploratory sweep:

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
  --accelerator-backend cuda \
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
  --output exploratory-sweep.json
```

Direct-host Metal, ROCm, and CPU runs use platform-neutral provenance instead
of fake CUDA or container values. For example, append the following runtime
controls to a pinned Apple Silicon run:

```sh
--accelerator-backend metal \
--accelerator 'Apple Silicon integrated GPU' \
--accelerator-fingerprint 'operator-private-stable-device-id' \
--accelerator-runtime-version 'MLX 0.32.0' \
--host-os-version 'macOS 15.0 build 24A335' \
--software-environment-digest 'python-environment@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
```

The software-environment digest must identify retained immutable evidence such
as a canonical dependency lock, installed-package manifest, or native binary
closure. Throttle validates the declaration and comparisons; it does not build
or independently inspect that environment. Use a non-secret label followed by
`@sha256:<64 lowercase hex>` (or a bare SHA-256 digest); URLs, credentials,
absolute paths, traversal segments, and control characters are rejected.

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
throttle benchmark --url https://... --model ... \
  --enable-cache \
  --cache-ttl-seconds 3600 \
  --cache-max-size 1000 \
  --cache-similarity-threshold 0.85
```

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

**Quick start** (start Ollama first with `ollama serve` and `ollama pull llama3.2:1b`):

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
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'

# Check cache stats - should show "misses": 1, "hits": 0
curl http://localhost:8080/health

# Second IDENTICAL request - cache hit (MUST match model, max_tokens, messages exactly)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'

# Verify cache hit - should show "hits": 1
curl http://localhost:8080/health
```

**Cache scope**: model, temperature, max_tokens, and all other sampling parameters must match exactly for a cache hit. Changing any parameter creates a different cache scope.

**Matching tiers**: the cache checks three tiers in order: exact match (O(1)), then lexical Jaccard token-overlap (threshold 0.85, always on), then an optional semantic embeddings tier.

By default (lexical-only), **paraphrases will miss** despite identical meaning. For example, `"optimize PostgreSQL queries"` vs `"optimize database queries in PostgreSQL"` has Jaccard similarity ~0.64, below the 0.85 threshold, so the second request hits the backend. Exact or near-exact token matches work well without any extra setup.

**Semantic embeddings (opt-in)**: enable with `--enable-embeddings` to catch paraphrases like the example above. Uses `sentence-transformers/all-MiniLM-L6-v2` via ONNX Runtime, threshold 0.95. Requires the `embeddings` extra:
```bash
pip install throttle-pro[embeddings]
throttle proxy --backend-url http://localhost:11434 --enable-cache --enable-embeddings --port 8080
```
If `--enable-embeddings` is passed without the extra installed, the proxy starts with embeddings marked `REQUESTED BUT UNAVAILABLE` and falls back to lexical-only matching rather than failing.

**Threshold behavior**: cosine similarity from this model encodes topic, not polarity. At threshold 0.95, `"Is it safe to use eval?"` vs `"Is it dangerous to use eval?"` scores 0.9874, above the threshold on similarity alone. This is a structural property of the embedding model, not something a higher threshold fixes, so the cache runs an explicit negation/antonym/version-conflict guard before accepting an embeddings-tier hit and skips the match if one is detected.

For detailed configuration, streaming behavior, error handling, and production deployment
considerations, see [PROXY_DEMO.md](PROXY_DEMO.md).

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
  --model Qwen/Qwen3-8B \
  --url https://inference.example/v1 \
  --api-key-env VLLM_API_KEY \
  --baseline-config max_num_seqs=8 \
  --candidate-config max_num_seqs=10 \
  --concurrency 16 \
  --cost-model dedicated-hourly \
  --gpus 1 \
  --total-hourly-price 0.50 \
  --cache-policy disabled \
  --model-revision 0123456789abcdef0123456789abcdef01234567 \
  --image-digest 'registry.example/vllm@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' \
  --gpu 'NVIDIA A100 80GB PCIe' \
  --gpu-fingerprint 'operator-private-stable-device-id' \
  --cuda-version 13.0 \
  --driver-version 580.42 \
  --server-version 0.27.1 \
  --engine-flag enable_chunked_prefill=true \
  --engine-flags-provenance runtime_verified \
  --p95-slo-ms 5000 \
  --ttft-slo-ms 1000 \
  --evidence-source live_inference \
  --output-dir golden-run-001
```

Replace those SLO examples with the operator's actual thresholds. A
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
  --probe-requests 20 \
  --output diagnose.json
```

The command runs 1 block of 20 requests per concurrency level with 3 warm-ups (maximum 200 total requests) under a strict 60-second execution ceiling.

Based on client-side timing heuristics, it classifies the server into one of five regimes:
- **dispatch-bound** (launch overhead): CPU overhead dominates; recommended tuning: `cuda_graph_capture`, `batch_size`.
- **orchestration-bound** (host jitter): high inter-request latency; recommended tuning: `scheduler_config`, `request_batching_strategy`, `python_vs_cpp_runtime`.
- **compute-bound** (GPU arithmetic): throughput scales linearly; recommended tuning: `max_num_seqs`, `max_num_batched_tokens`, quantization.
- **memory-bound** (VRAM limits): TTFT degrades sharply; recommended tuning: `kv_cache_block_size`, `prefix_caching`, `max_model_len`.
- **mixed**: multiple competing bottlenecks; run exploratory sweeps to isolate.

If the error rate exceeds 50%, or samples are insufficient, it returns `classification: inconclusive` (exit code `3`). It always sets `decision_eligible: false` and cannot be used with `throttle compare`.

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

The suite is offline-only and blocks non-loopback DNS/socket use:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

It covers modes, URL/proxy safety, response validation, streaming termination,
hard stops, partial reports, cost separation, open/closed-loop scheduling,
confidence and boundary logic, manifest tampering, saved comparisons, the
GuideLLM subprocess boundary, the six-run golden gate, and the opt-in
collector/analyzer/safety chain. Default commands are tested with collector
bombs so they cannot accidentally start experimental metric collection.

## Explicitly deferred

Throttle still does not build or perform automatic vLLM/TensorRT-LLM
reconfiguration, GPU/pod provisioning, replica autoscaling, GPU/instance
selection, spot orchestration, semantic/prefix caching, production traffic
proxying, async job queues, non-OpenAI backends, distributed multi-host tests,
accounts/teams, a hosted dashboard, a database/history/telemetry system,
production-log load discovery, monthly-savings claims, or a polished UI.

Remaining limitations and the current evidence boundary are listed in
[Known gaps](docs/KNOWN_GAPS.md).

## License

Throttle is released under the [MIT License](LICENSE).
