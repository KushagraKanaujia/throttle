# Golden live benchmark protocol

This protocol produces one sanitized, repeatable live comparison artifact from
six saved native Throttle reports. Throttle does not provision or reconfigure
the server; the operator performs each controlled change between runs.

The supported live workflow is one `throttle golden` invocation. It automates
the six measurements, persistence, ordering, and final validation, while
pausing at explicit operator checkpoints for each server-side transition.

A sanitized completed example is checked in under
`validation/golden-live-20260817`. It demonstrates this protocol for one pinned
model, accelerator, workload, and time window; it is not external validation or a
universal performance, savings, or production claim. The offline suite also
validates the gate and comparison mechanics independently.

## Controlled question

Test one change the workload can exercise. Both values must be distinct,
canonical positive integers: ASCII decimal digits without a sign, whitespace,
or leading zero, in the range 1 through 2,147,483,647.

```text
baseline:  max_num_seqs=8
candidate: max_num_seqs=10
```

Each golden position tests exactly one closed-loop condition. Its declared
client concurrency must be at least `max(baseline, candidate)`, and every
position must actually reach that declared level. Use the analyzer's original
offered concurrency when it is higher; the example pair above can therefore be
tested at concurrency 16. If no concurrency is supplied, the one-command flow
defaults to the larger treatment value. Lower concurrency levels are useful
exploratory controls, but they cannot exercise the whole treatment and belong
in separate benchmark reports. `256` versus `2048` at concurrency 8 is rejected
as unexercised. Chunked prefill must have the same effective state on both
variants and receives no credit; vLLM V1 already enables it by default when
possible.

This exercise check is deliberately client-scoped. Reaching the declared
concurrency proves sufficient offered demand; it does not prove direct
server-scheduler saturation or that `max_num_seqs` sequences were scheduled at
the same instant.

## Immutable/common controls

Before traffic, record and independently retain evidence for:

- model repository commit (full 40- or 64-hex revision, never `main`);
- accelerator backend/label and one stable private fingerprint (only its
  SHA-256 is reported);
- accelerator runtime, server, and Throttle versions;
- one immutable software-environment digest;
- for CUDA, the image name plus `@sha256:<64 hex>` digest and the CUDA/driver
  versions;
- for direct-host Metal, ROCm, or CPU, the host OS version;
- runtime-verified effective engine flags;
- model, the exact sealed request profile, fixed `max_tokens`, no stop tokens,
  streaming mode, request timeout, workload hash/order/seed, load shape,
  declared SLOs (or an explicit throughput-only objective), and safety caps;
- an explicit cache policy; and
- the same physical accelerator for all six positions.

Warm-ups must use the separate warm-up JSONL, share no canonical prompt with
the measured workload, and are excluded from measurements.
Choose `disabled`, `cold`, `warm`, or `representative` cache policy before the
first run, then hold it constant. Do not clear or warm caches opportunistically
between variants.

## Per-position measurement floor

Every position is a native `throttle benchmark` with at least three measured
blocks per condition. Each condition must achieve either:

- at least 200 valid measured requests; or
- at least 60 measured seconds.

The default three blocks × 67 requests gives 201 measured requests per
condition. The one-command v0.3 orchestrator currently requires count-bounded
positions. A time-bounded alternative (for example, three × 20-second blocks)
is available through the manual six-report flow and offline validator described
below. Every response must pass strict shape, finish, usage, stream-termination,
and size validation. One failed, malformed, incomplete, cancelled, or partial
response invalidates the position; do not delete a bad block and keep the rest.

## Order

Use one endpoint/accelerator sequentially, never two simultaneous endpoints:

```text
phase 1: B1 → C1 → B2
phase 2: C2 → B3 → C3
```

The six reports must not overlap. Use exactly these manifest values:

| Position | `--variant` | `--sequence-position` | `max_num_seqs` |
| --- | --- | --- | --- |
| B1 | baseline | B1 | baseline value |
| C1 | candidate | C1 | candidate value |
| B2 | baseline | B2 | baseline value |
| C2 | candidate | C2 | candidate value |
| B3 | baseline | B3 | baseline value |
| C3 | candidate | C3 | candidate value |

## One-command orchestration

Build the full zero-traffic session plan first:

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

The displayed SLO values are examples; replace them with the operator's actual
thresholds. Omitting both declares a throughput-only objective. Such a result
may still be decision-eligible, but its summary explicitly says that no latency
SLO was declared and makes no latency claim.

The plan lists every missing eligibility prerequisite and the aggregate
six-position request, output-token, elapsed, and spend ceilings. It does not
read the API-key environment variable, resolve DNS, construct an HTTP client,
prompt for a transition, create the output directory, or send traffic.

After review, set the key locally and rerun the same command without
`--dry-run`. Before each position, Throttle prints the required variant and
waits for an exact confirmation such as `B1 verified`. Reconfigure only
`max_num_seqs` in another terminal, restart if needed, verify the effective
runtime flags, then confirm. Throttle never executes a reconfiguration command.
It refuses an existing output directory, atomically saves every completed
position, and stops the sequence on the first failed, malformed, partial, or
underpowered position.

With count-bounded defaults, each position has 201 measured requests plus
three warm-ups: 204 calls per position and 1,224 for the complete session. The
default session elapsed ceiling is 5,400 seconds. Safety and billing inputs are
held byte-for-byte constant across all position manifests; operator transition
time counts against the outer elapsed/dedicated-cost guard.

The live orchestrator rejects `user-supplied` totals and a pre-filled
`--billed-active-seconds` value: either would be copied into every position and
would misstate a six-run session total as six separate run totals. Dedicated
hourly billing, explicitly acknowledged unknown billing, and serverless rate
limits without a pre-filled billed duration retain their existing tagged
semantics. Reconcile the final provider bill in the operator-controlled audit
record after the run.

Do not count server startup or Throttle's separate warm-ups as measurement.

## Manual/offline compatibility

The original position-specific flow remains available for existing automation
and for validating six already-saved reports. An individual position uses:

Example position-specific suffix:

```sh
--concurrency 16 \
--variant baseline \
--sequence-position B1 \
--engine-flag max_num_seqs=8 \
--engine-flag enable_chunked_prefill=true \
--engine-flags-provenance runtime_verified \
--evidence-source live_inference \
--output B1.json
```

The candidate uses the same command except `candidate`, `C1`, `10`, and its
output name. Repeat all six positions while preserving every other argument.
The offline validator independently infers the two values from the
runtime-verified effective flags; it does not trust a separate declaration.

## Validate and compare

```sh
throttle compare B1.json C1.json B2.json C2.json B3.json C3.json \
  --output golden.json
```

The golden gate recomputes schema, block count, response counts, zero-error
status, duration/request floors, manifest controls, condition sets, ordering,
non-overlap, pins, source, treatment, and the 5% completion-token tolerance.
The token tolerance is enforced across each of the six positions, so opposing
output-length mismatches cannot cancel in an aggregate. Every configured
latency/TTFT SLO must hold throughout the sequence using the repeated-block
Student-t interval over per-block p95s, never a request-bootstrap diagnostic.
It evaluates two order-balanced phase contrasts:

```text
C1 versus mean(B1, B2)
mean(C2, C3) versus B3
```

Each position contributes its arithmetic mean repeated-block throughput, the
same estimand used by native best-tested selection and block confidence
intervals. Pooled tokens/wall remains descriptive and is not substituted into
the phase contrasts.

The 95% interval may legitimately include zero. In that case the protocol is
eligible but the result is `inconclusive`; repeat or redesign rather than
claiming a win. A winning highest tested load remains a search-boundary result,
not an optimum.

When—and only when—the protocol is eligible and statistically supported,
`golden.json` adds a structured `decision_summary` and the terminal prints its
single `Golden recommendation — tested workload only` line. It names the
inferred winning `max_num_seqs` value, reports the candidate-relative
throughput delta and order-balanced 95% interval, states that the interval
excludes zero, and lists the declared E2E/TTFT SLO gates that passed. It does
not claim latency parity, server saturation, cost savings, an optimum, or
generality beyond the pinned workload. Every structurally valid complete
comparison includes a sanitized `treatment` block with the inferred pair and
common client concurrency, including when the interval is inconclusive.
Ineligible and statistically inconclusive sessions keep
`decision_summary: null` and emit no recommendation.

## Evidence boundary

The sanitized JSON deliberately omits endpoint, credentials, prompts,
responses, and raw server logs. Throttle checks internal consistency but cannot
independently prove operator-supplied hardware/runtime attestations. Keep the
provider allocation record, image inspection, model commit, effective server
startup/config output, and billing record in an operator-controlled audit
bundle. Never paste secrets or raw model content into the Throttle report.

The checked-in `validation/golden-live-20260817` artifact remains a valid
historical 1-versus-8 instance of this generalized protocol.
