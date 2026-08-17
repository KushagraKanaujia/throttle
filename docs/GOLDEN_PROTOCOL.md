# Golden live benchmark protocol

This protocol produces one sanitized, repeatable live comparison artifact from
six saved native Throttle reports. Throttle does not provision or reconfigure
the server; the operator performs each controlled change between runs.

A sanitized completed example is checked in under
`validation/golden-live-20260817`. It demonstrates this protocol for one pinned
model, GPU, workload, and time window; it is not external validation or a
universal performance, savings, or production claim. The offline suite also
validates the gate and comparison mechanics independently.

## Controlled question

Test one change the workload can exercise:

```text
baseline:  max_num_seqs=1
candidate: max_num_seqs=8
```

Each golden position tests exactly one condition: closed-loop concurrency 8.
Lower concurrency levels are useful exploratory controls, but they cannot
exercise the whole 1-versus-8 treatment and therefore belong in separate
benchmark reports, not the six golden positions. `256` versus `2048` at
concurrency 8 is rejected as unexercised. Chunked
prefill must have the same effective state on both variants and receives no
credit; vLLM V1 already enables it by default when possible.

## Immutable/common controls

Before traffic, record and independently retain evidence for:

- image name plus `@sha256:<64 hex>` digest;
- model repository commit (full 40- or 64-hex revision, never `main`);
- GPU label and one stable private fingerprint (only its SHA-256 is reported);
- CUDA, driver, server/vLLM, and Throttle versions;
- runtime-verified effective engine flags;
- model, `temperature=0`, fixed `max_tokens`, no stop tokens, streaming mode,
  request timeout, workload hash/order/seed, load shape, SLOs, and safety caps;
- an explicit cache policy; and
- the same physical GPU for all six positions.

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
condition. Use three × 20-second blocks for a time-bounded alternative. Every
response must pass strict shape, finish, usage, stream-termination, and size
validation. One failed, malformed, incomplete, cancelled, or partial response
invalidates the position; do not delete a bad block and keep the rest.

## Order

Use one endpoint/GPU sequentially, never two simultaneous endpoints:

```text
phase 1: B1 → C1 → B2
phase 2: C2 → B3 → C3
```

The six reports must not overlap. Use exactly these manifest values:

| Position | `--variant` | `--sequence-position` | `max_num_seqs` |
| --- | --- | --- | --- |
| B1 | baseline | B1 | 1 |
| C1 | candidate | C1 | 8 |
| B2 | baseline | B2 | 1 |
| C2 | candidate | C2 | 8 |
| B3 | baseline | B3 | 1 |
| C3 | candidate | C3 | 8 |

Run `throttle plan --run-mode benchmark` with each intended command before its
position. Confirm the exact request/token ceiling, spend bound, destination,
and privacy warning. Reconfigure only `max_num_seqs`, verify the effective
runtime flag, then execute the next position. Do not count server startup or
the first warm-up as measurement.

Example position-specific suffix:

```sh
--concurrency 8 \
--variant baseline \
--sequence-position B1 \
--engine-flag max_num_seqs=1 \
--engine-flag enable_chunked_prefill=true \
--engine-flags-provenance runtime_verified \
--evidence-source live_inference \
--output B1.json
```

The candidate uses the same command except `candidate`, `C1`, `8`, and its
output name. Repeat for all positions while preserving every other argument.

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

## Evidence boundary

The sanitized JSON deliberately omits endpoint, credentials, prompts,
responses, and raw server logs. Throttle checks internal consistency but cannot
independently prove operator-supplied hardware/runtime attestations. Keep the
provider allocation record, image inspection, model commit, effective server
startup/config output, and billing record in an operator-controlled audit
bundle. Never paste secrets or raw model content into the Throttle report.
