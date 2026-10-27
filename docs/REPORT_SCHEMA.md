# Report contract (schema 2.0)

Throttle writes mode-0600 JSON with `allow_nan=false`. Run artifacts use:

```text
schema_version       "2.0"
artifact_type        "throttle_run"
mode                 "smoke" | "benchmark"
status               running | complete | failed | stopped | cancelled
decision_eligible    boolean
manifest             versioned workload/runtime/request/cost/safety contract
conditions[]         warm-up diagnostics, repeated blocks, aggregate evidence
best_tested          descriptive tested-condition boundary/CI state
run_totals           started/completed/cancelled/error/token reservations
cost_summary         one tagged billing model only
stop_reason          fixed sanitized code or null
```

Comparison artifacts use `artifact_type: throttle_comparison`. Six-position
golden artifacts use `artifact_type: throttle_golden_live_comparison` and add
protocol eligibility/order-balanced contrasts. Supported, decision-eligible
golden artifacts additionally include `decision_summary`; it is null for every
ineligible or inconclusive result. A stopped orchestration writes
`artifact_type: throttle_golden_session` with completed-position names and a
fixed sanitized stop reason, never a recommendation.

Golden comparison and session artifacts include a sanitized treatment block:

```json
{
  "field": "max_num_seqs",
  "baseline_value": 8,
  "candidate_value": 10,
  "closed_loop_concurrency": 16
}
```

For a complete comparison, these values are inferred from the six saved
runtime-verified effective-flag and traffic manifests rather than trusted from
CLI input. The block remains present for a statistically inconclusive complete
comparison; malformed evidence may produce `treatment: null`. Reaching the
recorded client concurrency proves offered demand, not direct server-scheduler
saturation.

Golden artifacts label `run_fingerprint_basis` as
`validated_consumed_evidence_projection_v1`. Each `run_fingerprints` entry
hashes only the bounded, schema-validated evidence consumed by report and
Golden validation. Safe forward-extension fields are deliberately omitted so
the digest cannot become an oracle for ignored payloads; a structurally or
semantically invalid source report receives a null fingerprint.

A supported `decision_summary` contains a fixed workload-scoped label, the
winning variant and inferred `max_num_seqs` value, the candidate-relative
throughput delta, its order-balanced 95% interval/method,
`ci_excludes_zero: true`, the declared SLO gates that passed, and the exact
terminal summary text. It never contains a savings, optimum, or
universal-performance claim.

New run artifacts use runtime manifest `1.1`. It retains the manifest `1.0`
CUDA keys for compatibility and adds an accelerator backend, generic
accelerator identity/fingerprint, accelerator runtime version, host OS version,
and immutable software-environment digest. Saved manifest `1.0` CUDA reports
remain readable and comparable with other `1.0` reports; different manifest
versions are never treated as the same controlled runtime.

## Decision gate

`decision_eligible` is derived, not a caller assertion. A normal comparison
recomputes manifest completeness, disjoint warm-ups, block order/shape and
observed concurrency, request/token/run totals, timestamps, safety and billing
math, zero invalid responses, and the 200-request/60-second floor. Smoke,
partial, stopped, cancelled,
GuideLLM aggregate, legacy schema 1.0, unknown schema, missing controls, output
length mismatch, unexercised treatments, and unsupported confidence intervals
cannot become decision-grade by editing a boolean in JSON.

Open-loop manifests pin a 5% achieved-rate tolerance and a p95 scheduler-lag
ceiling of one target inter-arrival interval. Every block records its actual
launch window, achieved offered rate, relative error, and lag ratio. Saved-run
preflight recomputes those values; missing the offered load is
decision-ineligible.

Request-level `p95_ci` fields are bounded bootstrap diagnostics. Aggregate
E2E/TTFT distributions additionally carry `p95_repeated_block_ci`, a 95%
Student-t interval over the p95 from each measured block. SLO qualification and
saved/golden decisions use only that repeated-block interval, and saved-run
preflight recomputes it from the block evidence.

`metrics.output_tokens_per_second` is the pooled descriptive ratio of all valid
completion tokens to total measured condition wall time.
`block_mean_output_tokens_per_second` and its
`block_mean_output_tokens_per_second_ci` are the repeated-block decision
estimate and interval. Best-tested selection uses that block-mean estimate, so
the point estimate and CI always describe the same estimand.
The derived best-tested object also records a 5% completion-tokens-per-valid-
response tolerance and the observed relative spread across all valid tested
conditions before SLO filtering.
Exceeding it forces `state: inconclusive`; raw token totals are not compared
across duration-bounded conditions with different request counts.
Native multi-load execution is condition-major, so its best-tested object is
also forced inconclusive with `multi_condition_order_not_counterbalanced`.
This preserves a descriptive selected value without presenting an order-
confounded scan as a supported decision.

## Privacy contract

Persisted artifacts do not contain:

- endpoint URLs or hostnames;
- API keys, authorization headers, or inherited proxy values;
- prompts/messages or generated content;
- raw HTTP bodies, GuideLLM output, server exception strings, or temp paths; or
- raw accelerator fingerprints.

The manifest keeps canonical SHA-256 workload fingerprints, a hashed
accelerator fingerprint, validated non-secret runtime labels, engine-flag
names/values, and aggregate numeric samples. CUDA decision-grade reports
require the existing immutable image, CUDA, and driver evidence. Metal, ROCm,
and CPU reports instead require a host OS version and immutable
software-environment digest; all platforms require an accelerator runtime
version. Workload hashes reveal equality and may confirm a guessed low-entropy
workload; they are not encryption. `throttle plan` is terminal-only and
intentionally displays the destination before any traffic.

Artifact references use a bounded non-secret label with an optional immutable
`@sha256:<64 lowercase hex>` suffix. Generated and loaded reports reject URLs,
credential-like labels, absolute/traversal paths, and control characters before
runtime evidence can be compared. Safe mutable labels and non-SHA-256 digest
references remain structurally readable for descriptive manifest 1.0
comparisons, but never satisfy the immutable decision-grade pin.

Saved and in-memory report boundaries are iterative and explicit: depth is
limited to 64 edges, total JSON nodes to 100,000, aggregate UTF-8 key/value text
to 20 MB, individual values to 16 KiB, and keys to 1 KiB. Duplicate keys,
non-finite or unsafe-magnitude numbers, cycles, custom containers, unsafe
Unicode categories, and NFKC-normalized credential/path lookalikes fail with
fixed non-reflective reason codes.

Schema 1.0 validation JSON under `validation/` is historical smoke evidence.
It is not auto-upgraded and `throttle compare` rejects it with
`legacy_schema_not_decision_grade`.
