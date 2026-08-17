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
protocol eligibility/order-balanced contrasts.

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
- raw GPU fingerprints.

The manifest keeps canonical SHA-256 workload fingerprints, a hashed GPU
fingerprint, validated non-secret runtime labels, engine-flag names/values, and
aggregate numeric samples. Workload hashes reveal equality and may confirm a
guessed low-entropy workload; they are not encryption. `throttle plan` is
terminal-only and intentionally displays the destination before any traffic.

Schema 1.0 validation JSON under `validation/` is historical smoke evidence.
It is not auto-upgraded and `throttle compare` rejects it with
`legacy_schema_not_decision_grade`.
