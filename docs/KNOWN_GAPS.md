# Known gaps and evidence boundary

These limitations are explicit; none should be silently interpreted as a
successful optimization claim.

1. **The included golden run is narrow evidence, not external validation.** A
   sanitized six-position live comparison is retained under
   `validation/golden-live-20260817` and passes the protocol gate for its pinned
   model, GPU, workload, and window. It has not been independently replicated
   by an external operator and does not establish universal performance,
   savings, or production suitability.
2. **Native ITL is unavailable.** OpenAI SSE chunks are not guaranteed token
   boundaries. Native mode reports client chunk inter-arrival separately and
   refuses to relabel it as inter-token latency. TTFT, TPOT, and end-to-end
   latency are available.
3. **GuideLLM is cross-check-only.** Pinned GuideLLM 0.7.3 cannot prove finish
   reasons or usage provenance and has no endpoint-response byte cap. Throttle
   scrubs and constrains the child, but its imported aggregate is always
   decision/golden-ineligible and uses `synthetic_text`, not the supplied JSONL.
   Its tokenizer revision must already exist in the local cache because hidden
   tokenizer downloads are disabled. GuideLLM traffic is limited to POSIX
   platforms so forked descendants remain inside Throttle's termination
   boundary; Windows is rejected before traffic.
4. **Runtime provenance is attested.** Throttle validates shape, consistency,
   exact pins, and hashes, but cannot independently prove that an operator's
   accelerator, software environment, model commit, cache state, or effective
   flags are truthful. Keep external audit evidence. CUDA image digests and
   direct-host software-environment digests are operator-supplied identities,
   not remotely inspected environments.
5. **Intervals are deliberately conservative and small-sample.** Request
   percentiles use bounded deterministic bootstrap diagnostics; decisions use
   repeated blocks. The six-run golden result has only two order-balanced phase
   contrasts, so inconclusive results should be common.
6. **One load-generator host.** Open-loop timing uses one asyncio process and a
   monotonic clock. It is not a distributed generator and can itself saturate;
   scheduler lag/backpressure are reported rather than hidden.
7. **Cost is attribution, not billing discovery.** Dedicated pricing uses
   client wall time; exact serverless results require provider active seconds;
   user-supplied totals are labeled. Queue, storage, CPU, networking, and idle
   billing are not inferred.
8. **No retries in measurement.** A transient failure invalidates the block.
   This is intentional for zero-contamination evidence but may require a fresh
   full position after the underlying problem is resolved.
9. **Prompt hashes are not semantic workload analysis.** They prove exact
   canonical messages/order and disjoint warm-up identities, not
   representativeness. They also reveal equality and can confirm a guessed
   low-entropy workload; hashing is not encryption. The operator must choose
   approved prompts and an appropriate cache policy.
10. **Python compatibility is continuously checked, but platform evidence is
    still bounded.** Hosted CI runs warning-strict, offline-guarded source and
    clean-wheel tests on Python 3.11 through 3.14. The actual optional GuideLLM
    0.7.3 package and console command were additionally installed and checked
    on Python 3.13. CI does not replace live accelerator-specific endpoint
    validation on every supported operating system.
11. **Native multi-load order is not counterbalanced.** Conditions currently
    execute condition-major and use deterministic condition/block seeds. A
    multi-load best-tested value is therefore always descriptive/inconclusive,
    because repeated blocks within each condition cannot rule out time drift or
    a small prompt-mix remainder between conditions. Saved matched-run compare
    and the six-position golden protocol remain the decision paths. Removing
    this gate requires a block-major counterbalanced scheduler that reuses the
    same per-block prompt schedule across conditions and records that schedule.
12. **The one-command golden runner is count-bounded and operator-mediated.**
    It safely orchestrates the default 3 × 67 request positions, but does not
    yet run duration-bounded positions; those remain available through six
    manual benchmark reports plus offline validation. Arbitrary positive,
    distinct `max_num_seqs` pairs are accepted only when one common client
    concurrency is at least the larger value and is reached in every position.
    That proves offered client demand, not direct server-scheduler saturation
    or co-timed occupancy at the configured limit. Throttle times the operator
    confirmations and stops further client traffic at its session ceiling, but
    it cannot stop or bill-discover the provider resource. Keep a provider-side
    budget/auto-stop active during every transition.
13. **Sanitization is a boundary, not a secret detector.** Generated and loaded
    metadata reject the credential, URL, path, Unicode-spoofing, and structural
    shapes covered by the versioned test corpus, but no finite pattern set can
    prove arbitrary text is non-secret. Operators must never place credentials,
    endpoint details, private paths, prompts, or responses in provenance or
    engine-flag metadata. The Golden consumed-evidence fingerprint projection is
    versioned and must be updated whenever a future schema adds new decision
    evidence.
14. **The experimental agent chain has one isolated opt-in presenter.** The
    server metrics collector, suggestion-only bottleneck analyzer, and
    independent safety validator are reachable only through
    `throttle experimental-tuning`; no default, benchmark, comparison, or
    Golden path invokes them. The command writes an ordinary non-decision-grade
    smoke report and a separate fixed envelope whose SHA-256 binds the detached
    projection to that sanitized report. Every eligibility, action,
    Golden-bypass, CLI-self-authorization, and report-integration flag remains
    locked false. The digest proves file-content equality, not authorship. The
    path does not make process-wide exporter metrics request-scoped, prove the
    operator's same-deployment or traffic-isolation attestations, prove
    scheduler saturation, run Golden, sign the two files, or authorize a
    configuration change. Raw exporter snapshots and labels are intentionally
    discarded, so the derived server signals cannot be independently
    recomputed from the saved files. The projection includes only evidence used
    by the analyzer; exposing other collector diagnostics requires a newly
    reviewed allowlist. Like any in-process Python API, it is a validation
    boundary for data, not a sandbox against code that can modify module
    internals.
15. **Request acceptance is not semantic verification.** Native request
    profiles prove the exact safe fields sent by the client and require those
    fields to match across comparisons. A successful HTTP response does not
    independently prove that an engine honored a model-specific extension such
    as `enable_thinking`; retain engine logs or separate behavioral evidence.

Deferred by design: automatic engine reconfiguration, provisioning,
autoscaling, GPU selection, spot management, cache systems, production proxying,
queues, non-OpenAI backends, multi-host generation, accounts, dashboards,
databases/telemetry, production-log discovery, polished UI, and savings claims.
