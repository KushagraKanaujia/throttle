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
3. **Runtime provenance is attested.** Throttle validates shape, consistency,
   exact pins, and hashes, but cannot independently prove that an operator's
   accelerator, software environment, model commit, cache state, or effective
   flags are truthful. Keep external audit evidence. CUDA image digests and
   direct-host software-environment digests are operator-supplied identities,
   not remotely inspected environments.
4. **Intervals are deliberately conservative and small-sample.** Request
   percentiles use bounded deterministic bootstrap diagnostics; decisions use
   repeated blocks. The six-run golden result has only two order-balanced phase
   contrasts, so inconclusive results should be common.
5. **One load-generator host.** Open-loop timing uses one asyncio process and a
   monotonic clock. It is not a distributed generator and can itself saturate;
   scheduler lag/backpressure are reported rather than hidden.
6. **Cost is attribution, not billing discovery.** Dedicated pricing uses
   client wall time; exact serverless results require provider active seconds;
   user-supplied totals are labeled. Queue, storage, CPU, networking, and idle
   billing are not inferred.
7. **No retries in measurement.** A transient failure invalidates the block.
   This is intentional for zero-contamination evidence but may require a fresh
   full position after the underlying problem is resolved.
8. **Prompt hashes are not semantic workload analysis.** They prove exact
   canonical messages/order and disjoint warm-up identities, not
   representativeness. They also reveal equality and can confirm a guessed
   low-entropy workload; hashing is not encryption. The operator must choose
   approved prompts and an appropriate cache policy.
9. **Python compatibility is continuously checked, but platform evidence is
    still bounded.** Hosted CI runs warning-strict, offline-guarded source and
    clean-wheel tests on Python 3.11 through 3.14. CI does not replace live accelerator-specific endpoint
    validation on every supported operating system.
10. **Native multi-load order is not counterbalanced.** Conditions currently
    execute condition-major and use deterministic condition/block seeds. A
    multi-load best-tested value is therefore always descriptive/inconclusive,
    because repeated blocks within each condition cannot rule out time drift or
    a small prompt-mix remainder between conditions. Saved matched-run compare
    and the six-position golden protocol remain the decision paths. Removing
    this gate requires a block-major counterbalanced scheduler that reuses the
    same per-block prompt schedule across conditions and records that schedule.
11. **The one-command golden runner is count-bounded and operator-mediated.**
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
12. **Sanitization is a boundary, not a secret detector.** Generated and loaded
    metadata reject the credential, URL, path, Unicode-spoofing, and structural
    shapes covered by the versioned test corpus, but no finite pattern set can
    prove arbitrary text is non-secret. Operators must never place credentials,
    endpoint details, private paths, prompts, or responses in provenance or
    engine-flag metadata. The Golden consumed-evidence fingerprint projection is
    versioned and must be updated whenever a future schema adds new decision
    evidence.
