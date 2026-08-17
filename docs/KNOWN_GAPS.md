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
   GPU, image, model commit, cache state, or effective flags are truthful. Keep
   external audit evidence.
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
10. **Python compatibility evidence is local.** The full offline suite passes
    on isolated Python 3.11 and 3.13 interpreters as well as the workspace's
    3.14 interpreter. The actual optional GuideLLM 0.7.3 package and console
    command were additionally installed and checked on Python 3.13; no hosted
    CI matrix is included yet.
11. **Native multi-load order is not counterbalanced.** Conditions currently
    execute condition-major and use deterministic condition/block seeds. A
    multi-load best-tested value is therefore always descriptive/inconclusive,
    because repeated blocks within each condition cannot rule out time drift or
    a small prompt-mix remainder between conditions. Saved matched-run compare
    and the six-position golden protocol remain the decision paths. Removing
    this gate requires a block-major counterbalanced scheduler that reuses the
    same per-block prompt schedule across conditions and records that schedule.

Deferred by design: automatic engine reconfiguration, provisioning,
autoscaling, GPU selection, spot management, cache systems, production proxying,
queues, non-OpenAI backends, multi-host generation, accounts, dashboards,
databases/telemetry, production-log discovery, polished UI, and savings claims.
