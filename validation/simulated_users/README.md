# Simulated-user fixture suite

This directory is validation evidence for Throttle's local behavior. It is not
a GPU benchmark, model evaluation, operator pilot, or production performance
claim.

Seven prompt profiles exercise a deterministic loopback OpenAI-compatible
fixture at closed-loop concurrency 1, 2, 4, 8, and 16. Each condition uses one
warm-up plus 16 measured requests, so one profile plans exactly 85 calls. The
fixture models prompt-size-dependent delays and contention; it does not run a
model. The `stress_large` profile injects one HTTP 503 when at least twelve
large requests are simultaneously active. That failure must invalidate the
condition and suppress decision/cost claims.

The completed 2026-08-17 evidence bundle starts at
[`run-20260817/RUN_AUDIT.md`](run-20260817/RUN_AUDIT.md). Its expectations were
frozen before traffic, and it includes the seven mode-0600 schema-2 reports,
artifact hashes, a sanitized fixture snapshot, and the fail-closed audit.

Measured and warm-up prompts are separate. The server retains only aggregate
counts, prompt character ranges, maximum active requests, fixed parameter
sets, and sanitized outcome labels. Reports omit URLs, credentials, prompts,
and responses.
