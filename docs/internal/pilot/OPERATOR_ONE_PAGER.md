# What to expect from a Throttle pilot

This is a founder-guided benchmark against a staging vLLM endpoint you already
operate. The purpose is to answer one real pending configuration question—not
to demo features or collect opinions.

## Good fit

You have:

- authority to load-test a non-production vLLM endpoint;
- a current batching/`max_num_seqs` decision on the same GPU;
- a measurable latency, TTFT, throughput/goodput, or error target;
- approved measured and separate warm-up prompts; and
- immutable model/image revisions and runtime configuration evidence.

GPU-sizing and concurrency-only decisions are not accepted in this first pilot
because the current comparison contract cannot call them decision-grade.

## What happens

1. We spend about 15 minutes completing the run card: decision, SLO, workload,
   cache policy, traffic limits, billing basis, and stop conditions.
2. You install the pinned wheel and run Throttle locally. Your endpoint and
   short-lived key stay on your machine.
3. `throttle plan` shows the exact destination and request/token/time/spend
   ceilings without reading the key or sending traffic. You explicitly approve
   or stop.
4. A 27-request smoke checks connectivity and response validity. It is never a
   recommendation.
5. A baseline run sends at most 204 requests: 3 warm-ups plus 3 measured blocks
   of 67. Every measured response must be valid.
6. You change and verify the one predeclared candidate setting. Throttle does
   not touch your server. We repeat the identical plan and 204-request run.
7. `throttle compare` works offline and reports supported, inconclusive, or
   invalid honestly. A failed/malformed response or mismatched control prevents
   a decision claim.
8. You restore baseline, verify health, revoke the key, confirm traffic has
   drained, and review the sanitized report before sharing anything.

Each single fixed-load benchmark normally exits 3 because it is individually a
search-boundary observation. We inspect the saved JSON for complete,
decision-grade, zero-error evidence and then compare; exit 3 alone is not a
failed run.

Typical maximum traffic is 435 inference requests and 55,680 requested output
tokens across smoke, baseline, and candidate. Metered sessions use a USD $5 or
lower cap enforced by your provider guard or full-session wall-clock cutoff;
non-metered hardware uses an approved traffic/time ceiling. Already accepted
in-flight requests may finish after client cancellation, so the run card
reserves for them.

## What Throttle does and does not do

Throttle sends authorized chat-completion requests and writes local aggregate
reports. It does not provision, SSH, deploy, restart, scale, reconfigure,
delete, or call an admin/control-plane API. Inference still consumes capacity
and can affect server logs, caches, usage counters, and billing.

Reports omit the endpoint, credential, prompts, outputs, and raw logs. They do
contain workload fingerprints and operational metadata, so you review them
before optionally sharing a sanitized copy. Nothing is published or attributed
to you without separate written approval.

After the session, the only product question is:

> What, if anything, did you actually do because of the result?

The detailed terms are in [SAFETY_CONTRACT.md](SAFETY_CONTRACT.md); the exact
execution steps are in [WALKTHROUGH.md](WALKTHROUGH.md).
