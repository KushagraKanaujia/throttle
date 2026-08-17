# Throttle operator pilot

This folder is the operating packet for the first external Throttle pilot. It
does not add product functionality. The objective is to observe whether one
qualified vLLM operator uses a trustworthy result in a real configuration
decision.

## Current evidence boundary

As of 2026-08-17:

- engineering evidence: one valid six-position [live golden
  comparison](../validation/golden-live-20260817/golden.json) exists, with its
  [sanitized audit](../validation/golden-live-20260817/RUN_AUDIT.md);
- external qualified operators recruited: **0**;
- external staging endpoints tested: **0**;
- operator decisions influenced: **0**;
- unprompted returns or reruns: **0**; and
- paying operators: **0**.

The golden run proves benchmark mechanics. It is not demand, retention, or
willingness-to-pay evidence. Do not combine these categories.

## Pilot objective and exit condition

Recruit exactly one operator who:

1. owns or is authorized to test a self-hosted vLLM staging endpoint;
2. has a declared measurable latency, TTFT, throughput/goodput, or error target
   that Throttle reports;
3. has a real pending serving decision that the current tool can exercise;
4. can approve a bounded representative workload and either a known metered
   billing basis or a non-metered hard traffic/time window; and
5. agrees that we may observe what they actually do after the report.

The first-pilot milestone is complete only when the evidence ledger records a
qualified operator, an approved run card, traffic against their real staging
endpoint, the resulting sanitized artifact or exact failure state, teardown,
and a neutral 48–72 hour behavior follow-up. A compliment or install does not
complete it.

## Files

- [OUTREACH.md](OUTREACH.md): primary venue, ready-to-post copy, fallback, and
  qualification reply.
- [SAFETY_CONTRACT.md](SAFETY_CONTRACT.md): shareable execution and data
  agreement.
- [OPERATOR_ONE_PAGER.md](OPERATOR_ONE_PAGER.md): short prospective-operator
  walkthrough.
- [RUN_CARD.md](RUN_CARD.md): the per-session approval record completed before
  any traffic.
- [WALKTHROUGH.md](WALKTHROUGH.md): the exact operator experience and command
  skeleton.
- [EVIDENCE.md](EVIDENCE.md): signal rubric, follow-up questions, stage gates,
  and stop/pivot rules.
- [evidence-ledger.csv](evidence-ledger.csv): de-identified behavior ledger;
  intentionally contains no fabricated pilot row.
- [AUDIT.md](AUDIT.md): scope, zero-traffic command checks, independent review,
  evidence boundary, and artifact hashes.

## Next authorized action

The owner should review the outreach copy in their own voice and post it once
to the vLLM Forum Benchmarking category. No account was used and no external
message was posted while preparing this packet.
