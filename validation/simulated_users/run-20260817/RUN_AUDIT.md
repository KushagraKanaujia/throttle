# Simulated-user smoke suite — run audit

## Scope and claim boundary

This was a deterministic loopback fixture exercise on 2026-08-17. It used no
model, GPU, external endpoint, operator, or RunPod resource. The values below
describe only the fixture and Throttle's client/reporting behavior. They are
not inference performance, production capacity, an optimization result, a
recommended configuration, or a savings estimate.

The predeclared outcomes are in [EXPECTATIONS.md](EXPECTATIONS.md). Seven
separate workloads were used because schema 2 aggregates conditions within one
workload. Combining profiles would erase the prompt-size distinction.

## Exact traffic envelope

- Closed-loop concurrency: `1, 2, 4, 8, 16`, in ascending order.
- Per condition: one disjoint warm-up plus 16 measured requests.
- Per normal profile: 85 calls total, 80 measured.
- Fixed request: streaming chat completions, temperature 0, 64 maximum output
  tokens, no stop tokens, seed 424242, cache policy disabled.
- Limits: 85 requests, 5,440 reserved output tokens, 30 seconds, one error,
  concurrency 16, and 65,536 response bytes.
- Cost: tagged `unknown` and explicitly acknowledged. No cost-per-token or
  savings value was calculated. Actual external infrastructure spend was $0.
- Evidence source: `synthetic_validation`; every report is smoke mode and
  `decision_eligible=false`.

## Results

Cells show `synthetic output tokens/second / p95 end-to-end ms`. These are
fixture observations, not model measurements.

| Measured profile (characters/prompt) | c1 | c2 | c4 | c8 | c16 | Descriptive fixture high point |
|---|---:|---:|---:|---:|---:|---|
| Short chat (53–63) | 901.97 / 73.66 | 1793.79 / 76.44 | 3446.06 / 78.47 | 5779.44 / 105.97 | 6106.37 / 163.49 | c16, boundary; throughput nearly flattened while latency rose |
| Support ticket (456–530) | 904.56 / 74.98 | 1818.98 / 73.96 | 3319.82 / 80.88 | 5622.01 / 91.37 | 7681.58 / 126.19 | c16, boundary |
| Code assistant (494–699) | 875.40 / 77.23 | 1711.83 / 78.96 | 3331.19 / 80.84 | 5595.41 / 105.39 | 5999.81 / 162.96 | c16, boundary; throughput nearly flattened while latency rose |
| Retrieval QA (807–1024) | 853.28 / 79.35 | 1698.91 / 80.59 | 3293.44 / 83.34 | 4486.59 / 173.62 | 6979.92 / 139.89 | c16, boundary; non-monotonic one-block latency |
| Document summary (1891–2425) | 783.45 / 86.58 | 1548.89 / 86.68 | 2953.10 / 95.21 | 4719.97 / 124.16 | 4201.39 / 216.05 | c8; c16 throughput fell while p95 rose sharply |
| Mixed workload (49–1161) | 911.57 / 75.56 | 1724.10 / 78.66 | 3360.12 / 79.22 | 5828.21 / 101.67 | 8930.91 / 107.45 | c16, boundary; mixed-size smoke result only |
| Deliberate pressure (4303) | withheld: run stopped | withheld: run stopped | withheld: run stopped | withheld: run stopped | **invalid: fixed HTTP 503** | no performance/high point published for the stopped run |

The six normal reports completed 510/510 calls: 480/480 measured completions
and 30/30 warm-ups were valid. The pressure report completed its five warm-ups
and 64 measured requests through c8, then received the predeclared 503 at c16.
Throttle reserved the full c16 cohort, stopped at `max_errors`, cancelled 15
in-flight client requests, recorded one invalid completion, and ended with zero
requests in flight. Across all reports: 595 client requests reserved, 580
completed, 15 cancelled, 579 valid responses, and one invalid response.

## Fail-closed proof

The pressure artifact has:

- `status=stopped`, `stop_reason=max_errors`; terminal execution observed exit 1
  (the retained JSON itself proves status, not the shell exit code);
- c1/c2/c4/c8 valid at 16/16 measured responses each;
- c16 invalid at 0/1, status count `503`, error
  `non_200_response`, and `metrics=null`;
- `decision_eligible=false`;
- `best_tested.available=false`, state `inconclusive`, reason
  `partial_or_failed_run`;
- run totals: 85 started, 70 completed, 15 cancelled, one error, zero in flight,
  peak client in-flight 16, and 5,440 reserved output tokens.

The server-side snapshot in [FIXTURE_STATS.json](FIXTURE_STATS.json) records
the one injected pressure failure, 13 connections cancelled while writing,
zero active work at capture, and no raw prompt or response text. Two of the 15
client-cancelled requests were cancelled before reaching the server.

## Artifact integrity

All seven reports strict-parse as finite schema-2 JSON, are mode `0600`, use
`artifact_type=throttle_run`, mode `smoke`, and contain the exact load sweep.
Request/token/concurrency accounting reconciles. Measured and warm-up workload
hashes are distinct and the per-prompt identity sets are disjoint.

| Artifact | SHA-256 |
|---|---|
| `short_chat.json` | `7371b3e83fde68f4196814170d8e00044698cb39380681247411f639982646c9` |
| `support_ticket.json` | `f5096aadd7ab0e45690670bdbc9b99a2fce70d8504f495c00593b6d8084bb880` |
| `code_assistant.json` | `12a27cb09fd1a2383d2e8d9dba053e258332c5ae46e20ca35d0faeb814941e04` |
| `retrieval_qa.json` | `16c9e980c30008d19c3fa7f38d9566fd1cf7125733dbf3d7be97d9677029f5e0` |
| `document_summary.json` | `6cdcb96efceeae15a6d3a267d69dd15bfe7e4bb0d34f1c24a780369c05b1f7a1` |
| `mixed_workload.json` | `891d2de3a6d1b5c8cb50a4734295adfe31ebf941e92677ce6e3fadbc3c675d0f` |
| `stress_large.json` | `60aaa7293199fb52e198950577f520c1bfc0f2ca4ccbb4641895f11275c3b193` |

The serialized reports were scanned for the loopback address/port, fixture
credential, authorization strings, every measured and warm-up prompt, fixture
response text, injected error text, and payload keys such as `messages` and
`content`; none were present. Only stable workload fingerprints remain.

## What this exercise proved—and did not prove

It proved that Throttle can plan and run distinct closed-loop smoke shapes,
capture streaming latency/throughput diagnostics, retain strict completion
validity, show a plateau/degradation, stop on the first injected error, cancel
in-flight work, persist a sanitized partial report, and suppress contaminated
metrics and claims. The retained bundle proves the final partial artifact and
absence of temporary residue; it does not independently replay the filesystem
operation that performed the atomic replacement.

It did not test inference quality, a tokenizer, GPU memory, vLLM scheduling,
production traffic, billing, or an operator decision. Smoke has one measured
block, so even internally consistent values remain statistically inconclusive.

## Verification notes

- Fixture/source compile checks, dependency `pip check`, strict report audit,
  accounting reconciliation, sanitization scan, JSON parsing, and listener
  teardown all passed.
- The warning-strict repository suite ran 98 tests: 97 passed and one existing
  real-clock open-loop timing assertion failed because a three-request sample
  missed the deliberately strict 5% offered-rate gate. The production path
  failed closed (`open_loop_target_achieved=false`); this seven-profile suite is
  closed-loop and did not exercise or modify that path. The safety tolerance
  was not loosened to make the timing assertion pass. Before public v0.2.0
  distribution, that test-only fixture was extended to eight requests; the
  unchanged safety gate then passed 20/20 focused repetitions and the complete
  warning-strict suite passed 98/98. This paragraph retains the original run's
  historical verification result rather than rewriting it.
- This task added only `validation/simulated_users` evidence and fixture files.
  It did not edit Throttle source, pilot/outreach material, or infrastructure,
  and it did not contact RunPod or any external endpoint.
