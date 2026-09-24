# Operator-pilot packet audit

Audit date: 2026-08-17

## Scope

This pass created only the non-product files in `pilot/` and added a
supersession banner to `docs/USER_TESTING.md`. It did not modify `src/`,
`tests/`, packaging configuration, infrastructure, or any RunPod resource. No
external post, direct message, account action, or endpoint traffic was made.

## Verification

- The current official vLLM guidance routes fellow-user discussion to the
  forum, the Benchmarking category matches the request, and GitHub Discussions
  carries a retirement notice.
- The complete smoke skeleton passed `throttle plan --run-mode smoke` with exit
  0 and reported exactly 27 calls, 3,456 reserved output tokens, a 300-second
  ceiling, and a cost bound below its example spend cap. Plan sent zero traffic.
- The complete fixed-load benchmark skeleton passed
  `throttle plan --run-mode benchmark` with exit 0 and reported exactly 204
  calls, 26,112 reserved output tokens, a 600-second ceiling, and a cost bound
  below its example spend cap. Plan sent zero traffic.
- The walkthrough explains the expected exit-3 boundary state for each valid
  fixed-load position and gates comparison on the saved JSON evidence.
- The evidence ledger is header-only with 46 typed/code-only fields and no
  fabricated operator row.
- All local packet links resolve, trailing-whitespace checks pass, and no
  `src/` or `tests/` file changed after this pilot-doc pass began.
- Independent outreach, safety/command, evidence/YC, and whole-packet reviews
  returned no remaining blockers after corrections.

## Existing test-suite status

The final warning-strict offline run executed 98 tests: 97 passed and one
open-loop scheduling test failed. The failing fixture offered only three
requests at 100 requests/second and expected launch-rate error within 5%; this
machine observed about 92.9–93.8 requests/second (6.2–7.1% error), so Throttle
conservatively set `open_loop_target_achieved=false`. Repeated focused runs
failed the same timing assertion. This is a red release-test result and is not
hidden as a pass.

A read-only control used five 67-request blocks on the same runtime and achieved
99.78–99.96 requests/second (0.04–0.22% error); all met the target. The
production 5% gate is correctly fail-closed. The three-request assertion has
only two 10ms launch gaps and roughly a 1ms total timing budget, making the test
fixture brittle rather than revealing a sustained scheduler regression.

The operator packet uses closed-loop fixed-concurrency traffic, so this
open-loop timing assertion is outside its execution path. Per the explicit
no-code instruction, no source or test change was made here. It should be
handled later as a separately authorized correctness/test-reliability task.

### Distribution update

Before the public v0.2.0 repository push, the test-only open-loop fixture was
extended from three to eight requests. This preserves the production 5% gate
and tool logic while giving the real asyncio scheduler enough launch intervals
for a stable assertion. The warning-strict offline suite now passes 98/98. The
wheel was rebuilt after the stranger-facing README cleanup; its current hash is
recorded below. The historical pilot-pass result above is retained rather than
rewritten.

## Evidence boundary

Current external evidence remains zero: no qualified external operator, no
external endpoint run, no verified operator decision, no return, and no
payment. The existing golden artifact is linked only as engineering proof and
is not counted as demand evidence.

## SHA-256 manifest

```text
d15b52e640c0ed106b2e8342aeb678d1af0e2ee1541e64ac51ddea6293c72955  EVIDENCE.md
94f6f0d9f46e4466d971d01ea260231ac77217e34271e169c56a72f3416f9910  OPERATOR_ONE_PAGER.md
12bee2f2432edf316e73586170b485b25eb0d98a2785c108c566fe41010e8e72  OUTREACH.md
63bb9d8402ea2a9df209b158027ee9999698e9bd9488c67c0d4cba66c5860097  README.md
853a6562a9f563476afb92d3f235f12942c6767dd1b4c6e6370fedf7e3ea4860  RUN_CARD.md
28c8599f8e2de18cf6f1a2fe7907063939f741fd89bbe8fe74ca333c479eca7b  SAFETY_CONTRACT.md
3c7e574b8d0d570a6563cd7f510190c2df13d76f793614120fd2360e88e27165  WALKTHROUGH.md
078d1055ba66337372440c0659c6aea97e4fd254867609c4c19b71c57d79fbaf  evidence-ledger.csv
dd128c910c135f11219bb93037fa60b1c2c9365d86247b7c4020a2c084554dd5  throttle_bench-0.2.0-py3-none-any.whl
```

The next event that can change the external-evidence count is an explicitly
owner-authorized human post using `OUTREACH.md`.
