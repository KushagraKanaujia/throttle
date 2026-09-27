# Verdicts

*CHEAPER, MORE EXPENSIVE, NO WINNER and NOT CALIBRATED, and exactly when each is used.*

Every `throttle check` after the first compares with a baseline (the latest check of the same endpoint, or `--against CHECK_ID`) and prints one verdict.

| Verdict | When |
| --- | --- |
| **CHEAPER** | The measured change is larger than the noise bound **and** the 95% intervals don't overlap, and it's lower. |
| **MORE EXPENSIVE** | Same conditions, and it's higher. |
| **NO WINNER** | The change is inside the noise bound, or the intervals overlap, or the workloads differ (for example cold vs warm cache). Throttle names which condition failed. |
| **NOT CALIBRATED** | Fewer than 2 degrees of freedom of recent repeats, or the baseline is more than 24 hours old. Run the same check a few more times without changing anything. |

## Rules that keep it honest

- **The GPU rate alone is never a verdict.** $/M scales with the assumed rate, so a check with a different rate is first put on the baseline's rate; the rate-driven part is reported separately.
- **Different workloads are never ranked.** Prompts, requests per block, concurrency, max tokens and prompt cache mode must match.
- **The check being judged is never part of its own calibration.**

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Check done (cheaper, no winner, first check, or not calibrated without `--fail-if-costlier`) |
| 1 | Measurement failed; nothing recorded |
| 2 | Usage error |
| 4 | `--fail-if-costlier` tripped: calibrated MORE EXPENSIVE by at least the given percent |
| 5 | `--fail-if-costlier` given but the change couldn't be judged (NOT CALIBRATED, or a different workload) |
