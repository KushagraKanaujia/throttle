# Decisions Log

## 2026-08-24: Shift from task-based to outcome-based approach

**Decision:** Stop following PLAN.md tasks sequentially. Instead, work toward the 8 concrete outcomes in the updated "Definition of done" in CLAUDE.md.

**Reasoning:** The new definition of done provides verifiable, user-facing outcomes:
1. Clean install works
2. `throttle demo` runs (<5min, no GPU, shows cost comparison)
3. `throttle cost` and `throttle tune` work against local endpoint
4. `throttle validate-sim` exists and runs
5. Error handling is clear (no tracebacks, no silent hangs)
6. Help text is plain language, jargon-free
7. QUICKSTART.md exists and works verbatim
8. CI green

**What's being skipped from PLAN.md:**
- Task 8 (500 case suite) - replaced by requirement 8 (CI green with existing tests)
- Any PLAN.md tasks not needed for the 8 outcomes will be skipped

**What's being prioritized:**
- Simulator implementation (for `throttle demo`)
- `throttle cost`, `throttle tune`, `throttle validate-sim` commands
- QUICKSTART.md
- User-facing polish (error messages, help text)

**Alternative considered:** Continue with PLAN.md tasks 4-10 sequentially.

**Why this is better:** The new definition of done is concrete, verifiable, and user-focused. It ensures a working end-to-end experience rather than completing tasks that may not contribute to usability.

This aligns with CLAUDE.md line 161-163: "If a task in PLAN.md is blocking one of them, do the work needed. If something in PLAN.md is not needed for any of the 8, skip it and note that in DECISIONS.md."

## 2026-08-24: Bootstrap granularity in measure command

**Decision:** Use trial-level resampling (resample across trial medians) rather than per-request resampling for bootstrap confidence intervals in `throttle measure`.

**What was tried:** During development of the `measure` command, we considered two approaches for computing bootstrap confidence intervals:

1. **Trial-level resampling (chosen):** Resample from the `--repeat N` trial-level cost estimates. At `--repeat 10`, this resamples 10 median values with replacement 10,000 times.

2. **Per-request resampling (rejected):** Pool all per-request timings from all trials, then resample individual requests to create synthetic trials. At `--repeat 10` with `--num-requests 100`, this would pool 1,000 individual request timings and resample them.

**Empirical result:** Per-request resampling produced intervals that were **35-40% narrower** than trial-level resampling on synthetic data with known variance.

**Why narrower is wrong:**

Requests within a single trial are **not independent**. They share:
- Server state (cache warmth, memory layout, GPU kernel compilation state)
- Batch composition (if the server batches requests)
- Time-of-day effects (system load, thermal throttling)
- Network conditions during that trial's time window

Pooling all requests treats them as exchangeable, which fabricates precision by ignoring this hierarchical structure. The bootstrap assumes observations are independent draws from the same distribution, but per-request pooling violates this assumption.

**Why trial-level resampling is correct:**

Each trial is one independent observation of the configuration's cost under the fixed workload. Between-trial variance captures:
- Server variance across different time windows
- Cache state differences between trial starts
- Any systematic drift or instability in the server

Trial-level resampling correctly treats each trial as the unit of observation, preserving the hierarchical data structure (requests nested within trials).

**What the intervals represent:**

The confidence intervals from trial-level resampling reflect **between-trial variance at a fixed workload**, not variation in traffic mix. This is documented in the measure command output: "Interval reflects server variance at a fixed workload, not variation in traffic mix."

**Alternative considered:** Use per-request resampling to get tighter intervals.

**Why this approach is better:** Correctly reflects the actual uncertainty in the cost estimate. A tight interval that excludes the true value 30% of the time is worse than a wider interval with proper coverage.

**Follow-up change:** Increased `--repeat` default from 5 to 10 trials. Five observations is thin for a 95% confidence interval regardless of bootstrap method. Runtime cost noted in help text (~100 seconds at default arrival rate).
