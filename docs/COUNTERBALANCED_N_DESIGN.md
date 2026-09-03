# Generalizing Counterbalanced Ordering to N-Conditions (Williams Cross-Over Design)

## 1. The Core Limitation of Two-Arm Phase Contrast
The current golden protocol uses an alternating design (`B1/C1/B2/C2/B3/C3`) to compare exactly two arms (A vs B). This cancels out linear temporal drift (thermal throttling, GPU heat-up, and cache warming) by balancing the positional sequence.

However, simple pairwise alternation cannot generalize to $N > 2$ conditions. For instance, with 4 conditions (A/B/C/D), running a naive sequential sweep (`A/B/C/D/A/B/C/D...`) places condition A permanently at position 1 and 5, while condition D is stuck at position 4 and 8. Positional bias and cumulative thermal drift will completely invalidate the comparison.

---

## 2. The Solution: Williams Latin Squares (1949)
To sweep multiple configurations (e.g., evaluating 4 different concurrency levels or 3 parameter settings), we must transition to a **Williams Latin Square Design**. 

A Williams Square is a special class of Latin Square where:
1. Every condition appears exactly once in each row (block).
2. Every condition appears exactly once in each column (temporal position).
3. **Every condition precedes and follows every other condition exactly the same number of times** (balancing first-order carryover effects, such as residual GPU memory fragmentation or heat).

### Design Dimensions:
* For **even** $N$, a Williams Square requires exactly **$N$ sequences** ($N^2$ total runs).
* For **odd** $N$, it requires exactly **$2N$ sequences** ($2N^2$ total runs).

---

## 3. Runtime Complexity & Cloud Cost Analysis
Running multi-variable sweeps is an expensive statistical exercise. While $N=4$ is highly efficient due to its even structure, odd numbers like $N=5$ require a doubling of sequences to satisfy the balance constraint.

*Assumptions: Standard benchmark configuration running ~5 minutes per position/run (warmup, traffic run, and cooldown).*

| N (Conditions) | Active Sequences | Runs per Seq | Total Runs ($N^2$ or $2N^2$) | Wall Clock Time | Estimated A100 Cost ($1.39/hr) | Viable? |
|---|---|---|---|---|---|---|
| **2** (Current) | 3 | 2 | **6** | ~30 mins | ~$0.70 | **Yes (Current standard)** |
| **3** | 6 | 3 | **18** | ~90 mins | ~$2.09 | **Yes (Williams Odd)** |
| **4** | 4 | 4 | **16** | ~80 mins | ~$1.85 | **Yes (Sweet Spot — Williams Even)** |
| **5** | 10 | 5 | **50** | ~250 mins (4.1h) | ~$5.79 | **No (Too expensive / unstable)** |
| **6** | 6 | 6 | **36** | ~180 mins (3h) | ~$4.17 | **Marginal (Advisory limit)** |

### Architectural Cap:
Throttle will **hard-cap N-condition sweeps at $N \le 4$**. 
Sweeping more than 4 parameters simultaneously on single-tenant GPU runs is economically prohibitive and vulnerable to non-linear, long-term thermal drift that no Latin Square can statistically correct. If $N > 4$, the tool will refuse to run and suggest breaking the experiment into smaller, disjoint sub-sweeps.

---

## 4. Transitioning the Statistical Engine
Because conditions are no longer strictly paired, the traditional paired Welch's t-test must be replaced with a model capable of decomposing positional variance:

1. **The Model:** A **Linear Mixed-Effects (LME)** model (or a repeated-measures ANOVA) where:
   $$\text{Metric} = \beta_0 + \beta_1(\text{Condition}) + \gamma(\text{Sequence}) + \delta(\text{Position}) + \epsilon$$
   * **Fixed Effect:** The Condition (the parameters being benchmarked).
   * **Random Effects:** Sequence block and Temporal Position (accounting for hardware drift and warmup bias).
2. **The Decision Gate:** An omnibus **F-test** determines if a statistically significant difference exists between *any* of the conditions ($p < 0.05$).
3. **Ranking (Post-hoc):** If the F-test passes, we run a pairwise **Tukey Honest Significant Difference (HSD)** test with family-wise error rate control to determine the exact ranking (e.g., $C > A = B > D$ at $95\%$ confidence).

---

## 5. Mid-Run Failure Recovery & Checkpointing
With $N=4$ requiring 16 runs, the probability of a network glitch, OOM error, or timeout interrupting a run rises significantly. Aborting the entire 80-minute run due to a single position failure makes the feature practically unusable.

### Implementation:
1. **State Persistence (Checkpoint-per-Position):**
   * After every single position run completes, the benchmark state and raw CSV/JSON metrics are atomically persisted to a temporary checkpoint directory: `.throttle/checkpoints/<session_id>/pos_<idx>.json`.
2. **Resume Mechanics:**
   * If interrupted, running the same command detects the `.throttle/checkpoints/<session_id>` directory.
   * Throttle will prompt: `⚠️ Interrupted run detected at position 9/16. Resume from checkpoint? [Y/n]`.
   * On resume, it loads completed runs, warms up the GPU for 60s, and starts executing position 9.
3. **Imbalanced Data Handling (The "Graceful Degradation" Rule):**
   * If a run cannot be resumed and has missing positions (e.g., 15 out of 16 completed):
     * **Do NOT throw away the data.** 
     * The LME mixed model natively handles missing data cells (unlike classical RM-ANOVA which requires perfectly balanced matrices).
     * Throttle will compute the metrics but flag the output as `INCONCLUSIVE (Imbalanced Design)` and restrict confidence ratings to `MEDIUM`, warning the user that some positional/drift bias could not be mathematically canceled.
     * If more than 2 positions are missing, the run is rejected outright.
