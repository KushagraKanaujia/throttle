# Noise and calibration

*Why a single before/after run can lie, and how Throttle measures run-to-run noise.*

![](../../illustrations/noise.svg)

## Nothing changed, and it looked 27% more expensive

Four checks of the same unchanged Ollama server (`llama3.2:3b`, MacBook, GPU rate assumed at $1.50/hr):

| Run | $/M output tokens | Change vs previous |
| --- | ---: | ---: |
| 1 | $8.77 | |
| 2 | $8.86 | +1.0% |
| 3 | $11.26 | **+27.1%** |
| 4 | $10.02 | **−11.0%** |

If a setting had changed between runs 2 and 3, it would have taken the blame. Nothing changed.

## Why a 95% interval isn't enough

A check's 95% confidence interval is computed **across the blocks inside one run**. It can't see what changes *between* runs: machine load, thermal state, other tenants on the GPU. Two checks taken at different times aren't a controlled comparison.

## The noise bound

Throttle measures that drift from repeat checks of an unchanged config:

```text
noise bound = t(0.975, df) × √2 × run-to-run SD
```

- The run-to-run SD is the pooled relative SD of repeat checks' $/M, from the two configs being compared.
- It needs at least **2 degrees of freedom**: three earlier checks of one unchanged config, or two each of the before and after configs.
- Only checks from the **last 24 hours** count. An older baseline is NOT CALIBRATED.

In the example above, run 4 against run 3: SD 14.6% with 2 df gives a bound of ±89.1%. The −11.0% is well inside it, so the verdict is **NO WINNER**. (A laptop is noisy; a dedicated server usually has a much tighter bound.)

## A real change does separate

Same laptop, `llama3.2:3b` → `llama3.2:1b`: $8.71 → $5.47 per million output tokens, **−37.2%**, outside a 24.5% noise bound (3 df). Verdict: **CHEAPER**. (Whether the smaller model is good enough is your call; Throttle measures cost, not quality.)

- **[The four verdicts](verdicts.md)**: Exactly when Throttle says CHEAPER, MORE EXPENSIVE, NO WINNER or NOT CALIBRATED.
