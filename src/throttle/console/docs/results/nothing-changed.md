# Nothing changed: +27%, then −11%

*Four checks of an untouched server, and why the verdict is NO WINNER.*

Local Ollama, `llama3.2:3b`, MacBook, GPU rate assumed at $1.50/hr, Throttle 0.4.0. Same prompts, same settings, four checks back to back.

| Run | $/M output tokens | 95% CI | vs previous | Verdict |
| --- | ---: | --- | ---: | --- |
| 1 | $8.77 | $8.19 to $9.35 | | first check |
| 2 | $8.86 | $7.80 to $9.93 | +1.0% | NOT CALIBRATED |
| 3 | $11.26 | $10.31 to $12.21 | +27.1% | NOT CALIBRATED |
| 4 | $10.02 | $7.41 to $12.62 | −11.0% | **NO WINNER** |

By run 4 there were enough repeats to measure noise: run-to-run SD 14.6% with 2 degrees of freedom, so the bound is ±89.1%. A −11.0% change is well inside it.

A single before/after comparison would have reported a 27% regression and then an 11% saving, with nothing changed. That's the failure Throttle is built to prevent.

- **[How the noise bound works](../concepts/noise-and-calibration.md)**
