# Honest numbers

*How Throttle keeps every number it prints defensible.*

## Every number has a label

| Label | Means |
| --- | --- |
| MEASURED | Token counts and times we saw, and $/M tokens from a `throttle check` with a 95% interval |
| LIST PRICE | A published API price, with the date it was checked and its source page |
| ASSUMED | A number you gave us: your GPU rate, or a price from `--prices` |
| DEMO | Made up, from `throttle demo` |
| OPERATOR-REPORTED | Your invoice, from `route savings --owned-cost-usd` |

## Rules

- **Missing usage is excluded, never estimated.** A call with no token counts is left out of cost math and counted as excluded.
- **Quality needs enough evidence.** The judge runs both ways round. A position-biased judge gives INSUFFICIENT DATA. NO MATERIAL DIFFERENCE needs the loss rate shown to be at most 5%, which takes about 75 loss-free items. Small samples say NEEDS MORE DATA. See [Eval](eval.md).
- **Savings only from measured runs.** The report shows savings only with a measured `throttle check`, using the expensive end of its interval. The bill comes only from requests your model actually served.
- **Fallbacks are not savings.** They are already on your API bill.
- **Forced routes are not billable.**
- **Prompt caching is counted against us.** Avoided API cost uses the cached rate for the share of input the API cached, and assumes all input was cached when there is no history.
- **An always-on GPU is paid for every hour,** busy or not.
