# Cost per million tokens

*How Throttle turns GPU hours into dollars per million tokens.*

You pay for GPUs by the hour, but you budget, price and compare in **dollars per million tokens**. The conversion between the two is throughput, and throughput depends on the serving config: batch size, `max_num_seqs`, quantization, engine version, GPU type.

## The formula

```text
$ per M tokens = ($ per GPU-hour ÷ 3600) ÷ tokens per second × 1,000,000
```

Example with a real A100 measurement: $1.39/hr at 516.55 output tokens/s is

```text
(1.39 ÷ 3600) ÷ 516.55 × 1,000,000 = $0.7475 per million output tokens
```

## What's measured and what's assumed

| Quantity | Source | Label |
| --- | --- | --- |
| Tokens (input, output) | the endpoint's `usage` block | MEASURED |
| Wall-clock time | Throttle's clock around the workload | MEASURED |
| GPU $/hr | you (`--gpu-hourly-rate`) | ASSUMED |
| Monthly dollars | measured $/M × your `--monthly-tokens` | PROJECTED |

If the endpoint doesn't report `usage`, the check fails instead of guessing.

## Which tokens

`--metric` picks the headline: `output` (default), `input` or `total`. The same GPU time pays for all of them, so the figures aren't additive.

> **Tip:** Include every GPU the model uses in the hourly rate. For a model on 2× H100 at $2.49 each, pass `--gpu-hourly-rate 4.98`.

> **Warning:** Cost per token depends on load. Measure at the concurrency you actually run in production (`--concurrency`), or the number won't describe your bill.
