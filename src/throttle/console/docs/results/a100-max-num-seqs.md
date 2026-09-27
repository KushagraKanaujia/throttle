# One vLLM flag on an A100

*max_num_seqs 1 → 8: $0.746 → $0.234 per million output tokens, six-position counterbalanced.*

| | |
| --- | --- |
| Hardware | A100 PCIe 80GB (RunPod), $1.39/hr |
| Engine | vLLM 0.16.0 |
| Model | Qwen2.5-0.5B-Instruct |
| Load | 8 concurrent requests |
| Change | `max_num_seqs` 1 → 8 |
| Protocol | six positions, B1 C1 B2 C2 B3 C3 (counterbalanced) |
| Requests | 1,206 valid measured requests, 0 errors |

## Per position

| Position | `max_num_seqs` | Output tok/s (block mean) | $/M output tokens |
| --- | ---: | ---: | ---: |
| B1 | 1 | 516.55 | $0.7475 |
| C1 | 8 | 1,632.41 | $0.2371 |
| B2 | 1 | 517.89 | $0.7456 |
| C2 | 8 | 1,688.40 | $0.2288 |
| B3 | 1 | 518.55 | $0.7446 |
| C3 | 8 | 1,631.16 | $0.2367 |

Mean of the three positions each: **$0.746 → $0.234 per million output tokens (−68.6%)**. Throughput: **+189.5% to +246.2%** (95% CI, point estimate +217.8%). `decision_eligible: true`.

> **Note:** The same files also record a whole-run cost summary measured over a slightly different wall-time window: $0.778 → $0.261 (−66.5%). Both are in the saved artifacts.

## Why counterbalanced

Warm-up, thermals and drift change a GPU's speed over a session. Alternating the order (B C B, then C B C) spreads that drift across both configs instead of letting it favour whichever ran last.

> **Warning:** `max_num_seqs=1` is a deliberately bad baseline on a small model. This result shows the measurement method; it isn't a saving to expect on your setup.

Artifacts: [validation/golden-live-20260817](https://github.com/KushagraKanaujia/throttle/tree/main/validation/golden-live-20260817)
