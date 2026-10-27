# 4. Eval

*Is your model as good as the API model, per kind of work?*

```bash
export OPENAI_API_KEY=...
throttle eval --candidate qwen72b@vllm --judge-model gpt-4.1 --judge-key-env OPENAI_API_KEY
```

Eval pairs each captured call with your model's shadow answer and compares them two ways.

## Automatic checks

Scored the same way for both answers, as a pass rate with a 95% Wilson interval:

| Check | Passes when |
| --- | --- |
| `json_valid` | The request asked for JSON and the answer parses |
| `schema_valid` | The request gave a JSON schema and the answer fits it |
| `exact_match` | Classification only: the answer equals the API model's |
| `tool_calls_valid` | Every tool call names an offered tool and has JSON object arguments |
| `truncated` | The answer was not cut off by the token limit |

## The judge

The judge sees the conversation and both answers, and is asked twice with the answers swapped. It is a win or a loss only when both orders agree, and a tie only when both say tie. Anything else is "order disagree": it counts like a tie but is tracked on its own.

## Verdict per kind of work

In order:

1. **INSUFFICIENT DATA** if there are fewer than `--min-n` judged items (default 30), more than 20% judge errors, or more than 30% order disagreements (the judge is picking a position, not an answer).
2. **WORSE** if the 95% interval for win minus loss is entirely below minus the margin (default 5 points), or a check pass rate is clearly below the API model's.
3. **INCONCLUSIVE** if more than 10% of the answers are missing because of timeouts. They could be losses.
4. **BETTER** if the low end of the win rate beats the high end of the loss rate by more than the margin.
5. **NO MATERIAL DIFFERENCE** if the win-minus-loss interval sits inside plus or minus the margin AND the loss rate is shown to be at most the margin. That needs about 75 loss-free items, so a small sample of all ties is INCONCLUSIVE.
6. **INCONCLUSIVE** otherwise. Get more samples.

`--no-judge` runs the automatic checks only, and the verdict is CHECKS ONLY.

## Options

| Flag | Default | What it does |
| --- | --- | --- |
| `--candidate` | required | The label from `throttle shadow` |
| `--judge-model` | required unless `--no-judge` | Judge model name |
| `--judge-provider openai\|anthropic` | `openai` | Judge API format |
| `--judge-url` | the provider's public API | A judge you host |
| `--judge-key-env VAR` | none | Environment variable with the judge key. Never printed. |
| `--max-judge-calls` | 600 | Hard cap on judge calls |
| `--judge-max-usd` | none | Hard cap on judge spend, when its price is known |
| `--judge-concurrency` | 4 | Judge calls in flight |
| `--margin` | 0.05 | What counts as a material difference |
| `--min-n` | 30 | Minimum judged items per kind |
| `--json` | off | Print the result JSON |

> **Note:** The judge sees the captured (redacted) prompts. If it is a public API, they go to that provider. Use the provider you already call, or a judge you host.

Next: [Report](report.md).
