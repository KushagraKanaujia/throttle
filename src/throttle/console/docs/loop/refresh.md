# 7. Refresh

*When a new open model ships, test it on the same saved eval sets.*

```bash
throttle refresh --current qwen72b@vllm --new-label qwen3-72b@vllm \
    --new-url http://gpu-box:8001/v1 --new-model Qwen/Qwen3-72B \
    --judge-model gpt-4.1 --judge-key-env OPENAI_API_KEY
```

1. For each kind of work with a saved eval set for `--current`, the same requests are sent to the new model.
2. The new model is evaluated against the API model, like [Eval](eval.md).
3. The new model is judged directly against your current model on the same items, both orders, with the same verdict rules.

## Verdict per kind

| Verdict | When |
| --- | --- |
| **UPGRADE** | NO MATERIAL DIFFERENCE or BETTER against the API model, and against your current model. Not worse has to be shown; INCONCLUSIVE is not enough. |
| **STAY** | WORSE than the API model or than your current model |
| **NEEDS MORE DATA** | Anything else |

Refresh never measures cost. Run [`throttle check`](../cli/check.md) on the new model before you switch.

## Options

`--types` limits it to some kinds (comma separated). `--max-judge-calls` (default 1200) and `--judge-max-usd` cap the judge. `--margin` and `--min-n` work as in eval. `--json` prints the result.

`throttle refresh --watch-hf ORG/MODEL` only checks Hugging Face for a newer revision or a sibling model, then exits.
