# 3. Shadow

*Replay a sample of captured requests on your own open-weight model. Its answers are never served.*

Start your model on any OpenAI-compatible server (vLLM, SGLang, Ollama, LMDeploy), then:

```bash
throttle shadow --candidate-url http://gpu-box:8000/v1 \
    --candidate-model Qwen/Qwen2.5-72B-Instruct --label qwen72b@vllm
```

- It samples captured calls by kind of work, so each kind gets its share.
- Anthropic requests are translated to the OpenAI chat format first.
- Answers go to `~/.throttle/own/shadow/<label>.jsonl` for [Eval](eval.md). They are never returned to a client and never written into the captured call.
- A timeout or connection failure is retried once. If it fails again, it counts as a loss in eval. Context overflows, HTTP errors and empty answers also count as losses.
- Running it again skips calls that already have an answer for that label.

## Options

| Flag | Default | What it does |
| --- | --- | --- |
| `--candidate-url` | required | Your server, e.g. `http://gpu-box:8000/v1` |
| `--candidate-model` | required | Model name the server expects |
| `--label` | required | Name for this candidate. Eval, report and route use it. |
| `--api-key-env VAR` | none | Environment variable with the server's key, if it needs one |
| `--per-type` | 50 | Requests per kind of work |
| `--max-requests` | 500 | Total cap |
| `--concurrency` | 4 | Requests in flight |
| `--timeout` | 120 | Seconds per request |
| `--since` | all | Only calls newer than this, e.g. `7d` |
| `--seed` | 0 | Sampling seed |

Next: [Eval](eval.md).
