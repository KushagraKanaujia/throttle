# Cold vs warm cache

*Why checks are cold-cache by default, and when to use --warm-cache.*

Servers with prefix caching (vLLM V1 `--enable-prefix-caching`, on by default; SGLang RadixAttention; Ollama KV reuse) can serve repeated prompts from cache. If a benchmark resends identical prompts, $/M looks cheaper than real traffic.

## Default: cold

Each measured request's first user message starts with a unique tag such as `[run 482915 req 0012] `, from a random per-check run id stored with the record. A prefix cache can't serve repeats. The tag's own prompt tokens are measured separately and not counted as input tokens.

## `--warm-cache`

Resends identical prompts every block, to measure cache-friendly traffic on purpose.

```bash
throttle check --url http://localhost:8000 --model my-model \
  --gpu-hourly-rate 2.49 --warm-cache --label warm
```

> **Warning:** The cache mode is part of the workload identity. A cold check and a warm check are never compared with each other; the result is NO WINNER ("the prompt cache mode changed").

`throttle cost` and `throttle measure` tag prompts the same way and accept `--warm-cache` too.
