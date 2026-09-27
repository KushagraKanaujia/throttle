# vLLM

*Measure and gate the cost of a vLLM deployment.*

vLLM serves an OpenAI-compatible API (port 8000 by default) and exposes Prometheus metrics at `/metrics`, which Throttle uses for the config fingerprint.

### 1. Start vLLM as you normally do

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-num-seqs 8
```

### 2. Record the current config (three times)

```bash
throttle check --url http://localhost:8000 --model Qwen/Qwen2.5-0.5B-Instruct \
--gpu-hourly-rate 1.39 --config gpu=A100-80GB --label seqs-8 --concurrency 8
```

### 3. Change one flag, restart, check again

```bash
throttle check --url http://localhost:8000 --model Qwen/Qwen2.5-0.5B-Instruct \
--gpu-hourly-rate 1.39 --config gpu=A100-80GB --label seqs-16 --concurrency 8
```

## Passive monitoring

`throttle watch` reads vLLM's `/metrics` and translates generation throughput into $/M tokens without sending any requests:

```bash
throttle watch --metrics-url http://localhost:8000/metrics --gpu-rate-per-hour 1.39
```

It refuses to print a cost figure when generation throughput isn't available.

> **Tip:** vLLM's prefix caching is on by default in recent releases. `throttle check` is cold-cache by default, so cached prefixes don't make a config look cheaper than your real traffic. See [Cold vs warm cache](../concepts/cold-vs-warm-cache.md).

- **[Real result on an A100](../results/a100-max-num-seqs.md)**: `max_num_seqs` 1 → 8 on vLLM 0.16.0: $0.746 → $0.234 per million output tokens.
