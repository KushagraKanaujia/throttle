# What changed (config fingerprint)

*How Throttle records the serving config behind each check.*

Each check stores a fingerprint of the config it was measured on, so a verdict comes with a list of what changed.

It's built from:

- **The model** and your **label** (`--label fp8-batch256`).
- **Settings the server reports.** For vLLM, Throttle reads config labels from the Prometheus `/metrics` endpoint (default `<host>/metrics`; `--metrics-url` to override, `--no-metrics` to skip; skipped quietly if unreachable).
- **Settings you record** with `--config KEY=VALUE` (repeatable), for things the endpoint can't report: `--config gpu=H100 --config quantization=fp8`.
- **The workload**: prompts hash, requests per block, concurrency, max tokens, prompt cache mode.
- **The GPU rate** (ASSUMED) and the Throttle version.

```bash
throttle check --url http://localhost:8000 --model my-model \
  --gpu-hourly-rate 2.49 --config max_num_seqs=512 --label after
```

"What changed in the config" then lists the differences from the baseline, for example `max_num_seqs: 256 -> 512` or `workload.prompt_cache_mode: cold -> warm`.

> **Note:** `--config` only records what you tell it. Throttle never changes your server.
