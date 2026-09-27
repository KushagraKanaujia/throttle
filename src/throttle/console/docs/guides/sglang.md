# SGLang

*Measure the cost of an SGLang server.*

SGLang serves an OpenAI-compatible API (port 30000 by default). Point Throttle at it like any other endpoint:

```bash
throttle check --url http://localhost:30000 --model <model-name> \
  --gpu-hourly-rate 2.49 --config gpu=H100 --label baseline
```

Run it three times unchanged to calibrate, change one server argument, restart, and check again with a new `--label` and the changed setting in `--config`.

- `/metrics` is vLLM-specific; for SGLang, record the settings you care about with `--config KEY=VALUE`. Add `--no-metrics` to skip the metrics lookup entirely.
- SGLang's RadixAttention prefix cache is on unless you disable it. Throttle's default cold-cache prompts keep it from flattering the number.

> **Note:** SGLang was measured end to end on RunPod GPUs as part of Throttle's cross-engine validation (descriptive, not decision-grade).
