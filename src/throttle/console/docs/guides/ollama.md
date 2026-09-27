# Ollama

*Measure local models served by Ollama.*

Ollama listens on port 11434 and serves an OpenAI-compatible API. Throttle accepts the base URL with or without `/v1`.

```bash
ollama pull llama3.2:3b
throttle check --url http://localhost:11434 --model llama3.2:3b --gpu-hourly-rate 1.50
```

To test a setting, restart Ollama with it and record it:

```bash
OLLAMA_NUM_PARALLEL=4 ollama serve
throttle check --url http://localhost:11434 --model llama3.2:3b \
  --gpu-hourly-rate 1.50 --label after --config OLLAMA_NUM_PARALLEL=4
```

> **Note:** A laptop has no GPU invoice, so the hourly rate is an assumption. Use relative changes; tokens and time are still measured. Laptops are also noisy (other apps, thermals), so expect a wider noise bound than on a dedicated server.
