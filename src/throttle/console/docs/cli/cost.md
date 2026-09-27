# throttle cost

*Measure cost per million tokens once against a live endpoint.*

```bash
throttle cost --url http://localhost:11434 --model llama3.2:3b --gpu-hourly-rate 1.50
```

Sends a small workload (20 requests by default), measures throughput and timing, and prints input and output $/M tokens. Use `throttle check` instead when you want history and a verdict.

| Option | Default | Meaning |
| --- | --- | --- |
| `--endpoint-url` / `--url` | | Server URL, with or without `/v1` |
| `--model` | the server's only model | Model to request |
| `--gpu-hourly-rate` | | GPU $/hr (ASSUMED) |
| `--num-requests` | 20 | Test requests to send |
| `--api-key` | `OPENAI_API_KEY` | Bearer key |
| `--warm-cache` | off | Send prompts unchanged instead of cold-tagged |

<details><summary>Full `throttle cost --help` output (0.4.2)</summary>

```text
usage: throttle cost [-h] --endpoint-url ENDPOINT_URL [--model MODEL]
                     --gpu-hourly-rate GPU_HOURLY_RATE
                     [--num-requests NUM_REQUESTS] [--api-key API_KEY]
                     [--warm-cache]

Send a small workload to an OpenAI-compatible endpoint, measure actual
throughput and timing, and calculate dollars per million tokens. Requires a
running inference server.

options:
  -h, --help            show this help message and exit
  --endpoint-url, --url ENDPOINT_URL
                        inference server URL: http://host:port or
                        http://host:port/v1 (--url is accepted too)
  --model MODEL         model name to request (default: the server's only
                        model, read from /v1/models; required when it serves
                        several)
  --gpu-hourly-rate, --gpu-rate-per-hour, --total-hourly-price GPU_HOURLY_RATE
                        GPU hourly rate in dollars (e.g., 1.50 for A100 spot
                        pricing)
  --num-requests NUM_REQUESTS
                        number of test requests to send (default: 20)
  --api-key API_KEY     API key for authentication (also reads OPENAI_API_KEY
                        env var)
  --warm-cache          send the synthetic prompts unchanged (they share long
                        prefixes, so a prefix cache serves them warm). By
                        default each prompt starts with a unique tag such as
                        '[run 482915 req 0012] ' so a prefix cache cannot
                        serve it
```

</details>
