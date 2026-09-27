# throttle watch

*Read vLLM /metrics and report cost per million tokens without sending requests.*

```bash
throttle watch --metrics-url http://localhost:8000/metrics --gpu-rate-per-hour 1.39
```

Passively scrapes vLLM's Prometheus `/metrics` and turns generation throughput into $/M tokens. Nothing enters the request path. It refuses to print a cost figure when throughput isn't available.

| Option | Default | Meaning |
| --- | --- | --- |
| `--metrics-url` | `http://localhost:8000/metrics` | vLLM metrics endpoint |
| `--gpu-rate-per-hour` | required | GPU $/hr |
| `--interval` | 15 | Scrape interval in seconds |
| `--max-num-seqs` | | For batch-fill computation |
| `--json` | off | Raw JSON snapshots |

<details><summary>Full `throttle watch --help` output (0.4.2)</summary>

```text
usage: throttle watch [-h] [--metrics-url URL] --gpu-rate-per-hour DOLLARS
                      [--interval SECONDS] [--max-num-seqs N] [--json]

Passively reads vLLM /metrics (Prometheus text format) and translates
throughput into dollars per million tokens. Nothing enters the request path.
Requires --gpu-rate-per-hour. Refuses to print a cost figure when generation
throughput is unavailable.

options:
  -h, --help            show this help message and exit
  --metrics-url URL     vLLM /metrics endpoint (default:
                        http://localhost:8000/metrics)
  --gpu-rate-per-hour, --gpu-hourly-rate, --total-hourly-price DOLLARS
                        GPU cost in $/hr. Required — no default is honest.
  --interval SECONDS    scrape interval in seconds (default: 15)
  --max-num-seqs N      vLLM max_num_seqs for batch fill computation
                        (optional)
  --json                emit raw JSON snapshots instead of formatted text
```

</details>
