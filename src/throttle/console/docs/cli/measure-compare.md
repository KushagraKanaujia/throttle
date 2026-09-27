# measure, compare, report

*Repeated measurements to JSON, and offline comparison of saved runs.*

```bash
throttle measure --url http://localhost:8000 --model my-model --gpu-hourly-rate 2.49 --label baseline
throttle measure --url http://localhost:8000 --model my-model --gpu-hourly-rate 2.49 --label candidate
throttle compare baseline.json candidate.json
throttle report --out report.html baseline.json candidate.json
```

- `measure` repeats a workload (10 trials by default) and saves JSON named after `--label`.
- `compare` ranks two saved reports without network traffic, or six ordered B1 C1 B2 C2 B3 C3 reports for the golden protocol. It refuses to rank cold-cache against warm-cache files.
- `report` renders an HTML chart comparing two `measure` outputs.
- `golden-report` turns golden artifacts into a self-contained HTML deliverable.

<details><summary>Full `throttle measure --help` output (0.4.2)</summary>

```text
usage: throttle measure [-h] --endpoint-url ENDPOINT_URL [--model MODEL]
                        --gpu-hourly-rate GPU_HOURLY_RATE --label LABEL
                        [--repeat REPEAT] [--arrival-rate ARRIVAL_RATE]
                        [--num-requests NUM_REQUESTS] [--note NOTE]
                        [--warm-cache] [--api-key API_KEY]

Run repeated measurements against a live endpoint to measure cost per million
tokens with confidence intervals. Results are saved to JSON for later
comparison. The operator is responsible for server configuration - Throttle
does not restart or reconfigure the server.

options:
  -h, --help            show this help message and exit
  --endpoint-url, --url ENDPOINT_URL
                        inference server URL: http://host:port or
                        http://host:port/v1 (--url is accepted too)
  --model MODEL         model name (default: the server's only model, read
                        from /v1/models; required when it serves several)
  --gpu-hourly-rate, --gpu-rate-per-hour, --total-hourly-price GPU_HOURLY_RATE
                        GPU hourly rate in dollars
  --label LABEL         label for this measurement (used as output filename)
  --repeat REPEAT       number of times to repeat the workload (default: 10,
                        runtime ~100 seconds at default arrival rate)
  --arrival-rate ARRIVAL_RATE
                        arrival rate in requests per second (default: 10.0)
  --num-requests NUM_REQUESTS
                        number of requests per trial (default: 100)
  --note NOTE           optional note about server configuration
  --warm-cache          resend identical prompts in every trial (a prefix
                        cache serves them warm). By default each prompt starts
                        with a unique tag such as '[run 482915 req 0012] ', so
                        trials measure uncached prefill. 'throttle compare'
                        will not rank cold against warm files
  --api-key API_KEY     API key for authentication (also reads OPENAI_API_KEY
                        env var)
```

</details>

<details><summary>Full `throttle compare --help` output (0.4.2)</summary>

```text
usage: throttle compare [-h] [--output OUTPUT] reports [reports ...]

positional arguments:
  reports          two saved reports, or six ordered B1 C1 B2 C2 B3 C3 reports
                   for the golden protocol

options:
  -h, --help       show this help message and exit
  --output OUTPUT
```

</details>

<details><summary>Full `throttle report --help` output (0.4.2)</summary>

```text
usage: throttle report [-h] --out OUT reports reports

positional arguments:
  reports     two measure output JSON files to compare

options:
  -h, --help  show this help message and exit
  --out OUT   output HTML file path
```

</details>

<details><summary>Full `throttle golden-report --help` output (0.4.2)</summary>

```text
usage: throttle golden-report [-h] --golden-dir GOLDEN_DIR --output OUTPUT
                              [--gpu-hourly-rate GPU_HOURLY_RATE]
                              [--operator-name OPERATOR_NAME]
                              [--operator-email OPERATOR_EMAIL]

Transform golden protocol artifacts (golden.json + position reports) into a
self-contained HTML deliverable answering: should you change this setting
(yes/no), what it's worth (throughput + cost), what was tested, why believe
it, and what this doesn't prove. Single file, no external assets, opens
offline.

options:
  -h, --help            show this help message and exit
  --golden-dir GOLDEN_DIR
                        directory containing golden.json and position reports
                        (B1.json, C1.json, etc.)
  --output OUTPUT       output HTML file path
  --gpu-hourly-rate GPU_HOURLY_RATE
                        GPU hourly rate in dollars for cost calculation (e.g.,
                        1.39 for A100 80GB)
  --operator-name OPERATOR_NAME
                        operator name for report footer (default: 'Throttle')
  --operator-email OPERATOR_EMAIL
                        operator email for report footer (default:
                        'kushthrottle@gmail.com')
```

</details>
