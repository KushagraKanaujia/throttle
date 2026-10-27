# compare, golden-report

*Offline comparison of saved runs, and the golden HTML report.*

```bash
throttle compare baseline.json candidate.json
throttle golden-report --golden-dir throttle-golden --output golden.html
```

- `compare` ranks two saved reports without network traffic, or six ordered B1 C1 B2 C2 B3 C3 reports for the golden protocol. With `--baseline URL --candidate URL --eval-set FILE` it compares two live endpoints on your own prompts instead.
- `golden-report` turns golden artifacts into a self-contained HTML deliverable.
- To compare two `throttle check` results, use `throttle check --against ID` or `throttle gate`.

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
                        'kush@throttle-pro.com')
```

</details>
