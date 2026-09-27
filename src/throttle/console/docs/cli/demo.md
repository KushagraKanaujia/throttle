# throttle demo

*A simulated baseline vs tuned comparison. No GPU, no network.*

```bash
throttle demo
throttle demo --light-load
```

Generates a sample workload, simulates continuous batching for a baseline and a tuned configuration, and prints a side-by-side cost comparison. Every number is labelled SIMULATED and depends on assumed throughput parameters.

`--light-load` uses a workload that saturates neither configuration, to show the "no significant difference" path.

> **Note:** The demo shows the output format. For real numbers, run `throttle cost` or `throttle check` against an endpoint.

<details><summary>Full `throttle demo --help` output (0.4.2)</summary>

```text
usage: throttle demo [-h] [--light-load]

Generate a sample workload, simulate vLLM-like continuous batching inference
for two configurations (baseline and tuned), and print a side-by-side cost
comparison. Runs entirely locally with no GPU or network required. Completes
in under 5 minutes.

options:
  -h, --help    show this help message and exit
  --light-load  use a light workload that does not saturate either
                configuration (demonstrates NO SIGNIFICANT DIFFERENCE path)
```

</details>
