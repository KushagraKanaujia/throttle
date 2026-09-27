# Introduction

*Know what your LLM inference costs per million tokens, and whether your last serving change made it cheaper.*

![](../illustrations/quickstart.svg)

Throttle is an open-source CLI (MIT) that measures what a self-hosted model costs you in **dollars per million tokens**, on any OpenAI-compatible endpoint: vLLM, SGLang, Ollama, LMDeploy and similar servers.

Run it after every serving change (an engine flag, a batch size, a quantization, a GPU, an engine version). It measures again, compares with the last check, and gives one of four verdicts. It will not call a winner that is inside your machine's own run-to-run noise.

```text
change a serving setting ──► throttle check ──► $/M tokens (95% CI) + what changed + verdict
        ▲                                                                   │
        └──────────── keep the change, or roll it back ◄────────────────────┘
```

## Start here

- **[Quickstart](getting-started/quickstart.md)**: A real cost-per-million-tokens number in about two minutes.
- **[Installation](getting-started/installation.md)**: Install the CLI with pipx and check your setup.
- **[Noise and calibration](concepts/noise-and-calibration.md)**: Why a single before/after run can lie, and how Throttle avoids it.
- **[Gate costs in CI](guides/ci-github-action.md)**: Fail a deploy that makes every token more expensive.
- **[Compare GPUs](guides/compare-gpus.md)**: Measure the same model on two machines with the same workload.
- **[Results](results/overview.md)**: Real runs with the raw numbers, including an A100 counterbalanced test.

## What you get

- **Price it.** Dollars per million tokens from a live endpoint. The GPU hourly rate is yours and always labelled ASSUMED; tokens and time are MEASURED.
- **Re-check after every change.** A 95% confidence interval, a fingerprint of what changed in the config, and a verdict against the last check.
- **Refuse to lie.** CHEAPER or MORE EXPENSIVE only when the change is larger than a run-to-run noise bound measured from repeat checks and the confidence intervals don't overlap. Otherwise NO WINNER, or NOT CALIBRATED until the noise has been measured.
- **Gate it in CI.** `--fail-if-costlier` turns a costlier deploy into a failed job.

> **Note:** Throttle provisions nothing and never changes your server. It sends a small, capped workload to the endpoint you point it at, measures, and reports.
