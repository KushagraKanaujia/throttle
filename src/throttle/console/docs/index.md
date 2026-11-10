# Introduction

*Stop renting your AI.*

![](../illustrations/quickstart.svg)

Throttle cuts your AI bill. It is an open-source CLI (MIT) that sits in front of your OpenAI or Anthropic traffic. It tests each AI job against cheaper models, closed or open, on your own traffic. It moves a job only after it passes and you approve, with automatic fallback to the API. And it retests when a prompt or a model changes.

`throttle route savings` gives you a realized-savings statement you can audit (good for showing your boss).

```text
capture ──► workloads ──► shadow ──► eval ──► report ──► route ──► refresh
```

## Start here

- **[Quickstart](getting-started/quickstart.md)**: Run the whole loop on made-up data in a minute, then on your own traffic.
- **[The loop](loop/overview.md)**: One page per stage, with every command and flag.
- **[Honest numbers](loop/honest-numbers.md)**: The labels, and the rules that keep a savings number defensible.
- **[Privacy and data flow](more/privacy.md)**: What runs where, what is redacted, and what leaves your infra.
- **[Your first cost check](getting-started/first-check.md)**: Measure what your own model costs per million tokens on your server.
- **[Results](results/overview.md)**: Real runs with the raw numbers.

## What you get

- **See it.** Capture your API traffic locally and see what kinds of work it is and what each kind costs at list price.
- **Prove it.** Your own model answers the same requests in the shadow. A judge, run both ways round, plus automatic checks, gives a verdict per kind of work with 95% confidence intervals.
- **Price it.** `throttle check` measures what your model costs per million tokens on your GPUs. The report says MOVE, KEEP ON API, NEEDS FINETUNE or NEEDS MORE DATA per kind of work.
- **Move it.** Approved kinds of work are served from your model. Anything doubtful falls back to the API.
- **Keep it current.** Rerun the same eval sets on every new open model.

> **Note:** Everything runs on your infra: a proxy you run, your CI and the CLI. Nothing is sent anywhere except the API call your app already makes and the endpoints you configure (your candidate model and your judge). Throttle the company receives nothing. No telemetry.
