# Introduction

*Own your intelligence. Keep the savings.*

![](../illustrations/quickstart.svg)

Throttle is an open-source CLI (MIT) that sits in front of your OpenAI or Anthropic traffic. It proves which share of that traffic can move to an open-weight model you run yourself, without losing quality. It moves that share, with automatic fallback to the API. And it reruns your tests every time a new open model comes out.

We charge a share of the savings we measure: no savings, no bill.

```text
capture ──► workloads ──► shadow ──► eval ──► report ──► route ──► refresh
```

## Start here

- **[Quickstart](getting-started/quickstart.md)**: Run the whole loop on made-up data in a minute, then on your own traffic.
- **[The loop](loop/overview.md)**: One page per stage, with every command and flag.
- **[Honest numbers](loop/honest-numbers.md)**: The labels, and the rules that keep a savings number defensible.
- **[Privacy](more/privacy.md)**: What is stored, what is redacted, and what leaves your machine.
- **[Your first cost check](getting-started/first-check.md)**: Measure what your own model costs per million tokens on your server.
- **[Results](results/overview.md)**: Real runs with the raw numbers.

## What you get

- **See it.** Capture your API traffic locally and see what kinds of work it is and what each kind costs at list price.
- **Prove it.** Your own model answers the same requests in the shadow. A judge, run both ways round, plus automatic checks, gives a verdict per kind of work with 95% confidence intervals.
- **Price it.** `throttle check` measures what your model costs per million tokens on your GPUs. The report says MOVE, KEEP ON API, NEEDS FINETUNE or NEEDS MORE DATA per kind of work.
- **Move it.** Approved kinds of work are served from your model. Anything doubtful falls back to the API.
- **Keep it current.** Rerun the same eval sets on every new open model.

> **Note:** Everything runs on your machine. Nothing is sent anywhere except the API call your app already makes and the endpoints you configure (your model and your judge). No telemetry.
