# CLI overview

*Every Throttle command at a glance.*

```bash
throttle [--version] COMMAND ...
```

## Everyday commands

| Command | What it does | Sends traffic? |
| --- | --- | --- |
| [`check`](check.md) | After a serving change: measure $/M tokens, compare with the last check, give a verdict | Yes (small, capped) |
| [`cost`](cost.md) | Measure $/M tokens once against a live endpoint | Yes |
| [`watch`](watch.md) | Read vLLM `/metrics` and report $/M tokens | No |
| [`demo`](demo.md) | Simulated baseline vs tuned comparison | No |

## Evidence pipeline

| Command | What it does |
| --- | --- |
| [`plan`](evidence-pipeline.md) | Show traffic, cost, destination and privacy without sending traffic |
| [`smoke`](evidence-pipeline.md) | A short, explicitly non-decision-grade check |
| [`benchmark`](evidence-pipeline.md) | Sustained exploratory evidence blocks |
| [`golden`](evidence-pipeline.md) | The six-position counterbalanced decision protocol |
| [`diagnose`](evidence-pipeline.md) | Pre-flight bottleneck classification (not decision-grade) |
| [`experimental-tuning`](evidence-pipeline.md) | Opt-in, suggestion-only server-metrics analysis |

## Saved runs and reports

| Command | What it does |
| --- | --- |
| [`measure`](measure-compare.md) | Repeated measurements to JSON |
| [`compare`](measure-compare.md) | Compare two saved reports (or six golden positions) without traffic |
| [`report`](measure-compare.md) | HTML report comparing two `measure` outputs |
| [`golden-report`](measure-compare.md) | Self-contained HTML from golden artifacts |

## Proxy and agents

| Command | What it does |
| --- | --- |
| [`proxy`](proxy-sessions.md) | OpenAI-compatible caching proxy (cache off by default) |
| [`sessions`](proxy-sessions.md) | Agent session profiling from proxy data |

<details><summary>Full `throttle --help` output (0.4.2)</summary>

```text
usage: throttle [-h] [--version] COMMAND ...

Measure what LLM inference costs in dollars per million tokens, and which serving-config change lowers it.

positional arguments:
  COMMAND
    plan                show traffic, cost, destination, and privacy without
                        sending traffic
    smoke               run a short, explicitly non-decision-grade check
    benchmark           run sustained exploratory evidence blocks (sweeps are
                        not counterbalanced)
    diagnose            pre-flight bottleneck regime classification (not
                        decision-grade)
    experimental-tuning
                        run an opt-in, suggestion-only server-metrics analysis
    golden              orchestrate the six-position counterbalanced decision
                        protocol
    compare             compare two saved benchmark reports without network
                        traffic
    report              generate HTML report with chart comparing measure
                        outputs
    golden-report       generate self-contained HTML report from golden
                        protocol artifacts
    demo                run a fast simulator demo comparing baseline vs tuned
                        configuration
    cost                measure cost per million tokens against a live
                        endpoint
    measure             measure cost per million tokens with statistical rigor
    check               after a serving change, measure $/M tokens and compare
                        with the last check
    watch               read vLLM /metrics and report cost per million tokens
                        (no requests sent)
    proxy               run an OpenAI-compatible caching proxy server
    sessions            View agent session profiling data

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit

Get your first $/M-token number (3 commands):

  1. See the output without a server (simulated, no traffic):
       throttle demo

  2. Measure a real endpoint (local Ollama shown; any OpenAI-compatible URL works,
     with or without /v1). The hourly rate is what your GPU(s) cost you:
       throttle cost --url http://localhost:11434 --model llama3.2:3b --gpu-hourly-rate 1.50

  3. Catch cost drift: record today's config three times (the repeats measure
     run-to-run noise), change one server setting, check again. Each check is compared
     with the last one for that endpoint (95% CI and noise bound; overlap is NO WINNER,
     fewer than 3 recent repeats is NOT CALIBRATED). --config only records what you changed:
       throttle check --url http://localhost:11434 --model llama3.2:3b --gpu-hourly-rate 1.50 --label before
       (run it two more times, unchanged)
       (change a setting, e.g. restart Ollama with OLLAMA_NUM_PARALLEL=4, then:)
       throttle check --url http://localhost:11434 --model llama3.2:3b --gpu-hourly-rate 1.50 --label after --config OLLAMA_NUM_PARALLEL=4

Other commands: plan / smoke / benchmark / golden (capped, evidence-labelled runs),
measure + compare (repeated trials to JSON), watch (read vLLM /metrics),
proxy + sessions (optional cache and agent profiling), report, diagnose.
Run 'throttle COMMAND --help' for details.
```

</details>
