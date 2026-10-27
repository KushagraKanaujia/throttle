# CLI overview

*Every Throttle command at a glance.*

```bash
throttle [--version] COMMAND ...
```

## Everyday commands

| Command | What it does | Sends traffic? |
| --- | --- | --- |
| [`check`](check.md) | After a serving change: measure $/M tokens, compare with the last check, give a verdict | Yes (small, capped) |
| [`watch`](watch.md) | Read vLLM `/metrics` and report $/M tokens | No |
| `savings` | Verified-savings statement from two recorded checks | No |
| `autotune` | Find the cheapest serving config that passes OUTPUT CHANGED (free, local) | Yes |
| `gate` | CI cost gate between two recorded checks | No |
| `coldstart` | Time a server from launch to first token | Yes |
| `ui` | This console | No |

## Evidence pipeline

| Command | What it does |
| --- | --- |
| [`plan`](evidence-pipeline.md) | Show traffic, cost, destination and privacy without sending traffic |
| [`smoke`](evidence-pipeline.md) | A short, explicitly non-decision-grade check |
| [`benchmark`](evidence-pipeline.md) | Sustained exploratory evidence blocks |
| [`golden`](evidence-pipeline.md) | The six-position counterbalanced decision protocol |

## Saved runs and reports

| Command | What it does |
| --- | --- |
| [`compare`](compare.md) | Compare two saved reports (or six golden positions) without traffic, or two live endpoints on your prompts |
| [`golden-report`](compare.md) | Self-contained HTML from golden artifacts |

## Proxy and agents

| Command | What it does |
| --- | --- |
| [`proxy`](proxy-sessions.md) | OpenAI-compatible caching proxy (cache off by default) |
| [`sessions`](proxy-sessions.md) | Agent session profiling from proxy data |

Run `throttle --help` for the full list and `throttle COMMAND --help` for each command's options.
