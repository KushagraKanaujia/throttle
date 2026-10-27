# 1. Capture

*Record your OpenAI and Anthropic traffic locally, without changing it.*

```bash
throttle capture
```

Then point your app at the proxy:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8790/v1
export ANTHROPIC_BASE_URL=http://127.0.0.1:8790
```

- Every call goes to the API you already use, with the same headers. The answer comes back unchanged.
- Streaming is passed through chunk by chunk. A copy is put together on the side for the record.
- A redacted copy of each call is written to `~/.throttle/own/capture/`, one file per month.
- Nothing is cached. Auth headers are never written. Nothing is sent anywhere except the upstream.

## Options

| Flag | Default | What it does |
| --- | --- | --- |
| `--redact off\|mask\|hash\|metadata` | `mask` | How much of each body to keep. See [Privacy](../more/privacy.md). |
| `--port` | `8790` | Port to listen on |
| `--host` | `127.0.0.1` | Address to listen on. Anything else lets other machines send traffic through it. |
| `--openai-upstream` | `https://api.openai.com` | Where OpenAI calls go |
| `--anthropic-upstream` | `https://api.anthropic.com` | Where Anthropic calls go |
| `--route` | off | Serve approved kinds of work from your own model. See [Route](route.md). |
| `--dir` | `~/.throttle/own` | Data folder |

## See what you have

```bash
throttle capture stats
```

Calls, errors, streamed calls, models, redaction modes, and how many calls reported token usage. Calls without usage are left out of every cost number, never estimated.

Next: [Workloads](workloads.md).
