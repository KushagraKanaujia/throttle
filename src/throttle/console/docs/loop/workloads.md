# 2. Workloads

*What kinds of work your traffic is, and what each kind costs at list price.*

```bash
throttle workloads
throttle workloads --since 7d
```

Each captured call gets one type, from a fixed list of rules checked in order. The first match wins:

| Type | Main signals |
| --- | --- |
| agent | Tool calls or tool results in the conversation, or tools offered |
| extraction | `response_format` JSON or a JSON schema, one forced tool used as a schema, or a prompt that asks for JSON or fields |
| classification | A short answer to a classify or label prompt, or one picked from listed options |
| code | Code in the answer or the prompt, or a prompt that asks for code |
| summarization | A prompt that asks to summarize, or a long input with a short answer |
| chat | Everything else |

Calls captured with `--redact metadata` have no bodies, so they are `unknown`. Their tokens and spend still count, on their own row.

## Spend

Spend is the measured token counts times the published list price. The price table ships with Throttle. Each entry has the date it was checked and the page it came from, and every dollar figure built from it is labelled `LIST PRICE (<date>)`.

- A call with no usage, an unknown price or an error is excluded and counted, never estimated.
- Cached input tokens are priced at the cached rate.
- The monthly figure scales the capture window to 30 days. It is refused when the window is shorter than a day.

## Options

| Flag | What it does |
| --- | --- |
| `--since 30m\|24h\|7d\|2w\|DATE` | Only calls newer than this |
| `--prices FILE` | JSON price file that adds or replaces prices. Those prices are labelled ASSUMED. |
| `--embed` | Also split each type into clusters by prompt similarity. Needs `pip install "throttle-pro[embeddings]"`. |
| `--json` | Machine-readable output |
| `--dir` | Data folder |

Next: [Shadow](shadow.md).
