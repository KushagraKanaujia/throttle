# The loop

*Prove which share of your API traffic can move to a model you run, move it, and keep it current.*

```text
capture ──► workloads ──► shadow ──► eval ──► report ──► route ──► refresh
```

| Stage | Command | What it answers |
| --- | --- | --- |
| [Demo](demo.md) | `throttle demo` | What does the whole loop look like? (made-up data, offline) |
| [1. Capture](capture.md) | `throttle capture` | What does my app send to OpenAI or Anthropic? |
| [2. Workloads](workloads.md) | `throttle workloads` | What kinds of work is it, and what does each cost? |
| [3. Shadow](shadow.md) | `throttle shadow` | How does my own model answer the same requests? |
| [4. Eval](eval.md) | `throttle eval` | Is it as good, per kind of work? |
| [5. Report](report.md) | `throttle report` | What should move, and what does it save? |
| [6. Route](route.md) | `throttle route`, `throttle capture --route` | Serve it from my model, with API fallback. What did it save? |
| [7. Refresh](refresh.md) | `throttle refresh` | Is a new open model better than the one I run? |

All data lives in `~/.throttle/own` (or `$THROTTLE_OWN_DIR`). Every command takes `--dir` to use another folder.

How the numbers are kept honest: [Honest numbers](honest-numbers.md).
