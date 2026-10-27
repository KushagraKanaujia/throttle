# Demo

*The whole loop on made-up traffic, offline, in a few seconds.*

```bash
throttle demo
```

It runs every stage for real on synthetic data:

1. **Capture.** Made-up OpenAI and Anthropic requests for all six kinds of work go through the real capture proxy to a fake API.
2. **Shadow.** The real shadow code replays them on a fake open model. Its quality is set per kind of work on purpose, so the report shows each recommendation.
3. **Eval.** The real evaluator and statistics, with a fake judge.
4. **Owned cost.** A check record labelled DEMO.
5. **Report.** The real report.

It needs no API key, no GPU and no network. It writes `report.json` and `report.html` to `./throttle-demo/` (or `--out DIR`). `--seed` changes the data; `--requests` sets how many requests are captured.

> **Note:** Every number in the demo is labelled DEMO. None of it is a measurement of a real company or a real model. Run the same loop on your own traffic for real numbers: start with [Capture](capture.md).
