# 6. Route

*Serve approved kinds of work from your own model, fall back to the API when in doubt, and count what it saved.*

## Approve

```bash
throttle route approve --type extraction --candidate qwen72b@vllm \
    --url http://gpu-box:8000/v1 --model Qwen/Qwen2.5-72B-Instruct
```

Approval is refused unless the latest eval for that candidate said BETTER or NO MATERIAL DIFFERENCE for that kind, on the same `--model`. `throttle route approve --help` explains the gate.

| Flag | What it does |
| --- | --- |
| `--canary 0.1` | Serve only this share of the kind from your model |
| `--api-key-env VAR` | Environment variable with your server's key |
| `--force` | Approve without a passing eval. The route is marked FORCED everywhere and is never counted in the net. |

`throttle route list` shows routes. `throttle route revoke --type extraction` stops one on the next request.

## Serve

```bash
throttle capture --route
```

Each request is typed from the request alone and, if its kind is approved, sent to your model. The answer comes back in your client's format (OpenAI or Anthropic) with the header `x-throttle-served-by: owned`.

Anything doubtful falls back to the API with the original request, unchanged: a connection error, a timeout, an HTTP error, a bad, empty or cut-off answer, invalid JSON, a schema failure, an unknown tool, or bad tool arguments.

Limits:

- Owned answers are not truly streamed. Throttle waits for the full answer and replays it as a short stream.
- A fallback waits for the owned try first, up to the timeout.
- Requests with images, `n` > 1, or no messages are never routed.
- Anthropic `thinking`, `cache_control` and `tool_choice` are not sent to your model.

## Realized savings

```bash
throttle route savings --month 2026-11 --gpu-hourly-rate 2.49 --gpus 2
```

This is a realized-savings statement you can audit (good for showing your boss).

- **Avoided cost:** the tokens your model served, priced at the API model's list price. The share of input the API served from its prompt cache for that kind (measured from that month's captures) is priced at the cached rate. With no cache history, all input is priced at the cached rate.
- **Owned cost:** `--gpu-hourly-rate` x `--gpus` x every hour of that calendar month (for example "November: 720 h, calendar month"), or your invoice with `--owned-cost-usd` (labelled OPERATOR-REPORTED). The report's 730 h is an average month, used only for its projection.
- Fallbacks were answered by the API and are already on your API bill. They are not savings.
- Forced routes are shown on their own line and never counted in the net.
- If the net is zero or less, the statement says "No savings this month."

Next: [Refresh](refresh.md).
