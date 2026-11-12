# Scan your code

*Find the AI calls in your code, what each costs at list price, and which cheaper models to test. Offline, in seconds.*

```bash
throttle scan .
```

Needs Throttle 0.8.0 or later; 0.7.0 does not have it. It needs no API key, no network, no GPU and no traffic, and it writes nothing. It reads the Python, JavaScript and TypeScript files in the folder (skipping `node_modules`, virtualenvs, `dist`, `build` and similar) and finds the direct OpenAI and Anthropic SDK calls it can read, the same way `throttle jobs scan` does.

For each call it shows:

- **Where it is**: the job name (a `# throttle-job: name` tag, or the function) and `file:line`.
- **The model**: the literal name in the code. When the code sets it at runtime, the default written next to it (`os.getenv("MODEL", "gpt-4o")`, `process.env.MODEL ?? "gpt-4o"`, `def f(model="gpt-4o")`), marked "(default)", or the model you name with `--model NAME`, marked "(--model)".
- **What it costs**: per 1,000 calls at list price, from the bundled price table. With `--calls-per-day N` it shows dollars per month instead (N calls a day for each job, x 30 days).
- **Up to two cheaper models to test**, each marked UNTESTED.

The table lists the priced calls, most expensive first, up to 15 (`--all` lists every one). It ends with one summary line, for example "6 AI calls found, 4 priced. Moving the priced ones to the first candidate could cut about 61% of their list-price cost, if every call site runs equally often (ESTIMATE)", and the commands that test the first candidate on your own traffic.

Calls in tests, specs, examples, evals and benchmarks (`tests/`, `__tests__/`, `*.test.ts`, `test_*.py`, `examples/`, `evals/` and similar) are left out of the totals and counted on one line. `--all` puts them back. When every call is in one of them, they are all counted.

## What the numbers are

Every dollar figure is an **ESTIMATE**: list price times a token count per call.

| Input | Default | Change it with |
| --- | --- | --- |
| Input tokens per call | 1,000 (ASSUMED) | `--tokens-in N` |
| Output tokens per call | 300 (ASSUMED) | `--tokens-out N` |
| Calls per day, per job | none: shows $ per 1,000 calls | `--calls-per-day N` |
| Prices | bundled table, LIST PRICE with its date | `--prices FILE` (ASSUMED) |

When the code passes a literal output cap (`max_tokens`, `max_completion_tokens`, `max_output_tokens`, `maxTokens` or `maxOutputTokens`) that is smaller than the output tokens above, scan uses the cap and marks the model with `*`. Cached input is not counted.

## How the cheaper models are picked

All comparisons are list cost per call at the token counts above. A model must cost at least 10% less to be listed. Dated snapshots and models built only for code are left out.

1. **First pick, same provider.** One tier down in the same family: `gpt-4o` to `gpt-4o-mini`, `claude-sonnet-4-5` to `claude-haiku-4-5`. When the family has no cheaper tier, the same provider's next price step down.
2. **Second pick, other provider.** The model whose cost per call is closest to the first pick's, so it sits in the same price class.

> **Note:** A cheaper model is only cheaper. Scan does not test quality, so every pick is UNTESTED. A model with a lower list price can still cost more in real use if it writes longer answers or thinks longer. Prove it with the loop below, which uses measured tokens.

## What it cannot price

- **A model set at runtime with no default in the code** (`model=os.environ["CHAT_MODEL"]`, `model: config.model`, `create(**params)`). It is listed with the code that sets it. Rerun with `--model NAME` to price those calls as the model you run.
- **A model that is not in the price table**, such as a fine-tune, a deployment name, another provider's model or an older model the price page no longer lists. It is listed by name. Add a price with `--prices FILE`.
- **Calls through your own wrapper, another SDK or an agent framework.** They are not found. `throttle capture` sees them on real traffic.

When nothing is found, scan says what it looks for and exits 0.

## Prove it on your own traffic

```bash
throttle capture
export OPENAI_BASE_URL=http://127.0.0.1:8790/v1
export ANTHROPIC_BASE_URL=http://127.0.0.1:8790

# after real traffic has gone through
throttle shadow --candidate-url https://api.openai.com/v1 \
    --api-key-env OPENAI_API_KEY --candidate-model gpt-4o-mini --label gpt-4o-mini
throttle eval --candidate gpt-4o-mini --judge-model gpt-4.1 --judge-key-env OPENAI_API_KEY
throttle report --candidate gpt-4o-mini --candidate-model gpt-4o-mini
```

Scan prints these commands with its own first pick filled in. When the pick is a Claude model, the shadow step uses `https://api.anthropic.com/v1` (Anthropic's OpenAI-compatible endpoint, offered for testing) with `ANTHROPIC_API_KEY`, and when the code has no OpenAI calls the judge is a Claude model (`--judge-provider anthropic`). Each step has its own page under [The loop](../loop/overview.md).

## For tools

`throttle scan . --json` prints the same data as JSON with `"schema": "throttle.own.scan/1"`: the assumptions, a summary, one entry per call, tests and examples included with `"counted": false` (with `status` `priced`, `model_set_at_runtime`, `model_not_in_price_table` or `not_priced_at_this_size`) and each candidate with `"status": "UNTESTED"`.
