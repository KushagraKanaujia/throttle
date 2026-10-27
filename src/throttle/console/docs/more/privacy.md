# Privacy

*What Throttle sends, stores and never sees.*

Throttle runs on your machine. There is no telemetry.

## The loop (capture, shadow, eval, route)

- **Capture listens on `127.0.0.1` by default.** Every call goes to the API you already use. Nothing is sent anywhere else, and nothing is cached.
- **Captured calls are stored locally,** in `~/.throttle/own` (or `$THROTTLE_OWN_DIR`), after redaction.
- **Auth headers are never written.** Keys for your model and your judge are read from environment variables you name and are never printed or stored.
- **Only endpoints you configure get your data:** your candidate model (shadow, route, refresh) and your judge (eval, refresh). If the judge is a public API, the redacted prompts it judges go to that provider. Use the provider you already call, or a judge you host.

### Redaction modes (`throttle capture --redact`)

| Mode | What is stored |
| --- | --- |
| `mask` (default) | Bodies, with emails, phone numbers, US SSNs, card numbers (Luhn-checked), IP addresses and secrets (API keys, tokens, JWTs, passwords) replaced by placeholders like `<EMAIL>`. Inline images and audio become `<IMAGE>` or `<AUDIO>`. |
| `hash` | Like `mask`, plus a salted hash so repeats of one value can be matched, e.g. `<EMAIL:3f2a9c>`. Set the salt with `THROTTLE_OWN_SALT`. |
| `metadata` | No bodies. Only model, token counts and timing. Workloads still shows spend, but shadow and eval need bodies. |
| `off` | Bodies as sent |

> **Warning:** Redaction is pattern matching. It does not catch names, street addresses, dates of birth, or free-text medical or financial details. If that content cannot sit on your disk, use `--redact metadata`.

## Serving commands (check, autotune, compare, coldstart)

- **Requests go only to the endpoint you name.**
- **Keys** are read from an environment variable you name (`--api-key-env`), never stored.
- **Plain HTTP** to a non-loopback host is refused unless you pass `--allow-insecure-http`.
- **`throttle check` history** is a local file (default `~/.throttle/checks`). It keeps each check's endpoint URL so checks of the same endpoint can be compared. It holds no keys, prompts or responses.
- **Reports** from plan/smoke/benchmark/golden contain hashes and aggregate numbers, not endpoint URLs, hostnames, keys, prompts or responses.
- **`--share`** prints a summary without URLs, hostnames, IPs, local paths or keys, masks secret-looking values, and uploads nothing.
- **The session profiler** stores content hashes, timings and token counts, never prompt or completion text.
