# Privacy

*What Throttle sends, stores and never sees.*

- **Requests go only to the endpoint you name.** There's no telemetry.
- **Keys** are read from an environment variable you name (`--api-key-env`), never stored.
- **Plain HTTP** to a non-loopback host is refused unless you pass `--allow-insecure-http`.
- **`throttle check` history** is a local file (default `~/.throttle/checks`). It keeps each check's endpoint URL so checks of the same endpoint can be compared. It holds no keys, prompts or responses.
- **Reports** from plan/smoke/benchmark/golden contain hashes and aggregate numbers, not endpoint URLs, hostnames, keys, prompts or responses.
- **`--share`** prints a summary without URLs, hostnames, IPs, local paths or keys, masks secret-looking values, and uploads nothing.
- **The session profiler** stores content hashes, timings and token counts, never prompt or completion text.
