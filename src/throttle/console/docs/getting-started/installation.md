# Installation

*Install the Throttle CLI and confirm it works.*

Throttle needs **Python 3.11 or newer**. The command is `throttle`.

Install v0.8.0 from PyPI. Pin the version so what you install never changes under you.

```bash pip
python3 -m pip install "throttle-pro==0.8.0"
throttle demo
```

```bash pipx
pipx install "throttle-pro==0.8.0"
throttle --version
```

```bash uv
uvx --from "throttle-pro==0.8.0" throttle scan .
```

On a Mac the built-in `python3` is 3.9. There pip says `No matching distribution found for throttle-pro`: your Python is too old, the package is there. `uvx` (from [uv](https://docs.astral.sh/uv/)) fetches a recent Python by itself, so it works without installing one.

That gives you the whole loop (`capture`, `workloads`, `shadow`, `eval`, `report`, `route`, `refresh`, `demo`), `jobs`, `pr-check` and the serving tools such as `check`.

To install straight from the source you can read, use the release tag:

```bash
pip install "git+https://github.com/KushagraKanaujia/throttle@v0.8.0"
```


## Optional extras

| Extra | Install | What it adds |
| --- | --- | --- |
| `embeddings` | `pip install "throttle-pro[embeddings]==0.8.0"` | `throttle workloads --embed` (similarity clusters) and the opt-in semantic tier of the caching proxy. |

## Check your setup

```bash
throttle --help
```

With no server running, `throttle demo` shows the whole loop on made-up data. With a server, point `throttle check` at its OpenAI-compatible URL (with or without `/v1`).

> **Tip:** Throttle sends requests only to endpoints you name. Keys are never stored. The capture proxy sees your API keys in flight, so pin what you install by SHA. See [Privacy and data flow](../more/privacy.md).
