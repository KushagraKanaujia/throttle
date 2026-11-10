# Installation

*Install the Throttle CLI and confirm it works.*

Throttle needs **Python 3.11 or newer**. The command is `throttle`.

Install v0.7.0 from `main`, pinned to its full commit SHA so what you install never changes under you. These docs can be newer than that commit; each release moves the pin to itself.

```bash loop (v0.7.0)
pip install "git+https://github.com/KushagraKanaujia/throttle@64cb03e30b87b1de0e4bcc6bed55d0a54641eb95"  # v0.7.0
throttle demo
```

That gives you the whole loop (`capture`, `workloads`, `shadow`, `eval`, `report`, `route`, `refresh`, `demo`), `jobs`, `pr-check` and the serving tools.

The PyPI package `throttle-pro` is behind until 0.7.0 is published there. It does not have the loop yet, only the serving commands such as `check`:

```bash pipx
pipx install throttle-pro
throttle --version
```

```bash pip
python3 -m pip install throttle-pro
throttle --version
```


## Optional extras

| Extra | Install | What it adds |
| --- | --- | --- |
| `embeddings` | `pip install "throttle-pro[embeddings] @ git+https://github.com/KushagraKanaujia/throttle@64cb03e30b87b1de0e4bcc6bed55d0a54641eb95"` | `throttle workloads --embed` (similarity clusters) and the opt-in semantic tier of the caching proxy. |

## Check your setup

```bash
throttle --help
```

With no server running, `throttle demo` shows the whole loop on made-up data. With a server, point `throttle check` at its OpenAI-compatible URL (with or without `/v1`).

> **Tip:** Throttle sends requests only to endpoints you name. Keys are never stored. The capture proxy sees your API keys in flight, so pin what you install by SHA. See [Privacy and data flow](../more/privacy.md).
