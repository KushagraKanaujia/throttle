# Installation

*Install the Throttle CLI and confirm it works.*

Throttle needs **Python 3.11 or newer**. The command is `throttle`.

The own-your-intelligence loop (`capture`, `workloads`, `shadow`, `eval`, `report`, `route`, `refresh`, `demo`) lives on the `own/mvp` branch for now:

```bash loop (own/mvp)
pip install git+https://github.com/KushagraKanaujia/throttle@own/mvp
throttle demo
```

The PyPI package `throttle-pro` is behind and does not have the loop yet. It has the serving commands such as `check`:

```bash pipx
pipx install throttle-pro
throttle --version
```

```bash pip
python3 -m pip install throttle-pro
throttle --version
```

```bash upgrade
pipx upgrade throttle-pro
```


## Optional extras

| Extra | Install | What it adds |
| --- | --- | --- |
| `embeddings` | `pipx install 'throttle-pro[embeddings]'` | `throttle workloads --embed` (similarity clusters) and the opt-in semantic tier of the caching proxy. |

## Check your setup

```bash
throttle --help
```

With no server running, `throttle demo` shows the whole loop on made-up data. With a server, point `throttle check` at its OpenAI-compatible URL (with or without `/v1`).

> **Tip:** Throttle sends requests only to endpoints you name. Keys are never stored. See [Privacy](../more/privacy.md).
