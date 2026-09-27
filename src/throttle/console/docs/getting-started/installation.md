# Installation

*Install the Throttle CLI and confirm it works.*

Throttle needs **Python 3.11 or newer**. It's published on PyPI as `throttle-pro`; the command is `throttle`.

```bash pipx (recommended)
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

`throttle --version` should print `throttle 0.4.2` (or newer).

## Optional extras

| Extra | Install | What it adds |
| --- | --- | --- |
| `embeddings` | `pipx install 'throttle-pro[embeddings]'` | The opt-in semantic tier of the caching proxy. Not needed for cost checks. |

## Check your setup

```bash
throttle --help
```

With no server running, `throttle demo` confirms the install end to end. With a server, point `throttle check` at its OpenAI-compatible URL (with or without `/v1`).

> **Tip:** Throttle sends requests only to the endpoint you name. Prompts, responses and keys never leave your machine. See [Privacy](../more/privacy.md).
