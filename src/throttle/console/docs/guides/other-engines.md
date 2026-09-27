# Other OpenAI-compatible servers

*LMDeploy, TGI, llama.cpp server and anything that speaks the OpenAI API.*

Throttle talks to any server with an OpenAI-compatible chat completions API that returns a `usage` block with token counts.

```bash
throttle check --url http://your-server:PORT --model <model-name> \
  --gpu-hourly-rate <your $/hr> --label baseline
```

| Engine | Notes |
| --- | --- |
| LMDeploy | Measured end to end in Throttle's cross-engine validation. |
| TGI, llama.cpp server, others | Should work the same way if they expose OpenAI-compatible chat completions with `usage`. If the server omits `usage`, the check fails rather than guessing. |

For servers behind auth, put the key in an environment variable and pass its **name**:

```bash
export MY_KEY=...
throttle check --url https://inference.example.com --model my-model \
  --gpu-hourly-rate 2.49 --api-key-env MY_KEY
```

Plain HTTP to a non-loopback host is refused unless you pass `--allow-insecure-http` (your key would travel unencrypted).
