# Fewest GPUs for your model

*Before you rent cards: how many does your model need at your traffic?*

`throttle fit` is memory math. It runs on your laptop in a second, with no GPU and no keys, and every number it prints is an **estimate**.

```bash
throttle fit --model llama-3.3-70b --quant fp8 --context 32k --users 16 \
  --gpu h100 --gpu l40s --price h100=2.49 --price l40s=0.99 --current h100=4
```

For each GPU it shows:

| Column | Meaning |
|---|---|
| fewest cards | the smallest card count that holds the weights plus a full-length KV cache for every chat at once |
| chats they hold | how many full-length chats that many cards can hold |
| per month | that many cards at your `--price`, 730 hours a month |

With `--current`, it tells you if fewer cards look like enough, and what that would save **if a measurement agrees**.

### What it does not know

- **Speed.** More cards can be slower, because they have to talk to each other. Measure before you change anything.
- **Your real chat lengths.** It assumes every chat at the full `--context` at the same time. Real traffic is usually lighter, so this errs on the safe side.
- **Exotic attention.** Sliding-window and latent attention models need less KV memory than this math assumes.

### Then measure it

`throttle fit` ends with the command to run on your server, for example:

```bash
throttle autotune --url http://localhost:8000 --model meta-llama/Llama-3.3-70B-Instruct \
  --gpu-hourly-rate 2.49 \
  --launch "vllm serve {model} --tensor-parallel-size {tp}" --space tp=2,4
```

That gives you real cost per million tokens for each card count, with confidence intervals.

### Any model, any card

```bash
throttle fit --config ./config.json --params 32.8 --quant awq --context 8k --users 8
throttle fit --model qwen2.5-32b --quant q4_k_m --engine llamacpp --gpu rtx3090 --gpu 1060=6
```

`--list-models` and `--list-gpus` show what is built in. `--json` prints the result for scripts.
