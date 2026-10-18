# Throttle Cloud and the CLI

The Throttle CLI is free, local and open source (MIT). Throttle Cloud is the
paid side that cuts the bill: **Autotune** (`throttle autotune`), the **CI
cost gate** (`throttle gate`) and a shared **check history**
(`throttle check --push`). New accounts start with \$10 of credits, no card.
Plans, usage rates and the Enterprise pilot:
[throttle-pro.com/pricing](https://www.throttle-pro.com/pricing).

## Logging in

```sh
throttle login             # prompts for the key (not echoed, not in shell history)
throttle login --key thr_live_...
throttle whoami            # email, plan, credits left, and where the key came from
throttle logout            # deletes the saved key
```

`throttle login` verifies the key with `GET /api/v1/me` and only then saves it
to `~/.config/throttle/credentials` (or `$XDG_CONFIG_HOME/throttle/credentials`)
with mode 600, in a directory with mode 700. If the file is ever readable by
other users, every cloud command warns you.

`THROTTLE_API_KEY` overrides the saved key; use it in CI instead of logging in.
`THROTTLE_CLOUD_URL` points the CLI at another server (local development);
plain `http://` is refused except to localhost, because the key travels with
every request.

## When the CLI talks to Throttle Cloud

Only when you are logged in **and** run one of these:

| Command | Endpoint | Credits |
| --- | --- | --- |
| `throttle login`, `throttle whoami`, start of `throttle autotune` | `GET /api/v1/me` | free |
| `throttle check ... --push` (or `push: true` in `~/.throttle/config.yaml`) | `POST /api/v1/checks` | \$0.25 per check |
| `throttle autotune` (not `--dry-run`) | `POST /api/v1/autotune/trials` before every trial, `POST /api/v1/autotune/runs/{run_id}/result` at the end | \$1.00 per trial, the baseline included |
| `throttle gate` (not `--no-cloud`) | `POST /api/v1/ci/gate` | \$0.10 per gate |

Every other command, and every command when you are not logged in, sends
nothing to Throttle Cloud. `--no-push` overrides `push: true` for one check.

Every POST carries an `Idempotency-Key` header derived from what it describes
(`check:<check id>`, `autotune:<run id>:trial:<n>`, `autotune:<run id>:result`,
`gate:<hash of the two check ids>`), so a retried or repeated call is never
charged twice. A dropped connection or a 5xx is retried once with the same key.

## Errors and out of credits

| HTTP | What the CLI does |
| --- | --- |
| 401 | Says the key was rejected and how to fix it (`throttle login`, or check `THROTTLE_API_KEY`). |
| 402 | Says you are out of credits, with the link to add some. Autotune stops before the next trial and still prints the best config found so far. Nothing more is charged. |
| 403 | Says your plan does not include that feature, with the pricing link. |
| 422 | Says the upload was refused and why. |
| no connection | Says Throttle Cloud could not be reached; nothing was uploaded or charged. |

Every local feature keeps working in all of these cases: the check is still
measured, judged and saved locally, `throttle check --push` keeps its own exit
code (the upload failure is a warning), and `throttle gate` still judges
locally.

## Exactly what a pushed check contains

The upload is built from an allow-list by `throttle.cloud.check_payload`,
never by copying the local record. It contains:

- **Identity:** `record_type`, `record_version`, check `id`, `created_at`,
  `throttle_version`, `metric`, `monthly_tokens`, and `endpoint_sha256` (the
  first 16 hex digits of the SHA-256 of the endpoint URL, so checks of one
  server group together without the URL leaving your machine).
- **Model, engine, GPU:** `model`, `engine` and `gpu` as `throttle check
  --share` shows them.
- **Headline:** `usd_per_m`, `ci_low`, `ci_high`, `verdict`.
- **Config fingerprint:** the GPU $/hr you typed (ASSUMED), GPU count, label,
  and every config key/value Throttle shows in its "Config fingerprint"
  section (your `--config` pairs, vLLM info labels, `/v1/models` details).
  Values that look like secrets are masked; URLs, hostnames, IPs and
  absolute paths are removed.
- **Workload:** blocks, requests per block, concurrency, max tokens, warm-up
  requests, temperature, prompt count, cache mode, the SHA-256 of the prompt
  set, whether the prompts are built-in or custom, and for agent checks the
  measured shape (tokens per turn).
- **Measurements:** per block: input, output and tag tokens, max-token hits,
  wall-clock seconds, \$/M and GPU \$; the summary \$/M with 95% CI per metric.
- **Comparison** (when there is a baseline): baseline id, verdict and its
  reason (scrubbed), the baseline summary, deltas, the noise bound and the
  output-shape numbers (tokens per request, % of answers at the token cap).
- Optional `label` and `project` (`--project NAME`).

It never contains prompts, responses, the endpoint URL, hostnames, IPs, file
paths, API keys or authorization headers. Prompts and responses are not in a
local check record either: only the SHA-256 of the prompt set.

Autotune trials send the trial index and its config (secret-looking values
masked). The autotune result sends the best config, baseline and best \$/M,
the noise bound, the projected monthly saving and, per trial, its config,
GPU count, \$/M with CI and verdict. The launch command, endpoint URL and log
paths stay on your machine (in `~/.throttle/autotune/<run id>/`).

A gate sends the same allow-listed payload for the baseline and the
candidate (or just the baseline's cloud check id).

## Autotune in one screen

```sh
throttle autotune --dry-run --monthly-tokens 2B      # simulated, free, no login
throttle autotune --url http://localhost:8000 --model Qwen/Qwen2.5-7B-Instruct \
    --gpu-hourly-rate 2.49 --concurrency 16 --monthly-tokens 2B \
    --launch "vllm serve {model} --tensor-parallel-size {tp} --max-num-seqs {seqs} --gpu-memory-utilization {mem}" \
    --space tp=1,2 seqs=64,128,256 mem=0.85,0.9
```

- `--launch` runs with `/bin/sh`; `{model}` and every `--space` key are
  substituted verbatim. Throttle stops the whole process group after each
  trial (`--stop-command` runs an extra command, e.g. `docker rm -f vllm`).
  It refuses to start if something already answers at the health URL
  (default `<url>/v1/models`, or `--health-url`).
- `--space` takes `KEY=V1,V2,...` items (in order, so the search can walk up
  and down) or one YAML/JSON file mapping each key to a list. An empty value
  removes a flag, e.g. YAML `quant: ["", "--quantization fp8"]`.
- `--gpu-hourly-rate` is the price of **one** GPU. Each trial pays for its
  GPU count, read from `tp` / `tensor_parallel_size` / `gpus` / `num_gpus`
  (or `--gpus-param KEY`; `--gpus-param none` uses `--gpus` for every trial).
- All `throttle check` workload options apply (`--concurrency`, `--max-tokens`,
  `--blocks`, `--workload agent`, `--prompts`, safety limits). The priciest
  trial is checked against the safety limits before anything runs or is
  charged.
- Trial checks, logs and `autotune.json` are saved in
  `~/.throttle/autotune/<run id>/` (or `--history-dir`).
