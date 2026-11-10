# 5. Report

*Which kinds of work should move to a cheaper model, and what that saves per month.*

The report joins three things: captured API spend from [Workloads](workloads.md), the latest [Eval](eval.md) of your candidate, and the price of running that candidate. The candidate can be an open model you run yourself, an open model on a hosted endpoint, or a cheaper closed API model.

## Pick one way to price the candidate

Pick exactly one mode. If you give neither, or flags from both, `throttle report` stops with exit code 2 and says which flags to use.

### Per-token mode: no GPU to run

Use this when the candidate is billed per token: a cheaper API model, or an open model on a hosted endpoint.

```bash
# A cheaper API model that is in the bundled price list: LIST PRICE
throttle report --candidate mini --candidate-model gpt-4.1-mini

# A hosted open model, or any price you know: ASSUMED
throttle report --candidate qwen72b@host --candidate-price-in 0.20 --candidate-price-out 0.80
```

The prices in the second example are samples, not a quote from any provider. Use the prices on your provider's page.

- `--candidate-price-in` and `--candidate-price-out` are USD per 1M input and output tokens. They are labelled **ASSUMED**. Give both.
- `--candidate-model NAME` looks the model up in the bundled price list and is labelled **LIST PRICE** with the date the price was checked. The list holds API models only. For a hosted open model, pass the two price flags.
- If you give both, the price flags win.

Owned cost comes from **your candidate's own tokens**, measured on each shadow answer, never from the API model's token counts. A model with a cheaper list price can still cost more if it writes longer answers or thinks for longer. The report then says KEEP ON API.

Sample panel (made-up numbers, not a customer):

```text
Owned cost:                   per token, ASSUMED (operator price)
Candidate price:              $0.2/M in, $0.8/M out (open-70b)

type              API $/mo  owned $/mo (high)  recommendation
classification        $324             $32.83  MOVE
```

### GPU mode: you run the candidate

Use this when you serve the candidate on your own GPUs. A measured [`throttle check`](../cli/check.md) of that server gives its $/M tokens.

```bash
throttle check --url http://gpu-box:8000 --model Qwen/Qwen2.5-72B-Instruct \
    --total-hourly-price 4.98 --gpus 2 --label qwen72b@vllm
throttle report --candidate qwen72b@vllm --gpu-hourly-rate 2.49 --gpus 2
```

The GPU prices above are samples. `check` takes the hourly price of all GPUs behind the server and measures $/M tokens with a 95% interval. `report --gpu-hourly-rate` is the price of one GPU, and `--gpus` is how many an always-on deployment needs.

## Recommendations

| Recommendation | When |
| --- | --- |
| **MOVE** | Eval says BETTER or NO MATERIAL DIFFERENCE, the owned cost is known, and it is cheaper even at the high end of the cost interval. |
| **KEEP ON API** | The candidate costs more, or quality is WORSE and a format check (JSON, schema, tool calls) is clearly worse. |
| **NEEDS FINETUNE** | Quality is WORSE but the format checks hold. A recommendation only: Throttle does not fine-tune. |
| **NEEDS MORE DATA** | No shadow answers, an eval that is INSUFFICIENT DATA, INCONCLUSIVE or CHECKS ONLY, or no owned cost. |

The rule is the same in both modes. Nothing moves without your OK.

## Cost math

- Per-token mode: each shadow answer's input and output tokens are priced, and owned cost per kind = requests per month x the mean cost per request, with a 95% interval over requests. Cached input tokens use the cached price when the price list has one, otherwise the input price. No GPU floor.
- GPU mode: owned cost per kind = requests per month x your model's measured tokens per request x the check's $/M. A dedicated GPU is paid for every hour, so the owned total is never below `--gpu-hourly-rate` x `--gpus` x 730 hours, applied once across all MOVE kinds. If savings are not positive after that, every MOVE becomes KEEP ON API.
- Savings always use the high end of the 95% interval, not the middle. The per-kind column in the terminal and on the page shows that high end.
- Without a check record (GPU mode), or without shadow answers that report token usage (per-token mode), quality is still shown, but every cost and savings cell says "not measured" and has no number.

## Output

`report.json` and a self-contained `report.html`, in `./throttle-report/` or `--out DIR`. The HTML ends with how every number was produced and what it does not prove. `report.json` records the mode in `owned_cost_mode` (`gpu` or `per_token`) and, in per-token mode, the prices used and their label.

| Flag | Default | What it does |
| --- | --- | --- |
| `--candidate` | required | The label from shadow and eval |
| `--candidate-price-in` | none | Per-token mode: USD per 1M input tokens (ASSUMED) |
| `--candidate-price-out` | none | Per-token mode: USD per 1M output tokens (ASSUMED) |
| `--candidate-model` | none | Per-token mode: a model in the bundled price list (LIST PRICE) |
| `--gpu-hourly-rate` | none | GPU mode: price of one GPU per hour (ASSUMED) |
| `--gpus` | none | GPU mode: GPUs the always-on deployment needs |
| `--check ID` | latest for the candidate | GPU mode: which check record to price with |
| `--eval RUN_ID` | latest for the candidate | Which eval run to use |
| `--prices FILE` | bundled table | Extra or replacement prices (ASSUMED) |
| `--out DIR` | `./throttle-report` | Where the files go |
| `--json` | off | Also print the JSON |

Next: [Route](route.md).
