# 5. Report

*Which kinds of work should move to your model, and what that saves per month.*

The report joins three things: captured API spend from [Workloads](workloads.md), the latest [Eval](eval.md) of your candidate, and a measured [`throttle check`](../cli/check.md) of the server it runs on.

```bash
throttle check --url http://gpu-box:8000 --model Qwen/Qwen2.5-72B-Instruct \
    --total-hourly-price 4.98 --gpus 2 --label qwen72b@vllm
throttle report --candidate qwen72b@vllm --gpu-hourly-rate 2.49 --gpus 2
```

`check` takes the hourly price of all GPUs behind the server and measures $/M tokens with a 95% interval. `report --gpu-hourly-rate` is the price of one GPU, and `--gpus` is how many an always-on deployment needs.

## Recommendations

| Recommendation | When |
| --- | --- |
| **MOVE** | Eval says BETTER or NO MATERIAL DIFFERENCE, the owned cost is measured, and it is cheaper even at the high end of the cost interval. |
| **KEEP ON API** | Owning costs more, or quality is WORSE and a format check (JSON, schema, tool calls) is clearly worse. |
| **NEEDS FINETUNE** | Quality is WORSE but the format checks hold. A recommendation only: Throttle does not fine-tune. |
| **NEEDS MORE DATA** | No shadow answers, an eval that is INSUFFICIENT DATA, INCONCLUSIVE or CHECKS ONLY, or no measured owned cost. |

## Cost math

- Owned cost per kind = requests per month x your model's measured tokens per request x the check's $/M.
- Savings use the high end of the check's 95% interval, not the middle.
- A dedicated GPU is paid for every hour. The owned total is never below `--gpu-hourly-rate` x `--gpus` x 730 hours, applied once across all MOVE kinds. If savings are not positive after that, every MOVE becomes KEEP ON API.
- Without a check record, quality is still shown, but every cost and savings cell says "not measured" and has no number.

## Output

`report.json` and a self-contained `report.html`, in `./throttle-report/` or `--out DIR`. The HTML ends with how every number was produced and what it does not prove.

| Flag | Default | What it does |
| --- | --- | --- |
| `--candidate` | required | The label from shadow and eval |
| `--gpu-hourly-rate` | required | Price of one GPU per hour (ASSUMED) |
| `--gpus` | required | GPUs the always-on deployment needs |
| `--check ID` | latest for the candidate | Which check record to price with |
| `--eval RUN_ID` | latest for the candidate | Which eval run to use |
| `--prices FILE` | bundled table | Extra or replacement API prices (ASSUMED) |
| `--out DIR` | `./throttle-report` | Where the files go |
| `--json` | off | Also print the JSON |

Next: [Route](route.md).
