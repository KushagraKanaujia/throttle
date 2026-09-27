# Compare two GPUs

*Measure the same model on two machines and compare cost per token fairly.*

![](../../illustrations/gpus.svg)

Two GPUs are two endpoints. Compare them with the **same model, the same workload and a label per machine**, and point the final check at the other machine's check with `--against`.

### 1. Calibrate GPU A

Run the same check at least twice on machine A (three is better):
```bash
throttle check --url http://gpu-a:8000 --model my-model \
--gpu-hourly-rate 1.39 --config gpu=A100 --label a100 --concurrency 8
```

### 2. Calibrate GPU B

```bash
throttle check --url http://gpu-b:8000 --model my-model \
--gpu-hourly-rate 2.49 --config gpu=H100 --label h100 --concurrency 8
```

Run it at least twice too. Calibration pools repeats from both configs (2 each gives 2 degrees of freedom), within 24 hours.

### 3. Compare B against A

Find A's latest check id with `throttle check --history`, then:
```bash
throttle check --url http://gpu-b:8000 --model my-model \
--gpu-hourly-rate 2.49 --config gpu=H100 --label h100 --concurrency 8 \
--against <A-check-id>
```

## How the price difference is handled

The verdict judges what was **measured**. When the two checks use different hourly rates, Throttle puts the new check on the baseline's rate before judging, and reports the rate-driven part of the difference separately (it's your assumption, so it's never called significant). You get both: whether B is faster per token, and what that means at each machine's price.

> **Warning:** Keep every workload flag identical on both machines (`--blocks`, `--requests-per-block`, `--concurrency`, `--max-tokens`, prompts, cache mode). Different workloads are never ranked.
