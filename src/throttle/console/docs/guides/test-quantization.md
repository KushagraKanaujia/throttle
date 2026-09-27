# Test a quantization

*Check whether an FP8 or INT4 build is actually cheaper on your hardware.*

A quantized model usually changes throughput and memory, so its cost per token can move either way depending on your GPU and batch size. Measure it:

### 1. Baseline (three checks)

```bash
throttle check --url http://localhost:8000 --model my-model \
--gpu-hourly-rate 2.49 --config quantization=bf16 --label bf16 --concurrency 16
```

### 2. Serve the quantized build and check again

```bash
throttle check --url http://localhost:8000 --model my-model-fp8 \
--gpu-hourly-rate 2.49 --config quantization=fp8 --label fp8 --concurrency 16
```

If the verdict is CHEAPER, the saving is beyond run-to-run noise. If it's NO WINNER, the difference is too small to separate from noise at this sample size.

> **Warning:** Throttle measures cost, not output quality. Validate the quantized model's quality separately before shipping it.
