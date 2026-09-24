# RunPod Deployment Checklist for Config Comparison

## Problem Statement

RunPod's vLLM template runs vLLM as PID 1 under docker-init. Using `pkill` to restart with different flags DOES NOT WORK - supervisor respawns with the original flags from deployment.

**Solution:** Deploy two separate pods (or redeploy the same pod) with different `--max-num-seqs` values set in the template's Docker Override Command field AT DEPLOY TIME.

## Deployment Steps

### Pod A: Baseline (max_num_seqs=256, default)

1. **Create RunPod Instance:**
   - Template: `runpod/pytorch` or vLLM template
   - GPU: L40S or A100
   - Docker Override Command:
     ```
     vllm serve meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8000
     ```

2. **SSH into Pod A and verify:**
   ```bash
   # Check the actual running command
   cat /proc/$(pgrep -f "vllm serve" | head -1)/cmdline | tr '\0' ' '
   
   # Should show: vllm serve meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8000
   # Note: NO --max-num-seqs flag = defaults to 256
   ```

3. **Install throttle:**
   ```bash
   pip install git+https://github.com/KushagraKanaujia/throttle.git
   ```

4. **Run baseline measurement:**
   ```bash
   throttle measure \
     --endpoint-url http://localhost:8000 \
     --model meta-llama/Llama-3.2-3B-Instruct \
     --gpu-hourly-rate 0.69 \
     --label baseline_256 \
     --repeat 10 \
     --num-requests 100 \
     --arrival-rate 10
   ```

5. **Download the result:**
   ```bash
   # Copy baseline_256.json to your local machine
   cat baseline_256.json
   ```

### Pod B: Modified (max_num_seqs=32)

1. **Create RunPod Instance:**
   - Template: Same as Pod A
   - GPU: Same type as Pod A
   - Docker Override Command:
     ```
     vllm serve meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8000 --max-num-seqs 32
     ```

2. **SSH into Pod B and verify:**
   ```bash
   # Check the actual running command
   cat /proc/$(pgrep -f "vllm serve" | head -1)/cmdline | tr '\0' ' '
   
   # MUST show: vllm serve meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8000 --max-num-seqs 32
   # If --max-num-seqs 32 is missing, the deployment FAILED
   ```

3. **Install throttle:**
   ```bash
   pip install git+https://github.com/KushagraKanaujia/throttle.git
   ```

4. **Run modified measurement:**
   ```bash
   throttle measure \
     --endpoint-url http://localhost:8000 \
     --model meta-llama/Llama-3.2-3B-Instruct \
     --gpu-hourly-rate 0.69 \
     --label max_num_seqs_32 \
     --repeat 10 \
     --num-requests 100 \
     --arrival-rate 10 \
     --note "max_num_seqs=32"
   ```

5. **Download the result:**
   ```bash
   # Copy max_num_seqs_32.json to your local machine
   cat max_num_seqs_32.json
   ```

## Compare Results (Local Machine)

```bash
# With both JSON files downloaded locally
throttle compare baseline_256.json max_num_seqs_32.json
```

## Alternative: Single Pod Redeployed Twice

If using a single pod:

1. Deploy Pod with baseline command (no --max-num-seqs)
2. Run baseline measurement
3. Download baseline_256.json
4. **STOP and DELETE the pod** (don't just restart!)
5. Deploy NEW pod with --max-num-seqs 32 command
6. Run modified measurement
7. Download max_num_seqs_32.json
8. Compare locally

## Verification Checklist

Before running `throttle compare`, verify:

- [ ] Pod A cmdline shows NO --max-num-seqs flag
- [ ] Pod B cmdline shows --max-num-seqs 32
- [ ] Both JSON files exist and contain different labels
- [ ] Both measurements used same model, GPU type, and workload params
- [ ] Both pods completed all 10 trials successfully

## Expected Outcome

The `compare` command should show either:
- **Ranked output** if confidence intervals don't overlap (significant difference)
- **NO SIGNIFICANT DIFFERENCE** if intervals overlap

This will be the first real before/after comparison with genuinely different configs.
