# Known Issues and Findings

**Last Verified:** 2026-09-01 - All issues remain UNFIXED in current main branch.

## Wheel Parity Job - Cold Cache Failure (UNFIXED)

**Status**: Known failure on cold cache, masked by warm cache in CI

**Problem**: The wheel parity job will fail under PYTHONWARNINGS=error when HuggingFace Hub cache is cold due to deprecated hf-xet API usage.

**Evidence**:
- Run 32702491665 (cold cache): FAILED with `DeprecationWarning: hf_xet.download_files() is deprecated`
- Run 32704183533 (warm cache): PASSED - cache hit prevented network download

**Cache Configuration**:
```yaml
- name: Cache HuggingFace Hub directory
  uses: actions/cache@1bd1e32a3bdc45362d1e726936510720a7c30a57
  with:
    path: ~/.cache/huggingface/hub
    key: hf-hub-Linux-sentence-transformers-all-MiniLM-L6-v2
```

**Root Cause**: huggingface-hub 0.36.2 uses deprecated hf_xet.download_files() API internally when downloading model files. This triggers DeprecationWarning which fails under warning-strict mode.

**When Failure Occurs**:
- Cold cache (new cache key, first run, or cache expiration)
- Fork repositories (different cache namespace)
- Cache invalidation or manual cache clear

**Attempted Fixes**:
- Option a (pin/update huggingface-hub): No version without hf-xet deprecation exists
- Option b (env var to disable hf-xet): No such environment variable exists

**Workaround** (NOT IMPLEMENTED):
Add scoped filterwarnings to pyproject.toml:
```toml
[tool.pytest.ini_options]
filterwarnings = [
    "ignore:hf_xet\\.download_files\\(\\) is deprecated:DeprecationWarning:huggingface_hub\\.file_download",
]
```

**Decision**: Documented as known cold cache failure. Fix requires either:
1. huggingface-hub upstream fix (remove hf-xet dependency)
2. Implement scoped filterwarnings (relaxes warning-strict for this specific case)

**Impact**: Low - CI cache hit rate is high, forks can add filterwarnings if needed

---

## RunPod A100 Validation Results

**Setup**: A100 80GB PCIe, vLLM serving Qwen/Qwen3-8B, launched by RunPod template with `--enforce-eager --gpu-memory-utilization 0.95 --max-model-len 8128`. GPU rate $1.39/hr.

### Simulator Error by Scenario

| Scenario | Arrival Rate | Requests | Wall Clock Error | Output Throughput Error |
|----------|--------------|----------|------------------|-------------------------|
| Light    | 1 req/s      | 20       | -71.9%           | +285%                   |
| Medium   | 5 req/s      | 50       | -89.1%           | +1051%                  |
| Heavy    | 10 req/s     | 100      | -96.5%           | +3007%                  |

### Measured vs Simulated Output Throughput

- **Measured**: 36.9, 36.8, 34.9 tok/s (flat across scenarios)
- **Simulated**: 142.2, 423.7, 1083.4 tok/s (increases with arrival rate)

Measured throughput is **FLAT** across a 10x change in arrival rate.

Per-request timings from the JSON show roughly 36 tok/s per individual request. Total system throughput also 36 tok/s. **Total equals per-request, which is what you see when exactly one request is in flight at a time.**

## Finding 1: validate-sim Executes Requests Serially (UNFIXED)

**The three load levels never happened.** Every scenario measured single-stream decode and the arrival rate parameter did nothing.

### Code Evidence

Request dispatch loop in `src/throttle/cli.py` lines 2686-2697:

```python
for i, (_, prompt_tokens, max_tokens) in enumerate(workload):
    prompt = "Test " * prompt_tokens

    req_start = time.perf_counter_ns()
    response = client.post(
        f"{args.endpoint_url}/chat/completions",
        json={
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
    )
    req_end = time.perf_counter_ns()
```

**Analysis**:
- The loop calls `client.post()` which blocks until the response is received
- No `asyncio.gather`, no task group, no concurrent dispatch mechanism
- Each request waits for the previous one to complete before starting

### What the arrival_rate Parameter Does

Line 2657 passes `arrival_rate` to workload generation:

```python
workload = workload_gen.generate_chat_workload(
    num_requests=scenario['num_requests'],
    arrival_rate_requests_per_sec=scenario['arrival_rate'],
    mean_prompt_tokens=200,
    mean_output_tokens=150,
)
```

Line 2686 shows the arrival time is discarded:

```python
for i, (_, prompt_tokens, max_tokens) in enumerate(workload):
```

The `_` is the arrival time. It is generated but never used. Requests are sent serially regardless of their intended arrival times.

### Impact

- Light/Medium/Heavy scenarios all measured the same workload: single-stream sequential requests
- The measured throughput being flat (36.9, 36.8, 34.9 tok/s) confirms no concurrency
- All "load level" comparisons are invalid
- Simulator error percentages reflect serial execution, not the intended concurrent load

## Finding 2: No API Key Support (UNFIXED)

**Issue**: `validate-sim` has no flag for an API key or authorization header.

### Code Evidence

Connectivity test (lines 2590-2599):

```python
with httpx.Client(timeout=10.0) as client:
    response = client.post(
        f"{args.endpoint_url}/chat/completions",
        json={
            "model": args.model,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1,
        },
    )
```

Actual requests (lines 2690-2697):

```python
response = client.post(
    f"{args.endpoint_url}/chat/completions",
    json={
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    },
)
```

Neither passes any `Authorization` header. The argument parser has no `--api-key` option.

### Impact

- RunPod vLLM template sets `VLLM_API_KEY`, so every request returned Unauthorized
- Required manual proxy to inject bearer token
- Most production vLLM endpoints require authentication
- **Blocks real users from validating against their own secured endpoints**

## Status

Both findings are **unfixed**. No changes have been made to the code.
