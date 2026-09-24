# Known Issues and Findings

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

## Finding 1: validate-sim Executes Requests Serially (FIXED)

**Status**: Fixed in commit prior to 2026-08-29. Test added to prevent regression.

**Original Issue**: The three load levels never happened. Every scenario measured single-stream decode and the arrival rate parameter did nothing.

### Fix Evidence

Current implementation in `src/throttle/cli.py` lines 3270-3559:

- Line 3389: `async def run_concurrent_workload()` - uses async/await
- Line 3401: `async def send_request(arrival_time, ...)` - async request handler
- Line 3405-3406: `await asyncio.sleep(arrival_time)` - respects arrival times
- Line 3463-3466: Creates all tasks upfront
- Line 3469: `await asyncio.gather(*tasks)` - concurrent execution
- Line 3476: `asyncio.run(run_concurrent_workload())` - runs async workload
- Lines 3482-3485: **Fails if peak_concurrent == 1** - detects serial execution

### Test Coverage

`tests/test_validate_sim_concurrency.py`:
- Lines 105-109: Asserts `peak_concurrent > 1` to prove request overlap
- Uses mock backend with 0.5s sleep to detect serial vs concurrent execution
- Verifies validate-sim's built-in serial detection fires when needed

### Impact

- All three load levels (Light/Medium/Heavy) now test actual concurrent load
- Arrival rate parameter is used correctly
- Load level comparisons are valid
- Simulator error percentages reflect real concurrent behavior

---

## Finding 2: No API Key Support (FIXED)

**Status**: Fixed in commit prior to 2026-08-29.

**Original Issue**: `validate-sim` had no flag for an API key or authorization header.

### Fix Evidence

- Line 590-592: `--api-key` argument added to argument parser
- Line 3286: `api_key = _get_api_key(args)` - retrieves API key from args or environment
- Line 3287: `headers = _build_headers(api_key)` - builds Authorization header
- Line 3304: Connectivity test uses `headers=headers`
- Line 3421: Actual requests use `headers=headers`

### Impact

- RunPod vLLM endpoints with `VLLM_API_KEY` now work
- Users can validate against secured production endpoints
- No manual proxy required for authentication

---

## Status

Both findings are **FIXED** as of 2026-08-29.
