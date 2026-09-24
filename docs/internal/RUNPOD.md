# RunPod A100 Validation Instructions

Copy-paste these commands verbatim on a fresh RunPod A100 instance.

## Known RunPod Template Issues

**The RunPod vLLM template has three surprises that will block validation:**

1. **No git installed** - Cannot clone the repository without installing git first
2. **VLLM_API_KEY set** - All requests return Unauthorized without an API key
3. **--enforce-eager mode** - Disables CUDA graphs, significantly reduces decode throughput

These instructions work around all three issues.

## Step 1: Install Python, git, and dependencies

```bash
# Update package list
sudo apt-get update

# Install Python 3.11, pip, and git
sudo apt-get install -y python3.11 python3.11-venv python3-pip git

# Verify installation
python3.11 --version
git --version
```

## Step 2: Install vLLM

```bash
# Create venv for vLLM
python3.11 -m venv ~/vllm-env
source ~/vllm-env/bin/activate

# Install vLLM
pip install vllm

# Verify installation
python -c "import vllm; print(vllm.__version__)"
```

## Step 3: Start vLLM server with a small model

```bash
# Still in vllm-env
# Start vLLM with Qwen2.5-0.5B-Instruct (small, fast to download)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.9 \
  &

# Wait for model to load (30-60 seconds)
sleep 60

# Test that vLLM is running
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }'
```

If you see a JSON response with "choices", vLLM is running correctly.

## Step 4: Install throttle in a separate venv

```bash
# Deactivate vllm-env
deactivate

# Create new venv for throttle
python3.11 -m venv ~/throttle-env
source ~/throttle-env/bin/activate

# Install throttle directly from GitHub
pip install git+https://github.com/KushagraKanaujia/throttle.git

# Verify installation
throttle --help
```

## Step 5: Run validation

```bash
# Still in throttle-env
# Run validator against local vLLM
# GPU hourly rate: use your actual RunPod rate (typically $0.79-$1.14/hr for A100)
# --api-key uses $VLLM_API_KEY set by the RunPod template
throttle validate-sim \
  --endpoint-url http://localhost:8000/v1 \
  --model "Qwen/Qwen2.5-0.5B-Instruct" \
  --gpu-hourly-rate 1.00 \
  --api-key "$VLLM_API_KEY"

# This will:
# - Test connection to vLLM
# - Run 3 workloads (light, medium, heavy)
# - Compare simulator predictions to real measurements
# - Write validation_YYYYMMDD_HHMMSS.json with full results
```

## Step 6: Retrieve the output file

The validation command writes a JSON file named `validation_YYYYMMDD_HHMMSS.json` in the current directory.

```bash
# List validation files
ls -lh validation_*.json

# View the latest one
cat $(ls -t validation_*.json | head -1) | python3 -m json.tool
```

## Expected output structure

The validation will print a table for each load level:

```
Running Light load (1 req/sec, 20 requests)...

  Metric                      Simulated      Measured    Error
  -------------------------  -----------  -----------  --------
  Wall clock (s)                    X.XX         Y.YY   +/-Z.Z%
  Input tok/sec                     X.XX         Y.YY   +/-Z.Z%
  Output tok/sec                    X.XX         Y.YY   +/-Z.Z%
  TTFT (s)                           N/A         Y.YYY      N/A
  $/M input tokens                  X.XX         Y.YY   +/-Z.Z%
```

The JSON file contains:
- Endpoint URL and model name
- GPU hourly rate used
- All simulator parameters (prefill/decode throughput, max_num_seqs, etc.)
- Per-scenario results with simulated vs measured values
- Error percentages for each metric
- Raw per-request timings for all requests

## Troubleshooting

### vLLM fails to start

```bash
# Check CUDA is available
nvidia-smi

# Check GPU memory
nvidia-smi --query-gpu=memory.free --format=csv

# If out of memory, try smaller model or reduce --gpu-memory-utilization
```

### throttle validate-sim fails to connect

```bash
# Verify vLLM is listening
curl http://localhost:8000/health

# Check vLLM logs
# (vLLM outputs to the terminal where you started it)

# If vLLM crashed, restart it and wait longer for model load
```

### Validation JSON not created

Check the current directory for permission issues:

```bash
pwd
ls -la
# Ensure you can write to the current directory
```

## Clean shutdown

```bash
# Stop vLLM
pkill -f vllm

# Deactivate venv
deactivate

# Remove venvs if done
rm -rf ~/vllm-env ~/throttle-env
```

## Notes for different models

If using a different model (e.g., Llama-3.1-8B):

```bash
# Larger models take longer to download and load
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  &

# Wait 5-10 minutes for download and loading
sleep 600

# Test
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }'

# Then run validation with the model name
throttle validate-sim \
  --endpoint-url http://localhost:8000/v1 \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --gpu-hourly-rate 1.00
```
