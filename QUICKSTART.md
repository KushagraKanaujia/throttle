# Throttle Quickstart

This takes you from nothing to a simulated cost comparison, a live cache hit,
and a measured $/M-token figure in about five minutes. You don't need a GPU.

## Prerequisites

- Python 3.11 or later
- Step 1 needs nothing else
- Steps 2 and 3 need [Ollama](https://ollama.com/download) running locally
  with `ollama pull llama3.2:3b`

## Install

```bash
git clone https://github.com/KushagraKanaujia/throttle.git
cd throttle
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[embeddings]'
throttle --version
```

The PyPI release (`pipx install throttle-pro`, currently 0.3.0) predates the
agent session profiler. Install from source to get every command below.

## 1. Run the simulator demo

```bash
throttle demo
```

This command:
- generates a sample workload of 300 requests,
- simulates vLLM-style continuous batching at `max_num_seqs` 128 (baseline)
  and 256 (tuned),
- prints a cost table with confidence intervals and a sensitivity analysis,
- finishes in about a second and makes no network calls.

Output excerpt:

```text
Configuration being compared:
  Parameter: max_num_seqs
  Baseline: 128 concurrent sequences
  Tuned:    256 concurrent sequences
...
Metric                                       Baseline        Tuned        Delta
--------------------------------------------------------------------------------
Wall clock time (seconds)                       77.87        63.14       -14.73
GPU hours                                    0.021630     0.017538    -0.004092
Input cost ($/M tokens)                          0.35         0.28        -0.07
Output cost ($/M tokens)                         0.11         0.09        -0.02
Total cost ($)                                 0.0324       0.0263      -0.0061
...
All values above are [SIMULATED] - they depend entirely on assumed
throughput parameters, not real hardware measurements.
```

Every number in this output is simulated from assumed throughput. It shows
how the cost model works. It is not a measurement.

## 2. Put the caching proxy in front of Ollama

Terminal 1:

```bash
throttle proxy --backend-url http://localhost:11434 --port 8090 \
  --enable-cache --enable-embeddings
```

Terminal 2:

```bash
ask() { curl -s -o /dev/null -w "HTTP %{http_code}  %{time_total}s\n" \
  localhost:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"llama3.2:3b\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":60,\"temperature\":0}"; }
ask "Give me three tips for writing a good README."
ask "Give me three tips for writing a good README."
ask "give me three tips for writing a great README"
ask "Give me three tips for writing a good cover letter."
curl -s localhost:8090/health
```

On a laptop, the first call took about 1.75 s. The exact repeat took about
1 ms and the reworded prompt about 9.5 ms. The unrelated prompt went to the
model again. `/health` reported `"backend_calls":2` for the four requests.
Your timings will differ.

## 3. Measure real cost per million tokens

```bash
throttle cost \
  --endpoint-url http://localhost:11434/v1 \
  --model llama3.2:3b \
  --gpu-hourly-rate 1.50 \
  --num-requests 5
```

This sends real requests, measures throughput and timing, and converts the
results to dollars per million tokens at the hourly rate you supply. On a
laptop there is no real hourly price, so $1.50 is an assumption. The input
and output $/M figures each charge the full run cost to that token type, so
don't add them together.

## Next steps

- Profile an agent: `throttle proxy ... --enable-session-tracking`, then
  `throttle sessions` (see the README's "Agent session profiler" section)
- Before sending traffic to a real endpoint, preview it: `throttle plan --help`
- For decision-grade config comparisons, see `throttle golden --help` and
  [docs/GOLDEN_PROTOCOL.md](docs/GOLDEN_PROTOCOL.md)

## Getting help

```bash
throttle --help
throttle demo --help
throttle cost --help
throttle proxy --help
```

## Troubleshooting

### "Failed to connect to endpoint"

Make sure your inference server is running:

```bash
ollama serve
curl http://localhost:11434/api/tags
```

### The proxy says embeddings are unavailable

The embedding tier needs the `embeddings` extra (`pip install -e '.[embeddings]'`)
and a one-time download of `sentence-transformers/all-MiniLM-L6-v2`. Without
them, the proxy falls back to exact and lexical matching, so reworded prompts
will miss.

### Simulator costs differ from real measurements

That is expected. The simulator uses assumed throughput values. Use
`throttle cost` for measured numbers.
