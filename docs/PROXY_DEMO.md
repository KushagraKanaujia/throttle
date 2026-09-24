# Throttle Proxy - OpenAI-Compatible Caching Proxy

The Throttle proxy is a lightweight HTTP server that sits in front of real inference backends and caches responses using lexical similarity matching (Jaccard token-overlap).

**Verified compatible:** Ollama (CI integration tests pass)
**Expected compatible:** vLLM, SGLang, LMDeploy (GPU verification pending - see `validation/gpu_backend_verification.sh`)

## Features

- **OpenAI-Compatible API**: Drop-in replacement for `/v1/chat/completions` endpoint
- **Similarity Caching**: Uses Jaccard token-overlap for prompt matching (lexical, not semantic - see Limitations)
- **Streaming Support**: Handles both streaming and non-streaming requests
- **Real-time Cache**: In-memory cache with configurable TTL and max size
- **Zero Backend Changes**: Works with any OpenAI-compatible backend

## Quick Start

Follow these steps to go from clone to a verified cache hit:

### 1. Install Throttle and Start Ollama

```bash
# Install Ollama from https://ollama.com/download if not already installed
ollama pull llama3.2:1b
ollama serve  # Leave running in background
```

### 2. Start the Proxy

The proxy appends `/v1/chat/completions` to the backend URL automatically, so provide the base URL without `/v1`:

```bash
throttle proxy \
  --backend-url http://localhost:11434 \
  --enable-cache \
  --port 8080
```

You should see:
```
Starting Throttle proxy server on 127.0.0.1:8080
Backend: http://localhost:11434
Cache enabled: True
```

### 3. Send First Request (Cache Miss)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 50
  }'
```

Expected output (backend response):
```json
{
  "id": "...",
  "object": "chat.completion",
  "created": ...,
  "model": "llama3.2:1b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "The capital of France is Paris."
    },
    "finish_reason": "stop"
  }],
  ...
}
```

### 4. Check Cache Stats (1 Miss)

```bash
curl http://localhost:8080/health
```

Expected output:
```json
{
  "status": "ok",
  "cache_enabled": true,
  "cache_stats": {
    "hits": 0,
    "misses": 1,
    "evictions": 0,
    "backend_calls": 1
  }
}
```

### 5. Send Second IDENTICAL Request (Cache Hit)

**IMPORTANT**: All scope parameters must match exactly for a cache hit:
- Same `model`
- Same `messages` (exact text)
- Same `max_tokens`
- Same `temperature` (if specified)
- Any other sampling parameters

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 50
  }'
```

Expected output: Same response as before, but served from cache (sub-millisecond latency).

### 6. Verify Cache Hit

```bash
curl http://localhost:8080/health
```

Expected output shows `"hits": 1`:
```json
{
  "status": "ok",
  "cache_enabled": true,
  "cache_stats": {
    "hits": 1,
    "misses": 1,
    "evictions": 0,
    "backend_calls": 1
  }
}
```

**Success!** You've verified a cache hit. The backend received only 1 request despite 2 client requests.

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--backend-url` | *required* | Backend inference server URL |
| `--host` | `127.0.0.1` | Proxy server host |
| `--port` | `8080` | Proxy server port |
| `--enable-cache` | `false` | Enable semantic caching |
| `--cache-ttl-seconds` | `3600.0` | Cache entry TTL in seconds |
| `--cache-max-size` | `1000` | Maximum cache entries (FIFO eviction) |
| `--cache-similarity-threshold` | `0.85` | Jaccard similarity threshold (0.0-1.0) |

## How It Works

### Cache Miss Flow
1. Client sends request to proxy
2. Proxy checks cache using Jaccard similarity
3. **No match found** → Forward request to backend
4. Backend processes request and returns response
5. Proxy caches response and returns to client

### Cache Hit Flow
1. Client sends request to proxy
2. Proxy checks cache using Jaccard similarity
3. **Match found** (similarity ≥ threshold)
4. Proxy returns cached response immediately
5. No backend request needed

### Streaming Behavior
- **Cache miss**: Stream is passed through from backend in real-time
- **Cache hit**: Cached response is "fake-streamed" to maintain client compatibility

## Performance

Cache hit latency is sub-millisecond (served from memory). Cache miss latency equals whatever the backend inference costs, since the request is forwarded to the backend. Hit rate depends on traffic patterns, similarity threshold tuning, and the Jaccard limitation on paraphrases (see Limitations below).

## Use Cases

1. **Development/Testing**: Avoid repeatedly calling expensive inference APIs
2. **Chatbot FAQ**: Cache common questions with semantic matching
3. **API Cost Reduction**: Reduce backend load for similar queries
4. **Latency Optimization**: Sub-10ms responses for cached prompts

## Limitations

### Jaccard Similarity Cannot Match Paraphrases

The cache uses **lexical** Jaccard token-overlap similarity (threshold 0.85), NOT semantic embeddings. This means **paraphrases will miss the cache** despite having identical meaning.

**Concrete example that MISSES cache:**
- Request 1: `"optimize PostgreSQL queries"`
- Request 2: `"optimize database queries in PostgreSQL"`

These have Jaccard similarity ~0.64 (below the 0.85 threshold) because the token sets differ significantly, even though the semantic meaning is identical. The second request will hit the backend.

**What WILL hit cache:**
- Request 1: `"What is the capital of France?"`
- Request 2: `"What is the capital of France?"` (exact match)

**Why not use semantic embeddings?** Lexical similarity is fast (sub-millisecond) and requires no model loading. A semantic embedding approach (ONNX-based two-tier matching) would catch paraphrases but adds latency and complexity.

### Cache Scope: All Parameters Must Match Exactly

Changing ANY of these parameters creates a NEW cache scope (no hit):
- `model` - different models never share cache entries
- `max_tokens` - affects response length, separate scope
- `temperature` - affects randomness, separate scope
- `top_p`, `frequency_penalty`, `presence_penalty` - all create separate scopes
- Any custom backend parameters

**Example scope miss:**
```bash
# Request 1: max_tokens=50
curl ... -d '{"model": "llama3.2:1b", "messages": [...], "max_tokens": 50}'

# Request 2: max_tokens=100 - MISSES cache despite same messages
curl ... -d '{"model": "llama3.2:1b", "messages": [...], "max_tokens": 100}'
```

The `stream` parameter (true/false) does NOT create separate scopes - a cached response can be served as either streaming or non-streaming.

### Error Behavior

**Backend unreachable or timeout:**
- Client receives HTTP 504 Gateway Timeout
- Error message: `{"error": "Backend timeout: ..."}`
- Failed responses are NOT cached

**Backend returns error (4xx/5xx):**
- Proxy forwards the exact status code and error body from backend
- Error responses are NOT cached
- Example: Backend 404 → Client sees 404 with backend's error message

**Cache full (max_size reached):**
- Oldest entries are evicted (FIFO)
- No client-visible error
- Check `/health` for `"evictions"` count

### Other Limitations

- **In-memory only**: Cache is lost when proxy restarts
- **Single backend**: No multi-backend routing or load balancing
- **No persistence**: No disk-based cache or distributed caching
- **No authentication**: Proxy does not validate API keys (passes through to backend)

## Difference from Benchmark Cache

The proxy mode is **fundamentally different** from the benchmark cache:

| Feature | Proxy Cache | Benchmark Cache |
|---------|-------------|-----------------|
| Clients | **External HTTP clients** (curl, OpenAI SDK, etc.) | Throttle's internal load generator |
| Use case | Production traffic caching | Benchmark acceleration |
| Visibility | All clients benefit from cache | Only Throttle benchmark requests |
| Control | Separate server process | Integrated into benchmark runs |

## Example: Validating Cache Effectiveness

```python
import time
import httpx

proxy_url = "http://localhost:8080/v1/chat/completions"
prompt = "Explain quantum computing in simple terms"

# Measure cache miss
start = time.time()
response1 = httpx.post(proxy_url, json={
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 100
})
miss_time = time.time() - start
print(f"Cache miss: {miss_time*1000:.1f}ms")

# Measure cache hit
start = time.time()
response2 = httpx.post(proxy_url, json={
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 100
})
hit_time = time.time() - start
print(f"Cache hit: {hit_time*1000:.1f}ms")
print(f"Speedup: {miss_time/hit_time:.1f}x")
```

## Next Steps

- Try different similarity thresholds to tune cache effectiveness
- Monitor cache stats to track hit rates
- Test with your production traffic patterns
- Compare cache hit latency vs backend latency

For questions or issues, see the main Throttle documentation.
