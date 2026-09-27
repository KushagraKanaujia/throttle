# proxy and sessions

*An optional caching proxy and agent-session profiler.*

## Caching proxy

```bash
throttle proxy --backend-url http://localhost:11434 --port 8090 --enable-cache
```

An OpenAI-compatible server (`/v1/chat/completions`, `/health`) that sits in front of your inference server: change your client's base URL to it. The cache is **off unless you pass `--enable-cache`**. With it on, lookups try an exact match, then a lexical match (Jaccard token overlap of at least 0.85), always under the same model and sampling parameters. Verified against Ollama; vLLM, SGLang and LMDeploy are expected to work but not yet verified through the proxy on a GPU.

> **Warning:** The semantic tier (`--enable-embeddings`) is a separate opt-in with a known false-match risk: cosine similarity encodes topic, not polarity, so "Is it safe to use eval in Python?" and "Is it dangerous to use eval in Python?" can match. It stays off unless you turn it on.

## Agent session profiler

```bash
throttle proxy --backend-url http://localhost:11434 --port 8091 --enable-session-tracking
throttle sessions
throttle sessions <session-id>
```

Records every turn's timing and token counts to `~/.throttle/sessions.db` and shows where an agent's wall-clock time goes (generating vs waiting on the client), an estimate of redundant prefill, and rule-based suggestions naming backend flags to check. Prompt and completion text are never stored, only content hashes, timings and token counts.

<details><summary>Full `throttle proxy --help` output (0.4.2)</summary>

```text
usage: throttle proxy [-h] --backend-url BACKEND_URL [--host HOST]
                      [--port PORT] [--enable-cache]
                      [--cache-ttl-seconds CACHE_TTL_SECONDS]
                      [--cache-max-size CACHE_MAX_SIZE]
                      [--cache-similarity-threshold CACHE_SIMILARITY_THRESHOLD]
                      [--enable-embeddings | --no-embeddings]
                      [--embedding-threshold EMBEDDING_THRESHOLD]
                      [--embedding-max-entries-scanned EMBEDDING_MAX_ENTRIES_SCANNED]
                      [--backend-timeout-seconds BACKEND_TIMEOUT_SECONDS]
                      [--enable-session-tracking]

Start a lightweight HTTP proxy that sits in front of a real inference backend
and caches responses using semantic similarity matching. Verified compatible
with Ollama. Expected compatible with vLLM, SGLang, LMDeploy, and other
OpenAI-compatible servers (GPU verification pending, see
validation/gpu_backend_verification.sh).

options:
  -h, --help            show this help message and exit
  --backend-url, --url, --endpoint-url BACKEND_URL
                        backend inference server URL, with or without /v1
                        (e.g., http://localhost:8000; --url is accepted too)
  --host HOST           proxy server host (default: 127.0.0.1)
  --port PORT           proxy server port (default: 8080)
  --enable-cache        enable semantic similarity caching
  --cache-ttl-seconds CACHE_TTL_SECONDS
                        cache entry TTL in seconds (default: 3600)
  --cache-max-size CACHE_MAX_SIZE
                        maximum cache entries (default: 1000)
  --cache-similarity-threshold CACHE_SIMILARITY_THRESHOLD
                        Jaccard similarity threshold (0.0-1.0, default: 0.85)
  --enable-embeddings   enable ONNX semantic embedding tier (requires
                        embeddings extra); default: OFF
  --no-embeddings       force disable embeddings even if extra is installed
  --embedding-threshold EMBEDDING_THRESHOLD
                        semantic embedding similarity threshold (0.0-1.0,
                        default: 0.95)
  --embedding-max-entries-scanned EMBEDDING_MAX_ENTRIES_SCANNED
                        maximum cache entries scanned for embedding match
                        (default: 256)
  --backend-timeout-seconds BACKEND_TIMEOUT_SECONDS
                        backend request timeout in seconds (default: 120.0).
                        NOTE: this value is not evidence-based; 30 seconds
                        risks killing cold model loads and long generations.
  --enable-session-tracking
                        enable agent session profiling (records to
                        ~/.throttle/sessions.db)
```

</details>

<details><summary>Full `throttle sessions --help` output (0.4.2)</summary>

```text
usage: throttle sessions [-h] [--since SINCE] [--db DB] [--limit LIMIT]
                         [session_id]

Analyze multi-turn agent sessions captured by the proxy. Shows where wall
clock time is spent and provides actionable findings to improve performance.

positional arguments:
  session_id     Show detailed view of a specific session (full ID or unique
                 prefix)

options:
  -h, --help     show this help message and exit
  --since SINCE  Time window to query (e.g., '30m', '24h', '7d', '2w').
                 Default: 24h
  --db DB        Path to the session database (default:
                 ~/.throttle/sessions.db)
  --limit LIMIT  Maximum number of sessions to show (default: 50)
```

</details>
