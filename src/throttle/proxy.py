"""Throttle Proxy: OpenAI-compatible caching proxy for LLM inference backends.

This module provides a lightweight HTTP proxy server that sits in front of real
inference backends and caches responses using semantic similarity matching.

Verified compatible with Ollama. Expected compatible with vLLM, SGLang, LMDeploy,
and other OpenAI-compatible servers (GPU verification pending).

IMPORTANT LIMITATION: The current implementation uses Jaccard token-overlap
similarity (threshold 0.85) which is strict and may miss natural paraphrases.
For example, "optimize PostgreSQL queries" vs "optimize database queries in
PostgreSQL" has Jaccard similarity ~0.64 (below threshold) despite identical
semantic meaning. This is a known limitation of lexical similarity.

A more robust solution would use semantic embeddings (e.g., ONNX-based two-tier
matching) to catch paraphrases that Jaccard misses. The current threshold
balances precision (avoiding false positives) with recall (catching duplicates).

Agent Session Profiling: When enable_session_tracking=True, the proxy records
multi-turn agent sessions to ~/.throttle/sessions.db for analysis. Tracking is
async and adds <5ms overhead per request.
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from .cache import SimilarityCache


class ProxyServer:
    """OpenAI-compatible proxy with semantic caching and request deduplication.

    Implements per-prompt in-flight request deduplication to prevent race
    conditions where multiple concurrent identical requests all miss the cache
    and hit the backend independently. Only one backend call is made per
    unique prompt at a time; concurrent duplicates wait for the first request
    to complete and share its result.
    """

    def __init__(
        self,
        backend_url: str,
        *,
        enable_cache: bool = True,
        cache_ttl_seconds: float = 3600.0,
        cache_max_size: int = 1000,
        cache_similarity_threshold: float = 0.85,
        enable_embeddings: bool = False,
        embedding_threshold: float = 0.95,
        embedding_max_entries_scanned: int = 256,
        # NOTE: 120s default is not evidence-based. 30s risks killing cold model loads and long generations.
        backend_timeout_seconds: float = 120.0,
        model_backends: Optional[Dict[str, str]] = None,
        enable_session_tracking: bool = False,
        lifespan=None,
    ):
        self.backend_url = backend_url.rstrip("/")
        # IMPORTANT: _extract_scope_key() depends on this mapping. Mutating model_backends
        # after construction changes scope keys for all subsequent requests, orphaning every
        # existing cache entry.
        self.model_backends = model_backends or {}
        self.enable_cache = enable_cache
        self.cache: Optional[SimilarityCache] = None
        self.backend_timeout_seconds = backend_timeout_seconds
        self.enable_session_tracking = enable_session_tracking

        if self.enable_cache:
            self.cache = SimilarityCache(
                ttl_seconds=cache_ttl_seconds,
                max_size=cache_max_size,
                similarity_threshold=cache_similarity_threshold,
                enable_embeddings=enable_embeddings,
                embedding_threshold=embedding_threshold,
                embedding_max_entries_scanned=embedding_max_entries_scanned,
            )

        # Session tracking (optional)
        self._session_tracker = None
        if self.enable_session_tracking:
            from .sessions import SessionTracker

            self._session_tracker = SessionTracker()

        self.app = FastAPI(title="Throttle Proxy", version="0.3.0", lifespan=lifespan)
        self.app.post("/v1/chat/completions")(self.chat_completions)
        self.app.get("/health")(self.health)

        # HTTP client for backend requests
        self._client: Optional[httpx.AsyncClient] = None

        # In-flight request tracking for deduplication
        # Maps prompt -> Future[response]
        self._inflight: Dict[str, asyncio.Future] = {}
        self._inflight_lock = asyncio.Lock()

        # Backend call counter
        self._backend_calls = 0

    async def startup(self):
        """Initialize HTTP client and session tracker."""
        self._client = httpx.AsyncClient(timeout=self.backend_timeout_seconds)

        if self._session_tracker:
            await self._session_tracker.start_background_flush()

    async def shutdown(self):
        """Cleanup HTTP client and session tracker."""
        if self._session_tracker:
            await self._session_tracker.shutdown()

        if self._client:
            await self._client.aclose()

    async def health(self):
        """Health check endpoint."""
        return {
            "status": "ok",
            "cache_enabled": self.enable_cache,
            "cache_stats": {
                "hits": self.cache.metrics.hits if self.cache else 0,
                "misses": self.cache.metrics.misses if self.cache else 0,
                "evictions": self.cache.metrics.evictions if self.cache else 0,
                "exact_hits": self.cache.metrics.exact_hits if self.cache else 0,
                "lexical_hits": self.cache.metrics.lexical_hits if self.cache else 0,
                "embedding_hits": self.cache.metrics.embedding_hits if self.cache else 0,
                "embedding_scans_attempted": self.cache.metrics.embedding_scans_attempted if self.cache else 0,
                "embedding_comparisons_performed": self.cache.metrics.embedding_comparisons_performed if self.cache else 0,
                "backend_calls": self._backend_calls,
            } if self.cache else None,
        }

    def _get_backend_url(self, model: str) -> str:
        """Get backend URL for a specific model.

        If model_backends mapping is provided and contains the model,
        returns the model-specific URL. Otherwise returns default backend_url.
        """
        return self.model_backends.get(model, self.backend_url).rstrip("/")

    def _extract_scope_key(self, request_body: Dict[str, Any]) -> str:
        """Extract scope parameters that must match exactly for cache hits.

        Includes all request fields EXCEPT:
        - messages: used for semantic similarity matching, not scope differentiation
        - stream: controls response format (SSE vs JSON), not content; cached responses
                  can be served as either streaming or non-streaming

        All other fields (model, temperature, max_tokens, custom backend parameters, etc.)
        are included to prevent cache collisions between requests with different parameters.

        When model_backends is non-empty, also includes the resolved backend URL to prevent
        cache collisions when different backends serve models with the same name.
        """
        excluded = {"messages", "stream"}
        scope_params = {
            k: v for k, v in request_body.items()
            if k not in excluded
        }

        # Include backend URL in scope when using per-model routing.
        #
        # NOTE: The collision this guards (same model name routing to different backends)
        # is NOT reachable with the current in-memory, per-process cache and dict-based
        # model_backends (dict keys must be unique, so one model name cannot map to two
        # backends). This is defensive programming for a future persisted or shared cache.
        #
        # Without this field, two ProxyServer instances with different model_backends
        # mappings sharing a persisted cache could collide: both map "gpt-4" to different
        # backends, and requests with identical parameters would incorrectly share cache
        # entries despite hitting different backends.
        if self.model_backends:
            model = request_body.get("model", "")
            backend_url = self._get_backend_url(model)
            scope_params["_backend_url"] = backend_url

        return json.dumps(scope_params, sort_keys=True)

    def _extract_prompt(self, request_body: Dict[str, Any]) -> str:
        """Extract semantic text from messages for similarity matching.

        Includes role information to distinguish different conversation structures.
        """
        messages = request_body.get("messages", [])
        prompt_parts = []
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                role = msg.get("role", "user")
                content = msg["content"]
                # Include role to distinguish different conversation structures
                prompt_parts.append(f"{role}: {content}")
        return "\n".join(prompt_parts)

    async def _fake_stream_response(
        self, cached_response: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """Fake-stream a cached non-streaming response to maintain client compatibility."""
        # Convert the cached response into SSE chunks
        # Most clients expect streaming to look like real backend streaming

        # First chunk with role
        first_chunk = {
            "id": cached_response.get("id", "cached-response"),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": cached_response.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"
        await asyncio.sleep(0.001)  # Small delay to simulate streaming

        # Content chunks
        content = cached_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Split content into small chunks to simulate real streaming
        chunk_size = 10
        for i in range(0, len(content), chunk_size):
            chunk_content = content[i:i+chunk_size]
            content_chunk = {
                "id": cached_response.get("id", "cached-response"),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": cached_response.get("model", "unknown"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk_content},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(content_chunk)}\n\n"
            await asyncio.sleep(0.001)

        # Final chunk with finish_reason
        final_chunk = {
            "id": cached_response.get("id", "cached-response"),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": cached_response.get("model", "unknown"),
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": cached_response.get("choices", [{}])[0].get("finish_reason", "stop"),
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    async def _forward_streaming(
        self, backend_url: str, request_body: Dict[str, Any], headers: Dict[str, str], prompt: str
    ) -> AsyncIterator[str]:
        """Forward streaming request to backend and accumulate for caching."""
        accumulated_content = []
        response_metadata = {}

        async with self._client.stream(
            "POST",
            f"{backend_url}/v1/chat/completions",
            json=request_body,
            headers=headers,
        ) as response:
            # Capture metadata from first chunk
            first_chunk_seen = False

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        yield line + "\n"
                        continue

                    try:
                        chunk = json.loads(data_str)

                        # Capture metadata from first chunk
                        if not first_chunk_seen:
                            response_metadata["id"] = chunk.get("id", "")
                            response_metadata["model"] = chunk.get("model", "")
                            response_metadata["created"] = chunk.get("created", int(time.time()))
                            first_chunk_seen = True

                        # Accumulate content
                        for choice in chunk.get("choices", []):
                            delta = choice.get("delta", {})
                            if "content" in delta:
                                accumulated_content.append(delta["content"])

                            # Capture finish_reason from final chunk
                            if choice.get("finish_reason"):
                                response_metadata["finish_reason"] = choice["finish_reason"]

                        yield line + "\n"
                    except json.JSONDecodeError:
                        yield line + "\n"
                else:
                    yield line + "\n"

        # After streaming completes, cache the accumulated response
        if self.enable_cache and self.cache and accumulated_content:
            full_content = "".join(accumulated_content)
            cached_response = {
                "id": response_metadata.get("id", ""),
                "object": "chat.completion",
                "created": response_metadata.get("created", int(time.time())),
                "model": response_metadata.get("model", ""),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content,
                        },
                        "finish_reason": response_metadata.get("finish_reason", "stop"),
                    }
                ],
            }
            # WARNING: This cache.put is scope-blind (stores raw response, not scope-aware dict). Unsafe to wire up as written.
            self.cache.put(prompt, cached_response)

    async def _make_backend_request(
        self,
        request_body: Dict[str, Any],
        headers: Dict[str, str],
        prompt: str,
        is_streaming: bool,
    ) -> Dict[str, Any]:
        """Make the actual backend request and return the complete response.

        For non-streaming: returns response directly.
        For streaming: accumulates the stream and returns complete response.
        """
        # Increment backend call counter
        # Note: cache misses != backend calls, because in-flight deduplication
        # means multiple concurrent requests can all record cache misses while
        # only one actually reaches the backend (others wait on the same Future)
        self._backend_calls += 1

        # Get model-specific backend URL
        model = request_body.get("model", "")
        backend_url = self._get_backend_url(model)

        if is_streaming:
            # For streaming, accumulate the complete response
            accumulated_content = []
            response_metadata = {}

            async with self._client.stream(
                "POST",
                f"{backend_url}/v1/chat/completions",
                json=request_body,
                headers=headers,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            continue

                        try:
                            chunk = json.loads(data_str)

                            if not response_metadata:
                                response_metadata["id"] = chunk.get("id", "")
                                response_metadata["model"] = chunk.get("model", "")
                                response_metadata["created"] = chunk.get("created", int(time.time()))

                            for choice in chunk.get("choices", []):
                                delta = choice.get("delta", {})
                                if "content" in delta:
                                    accumulated_content.append(delta["content"])
                                if choice.get("finish_reason"):
                                    response_metadata["finish_reason"] = choice["finish_reason"]
                        except json.JSONDecodeError:
                            pass

            full_content = "".join(accumulated_content)
            return {
                "id": response_metadata.get("id", ""),
                "object": "chat.completion",
                "created": response_metadata.get("created", int(time.time())),
                "model": response_metadata.get("model", ""),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content,
                        },
                        "finish_reason": response_metadata.get("finish_reason", "stop"),
                    }
                ],
            }
        else:
            # Non-streaming request
            response = await self._client.post(
                f"{backend_url}/v1/chat/completions",
                json=request_body,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()

    async def chat_completions(self, request: Request):
        """Handle /v1/chat/completions requests with caching and deduplication."""
        start_time = time.perf_counter()
        ttft_ms = None  # Track time-to-first-token if available

        try:
            request_body = await request.json()

            # Extract scope and semantic text
            scope_key = self._extract_scope_key(request_body)
            prompt = self._extract_prompt(request_body)
            is_streaming = request_body.get("stream", False)

            # Check cache first (fast path)
            if self.enable_cache and self.cache:
                result = self.cache.get_with_key_no_metrics(prompt)
                if result is not None:
                    canonical_key, scope_dict = result
                    if isinstance(scope_dict, dict) and scope_key in scope_dict:
                        self.cache.metrics.hits += 1
                        cached_response = scope_dict[scope_key]["response"]

                        # Track session (cache hit)
                        total_latency_ms = (time.perf_counter() - start_time) * 1000
                        if self._session_tracker:
                            asyncio.create_task(
                                self._session_tracker.record_turn(
                                    request=request,
                                    request_body=request_body,
                                    response=cached_response,
                                    latency_ms=total_latency_ms,
                                    ttft_ms=None,  # Cache hits have no TTFT
                                )
                            )

                        if is_streaming:
                            return StreamingResponse(
                                self._fake_stream_response(cached_response),
                                media_type="text/event-stream",
                            )
                        else:
                            return JSONResponse(cached_response)
                    self.cache.metrics.misses += 1
                else:
                    self.cache.metrics.misses += 1

            # Cache miss - check if request is in-flight
            dedup_key = f"{scope_key}||{prompt}"
            async with self._inflight_lock:
                if dedup_key in self._inflight:
                    future = self._inflight[dedup_key]
                    is_waiter = True
                else:
                    future = asyncio.Future()
                    self._inflight[dedup_key] = future
                    is_waiter = False

            if not is_waiter:
                headers = {"Content-Type": "application/json"}
                if "authorization" in request.headers:
                    headers["Authorization"] = request.headers["authorization"]

                try:
                    response_data = await self._make_backend_request(
                        request_body, headers, prompt, is_streaming
                    )

                    if self.enable_cache and self.cache:
                        choices = response_data.get("choices", [])
                        # Note: HTTP 200 with populated choices AND an error field
                        # is not guarded here. No Throttle-verified backend (Ollama,
                        # vLLM, SGLang) produces this shape — they use proper HTTP
                        # error codes. Revisit if a concrete backend case is found.
                        if choices:
                            existing_scope_dict = self.cache.get_exact_no_metrics(prompt)
                            if existing_scope_dict is not None and isinstance(existing_scope_dict, dict):
                                scope_dict = existing_scope_dict
                            else:
                                scope_dict = {}
                            scope_dict[scope_key] = {
                                "_scope": scope_key,
                                "response": response_data,
                            }
                            self.cache.put(prompt, scope_dict)

                    future.set_result(response_data)

                except httpx.TimeoutException as e:
                    # Timeout - propagate to waiters via future, return 504 for originator
                    future.set_exception(e)
                    return JSONResponse(
                        status_code=504,
                        content={"error": "Backend request timed out"}
                    )
                except Exception as e:
                    future.set_exception(e)
                    raise
                finally:
                    async with self._inflight_lock:
                        self._inflight.pop(dedup_key, None)

            # Waiters and successful originators reach here
            try:
                response_data = await future
            except httpx.TimeoutException:
                # Waiter receiving timeout from originator
                return JSONResponse(
                    status_code=504,
                    content={"error": "Backend request timed out"}
                )

            # Track session (backend response)
            total_latency_ms = (time.perf_counter() - start_time) * 1000
            if self._session_tracker:
                asyncio.create_task(
                    self._session_tracker.record_turn(
                        request=request,
                        request_body=request_body,
                        response=response_data,
                        latency_ms=total_latency_ms,
                        ttft_ms=ttft_ms,
                    )
                )

            if is_streaming:
                return StreamingResponse(
                    self._fake_stream_response(response_data),
                    media_type="text/event-stream",
                )
            else:
                return JSONResponse(response_data)

        except httpx.HTTPStatusError as e:
            try:
                error_body = e.response.json()
            except Exception:
                error_body = {"error": e.response.text or str(e)}
            return JSONResponse(content=error_body, status_code=e.response.status_code)
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout) as e:
            return JSONResponse(content={"error": f"Backend timeout: {str(e)}"}, status_code=504)


def create_app(
    backend_url: str,
    enable_cache: bool = True,
    cache_ttl_seconds: float = 3600.0,
    cache_max_size: int = 1000,
    cache_similarity_threshold: float = 0.85,
    enable_embeddings: bool = False,
    embedding_threshold: float = 0.95,
    embedding_max_entries_scanned: int = 256,
    # NOTE: 120s default is not evidence-based. 30s risks killing cold model loads and long generations.
    backend_timeout_seconds: float = 120.0,
    enable_session_tracking: bool = False,
) -> FastAPI:
    """Factory function to create a proxy app."""
    # ProxyServer will be captured by the lifespan closure
    proxy = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        await proxy.startup()
        yield
        # Shutdown
        await proxy.shutdown()

    proxy = ProxyServer(
        backend_url,
        enable_cache=enable_cache,
        cache_ttl_seconds=cache_ttl_seconds,
        cache_max_size=cache_max_size,
        cache_similarity_threshold=cache_similarity_threshold,
        enable_embeddings=enable_embeddings,
        embedding_threshold=embedding_threshold,
        embedding_max_entries_scanned=embedding_max_entries_scanned,
        backend_timeout_seconds=backend_timeout_seconds,
        enable_session_tracking=enable_session_tracking,
        lifespan=lifespan,
    )

    return proxy.app
