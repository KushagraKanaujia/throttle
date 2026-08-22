"""Throttle Proxy: OpenAI-compatible caching proxy for LLM inference backends.

This module provides a lightweight HTTP proxy server that sits in front of real
inference backends (vLLM, Ollama, SGLang, LMDeploy) and caches responses using
semantic similarity matching.

IMPORTANT LIMITATION: The current implementation uses Jaccard token-overlap
similarity (threshold 0.85) which is strict and may miss natural paraphrases.
For example, "optimize PostgreSQL queries" vs "optimize database queries in
PostgreSQL" has Jaccard similarity ~0.64 (below threshold) despite identical
semantic meaning. This is a known limitation of lexical similarity.

A more robust solution would use semantic embeddings (e.g., ONNX-based two-tier
matching) to catch paraphrases that Jaccard misses. The current threshold
balances precision (avoiding false positives) with recall (catching duplicates).
"""

import asyncio
import json
import time
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
    ):
        self.backend_url = backend_url.rstrip("/")
        self.enable_cache = enable_cache
        self.cache: Optional[SimilarityCache] = None

        if self.enable_cache:
            self.cache = SimilarityCache(
                ttl_seconds=cache_ttl_seconds,
                max_size=cache_max_size,
                similarity_threshold=cache_similarity_threshold,
                enable_embeddings=True,
                embedding_threshold=0.95,
            )

        self.app = FastAPI(title="Throttle Proxy", version="0.3.0")
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
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(timeout=120.0)

    async def shutdown(self):
        """Cleanup HTTP client."""
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
                "backend_calls": self._backend_calls,
            } if self.cache else None,
        }

    def _extract_prompt(self, request_body: Dict[str, Any]) -> str:
        """Extract prompt string from request for cache key."""
        messages = request_body.get("messages", [])
        # Simple concatenation of all message content for cache key
        prompt_parts = []
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                prompt_parts.append(msg["content"])
        return " ".join(prompt_parts)

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
        if is_streaming:
            # For streaming, accumulate the complete response
            accumulated_content = []
            response_metadata = {}

            # Increment backend call counter
            # Note: cache misses != backend calls, because in-flight deduplication
            # means multiple concurrent requests can all record cache misses while
            # only one actually reaches the backend (others wait on the same Future)
            self._backend_calls += 1

            async with self._client.stream(
                "POST",
                f"{self.backend_url}/v1/chat/completions",
                json=request_body,
                headers=headers,
            ) as response:
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
            # Increment backend call counter
            # Note: cache misses != backend calls, because in-flight deduplication
            # means multiple concurrent requests can all record cache misses while
            # only one actually reaches the backend (others wait on the same Future)
            self._backend_calls += 1

            response = await self._client.post(
                f"{self.backend_url}/v1/chat/completions",
                json=request_body,
                headers=headers,
            )
            return response.json()

    async def chat_completions(self, request: Request):
        """Handle /v1/chat/completions requests with caching and deduplication."""
        request_body = await request.json()

        # Extract prompt for cache lookup
        prompt = self._extract_prompt(request_body)
        is_streaming = request_body.get("stream", False)

        # Check cache first (fast path)
        if self.enable_cache and self.cache:
            cached_response = self.cache.get(prompt)
            if cached_response is not None:
                # Cache hit
                if is_streaming:
                    # Fake-stream the cached response
                    return StreamingResponse(
                        self._fake_stream_response(cached_response),
                        media_type="text/event-stream",
                    )
                else:
                    # Return cached response directly
                    return JSONResponse(cached_response)

        # Cache miss - check if request is in-flight
        async with self._inflight_lock:
            if prompt in self._inflight:
                # Request is already in-flight, wait for it
                future = self._inflight[prompt]
                is_waiter = True
            else:
                # Create new future for this request
                future = asyncio.Future()
                self._inflight[prompt] = future
                is_waiter = False

        # If we're the primary request (not a waiter), make the backend call
        # IMPORTANT: This happens OUTSIDE the lock to avoid blocking waiters
        if not is_waiter:
            headers = {
                "Content-Type": "application/json",
            }
            if "authorization" in request.headers:
                headers["Authorization"] = request.headers["authorization"]

            try:
                # Make the backend request (accumulates streaming if needed)
                response_data = await self._make_backend_request(
                    request_body, headers, prompt, is_streaming
                )

                # Cache the response
                if self.enable_cache and self.cache:
                    self.cache.put(prompt, response_data)

                # Set the future result for all waiters
                future.set_result(response_data)

            except Exception as e:
                # Propagate error to all waiters
                future.set_exception(e)
                raise
            finally:
                # Clean up in-flight tracking
                async with self._inflight_lock:
                    self._inflight.pop(prompt, None)

        # All requests (both the one that made the call and waiters) reach here
        # Get the result from the future
        response_data = await future

        # Return response in the requested format
        if is_streaming:
            return StreamingResponse(
                self._fake_stream_response(response_data),
                media_type="text/event-stream",
            )
        else:
            return JSONResponse(response_data)


def create_app(
    backend_url: str,
    enable_cache: bool = True,
    cache_ttl_seconds: float = 3600.0,
    cache_max_size: int = 1000,
    cache_similarity_threshold: float = 0.85,
) -> FastAPI:
    """Factory function to create a proxy app."""
    proxy = ProxyServer(
        backend_url,
        enable_cache=enable_cache,
        cache_ttl_seconds=cache_ttl_seconds,
        cache_max_size=cache_max_size,
        cache_similarity_threshold=cache_similarity_threshold,
    )

    # Register startup/shutdown hooks
    @proxy.app.on_event("startup")
    async def startup_event():
        await proxy.startup()

    @proxy.app.on_event("shutdown")
    async def shutdown_event():
        await proxy.shutdown()

    return proxy.app
