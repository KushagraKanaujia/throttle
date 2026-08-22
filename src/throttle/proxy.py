"""Throttle Proxy: OpenAI-compatible caching proxy for LLM inference backends.

Uses the SimilarityCache Promise interface for unified semantic caching
and in-flight request deduplication. Concurrent paraphrase requests share
a single backend call via asyncio Futures managed by the cache layer.
"""

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from .cache import EntryStatus, SimilarityCache


class ProxyServer:
    """OpenAI-compatible proxy with semantic caching and Promise-based dedup."""

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
        self._backend_calls = 0

        if self.enable_cache:
            self.cache = SimilarityCache(
                ttl_seconds=cache_ttl_seconds,
                max_size=cache_max_size,
                similarity_threshold=cache_similarity_threshold,
                enable_embeddings=True,
                embedding_threshold=0.95,
            )

        self.app = FastAPI(title="Throttle Proxy", version="0.4.0")
        self.app.post("/v1/chat/completions")(self.chat_completions)
        self.app.get("/health")(self.health)
        self.app.get("/stats")(self.stats)

        self._client: Optional[httpx.AsyncClient] = None

    async def startup(self):
        self._client = httpx.AsyncClient(timeout=120.0)

    async def shutdown(self):
        if self._client:
            await self._client.aclose()

    async def health(self):
        return {
            "status": "ok",
            "cache_enabled": self.enable_cache,
            "cache_stats": self.cache.stats() if self.cache else None,
        }

    async def stats(self):
        return {
            "backend_calls": self._backend_calls,
            "cache": self.cache.stats() if self.cache else None,
        }

    def _extract_prompt(self, request_body: Dict[str, Any]) -> str:
        messages = request_body.get("messages", [])
        prompt_parts = []
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                prompt_parts.append(msg["content"])
        return " ".join(prompt_parts)

    async def _fake_stream_response(
        self, cached_response: Dict[str, Any]
    ) -> AsyncIterator[str]:
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
        await asyncio.sleep(0.001)

        content = cached_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        chunk_size = 10
        for i in range(0, len(content), chunk_size):
            content_chunk = {
                "id": cached_response.get("id", "cached-response"),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": cached_response.get("model", "unknown"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content[i : i + chunk_size]},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(content_chunk)}\n\n"
            await asyncio.sleep(0.001)

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

    async def _make_backend_request(
        self,
        request_body: Dict[str, Any],
        headers: Dict[str, str],
        is_streaming: bool,
    ) -> Dict[str, Any]:
        if is_streaming:
            accumulated_content = []
            response_metadata: Dict[str, Any] = {}

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
                            "content": "".join(accumulated_content),
                        },
                        "finish_reason": response_metadata.get("finish_reason", "stop"),
                    }
                ],
            }
        else:
            response = await self._client.post(
                f"{self.backend_url}/v1/chat/completions",
                json=request_body,
                headers=headers,
            )
            return response.json()

    async def chat_completions(self, request: Request):
        """Handle /v1/chat/completions with Promise Cache dedup."""
        request_body = await request.json()
        prompt = self._extract_prompt(request_body)
        is_streaming = request_body.get("stream", False)

        if self.enable_cache and self.cache:
            loop = asyncio.get_running_loop()
            status, result_or_future, entry = self.cache.get_or_create_promise(prompt, loop)

            if status == EntryStatus.READY:
                if is_streaming:
                    return StreamingResponse(
                        self._fake_stream_response(result_or_future),
                        media_type="text/event-stream",
                    )
                return JSONResponse(result_or_future)

            if result_or_future.done():
                try:
                    payload = result_or_future.result()
                    if is_streaming:
                        return StreamingResponse(
                            self._fake_stream_response(payload),
                            media_type="text/event-stream",
                        )
                    return JSONResponse(payload)
                except Exception:
                    pass

            is_owner = not result_or_future.done()

            if not is_owner:
                try:
                    payload = await asyncio.wait_for(result_or_future, timeout=30.0)
                    if is_streaming:
                        return StreamingResponse(
                            self._fake_stream_response(payload),
                            media_type="text/event-stream",
                        )
                    return JSONResponse(payload)
                except asyncio.TimeoutError:
                    return JSONResponse(
                        {"error": "in-flight request timed out"}, status_code=504
                    )
                except Exception as e:
                    return JSONResponse(
                        {"error": f"in-flight request failed: {e}"}, status_code=502
                    )

            self._backend_calls += 1
            headers = {"Content-Type": "application/json"}
            if "authorization" in request.headers:
                headers["Authorization"] = request.headers["authorization"]

            try:
                response_data = await self._make_backend_request(
                    request_body, headers, is_streaming
                )
                self.cache.resolve_promise(entry, response_data)

                if is_streaming:
                    return StreamingResponse(
                        self._fake_stream_response(response_data),
                        media_type="text/event-stream",
                    )
                return JSONResponse(response_data)

            except Exception as e:
                self.cache.reject_promise(entry, e)
                return JSONResponse(
                    {"error": f"backend request failed: {e}"}, status_code=502
                )

        self._backend_calls += 1
        headers = {"Content-Type": "application/json"}
        if "authorization" in request.headers:
            headers["Authorization"] = request.headers["authorization"]

        response_data = await self._make_backend_request(
            request_body, headers, is_streaming
        )
        if is_streaming:
            return StreamingResponse(
                self._fake_stream_response(response_data),
                media_type="text/event-stream",
            )
        return JSONResponse(response_data)


def create_app(
    backend_url: str,
    enable_cache: bool = True,
    cache_ttl_seconds: float = 3600.0,
    cache_max_size: int = 1000,
    cache_similarity_threshold: float = 0.85,
) -> FastAPI:
    proxy = ProxyServer(
        backend_url,
        enable_cache=enable_cache,
        cache_ttl_seconds=cache_ttl_seconds,
        cache_max_size=cache_max_size,
        cache_similarity_threshold=cache_similarity_threshold,
    )

    @proxy.app.on_event("startup")
    async def startup_event():
        await proxy.startup()

    @proxy.app.on_event("shutdown")
    async def shutdown_event():
        await proxy.shutdown()

    return proxy.app
