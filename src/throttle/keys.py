"""
throttle/keys.py
================
Shared cache key extraction functions.

Single implementation used by both the proxy (request path) and
forecast (offline simulation). Any change here affects both — that
is the point.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def extract_scope_key(
    request_body: Dict[str, Any],
    *,
    model_backends: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build a scope key from request parameters that affect which backend
    answers and what response is valid to cache.

    Excludes 'messages' (the prompt — handled separately) and 'stream'
    (a transport detail, not a semantic one).

    When model_backends routing is present, folds the resolved backend
    URL into the scope key. Two backends serving the same model name
    produce different scope keys and never collide in the cache.
    """
    excluded = {"messages", "stream"}
    scope_params = {
        k: v for k, v in request_body.items() if k not in excluded
    }

    if model_backends:
        model = request_body.get("model", "")
        backend_url = model_backends.get(model)
        if backend_url:
            scope_params["_backend_url"] = backend_url

    return json.dumps(scope_params, sort_keys=True)


def extract_prompt(request_body: Dict[str, Any]) -> str:
    """
    Extract a single string representation of the prompt from a
    chat-completions request body.

    Concatenates all message contents with role prefixes so that
    role changes (system vs user) produce different cache keys.
    """
    messages = request_body.get("messages", [])
    parts = []
    for msg in messages:
        if isinstance(msg, dict) and "content" in msg:
            role = msg.get("role", "user")
            parts.append(f"{role}: {msg['content']}")
    return "\n".join(parts)
