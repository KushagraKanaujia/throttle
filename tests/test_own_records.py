import json

from throttle.own import records as r


def _openai_exchange(**overrides):
    request = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Extract fields."},
            {"role": "user", "content": [{"type": "text", "text": "Invoice 42 from ACME"}]},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}},
        "max_tokens": 64,
        "temperature": 0,
    }
    response = {
        "choices": [{"message": {"role": "assistant", "content": '{"id": 42}'}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 4}},
    }
    data = dict(
        provider=r.PROVIDER_OPENAI, endpoint="/v1/chat/completions", model="gpt-4o-mini",
        request=request, response=response, status=200, latency_ms=12.5,
        usage=r.usage_from_openai(response),
    )
    data.update(overrides)
    return r.Exchange(**data)


def test_exchange_roundtrip_and_storage(tmp_path):
    ex = _openai_exchange()
    path = r.append_exchange(ex, tmp_path)
    assert path.name == ex.ts[:7] + ".jsonl"
    with path.open("a") as handle:
        handle.write('{"truncated": ')  # crash mid-write must not break readers
    loaded = list(r.iter_exchanges(tmp_path))
    assert len(loaded) == 1
    assert loaded[0].id == ex.id
    assert loaded[0].usage.input_tokens == 20 and loaded[0].usage.cached_input_tokens == 4
    assert loaded[0].ok and loaded[0].has_bodies


def test_metadata_capture_has_no_bodies():
    ex = _openai_exchange(request=None, response=None, redaction=r.REDACTION_METADATA)
    assert not ex.has_bodies
    assert list(r.iter_with_bodies([ex])) == []


def test_normalize_openai():
    norm = r.normalize(_openai_exchange())
    assert norm.system_text == "Extract fields."
    assert norm.last_user_text == "Invoice 42 from ACME"
    assert norm.response_format == "json_schema" and norm.json_schema == {"type": "object"}
    assert norm.output_text == '{"id": 42}' and norm.finish_reason == "stop"
    assert norm.max_tokens == 64 and norm.turns == 1


def test_normalize_anthropic_with_tools_and_cache_usage():
    request = {
        "model": "claude-sonnet",
        "system": [{"type": "text", "text": "You are an agent."}],
        "max_tokens": 100,
        "tools": [{"name": "lookup", "description": "d", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": "find order 7"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "lookup", "input": {"id": 7}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "shipped"}]},
        ],
    }
    response = {
        "content": [{"type": "text", "text": "It shipped."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "cache_read_input_tokens": 90, "cache_creation_input_tokens": 0, "output_tokens": 4},
    }
    norm = r.normalize_anthropic(request, response)
    assert [m.role for m in norm.messages] == ["system", "user", "assistant", "tool"]
    assert norm.messages[2].tool_calls[0]["name"] == "lookup"
    assert json.loads(norm.messages[2].tool_calls[0]["arguments"]) == {"id": 7}
    assert norm.tools[0]["function"]["name"] == "lookup"
    assert norm.output_text == "It shipped."
    usage = r.usage_from_anthropic(response)
    assert usage.input_tokens == 100 and usage.cached_input_tokens == 90 and usage.output_tokens == 4


def test_translate_to_openai_chat_for_candidate():
    norm = r.normalize(_openai_exchange())
    body = r.to_openai_chat_request(norm, model="qwen2.5-7b")
    assert body["model"] == "qwen2.5-7b" and body["stream"] is False
    assert body["response_format"]["type"] == "json_schema"
    assert body["max_tokens"] == 64
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_shadow_results_are_kept_per_candidate(tmp_path):
    result = r.ShadowResult(
        exchange_id="abc", candidate="qwen 72b @ vllm", candidate_model="qwen", output_text="hi",
        tool_calls=[], usage=r.Usage(3, 1), latency_ms=5.0, status=200,
    )
    path = r.append_shadow(result, tmp_path)
    assert path.name == "qwen-72b-vllm.jsonl"
    loaded = list(r.iter_shadow("qwen 72b @ vllm", tmp_path))
    assert loaded[0].ok and loaded[0].usage.output_tokens == 1


def test_bad_usage_values_are_not_trusted():
    assert r.usage_from_openai({"usage": {"prompt_tokens": -1, "completion_tokens": True}}).complete is False
