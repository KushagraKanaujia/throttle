import json
import os

import pytest

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


# ---------------------------------------------------------------------------
# Review fixes: translation keeps tool calls, tool_choice and stop
# ---------------------------------------------------------------------------

ANTHROPIC_AGENT = {
    "model": "claude-x", "max_tokens": 100, "system": [{"type": "text", "text": "sys"}],
    "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
    "tool_choice": {"type": "tool", "name": "get_weather"},
    "stop_sequences": ["END"],
    "messages": [
        {"role": "user", "content": "weather in SF?"},
        {"role": "assistant", "content": [{"type": "text", "text": "Let me check."},
                                          {"type": "tool_use", "id": "tu1", "name": "get_weather", "input": {"city": "SF"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "72F"}]},
    ],
}


def test_assistant_text_and_tool_calls_both_survive_translation():
    body = r.to_openai_chat_request(r.normalize_anthropic(ANTHROPIC_AGENT, None), model="m")
    assistant = body["messages"][2]
    assert assistant["role"] == "assistant" and assistant["content"].startswith("Let me check.\n")
    summary = json.loads(assistant["content"].split("\n", 1)[1])
    assert summary == [{"tool": "get_weather", "arguments": '{"city": "SF"}'}]
    # A call-only turn is just the summary.
    only = dict(ANTHROPIC_AGENT, messages=[ANTHROPIC_AGENT["messages"][0], {"role": "assistant", "content": [
        {"type": "tool_use", "id": "t", "name": "get_weather", "input": {}}]}])
    body = r.to_openai_chat_request(r.normalize_anthropic(only, None), model="m")
    assert json.loads(body["messages"][2]["content"])[0]["tool"] == "get_weather"


def test_anthropic_tool_choice_and_stop_sequences_are_forwarded():
    norm = r.normalize_anthropic(ANTHROPIC_AGENT, None)
    assert norm.tool_choice == {"type": "function", "function": {"name": "get_weather"}}
    assert norm.stop == ["END"]
    body = r.to_openai_chat_request(norm, model="m")
    assert body["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
    assert body["stop"] == ["END"]
    for raw, expected in [({"type": "any"}, "required"), ({"type": "auto"}, "auto"), ({"type": "none"}, "none"),
                          ({"type": "weird"}, None), ("auto", None)]:
        assert r.normalize_anthropic(dict(ANTHROPIC_AGENT, tool_choice=raw), None).tool_choice == expected


def test_openai_tool_choice_and_stop_are_forwarded():
    base = {"messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]}
    for raw, expected in [
        ("required", "required"), ("auto", "auto"), ("none", "none"), ("bogus", None),
        ({"type": "function", "function": {"name": "f"}}, {"type": "function", "function": {"name": "f"}}),
        ({"type": "function", "name": "f"}, {"type": "function", "function": {"name": "f"}}),
    ]:
        norm = r.normalize_openai(dict(base, tool_choice=raw), None)
        assert norm.tool_choice == expected
        body = r.to_openai_chat_request(norm, model="m")
        assert body.get("tool_choice") == expected
    assert r.normalize_openai(dict(base, stop="\n\n"), None).stop == ["\n\n"]
    body = r.to_openai_chat_request(r.normalize_openai(dict(base, stop=["a", "b"]), None), model="m")
    assert body["stop"] == ["a", "b"]
    # No tools: tool_choice is not sent (servers reject it), no stop means no key.
    plain = r.to_openai_chat_request(r.normalize_openai({"messages": base["messages"], "tool_choice": "auto"}, None),
                                     model="m")
    assert "tool_choice" not in plain and "stop" not in plain


def test_empty_shadow_answer_is_not_ok_but_tool_call_only_is():
    def result(text, calls):
        return r.ShadowResult(exchange_id="e", candidate="c", candidate_model="m", output_text=text,
                              tool_calls=calls, usage=r.Usage(), latency_ms=1.0, status=200)

    assert not result("", []).ok
    assert result("", [{"name": "f", "arguments": "{}"}]).ok
    assert result("hi", []).ok


def test_incumbent_empty_answer_still_normalizes():
    ex = _openai_exchange(response={"choices": [{"message": {"role": "assistant", "content": None},
                                                 "finish_reason": "stop"}]})
    norm = r.normalize(ex)
    assert ex.ok and norm.output_text == "" and norm.output_tool_calls == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_new_capture_ledger_and_lesson_files_are_private(tmp_path):
    import stat

    from throttle.own import jobs as jobs_mod
    from throttle.own import route as route_mod

    root = tmp_path / "data"
    exchange = r.Exchange(provider="openai", endpoint="/v1/chat/completions", model="m",
                          request={"messages": [{"role": "user", "content": "hi"}]}, response=None,
                          status=200, latency_ms=1.0, redaction=r.REDACTION_OFF)
    capture_file = r.append_exchange(exchange, root)
    ledger_file = route_mod.ledger_path(root, exchange.ts[:7])
    r.append_jsonl(ledger_file, {"id": exchange.id})
    lesson_file = jobs_mod.add_lesson(root, "app.py::f", source="test", reason=None, exchange_id=exchange.id,
                                      norm=None, preferred="a", rejected="b")
    registry_file = jobs_mod.save_registry(root, {})

    def mode(p):
        return stat.S_IMODE(p.stat().st_mode)

    for path in (capture_file, ledger_file, lesson_file, registry_file):
        assert mode(path) == 0o600, path
    for folder in {root, capture_file.parent, ledger_file.parent, lesson_file.parent, registry_file.parent}:
        assert mode(folder) == 0o700, folder
    # appending again keeps working and keeps the mode
    r.append_exchange(exchange, root)
    assert mode(capture_file) == 0o600 and len(list(r.iter_exchanges(root))) == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_existing_files_and_folders_keep_working(tmp_path):
    import stat

    folder = tmp_path / "capture"
    folder.mkdir(mode=0o755)
    os.chmod(folder, 0o755)
    old = folder / "2026-01.jsonl"
    old.write_text('{"a":1}\n')
    os.chmod(old, 0o644)
    r.append_jsonl(old, {"b": 2})
    assert [row for row in r.read_jsonl(old)] == [{"a": 1}, {"b": 2}]
    assert stat.S_IMODE(old.stat().st_mode) == 0o644      # not changed behind the owner's back
    assert stat.S_IMODE(folder.stat().st_mode) == 0o755
