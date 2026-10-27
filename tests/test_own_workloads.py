import json
from datetime import datetime, timedelta, timezone

import pytest

from throttle.cli import build_parser, main
from throttle.own import pricing, records as r, workloads as w

T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _ts(hours=0.0):
    return (T0 + timedelta(hours=hours)).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def oai(messages, output="ok", *, model="gpt-4o-mini", tools=None, tool_calls=None, usage=(100, 10, 0),
        hours=0.0, status=200, redaction=r.REDACTION_MASK, **request_extra):
    request = {"model": model, "messages": messages, **request_extra}
    if tools:
        request["tools"] = tools
    message = {"role": "assistant", "content": output}
    if tool_calls:
        message["tool_calls"] = tool_calls
    response = {"choices": [{"message": message, "finish_reason": "stop"}]}
    if usage is not None:
        response["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1],
                             "prompt_tokens_details": {"cached_tokens": usage[2]}}
    ex = r.Exchange(provider=r.PROVIDER_OPENAI, endpoint="/v1/chat/completions", model=model,
                    request=request, response=response, status=status, latency_ms=10,
                    usage=r.usage_from_openai(response), redaction=redaction, ts=_ts(hours))
    if redaction == r.REDACTION_METADATA:
        ex.request = None
        ex.response = None
    return ex


def user(text, system=None):
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + [{"role": "user", "content": text}]


TOOL = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
LONG_DOC = " ".join(["The quarterly numbers improved across every region we track."] * 60)


# ---------------------------------------------------------------------------
# One fixture per type
# ---------------------------------------------------------------------------

def test_agent_from_declared_tools():
    assert w.classify_reason(oai(user("What's the weather?"), tools=[TOOL])) == ("agent", "request offers 1 tool(s)")


def test_agent_from_tool_history():
    msgs = [
        {"role": "user", "content": "find order 7"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                                                                "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "shipped"},
        {"role": "user", "content": "and order 8?"},
    ]
    assert w.classify(oai(msgs)) == "agent"


def test_agent_from_response_tool_call():
    call = [{"id": "c", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
    assert w.classify(oai(user("hi"), output=None, tool_calls=call)) == "agent"


def test_agent_anthropic_tool_result_history():
    ex = r.Exchange(
        provider=r.PROVIDER_ANTHROPIC, endpoint="/v1/messages", model="claude-sonnet-5-5",
        request={"model": "claude-sonnet-5-5", "max_tokens": 100, "messages": [
            {"role": "user", "content": "check stock"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "stock", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "4 left"}]},
        ]},
        response={"content": [{"type": "text", "text": "4 left."}], "stop_reason": "end_turn"},
        status=200, latency_ms=5,
    )
    assert w.classify(ex) == "agent"


def test_extraction_from_response_format():
    ex = oai(user("Invoice 42 from ACME"), output='{"id": 42}', response_format={"type": "json_object"})
    assert w.classify_reason(ex) == ("extraction", "response_format json")


def test_extraction_from_prompt_asking_for_json():
    ex = oai(user("Pull the name and date. Respond only in JSON.", system="You read invoices."), output='{"a":1}')
    assert w.classify(ex) == "extraction"


def test_extraction_from_prompt_asking_for_fields():
    assert w.classify(oai(user("Extract the sender, amount and due date from this email: ..."))) == "extraction"


def test_extraction_single_forced_tool_is_structured_output():
    ex = oai(user("Invoice 42"), tools=[TOOL], tool_choice={"type": "function", "function": {"name": "lookup"}})
    assert w.classify_reason(ex) == ("extraction", "one forced tool used as an output schema")


def test_classification_short_label():
    ex = oai(user("Classify the sentiment: 'great service'. Labels: positive, negative, neutral."), output="positive")
    assert w.classify(ex) == "classification"


def test_classification_yes_no():
    ex = oai(user("Is this email spam? Answer yes or no.\n\nWIN A PRIZE NOW"), output="Yes")
    assert w.classify(ex) == "classification"


def test_code_from_fence():
    ex = oai(user("How do I reverse a list?"), output="Use slicing:\n```python\nxs[::-1]\n```")
    assert w.classify(ex) == "code"


def test_code_from_prompt():
    assert w.classify(oai(user("Write a Python function that parses ISO dates"), output="Sure.")) == "code"


def test_code_from_code_looking_output():
    out = "import os\ndef main():\n    return os.getcwd()"
    assert w.classify(oai(user("go"), output=out)) == "code"


def test_summarization_from_keyword():
    assert w.classify(oai(user("tl;dr of the meeting notes below: ..."), output="We agreed to ship.")) == "summarization"


def test_summarization_from_shape():
    ex = oai(user(LONG_DOC), output="Numbers went up everywhere this quarter, which is good news.")
    kind, reason = w.classify_reason(ex)
    assert kind == "summarization" and "long input" in reason


def test_chat_default():
    ex = oai(user("Tell me a fun fact about otters."), output="Otters hold hands while they sleep so they don't drift.")
    assert w.classify_reason(ex) == ("chat", "no other rule matched")


def test_metadata_only_is_unknown():
    ex = oai(user("secret"), redaction=r.REDACTION_METADATA)
    assert w.classify_reason(ex) == ("unknown", "captured without bodies (metadata only)")


# ---------------------------------------------------------------------------
# Tricky cases: rule order and near misses
# ---------------------------------------------------------------------------

def test_order_agent_beats_json():
    ex = oai(user("Return JSON"), tools=[TOOL], response_format={"type": "json_object"})
    assert w.classify(ex) == "agent"


def test_order_extraction_beats_classification():
    ex = oai(user("Classify this ticket. Respond in JSON."), output='{"label":"bug"}')
    assert w.classify(ex) == "extraction"


def test_classify_prompt_with_long_answer_is_not_classification():
    ex = oai(user("Classify these animals and explain your reasoning."),
             output="Whales are mammals because they breathe air and nurse their young, unlike fish.")
    assert w.classify(ex) != "classification"


def test_short_chat_reply_without_labels_is_chat():
    assert w.classify(oai(user("hi there"), output="Hello! How can I help?")) == "chat"


def test_mentioning_json_casually_is_not_extraction():
    ex = oai(user("What is the difference between JSON and YAML?"),
             output="JSON is stricter and YAML allows comments and is more readable for people.")
    assert w.classify(ex) == "chat"


def test_long_input_long_output_is_not_summarization():
    ex = oai(user(LONG_DOC), output=LONG_DOC)
    assert w.classify(ex) == "chat"


def test_failed_call_still_classified_from_request():
    ex = oai(user("Categorize: 'refund please'. Options: billing, tech."), output=None, usage=None,
             status=500, max_tokens=5)
    ex.response = None
    assert w.classify(ex) == "classification"


def test_rule_table_is_ordered_and_covers_every_type():
    assert [name for name, _ in w.RULES] == ["agent", "extraction", "classification", "code", "summarization", "chat"]
    assert {name for name, _ in w.RULES} == set(r.WORKLOAD_TYPES)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

@pytest.fixture
def prices():
    return pricing.load_prices()


def test_summarize_spend_tokens_and_exclusions(prices):
    exchanges = [
        oai(user("tl;dr: ..."), output="short", model="gpt-4o", usage=(1_000_000, 100_000, 200_000), hours=0),
        oai(user("tl;dr: ..."), output="short", model="gpt-4o-2024-08-06", usage=(1_000_000, 0, 0), hours=24),
        oai(user("tl;dr: ..."), output="short", usage=None, hours=1),            # no usage
        oai(user("tl;dr: ..."), output="short", model="mystery-1", hours=2),     # unknown price
        oai(user("tl;dr: ..."), output="short", status=429, hours=3),            # error
        oai(user("hi"), output="hello there friend", model="gpt-4o-mini", usage=(1000, 1000, 0), hours=48),
        oai(user("x"), model="gpt-4o", usage=(2_000_000, 0, 0), redaction=r.REDACTION_METADATA, hours=10),
    ]
    rows = {row.workload_type: row for row in w.summarize(exchanges, prices)}
    summ = rows["summarization"]
    assert summ.requests == 5 and summ.priced_requests == 2
    assert summ.input_tokens == 2_000_000 + 100 and summ.cached_input_tokens == 200_000
    expected = (0.8 * 2.5 + 0.2 * 1.25 + 0.1 * 10) + 2.5
    assert summ.spend_usd == pytest.approx(expected)
    assert summ.excluded == w.Excluded(no_usage=1, unknown_price=1, errors=1, metadata_only=0)
    assert summ.monthly_projection_usd == pytest.approx(expected / 2 * 30)  # whole capture window = 48h
    assert summ.projection_note.startswith("at captured rate")
    assert summ.first_ts == _ts(0) and summ.last_ts == _ts(24)
    assert summ.top_models[0] == ("gpt-4o-mini", 2)
    assert summ.price_labels == ["LIST PRICE (2026-10-27)"]
    unknown = rows["unknown"]
    assert unknown.excluded.metadata_only == 1 and unknown.spend_usd == pytest.approx(5.0)
    order = [row.workload_type for row in w.summarize(exchanges, prices)]
    assert order == ["summarization", "unknown", "chat"]


def test_projection_refused_under_one_day(prices):
    exchanges = [oai(user("hi"), model="gpt-4o", hours=0), oai(user("hi"), model="gpt-4o", hours=23.5)]
    (row,) = w.summarize(exchanges, prices)
    assert row.spend_usd is not None
    assert row.monthly_projection_usd is None
    assert "23.5 hours" in row.projection_note and "at least 1 day" in row.projection_note


def test_unpriced_type_has_no_spend(prices):
    (row,) = w.summarize([oai(user("hi"), model="mystery")], prices)
    assert row.spend_usd is None and row.monthly_projection_usd is None and row.excluded.unknown_price == 1


# ---------------------------------------------------------------------------
# Embedding sub-clusters
# ---------------------------------------------------------------------------

def test_subclusters_split_by_similarity():
    exchanges = [oai(user(f"weather in city {i}"), output="Sunny and warm, a lovely day out there today.") for i in range(4)]
    exchanges += [oai(user(f"recipe number {i}"), output="Boil the pasta for nine minutes then drain and serve.") for i in range(4)]

    def fake_embed(texts):
        return [[1.0, 0.0] if "weather" in t else [0.0, 1.0] for t in texts]

    result = w.subclusters(exchanges, embed=fake_embed)
    assert [c["requests"] for c in result["chat"]] == [4, 4]


def test_subclusters_refuse_zero_vectors():
    exchanges = [oai(user("a")), oai(user("b"))]
    with pytest.raises(w.EmbeddingsUnavailable, match="zero vectors"):
        w.subclusters(exchanges, embed=lambda texts: [[0.0, 0.0] for _ in texts])


def test_kmeans_is_deterministic():
    vecs = [[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [0.14, 0.99]]
    assert w.kmeans(vecs, 2) == w.kmeans(vecs, 2) == [0, 0, 1, 1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write(tmp_path, exchanges):
    for ex in exchanges:
        r.append_exchange(ex, tmp_path)


def test_cli_json_schema(tmp_path, capsys):
    _write(tmp_path, [
        oai(user("Classify: 'bad'. Labels: good, bad."), output="bad", model="gpt-4o-mini", hours=0),
        oai(user("Write a function to add numbers"), output="```py\ndef add(a,b): return a+b\n```",
            model="gpt-4.1", hours=50),
        oai(user("x"), redaction=r.REDACTION_METADATA, hours=1),
    ])
    assert main(["workloads", "--dir", str(tmp_path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema"] == "throttle.own.workloads/1"
    assert set(data) == {"schema", "dir", "window", "spend_basis", "projection_basis", "prices", "totals", "workloads"}
    assert set(data["window"]) == {"first_ts", "last_ts", "days", "projectable"}
    assert data["window"]["projectable"] is True
    assert set(data["prices"]) == {"labels", "sources", "override"}
    assert data["prices"]["labels"] == ["LIST PRICE (2026-10-27)"]
    assert set(data["totals"]) == {"requests", "priced_requests", "spend_usd", "monthly_projection_usd", "excluded"}
    assert set(data["totals"]["excluded"]) == {"no_usage", "unknown_price", "errors", "metadata_only"}
    assert data["totals"]["requests"] == 3 and data["totals"]["excluded"]["metadata_only"] == 1
    row = data["workloads"][0]
    assert set(row) == {
        "type", "requests", "priced_requests", "input_tokens", "output_tokens", "cached_input_tokens",
        "spend_usd", "first_ts", "last_ts", "monthly_projection_usd", "projection_note", "top_models",
        "excluded", "price_labels",
    }
    assert {x["type"] for x in data["workloads"]} == {"classification", "code", "unknown"}
    spends = [x["spend_usd"] for x in data["workloads"]]
    assert spends == sorted(spends, reverse=True)


def test_cli_table_and_projection_refusal(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    _write(tmp_path, [oai(user("hi"), model="gpt-4o", hours=0), oai(user("hi"), model="gpt-4o", hours=2)])
    assert main(["workloads", "--dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "chat" in out and "LIST PRICE (2026-10-27)" in out
    assert "No monthly projection" in out and "under 1 day" in out


def test_cli_override_prices_are_assumed(tmp_path, capsys):
    _write(tmp_path, [oai(user("hi"), model="my-ft")])
    prices = tmp_path / "p.json"
    prices.write_text(json.dumps([{"provider": "openai", "model": "my-ft", "input_per_m": 1, "output_per_m": 1}]))
    assert main(["workloads", "--dir", str(tmp_path), "--prices", str(prices), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["workloads"][0]["price_labels"] == ["ASSUMED (operator price, undated)"]
    assert data["prices"]["override"] == str(prices)


def test_cli_since_filters(tmp_path, capsys):
    now = datetime.now(timezone.utc)
    old = oai(user("hi"))
    old.ts = (now - timedelta(days=10)).isoformat().replace("+00:00", "Z")
    new = oai(user("hi"))
    new.ts = (now - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    _write(tmp_path, [old, new])
    assert main(["workloads", "--dir", str(tmp_path), "--since", "7d", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["totals"]["requests"] == 1


def test_cli_errors(tmp_path, capsys):
    assert main(["workloads", "--dir", str(tmp_path)]) == 1
    assert "no captured calls" in capsys.readouterr().err
    _write(tmp_path, [oai(user("hi"))])
    assert main(["workloads", "--dir", str(tmp_path), "--since", "yesterday"]) == 2
    assert "--since" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    assert main(["workloads", "--dir", str(tmp_path), "--prices", str(bad)]) == 2


def test_cli_embed_refuses_without_embeddings(tmp_path, capsys, monkeypatch):
    _write(tmp_path, [oai(user("hi"))])

    def unavailable(texts):
        raise w.EmbeddingsUnavailable("Embedding dependencies not installed")

    monkeypatch.setattr(w, "_embed", unavailable)
    assert main(["workloads", "--dir", str(tmp_path), "--embed"]) == 1
    assert "cannot cluster" in capsys.readouterr().err


def test_cli_embed_adds_clusters(tmp_path, capsys, monkeypatch):
    _write(tmp_path, [oai(user("hi")), oai(user("hey"))])
    monkeypatch.setattr(w, "_embed", lambda texts: [[1.0, 0.0] for _ in texts])
    assert main(["workloads", "--dir", str(tmp_path), "--embed", "--json"]) == 0
    row = json.loads(capsys.readouterr().out)["workloads"][0]
    assert row["clusters"] == [{"requests": 2, "example": "hi"}]


def test_parse_since():
    now = datetime(2026, 10, 27, tzinfo=timezone.utc)
    assert w.parse_since("7d", now=now) == now - timedelta(days=7)
    assert w.parse_since("24h", now=now) == now - timedelta(hours=24)
    assert w.parse_since("2026-10-01", now=now) == datetime(2026, 10, 1, tzinfo=timezone.utc)


def test_workloads_is_registered():
    parser = build_parser()
    args = parser.parse_args(["workloads", "--json"])
    assert args.command == "workloads"
