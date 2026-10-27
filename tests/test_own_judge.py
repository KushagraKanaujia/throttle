import asyncio
import json

import httpx
import pytest

from throttle.own import judge as j
from throttle.own import records as r

MSGS = [r.Message(role="system", text="Label it."), r.Message(role="user", text="great product")]


@pytest.mark.parametrize("text, expected", [
    ("Both fine.\nTIE", "TIE"),
    ("reasoning\n\nA\n", "A"),
    ("B", "B"),
    ("The answer is B", None),
    ("A.", "A"),
    ("**A**", "A"),
    ("a", "A"),
    ("A\nmore words", None),
    ("", None),
    (None, None),
])
def test_parse_verdict_is_strict(text, expected):
    assert j.parse_verdict(text) == expected


def _openai_judge(replies, seen=None, status=200):
    """Fake judge: ``replies`` maps (answer_a, answer_b) -> reply text."""

    def handler(request):
        body = json.loads(request.content)
        if seen is not None:
            seen.append((str(request.url), dict(request.headers), body))
        prompt = body["messages"][-1]["content"]
        a = prompt.split("=== Answer A ===\n")[1].split("\n\n=== Answer B")[0]
        b = prompt.split("=== Answer B ===\n")[1].split("\n\nWhich answer")[0]
        if status != 200:
            return httpx.Response(status, json={"error": "nope"})
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": replies(a, b)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        })

    return httpx.MockTransport(handler)


def _both(transport, budget=None, provider="openai"):
    config = j.JudgeConfig(provider=provider, model="judge-m", url="http://judge/v1/chat/completions"
                           if provider == "openai" else "http://judge/v1/messages", api_key="sk-judge")

    async def go():
        async with httpx.AsyncClient(transport=transport) as client:
            return await j.judge_both_orders(MSGS, "INC", "CAND", client=client, config=config,
                                             rubric=j.rubric_for("classification"), budget=budget)

    return asyncio.run(go())


def test_consistent_preference_is_a_win_or_loss():
    prefers_cand = lambda a, b: "ok\nA" if a == "CAND" else "ok\nB"
    prefers_inc = lambda a, b: "A" if a == "INC" else "B"
    assert _both(_openai_judge(prefers_cand)) == j.WIN
    assert _both(_openai_judge(prefers_inc)) == j.LOSS
    assert _both(_openai_judge(lambda a, b: "TIE")) == j.TIE


def test_order_disagreement_is_not_a_tie():
    always_a = lambda a, b: "A"   # pure position bias
    always_b = lambda a, b: "B"
    assert _both(_openai_judge(always_a)) == j.ORDER_DISAGREE
    assert _both(_openai_judge(always_b)) == j.ORDER_DISAGREE
    # TIE in one order and a preference in the other is a disagreement too.
    half = lambda a, b: "TIE" if a == "INC" else "A"
    assert _both(_openai_judge(half)) == j.ORDER_DISAGREE


def test_unparseable_or_failed_judge_is_error():
    assert _both(_openai_judge(lambda a, b: "I think A")) == j.ERROR
    assert _both(_openai_judge(lambda a, b: "A", status=500)) == j.ERROR


def test_openai_request_shape_and_key_header():
    seen = []
    _both(_openai_judge(lambda a, b: "TIE", seen))
    url, headers, body = seen[0]
    assert headers["authorization"] == "Bearer sk-judge"
    assert body["model"] == "judge-m" and body["temperature"] == 0 and "max_tokens" in body
    assert "great product" in body["messages"][-1]["content"]
    assert "correct" in body["messages"][-1]["content"]
    assert len(seen) == 2


def test_anthropic_provider():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((dict(request.headers), body))
        return httpx.Response(200, json={"content": [{"type": "text", "text": "fine\nTIE"}],
                                         "usage": {"input_tokens": 50, "output_tokens": 5}})

    assert _both(httpx.MockTransport(handler), provider="anthropic") == j.TIE
    headers, body = seen[0]
    assert headers["x-api-key"] == "sk-judge" and headers["anthropic-version"] == j.ANTHROPIC_VERSION
    assert body["system"] == j.JUDGE_SYSTEM and body["messages"][0]["role"] == "user"


def test_call_cap_raises_before_calling():
    seen = []
    budget = j.JudgeBudget(max_calls=3)
    transport = _openai_judge(lambda a, b: "TIE", seen)
    assert _both(transport, budget) == j.TIE
    with pytest.raises(j.BudgetExhausted):
        _both(transport, budget)
    assert len(seen) == 3 and budget.calls == 3


def test_usd_cap_uses_price_function():
    budget = j.JudgeBudget(max_calls=100, max_usd=0.025, price=lambda u: 0.01)
    transport = _openai_judge(lambda a, b: "TIE")
    _both(transport, budget)
    assert budget.usd == pytest.approx(0.02) and budget.reserved_usd == 0
    with pytest.raises(j.BudgetExhausted, match="spend cap"):
        _both(transport, budget)     # the next call's worst case would pass the cap
    assert budget.usd == pytest.approx(0.02) and budget.calls == 2


def test_pricing_missing_means_no_usd_cap(monkeypatch):
    import importlib

    real = importlib.import_module

    def fake(name, *a, **k):
        if name == "throttle.own.pricing":
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(j.importlib, "import_module", fake)
    assert j.pricing_for("openai", "gpt-4o") is None


def test_api_key_not_in_repr():
    config = j.JudgeConfig(provider="openai", model="m", url="u", api_key="sk-secret")
    assert "sk-secret" not in repr(config)


def test_rubrics_cover_every_workload_type():
    for kind in r.WORKLOAD_TYPES:
        assert kind in j.RUBRICS
    assert j.rubric_for("unknown") == j.DEFAULT_RUBRIC
    assert "TIE" in j.JUDGE_SYSTEM


def test_pricing_uses_own_pricing_table_when_present(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("throttle.own.pricing")

    class Table:
        def lookup(self, provider, model):
            return {"rate": 2.0} if model == "known" else None

    fake.load_prices = lambda: Table()
    fake.cost_usd = lambda entry, usage: entry["rate"] * (usage.input_tokens + usage.output_tokens) / 1e6
    monkeypatch.setitem(sys.modules, "throttle.own.pricing", fake)
    price = j.pricing_for("openai", "known")
    assert price(r.Usage(input_tokens=400_000, output_tokens=100_000)) == pytest.approx(1.0)
    assert j.pricing_for("openai", "unknown") is None


# ---------------------------------------------------------------------------
# Review fixes: lenient-but-strict parsing, USD cap reservations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("Final answer: **A**", "A"),
    ("**Final answer:** B", "B"),
    ("Verdict: TIE", "TIE"),
    ("verdict: tie.", "TIE"),
    ("thinking\nAnswer: _b_", "B"),
    ("__TIE__", "TIE"),
    ("The answer is B", None),
    ("Final answer: A or B", None),
    ("Final answer: A\nbecause", None),
    ("Verdict:", None),
    ("AB", None),
])
def test_parse_verdict_accepts_markup_and_prefixes(text, expected):
    assert j.parse_verdict(text) == expected


def test_usd_cap_holds_under_concurrency():
    sent = []

    async def handler(request):
        sent.append(1)
        await asyncio.sleep(0.01)
        return httpx.Response(200, json={"choices": [{"message": {"content": "TIE"}}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    budget = j.JudgeBudget(max_calls=100, max_usd=0.02, price=lambda u: 0.01)
    config = j.JudgeConfig(provider="openai", model="m", url="http://judge/v1/chat/completions")

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await asyncio.gather(*(
                j.judge_pair(MSGS, "x", "y", client=client, provider="openai", model="m", rubric="r",
                             budget=budget, config=config)
                for _ in range(6)
            ), return_exceptions=True)

    results = asyncio.run(go())
    assert len(sent) == 2 and budget.usd == pytest.approx(0.02) and budget.reserved_usd == 0
    assert sum(isinstance(x, j.BudgetExhausted) for x in results) == 4


def test_worst_case_reservation_uses_max_tokens_and_prompt_size():
    seen = []

    def price(u):
        seen.append((u.input_tokens, u.output_tokens))
        return (u.input_tokens + u.output_tokens) * 1e-6

    budget = j.JudgeBudget(max_calls=10, max_usd=1.0, price=price)
    assert budget.worst_case(prompt_chars=300, max_tokens=1024) == pytest.approx(1124e-6)
    assert seen[-1] == (100, 1024)
    # Too big to fit: refused before sending, and nothing is counted as spend.
    tight = j.JudgeBudget(max_calls=10, max_usd=0.001, price=price)
    with pytest.raises(j.BudgetExhausted, match="spend cap"):
        tight.reserve(tight.worst_case(prompt_chars=300, max_tokens=1024))
    assert tight.usd == 0 and tight.calls == 0


def test_missing_usage_stops_a_usd_capped_judge():
    seen = []
    no_usage = httpx.MockTransport(lambda req: (seen.append(1), httpx.Response(
        200, json={"choices": [{"message": {"content": "TIE"}}]}))[1])
    budget = j.JudgeBudget(max_calls=100, max_usd=5.0, price=lambda u: 0.01)
    with pytest.raises(j.BudgetExhausted, match="did not report usage"):
        _both(no_usage, budget)
    assert len(seen) == 1 and budget.usd == 0 and budget.stopped == j.NO_USAGE_STOP
    # Without a USD cap, missing usage is only counted.
    loose = j.JudgeBudget(max_calls=100, price=lambda u: 0.01)
    assert _both(no_usage, loose) == j.TIE and loose.unpriced_calls == 2


def test_failed_judge_call_releases_its_reservation():
    budget = j.JudgeBudget(max_calls=100, max_usd=0.05, price=lambda u: 0.01)
    assert _both(_openai_judge(lambda a, b: "A", status=500), budget) == j.ERROR
    assert budget.reserved_usd == 0 and budget.usd == 0 and budget.stopped is None
