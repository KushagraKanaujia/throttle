"""Tests for per-token candidate pricing (``throttle.own.candidate_cost``). All offline.

Every token count and per-token price in this file is a SAMPLE made up for the
test, except list prices, which are read from the bundled prices.json.
"""

from __future__ import annotations

import json
import math

import pytest

from own_eval_helpers import exchange
from throttle.own import candidate_cost as cc
from throttle.own import evaluate as ev
from throttle.own import pricing
from throttle.own import records as r

PRICES = pricing.load_prices()


def _usages(pairs):
    return [r.Usage(input_tokens=i, output_tokens=o) for i, o in pairs]


# ---------------------------------------------------------------------------
# Price source
# ---------------------------------------------------------------------------


def test_explicit_flags_are_assumed():
    price = cc.resolve_price(PRICES, price_in=0.2, price_out=0.8, model=None)   # sample prices
    assert price.label == cc.ASSUMED and price.label_detail.startswith("ASSUMED")
    assert price.price.input_per_m == 0.2 and price.price.output_per_m == 0.8
    data = price.to_json()
    assert data["label"] == "ASSUMED" and data["as_of"] is None and "--candidate-price-in" in data["source"]


def test_flags_beat_the_price_table():
    # gpt-4.1-mini is in prices.json, but explicit flags win and stay ASSUMED.
    price = cc.resolve_price(PRICES, price_in=0.2, price_out=0.8, model="gpt-4.1-mini")
    assert price.label == cc.ASSUMED and price.model == "gpt-4.1-mini"
    assert price.price.input_per_m == 0.2


def test_model_in_price_table_is_list_price():
    price = cc.resolve_price(PRICES, price_in=None, price_out=None, model="gpt-4.1-mini")
    table = PRICES.lookup("openai", "gpt-4.1-mini")
    assert price.label == cc.LIST_PRICE
    assert price.label_detail == f"LIST PRICE ({table.as_of})"
    assert (price.price.input_per_m, price.price.output_per_m) == (table.input_per_m, table.output_per_m)
    assert price.to_json()["as_of"] == table.as_of
    # Anthropic models are found too, and a dated snapshot matches its prefix.
    assert cc.lookup_price(PRICES, "claude-haiku-4-5-20251001").label == cc.LIST_PRICE


def test_unknown_model_and_no_flags_is_none():
    assert cc.resolve_price(PRICES, price_in=None, price_out=None, model="qwen2.5-72b-hosted") is None
    assert cc.resolve_price(PRICES, price_in=None, price_out=None, model=None) is None


def test_operator_price_file_entry_is_assumed(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps([{"provider": "openai", "model": "my-hosted-open", "input_per_m": 0.1,
                                 "output_per_m": 0.3}]))   # sample prices
    price = cc.lookup_price(pricing.load_prices(path), "my-hosted-open")
    assert price.label == cc.ASSUMED


@pytest.mark.parametrize("pin,pout", [(None, 0.8), (0.2, None)])
def test_one_price_flag_alone_is_an_error(pin, pout):
    with pytest.raises(ValueError, match="both --candidate-price-in and --candidate-price-out"):
        cc.resolve_price(PRICES, price_in=pin, price_out=pout, model=None)


@pytest.mark.parametrize("pin,pout", [(-1, 0.8), (0.2, math.nan), (0.2, math.inf), (0, 0)])
def test_bad_prices_are_rejected(pin, pout):
    with pytest.raises(ValueError):
        cc.assumed_price(pin, pout)


def test_model_matches_shadow_model():
    listed = cc.lookup_price(PRICES, "gpt-4.1-mini")
    assert cc.model_matches(listed, "gpt-4.1-mini-2025-04-14") is True
    assert cc.model_matches(listed, "gpt-4.1") is False
    assert cc.model_matches(listed, None) is None
    assumed = cc.assumed_price(0.2, 0.8, model="Qwen/Qwen2.5-72B")
    assert cc.model_matches(assumed, "qwen/qwen2.5-72b") is True
    assert cc.model_matches(cc.assumed_price(0.2, 0.8), "anything") is None


# ---------------------------------------------------------------------------
# Cost math: the candidate's own tokens, never the API's
# ---------------------------------------------------------------------------


def test_per_request_cost_uses_candidate_tokens_with_ci():
    price = cc.assumed_price(1.0, 2.0)                      # sample: $1/M in, $2/M out
    usages = _usages([(1000, 100), (1000, 300)] * 10)       # sample token counts
    stats, why = cc.per_request_cost(usages, price, kind="chat")
    assert why is None and stats["n"] == 20 and stats["excluded"] == 0
    # mean = 1000 x $1/M + 200 x $2/M = $0.0014
    assert stats["mean"] == pytest.approx(0.0014)
    assert stats["low"] <= stats["mean"] <= stats["high"]
    assert stats["low"] < stats["high"]
    again, _ = cc.per_request_cost(usages, price, kind="chat")
    assert again == stats                                   # deterministic


def test_monthly_cost_point_and_ci():
    price = cc.assumed_price(1.0, 2.0)
    cost = cc.monthly_cost(_usages([(1000, 100), (1000, 300)] * 10), price, 900.0, kind="chat")
    assert cost.owned["point"] == pytest.approx(900 * 0.0014)
    assert cost.owned["low"] == pytest.approx(900 * cost.per_request["low"])
    assert cost.owned["high"] == pytest.approx(900 * cost.per_request["high"])
    assert "ASSUMED" in cost.note and "20 shadow answers" in cost.note


def test_cheaper_list_price_can_cost_more_with_more_output_tokens():
    # The API model: gpt-4.1. The candidate: o4-mini, cheaper per token on the list,
    # but it writes (reasons) far more output tokens on the same requests.
    api = PRICES.lookup("openai", "gpt-4.1")
    cand = cc.lookup_price(PRICES, "o4-mini")
    assert cand.price.input_per_m < api.input_per_m and cand.price.output_per_m < api.output_per_m
    api_per_request = pricing.cost_usd(api, r.Usage(input_tokens=1000, output_tokens=200))   # sample counts
    usages = _usages([(1000, 800), (1000, 1000)] * 15)                                        # sample counts
    stats, _ = cc.per_request_cost(usages, cand)
    assert stats["low"] > api_per_request          # more expensive even at the low end
    # Pricing the API's own token counts at the candidate's list price would wrongly look cheaper.
    naive = pricing.cost_usd(cand.price, r.Usage(input_tokens=1000, output_tokens=200))
    assert naive < api_per_request


def test_cached_input_uses_cached_price_only_when_listed():
    usage = [r.Usage(input_tokens=1000, output_tokens=0, cached_input_tokens=800)] * 2   # sample counts
    listed = cc.lookup_price(PRICES, "gpt-4.1-mini")
    stats, _ = cc.per_request_cost(usage, listed)
    p = listed.price
    assert stats["mean"] == pytest.approx((200 * p.input_per_m + 800 * p.cached_input_per_m) / 1e6)
    assumed, _ = cc.per_request_cost(usage, cc.assumed_price(1.0, 2.0))
    assert assumed["mean"] == pytest.approx(1000 * 1.0 / 1e6)   # no cached price given: input price


def test_incomplete_usage_is_excluded_not_estimated():
    usages = _usages([(1000, 100)] * 3) + [r.Usage(input_tokens=1000), r.Usage()]
    stats, _ = cc.per_request_cost(usages, cc.assumed_price(1.0, 2.0))
    assert stats["n"] == 3 and stats["excluded"] == 2


def test_too_few_priced_answers_has_no_number():
    cost = cc.monthly_cost(_usages([(1000, 100)]), cc.assumed_price(1.0, 2.0), 900.0)
    assert cost.owned is None and cost.note.startswith("not measured: only 1 shadow answer")
    assert cc.monthly_cost([], cc.assumed_price(1.0, 2.0), 900.0).note.startswith("not measured: no shadow answers")
    assert cc.monthly_cost(None, cc.assumed_price(1.0, 2.0), 900.0).owned is None


def test_short_window_is_not_projected():
    cost = cc.monthly_cost(_usages([(1000, 100)] * 5), cc.assumed_price(1.0, 2.0), None)
    assert cost.owned is None and cost.note.startswith("not projected")


def test_unpriced_long_context_leaves_the_type_unpriced():
    price = cc.lookup_price(PRICES, "gpt-5.4")
    above = PRICES.lookup("openai", "gpt-5.4").long_context.above_input_tokens
    usages = _usages([(1000, 100)] * 5 + [(above + 1, 100)])
    cost = cc.monthly_cost(usages, price, 900.0)
    assert cost.owned is None and "long-context" in cost.note


# ---------------------------------------------------------------------------
# Loading measured usage from shadow records
# ---------------------------------------------------------------------------


def _shadow(ex: r.Exchange, out_tokens: int, *, ok=True, candidate="mini") -> r.ShadowResult:
    return r.ShadowResult(
        exchange_id=ex.id, candidate=candidate, candidate_model="gpt-4.1-mini",
        output_text="answer" if ok else None, tool_calls=[],
        usage=r.Usage(input_tokens=900, output_tokens=out_tokens), latency_ms=10.0,
        status=200 if ok else 500, error=None if ok else "http_500: boom",
    )


def test_load_samples_reads_each_shadow_answer(tmp_path, monkeypatch):
    from throttle.own import workloads
    monkeypatch.setattr(workloads, "classify", lambda ex, norm=None: ex.workload_type or "chat")
    exchanges = [exchange(f"{kind} {i}", kind=kind) for kind in ("chat", "extraction") for i in range(4)]
    for i, ex in enumerate(exchanges):
        r.append_exchange(ex, tmp_path)
        r.append_shadow(_shadow(ex, 50 + i, ok=i != 0), tmp_path)
    samples = cc.load_samples(tmp_path, "mini", None)
    assert set(samples) == {"chat", "extraction"}
    assert len(samples["chat"]) == 3            # the failed answer has no usage to price
    assert sorted(u.output_tokens for u in samples["extraction"]) == [54, 55, 56, 57]

    # With the eval's saved eval sets, exactly those requests are used.
    sets = ev.build_eval_sets(tmp_path, "mini")
    types = {}
    for kind, pairs in sets.items():
        _, path = ev.save_evalset(tmp_path, kind, "mini", pairs[:2])
        types[kind] = {"evalset_path": str(path)}
    samples = cc.load_samples(tmp_path, "mini", {"types": types})
    assert len(samples["extraction"]) == 2

    # An eval set for another candidate is ignored: every shadow answer is used.
    _, other = ev.save_evalset(tmp_path, "chat", "other", sets["chat"][:1])
    samples = cc.load_samples(tmp_path, "mini", {"types": {"chat": {"evalset_path": str(other)}}})
    assert len(samples["extraction"]) == 4
