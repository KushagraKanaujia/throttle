import json
import re
import tomllib
from fnmatch import fnmatch
from pathlib import Path

import pytest

from throttle.own import pricing as p
from throttle.own.records import Usage

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def table():
    return p.load_prices()


def test_bundled_table_entries_are_complete_and_sourced(table):
    assert len(table) >= 40
    for price in table:
        assert price.origin == p.ORIGIN_BUNDLED
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", price.as_of)
        assert price.source.startswith("https://")
        assert price.input_per_m > 0 and price.output_per_m > 0
        if price.cached_input_per_m is not None:
            assert price.cached_input_per_m <= price.input_per_m
        assert price.label == f"LIST PRICE ({price.as_of})"
    providers = {price.provider for price in table}
    assert providers == {"openai", "anthropic"}


@pytest.mark.parametrize("model", [
    "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-4o", "gpt-4o-mini", "o1", "o3", "o3-mini", "o4-mini",
])
def test_core_openai_models_are_covered(table, model):
    assert table.lookup("openai", model).model == model


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-haiku-4-5"])
def test_core_anthropic_models_are_covered(table, model):
    assert table.lookup("anthropic", model).model == model


def test_prices_json_ships_in_the_wheel():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    globs = config["tool"]["setuptools"]["package-data"]["throttle"]
    assert any(fnmatch("own/prices.json", g) for g in globs)
    assert (ROOT / "src/throttle/own/prices.json").is_file()


@pytest.mark.parametrize("provider,model,expected", [
    ("openai", "gpt-4o", "gpt-4o"),
    ("openai", "gpt-4o-2024-08-06", "gpt-4o"),
    ("openai", "GPT-4o-2024-08-06", "gpt-4o"),
    ("openai", "gpt-4o-mini", "gpt-4o-mini"),
    ("openai", "gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
    ("openai", "gpt-4o-2024-05-13", "gpt-4o-2024-05-13"),  # its own, older price
    ("openai", "gpt-5-mini-2025-08-07", "gpt-5-mini"),
    ("openai", "gpt-4.1-nano", "gpt-4.1-nano"),
    ("openai", "o3-mini", "o3-mini"),
    ("openai", "o3-2025-04-16", "o3"),
    ("anthropic", "claude-opus-4-5-20251101", "claude-opus-4-5"),
    ("anthropic", "claude-haiku-4-5@20251001", "claude-haiku-4-5"),
    ("anthropic", "claude-opus-5", "claude-opus-5"),
    ("anthropic", "claude-opus-5-5", "claude-opus-5-5"),
])
def test_lookup_longest_prefix(table, provider, model, expected):
    assert table.lookup(provider, model).model == expected


@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-4o-audio-preview"),   # a variant, not a snapshot
    ("openai", "gpt-5-chat-latest"),
    ("openai", "gpt-4"),
    ("anthropic", "claude-opus-5-7"),     # unknown model must not inherit opus-5's price
    ("anthropic", "gpt-4o"),              # wrong provider
    ("openai", ""),
])
def test_lookup_refuses_lookalikes(table, provider, model):
    assert table.lookup(provider, model) is None


def _price(**kw):
    data = dict(provider="openai", model="m", input_per_m=2.0, output_per_m=8.0,
                cached_input_per_m=0.5, as_of="2026-10-27", source="https://example.test")
    data.update(kw)
    return p.Price(**data)


def test_cost_bills_cached_tokens_at_cached_rate():
    cost = p.cost_usd(_price(), Usage(input_tokens=1_000_000, output_tokens=500_000, cached_input_tokens=400_000))
    assert cost == pytest.approx(0.6 * 2 + 0.4 * 0.5 + 0.5 * 8)


def test_cost_without_cached_rate_bills_cached_at_input_rate():
    cost = p.cost_usd(_price(cached_input_per_m=None), Usage(1_000_000, 0, 400_000))
    assert cost == pytest.approx(2.0)


def test_cost_with_unreported_cache_count_bills_all_input_at_base():
    assert p.cost_usd(_price(), Usage(1_000_000, 1_000_000, None)) == pytest.approx(10.0)


@pytest.mark.parametrize("usage", [Usage(), Usage(100, None), Usage(None, 100)])
def test_cost_refuses_incomplete_usage(usage):
    assert p.cost_usd(_price(), usage) is None


def test_cost_refuses_impossible_usage_and_missing_price():
    assert p.cost_usd(_price(), Usage(10, 10, 20)) is None
    assert p.cost_usd(None, Usage(10, 10)) is None


def test_long_context_unpublished_price_is_left_unpriced(table):
    price = table.lookup("openai", "gpt-5.5")
    assert p.cost_usd(price, Usage(272_000, 100)) is not None
    assert p.cost_usd(price, Usage(272_001, 100)) is None


def test_long_context_published_price_applies_above_threshold(table):
    price = table.lookup("anthropic", "claude-haiku-5-5")
    short = p.cost_usd(price, Usage(100_000, 0))
    long = p.cost_usd(price, Usage(100_001, 0))
    assert short == pytest.approx(100_000 * 0.10 / 1e6)
    assert long == pytest.approx(100_001 * 0.50 / 1e6)


def test_override_replaces_and_extends(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"prices": [
        {"provider": "openai", "model": "gpt-4o", "input_per_m": 1, "output_per_m": 2, "as_of": "2026-01-01"},
        {"provider": "openai", "model": "my-finetune", "input_per_m": 3, "output_per_m": 4},
    ]}))
    table = p.load_prices(path)
    replaced = table.lookup("openai", "gpt-4o-2024-08-06")
    assert replaced.input_per_m == 1 and replaced.origin == p.ORIGIN_OVERRIDE
    assert replaced.label == "ASSUMED (operator price, 2026-01-01)"
    added = table.lookup("openai", "my-finetune")
    assert added.output_per_m == 4 and str(path) in added.source
    assert table.lookup("openai", "gpt-4o-mini").origin == p.ORIGIN_BUNDLED


def test_override_accepts_a_bare_list(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps([{"provider": "anthropic", "model": "x", "input_per_m": 1, "output_per_m": 1}]))
    assert p.load_prices(path).lookup("anthropic", "x") is not None


@pytest.mark.parametrize("content,message", [
    ("{nope", "not valid JSON"),
    (json.dumps({"prices": [{"provider": "google", "model": "g", "input_per_m": 1, "output_per_m": 1}]}), "provider"),
    (json.dumps([{"provider": "openai", "model": "g", "input_per_m": -1, "output_per_m": 1}]), "non-negative"),
    (json.dumps([{"provider": "openai", "model": "g", "input_per_m": 1}]), "output_per_m"),
    (json.dumps({"x": 1}), "expected a list"),
])
def test_override_errors_are_clear(tmp_path, content, message):
    path = tmp_path / "prices.json"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        p.load_prices(path)


def test_missing_override_file(tmp_path):
    with pytest.raises(ValueError, match="cannot read"):
        p.load_prices(tmp_path / "nope.json")
