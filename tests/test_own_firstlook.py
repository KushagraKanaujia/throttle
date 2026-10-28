"""First look at a brand-new AI job (throttle.own.firstlook). Offline: every
model is an in-process fake."""

from __future__ import annotations

import json

import httpx

from throttle.own import firstlook as fl
from throttle.own import jobs
from throttle.own import judge as judge_mod
from throttle.own import pricing

from own_job_helpers import judge_prefers, py_call, sites_for

PROMPT = "You draft a short, friendly reply to a customer about their {product} problem."
SITE_SRC = py_call("draft_reply", PROMPT, "Customer wrote: {message}", model="gpt-4o-mini").replace(
    "{'role': 'system', 'content': 'You", "{'role': 'system', 'content': f'You")


def site():
    found = sites_for({"app/reply.py": SITE_SRC})[0]
    assert found.system.slots == ["product"] and found.user.slots == ["message"]
    return found


def config(**kw):
    judge = judge_mod.JudgeConfig(provider="openai", model="judge-m", url="http://judge.test/v1/chat/completions")
    return fl.FirstLookConfig(candidate_url="http://cand.test/v1", candidate_model="qwen-7b", candidate_label="qwen",
                              judge=judge, api_url="http://api.test/v1", **kw)


def fake_world(calls, *, examples=None, candidate_text="CANDIDATE reply", api_status=200):
    examples = examples if examples is not None else [
        {"product": "printer", "message": "It jams."}, {"product": "app", "message": "Login fails."},
        {"product": "router", "message": "No wifi since Monday."},
    ]
    judge = judge_prefers("API reply")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((request.url.host, body))
        if request.url.host == "judge.test":
            if body["messages"][0]["content"] == fl.GEN_SYSTEM:
                assert '"product"' in body["messages"][1]["content"] and body["response_format"]["type"] == "json_object"
                return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"examples": examples})}}]})
            return judge(request)
        if request.url.host == "api.test":
            if api_status != 200:
                return httpx.Response(api_status, json={})
            return httpx.Response(200, json={"choices": [{"message": {"content": "API reply"}, "finish_reason": "stop"}],
                                             "usage": {"prompt_tokens": 1000, "completion_tokens": 100}})
        return httpx.Response(200, json={"choices": [{"message": {"content": candidate_text}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 900, "completion_tokens": 80}})

    return httpx.MockTransport(handler)


async def test_first_look_runs_and_is_labelled(tmp_path):
    calls: list = []
    doc = await fl.run_first_look(site(), config(), transport=fake_world(calls), root=tmp_path,
                                  prices=pricing.load_prices())
    assert doc["label"] == fl.LABEL and "not enough data yet" in doc["note"]
    assert doc["inputs"] == 3 and doc["losses"] == 3 and doc["wins"] == 0
    assert doc["calls"] == {"generator": 1, "api": 3, "candidate": 3, "judge": 6}
    # filled prompts went to both models
    api_bodies = [b for host, b in calls if host == "api.test"]
    assert {b["messages"][0]["content"] for b in api_bodies} >= {PROMPT.replace("{product}", "printer")}
    assert any(b["messages"][1]["content"] == "Customer wrote: It jams." for b in api_bodies)
    # gpt-4o-mini has a bundled list price: 1000 in + 100 out per call
    price = pricing.load_prices().lookup("openai", "gpt-4o-mini")
    expected = (1000 * price.input_per_m + 100 * price.output_per_m) / 1_000_000
    assert abs(doc["api_cost_per_call_usd"] - expected) < 1e-12 and doc["api_cost_label"].startswith("LIST PRICE")
    assert doc["candidate_usage"]["input_tokens"] == 2700
    assert len(doc["examples"]) == 3 and doc["examples"][0]["outcome"] == judge_mod.LOSS
    # losses are saved as lessons for later
    assert jobs.lesson_counts(tmp_path)["app/reply.py::draft_reply"]["first_look_loss"] == 3


async def test_api_failure_is_counted_not_judged(tmp_path):
    calls: list = []
    doc = await fl.run_first_look(site(), config(), transport=fake_world(calls, api_status=500), prices=pricing.load_prices())
    assert doc["api_errors"] == 3 and doc["judged"] == 0 and doc["api_cost_per_call_usd"] is None
    assert not [c for c in calls if c[0] == "judge.test" and c[1]["messages"][0]["content"] != fl.GEN_SYSTEM]


async def test_skips_when_there_is_nothing_to_test():
    dynamic = sites_for({"a.py": "def f(m):\n    return c.chat.completions.create(model='gpt-4o', messages=m)\n"})[0]
    doc = await fl.run_first_look(dynamic, config(), transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert "built at run time" in doc["skipped"]
    no_model = sites_for({"a.py": py_call("f", PROMPT, model="x").replace("model='x'", "model=M")})[0]
    doc = await fl.run_first_look(no_model, config(), transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert "--api-model" in doc["skipped"]


def test_parse_examples_tolerates_small_model_output():
    assert fl.parse_examples('```json\n{"examples": [{"a": "x"}, {"a": 3}]}\n```', ["a"], 5) == [{"a": "x"}, {"a": "3"}]
    assert fl.parse_examples('Sure! {"inputs": ["one", "two"]} hope that helps', ["a"], 5) == [{"a": "one"}, {"a": "two"}]
    assert fl.parse_examples('[{"text": "only key"}]', ["a"], 5) == [{"a": "only key"}]
    assert fl.parse_examples("1. first\n2. second", ["a"], 1) == [{"a": "first"}]
    assert fl.parse_examples("nothing useful", ["a", "b"], 3) == []


def test_cached_first_look_is_reused(tmp_path):
    calls: list = []
    first = fl.cached_or_run(site(), config(), root=tmp_path, transport=fake_world(calls), prices=pricing.load_prices())
    n = len(calls)
    again = fl.cached_or_run(site(), config(), root=tmp_path, transport=fake_world(calls), prices=pricing.load_prices())
    assert len(calls) == n and again["cached"] and again["losses"] == first["losses"]
    other = fl.cached_or_run(site(), config(inputs=2), root=tmp_path, transport=fake_world(calls), prices=pricing.load_prices())
    assert len(calls) > n and other["inputs"] == 2
