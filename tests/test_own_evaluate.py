import json

import httpx
import pytest

from throttle import statistics as st
from throttle.cli import build_parser
from throttle.own import evaluate as ev
from throttle.own import judge as j
from throttle.own import records as r

from own_eval_helpers import by_workload_field, exchange


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k, n, low, high", [
    (5, 10, 0.2366, 0.7634),
    (0, 10, 0.0, 0.2775),
    (10, 10, 0.7225, 1.0),
    (81, 263, 0.2553, 0.3662),
    (1, 1, 0.2065, 1.0),
])
def test_wilson_known_values(k, n, low, high):
    ci = st.wilson_interval_95(k, n)
    assert ci["low"] == pytest.approx(low, abs=1e-4)
    assert ci["high"] == pytest.approx(high, abs=1e-4)
    assert ci["point"] == pytest.approx(k / n) and ci["n"] == n


def test_wilson_empty_and_invalid():
    assert st.wilson_interval_95(0, 0) == {"low": None, "high": None, "point": None, "n": 0}
    with pytest.raises(ValueError):
        st.wilson_interval_95(3, 2)


def test_paired_bootstrap():
    diffs = [1] * 50 + [-1] * 50 + [0] * 100
    a = st.paired_bootstrap_interval_95(diffs, seed=3)
    b = st.paired_bootstrap_interval_95(diffs, seed=3)
    assert a == b
    assert a["estimate"] == 0 and a["n"] == 200 and a["resamples"] == 2000
    # SE = sqrt(0.5 / 200) = 0.05, so the 95% CI is about +/-0.098.
    assert -0.12 < a["low"] < -0.08 and 0.08 < a["high"] < 0.12
    assert st.paired_bootstrap_interval_95([0] * 40, seed=1)[["low", "high"][0]] == 0
    ones = st.paired_bootstrap_interval_95([1] * 40, seed=1)
    assert ones["low"] == ones["high"] == 1
    single = st.paired_bootstrap_interval_95([1], seed=1)
    assert single["estimate"] == 1 and single["low"] is None
    assert st.paired_bootstrap_interval_95([], seed=1)["estimate"] is None


# ---------------------------------------------------------------------------
# schema validator
# ---------------------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "title": "Invoice",
    "properties": {
        "id": {"type": "integer"},
        "status": {"type": "string", "enum": ["paid", "open"]},
        "lines": {"type": "array", "items": {"type": "object", "properties": {"sku": {"type": "string"}},
                                             "required": ["sku"]}},
        "note": {"type": ["string", "null"]},
    },
    "required": ["id", "status"],
    "additionalProperties": False,
}


@pytest.mark.parametrize("value, ok", [
    ({"id": 1, "status": "paid"}, True),
    ({"id": 1.0, "status": "open", "lines": [{"sku": "a"}], "note": None}, True),
    ({"id": "1", "status": "paid"}, False),
    ({"id": True, "status": "paid"}, False),
    ({"id": 1}, False),
    ({"id": 1, "status": "late"}, False),
    ({"id": 1, "status": "paid", "extra": 1}, False),
    ({"id": 1, "status": "paid", "lines": [{}]}, False),
    ({"id": 1, "status": "paid", "note": 3}, False),
    ([1], False),
])
def test_schema_validator(value, ok):
    assert ev.validate_schema(SCHEMA, value) is ok


@pytest.mark.parametrize("schema", [
    {"type": "string", "pattern": "^a"},
    {"anyOf": [{"type": "string"}]},
    {"type": "object", "additionalProperties": {"type": "string"}},
    {"$ref": "#/defs/x"},
    {"type": "date"},
    {"type": "object", "properties": {"x": {"type": "string", "minLength": 1}}},
])
def test_schema_unsupported_is_none(schema):
    assert ev.validate_schema(schema, "a") is None


def test_enum_distinguishes_bool_from_int():
    assert ev.validate_schema({"enum": [1]}, True) is False
    assert ev.validate_schema({"enum": [True]}, True) is True


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def test_checks_json_schema_tools_truncation():
    ex = exchange("q", output='{"id": 1, "status": "paid"}',
                  response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA}})
    norm = r.normalize(ex)
    good = ev.run_checks(norm, "extraction", ev.Side('```json\n{"id": 2, "status": "open"}\n```', [], "stop"), None)
    assert good == {"json_valid": True, "schema_valid": True, "truncated": True}
    bad = ev.run_checks(norm, "extraction", ev.Side('{"id": 2', [], "length"), None)
    assert bad == {"json_valid": False, "schema_valid": False, "truncated": False}
    failed = ev.run_checks(norm, "extraction", ev.Side(None, [], None, failed=True), None)
    assert set(failed.values()) == {False}

    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    tnorm = r.normalize(exchange("t", tools=tools))
    ok_call = [{"name": "lookup", "arguments": '{"q": 1}'}]
    assert ev.run_checks(tnorm, "agent", ev.Side("", ok_call, "tool_calls"), None)["tool_calls_valid"] is True
    for calls in ([{"name": "other", "arguments": "{}"}], [{"name": "lookup", "arguments": "{bad"}],
                  [{"name": "lookup", "arguments": "[1]"}]):
        assert ev.run_checks(tnorm, "agent", ev.Side("", calls, "tool_calls"), None)["tool_calls_valid"] is False

    cnorm = r.normalize(exchange("c", output="Positive."))
    assert ev.run_checks(cnorm, "classification", ev.Side("positive", [], "stop"), "Positive.")["exact_match"] is True
    assert ev.run_checks(cnorm, "classification", ev.Side("negative", [], "stop"), "Positive.")["exact_match"] is False
    assert "exact_match" not in ev.run_checks(cnorm, "chat", ev.Side("x", [], "stop"), "Positive.")


# ---------------------------------------------------------------------------
# verdict: every branch
# ---------------------------------------------------------------------------


def _ci(k, n):
    return st.wilson_interval_95(k, n)


SAME = {"json_valid": {"incumbent": _ci(50, 50), "candidate": _ci(49, 50)}}
WORSE_CHECK = {"json_valid": {"incumbent": _ci(50, 50), "candidate": _ci(30, 50)}}


def _decide(low, high, checks=SAME, **kw):
    params = dict(judged=True, n=50, min_n=30, margin=0.05, net_ci={"low": low, "high": high}, checks=checks)
    params.update(kw)
    return ev.decide(**params)[0]


def test_verdict_branches():
    assert _decide(-0.03, 0.04) == ev.NO_DIFF
    assert _decide(0.08, 0.3) == ev.BETTER
    assert _decide(-0.4, -0.1) == ev.WORSE
    assert _decide(-0.2, 0.2) == ev.INCONCLUSIVE          # contains 0 but too wide
    assert _decide(0.02, 0.2) == ev.INCONCLUSIVE          # positive, not above margin
    assert _decide(0.08, 0.3, checks=WORSE_CHECK) == ev.WORSE   # a worse check beats a judge win
    assert _decide(-0.03, 0.04, checks=WORSE_CHECK) == ev.WORSE
    assert _decide(-0.03, 0.04, n=29) == ev.INSUFFICIENT
    assert _decide(-0.03, 0.04, judge_errors=11, judge_attempts=50) == ev.INSUFFICIENT
    assert _decide(-0.03, 0.04, judge_errors=10, judge_attempts=50) == ev.NO_DIFF
    assert _decide(None, None) == ev.INSUFFICIENT


def test_verdict_checks_only_is_never_better():
    assert ev.decide(judged=False, n=50, min_n=30, margin=0.05, net_ci=None, checks=SAME)[0] == ev.CHECKS_ONLY
    assert ev.decide(judged=False, n=50, min_n=30, margin=0.05, net_ci=None, checks=WORSE_CHECK)[0] == ev.WORSE
    assert ev.decide(judged=False, n=5, min_n=30, margin=0.05, net_ci=None, checks=SAME)[0] == ev.INSUFFICIENT


def test_verdict_reasons_name_the_check():
    verdict, reasons = ev.decide(judged=True, n=50, min_n=30, margin=0.05,
                                 net_ci={"low": 0.0, "high": 0.01}, checks=WORSE_CHECK)
    assert verdict == ev.WORSE and any("json_valid" in x for x in reasons)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


def _shadow(ex, text, *, status=200, error=None, finish="stop"):
    return r.ShadowResult(exchange_id=ex.id, candidate="cand", candidate_model="qwen", output_text=text,
                          tool_calls=[], usage=r.Usage(input_tokens=10, output_tokens=2) if text else r.Usage(),
                          latency_ms=100.0 + len(text or ""), status=status, finish_reason=finish, error=error)


def _populate(root, n_class=40, n_extract=10):
    pairs = []
    for i in range(n_class):
        ex = exchange(f"review {i}", output="positive", kind="classification")
        r.append_exchange(ex, root)
        r.append_shadow(_shadow(ex, "positive" if i % 10 else "negative"), root)
        pairs.append(ex)
    dup = exchange("review 0", output="positive", kind="classification")   # same prompt again
    r.append_exchange(dup, root)
    r.append_shadow(_shadow(dup, "positive"), root)
    fmt = {"type": "json_object"}
    for i in range(n_extract):
        ex = exchange(f"invoice {i}", output='{"id": %d}' % i, kind="extraction", response_format=fmt)
        r.append_exchange(ex, root)
        r.append_shadow(_shadow(ex, '{"id": %d}' % i), root)
    lone = exchange("unshadowed", kind="chat")
    r.append_exchange(lone, root)
    down = exchange("transport", kind="chat")
    r.append_exchange(down, root)
    r.append_shadow(_shadow(down, None, status=0, error="connect: refused"), root)
    return pairs


def _judge_transport(rule, seen):
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        prompt = body["messages"][-1]["content"]
        a = prompt.split("=== Answer A ===\n")[1].split("\n\n=== Answer B")[0]
        b = prompt.split("=== Answer B ===\n")[1].split("\n\nWhich answer")[0]
        return httpx.Response(200, json={"choices": [{"message": {"content": rule(a, b)}}]})
    return httpx.MockTransport(handler)


def test_run_eval_end_to_end(tmp_path):
    _populate(tmp_path)
    seen = []
    tie_unless_negative = lambda a, b: "B" if a == "negative" else ("A" if b == "negative" else "TIE")
    config = j.JudgeConfig(provider="openai", model="judge-m", url="http://judge/v1/chat/completions")
    doc = ev.run_eval(tmp_path, "cand", judge_config=config, budget=j.JudgeBudget(max_calls=1000),
                      classify=by_workload_field, transport=_judge_transport(tie_unless_negative, seen),
                      run_id="run1")
    assert set(doc["types"]) == {"classification", "extraction"}   # chat had no usable shadow
    cls = doc["types"]["classification"]
    assert cls["n"] == 40                                         # duplicate prompt dropped
    assert (cls["wins"], cls["ties"], cls["losses"]) == (0, 36, 4)
    assert cls["judged"] == 40 and cls["judge_errors"] == 0
    assert cls["checks"]["exact_match"]["incumbent"]["point"] == 1.0
    assert cls["checks"]["exact_match"]["candidate"]["point"] == pytest.approx(0.9)
    assert cls["net_ci"]["estimate"] == pytest.approx(-0.1)
    assert cls["verdict"] == ev.INCONCLUSIVE
    assert cls["candidate_usage"] == {"input_tokens": 400, "output_tokens": 80, "excluded": 0}
    assert cls["candidate_latency_ms_p50"] == pytest.approx(108.0)
    ext = doc["types"]["extraction"]
    assert ext["verdict"] == ev.INSUFFICIENT and ext["checks"]["json_valid"]["candidate"]["point"] == 1.0
    assert len(seen) == 100 and doc["judge_calls"] == 100
    assert doc["judge"] == {"provider": "openai", "model": "judge-m"} and doc["candidate_model"] == "qwen"

    saved = json.loads((tmp_path / "evals" / "run1.json").read_text())
    assert saved["types"]["classification"]["verdict"] == ev.INCONCLUSIVE
    for key in ("run_id", "ts", "candidate", "candidate_model", "judge", "margin", "min_n", "types"):
        assert key in saved

    kind, cand, ids = ev.load_evalset(cls["evalset_path"])
    assert kind == "classification" and cand == "cand" and len(ids) == 40
    assert cls["evalset_sha256"] == ev.evalset_sha256("cand", ids)
    assert ids == sorted(ids)
    again = ev.build_eval_sets(tmp_path, "cand", only_ids={"classification": ids})
    assert [p.exchange.id for p in again["classification"]] == ids


def test_candidate_failure_counts_as_loss_without_a_judge_call(tmp_path):
    for i in range(3):
        ex = exchange(f"q{i}", kind="chat")
        r.append_exchange(ex, tmp_path)
        if i == 0:
            r.append_shadow(_shadow(ex, None, status=400, error="context_overflow: too long"), tmp_path)
        else:
            r.append_shadow(_shadow(ex, "ok"), tmp_path)
    seen = []
    config = j.JudgeConfig(provider="openai", model="m", url="http://judge/v1/chat/completions")
    doc = ev.run_eval(tmp_path, "cand", judge_config=config, min_n=1, classify=by_workload_field,
                      transport=_judge_transport(lambda a, b: "TIE", seen))
    t = doc["types"]["chat"]
    assert (t["wins"], t["ties"], t["losses"], t["candidate_errors"]) == (0, 2, 1, 1)
    assert len(seen) == 4
    assert t["checks"]["truncated"]["candidate"]["point"] == pytest.approx(2 / 3)


def test_judge_budget_stops_judging(tmp_path):
    _populate(tmp_path, n_class=10, n_extract=0)
    seen = []
    config = j.JudgeConfig(provider="openai", model="m", url="http://judge/v1/chat/completions")
    doc = ev.run_eval(tmp_path, "cand", judge_config=config, budget=j.JudgeBudget(max_calls=6), min_n=1,
                      classify=by_workload_field, judge_concurrency=1,
                      transport=_judge_transport(lambda a, b: "TIE", seen))
    t = doc["types"]["classification"]
    assert len(seen) == 6 and t["judged"] == 3 and t["judge_skipped"] == 7
    assert "judge call cap" in doc["judge_stopped"]


def test_judge_errors_over_20_percent_is_insufficient(tmp_path):
    _populate(tmp_path, n_class=40, n_extract=0)
    count = {"n": 0}

    def flaky(a, b):
        count["n"] += 1
        return "garbage" if count["n"] % 3 == 0 else "TIE"

    config = j.JudgeConfig(provider="openai", model="m", url="http://judge/v1/chat/completions")
    doc = ev.run_eval(tmp_path, "cand", judge_config=config, classify=by_workload_field, judge_concurrency=1,
                      transport=_judge_transport(flaky, []))
    t = doc["types"]["classification"]
    assert t["judge_errors"] > 0.2 * (t["judged"] + t["judge_errors"])
    assert t["verdict"] == ev.INSUFFICIENT


def test_cli_no_judge_table_and_json(tmp_path, capsys):
    _populate(tmp_path)
    args = build_parser().parse_args(["eval", "--candidate", "cand", "--no-judge", "--dir", str(tmp_path)])
    assert ev.handle(args, classify=by_workload_field) == 0
    out = capsys.readouterr().out
    assert "classification" in out and "CHECKS ONLY" in out and "INSUFFICIENT DATA" in out
    args = build_parser().parse_args(["eval", "--candidate", "cand", "--no-judge", "--json", "--dir", str(tmp_path)])
    assert ev.handle(args, classify=by_workload_field) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["judge"] is None
    # 36/40 exact matches: Wilson CIs [0.77, 0.96] vs [0.91, 1.0] overlap, so no check is worse.
    assert doc["types"]["classification"]["verdict"] == ev.CHECKS_ONLY
    assert doc["types"]["extraction"]["verdict"] == ev.INSUFFICIENT
    assert all(t["verdict"] != ev.BETTER for t in doc["types"].values())


def test_cli_usage_errors(tmp_path, capsys, monkeypatch):
    parse = build_parser().parse_args
    assert ev.handle(parse(["eval", "--candidate", "none", "--dir", str(tmp_path)])) == 2
    assert "--judge-model" in capsys.readouterr().err
    assert ev.handle(parse(["eval", "--candidate", "none", "--no-judge", "--dir", str(tmp_path)])) == 2
    assert "throttle shadow" in capsys.readouterr().err
    monkeypatch.delenv("JKEY", raising=False)
    assert ev.handle(parse(["eval", "--candidate", "x", "--judge-model", "m", "--judge-key-env", "JKEY",
                            "--dir", str(tmp_path)])) == 2
