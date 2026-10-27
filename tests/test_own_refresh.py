import asyncio
import json

import httpx
import pytest

from throttle.cli import build_parser
from throttle.own import evaluate as ev
from throttle.own import judge as j
from throttle.own import records as r
from throttle.own import refresh as rf
from throttle.own import shadow as sh

from own_eval_helpers import by_workload_field, candidate_transport, chat_response, exchange, last_user

JUDGE = j.JudgeConfig(provider="openai", model="judge-m", url="http://judge/v1/chat/completions")


def _shadow(ex, text, label="cur"):
    return r.ShadowResult(exchange_id=ex.id, candidate=label, candidate_model="old-m", output_text=text,
                          tool_calls=[], usage=r.Usage(input_tokens=10, output_tokens=2),
                          latency_ms=100.0, status=200, finish_reason="stop")


def _setup(root, n=40, current_answer="positive"):
    """Capture n classification exchanges, shadow them on 'cur', save its eval set,
    then capture more traffic that is NOT in the saved set."""

    for i in range(n):
        ex = exchange(f"review {i}", output="positive", kind="classification")
        r.append_exchange(ex, root)
        r.append_shadow(_shadow(ex, current_answer), root)
    ev.run_eval(root, "cur", judge_config=None, min_n=1, classify=by_workload_field, run_id="first")
    for i in range(5):
        r.append_exchange(exchange(f"later {i}", output="positive", kind="classification"), root)
    [path] = (root / "evals").glob("evalset-*.json")
    return ev.load_evalset(path)[2]


def _judge_transport(rule, seen=None):
    def handler(request):
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        prompt = body["messages"][-1]["content"]
        a = prompt.split("=== Answer A ===\n")[1].split("\n\n=== Answer B")[0]
        b = prompt.split("=== Answer B ===\n")[1].split("\n\nWhich answer")[0]
        return httpx.Response(200, json={"choices": [{"message": {"content": "Reason.\n" + rule(a, b)}}]})
    return httpx.MockTransport(handler)


def prefer_positive(a, b):
    if a == b:
        return "TIE"
    return "A" if a == "positive" else ("B" if b == "positive" else "TIE")


def _run(root, new_answer, *, calls=None, min_n=30, judge_rule=prefer_positive, **kw):
    return rf.run_refresh(
        root, current="cur", new_label="new", new_url="http://gpu:8000/v1", new_model="new-m",
        judge_config=JUDGE, min_n=min_n, run_id="r1",
        candidate_transport=candidate_transport(lambda body: chat_response(new_answer(last_user(body))), calls),
        judge_transport=_judge_transport(judge_rule), **kw,
    )


def test_reuses_exactly_the_saved_ids(tmp_path):
    ids = _setup(tmp_path)
    calls = []
    doc = _run(tmp_path, lambda text: "positive", calls=calls)
    assert len(calls) == 40 and all(not last_user(c[2]).startswith("later") for c in calls)
    assert {res.exchange_id for res in r.iter_shadow("new", tmp_path)} == set(ids)
    t = doc["types"]["classification"]
    assert t["exchange_ids"] == 40 and t["compared"] == 40 and t["missing_ids"] == []
    assert t["evalset_sha256"] == ev.evalset_sha256("cur", ids)
    assert t["vs_api"]["new_evalset_sha256"] == ev.evalset_sha256("new", ids)
    # A second run sends nothing new: every id already has an answer.
    calls2 = []
    doc2 = _run(tmp_path, lambda text: "positive", calls=calls2)
    assert calls2 == [] and doc2["shadow"]["already_had"] == 40


def test_backfill_only_ids_filter(tmp_path):
    exs = [exchange(f"q{i}", kind="chat") for i in range(6)]
    picked = sh.select_backfill(exs, only_ids={exs[1].id, exs[4].id}, classify=by_workload_field)
    assert {e.id for e, _, _ in picked} == {exs[1].id, exs[4].id}
    assert len(sh.select_backfill(exs, classify=by_workload_field)) == 6


# NO MATERIAL DIFFERENCE needs the loss-rate 95% upper bound <= the 0.05
# margin, which zero losses reach at about 75 items; 80 is used below.
def test_upgrade_when_equal(tmp_path):
    _setup(tmp_path, n=80)
    t = _run(tmp_path, lambda text: "positive")["types"]["classification"]
    assert t["vs_api"]["verdict"] == ev.NO_DIFF and t["vs_current"]["verdict"] == ev.NO_DIFF
    assert t["verdict"] == rf.UPGRADE and t["better_than_current"] is False
    assert rf.COST_NOTE in t["notes"] and t["cost"] == {"measured": False}


def test_upgrade_and_better_than_current(tmp_path):
    _setup(tmp_path, n=80, current_answer="negative")
    t = _run(tmp_path, lambda text: "positive")["types"]["classification"]
    assert t["vs_api"]["verdict"] == ev.NO_DIFF
    assert t["vs_current"]["verdict"] == ev.BETTER and t["vs_current"]["wins"] == 80
    assert t["verdict"] == rf.UPGRADE and t["better_than_current"] is True
    assert "better than the current owned model" in " ".join(t["notes"])


def test_stay_when_worse(tmp_path):
    _setup(tmp_path)
    t = _run(tmp_path, lambda text: "negative")["types"]["classification"]
    assert t["vs_api"]["verdict"] == ev.WORSE and t["vs_current"]["verdict"] == ev.WORSE
    assert t["verdict"] == rf.STAY and rf.COST_NOTE not in t["notes"]


def test_needs_more_data_when_too_few(tmp_path):
    _setup(tmp_path, n=10)
    t = _run(tmp_path, lambda text: "positive")["types"]["classification"]
    assert t["vs_api"]["verdict"] == ev.INSUFFICIENT
    assert t["verdict"] == rf.NEEDS_MORE_DATA


@pytest.mark.parametrize("vs_api, vs_current, verdict, better", [
    (ev.NO_DIFF, ev.NO_DIFF, rf.UPGRADE, False),
    (ev.BETTER, ev.BETTER, rf.UPGRADE, True),
    (ev.NO_DIFF, ev.BETTER, rf.UPGRADE, True),
    (ev.WORSE, ev.BETTER, rf.STAY, False),
    (ev.BETTER, ev.WORSE, rf.STAY, False),
    (ev.NO_DIFF, ev.INCONCLUSIVE, rf.NEEDS_MORE_DATA, False),
    (ev.INCONCLUSIVE, ev.NO_DIFF, rf.NEEDS_MORE_DATA, False),
    (ev.INSUFFICIENT, ev.INSUFFICIENT, rf.NEEDS_MORE_DATA, False),
])
def test_decide_table(vs_api, vs_current, verdict, better):
    got, got_better, notes = rf.decide(vs_api, vs_current)
    assert (got, got_better) == (verdict, better)
    assert (rf.COST_NOTE in notes) == (verdict == rf.UPGRADE)


def _pair(text, label, ok=True):
    ex = exchange("q", output="api")
    norm = r.normalize(ex)
    res = _shadow(ex, text, label)
    if not ok:
        res = r.ShadowResult(exchange_id=ex.id, candidate=label, candidate_model="m", output_text=None,
                             tool_calls=[], usage=r.Usage(), latency_ms=1.0, status=400,
                             error="context_overflow: too long")
    return ev.Pair(ex, norm, res, "chat")


def _pairwise(items, rule):
    seen = []
    out, _ = asyncio.run(rf._judge_current_vs_new(
        {"chat": items}, config=JUDGE, budget=j.JudgeBudget(max_calls=100), concurrency=1,
        transport=_judge_transport(rule, seen),
    ))
    return out["chat"].outcomes, seen


def test_pairwise_both_orders_and_disagreement_is_order_disagree():
    outcomes, seen = _pairwise([(_pair("old", "cur"), _pair("new", "new"))], lambda a, b: "A")
    assert outcomes == [j.ORDER_DISAGREE] and len(seen) == 2       # always "A": the orders disagree
    first, second = (s["messages"][-1]["content"] for s in seen)
    assert "=== Answer A ===\nold" in first and "=== Answer A ===\nnew" in second
    outcomes, _ = _pairwise([(_pair("old", "cur"), _pair("new", "new"))],
                            lambda a, b: "A" if a == "new" else "B")
    assert outcomes == [j.WIN]
    outcomes, _ = _pairwise([(_pair("old", "cur"), _pair("new", "new"))],
                            lambda a, b: "A" if a == "old" else "B")
    assert outcomes == [j.LOSS]


def test_pairwise_failures_skip_the_judge():
    items = [
        (_pair("old", "cur"), _pair(None, "new", ok=False)),
        (_pair(None, "cur", ok=False), _pair("new", "new")),
        (_pair(None, "cur", ok=False), _pair(None, "new", ok=False)),
    ]
    outcomes, seen = _pairwise(items, lambda a, b: "A")
    assert outcomes == [j.LOSS, j.WIN, j.TIE] and seen == []


def test_no_saved_evalset_is_a_clear_error(tmp_path, capsys):
    with pytest.raises(rf.NoEvalSet, match="throttle eval"):
        rf.find_evalsets(tmp_path, "cur")
    args = build_parser().parse_args([
        "refresh", "--current", "cur", "--new-label", "new", "--new-url", "http://x", "--new-model", "m",
        "--judge-model", "judge-m", "--dir", str(tmp_path),
    ])
    assert rf.handle(args) == 2
    assert "Run `throttle eval --candidate cur` first" in capsys.readouterr().err


def test_newest_evalset_per_type_and_types_filter(tmp_path):
    root = tmp_path
    (root / "evals").mkdir()
    for created, ids in (("2026-10-01T00:00:00Z", ["a"]), ("2026-10-20T00:00:00Z", ["a", "b"])):
        sha = ev.evalset_sha256("cur", ids)
        (root / "evals" / f"evalset-chat-{sha[:12]}.json").write_text(json.dumps(
            {"type": "chat", "candidate": "cur", "sha256": sha, "exchange_ids": ids, "created": created}))
    sha = ev.evalset_sha256("other", ["z"])
    (root / "evals" / f"evalset-code-{sha[:12]}.json").write_text(json.dumps(
        {"type": "code", "candidate": "other", "sha256": sha, "exchange_ids": ["z"], "created": "x"}))
    found = rf.find_evalsets(root, "cur")
    assert list(found) == ["chat"] and found["chat"]["ids"] == ["a", "b"]
    with pytest.raises(rf.NoEvalSet):
        rf.find_evalsets(root, "cur", ["code"])


def test_json_schema_and_cli(tmp_path, capsys):
    _setup(tmp_path, n=80)
    args = build_parser().parse_args([
        "refresh", "--current", "cur", "--new-label", "new", "--new-url", "http://gpu:8000/v1",
        "--new-model", "new-m", "--judge-model", "judge-m", "--judge-url", "http://judge/v1",
        "--min-n", "30", "--dir", str(tmp_path), "--json",
    ])
    cand = candidate_transport(lambda body: chat_response("positive"))
    assert rf.handle(args, transport=cand, judge_transport=_judge_transport(prefer_positive)) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == rf.SCHEMA_ID
    for key in ("run_id", "ts", "current", "new", "new_model", "judge", "margin", "min_n", "judge_calls",
                "judge_stopped", "shadow", "eval_run_id", "eval_path", "types"):
        assert key in doc
    t = doc["types"]["classification"]
    for key in ("evalset_sha256", "evalset_path", "exchange_ids", "compared", "missing_ids", "vs_api",
                "vs_current", "verdict", "better_than_current", "cost", "notes"):
        assert key in t
    assert set(t["vs_current"]) >= {"verdict", "n", "wins", "ties", "losses", "order_disagree", "net_ci",
                                    "checks", "reasons"}
    assert set(t["vs_current"]["checks"]["exact_match"]) == {"current", "new", "unsupported"}
    saved = json.loads((tmp_path / "evals" / f"refresh-{doc['run_id']}.json").read_text())
    assert saved["types"]["classification"]["verdict"] == rf.UPGRADE
    assert doc["judge_calls"] == 320    # 80 items x 2 orders x (vs API, vs current)

    args.json = False
    assert rf.handle(args, transport=cand, judge_transport=_judge_transport(prefer_positive)) == 0
    out = capsys.readouterr().out
    assert "UPGRADE" in out and t["evalset_sha256"][:12] in out and rf.COST_NOTE in out


def test_cli_requires_flags(tmp_path, capsys):
    args = build_parser().parse_args(["refresh", "--current", "cur", "--dir", str(tmp_path)])
    assert rf.handle(args) == 2
    assert "--new-label" in capsys.readouterr().err


def test_watch_hf_is_mocked_and_lists_newer(tmp_path, capsys):
    seen = []

    def handler(request):
        seen.append(request.url)
        if request.url.path == "/api/models/Qwen/Qwen2.5-7B-Instruct":
            return httpx.Response(200, json={"id": "Qwen/Qwen2.5-7B-Instruct", "sha": "abc123def4567890",
                                             "lastModified": "2025-01-01T00:00:00.000Z"})
        assert request.url.params["author"] == "Qwen" and request.url.params["search"] == "Qwen2.5"
        return httpx.Response(200, json=[
            {"id": "Qwen/Qwen2.5-7B-Instruct-1M", "lastModified": "2025-06-01T00:00:00.000Z"},
            {"id": "Qwen/Qwen2.5-7B-Instruct", "lastModified": "2025-01-01T00:00:00.000Z"},
            {"id": "Qwen/Qwen2.5-0.5B", "lastModified": "2024-09-01T00:00:00.000Z"},
        ])

    args = build_parser().parse_args(["refresh", "--watch-hf", "Qwen/Qwen2.5-7B-Instruct", "--dir", str(tmp_path)])
    assert rf.handle(args, hf_transport=httpx.MockTransport(handler)) == 0
    out = capsys.readouterr().out
    assert "abc123def456" in out and "Qwen/Qwen2.5-7B-Instruct-1M" in out and "0.5B" not in out
    assert len(seen) == 2 and all(u.host == "huggingface.co" for u in seen)
    args = build_parser().parse_args(["refresh", "--watch-hf", "no-slash"])
    assert rf.handle(args, hf_transport=httpx.MockTransport(handler)) == 2


def test_name_stem():
    assert rf.name_stem("Qwen2.5-72B-Instruct") == "Qwen2.5"
    assert rf.name_stem("Llama-3.1-8B") == "Llama-3.1"
    assert rf.name_stem("gemma-2-9b-it") == "gemma-2"


def test_position_biased_judge_cannot_upgrade(tmp_path):
    _setup(tmp_path, n=80)
    t = _run(tmp_path, lambda text: "positive", judge_rule=lambda a, b: "A")["types"]["classification"]
    assert t["vs_current"]["order_disagree"] == 80 and t["vs_current"]["verdict"] == ev.INSUFFICIENT
    assert t["vs_api"]["order_disagree"] == 80 and t["vs_api"]["verdict"] == ev.INSUFFICIENT
    assert t["verdict"] == rf.NEEDS_MORE_DATA


def test_small_all_tie_refresh_is_not_an_upgrade(tmp_path):
    _setup(tmp_path, n=40)
    t = _run(tmp_path, lambda text: "positive")["types"]["classification"]
    assert t["vs_current"]["verdict"] == ev.INCONCLUSIVE and t["verdict"] == rf.NEEDS_MORE_DATA
