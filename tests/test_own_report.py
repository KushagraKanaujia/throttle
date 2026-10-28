"""Tests for ``throttle report`` (stage 5). All offline."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from own_eval_helpers import exchange
from throttle import check as check_mod
from throttle.own import evaluate as ev
from throttle.own import pricing
from throttle.own import records as r
from throttle.own import report as rep

START = datetime(2026, 10, 1, tzinfo=timezone.utc)
PRICES = pricing.load_prices()


def _ts(hours: float) -> str:
    return (START + timedelta(hours=hours)).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _traffic(per_type: dict[str, int], *, days: float = 2.0) -> list[r.Exchange]:
    """gpt-4.1 exchanges typed by the request text; 1000 in / 200 out tokens each."""

    out = []
    total = sum(per_type.values())
    i = 0
    for kind, n in per_type.items():
        for _ in range(n):
            ex = exchange(f"{kind} {i}", kind=kind, ts=_ts(days * 24 * i / max(1, total - 1)))
            ex.model = "gpt-4.1"
            ex.request["model"] = "gpt-4.1"
            ex.usage = r.Usage(input_tokens=1000, output_tokens=200)
            out.append(ex)
            i += 1
    return out


def _typed(ex: r.Exchange, norm=None) -> str:
    return ex.workload_type or "chat"


@pytest.fixture(autouse=True)
def _classifier(monkeypatch):
    # Classify by the fixture's workload_type field, like the eval tests do.
    monkeypatch.setattr(rep.wl, "classify", _typed)


def _wilson(k: int, n: int) -> dict:
    from throttle import statistics as stats
    return stats.wilson_interval_95(k, n)


def _type_eval(verdict: str, *, wins=2, ties=36, losses=2, net=(-0.04, 0.04), checks=None,
               cand_tokens=(500, 100), n=40, excluded=0, errors=0) -> dict:
    return {
        "n": n, "judged": wins + ties + losses, "wins": wins, "ties": ties, "losses": losses,
        "candidate_errors": errors,
        "net_ci": {"estimate": 0.0, "low": net[0], "high": net[1]} if net else None,
        "checks": checks or {},
        "candidate_usage": {"input_tokens": cand_tokens[0] * (n - errors - excluded),
                            "output_tokens": cand_tokens[1] * (n - errors - excluded), "excluded": excluded},
        "verdict": verdict, "reasons": [f"because {verdict}"],
    }


def _eval_doc(types: dict, candidate="cand") -> dict:
    return {"schema": ev.SCHEMA_ID, "run_id": "run1", "ts": "2026-10-03T00:00:00.000Z", "candidate": candidate,
            "candidate_model": "open-70b", "judge": {"provider": "openai", "model": "j"}, "types": types}


def _check(mean=0.30, low=0.27, high=0.33, model="open-70b", rate=4.0, gpus=2, cid="c1",
           created="2026-10-02T00:00:00Z") -> dict:
    return {
        "record_type": check_mod.RECORD_TYPE, "record_version": 1, "id": cid, "created_at": created,
        "endpoint": "http://gpu:8000/v1/chat/completions", "metric": "total",
        "fingerprint": {"model": model, "gpu_hourly_rate_usd": rate, "gpu_count": gpus},
        "workload": {},
        "result": {"total": {"mean": mean, "ci_low": low, "ci_high": high, "n_blocks": 5},
                   "output": {"mean": mean * 5, "ci_low": low * 5, "ci_high": high * 5, "n_blocks": 5}},
    }


def _build(per_type, types, check=None, *, rate=2.0, gpus=1, scale=1):
    return rep.build_report(_traffic(per_type), PRICES, candidate="cand", eval_doc=_eval_doc(types),
                            check_record=check, gpu_hourly_rate=rate, gpus=gpus, traffic_scale=scale,
                            generated_at="2026-10-03T00:00:00Z")


def _row(report, kind):
    return next(w for w in report["workloads"] if w["type"] == kind)


FMT_OK = {"tool_calls_valid": {"incumbent": _wilson(40, 40), "candidate": _wilson(39, 40), "unsupported": 0}}
FMT_BAD = {"tool_calls_valid": {"incumbent": _wilson(40, 40), "candidate": _wilson(20, 40), "unsupported": 0}}


# ---------------------------------------------------------------------------
# Recommendation branches
# ---------------------------------------------------------------------------


def test_every_recommendation_branch():
    types = {
        "classification": _type_eval(ev.NO_DIFF),
        "extraction": _type_eval(ev.BETTER, wins=20, ties=18, losses=2, net=(0.2, 0.6)),
        "agent": _type_eval(ev.WORSE, wins=2, ties=10, losses=28, net=(-0.8, -0.5), checks=FMT_BAD),
        "code": _type_eval(ev.WORSE, wins=2, ties=10, losses=28, net=(-0.8, -0.5), checks=FMT_OK),
        "summarization": _type_eval(ev.INCONCLUSIVE, net=(-0.2, 0.1)),
        "chat": _type_eval(ev.INSUFFICIENT, n=10, ties=10, wins=0, losses=0),
    }
    per_type = {k: 60 for k in types} | {"other": 20}
    report = _build(per_type, types, _check(), scale=1000)
    got = {w["type"]: w["recommendation"] for w in report["workloads"]}
    assert got == {
        "classification": rep.MOVE, "extraction": rep.MOVE, "agent": rep.KEEP, "code": rep.NEEDS_FINETUNE,
        "summarization": rep.MORE_DATA, "chat": rep.MORE_DATA, "other": rep.MORE_DATA,
    }
    assert "tool_calls_valid" in _row(report, "agent")["reasons"][0]
    assert "fine-tuning" in _row(report, "code")["reasons"][1]
    assert "no shadow answers" in _row(report, "other")["reasons"][0]
    for w in report["workloads"]:
        assert w["reasons"] and all(isinstance(x, str) and x for x in w["reasons"])


def test_worse_without_format_checks_is_needs_finetune():
    q = rep._quality(_type_eval(ev.WORSE, net=(-0.5, -0.2)))
    rec, reasons = rep.recommend(q, owned_measured=False, conservative_savings=None, api_usd=None,
                                 owned_high=None, not_measured="not measured")
    assert rec == rep.NEEDS_FINETUNE and "no format checks apply" in reasons[1]


def test_checks_only_and_no_shadow_need_more_data():
    q = rep._quality(_type_eval(ev.CHECKS_ONLY, net=None, wins=0, ties=0, losses=0))
    assert rep.recommend(q, owned_measured=True, conservative_savings=10.0, api_usd=20.0, owned_high=10.0,
                         not_measured="")[0] == rep.MORE_DATA
    assert rep.recommend(None, owned_measured=True, conservative_savings=10.0, api_usd=20.0, owned_high=10.0,
                         not_measured="")[0] == rep.MORE_DATA


def test_measured_cost_without_saving_is_keep_on_api():
    # Owned $/M far above the API's per-token price.
    report = _build({"classification": 60}, {"classification": _type_eval(ev.NO_DIFF)},
                    _check(mean=900, low=800, high=1000), scale=10)
    row = _row(report, "classification")
    assert row["recommendation"] == rep.KEEP
    assert row["conservative_savings_usd_per_month"] <= 0
    assert "no saving" in row["reasons"][0]
    assert report["totals"]["conservative_savings_usd"] is None


def test_good_quality_without_check_is_needs_more_data_not_move():
    report = _build({"classification": 60}, {"classification": _type_eval(ev.NO_DIFF)}, None, scale=1000)
    row = _row(report, "classification")
    assert row["recommendation"] == rep.MORE_DATA
    assert "throttle check --url ... --model open-70b" in row["reasons"][0]


# ---------------------------------------------------------------------------
# Evidence gate: no number without a check record
# ---------------------------------------------------------------------------


def test_no_owned_or_savings_number_without_check(tmp_path):
    types = {"classification": _type_eval(ev.NO_DIFF), "code": _type_eval(ev.WORSE, net=(-0.5, -0.2))}
    report = _build({"classification": 60, "code": 60}, types, None, scale=1000)
    assert report["owned_cost"] is None
    for w in report["workloads"]:
        assert w["owned_usd_per_month"] is None
        assert w["conservative_savings_usd_per_month"] is None
        assert w["owned_note"].startswith("not measured: run `throttle check --url ... --model open-70b`")
        assert w["quality"] is not None          # quality still renders
    totals = report["totals"]
    assert totals["owned_usd_per_month"] is None
    assert totals["conservative_savings_usd"] is None and totals["point_savings_usd"] is None
    assert totals["savings_note"].startswith("not measured")
    _, html_path = rep.write_outputs(report, tmp_path)
    page = html_path.read_text()
    assert "not measured: run `throttle check --url ... --model open-70b`" in page
    assert "Conservative monthly savings" in page
    # The only $ figures are API spend: none of the owned/savings cells carry one.
    owned_cells = re.findall(r'<b class="cost-own[^"]*">(.*?)</b>', page)
    assert owned_cells and all(c == "not measured" for c in owned_cells)
    assert re.findall(r'<b class="cost-api">\$', page)
    assert "NOT MEASURED" in page and 'aria-label="no savings claim"' in page


def test_check_without_ci_is_not_measured():
    check = _check()
    check["result"]["total"]["ci_low"] = None
    report = _build({"classification": 60}, {"classification": _type_eval(ev.NO_DIFF)}, check, scale=1000)
    assert report["owned_cost"] is None
    assert "no 95% CI" in report["not_measured_hint"]
    assert _row(report, "classification")["owned_usd_per_month"] is None


def test_short_capture_window_projects_nothing():
    exchanges = _traffic({"classification": 40}, days=0.25)
    report = rep.build_report(exchanges, PRICES, candidate="cand", eval_doc=_eval_doc({"classification": _type_eval(ev.NO_DIFF)}),
                              check_record=_check(), gpu_hourly_rate=2.0, gpus=1)
    row = _row(report, "classification")
    assert row["api_usd_per_month"] is None and row["owned_usd_per_month"] is None
    assert row["recommendation"] == rep.MORE_DATA
    assert report["totals"]["conservative_savings_usd"] is None


# ---------------------------------------------------------------------------
# Cost math
# ---------------------------------------------------------------------------


def test_per_type_owned_cost_uses_candidate_tokens_and_check_ci():
    report = _build({"classification": 60}, {"classification": _type_eval(ev.NO_DIFF, cand_tokens=(700, 50), excluded=4)},
                    _check(mean=0.30, low=0.20, high=0.40), scale=1)
    row = _row(report, "classification")
    # 60 requests over 2 days -> 900 / month; candidate 750 tokens per request.
    assert row["requests_per_month"] == pytest.approx(900)
    assert row["candidate_tokens_per_request"]["total"] == pytest.approx(750)
    assert row["candidate_tokens_per_request"]["n"] == 36
    million = 900 * 750 / 1e6
    assert row["owned_usd_per_month"] == pytest.approx({"point": million * 0.30, "low": million * 0.20, "high": million * 0.40})
    # API: 1000 in x $2/M + 200 out x $8/M = $0.0036 per request.
    assert row["api_usd_per_month"] == pytest.approx(900 * 0.0036)
    assert row["conservative_savings_usd_per_month"] == pytest.approx(900 * 0.0036 - million * 0.40)


def test_floor_is_applied_once_across_move_types():
    types = {"classification": _type_eval(ev.NO_DIFF), "extraction": _type_eval(ev.NO_DIFF), "chat": _type_eval(ev.NO_DIFF)}
    report = _build({k: 60 for k in types}, types, _check(), rate=2.0, gpus=1, scale=1000)
    floor = 2.0 * 1 * 730
    assert report["deployment"]["floor_usd_per_month"] == floor
    totals = report["totals"]
    moving = [w for w in report["workloads"] if w["recommendation"] == rep.MOVE]
    assert len(moving) == 3
    token_high = sum(w["owned_usd_per_month"]["high"] for w in moving)
    assert token_high < floor
    assert totals["floor_applied"] is True
    assert totals["owned_usd_per_month"]["high"] == pytest.approx(floor)          # once, not 3x
    movable = sum(w["api_usd_per_month"] for w in moving)
    assert totals["movable_api_usd"] == pytest.approx(movable)
    assert totals["conservative_savings_usd"] == pytest.approx(movable - floor)
    assert "counted once" in totals["savings_note"]


def test_conservative_math_uses_ci_high_when_tokens_exceed_floor():
    types = {"classification": _type_eval(ev.NO_DIFF), "chat": _type_eval(ev.NO_DIFF)}
    report = _build({k: 60 for k in types}, types, _check(mean=0.30, low=0.25, high=0.50), rate=0.01, gpus=1, scale=1000)
    totals = report["totals"]
    moving = [w for w in report["workloads"] if w["recommendation"] == rep.MOVE]
    high = sum(w["owned_usd_per_month"]["high"] for w in moving)
    point = sum(w["owned_usd_per_month"]["point"] for w in moving)
    movable = sum(w["api_usd_per_month"] for w in moving)
    assert totals["floor_applied"] is False
    assert totals["conservative_savings_usd"] == pytest.approx(movable - high)
    assert totals["point_savings_usd"] == pytest.approx(movable - point)
    assert totals["conservative_savings_usd"] < totals["point_savings_usd"]
    assert totals["movable_share"] == pytest.approx(movable / totals["monthly_api_usd"])


def test_floor_bigger_than_movable_spend_means_no_claim_and_no_move():
    types = {"classification": _type_eval(ev.NO_DIFF)}
    report = _build({"classification": 60}, types, _check(), rate=50.0, gpus=8, scale=10)
    row = _row(report, "classification")
    assert row["recommendation"] == rep.KEEP
    assert "deployment floor" in row["reasons"][-1]
    totals = report["totals"]
    assert totals["conservative_savings_usd"] is None
    assert totals["movable_api_usd"] == 0
    assert "no savings claim" in totals["savings_note"]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def test_load_eval_latest_for_candidate(tmp_path):
    evals = tmp_path / "evals"
    evals.mkdir()
    for run, ts, cand in (("a", "2026-10-01T00:00:00Z", "cand"), ("b", "2026-10-02T00:00:00Z", "cand"),
                          ("c", "2026-10-03T00:00:00Z", "other")):
        (evals / f"{run}.json").write_text(json.dumps({"schema": ev.SCHEMA_ID, "run_id": run, "ts": ts, "candidate": cand, "types": {}}))
    (evals / "evalset-chat-abc.json").write_text(json.dumps({"type": "chat", "candidate": "cand"}))
    assert rep.load_eval(tmp_path, "cand")["run_id"] == "b"
    assert rep.load_eval(tmp_path, "cand", "a")["run_id"] == "a"
    with pytest.raises(ValueError):
        rep.load_eval(tmp_path, "cand", "c")
    with pytest.raises(ValueError):
        rep.load_eval(tmp_path, "cand", "missing")
    assert rep.load_eval(tmp_path / "nope", "cand") is None


def test_load_check_matches_candidate_model(tmp_path):
    check_mod.append_history(tmp_path, _check(cid="old", created="2026-10-01T00:00:00Z"))
    check_mod.append_history(tmp_path, _check(cid="new", created="2026-10-02T00:00:00Z"))
    check_mod.append_history(tmp_path, _check(cid="else", model="other-model", created="2026-10-03T00:00:00Z"))
    assert rep.load_check(tmp_path, ["open-70b"])["id"] == "new"
    assert rep.load_check(tmp_path, ["open-70b"], "old")["id"] == "old"
    assert rep.load_check(tmp_path, ["nothing"]) is None
    with pytest.raises(ValueError, match="not the candidate"):
        rep.load_check(tmp_path, ["open-70b"], "else")
    with pytest.raises(ValueError, match="no check"):
        rep.load_check(tmp_path, ["open-70b"], "missing")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def _full_report():
    types = {"classification": _type_eval(ev.NO_DIFF), "agent": _type_eval(ev.WORSE, net=(-0.8, -0.5), checks=FMT_BAD)}
    return _build({"classification": 60, "agent": 60}, types, _check(rate=0.02, gpus=1), rate=0.02, scale=1)


def test_html_is_self_contained(tmp_path):
    _, html_path = rep.write_outputs(_full_report(), tmp_path)
    page = html_path.read_text()
    assert page.startswith("<!doctype html>")
    assert not re.search(r"""(src|href)\s*=\s*["']?\s*(https?:)?//""", page, re.I)
    assert "<script" not in page.lower()
    assert "@import" not in page and "url(" not in page
    assert "<link" not in page.lower()
    for section in ("How we got these numbers", "What this does not prove", "Monthly API spend",
                    "Share movable now", "Conservative monthly savings"):
        assert section in page
    for label in ("MEASURED", "LIST PRICE (", "ASSUMED"):
        assert label in page
    assert "DEMO" not in page


def test_html_escapes_text(tmp_path):
    report = _full_report()
    report["candidate"] = "<script>alert(1)</script>"
    page = rep.render_html(report)
    assert "<script>alert" not in page and "&lt;script&gt;" in page


def test_json_schema(tmp_path):
    report = _full_report()
    json_path, _ = rep.write_outputs(report, tmp_path)
    data = json.loads(json_path.read_text())
    assert data["schema"] == rep.SCHEMA_ID
    assert set(data) >= {"schema", "generated_at", "demo", "candidate", "candidate_model", "eval_run_id", "check_id",
                         "window", "traffic_scale", "owned_cost", "not_measured_hint", "deployment", "workloads",
                         "totals", "labels", "does_not_prove"}
    assert data["demo"] is False
    assert set(data["owned_cost"]) == {"check_id", "model", "label", "usd_per_m_tokens", "gpu_hourly_rate_usd",
                                       "gpu_count", "rate_label", "endpoint", "created_at"}
    assert data["owned_cost"]["label"] == "MEASURED" and data["owned_cost"]["rate_label"] == "ASSUMED"
    assert set(data["deployment"]) == {"gpu_hourly_rate_usd", "gpus", "hours_per_month", "floor_usd_per_month", "label"}
    for w in data["workloads"]:
        assert set(w) == {"type", "requests_captured", "requests_per_month", "api_usd_per_month", "api_note",
                          "candidate_tokens_per_request", "owned_usd_per_month", "owned_note",
                          "conservative_savings_usd_per_month", "quality", "recommendation", "reasons"}
        assert w["recommendation"] in rep.RECOMMENDATIONS
        assert set(w["quality"]) == {"verdict", "n", "judged", "wins", "ties", "losses", "order_disagree",
                                     "win_rate", "tie_rate", "loss_rate", "order_disagree_rate", "net_ci",
                                     "win_rate_ci", "tie_rate_ci", "loss_rate_ci", "format_checks", "reasons"}
    assert set(data["totals"]) == {"monthly_api_usd", "movable_api_usd", "movable_share", "owned_usd_per_month",
                                   "floor_applied", "conservative_savings_usd", "point_savings_usd", "savings_note"}
    for item in data["labels"]:
        assert set(item) == {"what", "label", "detail"}


def test_gpu_price_mismatch_is_flagged():
    report = _build({"classification": 60}, {"classification": _type_eval(ev.NO_DIFF)}, _check(rate=4.0, gpus=2),
                    rate=3.0, gpus=1, scale=1000)
    assert any(item["label"] == "WARNING" for item in report["labels"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser():
    parser = argparse.ArgumentParser()
    rep.add_parser(parser.add_subparsers(dest="command"))
    return parser


def _populate(root: Path, check_dir: Path | None):
    for ex in _traffic({"classification": 60, "code": 60}):
        r.append_exchange(ex, root)
    doc = _eval_doc({"classification": _type_eval(ev.NO_DIFF), "code": _type_eval(ev.WORSE, net=(-0.5, -0.2))})
    (root / "evals").mkdir(parents=True)
    (root / "evals" / "run1.json").write_text(json.dumps(doc))
    if check_dir is not None:
        check_mod.append_history(check_dir, _check())


def test_cli_writes_both_files(tmp_path, monkeypatch, capsys):
    root, checks, out = tmp_path / "own", tmp_path / "checks", tmp_path / "out"
    _populate(root, checks)
    monkeypatch.setenv(check_mod.HISTORY_ENV, str(checks))
    args = _parser().parse_args(["report", "--candidate", "cand", "--gpu-hourly-rate", "2", "--gpus", "1",
                                 "--dir", str(root), "--out", str(out)])
    assert rep.handle(args) == 0
    text = capsys.readouterr().out
    assert "report.html" in text and "Share movable now" in text
    data = json.loads((out / "report.json").read_text())
    assert data["check_id"] == "c1" and data["eval_run_id"] == "run1"
    assert (out / "report.html").exists()


def test_cli_without_check_shows_hint(tmp_path, monkeypatch, capsys):
    root, out = tmp_path / "own", tmp_path / "out"
    _populate(root, None)
    monkeypatch.setenv(check_mod.HISTORY_ENV, str(tmp_path / "empty"))
    args = _parser().parse_args(["report", "--candidate", "cand", "--gpu-hourly-rate", "2", "--gpus", "1",
                                 "--dir", str(root), "--out", str(out), "--json"])
    assert rep.handle(args) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["owned_cost"] is None and data["totals"]["conservative_savings_usd"] is None


def test_cli_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(check_mod.HISTORY_ENV, str(tmp_path / "checks"))
    base = ["report", "--candidate", "cand", "--gpu-hourly-rate", "2", "--gpus", "1", "--out", str(tmp_path / "o")]
    assert rep.handle(_parser().parse_args(base + ["--dir", str(tmp_path / "none")])) == 1
    _populate(tmp_path / "own", None)
    assert rep.handle(_parser().parse_args(base + ["--dir", str(tmp_path / "own"), "--check", "nope"])) == 2
    assert rep.handle(_parser().parse_args(base + ["--dir", str(tmp_path / "own"), "--eval", "nope"])) == 2
    bad = ["report", "--candidate", "cand", "--gpu-hourly-rate", "0", "--gpus", "1", "--dir", str(tmp_path / "own")]
    assert rep.handle(_parser().parse_args(bad)) == 2


def test_real_cli_dispatches_report(tmp_path, monkeypatch, capsys):
    from throttle.cli import main

    root, checks, out = tmp_path / "own", tmp_path / "checks", tmp_path / "out"
    _populate(root, checks)
    monkeypatch.setenv(check_mod.HISTORY_ENV, str(checks))
    code = main(["report", "--candidate", "cand", "--gpu-hourly-rate", "2", "--gpus", "1",
                 "--dir", str(root), "--out", str(out)])
    assert code == 0
    assert (out / "report.json").exists() and (out / "report.html").exists()
    assert "Report: cand vs API" in capsys.readouterr().out


def test_panel_says_not_projected_when_the_window_is_too_short():
    from throttle.own.report import _owned_gap
    assert _owned_gap({"owned_note": "not projected: capture window under 1 day"}) == "not projected"
    assert _owned_gap({"owned_note": "not measured: run throttle check"}) == "not measured"
    assert _owned_gap({}) == "not measured"


def test_meter_page_shows_replay_source_and_caveats():
    source = {"kind": "public_replay", "headline_label": "public dataset replay, local models, GPU price assumed",
              "dataset": "allenai/WildChat-1M", "license": "ODC-BY 1.0", "requests": 500,
              "api_stand_in": "llama3.2:3b", "priced_as": "gpt-4.1-mini",
              "caveats": ["Small local models stand in for frontier API models."]}
    types = {"classification": _type_eval(ev.NO_DIFF)}
    report = rep.build_report(_traffic({"classification": 60}), PRICES, candidate="cand", eval_doc=_eval_doc(types),
                              check_record=_check(rate=0.02, gpus=1), gpu_hourly_rate=0.02, gpus=1,
                              traffic_scale=10, source=source)
    assert report["source"]["dataset"] == "allenai/WildChat-1M"
    assert report["does_not_prove"][0] == "Small local models stand in for frontier API models."
    assert any(i["what"] == "Traffic volume" and i["label"] == "ASSUMED" for i in report["labels"])
    assert any(i["label"] == "PUBLIC DATASET" for i in report["labels"])
    page = rep.render_html(report)
    assert "public dataset replay, local models, GPU price assumed" in page
    assert "allenai/WildChat-1M" in page and "Small local models stand in" in page
    assert '<span class="tag t-replay">PUBLIC DATASET REPLAY</span>' in page
    assert "<title>Replay Throttle report" in page
    assert "<script" not in page.lower() and "prefers-reduced-motion" in page


def test_meter_odometer_and_receipt(tmp_path):
    report = _full_report()
    page = rep.render_html(report)
    assert 'class="odometer"' in page and 'class="receipt"' in page and "<details" in page
    assert page.count('<li class="line ') == len(report["workloads"])
    for w in report["workloads"]:
        q = w["quality"]
        assert f"As good or better in {q['wins'] + q['ties']} of {q['judged']} tests" in page
