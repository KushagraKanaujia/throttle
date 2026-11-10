"""Tests for ``throttle demo``: the full loop on synthetic data, offline."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter

import pytest

from throttle.own import demo
from throttle.own import records as r
from throttle.own import report as rep
from throttle.own import workloads as wl


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    base = tmp_path_factory.mktemp("demo")
    started = time.perf_counter()
    result = demo.run_demo(base / "out", seed=7, requests=600, work_dir=base / "work")
    return result, time.perf_counter() - started, base


def _numbers(report: dict) -> dict:
    """Everything that should not change between runs of one seed."""

    rows = {w["type"]: (w["recommendation"], w["requests_captured"], w["api_usd_per_month"],
                        w["owned_usd_per_month"], w["quality"]) for w in report["workloads"]}
    return {"rows": rows, "totals": report["totals"]}


def test_demo_runs_end_to_end_quickly(run):
    result, seconds, _ = run
    assert seconds < 60
    assert result.captured == 600
    assert result.judge_unmatched == 0
    assert result.html_path.exists() and result.json_path.exists()


def test_synthetic_traffic_hits_every_type_through_the_real_classifier(run):
    _, _, base = run
    exchanges = list(r.iter_exchanges(base / "work"))
    kinds = Counter(wl.classify(e) for e in exchanges)
    assert set(kinds) == set(r.WORKLOAD_TYPES)
    assert min(kinds.values()) >= 60
    providers = Counter(e.provider for e in exchanges)
    assert providers[r.PROVIDER_OPENAI] > 100 and providers[r.PROVIDER_ANTHROPIC] > 100
    assert {e.model for e in exchanges} == {demo.OPENAI_MODEL, demo.ANTHROPIC_MODEL}
    assert all(e.usage.complete and e.redaction == r.REDACTION_MASK for e in exchanges)
    assert any(e.provider == r.PROVIDER_ANTHROPIC and r.normalize(e).output_tool_calls for e in exchanges)


def test_demo_produces_the_designed_mix(run):
    result, _, _ = run
    got = {w["type"]: w["recommendation"] for w in result.report["workloads"]}
    assert got == {
        "classification": rep.MOVE, "extraction": rep.MOVE, "chat": rep.MOVE,
        "summarization": rep.NEEDS_FINETUNE, "code": rep.NEEDS_FINETUNE, "agent": rep.KEEP,
    }
    agent = next(w for w in result.report["workloads"] if w["type"] == "agent")
    assert not agent["quality"]["format_checks"]["tool_calls_valid"]["overlap"]
    totals = result.report["totals"]
    assert totals["conservative_savings_usd"] > 0
    assert 0 < totals["movable_share"] < 1
    assert result.route_lines[0].startswith("DEMO route approved: extraction")
    assert result.route_lines[1].startswith("DEMO route refused for agent")


def test_demo_is_deterministic_for_a_seed(run, tmp_path):
    result, _, _ = run
    again = demo.run_demo(tmp_path / "out", seed=7, requests=600)
    assert _numbers(again.report) == _numbers(result.report)
    other = demo.run_demo(tmp_path / "out2", seed=8, requests=600)
    assert _numbers(other.report) != _numbers(result.report)


def test_every_number_is_labelled_demo(run):
    result, _, _ = run
    report = json.loads(result.json_path.read_text())
    assert report["demo"] is True
    assert report["owned_cost"]["label"] == "DEMO" and report["owned_cost"]["rate_label"] == "DEMO"
    assert report["deployment"]["label"] == "DEMO"
    assert all("DEMO" in item["label"] for item in report["labels"])
    assert "synthetic" in report["does_not_prove"][0]
    page = result.html_path.read_text()
    assert rep.DEMO_BANNER in page
    assert "<title>DEMO " in page
    # The savings figure and every section heading carry the DEMO tag.
    assert re.search(r'Saved per month <span class="tag t-demo">DEMO</span>', page)
    for heading in ("Job by job", "The proof, one click deeper", "How we got these numbers"):
        assert re.search(rf'<h2>{heading} <span class="tag t-demo">DEMO</span></h2>', page)
    assert '<span class="stamp st-demo">DEMO</span>' in page
    assert not re.search(r"""(src|href)\s*=\s*["']?\s*(https?:)?//""", page, re.I)


def test_site_json_matches_landing_schema(run):
    result, _, _ = run
    site = result.site
    assert set(site) == {"placeholder", "generated_at", "label", "workloads", "totals"}
    assert site["placeholder"] is False and site["label"] == "DEMO" and site["generated_at"]
    assert set(site["totals"]) == {"monthly_api_usd", "movable_usd", "projected_savings_usd"}
    assert all(isinstance(v, float) for v in site["totals"].values())
    landing_verdicts = {"MOVE", "KEEP ON API", "NEEDS FINETUNE", "INSUFFICIENT DATA"}
    for w in site["workloads"]:
        assert set(w) == {"type", "monthly_api_usd", "projected_owned_usd", "verdict", "quality"}
        assert w["verdict"] in landing_verdicts
        assert isinstance(w["monthly_api_usd"], float) and isinstance(w["projected_owned_usd"], float)
        q = w["quality"]
        assert set(q) == {"win", "tie", "judge_unsure", "loss", "net_ci"}
        assert all(0 <= q[k] <= 1 for k in ("win", "tie", "judge_unsure", "loss"))
        assert q["win"] + q["tie"] + q["judge_unsure"] + q["loss"] == pytest.approx(1.0, abs=1e-3)
        assert len(q["net_ci"]) == 2 and q["net_ci"][0] <= q["net_ci"][1]


def test_site_json_uses_null_without_numbers():
    report = {"generated_at": "x", "workloads": [{"type": "chat", "api_usd_per_month": None, "owned_usd_per_month": None,
                                                  "recommendation": rep.MORE_DATA, "quality": None}],
              "totals": {"monthly_api_usd": None, "movable_api_usd": None, "conservative_savings_usd": None}}
    site = demo.site_json(report)
    row = site["workloads"][0]
    assert row["monthly_api_usd"] is None and row["projected_owned_usd"] is None
    assert row["quality"] == {"win": None, "tie": None, "judge_unsure": None, "loss": None, "net_ci": [None, None]}
    assert site["totals"] == {"monthly_api_usd": None, "movable_usd": None, "projected_savings_usd": None}


def test_demo_cli(tmp_path, capsys):
    parser = argparse.ArgumentParser()
    demo.add_parser(parser.add_subparsers(dest="command"))
    site = tmp_path / "site" / "demo-report.json"
    args = parser.parse_args(["demo", "--out", str(tmp_path / "out"), "--site-json", str(site), "--requests", "300"])
    assert demo.handle(args) == 0
    text = capsys.readouterr().out
    assert "DEMO" in text and "report.html" in text and "Run this on your own traffic: throttle capture" in text
    assert json.loads(site.read_text())["label"] == "DEMO"
    assert demo.handle(parser.parse_args(["demo", "--requests", "10"])) == 2


def test_real_cli_dispatches_demo(tmp_path, capsys):
    from throttle.cli import main

    assert main(["demo", "--out", str(tmp_path / "out"), "--seed", "3", "--requests", "300"]) == 0
    assert (tmp_path / "out" / "report.html").exists()
    assert "Report: llama-3.3-70b@vllm vs API (DEMO)" in capsys.readouterr().out


def test_site_json_tie_is_ties_only_and_unsure_is_separate(run):
    result, _, _ = run
    by_type = {w["type"]: w for w in result.report["workloads"]}
    for row in result.site["workloads"]:
        q = by_type[row["type"]]["quality"]
        assert row["quality"]["tie"] == pytest.approx(round(q["ties"] / q["judged"], 4))
        assert row["quality"]["judge_unsure"] == pytest.approx(round(q["order_disagree"] / q["judged"], 4))


def test_site_json_with_judge_unsure_items():
    quality = {"win_rate": 0.25, "tie_rate": 0.25, "order_disagree_rate": 0.2, "loss_rate": 0.3,
               "net_ci": [-0.2, 0.1]}
    report = {"generated_at": "x", "workloads": [{
        "type": "chat", "api_usd_per_month": 100.0, "owned_usd_per_month": {"point": 40.0, "low": 30.0, "high": 55.0},
        "recommendation": rep.MORE_DATA, "quality": quality}],
        "totals": {"monthly_api_usd": 100.0, "movable_api_usd": 0.0, "conservative_savings_usd": None}}
    row = demo.site_json(report)["workloads"][0]
    assert row["quality"]["tie"] == 0.25 and row["quality"]["judge_unsure"] == 0.2   # not folded together
    assert sum(row["quality"][k] for k in ("win", "tie", "judge_unsure", "loss")) == pytest.approx(1.0)
    assert row["projected_owned_usd"] == 55.0   # the high end, as on the report page


def test_report_page_and_site_json_show_the_same_owned_cost(run):
    """Cross-lane contract: owned $/mo is the high end of the 95% CI everywhere."""
    from throttle.own import report_page

    result, _, _ = run
    page = result.html_path.read_text()
    assert report_page.OWNED_COLUMN in page
    by_type = {w["type"]: w for w in result.report["workloads"]}
    for row in result.site["workloads"]:
        owned = by_type[row["type"]]["owned_usd_per_month"]
        assert row["projected_owned_usd"] == pytest.approx(round(owned["high"], 2))
        # The page's job row for this type prints the same high-end figure.
        name = report_page.job_name(row["type"])
        job_row = re.search(rf'<span class="job-name">{re.escape(name)}</span>.*?</tr>', page, re.S).group(0)
        assert report_page.money(owned["high"]) in job_row, (row["type"], report_page.money(owned["high"]))
