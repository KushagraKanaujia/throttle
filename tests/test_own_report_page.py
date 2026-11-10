"""The report page's words: plain, consistent with the numbers, and 'your model' throughout."""

from __future__ import annotations

import html
import re

import pytest

from throttle.own import demo
from throttle.own import report as rep
from throttle.own import report_page as rp


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    base = tmp_path_factory.mktemp("page")
    result = demo.run_demo(base / "out", seed=7, requests=600, work_dir=base / "work")
    return result, html.unescape(result.html_path.read_text())


def text_of(markup: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", markup).split())


def test_page_says_your_model_and_drops_the_finetune_promise(page):
    _, markup = page
    words = text_of(markup)
    assert "the open model" not in words.lower()
    assert "usually closes" not in words
    assert "your model loses on quality" in words


def test_hero_does_not_call_finetune_jobs_still_testing(page):
    result, markup = page
    jobs = result.report["workloads"]
    finetune = sum(1 for w in jobs if w["recommendation"] == rep.NEEDS_FINETUNE)
    keep = sum(1 for w in jobs if w["recommendation"] == rep.KEEP)
    assert finetune and keep
    words = text_of(markup)
    assert "Still testing" not in markup and "still testing" not in markup
    tile = re.search(r'<div class="fact f-test"><dt>Not ready</dt><dd><b>(\d+)</b><span>([^<]*)</span>', markup)
    assert tile, "the Not ready tile is missing"
    more_data = sum(1 for w in jobs if w["recommendation"] == rep.MORE_DATA)
    # Not ready counts only jobs that could still move; KEEP ON API is final and has its own tile.
    assert int(tile.group(1)) == finetune + more_data
    assert tile.group(2).startswith(f"{finetune} need")
    assert "stay" not in tile.group(2)
    stay = re.search(r'<div class="fact f-api"><dt>Stay on the API</dt><dd><b>(\d+)</b>', markup)
    assert stay and int(stay.group(1)) == keep
    assert "Not ready" in words


def test_lede_skips_zero_not_ready_when_every_job_stays(page):
    import copy

    result, _ = page
    report = copy.deepcopy(result.report)
    for w in report["workloads"]:
        w["recommendation"] = rep.KEEP
    report["totals"]["movable_api_usd"] = 0
    words = text_of(html.unescape(rp._hero(report)))
    assert "0 jobs need" not in words
    assert "need a fine-tune or more data first" not in words
    n = len(report["workloads"])
    assert f"{n} jobs stay on the API." in words


def test_owned_column_says_high_end(page):
    _, markup = page
    assert '<th scope="col" class="num">Your model $/mo (high end of 95% CI)</th>' in markup


def test_check_names_are_plain_words(page):
    _, markup = page
    words = text_of(markup)
    assert "tool_calls_valid" not in words and "json_valid" not in words
    assert "valid tool calls" in words
    assert rp.check_words("tool_calls_valid") == "valid tool calls"
    assert rp.check_words("json_valid") == "valid JSON"
    assert rp.check_words("some_new_check") == "some new check"


def test_plain_keeps_ordinary_words():
    assert rp.plain("the answer was truncated") == "the answer was truncated"
    assert rp.plain("the open model loses on quality (x); this kind of gap is what fine-tuning on your own "
                    "traffic usually closes") == "your model loses on quality (x)"


def test_worse_rows_lead_with_worse_on():
    q = {"verdict": "WORSE", "judged": 40, "wins": 2, "ties": 10, "losses": 25, "order_disagree": 3}
    assert rp.proof_words(q) == "Worse on 25 of 40 tests; as good or better in 12; judge unsure on 3."
    ok = {"verdict": "NO MATERIAL DIFFERENCE", "judged": 40, "wins": 5, "ties": 35, "losses": 0, "order_disagree": 0}
    assert rp.proof_words(ok) == "As good or better in 40 of 40 tests."


def _quality_with_unsure() -> dict:
    judged, wins, ties, unsure, losses = 40, 10, 10, 8, 12
    from throttle import statistics as stats

    ci = lambda k: [stats.wilson_interval_95(k, judged)["low"], stats.wilson_interval_95(k, judged)["high"]]  # noqa: E731
    return {"verdict": "INCONCLUSIVE", "n": judged, "judged": judged, "wins": wins, "ties": ties, "losses": losses,
            "order_disagree": unsure, "win_rate": wins / judged, "tie_rate": ties / judged,
            "loss_rate": losses / judged, "order_disagree_rate": unsure / judged, "net_ci": [-0.3, 0.2],
            "win_rate_ci": ci(wins), "tie_rate_ci": ci(ties), "loss_rate_ci": ci(losses),
            "format_checks": {}, "reasons": []}


def test_same_bar_is_ties_only_and_judge_unsure_has_its_own_bar():
    q = _quality_with_unsure()
    report = {"candidate": "c", "generated_at": "2026-11-09T00:00:00Z", "demo": True,
              "workloads": [{"type": "chat", "recommendation": rep.MORE_DATA, "api_usd_per_month": 10.0,
                             "owned_usd_per_month": None, "quality": q, "reasons": ["quality verdict is INCONCLUSIVE"]}],
              "totals": {"monthly_api_usd": 10.0, "movable_api_usd": 0.0, "movable_share": 0.0,
                         "owned_usd_per_month": None, "conservative_savings_usd": None},
              "labels": [], "does_not_prove": [], "owned_cost": None}
    markup = rp._proof(report)

    def bar(label: str) -> tuple[float, float, float]:
        m = re.search(rf'<span class="rate-name">{label}</span>.*?<span class="rate-num">(\d+)%<small>95% CI '
                      r"(\d+)% to (\d+)%</small>", markup, re.S)
        assert m, label
        return tuple(int(x) / 100 for x in m.groups())  # type: ignore[return-value]

    same, unsure = bar("Same"), bar("Judge unsure")
    assert same[0] == 0.25 and unsure[0] == 0.20
    for point, low, high in (same, unsure):
        assert low <= point <= high
    shares = [bar(label)[0] for label in ("Your model better", "Same", "Judge unsure", "Your model worse")]
    assert sum(shares) == pytest.approx(1.0)
    assert "shown as judge unsure, not as same" in markup
    assert rp.unsure_ci(q)[0] <= 0.2 <= rp.unsure_ci(q)[1]
    assert rp.unsure_ci({**q, "order_disagree_rate_ci": [0.1, 0.3]}) == [0.1, 0.3]
    assert rp.unsure_ci({"judged": 0}) is None
