"""AI jobs: matching traffic, the registry following the code, approvals,
job-level evals and lessons (throttle.own.jobs). Offline."""

from __future__ import annotations

import json

import httpx

from throttle.cli import build_parser
from throttle.own import evaluate as ev
from throttle.own import jobs
from throttle.own import judge as judge_mod
from throttle.own import records as r
from throttle.own import route as rt

from own_job_helpers import SUMMARY, TRIAGE, exchange, judge_prefers, py_call, shadow, sites_for, write_job_eval

BASE = {"app/support.py": py_call("summarize", SUMMARY), "app/triage.py": py_call("triage", TRIAGE, "{ticket}")}


def sites(files=None):
    return sites_for(files or BASE)


def test_matcher_uses_header_then_most_specific_template():
    m = jobs.JobMatcher(sites())
    ex = exchange(SUMMARY, "Ticket: printer on fire")
    assert m.match(ex, r.normalize(ex)) == "app/support.py::summarize"
    tri = exchange("  " + TRIAGE.replace(" ", "\n  "), "refund me")     # whitespace does not matter
    assert m.match(tri, r.normalize(tri)) == "app/triage.py::triage"
    other = exchange("You write poems.", "Ticket: x")
    assert m.match(other, r.normalize(other)) is None
    # a header naming a known job wins, but only while the request fits its prompt
    tagged = exchange(SUMMARY, "Ticket: y", job="app/support.py::summarize")
    assert m.match(tagged, r.normalize(tagged)) == "app/support.py::summarize"
    stale = exchange("An older summary prompt that was edited since.", "Ticket: y", job="app/support.py::summarize")
    assert m.match(stale, r.normalize(stale)) is None
    # metadata-only captures can still be matched by header
    meta = exchange(SUMMARY, "x", job="app/triage.py::triage")
    meta.redaction, meta.request = r.REDACTION_METADATA, None
    assert m.match(meta, None) == "app/triage.py::triage"


def test_matcher_prefers_longer_template():
    generic = py_call("generic", "You are a helpful assistant for {team}.", "{ticket}")
    specific = py_call("specific", "You are a helpful assistant for billing. Answer refund questions only.", "{ticket}")
    m = jobs.JobMatcher(sites({"a.py": generic.replace("'You are", "f'You are"), "b.py": specific}))
    ex = exchange("You are a helpful assistant for billing. Answer refund questions only.", "hi")
    assert m.match(ex, r.normalize(ex)) == "b.py::specific"


def test_changes_in_plain_words():
    before = sites()[0]
    after = sites({"app/support.py": py_call("summarize", SUMMARY + " Use bullets.", model="gpt-4.1",
                                             extra=", response_format={'type': 'json_object'}")})[0]
    assert jobs.changes(before, after) == ["prompt edited", "model changed from gpt-4o-mini to gpt-4.1", "JSON output turned on"]
    assert jobs.changes(before, before) == []


def _moved(root, job="app/support.py::summarize", fingerprint=None, run_id="jobeval-1"):
    registry = jobs.load_registry(root)
    entry = registry[job]
    entry.status = jobs.MOVED
    entry.route = {"candidate_label": "qwen", "url": "http://gpu:8000/v1", "model": "qwen-7b",
                   "approved_eval_run_id": run_id, "eval_verdict": ev.NO_DIFF, "mode": "live",
                   "prompt_fingerprint": fingerprint or entry.call_site.prompt_fingerprint}
    jobs.save_registry(root, registry)


def test_sync_adds_suspends_retires_and_follows_moves(tmp_path):
    events = jobs.sync(tmp_path, sites(), ref="aaa")
    assert {e.event for e in events} == {"added"}
    assert all(e.status == jobs.TESTING for e in jobs.load_registry(tmp_path).values())
    _moved(tmp_path)

    # the prompt of the moved job changes and triage is deleted
    edited = {"app/support.py": py_call("summarize", SUMMARY + " Mention the plan.")}
    registry, events = jobs.plan_sync(jobs.load_registry(tmp_path), sites(edited), ref="bbb")
    kinds = {e.job: e.event for e in events}
    assert kinds == {"app/support.py::summarize": "suspended", "app/triage.py::triage": "retired"}
    summary = registry["app/support.py::summarize"]
    assert summary.status == jobs.TESTING and summary.route is None
    assert summary.suspended["route"]["candidate_label"] == "qwen" and summary.suspended["reason"] == "prompt edited"
    assert registry["app/triage.py::triage"].status == jobs.RETIRED
    assert [h["event"] for h in summary.history] == ["added", "suspended"]
    # plan_sync is pure
    assert jobs.load_registry(tmp_path)["app/support.py::summarize"].status == jobs.MOVED

    # the same prompt moved to a new file keeps its state under the new id
    moved = {"app/new_place.py": py_call("summarize", SUMMARY), "app/triage.py": BASE["app/triage.py"]}
    registry, events = jobs.plan_sync(jobs.load_registry(tmp_path), sites(moved))
    assert "app/support.py::summarize" not in registry
    assert registry["app/new_place.py::summarize"].status == jobs.MOVED
    assert [e.event for e in events] == ["moved"]

    # a retired job that comes back starts in testing
    jobs.save_registry(tmp_path, jobs.plan_sync(jobs.load_registry(tmp_path), sites(edited))[0])
    registry, events = jobs.plan_sync(jobs.load_registry(tmp_path), sites())
    assert registry["app/triage.py::triage"].status == jobs.TESTING
    assert {e.job: e.event for e in events}["app/triage.py::triage"] == "back"


def test_approve_job_needs_a_passing_eval_of_the_current_prompt(tmp_path):
    jobs.sync(tmp_path, sites())
    job = "app/support.py::summarize"
    fp = jobs.load_registry(tmp_path)[job].call_site.prompt_fingerprint
    entry, reason = jobs.approve_job(tmp_path, job, "qwen", "http://gpu/v1", "qwen-7b")
    assert entry is None and "no job eval" in reason
    write_job_eval(tmp_path, job, ev.NO_DIFF, "an-older-prompt")
    entry, reason = jobs.approve_job(tmp_path, job, "qwen", "http://gpu/v1", "qwen-7b")
    assert entry is None and "older prompt" in reason
    write_job_eval(tmp_path, job, ev.WORSE, fp, ts="2026-10-27T11:00:00.000Z")
    entry, reason = jobs.approve_job(tmp_path, job, "qwen", "http://gpu/v1", "qwen-7b")
    assert entry is None and "WORSE" in reason
    run = write_job_eval(tmp_path, job, ev.BETTER, fp, ts="2026-10-27T12:00:00.000Z")
    entry, reason = jobs.approve_job(tmp_path, job, "qwen", "http://gpu/v1", "other-model")
    assert entry is None and "not other-model" in reason
    entry, reason = jobs.approve_job(tmp_path, job, "qwen", "http://gpu/v1", "qwen-7b")
    assert entry.status == jobs.MOVED and entry.route["approved_eval_run_id"] == run
    assert entry.route["prompt_fingerprint"] == fp
    # type-level eval docs are untouched: route.latest_eval does not see job evals
    assert rt.latest_eval(tmp_path, "qwen") is None
    # pinned jobs cannot be moved, undo returns to testing
    jobs.set_status(tmp_path, job, jobs.PINNED, "pinned")
    entry, reason = jobs.approve_job(tmp_path, job, "qwen", "http://gpu/v1", "qwen-7b")
    assert entry is None and "pinned" in reason
    assert jobs.set_status(tmp_path, job, jobs.TESTING, "undone").status == jobs.TESTING


def test_cli_scan_sync_list_pin_log(tmp_path, capsys):
    repo = tmp_path / "repo"
    for path, text in BASE.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    data = tmp_path / "data"
    parser = build_parser()

    def run(*argv):
        args = parser.parse_args(["jobs", *argv, "--dir", str(data)])
        return jobs.handle(args)

    assert run("scan", "--repo", str(repo)) == 0
    assert "2 AI calls found" in capsys.readouterr().out
    assert run("sync", "--repo", str(repo), "--dry-run") == 0
    assert "Would change" in capsys.readouterr().out and not jobs.registry_path(data).exists()
    assert run("sync", "--repo", str(repo)) == 0
    assert run("pin", "app/triage.py::triage") == 0
    capsys.readouterr()
    assert run("list") == 0
    out = capsys.readouterr().out
    assert "pinned to the API" in out and "on the API, being tested" in out
    assert run("log", "app/triage.py::triage") == 0
    assert "pinned to the API by the team" in capsys.readouterr().out
    assert run("approve", "app/triage.py::triage", "--candidate", "q", "--url", "u", "--model", "m") == 1
    assert "pinned" in capsys.readouterr().err
    assert run("lessons") == 0


def test_fallback_and_loss_lessons(tmp_path):
    ex = exchange(SUMMARY, "Ticket: z", job="app/support.py::summarize", fallback=True)
    ex.route_reason = "invalid_json"
    path = jobs.lesson_from_fallback(tmp_path, ex)
    lesson = next(r.read_jsonl(path))
    assert lesson["source"] == "fallback" and lesson["reason"] == "invalid_json"
    assert lesson["preferred"] == "API ANSWER" and lesson["messages"][0]["role"] == "system"
    assert jobs.lesson_from_fallback(tmp_path, exchange(SUMMARY, "x")) is None
    assert jobs.lesson_counts(tmp_path) == {"app/support.py::summarize": {"fallback": 1}}


def test_job_eval_groups_by_job_and_current_prompt(tmp_path):
    exchanges = [exchange(SUMMARY, f"Ticket: {i}", ts=f"2026-10-01T00:00:{i:02d}.000Z") for i in range(4)]
    exchanges += [exchange(TRIAGE, f"ticket {i}", output="bug") for i in range(3)]
    exchanges += [exchange("An old summary prompt.", "Ticket: old")]      # fits no current prompt
    for ex in exchanges:
        r.append_exchange(ex, tmp_path)
        text = "CANDIDATE ANSWER" if ex.request["messages"][0]["content"] == SUMMARY else "bug"
        r.append_shadow(shadow(ex.id, text), tmp_path)
    config = judge_mod.JudgeConfig(provider="openai", model="judge", url="http://judge/v1/chat/completions")
    doc = jobs.run_job_eval(tmp_path, "qwen", sites(), judge_config=config, min_n=2,
                            transport=httpx.MockTransport(judge_prefers("API ANSWER")))
    assert set(doc["jobs"]) == {"app/support.py::summarize", "app/triage.py::triage"}
    summary = doc["jobs"]["app/support.py::summarize"]
    assert summary["n"] == 4 and summary["losses"] == 4 and summary["verdict"] == ev.WORSE
    assert summary["prompt_fingerprint"] == sites()[0].prompt_fingerprint
    triage = doc["jobs"]["app/triage.py::triage"]
    assert triage["ties"] == 3 and triage["workload_type"] == "classification"
    assert (jobs.job_eval_dir(tmp_path) / f"{doc['run_id']}.json").exists()
    assert not list((tmp_path / "evals").glob("*.json")) if (tmp_path / "evals").exists() else True
    # every judged loss is saved as a lesson with both answers
    counts = jobs.lesson_counts(tmp_path)
    assert counts["app/support.py::summarize"]["judged_loss"] == 4
    found = jobs.latest_job_result(tmp_path, "app/support.py::summarize", "qwen", summary["prompt_fingerprint"])
    assert found and found[1]["verdict"] == ev.WORSE
