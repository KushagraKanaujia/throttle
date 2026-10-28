"""throttle pr-check: diffing AI jobs between refs and writing the comment. Offline."""

from __future__ import annotations

import json

import httpx

from throttle.cli import build_parser
from throttle.own import evaluate as ev
from throttle.own import firstlook as fl
from throttle.own import jobs
from throttle.own import judge as judge_mod
from throttle.own import prcheck as pc
from throttle.own import records as r
from throttle.own import scan

from own_job_helpers import SUMMARY, TRIAGE, commit_files, exchange, judge_prefers, py_call, write_job_eval

SUMMARY2 = SUMMARY + " Then list what the customer already tried."
REPLY = "You draft a short, friendly reply to a customer who reported a problem."
BASE = {
    "app/support.py": py_call("summarize", SUMMARY),
    "app/triage.py": py_call("triage", TRIAGE, "{ticket}"),
    "app/legacy.py": py_call("old_digest", "Write a weekly digest of all tickets for the team lead."),
}
HEAD = {
    "app/support.py": py_call("summarize", SUMMARY2),
    "app/legacy.py": None,
    "app/reply.py": py_call("draft_reply", REPLY, "Customer wrote: {ticket}"),
}


def setup(tmp_path):
    repo = tmp_path / "repo"
    base = commit_files(repo, dict(BASE), "base")
    head = commit_files(repo, dict(HEAD), "head")
    data = tmp_path / "data"
    days = ["2026-10-01", "2026-10-02", "2026-10-03"]
    for i in range(30):
        day = days[i % 3]
        r.append_exchange(exchange(SUMMARY, f"Ticket: {i}", ts=f"{day}T00:00:{i:02d}.000Z"), data)
        r.append_exchange(exchange(TRIAGE, f"ticket {i}", output="bug", ts=f"{day}T01:00:{i:02d}.000Z",
                                   job="app/triage.py::triage"), data)
    jobs.sync(data, scan.scan_ref(repo, base))
    registry = jobs.load_registry(data)
    entry = registry["app/support.py::summarize"]
    entry.status = jobs.MOVED
    entry.route = {"candidate_label": "qwen", "model": "qwen-7b", "url": "u", "prompt_fingerprint": entry.call_site.prompt_fingerprint,
                   "site_key": entry.call_site.approval_key}
    jobs.save_registry(data, registry)
    return repo, base, head, data


def test_diff_sites_kinds():
    base = scan.scan_files({k: v for k, v in BASE.items()})
    head_files = {**BASE, "app/support.py": HEAD["app/support.py"], "app/reply.py": HEAD["app/reply.py"]}
    head_files.pop("app/legacy.py")
    head_files["app/digest.py"] = BASE["app/legacy.py"]          # same prompt, new file
    changes, unchanged = pc.diff_sites(base, scan.scan_files(head_files))
    assert [(c.kind, c.job) for c in changes] == [
        ("added", "app/reply.py::draft_reply"),
        ("changed", "app/support.py::summarize"),
        ("moved", "app/digest.py::old_digest"),
    ]
    assert changes[1].what == ["prompt edited"] and unchanged == 1


def test_run_check_actions_costs_and_markdown(tmp_path):
    repo, base, head, data = setup(tmp_path)
    doc = pc.run_check(data, scan.scan_ref(repo, base), scan.scan_ref(repo, head), base_ref=base, head_ref=head)
    rows = {row["job"]: row for row in doc["jobs"]}
    assert set(rows) == {"app/reply.py::draft_reply", "app/support.py::summarize", "app/legacy.py::old_digest"}
    assert rows["app/reply.py::draft_reply"]["action"] == "test_more"
    changed = rows["app/support.py::summarize"]
    assert changed["action"] == "test_again" and "never moved silently" in changed["action_text"]
    # its traffic is on the old prompt: priced, but labelled as the old version
    assert changed["traffic_basis"] == "old version" and changed["cost"]["requests"] == 30
    assert changed["cost"]["monthly_usd"] is not None
    assert rows["app/legacy.py::old_digest"]["action"] == "retire"
    assert doc["unchanged_jobs"] == 1 and doc["capture"]["monthly_projection"]
    # money first: old-version traffic is not added to the monthly headline
    assert doc["totals"]["monthly_api_usd"] is None
    md = pc.render_markdown(doc)
    assert md.startswith(pc.MARKER)
    assert "### Throttle: 3 AI calls changed in this pull request" in md
    assert "| `app/support.py::summarize` | prompt edited |" in md and "(old version)" in md
    assert "**Back to testing**" in md and "**Route retired**" in md and "**Test it more**" in md
    assert "\n\n1 other AI job in this code is unchanged." in md   # not glued to the table
    assert "\u2014" not in md and "\u2013" not in md      # no dashes that read like AI copy


def test_passing_job_eval_and_owned_cost_lead_to_move(tmp_path):
    repo, base, _, data = setup(tmp_path)
    head = commit_files(repo, {"app/triage.py": py_call("triage", TRIAGE, "{ticket}", model="gpt-4.1-mini")}, "model")
    head_sites = scan.scan_ref(repo, head)
    new_site = [s for s in head_sites if s.job == "app/triage.py::triage"][0]
    write_job_eval(data, "app/triage.py::triage", ev.NO_DIFF, new_site)
    doc = pc.run_check(data, scan.scan_ref(repo, base), head_sites, base_ref=base, head_ref=head,
                       options=pc.Options(candidate="qwen"))
    row = [x for x in doc["jobs"] if x["job"] == "app/triage.py::triage"][0]
    assert row["what"] == ["model changed from gpt-4o-mini to gpt-4.1-mini"]
    assert row["action"] == "measure" and row["quality"]["verdict"] == ev.NO_DIFF
    owned = {"usd_per_m_tokens": {"mean": 0.1, "low": 0.05, "high": 0.2}}
    usage = {"calls": 40, "input_tokens": 40_000, "output_tokens": 4_000}
    action, words = pc.decide(pc.JobChange("changed", None, head_sites[0]), None, {"verdict": ev.BETTER}, None,
                              pc.owned_per_call(usage, owned), 0.01)
    assert action == "move" and "throttle jobs approve" in words
    action, _ = pc.decide(pc.JobChange("changed", None, head_sites[0]), None, {"verdict": ev.BETTER}, None, 0.02, 0.01)
    assert action == "keep"
    action, words = pc.decide(pc.JobChange("added", None, head_sites[0]), None,
                              {"verdict": ev.WORSE, "format_broken": ["json_valid"]}, None, None, None)
    assert action == "keep" and "json_valid" in words
    action, _ = pc.decide(pc.JobChange("added", None, head_sites[0]), None, {"verdict": ev.WORSE, "format_broken": []},
                          None, None, None)
    assert action == "finetune"
    action, words = pc.decide(pc.JobChange("added", None, head_sites[0]), None,
                              {"verdict": ev.INSUFFICIENT, "judged": 12, "min_n": 30}, None, None, None)
    assert action == "test_more" and "12 judged so far, needs 30" in words


def test_first_look_for_new_job(tmp_path):
    repo, base, head, data = setup(tmp_path)
    judge = judge_mod.JudgeConfig(provider="openai", model="judge-m", url="http://judge.test/v1/chat/completions")
    look = fl.FirstLookConfig(candidate_url="http://cand.test/v1", candidate_model="qwen-7b", candidate_label="qwen",
                              judge=judge, api_url="http://api.test/v1", inputs=2)
    judged = judge_prefers(None)

    def handler(request):
        body = json.loads(request.content)
        if request.url.host == "judge.test":
            if body["messages"][0]["content"] == fl.GEN_SYSTEM:
                return httpx.Response(200, json={"choices": [{"message": {"content": '{"examples": [{"ticket": "a"}, {"ticket": "b"}]}'}}]})
            return judged(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 100, "completion_tokens": 10}})

    doc = pc.run_check(data, scan.scan_ref(repo, base), scan.scan_ref(repo, head), base_ref=base, head_ref=head,
                       options=pc.Options(first_look=look, transport=httpx.MockTransport(handler)))
    rows = {row["job"]: row for row in doc["jobs"]}
    new = rows["app/reply.py::draft_reply"]
    assert new["first_look"]["label"] == fl.LABEL and new["first_look"]["ties"] == 2
    # the edited prompt has no traffic of its own yet, so it gets a first look too
    assert rows["app/support.py::summarize"]["first_look"]["inputs"] == 2
    md = pc.render_markdown(doc)
    assert "first look: as good on 2 of 2" in md
    assert "**Proof (first look, not enough data yet):**" in md and "This is not a verdict and moves nothing." in md


def _look_world(answer="ok"):
    judged = judge_prefers(None)

    def handler(request):
        body = json.loads(request.content)
        if request.url.host == "judge.test":
            if body["messages"][0]["content"] == fl.GEN_SYSTEM:
                return httpx.Response(200, json={"choices": [{"message": {"content": '{"examples": [{"ticket": "a"}, {"ticket": "b"}]}'}}]})
            return judged(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 100, "completion_tokens": 10}})

    return handler


def _look_config(inputs=2):
    judge = judge_mod.JudgeConfig(provider="openai", model="judge-m", url="http://judge.test/v1/chat/completions")
    return fl.FirstLookConfig(candidate_url="http://cand.test/v1", candidate_model="qwen-7b", candidate_label="qwen",
                              judge=judge, api_url="http://api.test/v1", inputs=inputs)


def test_pr_check_first_looks_are_capped_write_no_lessons_and_render_answers_inert(tmp_path):
    repo, base, head, data = setup(tmp_path)
    many = {f"app/new{i}.py": py_call(f"job{i}", f"You are helper number {i}. Answer the ticket politely.") for i in range(4)}
    head = commit_files(repo, many, "many new jobs")
    answer = "Sure! @octocat see ![x](https://evil.test/p.png) <img src=x> [click](https://evil.test)"
    doc = pc.run_check(data, scan.scan_ref(repo, base), scan.scan_ref(repo, head), base_ref=base, head_ref=head,
                       options=pc.Options(first_look=_look_config(), transport=httpx.MockTransport(_look_world(answer)),
                                          max_first_looks=2))
    looks = [row["first_look"] for row in doc["jobs"] if row.get("first_look")]
    ran = [x for x in looks if not x.get("skipped")]
    assert len(ran) == 2
    assert any("--max-first-looks" in (x.get("skipped") or "") for x in looks)
    # a pull request (maybe never merged) never adds training lessons
    assert not jobs.lesson_counts(data)
    md = pc.render_markdown(doc)
    for line in md.splitlines():
        if line.startswith("**API:**") or line.startswith("**Your model:**"):
            assert line.split(":**", 1)[1].strip().startswith("`"), line


def test_model_change_is_not_priced_on_the_old_models_calls_as_current(tmp_path):
    repo, base, _, data = setup(tmp_path)
    head = commit_files(repo, {"app/triage.py": py_call("triage", TRIAGE, "{ticket}", model="gpt-4.1")}, "model")
    doc = pc.run_check(data, scan.scan_ref(repo, base), scan.scan_ref(repo, head), base_ref=base, head_ref=head,
                       options=pc.Options(candidate="qwen"))
    row = [x for x in doc["jobs"] if x["job"] == "app/triage.py::triage"][0]
    # the captured calls were all gpt-4o-mini: they are the OLD version's cost
    assert row["traffic_basis"] == "old version"
    assert "old version" in pc._api_cell(row)


def test_cli_writes_markdown_and_json(tmp_path, capsys):
    repo, base, head, data = setup(tmp_path)
    args = build_parser().parse_args([
        "pr-check", "--repo", str(repo), "--base", base, "--head", head, "--dir", str(data),
        "--markdown", str(tmp_path / "c.md"), "--json", str(tmp_path / "c.json"),
        "--candidate-url", "http://never.test", "--candidate-model", "m", "--judge-model", "j", "--no-first-look",
    ])
    assert pc.handle(args) == 0
    assert "3 AI jobs changed" in capsys.readouterr().out
    assert (tmp_path / "c.md").read_text().startswith(pc.MARKER)
    doc = json.loads((tmp_path / "c.json").read_text())
    assert doc["schema"] == pc.SCHEMA and len(doc["jobs"]) == 3
    bad = build_parser().parse_args(["pr-check", "--repo", str(repo), "--base", "nope", "--dir", str(data)])
    assert pc.handle(bad) == 2


def test_no_changes_comment(tmp_path):
    repo = tmp_path / "repo"
    base = commit_files(repo, dict(BASE), "base")
    head = commit_files(repo, {"README.md": "docs only"}, "docs")
    sites = scan.scan_ref(repo, base)
    doc = pc.run_check(tmp_path / "data", sites, scan.scan_ref(repo, head), base_ref=base, head_ref=head)
    md = pc.render_markdown(doc)
    assert "no AI calls changed" in md and "tracks 3" in md


def test_money_formatting():
    assert pc.money(1234.6) == "$1,235"
    assert pc.money(12.345) == "$12.35" and pc.money(0.42) == "$0.42"
    assert pc.money(0.00123) == "$0.0012" and pc.money(None) == "-" and pc.money(0) == "$0"
