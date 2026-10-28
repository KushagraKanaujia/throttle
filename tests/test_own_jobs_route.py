"""Routing that follows the code: tracked jobs in capture --route. Offline."""

from __future__ import annotations

import json

import httpx

from throttle.own import jobs
from throttle.own import records as r
from throttle.own import route as rt
from throttle.own.capture import create_capture_app

from own_job_helpers import SUMMARY, TRIAGE, py_call, sites_for

OPENAI = "https://api.openai.test"
ANTHROPIC = "https://api.anthropic.test"
JOB = "app/support.py::summarize"
BASE = {"app/support.py": py_call("summarize", SUMMARY), "app/triage.py": py_call("triage", TRIAGE, "{ticket}")}
API_ANSWER = {"id": "x", "object": "chat.completion", "model": "gpt-4o-mini",
              "choices": [{"index": 0, "message": {"role": "assistant", "content": "from the API"}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 10, "completion_tokens": 3}}


def owned_answer(content="from my model", finish="stop"):
    return {"id": "o", "object": "chat.completion", "model": "qwen-7b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4}}


class Recorder:
    def __init__(self, respond):
        self.respond = respond
        self.seen: list[httpx.Request] = []

    async def __call__(self, request):
        await request.aread()
        self.seen.append(request)
        return self.respond(request)


def make_app(root, owned=lambda req: httpx.Response(200, json=owned_answer()), *, routed=True):
    upstream = Recorder(lambda req: httpx.Response(200, json=API_ANSWER))
    owned_rec = Recorder(owned)
    router = rt.Router(root, transport=httpx.MockTransport(owned_rec), seed=0) if routed else None
    app = create_capture_app(OPENAI, ANTHROPIC, redaction=r.REDACTION_OFF, root=root, salt="s",
                             client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)), router=router)
    return app, upstream, owned_rec


async def post(app, system, user, headers=None, **extra):
    body = {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    body.update(extra)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://capture") as client:
        resp = await client.post("/v1/chat/completions", content=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", **(headers or {})})
    await app.state.capture.drain()
    return resp


def move(root, job=JOB):
    jobs.sync(root, sites_for(BASE))
    registry = jobs.load_registry(root)
    entry = registry[job]
    entry.status = jobs.MOVED
    entry.route = {"candidate_label": "qwen", "url": "http://gpu:8000/v1", "model": "qwen-7b", "mode": "live",
                   "approved_eval_run_id": "jobeval-1", "eval_verdict": "BETTER",
                   "prompt_fingerprint": entry.call_site.prompt_fingerprint, "site_key": entry.call_site.approval_key}
    jobs.save_registry(root, registry)


async def test_moved_job_is_served_by_owned_model_and_tagged(tmp_path):
    move(tmp_path)
    app, upstream, owned = make_app(tmp_path)
    resp = await post(app, SUMMARY, "Ticket: printer jam")
    assert resp.headers["x-throttle-served-by"] == "owned" and not upstream.seen
    ex = list(r.iter_exchanges(tmp_path))[0]
    assert ex.job == JOB and ex.served_by == r.SERVED_OWNED
    ledger = [line for p in (tmp_path / "ledger").glob("*.jsonl") for line in r.read_jsonl(p)]
    assert ledger[0]["job"] == JOB


async def test_changed_prompt_is_never_served_silently(tmp_path):
    move(tmp_path)
    app, upstream, owned = make_app(tmp_path)
    # the code changed the prompt but nobody ran sync yet: the request no longer fits
    resp = await post(app, SUMMARY + " Also list the steps already tried.", "Ticket: printer jam",
                      headers={"X-Throttle-Job": JOB})
    assert "x-throttle-served-by" not in resp.headers and len(upstream.seen) == 1 and not owned.seen
    # after sync the job is back in testing, so even the old prompt goes to the API
    edited = {**BASE, "app/support.py": py_call("summarize", SUMMARY + " Also list the steps already tried.")}
    jobs.sync(tmp_path, sites_for(edited))
    resp = await post(app, SUMMARY, "Ticket: printer jam")
    assert "x-throttle-served-by" not in resp.headers and not owned.seen


async def test_tracked_jobs_ignore_type_routes_untracked_traffic_uses_them(tmp_path):
    jobs.sync(tmp_path, sites_for(BASE))
    doc = {"schema": "throttle.own.eval/1", "run_id": "e1", "ts": "2026-10-27T00:00:00.000Z", "candidate": "qwen",
           "candidate_model": "qwen-7b", "types": {"chat": {"verdict": "BETTER"}, "classification": {"verdict": "BETTER"}}}
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "e1.json").write_text(json.dumps(doc))
    for kind in ("chat", "classification"):
        assert rt.approve(tmp_path, kind, "qwen", "http://gpu:8000/v1", "qwen-7b")[0] is not None
    app, upstream, owned = make_app(tmp_path)
    await post(app, SUMMARY, "Ticket: a")               # tracked, testing: API
    assert len(upstream.seen) == 1 and not owned.seen
    resp = await post(app, "You write haiku about anything.", "the sea")   # not a job: type route
    assert resp.headers.get("x-throttle-served-by") == "owned"


async def test_job_header_is_read_not_forwarded_and_fallback_becomes_lesson(tmp_path):
    move(tmp_path)
    app, upstream, owned = make_app(tmp_path, owned=lambda req: httpx.Response(200, json=owned_answer("", "length")))
    await post(app, SUMMARY, "Ticket: b", headers={"X-Throttle-Job": JOB})
    assert len(upstream.seen) == 1
    assert "x-throttle-job" not in {k.lower() for k in upstream.seen[0].headers}
    ex = list(r.iter_exchanges(tmp_path))[0]
    assert ex.fallback and ex.job == JOB and ex.route_reason == "truncated"
    lessons = list(r.read_jsonl(jobs.lessons_path(tmp_path, JOB)))
    assert lessons[0]["source"] == "fallback" and lessons[0]["preferred"] == "from the API"


async def test_capture_without_router_tags_jobs_from_header_and_registry(tmp_path):
    app, upstream, _ = make_app(tmp_path, routed=False)
    await post(app, "Any prompt at all for this test.", "x", headers={"X-Throttle-Job": "my-job"})
    jobs.sync(tmp_path, sites_for(BASE))
    await post(app, TRIAGE, "refund please")
    await post(app, "Unknown prompt here, matches nothing.", "y")
    got = [e.job for e in r.iter_exchanges(tmp_path)]
    assert got == ["my-job", "app/triage.py::triage", None]
    assert all("x-throttle-job" not in {k.lower() for k in req.headers} for req in upstream.seen)


async def test_moved_job_is_not_served_when_the_request_is_another_version_of_the_call(tmp_path):
    # The code shipped a new model, tools, JSON mode or user prompt before
    # anyone ran `jobs sync`. None of these was approved, so all go to the API.
    move(tmp_path)
    app, upstream, owned = make_app(tmp_path)
    tool = [{"type": "function", "function": {"name": "refund", "parameters": {"type": "object"}}}]
    await post(app, SUMMARY, "Ticket: a", model="gpt-4o")
    await post(app, SUMMARY, "Ticket: a", tools=tool)
    await post(app, SUMMARY, "Ticket: a", response_format={"type": "json_object"})
    await post(app, SUMMARY, "Translate this ticket to French: a")      # user prompt edited, system kept
    assert len(upstream.seen) == 4 and not owned.seen
    resp = await post(app, SUMMARY, "Ticket: a")                        # the approved version still moves
    assert resp.headers.get("x-throttle-served-by") == "owned"


async def test_a_route_not_tied_to_this_version_of_the_call_never_serves(tmp_path):
    move(tmp_path)
    registry = jobs.load_registry(tmp_path)
    registry[JOB].route.pop("site_key")             # approved before call settings were tracked
    jobs.save_registry(tmp_path, registry)
    app, upstream, owned = make_app(tmp_path)
    await post(app, SUMMARY, "Ticket: a")
    assert len(upstream.seen) == 1 and not owned.seen
