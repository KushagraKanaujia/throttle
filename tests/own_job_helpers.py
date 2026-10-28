"""Shared fakes for the AI-job tests (scan, jobs, first look, pr-check). In-process only."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import httpx

from throttle.own import jobs
from throttle.own import records as r
from throttle.own import scan

SUMMARY = "You are a support lead. Summarize the customer ticket in two short sentences."
TRIAGE = "Classify the support ticket into exactly one label: billing, bug, account or other."


def py_call(func: str, system: str, user: str = "Ticket: {ticket}", *, model: str = "gpt-4o-mini", extra: str = "") -> str:
    return (
        f"def {func}(ticket):\n"
        f"    return client.chat.completions.create(model={model!r}, messages=[\n"
        f"        {{'role': 'system', 'content': {system!r}}},\n"
        f"        {{'role': 'user', 'content': f{user!r}}}]{extra})\n"
    )


def sites_for(files: dict[str, str]) -> list[scan.CallSite]:
    return scan.scan_files(files)


def exchange(system: str, user: str, output: str = "API ANSWER", *, job: str | None = None, model: str = "gpt-4o-mini",
             ts: str | None = None, usage: tuple[int, int] = (1000, 200), fallback: bool = False,
             served_by: str = r.SERVED_API) -> r.Exchange:
    request = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "max_tokens": 200}
    response = {"choices": [{"message": {"role": "assistant", "content": output}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}
    data: dict[str, Any] = dict(provider=r.PROVIDER_OPENAI, endpoint="/v1/chat/completions", model=model,
                                request=request, response=response, status=200, latency_ms=10.0,
                                usage=r.usage_from_openai(response), job=job, fallback=fallback, served_by=served_by,
                                redaction=r.REDACTION_MASK)
    if ts:
        data["ts"] = ts
    return r.Exchange(**data)


def shadow(exchange_id: str, text: str = "CANDIDATE ANSWER", *, label: str = "qwen", model: str = "qwen-7b") -> r.ShadowResult:
    return r.ShadowResult(exchange_id=exchange_id, candidate=label, candidate_model=model, output_text=text,
                          tool_calls=[], usage=r.Usage(900, 150), latency_ms=5.0, status=200, finish_reason="stop")


def judge_prefers(marker: str | None, calls: list | None = None) -> Callable[[httpx.Request], httpx.Response]:
    """A judge that picks the answer containing ``marker`` (TIE when neither
    or both do, or when marker is None)."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"]
        if calls is not None:
            calls.append(prompt)
        a = prompt.split("=== Answer A ===\n")[1].split("\n\n=== Answer B")[0]
        b = prompt.split("=== Answer B ===\n")[1].split("\n\nWhich answer")[0]
        verdict = "TIE"
        if marker is not None and (marker in a) != (marker in b):
            verdict = "A" if marker in a else "B"
        return httpx.Response(200, json={"choices": [{"message": {"content": f"thinking\n{verdict}"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 50, "completion_tokens": 5}})

    return handler


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    return done.stdout.strip()


def commit_files(repo: Path, files: dict[str, str | None], message: str) -> str:
    if not (repo / ".git").exists():
        repo.mkdir(parents=True, exist_ok=True)
        git(repo, "init", "-q", "-b", "main")
    for path, text in files.items():
        target = repo / path
        if text is None:
            target.unlink()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


def write_job_eval(root, job, verdict, fingerprint, *, label="qwen", model="qwen-7b", ts="2026-10-27T10:00:00.000Z", run_id=None):
    run_id = run_id or f"jobeval-{ts[:13]}-{verdict[:4]}"
    doc = {"schema": jobs.EVAL_SCHEMA, "run_id": run_id, "ts": ts, "candidate": label, "candidate_model": model,
           "min_n": 30, "jobs": {job: {"verdict": verdict, "prompt_fingerprint": fingerprint, "judged": 40,
                                       "wins": 5, "ties": 33, "losses": 2, "n": 40, "reasons": ["r"]}}}
    path = jobs.job_eval_dir(root) / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return run_id
