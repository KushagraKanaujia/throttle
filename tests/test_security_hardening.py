"""Regression tests for the security pass (proxy cache scope, capture guards,
PR-comment injection, redaction, file modes, judge injection, action pins)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from throttle.own import records as r
from throttle.own.capture import create_capture_app
from throttle.proxy import ProxyServer, create_app

REPO = Path(__file__).resolve().parents[1]

CHAT = {"model": "m", "messages": [{"role": "user", "content": "same prompt for everyone"}]}


def _completion(text: str) -> dict:
    return {"id": "x", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


class Backend:
    """Answers with the caller's Authorization value, so a leak is visible."""

    def __init__(self, delay: float = 0.0) -> None:
        self.calls: list[httpx.Request] = []
        self.delay = delay

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        who = request.headers.get("authorization") or request.headers.get("x-api-key") or "anonymous"
        return httpx.Response(200, json=_completion(f"answer for {who}"))


def _proxy(backend: Backend, **kwargs) -> ProxyServer:
    proxy = ProxyServer("http://backend.test", enable_cache=True, **kwargs)
    proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    return proxy


def _front(app, base_url: str = "http://127.0.0.1:8080"):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)


# ---------------------------------------------------------------------------
# proxy.py: cached answers never cross callers
# ---------------------------------------------------------------------------


async def test_proxy_different_keys_both_reach_backend():
    backend = Backend()
    proxy = _proxy(backend)
    async with _front(proxy.app) as client:
        a = await client.post("/v1/chat/completions", json=CHAT, headers={"Authorization": "Bearer key-A"})
        b = await client.post("/v1/chat/completions", json=CHAT, headers={"Authorization": "Bearer key-B"})
    assert len(backend.calls) == 2
    assert a.json()["choices"][0]["message"]["content"] == "answer for Bearer key-A"
    assert b.json()["choices"][0]["message"]["content"] == "answer for Bearer key-B"


async def test_proxy_same_key_is_still_cached():
    backend = Backend()
    proxy = _proxy(backend)
    async with _front(proxy.app) as client:
        for _ in range(2):
            resp = await client.post("/v1/chat/completions", json=CHAT, headers={"Authorization": "Bearer key-A"})
            assert resp.status_code == 200
    assert len(backend.calls) == 1


@pytest.mark.parametrize("header", ["x-api-key", "OpenAI-Organization", "OpenAI-Project"])
async def test_proxy_other_identity_headers_split_the_cache(header):
    backend = Backend()
    proxy = _proxy(backend)
    base = {"Authorization": "Bearer shared"} if header != "x-api-key" else {}
    async with _front(proxy.app) as client:
        await client.post("/v1/chat/completions", json=CHAT, headers={**base, header: "one"})
        await client.post("/v1/chat/completions", json=CHAT, headers={**base, header: "two"})
    assert len(backend.calls) == 2
    assert {c.headers.get(header) for c in backend.calls} == {"one", "two"}


async def test_proxy_request_without_auth_is_never_cached_or_served_from_cache():
    backend = Backend()
    proxy = _proxy(backend)
    async with _front(proxy.app) as client:
        # an authenticated caller fills the cache for this prompt
        await client.post("/v1/chat/completions", json=CHAT, headers={"Authorization": "Bearer key-A"})
        anon1 = await client.post("/v1/chat/completions", json=CHAT)
        anon2 = await client.post("/v1/chat/completions", json=CHAT)
    assert len(backend.calls) == 3
    assert anon1.json()["choices"][0]["message"]["content"] == "answer for anonymous"
    assert anon2.json()["choices"][0]["message"]["content"] == "answer for anonymous"
    assert proxy.cache is not None and proxy.cache.metrics.hits == 0


async def test_proxy_anonymous_requests_are_not_deduplicated():
    backend = Backend(delay=0.05)
    proxy = _proxy(backend)
    async with _front(proxy.app) as client:
        await asyncio.gather(*[client.post("/v1/chat/completions", json=CHAT) for _ in range(3)])
    assert len(backend.calls) == 3


async def test_proxy_inflight_dedup_is_per_caller():
    backend = Backend(delay=0.05)
    proxy = _proxy(backend)
    async with _front(proxy.app) as client:
        results = await asyncio.gather(
            client.post("/v1/chat/completions", json=CHAT, headers={"Authorization": "Bearer key-A"}),
            client.post("/v1/chat/completions", json=CHAT, headers={"Authorization": "Bearer key-B"}),
            client.post("/v1/chat/completions", json=CHAT, headers={"Authorization": "Bearer key-A"}),
        )
    assert len(backend.calls) == 2
    texts = [res.json()["choices"][0]["message"]["content"] for res in results]
    assert texts == ["answer for Bearer key-A", "answer for Bearer key-B", "answer for Bearer key-A"]


def test_proxy_scope_key_hashes_the_caller_and_never_holds_the_key():
    proxy = ProxyServer("http://backend.test", enable_cache=False)
    a = proxy._caller_scope({"authorization": "Bearer sk-secret-A"})
    b = proxy._caller_scope({"authorization": "Bearer sk-secret-B"})
    assert a and b and a != b
    assert proxy._caller_scope({}) is None
    assert proxy._caller_scope({"openai-organization": "org-1"}) is None  # an org alone is not a key
    key = proxy._extract_scope_key(CHAT, a)
    assert "sk-secret-A" not in key and a in key
    assert proxy._extract_scope_key(CHAT) == json.dumps({"model": "m"}, sort_keys=True)


@pytest.mark.parametrize("host,status", [
    ("http://127.0.0.1:8080", 200), ("http://localhost:8080", 200), ("http://[::1]:8080", 200),
    ("http://attacker.example", 403), ("http://attacker.example:8080", 403), ("http://10.0.0.5:8080", 403),
])
async def test_proxy_host_allowlist_when_bound_to_loopback(host, status):
    proxy = _proxy(Backend(), bind_host="127.0.0.1")
    async with _front(proxy.app, base_url=host) as client:
        resp = await client.get("/health")
    assert resp.status_code == status


async def test_proxy_no_host_check_when_bound_to_all_interfaces():
    proxy = _proxy(Backend(), bind_host="0.0.0.0")
    async with _front(proxy.app, base_url="http://proxy.internal:8080") as client:
        assert (await client.get("/health")).status_code == 200


async def test_proxy_create_app_defaults_to_the_loopback_guard():
    app = create_app("http://backend.test")
    async with _front(app, base_url="http://attacker.example") as client:
        assert (await client.get("/health")).status_code == 403


def test_proxy_docs_say_single_tenant():
    import throttle.proxy as proxy_module
    from throttle.cli import build_parser

    assert "Single tenant" in (proxy_module.__doc__ or "")
    parser = build_parser()
    sub = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction")
    assert "Single tenant" in sub.choices["proxy"].description


# ---------------------------------------------------------------------------
# capture.py: request guards and routing needs a key
# ---------------------------------------------------------------------------


class _Upstream:
    def __init__(self) -> None:
        self.seen: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.seen.append(request)
        return httpx.Response(200, json=_completion("from the API"))


def _capture(tmp_path, upstream, **kwargs):
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    return create_capture_app("https://api.openai.test", "https://api.anthropic.test", redaction=r.REDACTION_MASK,
                              root=tmp_path, salt="s", client=client, **kwargs)


KEY = {"authorization": "Bearer sk-test-key"}


@pytest.mark.parametrize("base_url", ["http://attacker.example:8790", "http://192.168.1.4:8790"])
async def test_capture_rejects_non_loopback_host_when_bound_to_loopback(tmp_path, base_url):
    upstream = _Upstream()
    app = _capture(tmp_path, upstream, bind_host="127.0.0.1")
    async with _front(app, base_url=base_url) as client:
        resp = await client.post("/v1/chat/completions", json=CHAT, headers=KEY)
        other = await client.get("/v1/models", headers=KEY)
        health = await client.get("/health")
    assert resp.status_code == other.status_code == health.status_code == 403
    assert upstream.seen == []
    assert list(r.iter_exchanges(tmp_path)) == []


@pytest.mark.parametrize("base_url", ["http://127.0.0.1:8790", "http://localhost:8790", "http://[::1]:8790"])
async def test_capture_accepts_loopback_host(tmp_path, base_url):
    upstream = _Upstream()
    app = _capture(tmp_path, upstream, bind_host="127.0.0.1")
    async with _front(app, base_url=base_url) as client:
        resp = await client.post("/v1/chat/completions", json=CHAT, headers=KEY)
    assert resp.status_code == 200 and len(upstream.seen) == 1


async def test_capture_rejects_browser_origin(tmp_path):
    upstream = _Upstream()
    app = _capture(tmp_path, upstream)
    async with _front(app) as client:
        resp = await client.post("/v1/chat/completions", json=CHAT, headers={**KEY, "Origin": "https://evil.example"})
        other = await client.get("/v1/models", headers={**KEY, "Origin": "null"})
    assert resp.status_code == other.status_code == 403
    assert upstream.seen == []


@pytest.mark.parametrize("content_type", ["text/plain", "application/x-www-form-urlencoded", None])
async def test_capture_requires_json_on_post(tmp_path, content_type):
    upstream = _Upstream()
    app = _capture(tmp_path, upstream)
    headers = dict(KEY)
    if content_type:
        headers["content-type"] = content_type
    async with _front(app) as client:
        resp = await client.post("/v1/chat/completions", content=json.dumps(CHAT).encode(), headers=headers)
        messages = await client.post("/v1/messages", content=json.dumps(CHAT).encode(), headers=headers)
        other = await client.post("/v1/files", content=b"x=1", headers=headers)
    assert resp.status_code == messages.status_code == other.status_code == 415
    assert upstream.seen == []


async def test_capture_still_allows_json_with_charset_multipart_and_empty_posts(tmp_path):
    upstream = _Upstream()
    app = _capture(tmp_path, upstream)
    async with _front(app) as client:
        ok = await client.post("/v1/chat/completions", content=json.dumps(CHAT).encode(),
                               headers={**KEY, "content-type": "application/json; charset=utf-8"})
        upload = await client.post("/v1/files", files={"file": ("a.jsonl", b"{}")}, data={"purpose": "batch"},
                                   headers=KEY)
        cancel = await client.post("/v1/batches/b1/cancel", headers=KEY)
    assert ok.status_code == upload.status_code == cancel.status_code == 200
    assert len(upstream.seen) == 3


class _StubRouter:
    def __init__(self) -> None:
        self.calls = 0

    async def route(self, provider, endpoint, request_json, stream, job=None):  # noqa: ARG002
        self.calls += 1
        return SimpleNamespace(served=True, body=_completion("from your model"), sse=[], owned_latency_ms=1.0,
                               served_by=r.SERVED_OWNED, fallback=False, route_reason="approved",
                               workload_type="chat", job=None)

    def record(self, exchange, decision):  # noqa: ARG002
        return None

    async def aclose(self) -> None:
        return None


async def test_capture_route_needs_a_key_before_the_owned_model_answers(tmp_path):
    upstream = _Upstream()
    router = _StubRouter()
    app = _capture(tmp_path, upstream, router=router)
    async with _front(app) as client:
        anon = await client.post("/v1/chat/completions", json=CHAT)
        keyed = await client.post("/v1/chat/completions", json=CHAT, headers=KEY)
        anthropic = await client.post("/v1/messages", json=CHAT, headers={"x-api-key": "sk-ant-x"})
        await app.state.capture.drain()
    # no key: straight to the upstream, unchanged, which does the key check
    assert anon.json()["choices"][0]["message"]["content"] == "from the API"
    assert "x-throttle-served-by" not in anon.headers
    assert len(upstream.seen) == 1 and "authorization" not in upstream.seen[0].headers
    # with a key: the owned model may answer
    assert keyed.headers["x-throttle-served-by"] == "owned"
    assert anthropic.headers["x-throttle-served-by"] == "owned"
    assert router.calls == 2


@pytest.mark.parametrize("value,recorded", [
    ("ticket-summary", "ticket-summary"),
    ("team/app:job@v2", "team/app:job@v2"),
    ("x` [click](https://attacker.example) `", None),
    ("has space", None),
    ("a" * 121, None),
])
async def test_capture_validates_x_throttle_job(tmp_path, value, recorded):
    upstream = _Upstream()
    app = _capture(tmp_path, upstream)
    async with _front(app) as client:
        resp = await client.post("/v1/chat/completions", json=CHAT, headers={**KEY, "X-Throttle-Job": value})
        await app.state.capture.drain()
    assert resp.status_code == 200
    assert "x-throttle-job" not in upstream.seen[0].headers  # never forwarded
    (exchange,) = list(r.iter_exchanges(tmp_path))
    if recorded is None:
        assert exchange.job != value
        assert "attacker" not in json.dumps(exchange.to_json())
    else:
        assert exchange.job == recorded


def test_tag_valid_rejects_a_trailing_newline():
    from throttle.own.tag import _VALID, throttle_job

    assert _VALID.match("job-1")
    assert not _VALID.match("job-1\n")
    with pytest.raises(ValueError):
        throttle_job("job-1\n")


def test_capture_help_is_honest_about_routing_and_defaults():
    from throttle.cli import build_parser

    parser = build_parser()
    sub = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction")
    capture = sub.choices["capture"]
    desc = " ".join(capture.description.split())
    assert "By default every call goes to your upstream unchanged and is recorded locally" in desc
    assert "With --route" in desc
    for reason in ("errors", "timeouts", "empty or cut-off answers", "bad JSON", "schema or tool-call failures"):
        assert reason in desc
    helps = {a.dest: a.help or "" for a in capture._actions}
    assert "default: 127.0.0.1" in helps["host"]
    assert "default: 8790" in helps["port"]
    assert "—" not in desc


# ---------------------------------------------------------------------------
# prcheck.py: PR-derived text is inert in the comment
# ---------------------------------------------------------------------------

from throttle.own import jobs as jobs_mod  # noqa: E402
from throttle.own import prcheck as pc  # noqa: E402
from throttle.own import scan  # noqa: E402

_BASE_APP = ('from openai import OpenAI\nc=OpenAI()\ndef f():\n    return c.chat.completions.create(model="gpt-4o-mini", '
             'messages=[{"role":"system","content":"You summarize support tickets for the team please"}])\n')
EVIL_MODEL = ("gpt-4o-mini ![x](https://attacker.example/p.png) [Approve here](https://attacker.example) "
              "@acme/security-team")
EVIL_TAG = "x` [click](https://attacker.example) `"


def _outside_code(md: str) -> str:
    """The comment with every inline code span removed: what renders as Markdown."""
    return re.sub(r"`[^`\n]*`", "", md)


def _assert_inert(md: str) -> None:
    live = _outside_code(md)
    assert "](http" not in live
    assert "![" not in live
    assert "@acme" not in live
    assert "<img" not in live.lower()
    assert "https://attacker" not in live  # no bare URL to autolink either
    # code spans are balanced on every line: nothing broke out of one
    for line in md.splitlines():
        assert line.count("`") % 2 == 0, line


def _comment(base_files, head_files, tmp_path):
    doc = pc.run_check(tmp_path / "data", scan.scan_files(base_files), scan.scan_files(head_files),
                       base_ref="a" * 40, head_ref="b" * 40)
    return doc, pc.render_markdown(doc)


def test_pr_comment_model_injection_renders_inert(tmp_path):
    head = {"app.py": _BASE_APP.replace('model="gpt-4o-mini"', f'model="{EVIL_MODEL}"')}
    doc, md = _comment({"app.py": _BASE_APP}, head, tmp_path)
    assert doc["jobs"] and doc["jobs"][0]["change"] == "changed"
    assert "attacker.example" in md  # it is shown, as text
    _assert_inert(md)


def test_pr_comment_added_job_with_evil_model_renders_inert(tmp_path):
    head = {"app.py": _BASE_APP, "evil.py": _BASE_APP.replace('model="gpt-4o-mini"', f'model="{EVIL_MODEL}"')}
    _, md = _comment({"app.py": _BASE_APP}, head, tmp_path)
    _assert_inert(md)


def test_scan_ignores_an_invalid_tag_with_a_warning(caplog):
    src = _BASE_APP.replace("messages=", 'extra_headers={"X-Throttle-Job": "' + EVIL_TAG.replace("`", "\\x60") + '"}, messages=')
    with caplog.at_level("WARNING", logger="throttle.own.scan"):
        (site,) = scan.scan_files({"app.py": src})
    assert site.tag is None and site.job == "app.py::f"
    assert "invalid X-Throttle-Job tag" in caplog.text


def test_scan_ignores_invalid_tags_in_javascript_and_comments(caplog):
    js = ('import OpenAI from "openai";\nconst c = new OpenAI();\nexport async function f() {\n'
          '  return c.chat.completions.create({model: "gpt-4o-mini", messages: [{role: "system", content: "You summarize tickets for the team"}]},'
          ' { headers: { "X-Throttle-Job": "[click](https://attacker.example)" } });\n}\n')
    py = "# throttle-job: " + "a" * 130 + "\n" + _BASE_APP.replace("def f():\n    return", "def f():\n    # throttle-job: " + "b" * 130 + "\n    return")
    with caplog.at_level("WARNING", logger="throttle.own.scan"):
        sites = scan.scan_files({"web.js": js, "app.py": py})
    assert all(s.tag is None for s in sites), [s.tag for s in sites]
    assert caplog.text.count("invalid X-Throttle-Job tag") >= 2


def test_pr_comment_tag_cannot_break_out_of_its_code_span(tmp_path):
    # A tag that reached the comment some other way (an old registry, a fork
    # of the scanner) still stays inside its code span.
    (base_site,) = scan.scan_files({"app.py": _BASE_APP})
    evil = scan.CallSite.from_json({**base_site.to_json(), "tag": EVIL_TAG, "path": "new.py"})
    doc = pc.run_check(tmp_path / "data", [base_site], [base_site, evil], base_ref="a" * 40, head_ref="b" * 40)
    md = pc.render_markdown(doc)
    assert pc._code(EVIL_TAG).count("`") == 2
    _assert_inert(md)


def test_pr_comment_evil_path_and_move_text_are_inert(tmp_path):
    (site,) = scan.scan_files({"app.py": _BASE_APP})
    evil_path = scan.CallSite.from_json({**site.to_json(), "path": "docs/![x](https://attacker.example/p.png).py"})
    doc = pc.run_check(tmp_path / "data", [site], [evil_path], base_ref="a" * 40, head_ref="b" * 40)
    assert doc["jobs"][0]["change"] == "moved"
    _assert_inert(pc.render_markdown(doc))


def test_approve_action_text_keeps_the_job_in_a_code_span():
    (site,) = scan.scan_files({"app.py": _BASE_APP})
    evil = scan.CallSite.from_json({**site.to_json(), "tag": EVIL_TAG})
    action, words = pc.decide(pc.JobChange("changed", site, evil), None, {"verdict": "BETTER"}, None, 0.001, 0.01)
    assert action == "move"
    assert "throttle jobs approve" in words
    _assert_inert(words)


@pytest.mark.parametrize("raw", [EVIL_MODEL, EVIL_TAG, "<script>x</script>", "a\n\n# heading", "*bold* _it_ ~~s~~ | t |",
                                 "see www.attacker.example", "&#64;acme"])
def test_md_escape_neutralises_markdown(raw):
    out = pc.md_escape(raw)
    assert "\n" not in out
    assert "](" not in out and "![" not in out and "@acme" not in out
    assert "<script>" not in out and "://" not in out and "www.attacker" not in out
    for ch in "[]()!<>*_~`|":
        assert ch not in out.replace("\\" + ch, "")


def test_changes_wrap_model_names_in_code_spans():
    (a,) = scan.scan_files({"app.py": _BASE_APP})
    b = scan.CallSite.from_json({**a.to_json(), "model": "gpt-4.1 `x` [y](https://z)"})
    (what,) = jobs_mod.changes(a, b)
    assert what == "model changed from `gpt-4o-mini` to `gpt-4.1 'x' [y](https://z)`"


# ---------------------------------------------------------------------------
# prcheck.py / jobs.py: what the comment promises
# ---------------------------------------------------------------------------


def test_added_job_with_unreadable_prompt_says_it_needs_a_tag():
    src = ('from openai import OpenAI\nc=OpenAI()\ndef f(msgs):\n'
           '    return c.chat.completions.create(model="gpt-4o-mini", messages=msgs)\n')
    (site,) = scan.scan_files({"app.py": src})
    _, words = pc.decide(pc.JobChange("added", None, site), None, None, None, None, None)
    assert words == "Throttle cannot read this prompt, so it stays on the API. Tag it with X-Throttle-Job to test it."
    tagged = scan.CallSite.from_json({**site.to_json(), "tag": "ticket-summary"})
    _, words = pc.decide(pc.JobChange("added", None, tagged), None, None, None, None, None)
    assert words == "Stays on the API. It can be tested once there is captured traffic and a candidate."


def test_added_job_with_readable_prompt_promises_nothing_automatic():
    (site,) = scan.scan_files({"app.py": _BASE_APP})
    _, words = pc.decide(pc.JobChange("added", None, site), None, None, None, None, None)
    assert words == "Stays on the API. It can be tested once there is captured traffic and a candidate."
    assert "starts testing" not in words


def test_ready_to_move_cell_says_per_call_and_points_to_the_report():
    cell = pc._action_cell({"action": "move", "action_title": pc.ACTIONS["move"]})
    assert cell.startswith("**Ready to move**")
    assert "per call; see throttle report for the full monthly cost" in cell
    assert pc._action_cell({"action": "keep", "action_title": pc.ACTIONS["keep"]}) == "**Keep on the API**"


def test_footer_says_failed_answers_fall_back(tmp_path):
    head = {"app.py": _BASE_APP.replace("gpt-4o-mini", "gpt-4.1")}
    _, md = _comment({"app.py": _BASE_APP}, head, tmp_path)
    assert "falls back to the API" in md.splitlines()[-1]
    assert "—" not in md


def test_testing_status_words():
    assert jobs_mod.STATUS_WORDS[jobs_mod.TESTING] == "on the API, not tested yet"


# ---------------------------------------------------------------------------
# GitHub Actions: pinned installs, no pull_request_target with keys, pinned uses
# ---------------------------------------------------------------------------

import subprocess  # noqa: E402

import yaml  # noqa: E402

PR_ACTION = REPO / ".github" / "actions" / "throttle-pr" / "action.yml"
GH_SHA = "0123456789abcdef0123456789abcdef01234567"


def _step(name: str) -> str:
    steps = yaml.safe_load(PR_ACTION.read_text())["runs"]["steps"]
    return next(s["run"] for s in steps if s["name"] == name)


def _run_step(tmp_path, name: str, env: dict) -> subprocess.CompletedProcess:
    script = tmp_path / "step.sh"
    script.write_text(_step(name))
    full = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "RUNNER_TEMP": str(tmp_path / "runner"),
            "GITHUB_ENV": str(tmp_path / "github_env"), **env}
    (tmp_path / "runner").mkdir(exist_ok=True)
    return subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)], env=full,
                          capture_output=True, text=True, check=False)


def _fake_python(tmp_path) -> Path:
    """A python3 on PATH whose venv's pip records what it was asked to install."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / "pip.log"
    fake = bindir / "python3"
    fake.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then\n'
        '  mkdir -p "$3/bin"\n'
        f'  printf \'#!/bin/bash\\necho "$@" >> {log}\\n\' > "$3/bin/python"\n'
        '  chmod +x "$3/bin/python"\n'
        "fi\n"
    )
    fake.chmod(0o755)
    return bindir


def test_pr_action_has_no_branch_fallback():
    text = PR_ACTION.read_text()
    assert ":-main" not in text
    assert "@main" not in _step("Install Throttle")


@pytest.mark.skipif(sys.platform == "win32", reason="bash step")
@pytest.mark.parametrize("ref", [GH_SHA, "v0.7.0", "0.7.1"])
def test_pr_action_installs_its_own_tag_or_sha(tmp_path, ref):
    bindir = _fake_python(tmp_path)
    done = _run_step(tmp_path, "Install Throttle", {"SPEC": "", "ACTION_REF": ref,
                                                   "PATH": f"{bindir}:{os.environ['PATH']}"})
    assert done.returncode == 0, done.stderr + done.stdout
    assert f"git+https://github.com/KushagraKanaujia/throttle@{ref}" in (tmp_path / "pip.log").read_text()


@pytest.mark.skipif(sys.platform == "win32", reason="bash step")
@pytest.mark.parametrize("ref", ["", "main", "feature/x", GH_SHA[:12], "v1"])
def test_pr_action_fails_on_a_branch_or_empty_ref(tmp_path, ref):
    bindir = _fake_python(tmp_path)
    done = _run_step(tmp_path, "Install Throttle", {"SPEC": "", "ACTION_REF": ref,
                                                   "PATH": f"{bindir}:{os.environ['PATH']}"})
    assert done.returncode == 1
    assert "Pin Throttle for GitHub to a release tag or a full commit SHA" in done.stdout
    assert not (tmp_path / "pip.log").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="bash step")
def test_pr_action_explicit_version_works_without_a_pinned_ref(tmp_path):
    bindir = _fake_python(tmp_path)
    done = _run_step(tmp_path, "Install Throttle", {"SPEC": "throttle-pro==0.7.0", "ACTION_REF": "",
                                                   "PATH": f"{bindir}:{os.environ['PATH']}"})
    assert done.returncode == 0, done.stderr
    assert "throttle-pro==0.7.0" in (tmp_path / "pip.log").read_text()


_GUARD = "Refuse to run fork code with API keys"


@pytest.mark.skipif(sys.platform == "win32", reason="bash step")
@pytest.mark.parametrize("env, code", [
    ({"EVENT_NAME": "pull_request_target", "IN_API_KEY_ENV": "MY_OPENAI", "MY_OPENAI": "sk-x"}, 1),
    ({"EVENT_NAME": "pull_request_target", "IN_JUDGE_KEY_ENV": "JUDGE", "JUDGE": "k"}, 1),
    ({"EVENT_NAME": "pull_request_target", "OPENAI_API_KEY": "sk-x"}, 1),
    ({"EVENT_NAME": "pull_request_target", "ANTHROPIC_API_KEY": "sk-ant"}, 1),
    ({"EVENT_NAME": "pull_request_target", "IN_API_KEY_ENV": "UNSET_VAR"}, 0),
    ({"EVENT_NAME": "pull_request_target"}, 0),
    ({"EVENT_NAME": "pull_request", "IN_API_KEY_ENV": "MY_OPENAI", "MY_OPENAI": "sk-x"}, 0),
])
def test_pr_action_refuses_pull_request_target_with_keys(tmp_path, env, code):
    done = _run_step(tmp_path, _GUARD, env)
    assert done.returncode == code, done.stdout + done.stderr
    if code:
        assert "pull_request_target" in done.stdout


def test_pr_action_guard_runs_before_install():
    names = [s["name"] for s in yaml.safe_load(PR_ACTION.read_text())["runs"]["steps"]]
    assert names.index(_GUARD) < names.index("Install Throttle")


# Workflow pinning tests ship with the .github/workflows change (separate PR).


def test_readme_pins_the_pr_action():
    readme = (REPO / "README.md").read_text()
    assert "throttle-pr@main" not in readme
    assert re.search(r"throttle-pr@(v\d+\.\d+\.\d+|[0-9a-f]{40})", readme)
