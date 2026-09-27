"""Tests for the Throttle Console (`throttle ui`) HTTP API.

The fixture history is six real `throttle check` runs against a local Ollama
(four unchanged llama3.2:3b checks, then two llama3.2:1b checks). The verdicts
the CLI printed for them, oldest first, were: first check, NOT CALIBRATED,
NOT CALIBRATED, NO WINNER, CHEAPER (-35.9%), NOT CALIBRATED.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import asyncio

import httpx
import pytest

import throttle.check as check_module
from throttle.cli import build_parser
from throttle.console import server as console_server
from throttle.console.server import DOCS_NAV, create_app, is_loopback, mask_endpoint

FIXTURE = Path(__file__).parent / "fixtures" / "console_checks.ndjson"
CLI_VERDICTS_OLDEST_FIRST = [
    None,
    "NOT CALIBRATED",
    "NOT CALIBRATED",
    "NO WINNER",
    "CHEAPER",
    "NOT CALIBRATED",
]


class Client:
    """Minimal sync client over the ASGI app (no network, no TestClient)."""

    def __init__(self, app, base_url: str) -> None:
        self.app, self.base_url = app, base_url

    def get(self, path: str, params: dict | None = None) -> httpx.Response:
        async def go() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url=self.base_url) as c:
                return await c.get(path, params=params)

        return asyncio.run(go())



@pytest.fixture
def history(tmp_path: Path) -> Path:
    directory = tmp_path / "checks"
    directory.mkdir()
    shutil.copy(FIXTURE, directory / check_module.HISTORY_FILENAME)
    return directory


@pytest.fixture
def client(history: Path, tmp_path: Path):
    yield Client(create_app(history, cwd=tmp_path), "http://127.0.0.1:8787")


def records(history: Path) -> list[dict]:
    loaded, skipped = check_module.load_history(history)
    assert skipped == 0
    return loaded


def test_checks_list_matches_cli_verdicts(client: Client) -> None:
    response = client.get("/api/checks")
    assert response.status_code == 200
    cards = response.json()
    assert [c["verdict"] for c in reversed(cards)] == CLI_VERDICTS_OLDEST_FIRST
    cheaper = next(c for c in cards if c["verdict"] == "CHEAPER")
    assert cheaper["delta_percent"] == pytest.approx(-35.9, abs=0.05)
    assert {change["key"] for change in cheaper["changes"]} == {"label", "request.model"}
    assert all(c["endpoint_masked"] == "localhost:11434" for c in cards)


def test_summary_counts_the_history(client: Client, history: Path) -> None:
    body = client.get("/api/summary").json()
    assert body["checks"] == 6
    assert body["endpoints"] == 1
    assert body["history_file"] == str(history / check_module.HISTORY_FILENAME)


def test_check_detail_uses_judge_recorded(client: Client, history: Path) -> None:
    loaded = records(history)
    index = CLI_VERDICTS_OLDEST_FIRST.index("CHEAPER")
    _baseline, expected = check_module.judge_recorded(loaded, index)
    body = client.get(f"/api/checks/{loaded[index]['id']}").json()
    assert body["comparison"]["verdict"] == expected["verdict"] == "CHEAPER"
    assert body["comparison"]["reason"] == expected["reason"]
    assert body["baseline"]["id"] == expected["previous_id"]
    assert body["share_command"] == f"throttle check --share-id {loaded[index]['id']}"
    assert len(body["blocks"]) == 3
    assert "prompt_nonce" not in body["workload"]


def test_first_check_has_no_comparison(client: Client, history: Path) -> None:
    first = records(history)[0]
    body = client.get(f"/api/checks/{first['id']}").json()
    assert body["comparison"] is None
    assert body["verdict"] is None


def test_unknown_check_is_404(client: Client) -> None:
    assert client.get("/api/checks/nope").status_code == 404


def test_compare_matches_compare_checks(client: Client, history: Path) -> None:
    loaded = records(history)
    a, b = loaded[3], loaded[4]
    expected = check_module.compare_checks(a, b, loaded[:5])
    body = client.get("/api/compare", params={"a": a["id"], "b": b["id"]}).json()
    assert body["comparison"]["verdict"] == expected["verdict"] == "CHEAPER"
    assert body["comparison"]["noise"]["floor_percent"] == pytest.approx(
        expected["noise"]["floor_percent"]
    )
    assert client.get("/api/compare", params={"a": a["id"], "b": a["id"]}).status_code == 400
    assert client.get("/api/compare", params={"a": a["id"], "b": "nope"}).status_code == 404


def test_endpoints_group_by_model(client: Client) -> None:
    groups = client.get("/api/endpoints").json()
    by_model = {g["model"]: g for g in groups}
    assert set(by_model) == {"llama3.2:3b", "llama3.2:1b"}
    assert len(by_model["llama3.2:3b"]["series"]) == 4
    assert by_model["llama3.2:1b"]["labels"] == ["smaller-1b"]


def test_quickstart_status(client: Client, tmp_path: Path) -> None:
    status = {s["key"]: s["done"] for s in client.get("/api/quickstart-status").json()}
    assert status == {"connect": True, "first": True, "calibrate": True, "ci": False}
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "cost.yml").write_text("steps:\n  - uses: KushagraKanaujia/throttle@v0.4.2\n")
    status = {s["key"]: s["done"] for s in client.get("/api/quickstart-status").json()}
    assert status["ci"] is True


def test_empty_history(tmp_path: Path) -> None:
    test_client = Client(create_app(tmp_path / "empty", cwd=tmp_path), "http://127.0.0.1")
    assert test_client.get("/api/checks").json() == []
    status = {s["key"]: s["done"] for s in test_client.get("/api/quickstart-status").json()}
    assert not any(status.values())


def test_every_nav_page_is_bundled_and_renders(client: Client) -> None:
    for group in DOCS_NAV:
        for slug, _title in group["pages"]:
            response = client.get(f"/api/docs/{slug}")
            assert response.status_code == 200, slug
            assert response.json()["markdown"].lstrip().startswith("#"), slug


def test_internal_doc_links_resolve() -> None:
    import re

    docs = console_server.DOCS_DIR
    for path in docs.rglob("*.md"):
        for target in re.findall(r"\]\(([^)#\s]+\.md)", path.read_text(encoding="utf-8")):
            if target.startswith("http"):
                continue
            assert (path.parent / target).resolve().is_file(), f"{path.name} -> {target}"


@pytest.mark.parametrize(
    "slug",
    ["../server", "..%2Fserver", "concepts/../../server", "Concepts/verdicts", "concepts/verdicts.md", "x" * 300],
)
def test_docs_path_traversal_is_blocked(client: Client, slug: str) -> None:
    assert client.get(f"/api/docs/{slug}").status_code == 404


def test_search_finds_docs_and_checks(client: Client) -> None:
    hits = client.get("/api/search", params={"q": "noise bound"}).json()
    assert any(h["kind"] == "doc" for h in hits)
    hits = client.get("/api/search", params={"q": "smaller-1b"}).json()
    assert any(h["kind"] == "check" for h in hits)
    assert client.get("/api/search", params={"q": "x"}).json() == []


def test_index_and_static_are_served_with_csp(client: Client) -> None:
    index = client.get("/")
    assert index.status_code == 200
    assert "Throttle Console" in index.text
    csp = index.headers["content-security-policy"]
    assert "default-src 'none'" in csp and "frame-ancestors 'none'" in csp
    for asset in ("/static/app.js", "/static/app.css", "/static/vendor/marked.min.js",
                  "/static/illustrations/noise.svg"):
        assert client.get(asset).status_code == 200, asset


def test_non_loopback_host_header_is_refused(history: Path, tmp_path: Path) -> None:
    test_client = Client(create_app(history, cwd=tmp_path), "http://evil.example")
    assert test_client.get("/api/checks").status_code == 403


def test_masking_and_loopback_helpers() -> None:
    assert mask_endpoint("http://localhost:11434/v1/chat/completions") == "localhost:11434"
    assert mask_endpoint("http://127.0.0.1:8000/v1/chat/completions") == "127.0.0.1:8000"
    masked = mask_endpoint("https://gpu-7.internal.example.com/v1/chat/completions")
    assert "internal" not in masked and "gpu-7" not in masked
    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
    assert not is_loopback("0.0.0.0") and not is_loopback("192.168.1.5")


def test_ui_parser_defaults_to_loopback() -> None:
    args = build_parser().parse_args(["ui"])
    assert args.host == "127.0.0.1"
    assert args.port == 8787
    assert args.no_open is False
