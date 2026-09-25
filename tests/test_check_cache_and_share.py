"""`throttle check` 0.4.1: cold/warm prompt-cache mode and `--share`.

Drives the real CLI (`throttle.cli.main`) against the mock backend from
test_check.py and asserts on what the user sees: printed output, exit codes,
the prompts that actually reached the server, the history file, and the
pre-filled GitHub issue link.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

import throttle.check as check_module
from throttle.benchmark import load_prompts
from throttle.cli import main

from test_check import FakeServer, constant, history, run_check  # noqa: F401  (fixture)

TAG = re.compile(r"^\[run (\d{6}) req (\d{4})\] ")


class RecordingServer(FakeServer):
    """FakeServer that keeps every chat request body it receives."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bodies: list[dict] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            self.bodies.append(json.loads(request.content))
        return await super().handle(request)

    def probes(self) -> list[dict]:
        """A cold check's untagged tag-cost probes (1 output token each)."""
        return [b for b in self.bodies if b["max_tokens"] == check_module.TAG_PROBE_MAX_TOKENS]

    def measured(self, warmup: int = 1) -> list[list[dict]]:
        return [
            body["messages"] for body in self.bodies[warmup:]
            if body["max_tokens"] != check_module.TAG_PROBE_MAX_TOKENS
        ]


def first_user(messages: list[dict]) -> str:
    return next(m["content"] for m in messages if m["role"] == "user")


def records(history_dir) -> list[dict]:
    path = history_dir / "checks.ndjson"
    return [json.loads(line) for line in path.read_text().splitlines()]


# --------------------------------------------------------------------------
# Prompt cache mode
# --------------------------------------------------------------------------


def test_default_check_sends_a_unique_tagged_prompt_per_request(history, monkeypatch, capsys):
    server = RecordingServer(constant(0.0))
    # 6 per block reaches built-in prompt 5, which has a system message.
    code, out = run_check(monkeypatch, capsys, server, "--requests-per-block", "6")
    assert code == 0, out
    assert "cache      COLD (default)" in out

    measured = server.measured()
    assert len(measured) == 5 * 6
    firsts = [first_user(messages) for messages in measured]
    assert len(set(firsts)) == len(firsts)  # no two measured prompts are identical
    tags = [TAG.match(text) for text in firsts]
    assert all(tags), firsts[:3]
    run_ids = {tag.group(1) for tag in tags}
    assert len(run_ids) == 1
    assert sorted(int(tag.group(2)) for tag in tags) == list(range(30))
    # The tag is the very start of the first user message; a system message
    # ahead of it is sent unchanged.
    with_system = [m for m in measured if m[0]["role"] == "system"]
    assert with_system and all(not TAG.match(m[0]["content"]) for m in with_system)
    # The warm-up request is tagged too, so it never warms a measured prefix.
    assert first_user(server.bodies[0]["messages"]).startswith(f"[run {run_ids.pop()} warmup 0000] ")


def test_cold_prompts_are_reproducible_from_the_recorded_run_id(history, monkeypatch, capsys):
    server = RecordingServer(constant(0.0))
    code, out = run_check(monkeypatch, capsys, server)
    assert code == 0, out
    workload = records(history)[0]["workload"]
    assert workload["prompt_cache_mode"] == "cold"
    run_id = workload["prompt_nonce"]["run_id"]
    assert f"'[run {run_id} req 0000] '" in out  # the header names the run id
    rebuilt = check_module.measured_prompts(load_prompts(), 5, 4, "cold", run_id)
    expected = sorted(json.dumps(list(m)) for row in rebuilt for m in row)
    assert sorted(json.dumps(m) for m in server.measured()) == expected
    assert workload["sent_prompts_sha256"] == check_module.canonical_workload_hash(
        tuple(tuple(m) for row in rebuilt for m in row)
    )
    # The workload identity still hashes the prompt file itself, as smoke,
    # benchmark and golden do, so it is the same for every run.
    assert workload["prompts_sha256"] == check_module.canonical_workload_hash(load_prompts())


def test_warm_cache_resends_identical_prompts(history, monkeypatch, capsys):
    server = RecordingServer(constant(0.0))
    code, out = run_check(monkeypatch, capsys, server, "--warm-cache")
    assert code == 0, out
    assert "cache      WARM (--warm-cache)" in out
    firsts = [first_user(messages) for messages in server.measured()]
    assert not any(TAG.match(text) or "[run " in text for text in firsts)
    counts = Counter(firsts)
    assert len(counts) == 4 and set(counts.values()) == {5}  # 4 prompts, each once per block
    workload = records(history)[0]["workload"]
    assert workload["prompt_cache_mode"] == "warm" and workload["prompt_nonce"] is None


class TokenizingServer(RecordingServer):
    """RecordingServer whose prompt_tokens mimic a BPE tokenizer: digits in
    groups of up to 3 (as Llama 3 / tiktoken split them), letter runs, and
    single punctuation marks. With this count a hex run id such as 'ab12cd'
    (3 tokens) and '7f3a2c' (6 tokens) cost different amounts."""

    @staticmethod
    def count(text: str) -> int:
        return len(re.findall(r"\d{1,3}|[A-Za-z]+|[^\sA-Za-z\d]", text))

    async def handle(self, request: httpx.Request) -> httpx.Response:
        response = await super().handle(request)
        if request.url.path != "/v1/chat/completions":
            return response
        body = json.loads(request.content)
        payload = response.json()
        payload["usage"]["prompt_tokens"] = self.count(
            " ".join(m["content"] for m in body["messages"])
        )
        return httpx.Response(200, json=payload)


def test_run_id_is_fixed_length_decimal():
    # Digits tokenize into a fixed number of tokens; hex does not.
    ids = [check_module.new_run_id() for _ in range(200)]
    assert all(re.fullmatch(r"\d{6}", run_id) for run_id in ids), ids[:5]
    assert len(set(ids)) > 150  # still random per check


def test_cold_input_tokens_exclude_the_tag_and_match_a_warm_check(
    history, monkeypatch, capsys, tmp_path
):
    # 6 requests per block over 8 built-in prompts: 6 distinct base prompts.
    extra = ("--requests-per-block", "6", "--metric", "input")
    colds = []
    for run in range(3):
        server = TokenizingServer(constant(0.0))
        path = tmp_path / f"cold{run}.json"
        code, out = run_check(monkeypatch, capsys, server, *extra, "--json", str(path))
        assert code == 0, out
        colds.append((json.loads(path.read_text()), out, server))
    warm_server = TokenizingServer(constant(0.0))
    code, out_warm = run_check(
        monkeypatch, capsys, warm_server, *extra, "--warm-cache",
        "--json", str(tmp_path / "warm.json"),
    )
    assert code == 0, out_warm
    warm = json.loads((tmp_path / "warm.json").read_text())
    assert warm_server.probes() == []  # a warm check sends no tags, so no probes

    warm_input = {block["input_tokens"] for block in warm["blocks"]}
    assert len(warm_input) == 1
    (n,) = warm_input
    for record, out, server in colds:
        # Each distinct base prompt was sent once, untagged, as a probe.
        probes = server.probes()
        assert [first_user(p["messages"]) for p in probes] == [
            first_user(list(prompt)) for prompt in load_prompts()[:6]
        ]
        assert not any("[run " in first_user(p["messages"]) for p in probes)
        # Input tokens count the real prompts only: the same as a warm check.
        assert {block["input_tokens"] for block in record["blocks"]} == warm_input
        # The tag's tokens are measured, constant, and shown as not counted.
        tag = {block["tag_input_tokens"] for block in record["blocks"]}
        assert len(tag) == 1 and tag.pop() > 0
        assert f": {n} in (+" in out and " tag, not counted) / " in out
        assert "6 unmeasured request(s) send each base prompt once untagged" in out
        nonce = record["workload"]["prompt_nonce"]
        assert sum(nonce["untagged_prompt_tokens"]) == n
        assert nonce["tag_input_tokens"] == sum(b["tag_input_tokens"] for b in record["blocks"])
        # Input $/M is GPU dollars per real prompt token.
        for block in record["blocks"]:
            assert block["dollars_per_million"]["input"] == pytest.approx(
                block["gpu_dollars"] / n * 1_000_000, rel=1e-9
            )
        assert "limits     37 requests" in out  # 5 x 6 + 1 warm-up + 6 probes
    # Same flags, same tag cost in every run: the fixed-length id tokenizes
    # to the same count whatever its digits.
    tags = {b["tag_input_tokens"] for record, _, _ in colds for b in record["blocks"]}
    assert len(tags) == 1, tags


def test_probes_count_toward_the_request_limit(history, monkeypatch, capsys):
    # 5 x 4 measured + 1 warm-up = 21; the 4 cold probes make 25.
    server = RecordingServer(constant(0.0))
    code, out = run_check(monkeypatch, capsys, server, "--max-requests", "24")
    assert code == 2, out
    assert "+ 4 untagged tag-cost probes), above --max-requests 24" in out
    assert server.bodies == []
    code, out = run_check(monkeypatch, capsys, server, "--max-requests", "24", "--warm-cache")
    assert code == 0, out


def test_cold_and_warm_checks_are_never_compared(history, monkeypatch, capsys):
    # Warm is 8x faster (as a prefix cache might make it): still no winner.
    code, out = run_check(monkeypatch, capsys, FakeServer(constant(0.08)))
    assert code == 0, out
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)),
        "--warm-cache", "--fail-if-costlier", "0", "--share",
    )
    # The gate could not judge a cold vs warm pair: exit 5 with a warning.
    assert code == 5, out
    assert "WARNING: the baseline ran a different workload (prompt_cache_mode)" in out
    # The terminal names the mode change, as the shared summary does.
    assert "  What changed in the config:\n    workload.prompt_cache_mode: cold -> warm\n" in out
    assert "nothing Throttle can see" not in out
    assert "Verdict: NO WINNER, the workload differs (prompt_cache_mode)" in out
    assert "The prompt cache mode changed (cold -> warm)" in out
    assert "Add or drop --warm-cache to match" in out
    assert "change  not computed: the two checks ran different workloads" in out
    assert "CHEAPER" not in out
    summary = share_block(out)
    assert "- **Verdict: NO WINNER**, the workload differs (prompt_cache_mode)" in summary
    assert "- `workload.prompt_cache_mode`: cold -> warm" in summary
    assert "| Prompt cache | warm (identical prompts repeated) |" in summary


def test_pre_041_checks_count_as_warm_cache(history, monkeypatch, capsys):
    run_check(monkeypatch, capsys, FakeServer(constant(0.01)))
    path = history / "checks.ndjson"
    legacy = records(history)
    for record in legacy:
        for key in ("prompt_cache_mode", "prompt_nonce", "sent_prompts_sha256"):
            record["workload"].pop(key)
    path.write_text("".join(json.dumps(r) + "\n" for r in legacy))

    code, out = run_check(monkeypatch, capsys, FakeServer(constant(0.01)))
    assert code == 0, out
    assert "The prompt cache mode changed (warm -> cold (checks before Throttle 0.4.1" in out
    code, out = run_check(monkeypatch, capsys, FakeServer(constant(0.01)), "--warm-cache",
                          "--against", legacy[0]["id"], "--no-save")
    assert code == 0, out
    assert "workload differs" not in out


# --------------------------------------------------------------------------
# --share
# --------------------------------------------------------------------------

PRIVATE_URL = "http://gpu-box.internal.example.com:8000"

def metrics(root: str, extra: str = "") -> str:
    return (
        "# TYPE vllm:model_config_info gauge\n"
        f'vllm:model_config_info{{model="/home/kush/models/llama",served_url="{root}",'
        f'hf_token="hf_AbCdEfGhIjKlMnOp12345",max_num_seqs="256"{extra}}} 1.0\n'
    )


def share_block(out: str) -> str:
    begin = out.index(check_module.SHARE_BEGIN) + len(check_module.SHARE_BEGIN) + 1
    return out[begin:out.index(check_module.SHARE_END)].rstrip("\n")


def share_url(out: str) -> str:
    links = re.findall(r"https://github\.com/\S+", out)
    assert len(links) == 1, links
    return links[0]


def assert_no_private_data(text: str) -> None:
    for leaked in (
        "://", "http", "127.0.0.1", "10.1.2.3", "10.0.0.5", "192.168.7.20",
        "gpu-box", "example.com", "internal", "localhost", ":8000",
        "hf_AbCdEf", "sk-live", "/home/kush", "ghp_",
    ):
        assert leaked not in text, leaked
    assert not re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text)


def test_share_summary_has_no_urls_hosts_ips_or_secrets(history, monkeypatch, capsys):
    base = ("--allow-insecure-http", "--config", "gpu=H100", "--config", "engine=vLLM 0.11")
    run_check(
        monkeypatch, capsys,
        FakeServer(constant(0.01), metrics_body=metrics("http://10.1.2.3:8000/v1")),
        *base, "--config", "note=sk-liveABCDEF1234567890abcdef", "--label", "before",
        url=PRIVATE_URL,
    )
    code, out = run_check(
        monkeypatch, capsys,
        FakeServer(constant(0.01), metrics_body=metrics("http://127.0.0.1:9000")),
        *base, "--config", "note=ghp_ZYXWVUTSRQ0987654321zyxw", "--config",
        "deploy_host=10.0.0.5", "--label", "after on 192.168.7.20", "--share",
        url=PRIVATE_URL,
    )
    assert code == 0, out
    summary = share_block(out)
    assert_no_private_data(summary)
    assert "[masked: looks like a secret]" in summary
    assert "| Engine | vLLM 0.11 (from --config) |" in summary
    assert "| GPU | H100 |" in summary
    assert "| GPU hourly rate | $2.00/hr (ASSUMED, user-supplied) |" in summary
    assert f"| Throttle version | {check_module.__version__} |" in summary
    assert "| Prompt cache | cold (unique prompt per request) |" in summary
    assert "per million output tokens (95% CI $" in summary
    assert "- before: $" in summary and "- after: $" in summary
    assert "- noise floor: NOT CALIBRATED (0 df from repeat checks, 2 needed)" in summary
    assert "**Verdict: NOT CALIBRATED**" in summary
    assert "- `config.deploy_host`: (not set) -> [host removed]" in summary
    assert "- `label`: before -> after on [host removed]" in summary

    url = share_url(out)
    parts = urlsplit(url)
    assert (parts.scheme, parts.netloc, parts.path) == (
        "https", "github.com", "/KushagraKanaujia/throttle/issues/new"
    )
    query = parse_qs(parts.query)
    assert query["template"] == ["share-results.yml"]
    assert query["summary"] == [summary]  # decodes back to the printed summary
    assert query["engine"] == ["vLLM 0.11 (from --config)"]
    assert query["gpu"] == ["H100"]
    assert query["model"] == ["mock-model"]
    assert query["verdict"] == ["NOT CALIBRATED"]
    for key in ("title", "engine", "gpu", "model", "verdict"):
        assert_no_private_data(query[key][0])
    assert len(url) <= check_module.MAX_SHARE_URL_CHARS
    assert "nothing is uploaded" in out


def test_share_masks_secret_named_keys_and_prefixless_random_tokens(
    history, monkeypatch, capsys
):
    # A server label whose NAME says secret (value looks harmless), and a
    # --config value that is a random token with no known key prefix.
    run_check(
        monkeypatch, capsys,
        FakeServer(constant(0.01), metrics_body=metrics("x", ',api_key="plainvalue1"')),
        "--config", "build=Zq8x7Rt2Lm9Pw4Kd6Vn3Yb5",
    )
    code, out = run_check(
        monkeypatch, capsys,
        FakeServer(constant(0.01), metrics_body=metrics("x", ',api_key="plainvalue2"')),
        "--config", "build=Hy3Nc7Qe1Wa8Ts5Ub2Jf6Gk", "--share",
    )
    assert code == 0, out
    summary = share_block(out)
    for leaked in ("plainvalue", "Zq8x7Rt2", "Hy3Nc7Qe"):
        assert leaked not in summary, leaked
    assert f"- `vllm.model_config.api_key`: {check_module.MASKED} -> {check_module.MASKED}" in summary
    assert f"- `config.build`: {check_module.MASKED} -> {check_module.MASKED}" in summary
    query = parse_qs(urlsplit(share_url(out)).query)
    assert "plainvalue" not in query["summary"][0] and "Hy3Nc7Qe" not in query["summary"][0]


ADVERSARIAL_CONFIG = {
    "upstream": "gpu01:443",
    "fallback": "gpu01:80",
    "peer": "::ffff:10.1.2.3",
    "router": "admin:s3cret@gpu-box",
    "proxy": "user:hunter2@10.0.0.5",
    "dns": "node7.cluster.k8s",
    "note": "token=abcdefghijklmnop",
    "billing": "retry with tok_live_abcdefghijklmnopqrstuvwxyz",
    "rack.node9.cluster.k8s": "1",
    "gpu": "H100",
    "weights": "meta-llama/Llama-3.1-8B-Instruct",
    "quant": "model.Q4.gguf",
}
ADVERSARIAL_LEAKS = (
    "hunter2", "s3cret", "admin", "gpu-box", "gpu01", "ffff", ".1.2.3", "10.0.0.5",
    "node7", "node9", "cluster", "k8s", ":443", ":80", "abcdefghijklmnop",
    "tok_live", "user:",
)


def test_share_scrubs_credentials_mapped_ips_bare_hosts_ports_and_free_text_tokens(
    history, monkeypatch, capsys
):
    config = [arg for key, value in ADVERSARIAL_CONFIG.items()
              for arg in ("--config", f"{key}={value}")]
    run_check(monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "plain")
    code, out = run_check(
        monkeypatch, capsys, FakeServer(constant(0.01)),
        *config, "--label", "user:hunter2@10.0.0.5 via tok_live_abcdefghijklmnopqrstuvwxyz",
    )
    assert code == 0, out
    check_id = records(history)[-1]["id"]

    code = main(["check", "--share-id", check_id])
    out = capsys.readouterr().out
    assert code == 0, out
    summary = share_block(out)
    query = parse_qs(urlsplit(share_url(out)).query)
    for text in (summary, *(value[0] for value in query.values())):
        for leaked in ADVERSARIAL_LEAKS:
            assert leaked not in text, (leaked, text)
    host, masked = check_module.HOST_REMOVED, check_module.MASKED
    assert f"| Label | {host} via {masked} |" in summary
    for key in ("upstream", "fallback", "peer", "router", "proxy", "dns"):
        assert f"- `config.{key}`: (not set) -> {host}" in summary, key
    assert f"- `config.note`: (not set) -> token={masked}" in summary
    assert f"- `config.billing`: (not set) -> retry with {masked}" in summary
    assert f"- `config.{host}`: (not set) -> 1" in summary
    # Ordinary settings, model names and file names are still readable.
    assert "- `config.gpu`: (not set) -> H100" in summary
    assert "- `config.weights`: (not set) -> meta-llama/Llama-3.1-8B-Instruct" in summary
    assert "- `config.quant`: (not set) -> model.Q4.gguf" in summary
    assert "| Model | mock-model |" in summary


def test_share_a_recorded_check_by_id_or_the_latest_without_traffic(
    history, monkeypatch, capsys
):
    run_check(monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "first-cfg")
    run_check(monkeypatch, capsys, FakeServer(constant(0.01)), "--label", "second-cfg")
    first, second = records(history)

    def no_traffic(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"--share of a recorded check sent {request.url}")

    monkeypatch.setattr(check_module, "_TRANSPORT", httpx.MockTransport(no_traffic))
    code = main(["check", "--history", "--share"])
    out = capsys.readouterr().out
    assert code == 0, out
    summary = share_block(out)
    assert f"| Check | {second['id']} " in summary
    assert f"**Compared with** check {first['id']}:" in summary
    assert "- `label`: first-cfg -> second-cfg" in summary
    assert "**Verdict: NOT CALIBRATED**" in summary
    assert parse_qs(urlsplit(share_url(out)).query)["summary"] == [summary]
    assert_no_private_data(summary)

    code = main(["check", "--share-id", first["id"]])
    out = capsys.readouterr().out
    assert code == 0, out
    summary = share_block(out)
    assert f"| Check | {first['id']} " in summary
    assert "**Compared with:** nothing (first check of this endpoint)." in summary
    assert parse_qs(urlsplit(share_url(out)).query)["verdict"] == ["first check (no comparison)"]

    code = main(["check", "--share-id", "no-such-check"])
    captured = capsys.readouterr()
    assert code == 2
    assert "no check with id 'no-such-check'" in captured.err


def test_share_link_is_truncated_gracefully_to_fit(history, monkeypatch, capsys):
    many = "".join(f',setting_{i:02d}="{"a" * 60}{i}"' for i in range(60))
    changed = "".join(f',setting_{i:02d}="{"b" * 60}{i}"' for i in range(60))
    run_check(monkeypatch, capsys, FakeServer(constant(0.01), metrics_body=metrics("x", many)))
    code, out = run_check(
        monkeypatch, capsys,
        FakeServer(constant(0.01), metrics_body=metrics("x", changed)), "--share",
    )
    assert code == 0, out
    summary = share_block(out)
    assert summary.count("vllm.model_config.setting_") == 60  # the terminal gets it all
    url = share_url(out)
    assert len(url) <= check_module.MAX_SHARE_URL_CHARS
    sent = parse_qs(urlsplit(url).query)["summary"][0]
    assert sent.endswith(check_module.SHARE_TRUNCATED_NOTE)
    kept = sent[: -len(check_module.SHARE_TRUNCATED_NOTE)].rstrip()
    assert kept and summary.startswith(kept)
    assert "the summary was cut to keep the link under 7,000 characters" in out


# --------------------------------------------------------------------------
# cost and measure use the same default
# --------------------------------------------------------------------------


class ContentRecorder:
    """Minimal OpenAI-compatible server that keeps each chat prompt."""

    def __init__(self) -> None:
        self.contents: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        body = json.loads(request.content)
        self.contents.append(body["messages"][0]["content"])
        return httpx.Response(200, json={
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        })


@pytest.fixture
def recorder(monkeypatch, tmp_path) -> ContentRecorder:
    server = ContentRecorder()
    real_client, real_async = httpx.Client, httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(server.handle)
        return real_client(*args, **kwargs)

    def async_client(*args, **kwargs):
        async def handler(request):
            return server.handle(request)

        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setattr(httpx, "AsyncClient", async_client)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    return server


COST = ("cost", "--url", "http://127.0.0.1:8000", "--model", "m",
        "--gpu-hourly-rate", "1.5", "--num-requests", "6")


def test_cost_tags_each_prompt_by_default_and_warm_cache_does_not(recorder, capsys):
    assert main(list(COST)) == 0
    out = capsys.readouterr().out
    assert "Prompt cache: COLD (default)" in out
    tags = [TAG.match(text) for text in recorder.contents]
    assert all(tags) and len({t.group(1) for t in tags}) == 1
    assert len(set(recorder.contents)) == len(recorder.contents) == 6

    recorder.contents.clear()
    assert main([*COST, "--warm-cache"]) == 0
    out = capsys.readouterr().out
    assert "Prompt cache: WARM (--warm-cache)" in out
    assert all(text.startswith("Test ") for text in recorder.contents)


MEASURE = ("measure", "--url", "http://127.0.0.1:8000", "--model", "m",
           "--gpu-hourly-rate", "1.5", "--repeat", "2", "--num-requests", "3",
           "--arrival-rate", "1000")


def test_measure_records_the_mode_and_compare_refuses_cold_vs_warm(recorder, capsys, tmp_path):
    assert main([*MEASURE, "--label", "cold-run"]) == 0
    probe, *measured = recorder.contents  # the connectivity probe sends "test"
    assert probe == "test"
    assert len(set(measured)) == len(measured) == 6 and all(TAG.match(t) for t in measured)
    assert main([*MEASURE, "--label", "warm-run", "--warm-cache"]) == 0
    capsys.readouterr()
    cold = json.loads((tmp_path / "cold-run.json").read_text())
    warm = json.loads((tmp_path / "warm-run.json").read_text())
    assert cold["workload"]["prompt_cache_mode"] == "cold"
    assert TAG.match(measured[0]).group(1) == cold["workload"]["prompt_nonce"]["run_id"]
    assert warm["workload"]["prompt_cache_mode"] == "warm"

    code = main(["compare", "cold-run.json", "warm-run.json"])
    out = capsys.readouterr().out
    assert code == 0
    assert "NO WINNER: these measurements used different prompt cache modes:" in out
    assert "cold-run.json (label cold-run): cold" in out
    assert "warm-run.json (label warm-run): warm" in out
    assert "Ranked by" not in out and "NO SIGNIFICANT DIFFERENCE" not in out


def test_compare_refuses_cold_vs_warm_even_when_the_labels_match(recorder, capsys, tmp_path):
    # measure names its file after the label, so move each run aside.
    assert main([*MEASURE, "--label", "baseline"]) == 0
    (tmp_path / "baseline.json").rename(tmp_path / "cold.json")
    assert main([*MEASURE, "--label", "baseline", "--warm-cache"]) == 0
    (tmp_path / "baseline.json").rename(tmp_path / "warm.json")
    capsys.readouterr()
    assert json.loads((tmp_path / "cold.json").read_text())["label"] == "baseline"
    assert json.loads((tmp_path / "warm.json").read_text())["label"] == "baseline"

    code = main(["compare", "cold.json", "warm.json"])
    out = capsys.readouterr().out
    assert code == 0
    assert "NO WINNER: these measurements used different prompt cache modes:" in out
    assert "cold.json (label baseline): cold" in out
    assert "warm.json (label baseline): warm" in out
    assert "Ranked by" not in out and "NO SIGNIFICANT DIFFERENCE" not in out
