"""`throttle scan`: the offline first run (find, price, suggest, never claim)."""

from __future__ import annotations

import json
import math
import socket
from pathlib import Path

import pytest

from throttle.cli import GETTING_STARTED, HELP_ORDER, build_parser, main
from throttle.own import pricing, scan, scan_cost
from throttle.own.scan_cost import (
    MIN_CUT, SCHEMA, STATUS_PRICED, STATUS_RUNTIME, STATUS_UNKNOWN, Shape, WHY_FAMILY, WHY_OTHER,
    WHY_OTHER_ALONE, WHY_PROVIDER, family, per_call_usd, pick_candidates,
)

REPO = Path(__file__).resolve().parents[1]

# A small app with Python and JS/TS calls: two priced OpenAI calls (one with a
# literal max_tokens under the assumed output), two priced Anthropic calls, a
# model set at runtime and a model missing from the price table.
FIXTURE = {
    "app/summarize.py": '''from openai import OpenAI

client = OpenAI()


def summarize(text: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=200,
        messages=[
            {"role": "system", "content": "Summarize the support ticket in two sentences."},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content
''',
    "app/classify.py": '''import anthropic

client = anthropic.Anthropic()


def classify(ticket: str) -> str:
    # throttle-job: ticket-triage
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system="Label the ticket as billing, bug or other. Reply with the label only.",
        messages=[{"role": "user", "content": ticket}],
    )
    return msg.content[0].text
''',
    "app/router.py": '''import os

from openai import OpenAI

client = OpenAI()


def answer(question: str) -> str:
    resp = client.chat.completions.create(
        model=os.environ["CHAT_MODEL"],
        messages=[{"role": "user", "content": question}],
    )
    return resp.choices[0].message.content
''',
    "web/src/extract.ts": '''import OpenAI from "openai";

const client = new OpenAI();

export async function extractInvoice(text: string) {
  const res = await client.responses.create({
    model: "gpt-4.1",
    instructions: "Extract the invoice number, total and due date as JSON.",
    input: text,
  });
  return res.output_text;
}

export async function draftReply(text: string) {
  const res = await client.chat.completions.create({
    model: "acme-finetune-v2",
    messages: [{ role: "user", content: text }],
  });
  return res.choices[0].message.content;
}
''',
    "web/src/chat.js": '''import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic();

export async function chat(history) {
  const msg = await anthropic.messages.create({
    model: "claude-opus-4-5",
    max_tokens: 2048,
    messages: history,
  });
  return msg.content[0].text;
}
''',
}


def write_fixture(root: Path) -> Path:
    for rel, source in FIXTURE.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


def price(model: str, provider: str = "openai", inp: float = 1.0, out: float = 4.0, **extra) -> pricing.Price:
    return pricing.Price(provider=provider, model=model, input_per_m=inp, output_per_m=out,
                         cached_input_per_m=None, as_of="2026-10-27", source="test", **extra)


@pytest.fixture
def plain(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.delenv("FORCE_COLOR", raising=False)


def flatten(out: str) -> str:
    """The output as one line of words, without the panel frame."""
    lines = []
    for line in out.splitlines():
        if line[:1] in "│|" and line[-1:] in "│|":
            line = line[1:-1]
        lines.append(line)
    return " ".join(" ".join(lines).split())


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(["scan", *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model,provider,expected", [
    ("gpt-4o-mini", "openai", "gpt-4o"),
    ("gpt-5-pro", "openai", "gpt-5"),
    ("gpt-5.4-nano", "openai", "gpt-5.4"),
    ("o3-mini", "openai", "o3"),
    ("gpt-6-astra", "openai", "gpt-6"),
    ("claude-haiku-4-5", "anthropic", "4-5"),
    ("claude-opus-4-5", "anthropic", "4-5"),
    ("claude-sonnet-5", "anthropic", "5"),
    ("plain", "openai", "plain"),
])
def test_family_is_read_from_the_name(model, provider, expected) -> None:
    assert family(price(model, provider)) == (provider, expected)


def test_first_pick_is_one_tier_down_in_the_family_then_other_provider_at_a_similar_price() -> None:
    table = [
        price("big-9", inp=10, out=40),
        price("big-9-mid", inp=3, out=12),
        price("big-9-small", inp=0.5, out=2),
        price("other-1", inp=2, out=8),                       # same provider, not in the family
        price("claude-top-9", "anthropic", inp=20, out=80),  # dearer: never suggested
        price("claude-mid-9", "anthropic", inp=3, out=12),
        price("claude-low-9", "anthropic", inp=0.4, out=1.6),
    ]
    picks = pick_candidates(table[0], table, 1000, 300)
    assert [(c.price.model, c.why) for c in picks] == [("big-9-mid", WHY_FAMILY), ("claude-mid-9", WHY_OTHER)]
    assert all(c.per_call_usd < per_call_usd(table[0], 1000, 300) for c in picks)


def test_no_cheaper_tier_in_family_falls_back_to_next_price_step_of_the_provider() -> None:
    table = [
        price("solo-3", inp=4, out=16),
        price("other-2", inp=3, out=12),
        price("other-1", inp=1, out=4),
        price("claude-mid-9", "anthropic", inp=3, out=12),
    ]
    picks = pick_candidates(table[0], table, 1000, 300)
    assert [(c.price.model, c.why) for c in picks] == [("other-2", WHY_PROVIDER), ("claude-mid-9", WHY_OTHER)]


def test_cheapest_model_gets_only_a_cheaper_other_provider_model_or_nothing() -> None:
    table = [price("tiny-1", inp=1, out=4), price("claude-x-1", "anthropic", inp=0.5, out=2)]
    picks = pick_candidates(table[0], table, 1000, 300)
    assert [(c.price.model, c.why) for c in picks] == [("claude-x-1", WHY_OTHER_ALONE)]
    assert pick_candidates(table[1], table, 1000, 300) == []


def test_pool_skips_snapshots_specialized_models_and_tiny_savings() -> None:
    now = price("big-9", inp=10, out=40)
    table = [
        now,
        price("big-9-2024-05-13", inp=1, out=4),         # dated snapshot
        price("big-9-codex", inp=1, out=4),              # built for code only
        price("big-9-nearly", inp=9.5, out=38),          # only 5% cheaper
    ]
    assert pick_candidates(now, table, 1000, 300) == []
    cheaper_enough = price("big-9-ok", inp=10 * (1 - MIN_CUT), out=40 * (1 - MIN_CUT))
    assert [c.price.model for c in pick_candidates(now, [*table, cheaper_enough], 1000, 300)] == ["big-9-ok"]


def test_ties_go_to_the_entry_listed_first_and_limit_is_respected() -> None:
    now = price("claude-top-9", "anthropic", inp=10, out=40)
    table = [now, price("claude-a-9", "anthropic", inp=2, out=8), price("claude-b-9", "anthropic", inp=2, out=8),
             price("x-1", inp=2, out=8), price("y-1", inp=2, out=8)]
    picks = pick_candidates(now, table, 1000, 300)
    assert [c.price.model for c in picks] == ["claude-a-9", "x-1"]
    assert [c.price.model for c in pick_candidates(now, table, 1000, 300, limit=1)] == ["claude-a-9"]
    assert pick_candidates(now, table, 1000, 300, limit=0) == []


def test_picks_depend_on_the_token_shape() -> None:
    # cheap input, dear output vs the reverse: which is cheaper flips with the shape
    now = price("base-1", inp=5, out=5)
    out_heavy = price("base-1-a", inp=1, out=6)
    in_heavy = price("base-1-b", inp=6, out=1)
    table = [now, out_heavy, in_heavy]
    assert [c.price.model for c in pick_candidates(now, table, 10_000, 10)] == ["base-1-a"]
    assert [c.price.model for c in pick_candidates(now, table, 10, 10_000)] == ["base-1-b"]


def test_bundled_table_picks_are_cheaper_and_same_provider_first() -> None:
    table = pricing.load_prices()
    for entry in table:
        picks = pick_candidates(entry, table, 1000, 300)
        now = per_call_usd(entry, 1000, 300)
        assert len(picks) <= 2
        for pick in picks:
            assert pick.per_call_usd <= now * (1 - MIN_CUT) + 1e-15, (entry.model, pick.price.model)
            assert pick.price.model != entry.model
        if picks and picks[0].why in (WHY_FAMILY, WHY_PROVIDER):
            assert picks[0].price.provider == entry.provider
    gpt4o = table.lookup("openai", "gpt-4o")
    assert pick_candidates(gpt4o, table, 1000, 300)[0].price.model == "gpt-4o-mini"
    sonnet = table.lookup("anthropic", "claude-sonnet-4-5")
    assert pick_candidates(sonnet, table, 1000, 300)[0].price.model == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Pricing math
# ---------------------------------------------------------------------------


def test_per_call_cost_is_list_price_times_tokens() -> None:
    p = price("m-1", inp=2.5, out=10)
    assert per_call_usd(p, 1000, 300) == pytest.approx((2.5 * 1000 + 10 * 300) / 1_000_000)


def test_long_prompt_over_an_unpriced_threshold_is_not_priced() -> None:
    p = price("m-1", inp=2.5, out=10, long_context=pricing.LongContext(above_input_tokens=5000))
    assert per_call_usd(p, 6000, 300) is None
    site = scan.CallSite(path="a.py", line=1, function="f", language="python", provider="openai",
                         api="chat.completions", model="m-1")
    job = scan_cost.cost_job(site, pricing.PriceTable([p]), Shape(tokens_in=6000))
    assert job.status == scan_cost.STATUS_UNPRICED and job.per_call_usd is None


def test_max_tokens_caps_output_and_monthly_uses_calls_per_day() -> None:
    table = pricing.PriceTable([price("m-1", inp=2, out=10), price("m-1-mini", inp=0.2, out=1)])
    site = scan.CallSite(path="a.py", line=3, function="f", language="python", provider="openai",
                         api="chat.completions", model="m-1", max_tokens=100)
    shape = Shape(tokens_in=1000, tokens_out=300, calls_per_day=1000)
    job = scan_cost.cost_job(site, table, shape)
    assert job.capped and job.tokens_out == 100
    assert job.per_call_usd == pytest.approx((2 * 1000 + 10 * 100) / 1e6)
    summary = scan_cost.summarize([job], shape)
    assert summary["monthly_usd"] == pytest.approx(job.per_call_usd * 1000 * 30)
    assert summary["per_1k_calls_usd"] == pytest.approx(job.per_call_usd * 1000)
    assert summary["cut_pct"] == pytest.approx(90.0)
    # a cap above the assumed output changes nothing
    site.max_tokens = 4096
    assert not scan_cost.cost_job(site, table, shape).capped


def test_summary_cut_counts_jobs_without_a_candidate_at_their_own_cost() -> None:
    table = pricing.PriceTable([price("m-1", inp=2, out=10), price("m-1-mini", inp=0.2, out=1),
                                price("solo", inp=0.1, out=0.5)])
    sites = [scan.CallSite(path="a.py", line=i, function=f"f{i}", language="python", provider="openai",
                           api="chat.completions", model=m) for i, m in enumerate(["m-1", "solo"], 1)]
    shape = Shape()
    jobs = [scan_cost.cost_job(s, table, shape) for s in sites]
    assert jobs[1].candidates == []
    now = jobs[0].per_call_usd + jobs[1].per_call_usd
    after = jobs[0].candidates[0].per_call_usd + jobs[1].per_call_usd
    assert scan_cost.summarize(jobs, shape)["cut_pct"] == round((1 - after / now) * 100, 1)


def test_litellm_style_and_unknown_provider_names_are_looked_up() -> None:
    table = pricing.load_prices()
    for provider, model in (("unknown", "claude-haiku-4-5"), ("openai", "anthropic/claude-haiku-4-5"),
                            ("unknown", "gpt-4o-mini")):
        site = scan.CallSite(path="a.py", line=1, function="f", language="python", provider=provider,
                             api="litellm", model=model)
        assert scan_cost.cost_job(site, table, Shape()).status == STATUS_PRICED, model


# ---------------------------------------------------------------------------
# Scanner additions: literal max_tokens, pruned walk
# ---------------------------------------------------------------------------


def test_scanner_reads_literal_max_tokens_in_python_and_js_but_not_into_identity() -> None:
    py = scan.scan_python("a.py", 'LIMIT = 64\nclient.chat.completions.create(model="gpt-4o", messages=[], '
                                  'max_tokens=LIMIT)\nclient.chat.completions.create(model="gpt-4o", messages=[], '
                                  'max_tokens=n)\n')
    assert [s.max_tokens for s in py] == [64, None]
    js = scan.scan_javascript("a.ts", 'await client.messages.create({ model: "claude-haiku-4-5", '
                                      'max_tokens: 1_024, messages: [] });\n'
                                      'await generateText({ model: openai("gpt-4o"), maxOutputTokens: 50, prompt: "hi" });\n')
    assert [s.max_tokens for s in js] == [1024, 50]
    assert "max_tokens" not in py[0].to_json()
    assert scan.CallSite.from_json(py[0].to_json()) == py[0]


def test_scan_dir_never_enters_skipped_folders(tmp_path) -> None:
    write_fixture(tmp_path)
    for skipped in ("node_modules/pkg", ".venv/lib", "dist"):
        target = tmp_path / skipped / "x.py"
        target.parent.mkdir(parents=True)
        target.write_text('client.chat.completions.create(model="gpt-4o", messages=[])\n')
    found = {s.path for s in scan.scan_dir(tmp_path)}
    assert found == {"app/summarize.py", "app/classify.py", "app/router.py", "web/src/extract.ts",
                     "web/src/chat.js"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argv,needle", [
    (["--tokens-in", "0"], "greater than zero"),
    (["--tokens-in", "-5"], "greater than zero"),
    (["--tokens-out", "lots"], "whole number"),
    (["--tokens-out", "20000001"], "at most"),
    (["--calls-per-day", "0"], "greater than zero"),
    (["--calls-per-day", "1.5"], "whole number"),
])
def test_bad_flags_are_usage_errors(capsys, argv, needle) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["scan", *argv])
    assert exc.value.code == 2
    assert needle in capsys.readouterr().err


def test_good_flags_parse_with_commas() -> None:
    args = build_parser().parse_args(["scan", "src", "--tokens-in", "1,500", "--tokens-out", "200",
                                      "--calls-per-day", "10_000", "--json"])
    assert (args.path, args.tokens_in, args.tokens_out, args.calls_per_day, args.json) == ("src", 1500, 200,
                                                                                           10000, True)


def test_missing_folder_is_a_usage_error(tmp_path, capsys, plain) -> None:
    code, out, err = run(capsys, str(tmp_path / "nope"))
    assert code == 2 and "is not a folder" in err and out == ""


def test_empty_repo_says_what_scan_looks_for(tmp_path, capsys, plain, monkeypatch) -> None:
    (tmp_path / "main.py").write_text("print('no AI here')\n")
    monkeypatch.chdir(tmp_path)
    code, out, _ = run(capsys, ".")
    assert code == 0
    flat = flatten(out)
    assert f"No OpenAI or Anthropic calls found in {tmp_path.name}." in flat
    assert "chat.completions.create" in flat and "generateText" in flat and "node_modules" in flat
    assert "throttle demo" in flat
    code, out, _ = run(capsys, str(tmp_path), "--json")
    data = json.loads(out)
    assert data["summary"]["calls_found"] == 0 and data["jobs"] == [] and data["next_steps"] == []


def test_runtime_and_unknown_models_are_listed_not_priced(tmp_path, capsys, plain) -> None:
    write_fixture(tmp_path)
    code, out, _ = run(capsys, str(tmp_path), "--json")
    assert code == 0
    jobs = {j["function"]: j for j in json.loads(out)["jobs"]}
    runtime = jobs["answer"]
    assert runtime["status"] == STATUS_RUNTIME and runtime["model"] is None
    assert runtime["model_source"] == "runtime" and runtime["model_expr"] == "os.environ['CHAT_MODEL']"
    assert runtime["per_call_usd"] is None and runtime["candidates"] == []
    unknown = jobs["draftReply"]
    assert unknown["status"] == STATUS_UNKNOWN and unknown["model"] == "acme-finetune-v2"
    assert unknown["price"] is None and unknown["candidates"] == []
    code, out, _ = run(capsys, str(tmp_path))
    flat = flatten(out)
    assert "Not priced (2):" in flat
    assert "app/router.py:9 model set at runtime (os.environ['CHAT_MODEL'])" in flat
    assert "acme-finetune-v2 is not in the price table; add it with --prices FILE" in flat


def test_prices_file_prices_an_unknown_model_as_assumed(tmp_path, capsys, plain) -> None:
    app = write_fixture(tmp_path / "app")
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps([{"provider": "openai", "model": "acme-finetune-v2",
                                   "input_per_m": 3, "output_per_m": 12}]))
    code, out, _ = run(capsys, str(app), "--json", "--prices", str(prices))
    assert code == 0
    job = next(j for j in json.loads(out)["jobs"] if j["model"] == "acme-finetune-v2")
    assert job["status"] == STATUS_PRICED and job["price"]["label"].startswith("ASSUMED")
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    code, _, err = run(capsys, str(app), "--prices", str(bad))
    assert code == 2 and "not valid JSON" in err


def test_json_schema_is_stable(tmp_path, capsys, plain) -> None:
    write_fixture(tmp_path)
    code, out, _ = run(capsys, str(tmp_path), "--json", "--calls-per-day", "2000")
    assert code == 0
    data = json.loads(out)
    assert data["schema"] == SCHEMA == "throttle.own.scan/1"
    assert set(data) == {"schema", "path", "label", "assumptions", "summary", "jobs", "next_steps"}
    assert data["label"] == "ESTIMATE" and data["summary"]["label"] == "ESTIMATE"
    a = data["assumptions"]
    assert (a["tokens_in"], a["tokens_in_source"], a["tokens_out"], a["tokens_out_source"]) == (
        1000, "ASSUMED", 300, "ASSUMED")
    assert (a["calls_per_day"], a["calls_per_day_source"], a["days_per_month"], a["quality"]) == (
        2000, "flag", 30, "UNTESTED")
    assert a["price_labels"] and all(label.startswith("LIST PRICE (") for label in a["price_labels"])
    s = data["summary"]
    assert (s["calls_found"], s["priced"], s["not_priced"], s["with_candidate"]) == (6, 4, 2, 4)
    job_keys = {"job", "path", "line", "function", "language", "provider", "api", "model", "model_expr",
                "model_source", "status", "tokens_in", "tokens_out", "max_tokens",
                "tokens_out_capped_by_max_tokens", "price", "per_call_usd", "per_1k_calls_usd", "monthly_usd",
                "candidates"}
    for job in data["jobs"]:
        assert set(job) == job_keys
        for c in job["candidates"]:
            assert set(c) == {"model", "provider", "status", "why", "price", "per_call_usd", "per_1k_calls_usd",
                              "monthly_usd", "cut_pct"}
            assert c["status"] == "UNTESTED" and c["per_call_usd"] < job["per_call_usd"]
            assert c["monthly_usd"] == pytest.approx(c["per_call_usd"] * 2000 * 30, rel=1e-6)
    summarize = next(j for j in data["jobs"] if j["function"] == "summarize")
    assert summarize["job"] == "app/summarize.py::summarize" and summarize["max_tokens"] == 200
    assert summarize["tokens_out"] == 200 and summarize["tokens_out_capped_by_max_tokens"] is True
    assert summarize["per_1k_calls_usd"] == pytest.approx((2.5 * 1000 + 10 * 200) / 1000)
    assert next(j for j in data["jobs"] if j["function"] == "classify")["job"] == "ticket-triage"
    total = sum(j["monthly_usd"] for j in data["jobs"] if j["monthly_usd"] is not None)
    assert s["monthly_usd"] == pytest.approx(total)
    assert math.isclose(s["cut_pct"], round((1 - s["first_candidates_per_1k_calls_usd"]
                                             / s["per_1k_calls_usd"]) * 100, 1), abs_tol=0.05)


def test_end_to_end_on_a_python_and_js_repo_offline_and_writes_nothing(tmp_path, capsys, plain,
                                                                        monkeypatch) -> None:
    repo = write_fixture(tmp_path / "repo")
    data_dir = tmp_path / "throttle-data"
    monkeypatch.setenv("THROTTLE_OWN_DIR", str(data_dir))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(repo)

    def no_network(*_a, **_k):
        raise AssertionError("throttle scan must not use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    code, out, err = run(capsys, ".")
    after = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    assert code == 0 and err == ""
    assert before == after and not data_dir.exists()

    lines = out.splitlines()
    assert lines[0].startswith("┌─ Throttle scan: repo ") or lines[0].startswith("+- Throttle scan: repo ")
    assert all(len(line) <= 80 for line in lines), [line for line in lines if len(line) > 80]
    flat = flatten(out)
    for needed in ("JOB", "MODEL", "$ / 1K CALLS", "CHANGE",
                   "ticket-triage", "app/classify.py:8", "claude-sonnet-4-5",
                   "summarize", "app/summarize.py:7", "gpt-4o*", "gpt-4o-mini",
                   "extractInvoice", "web/src/extract.ts:6", "gpt-4.1",
                   "chat", "web/src/chat.js:6", "claude-opus-4-5",
                   "cheaper at list price, UNTESTED on your traffic.",
                   "* output capped by a literal max_tokens in the code.",
                   "6 AI calls found, 4 priced; moving the priced ones to the first candidate could cut about",
                   "of their list-price cost, ESTIMATE.",
                   "Per call: 1,000 tokens in, 300 out, ASSUMED.",
                   "Add --calls-per-day N for a monthly figure.",
                   "Quality is not tested here.",
                   'Prove it on your own traffic (see README "The loop")',
                   "throttle capture", "export OPENAI_BASE_URL=http://127.0.0.1:8790/v1",
                   "throttle shadow --candidate-url https://api.openai.com/v1 \\ --api-key-env OPENAI_API_KEY \\ "
                   "--candidate-model gpt-4o-mini --label gpt-4o-mini",
                   "throttle eval --candidate gpt-4o-mini \\ --judge-model gpt-4.1 --judge-key-env OPENAI_API_KEY",
                   "throttle report --candidate gpt-4o-mini --candidate-model gpt-4o-mini"):
        assert needed in flat, needed
    # never a quality claim, never an em dash
    lowered = flat.lower()
    assert "as good" not in lowered and "just as well" not in lowered and "\u2014" not in out


def test_monthly_output_and_given_tokens_are_labelled(tmp_path, capsys, plain) -> None:
    write_fixture(tmp_path)
    code, out, _ = run(capsys, str(tmp_path), "--calls-per-day", "2000", "--tokens-in", "500")
    assert code == 0
    flat = flatten(out)
    assert "$ / MONTH" in flat
    assert "At 2,000 calls a day per job: about $" in flat and "/mo on the first candidates (ESTIMATE)" in flat
    assert "Per call: 500 tokens in (your flag), 300 out (ASSUMED)." in flat
    assert "--calls-per-day 2,000 x 30 days, the same for every job." in flat
    assert "Add --calls-per-day" not in flat


def test_scan_is_first_in_help_welcome_and_readme() -> None:
    assert HELP_ORDER[0] == "scan"
    assert GETTING_STARTED.index("throttle scan .") < GETTING_STARTED.index("throttle demo")
    from throttle.cli import WELCOME_STEPS
    assert WELCOME_STEPS[0][2] == ("throttle scan .",)
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    start = readme.index("## Try it in a minute")
    block = readme[readme.index("```sh", start):readme.index("```\n", readme.index("```sh", start) + 5)]
    assert block.splitlines()[1:] == ['pip install "throttle-pro==0.7.0"', "throttle scan .", "throttle demo"]
    assert "throttle.own.scan/1" in readme
