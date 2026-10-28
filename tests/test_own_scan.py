"""Finding AI call sites in Python and JS/TS (throttle.own.scan). Offline."""

from __future__ import annotations

import subprocess

import pytest

from throttle.own import scan
from throttle.own.scan import Template


def one(sites, job):
    found = [s for s in sites if s.job == job]
    assert found, [s.job for s in sites]
    return found[0]


PY = '''
import textwrap
import litellm
from litellm import acompletion
from openai import OpenAI
from langchain_openai import ChatOpenAI

client = OpenAI()
SYSTEM = """You are a support agent.
Summarize the ticket in two sentences."""
MODEL = "gpt-4o-mini"


def summarize(ticket):
    return client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Ticket: {ticket.body}"}],
    )


class Bot:
    async def reply(self, q):
        prompt = textwrap.dedent("""
            Answer the customer politely and briefly: {q}
        """).format(q=q)
        # throttle-job: polite-reply
        return await self.client.messages.create(model="claude-sonnet-4-5", max_tokens=300,
            system="You are kind and very brief.", messages=[{"role": "user", "content": prompt}])


def many(x, model_name):
    a = client.responses.create(model=model_name, instructions="Classify the sentiment as positive or negative.",
                                input=x, extra_headers={"X-Throttle-Job": "sentiment"})
    t = client.beta.threads.messages.create(thread_id="t", role="user", content="hi")
    b = litellm.completion(model="claude-3-haiku", messages=[{"role": "user", "content": "Translate to French: " + x}])
    c = client.chat.completions.create(model="gpt-4.1", messages=[{"role": "user", "content": x}],
                                       tools=[{"type": "function"}], response_format={"type": "json_object"})
    d = client.chat.completions.create(model="gpt-4.1", messages=history)
    e = client.completions.create(model="gpt-3.5-turbo-instruct", prompt="Say %s to %s" % (x, x))
    f = ChatOpenAI(model="gpt-4o")
    g = acompletion(model="gpt-4o", messages=[{"role": "user", "content": x}])
    h = client.chat.completions.create(model="gpt-4.1", messages=[{"role": "user", "content": x}],
                                       extra_headers=throttle_job("tagged-helper"))
    return a, t, b, c, d, e, f, g, h
'''


def test_python_call_sites_and_prompts():
    sites = scan.scan_files({"app/support.py": PY})
    jobs = [s.job for s in sites]
    # the Assistants API threads.messages.create is not a model call; tagged
    # calls take their tag as the job id and do not use up an ordinal
    assert len(sites) == 10, jobs
    s = one(sites, "app/support.py::summarize")
    assert (s.provider, s.api, s.model, s.line) == ("openai", "chat.completions", "gpt-4o-mini", 15)
    assert s.system.text.startswith("You are a support agent.")
    assert s.user.text == "Ticket: {ticket.body}"
    assert s.prompt_fingerprint and not s.prompt_dynamic

    bot = one(sites, "polite-reply")                       # comment tag
    assert (bot.provider, bot.api, bot.function) == ("anthropic", "messages", "Bot.reply")
    assert bot.system.text == "You are kind and very brief."
    assert "Answer the customer politely and briefly: {q}" in bot.user.text

    sentiment = one(sites, "sentiment")                   # header tag
    assert sentiment.api == "responses" and sentiment.model is None and sentiment.model_expr == "model_name"
    assert sentiment.system.text.startswith("Classify the sentiment")

    lite = one(sites, "app/support.py::many")
    assert (lite.api, lite.provider, lite.model) == ("litellm", "anthropic", "claude-3-haiku")
    assert lite.user.text == "Translate to French: {x}"

    tools = one(sites, "app/support.py::many#2")
    assert tools.has_tools and tools.wants_json and tools.prompt_fingerprint is None   # user text is all input
    assert one(sites, "app/support.py::many#3").prompt_dynamic                          # messages=history
    legacy = one(sites, "app/support.py::many#4")
    assert legacy.api == "completions" and legacy.user.text == "Say {arg1} to {arg2}"
    lang = one(sites, "app/support.py::many#5")
    assert (lang.api, lang.model, lang.prompt_dynamic) == ("langchain", "gpt-4o", True)
    assert one(sites, "app/support.py::many#6").api == "litellm"
    assert one(sites, "tagged-helper").model == "gpt-4.1"


def test_unparseable_python_is_skipped():
    assert scan.scan_files({"bad.py": "def (:\n  client.chat.completions.create("}) == []


def test_fingerprint_ignores_whitespace_and_slot_names_but_not_words():
    base = 'def f(x):\n    return c.chat.completions.create(model="m", messages=[{"role": "system", "content": "Summarize   this ticket now."}, {"role": "user", "content": f"T: {x}"}])\n'
    spaced = base.replace("Summarize   this", "Summarize \t this").replace("{x}", "{y}")
    edited = base.replace("Summarize", "Translate")
    fp = lambda src: scan.scan_files({"a.py": src})[0].prompt_fingerprint  # noqa: E731
    assert fp(base) == fp(spaced)
    assert fp(base) != fp(edited)


JS = '''
import OpenAI from "openai";
const client = new OpenAI();
const SYSTEM = "You extract invoice fields as JSON.";
export async function extractInvoice(text: string): Promise<string> {
  const r = await client.chat.completions.create({
    model: "gpt-4o-mini",
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: SYSTEM },
      { role: "user", content: `Invoice:\\n${text}` },
    ],
  }, { headers: { "X-Throttle-Job": "invoice" } });
  return r.choices[0].message.content;
}
class Svc {
  async tagIt(t) {
    // a comment with a quote ' and a bracket ( in it
    return this.anthropic.messages.create({ model: "claude-haiku-4-5", max_tokens: 50, system: "Tag this text with one word.", messages: [{role: "user", content: t}] });
  }
}
const summarize = async (doc, n) => {
  return generateText({ model: openai("gpt-4.1-mini"), system: `Summarize in ${n} bullet points for an executive.`, prompt: doc });
};
function noModel() {
  return thread.messages.create({ role: "user", content: "hi" });
}
const odd = "client.chat.completions.create({ model: 'not-a-call' })";
'''


def test_javascript_call_sites():
    sites = scan.scan_files({"web/invoice.ts": JS})
    assert [s.job for s in sites] == ["invoice", "web/invoice.ts::Svc.tagIt", "web/invoice.ts::summarize"]
    inv = sites[0]
    assert (inv.function, inv.model, inv.wants_json, inv.line) == ("extractInvoice", "gpt-4o-mini", True, 6)
    assert inv.system.text == "You extract invoice fields as JSON."
    assert inv.user.text == "Invoice:\n{text}"
    tag = sites[1]
    assert (tag.provider, tag.api, tag.model) == ("anthropic", "messages", "claude-haiku-4-5")
    assert tag.system.text == "Tag this text with one word."
    sdk = sites[2]
    assert (sdk.api, sdk.provider, sdk.model) == ("ai-sdk", "openai", "gpt-4.1-mini")
    assert sdk.system.text == "Summarize in {n} bullet points for an executive."
    assert sdk.user.text == "{doc}"


def test_javascript_concatenation_comment_tag_and_responses():
    src = '''
const PREFIX = "You are a careful translator. ";
// throttle-job: translate
const out = await client.responses.create({ model: MODEL, instructions: PREFIX + "Translate to " + lang + ".", input: text });
'''
    site = scan.scan_files({"t.js": src})[0]
    assert site.job == "translate" and site.api == "responses" and site.model_expr == "MODEL"
    assert site.system.text == "You are a careful translator. Translate to {lang}."


def test_template_matching():
    t = Template(("Summarize the ticket for ", "who", ". Be brief."))
    assert t.matches("Summarize the ticket for the   on-call engineer.\n Be brief.")
    assert not t.matches("Summarize the invoice for x. Be brief.")
    assert Template.slot("x").regex() is None          # too little fixed text to recognise anything
    assert Template.literal("short").regex() is None
    assert scan._brace_template("Hi {name}, {{literal}} {0:>3}").parts == ("Hi ", "name", ", {literal} ", "0", "")
    assert Template.from_json(t.to_json()) == t
    assert Template.from_json(["a", "b"]) is None


def test_call_site_json_round_trip():
    site = scan.scan_files({"app/support.py": PY})[0]
    again = scan.CallSite.from_json(site.to_json())
    assert again.job == site.job and again.prompt_fingerprint == site.prompt_fingerprint
    assert again.system == site.system


def test_skips_vendored_dirs():
    assert not scan.wanted("node_modules/openai/index.js")
    assert not scan.wanted("app/.venv/lib/x.py")
    assert scan.wanted("src/app.tsx") and not scan.wanted("README.md")


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_scan_ref_reads_git_without_checkout(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text('def f(x):\n    return c.chat.completions.create(model="m", messages=[{"role": "system", "content": "Answer as a pirate would."}])\n')
    (repo / "b.py").write_text("print('no ai here')\n")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "one")
    (repo / "a.py").write_text("print('gone')\n")      # working tree differs from the commit
    sites = scan.scan_ref(repo, "main")
    assert [s.job for s in sites] == ["a.py::f"]
    assert scan.scan_dir(repo) == []
    with pytest.raises(scan.GitError):
        scan.scan_ref(repo, "no-such-ref")


def test_template_matching_edge_cases_and_speed():
    t = Template(("Reply to ", "name", " about ", "topic", ". Be kind."))
    assert t.matches("Reply to  about . Be kind.")                 # empty slots
    assert t.matches("Reply to Ann about about about. Be kind.")   # literal repeated inside a slot
    assert not t.matches("Reply to Ann. Be kind.")
    many = Template(tuple(x for i in range(30) for x in ("word ", f"s{i}")) + ("the end of the prompt",))
    import time
    started = time.perf_counter()
    assert not many.matches("word " * 5000 + "no ending here")
    assert time.perf_counter() - started < 0.5


def test_deeply_nested_or_generated_python_does_not_break_the_scan():
    deep = "x = " + "+".join(["a"] * 3000) + "\n"
    ok = "def f():\n    return client.chat.completions.create(model='gpt-4o-mini', messages=[])\n"
    sites = scan.scan_files({"gen.py": deep, "ok.py": ok})
    assert [s.job for s in sites] == ["ok.py::f"]


def test_scan_ref_reads_paths_with_non_ascii_letters(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "résumé.py").write_text("def f():\n    return client.chat.completions.create(model='gpt-4o-mini', messages=[])\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-qm", "x"], check=True)
    assert [s.path for s in scan.scan_ref(repo, "HEAD")] == [s.path for s in scan.scan_dir(repo)] == ["résumé.py"]


def test_scan_dir_does_not_follow_a_symlink_out_of_the_repo(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("def f():\n    return client.chat.completions.create(model='gpt-4o-mini', messages=[])\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "link.py").symlink_to(outside)
    assert scan.scan_dir(repo) == []
