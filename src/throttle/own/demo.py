"""``throttle demo``: the whole own-your-intelligence loop on synthetic data.

Offline, in a temporary data directory, deterministic for a seed:

1. CAPTURE: synthetic requests for a made-up company (all six workload types,
   OpenAI and Anthropic wire shapes, tool-calling agent turns) go through the
   real capture proxy app. The upstream is an in-process fake API that
   answers with plausible text and token usage. Only the clock and the
   exchange ids are replaced (spread over one day, seeded) so the run is
   reproducible.
2. SHADOW: the real backfill replays them against a fake open-weight model
   whose quality is designed per type: classification, extraction and chat
   about equal; summarization slightly worse; code worse; agent worse with
   malformed tool calls.
3. EVAL: the real evaluator and statistics, with a deterministic fake judge
   that reads the outcome the fake model was designed to have (plus a little
   position noise, which the both-orders rule turns into ties).
4. OWNED COST: a check-style record labelled DEMO (assumed GPU price,
   made-up $/M), read back through ``throttle check``'s history loader.
5. REPORT: the real report, every number labelled DEMO.

Nothing here is a measurement of any real company or model.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import httpx

from . import evaluate as ev
from . import judge as judge_mod
from . import pricing as pricing_mod
from . import records
from . import report as report_mod
from . import shadow as shadow_mod

OPENAI_MODEL = "gpt-4.1"
ANTHROPIC_MODEL = "claude-sonnet-4-5"
CANDIDATE_MODEL = "llama-3.3-70b-instruct"
CANDIDATE_LABEL = "llama-3.3-70b@vllm"
CANDIDATE_URL = "http://candidate.demo.invalid/v1"
JUDGE_URL = "http://judge.demo.invalid/v1/chat/completions"
GPU_HOURLY_RATE = 2.49   # per GPU, DEMO
GPUS = 2
CHECK_USD_PER_M = 0.31   # DEMO
TRAFFIC_SCALE = 2500     # each synthetic request stands for this many
WINDOW_HOURS = 24.5
DEFAULT_SEED = 7
DEFAULT_REQUESTS = 600

TYPE_WEIGHTS = {
    "classification": 0.22, "extraction": 0.17, "summarization": 0.15,
    "code": 0.12, "agent": 0.18, "chat": 0.16,
}
# (win share, loss share); the rest are ties. Designed, not measured.
QUALITY_DESIGN = {
    "classification": (0.02, 0.02),
    "extraction": (0.01, 0.02),
    "chat": (0.02, 0.02),
    "summarization": (0.08, 0.30),
    "code": (0.10, 0.50),
    "agent": (0.04, 0.46),
}
MALFORMED_AGENT_SHARE = 0.22   # of agent tool-call turns
JUDGE_POSITION_NOISE = 0.04
ANTHROPIC_SHARE = {"classification": 0.3, "extraction": 0.4, "summarization": 0.7,
                   "code": 0.2, "agent": 0.4, "chat": 0.5}

SID = re.compile(r"\b[A-Z]{2,3}-\d{5}\b")


@dataclass
class Scenario:
    sid: str
    kind: str
    provider: str
    model: str
    body: dict[str, Any]
    text: str | None
    calls: list[dict[str, Any]]          # [{"name", "input": dict}]
    tag: str = judge_mod.TIE
    cand_text: str = ""
    cand_calls: list[dict[str, str]] = field(default_factory=list)   # [{"name", "arguments": str}]
    in_tokens: int = 0
    out_tokens: int = 0
    cand_in: int = 0
    cand_out: int = 0
    ts: str = ""
    exchange_id: str = ""


def _tok(text: str) -> int:
    return max(1, round(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# Synthetic content
# ---------------------------------------------------------------------------

LABELS = {
    "billing": ["I was charged twice for the {plan} plan this month.", "Can I get a refund for the {n} unused seats on our invoice?",
                "Our card was declined but the invoice still says paid.", "Why did the price of the {plan} plan go up?"],
    "bug": ["The {page} page shows a blank screen after the last update.", "Exporting a board with {n} cards times out every time.",
            "Comments disappear when I switch to the {page} view.", "Drag and drop stopped working on the {page} board."],
    "feature_request": ["It would help a lot if the {page} view had a dark mode.", "Please add recurring tasks to the {page} board.",
                        "Could you support Gantt charts for projects with {n} people?", "We would love a way to archive old boards in bulk."],
    "account_access": ["I cannot log in since we turned on single sign-on.", "My two-factor codes stopped working on my new phone.",
                       "A teammate left and we need to move {n} boards off their account.", "The invite link for our workspace says it expired."],
    "other": ["Do you have an office in Berlin?", "Is there a discount for nonprofits with {n} members?",
              "Who should I talk to about a partnership?", "Can I speak at your user conference next spring?"],
}
PLANS = ["Team", "Business", "Starter", "Enterprise"]
PAGES = ["Timeline", "Calendar", "Roadmap", "Kanban", "Reports"]
VENDORS = ["Northwind Supply", "Bluepeak Hosting", "Atlas Office Co", "Kestrel Logistics", "Harbor Print Works", "Vela Cloud"]
CURRENCIES = ["USD", "EUR", "GBP"]
CALL_LINES = [
    ("Customer", "Hi, our team moved to the Business plan last week and the seat count looks wrong."),
    ("Agent", "Thanks for calling. I can see the upgrade on the account, let me look at the seats."),
    ("Customer", "We have {n} people but the bill shows {m} seats."),
    ("Agent", "I see two seats still assigned to people who left in September."),
    ("Customer", "Can you remove those and credit us for the difference?"),
    ("Agent", "Yes, I removed them and added a credit of {amt} dollars to the next invoice."),
    ("Customer", "Great. Also the Timeline view has been slow for the whole team since Monday."),
    ("Agent", "Our engineers are rolling out a fix for Timeline speed this week."),
    ("Customer", "Will we get an email when it is done?"),
    ("Agent", "Yes, I subscribed your workspace to the status updates for that issue."),
    ("Customer", "One more thing, can our admin see who changed the board permissions?"),
    ("Agent", "The audit log under Settings shows every permission change with the person and the time."),
    ("Customer", "Perfect, that is all for today."),
    ("Agent", "Happy to help. Your reference for this call is in the email I just sent."),
]
CODE_TASKS = [
    ("chunk_list", "splits a list into chunks of size n and keeps the last short chunk",
     "def chunk_list(items, n):\n    return [items[i:i + n] for i in range(0, len(items), n)]"),
    ("dedupe_keep_order", "removes duplicates from a list but keeps the first-seen order",
     "def dedupe_keep_order(items):\n    seen = set()\n    out = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            out.append(item)\n    return out"),
    ("parse_duration", "turns strings like '1h30m' or '45s' into a number of seconds",
     "import re\n\ndef parse_duration(text):\n    units = {'h': 3600, 'm': 60, 's': 1}\n    return sum(int(n) * units[u] for n, u in re.findall(r'(\\d+)([hms])', text))"),
    ("retry", "calls a callable up to n times with exponential backoff and re-raises the last error",
     "import time\n\ndef retry(fn, n=3, base=0.5):\n    for attempt in range(n):\n        try:\n            return fn()\n        except Exception:\n            if attempt == n - 1:\n                raise\n            time.sleep(base * 2 ** attempt)"),
    ("moving_average", "returns the moving average of a list of numbers over a window k",
     "def moving_average(values, k):\n    if k <= 0:\n        raise ValueError('k must be positive')\n    return [sum(values[i - k + 1:i + 1]) / k for i in range(k - 1, len(values))]"),
    ("slugify", "lowercases a title, keeps letters and digits and joins words with dashes",
     "import re\n\ndef slugify(title):\n    words = re.findall(r'[a-z0-9]+', title.lower())\n    return '-'.join(words)"),
]
CHAT_QUESTIONS = [
    ("How do I move a card to another board?", "Open the card, click the three dots in the top corner and pick Move. Choose the board and the list, and the card keeps its comments and attachments."),
    ("Can guests see private boards?", "No. Guests only see boards they are invited to, and a private board stays hidden from them until an admin adds them to it."),
    ("How do I set a due date reminder?", "Open the card, set the due date, then pick a reminder time under it. Lumen sends the reminder to everyone who follows the card."),
    ("What happens to my boards if I downgrade?", "Your boards stay as they are. Features from the higher plan, like timeline view, become read only until you upgrade again."),
    ("Can I import tasks from a spreadsheet?", "Yes. Go to Settings, then Import, and upload a CSV file. Each row becomes a card and you can map columns to card details."),
    ("How do I change the owner of a workspace?", "An owner can open Settings, then Members, and pick Make owner next to another admin. The old owner stays on as an admin."),
]
AGENT_TOOLS = [
    ("get_order", "Look up an order by id.", {"order_id": {"type": "string"}}, ["order_id"]),
    ("refund_order", "Refund all or part of an order.", {"order_id": {"type": "string"}, "amount": {"type": "number"}, "reason": {"type": "string"}}, ["order_id", "amount"]),
    ("track_shipment", "Get the carrier status for an order.", {"order_id": {"type": "string"}}, ["order_id"]),
    ("send_reply", "Send a reply to the customer.", {"text": {"type": "string"}}, ["text"]),
]
AGENT_ASKS = [
    ("Where is my order {oid}? It was supposed to arrive on Friday.", "track_shipment"),
    ("Order {oid} arrived broken, I want my money back.", "get_order"),
    ("Can you check what I ordered in {oid}? I forgot.", "get_order"),
    ("Please refund the shipping on {oid}, it came a week late.", "get_order"),
]


def _openai_tools() -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": n, "description": d,
             "parameters": {"type": "object", "properties": p, "required": r}}} for n, d, p, r in AGENT_TOOLS]


def _anthropic_tools() -> list[dict[str, Any]]:
    return [{"name": n, "description": d, "input_schema": {"type": "object", "properties": p, "required": r}}
            for n, d, p, r in AGENT_TOOLS]


def _messages(provider: str, system: str, user: str) -> dict[str, Any]:
    if provider == records.PROVIDER_ANTHROPIC:
        return {"system": system, "messages": [{"role": "user", "content": user}]}
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}


def _classification(rng: random.Random, sid: str, provider: str, tag: str) -> Scenario:
    label = rng.choice(sorted(LABELS))
    msg = rng.choice(LABELS[label]).format(plan=rng.choice(PLANS), page=rng.choice(PAGES), n=rng.randint(2, 60))
    system = ("You triage support tickets for Lumen, a project-management app. Classify the ticket into exactly one "
              "category: billing, bug, feature_request, account_access, other. Reply with the category name only.")
    body = {**_messages(provider, system, f"Ticket {sid}: {msg}"), "max_tokens": 10, "temperature": 0}
    others = [x for x in sorted(LABELS) if x != label]
    wrong = rng.choice(others)
    inc, cand = label, label
    if tag == judge_mod.LOSS:
        cand = wrong
    elif tag == judge_mod.WIN:
        inc = wrong   # the API model got this one wrong
    return Scenario(sid, "classification", provider, "", body, inc, [], tag, cand_text=cand)


def _extraction(rng: random.Random, sid: str, provider: str, tag: str, broken: bool) -> Scenario:
    vendor = rng.choice(VENDORS)
    day = datetime(2026, 9, 1) + timedelta(days=rng.randint(0, 50))
    total = round(rng.uniform(80, 9000), 2)
    cur = rng.choice(CURRENCIES)
    lines = rng.randint(2, 5)
    items = "\n".join(f"  {rng.choice(['Licenses', 'Support hours', 'Storage', 'Shipping', 'Printing'])}  {round(total / lines, 2)}"
                      for _ in range(lines))
    doc = (f"{vendor}\nInvoice {sid}\nDate: {day:%d %B %Y}\nBill to: Lumen Inc, Accounts Payable\n"
           f"Items:\n{items}\nTotal due: {total:,.2f} {cur}\nPayment terms: net 30")
    value = {"vendor": vendor, "invoice_number": sid, "date": f"{day:%Y-%m-%d}", "total": total, "currency": cur}
    schema = {"type": "object", "additionalProperties": False, "required": list(value),
              "properties": {"vendor": {"type": "string"}, "invoice_number": {"type": "string"},
                             "date": {"type": "string"}, "total": {"type": "number"}, "currency": {"type": "string"}}}
    system = "You read invoices for the finance team. Pull out vendor, invoice_number, date (YYYY-MM-DD), total and currency."
    if provider == records.PROVIDER_ANTHROPIC:
        system += " Respond with JSON only."
    body = {**_messages(provider, system, doc), "max_tokens": 200, "temperature": 0}
    if provider == records.PROVIDER_OPENAI:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "invoice", "schema": schema}}
    inc = json.dumps(value)
    cand_value = dict(value)
    if tag == judge_mod.LOSS:
        cand_value["total"] = round(total / lines, 2)   # picked a line item, not the total
    cand = json.dumps(cand_value, indent=1)
    if broken:
        cand = cand[:-3]   # cut off mid-object
    return Scenario(sid, "extraction", provider, "", body, inc, [], tag, cand_text=cand)


def _summarization(rng: random.Random, sid: str, provider: str, tag: str) -> Scenario:
    n = rng.randint(8, 40)
    amt = rng.randint(20, 400)
    start = rng.randint(0, 3)
    picked = CALL_LINES[start:start + rng.randint(9, 11)]
    transcript = "\n".join(f"{who}: {line.format(n=n, m=n + 2, amt=amt)}" for who, line in picked)
    system = "You help the support team keep short notes on customer calls."
    body = {**_messages(provider, system, f"Summarize call {sid} in three bullet points for the account notes.\n\n{transcript}"),
            "max_tokens": 300}
    inc = (f"- The customer's seat count was wrong after the Business upgrade; two former staff seats were removed.\n"
           f"- A credit of {amt} dollars goes on the next invoice.\n"
           "- Timeline speed is being fixed this week and the workspace will get status emails.")
    if tag == judge_mod.LOSS:
        cand = (f"- The customer called about their plan and some seats.\n- The agent offered a refund of {amt + 50} dollars.\n"
                "- The customer also mentioned the app being slow.")
    elif tag == judge_mod.WIN:
        cand = (f"- Seat count fixed: two seats of former staff removed, {amt} dollar credit on the next invoice.\n"
                "- Timeline slowness is known; the fix ships this week and the workspace is subscribed to updates.\n"
                "- Admins can see permission changes in the audit log.")
    else:
        cand = (f"- Two seats of people who left were removed after the Business plan upgrade.\n"
                f"- The account gets a {amt} dollar credit on the next invoice.\n"
                "- A fix for the slow Timeline view ships this week; status emails are on.")
    return Scenario(sid, "summarization", provider, "", body, inc, [], tag, cand_text=cand)


def _code(rng: random.Random, sid: str, provider: str, tag: str) -> Scenario:
    name, desc, solution = rng.choice(CODE_TASKS)
    system = "You are a senior Python engineer. Answer with code only, in one fenced block."
    body = {**_messages(provider, system, f"Task {sid}: Write a Python function `{name}` that {desc}."), "max_tokens": 400}
    inc = f"```python\n{solution}\n```"
    if tag == judge_mod.LOSS:
        broken = solution.replace("range(0, len(items), n)", "range(1, len(items), n)").replace("k - 1", "k").replace(
            "seen.add(item)\n", "").replace("2 ** attempt", "2").replace("[a-z0-9]+", "[a-z]+").replace("re.findall(r'(\\d+)([hms])', text)", "re.findall(r'(\\d)([hms])', text)")
        cand = f"Here is the function:\n\n```python\n{broken}\n```\n\nThis should work for most inputs."
    elif tag == judge_mod.WIN:
        cand = f"```python\n{solution}\n```\n\nIt handles empty input as well."
    else:
        cand = f"```python\n{solution}\n```"
    return Scenario(sid, "code", provider, "", body, inc, [], tag, cand_text=cand)


def _agent(rng: random.Random, sid: str, provider: str, tag: str, malformed: bool) -> Scenario:
    ask, first_tool = rng.choice(AGENT_ASKS)
    user = ask.format(oid=sid)
    system = ("You are the support agent for Lumen Goods, an online store. Use the tools to look things up before "
              "you answer. Never promise a refund over the order total.")
    follow_up = rng.random() < 0.35
    total = round(rng.uniform(20, 300), 2)
    tools = _anthropic_tools() if provider == records.PROVIDER_ANTHROPIC else _openai_tools()
    if not follow_up:
        messages_body = _messages(provider, system, user)
        inc_calls = [{"name": first_tool, "input": {"order_id": sid}}]
        inc_text = None
        good_calls = [{"name": first_tool, "arguments": json.dumps({"order_id": sid})}]
        bad_calls = [{"name": "get_order" if first_tool == "track_shipment" else "send_reply",
                      "arguments": json.dumps({"order_id": sid} if first_tool == "track_shipment" else {"text": "Sorry about that."})}]
    else:
        result = json.dumps({"order_id": sid, "status": "delivered", "total": total, "items": 2})
        if provider == records.PROVIDER_ANTHROPIC:
            messages_body = {"system": system, "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_order", "input": {"order_id": sid}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": result}]},
            ]}
        else:
            messages_body = {"messages": [
                {"role": "system", "content": system}, {"role": "user", "content": user},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                 "function": {"name": "get_order", "arguments": json.dumps({"order_id": sid})}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": result},
            ]}
        refund = round(total * 0.2, 2)
        inc_calls = [{"name": "refund_order", "input": {"order_id": sid, "amount": refund, "reason": "late or damaged delivery"}}]
        inc_text = None
        good_calls = [{"name": "refund_order", "arguments": json.dumps({"order_id": sid, "amount": refund, "reason": "late delivery"})}]
        bad_calls = [{"name": "refund_order", "arguments": json.dumps({"order_id": sid, "amount": round(total * 1.5, 2)})}]
    body = {**messages_body, "tools": tools, "max_tokens": 300}
    if malformed:
        name = good_calls[0]["name"]
        cand_calls = ([{"name": name, "arguments": good_calls[0]["arguments"][:-2]}] if rng.random() < 0.6
                      else [{"name": "lookup_order", "arguments": json.dumps({"order_id": sid})}])
    elif tag == judge_mod.LOSS:
        cand_calls = bad_calls
    else:
        cand_calls = good_calls
    return Scenario(sid, "agent", provider, "", body, inc_text, inc_calls, tag, cand_text="", cand_calls=cand_calls)


def _chat(rng: random.Random, sid: str, provider: str, tag: str) -> Scenario:
    question, answer = rng.choice(CHAT_QUESTIONS)
    system = "You are Lumen's in-app assistant. Answer questions about using Lumen in two or three sentences."
    opener = rng.choice(["Hi!", "Quick one.", "Hello,", "Hey team,"])
    body = {**_messages(provider, system, f"{opener} {question} (ref {sid})"), "max_tokens": 250}
    if tag == judge_mod.LOSS:
        cand = "That depends on your setup. Please check the help center or contact your workspace admin for details."
    elif tag == judge_mod.WIN:
        cand = answer + " If you get stuck, the help center has a short video on it."
    else:
        cand = answer.replace(", and", ". And", 1) if ", and" in answer else answer + " "
        cand = cand.strip()
    return Scenario(sid, "chat", provider, "", body, answer, [], tag, cand_text=cand)


def build_scenarios(seed: int, n: int) -> list[Scenario]:
    """Seeded synthetic traffic with designed candidate outcomes per type."""

    rng = random.Random(seed)
    kinds = list(TYPE_WEIGHTS)
    counts = {k: int(n * TYPE_WEIGHTS[k]) for k in kinds}
    for i in range(n - sum(counts.values())):
        counts[kinds[i % len(kinds)]] += 1
    prefixes = {"classification": "TK", "extraction": "INV", "summarization": "CL", "code": "PR", "agent": "OR", "chat": "CV"}
    numbers = rng.sample(range(10000, 99999), n)
    out: list[Scenario] = []
    for kind in kinds:
        k = counts[kind]
        win, loss = QUALITY_DESIGN[kind]
        tags = [judge_mod.WIN] * round(win * k) + [judge_mod.LOSS] * round(loss * k)
        tags += [judge_mod.TIE] * (k - len(tags))
        rng.shuffle(tags)
        malformed = set()
        if kind == "agent":
            losses = [i for i, t in enumerate(tags) if t == judge_mod.LOSS]
            malformed = set(rng.sample(losses, min(len(losses), round(MALFORMED_AGENT_SHARE * k))))
        broken = set()
        if kind == "extraction":
            losses = [i for i, t in enumerate(tags) if t == judge_mod.LOSS]
            broken = set(losses[:1])
        for i, tag in enumerate(tags):
            sid = f"{prefixes[kind]}-{numbers.pop()}"
            provider = records.PROVIDER_ANTHROPIC if rng.random() < ANTHROPIC_SHARE[kind] else records.PROVIDER_OPENAI
            if kind == "classification":
                s = _classification(rng, sid, provider, tag)
            elif kind == "extraction":
                s = _extraction(rng, sid, provider, tag, i in broken)
            elif kind == "summarization":
                s = _summarization(rng, sid, provider, tag)
            elif kind == "code":
                s = _code(rng, sid, provider, tag)
            elif kind == "agent":
                s = _agent(rng, sid, provider, tag, i in malformed)
            else:
                s = _chat(rng, sid, provider, tag)
            s.model = ANTHROPIC_MODEL if provider == records.PROVIDER_ANTHROPIC else OPENAI_MODEL
            s.body = {"model": s.model, **s.body}
            prompt_text = json.dumps(s.body)
            s.in_tokens = _tok(prompt_text.replace('"', " ").replace(",", " "))
            answer_text = (s.text or "") + "".join(json.dumps(c["input"]) for c in s.calls)
            s.out_tokens = _tok(answer_text.replace('"', " ")) + 2
            s.cand_in = round(s.in_tokens * 1.07)
            s.cand_out = _tok((s.cand_text + "".join(c["arguments"] for c in s.cand_calls)).replace('"', " ")) + 2
            out.append(s)
    rng.shuffle(out)
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=WINDOW_HOURS + 1)
    offsets = sorted(rng.uniform(0, WINDOW_HOURS * 3600) for _ in out)
    offsets[0], offsets[-1] = 0.0, WINDOW_HOURS * 3600
    for s, offset in zip(out, offsets):
        s.ts = (start + timedelta(seconds=offset)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        s.exchange_id = hashlib.sha256(f"demo:{seed}:{s.sid}".encode()).hexdigest()[:32]
    return out


# ---------------------------------------------------------------------------
# Fakes: API upstream, open-weight candidate, judge
# ---------------------------------------------------------------------------


def _find(text: str, registry: Mapping[str, Scenario]) -> Scenario | None:
    for match in SID.finditer(text):
        if match.group(0) in registry:
            return registry[match.group(0)]
    return None


def upstream_handler(registry: Mapping[str, Scenario]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        s = _find(json.dumps(body.get("messages")), registry)
        if s is None:
            return httpx.Response(400, json={"error": {"type": "invalid_request_error", "message": "unknown demo request"}})
        if request.url.path.endswith("/messages"):
            content: list[dict[str, Any]] = [{"type": "text", "text": s.text}] if s.text else []
            content += [{"type": "tool_use", "id": f"toolu_{i}{s.sid[-3:]}", "name": c["name"], "input": c["input"]}
                        for i, c in enumerate(s.calls)]
            return httpx.Response(200, json={
                "id": f"msg_{s.exchange_id[:12]}", "type": "message", "role": "assistant", "model": s.model,
                "content": content, "stop_reason": "tool_use" if s.calls else "end_turn",
                "usage": {"input_tokens": s.in_tokens, "output_tokens": s.out_tokens},
            })
        message: dict[str, Any] = {"role": "assistant", "content": s.text}
        if s.calls:
            message["tool_calls"] = [{"id": f"call_{i}{s.sid[-3:]}", "type": "function",
                                      "function": {"name": c["name"], "arguments": json.dumps(c["input"])}}
                                     for i, c in enumerate(s.calls)]
        return httpx.Response(200, json={
            "id": f"chatcmpl-{s.exchange_id[:12]}", "object": "chat.completion", "model": s.model,
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if s.calls else "stop"}],
            "usage": {"prompt_tokens": s.in_tokens, "completion_tokens": s.out_tokens,
                      "total_tokens": s.in_tokens + s.out_tokens},
        })

    return handler


def candidate_handler(registry: Mapping[str, Scenario]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        s = _find(json.dumps(body.get("messages")), registry)
        if s is None:
            return httpx.Response(400, json={"error": {"message": "unknown demo request"}})
        message: dict[str, Any] = {"role": "assistant", "content": s.cand_text}
        if s.cand_calls:
            message["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                                      "function": {"name": c["name"], "arguments": c["arguments"]}}
                                     for i, c in enumerate(s.cand_calls)]
        return httpx.Response(200, json={
            "id": f"cmpl-{s.exchange_id[:12]}", "object": "chat.completion", "model": CANDIDATE_MODEL,
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if s.cand_calls else "stop"}],
            "usage": {"prompt_tokens": s.cand_in, "completion_tokens": s.cand_out, "total_tokens": s.cand_in + s.cand_out},
        })

    return handler


@dataclass
class FakeJudge:
    """Reads the designed outcome; never looks at answer quality itself."""

    registry: Mapping[str, Scenario]
    seed: int
    unmatched: int = 0
    calls: int = 0

    def decide(self, prompt: str) -> str:
        request = prompt.split("=== Request ===\n", 1)[-1].split("\n\n=== Answer A ===", 1)[0]
        a = prompt.split("=== Answer A ===\n", 1)[1].split("\n\n=== Answer B ===", 1)[0]
        b = prompt.split("=== Answer B ===\n", 1)[1].split("\n\nWhich answer", 1)[0]
        s = _find(request, self.registry)
        if s is None:
            self.unmatched += 1
            return "TIE"
        cand = judge_mod.render_answer(s.cand_text, s.cand_calls)
        if a == b or s.tag == judge_mod.TIE:
            return "TIE"
        if cand == b:
            first = True      # order 1: A is the API model
        elif cand == a:
            first = False
        else:
            self.unmatched += 1
            return "TIE"
        noise = int(hashlib.sha256(f"{self.seed}:{s.sid}:{first}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if noise < JUDGE_POSITION_NOISE:
            return "A"        # position bias; the other order cancels it into a TIE
        cand_letter, inc_letter = ("B", "A") if first else ("A", "B")
        return cand_letter if s.tag == judge_mod.WIN else inc_letter

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        body = json.loads(request.content)
        verdict = self.decide(body["messages"][-1]["content"])
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": f"Compared both against the rubric.\n{verdict}"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 400, "completion_tokens": 12},
        })


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


async def capture(root: Path, scenarios: list[Scenario]) -> int:
    from .capture import create_capture_app

    registry = {s.sid: s for s in scenarios}
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler(registry)))
    app = create_capture_app("https://api.openai.com", "https://api.anthropic.com", redaction=records.REDACTION_MASK,
                             root=root, salt=b"throttle-demo", client=upstream)
    state = app.state.capture
    real_append = state._append

    def stamped(exchange: records.Exchange) -> None:
        # Demo clock and ids: spread over a day, reproducible for the seed.
        s = _find(json.dumps((exchange.request or {}).get("messages")), registry)
        if s is not None:
            exchange.ts, exchange.id = s.ts, s.exchange_id
        real_append(exchange)

    state._append = stamped
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://capture.demo") as client:
        for s in sorted(scenarios, key=lambda x: x.ts):
            if s.provider == records.PROVIDER_ANTHROPIC:
                await client.post("/v1/messages", json=s.body,
                                  headers={"x-api-key": "demo-key", "anthropic-version": "2023-06-01"})
            else:
                await client.post("/v1/chat/completions", json=s.body, headers={"authorization": "Bearer demo-key"})
    await state.drain()
    await upstream.aclose()
    return state.written


def write_check(check_dir: Path, seed: int) -> dict[str, Any]:
    """A ``throttle check``-shaped record, labelled DEMO, for the fake model."""

    from .. import __version__
    from .. import check as check_mod

    rng = random.Random(seed * 31 + 1)
    blocks = []
    for _ in range(6):
        total = CHECK_USD_PER_M * rng.uniform(0.94, 1.06)
        blocks.append({"dollars_per_million": {"total": total, "input": total * 1.25, "output": total * 5.0},
                       "demo": True})
    result = {key: check_mod.summarize([b["dollars_per_million"][key] for b in blocks]) for key in ("input", "output", "total")}
    created = records.utc_now_iso()
    record = {
        "record_type": check_mod.RECORD_TYPE, "record_version": check_mod.RECORD_VERSION,
        "id": f"demo-check-seed{seed}", "created_at": created, "throttle_version": __version__,
        "endpoint": CANDIDATE_URL + "/chat/completions", "metric": "total", "demo": True,
        "label": report_mod.DEMO_LABEL,
        "fingerprint": {"model": CANDIDATE_MODEL, "label": CANDIDATE_LABEL,
                        "gpu_hourly_rate_usd": GPU_HOURLY_RATE * GPUS,
                        "gpu_hourly_rate_source": "DEMO (made-up rate for the demo)", "gpu_count": GPUS,
                        "user_config": {}, "server_models": {"status": "DEMO"}, "server_metrics": {"status": "DEMO"}},
        "workload": {"demo": True}, "blocks": blocks, "result": result,
    }
    check_mod.append_history(check_dir, record)
    return record


def site_json(report: Mapping[str, Any]) -> dict[str, Any]:
    """The landing page's demo-report.json format; null where the report has no number."""

    rows = []
    for w in report["workloads"]:
        q = w.get("quality") or {}
        owned = w.get("owned_usd_per_month")
        rows.append({
            "type": w["type"],
            "monthly_api_usd": _round(w.get("api_usd_per_month")),
            "projected_owned_usd": _round(owned["point"]) if owned else None,
            "verdict": w["recommendation"],
            "quality": {"win": _round(q.get("win_rate"), 4), "tie": _round(q.get("tie_rate"), 4),
                        "loss": _round(q.get("loss_rate"), 4),
                        "net_ci": [_round(x, 4) for x in q["net_ci"]] if q.get("net_ci") else [None, None]},
        })
    totals = report["totals"]
    return {
        "placeholder": False,
        "generated_at": report["generated_at"],
        "label": report_mod.DEMO_LABEL,
        "workloads": rows,
        "totals": {"monthly_api_usd": _round(totals.get("monthly_api_usd")),
                   "movable_usd": _round(totals.get("movable_api_usd")),
                   "projected_savings_usd": _round(totals.get("conservative_savings_usd"))},
    }


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(float(value), digits)


@dataclass
class DemoResult:
    report: dict[str, Any]
    json_path: Path
    html_path: Path
    site: dict[str, Any]
    seconds: float
    judge_unmatched: int
    captured: int


def run_demo(out: Path, *, seed: int = DEFAULT_SEED, requests: int = DEFAULT_REQUESTS,
             work_dir: Path | None = None) -> DemoResult:
    started = time.perf_counter()
    temp = Path(tempfile.mkdtemp(prefix="throttle-demo-")) if work_dir is None else work_dir
    previous = os.environ.get("THROTTLE_OWN_DIR")
    os.environ["THROTTLE_OWN_DIR"] = str(temp)
    try:
        root = temp
        scenarios = build_scenarios(seed, requests)
        registry = {s.sid: s for s in scenarios}
        captured = asyncio.run(capture(root, scenarios))

        selection = shadow_mod.select_backfill(records.iter_exchanges(root), per_type=requests,
                                               max_requests=requests, seed=seed)
        asyncio.run(shadow_mod.run_backfill(
            selection, url=CANDIDATE_URL, model=CANDIDATE_MODEL, label=CANDIDATE_LABEL, root=root,
            concurrency=8, transport=httpx.MockTransport(candidate_handler(registry)),
        ))

        judge = FakeJudge(registry, seed)
        config = judge_mod.JudgeConfig(provider=records.PROVIDER_OPENAI, model="demo-judge", url=JUDGE_URL)
        ev.run_eval(root, CANDIDATE_LABEL, judge_config=config,
                    budget=judge_mod.JudgeBudget(max_calls=4 * requests + 10),
                    transport=httpx.MockTransport(judge.handler), judge_concurrency=8,
                    run_id=f"demo-eval-seed{seed}")

        check_dir = root / "checks"
        write_check(check_dir, seed)
        eval_doc = report_mod.load_eval(root, CANDIDATE_LABEL)
        models = [m for m in ((eval_doc or {}).get("candidate_model"), CANDIDATE_LABEL) if m]
        record = report_mod.load_check(check_dir, models)
        report = report_mod.build_report(
            list(records.iter_exchanges(root)), pricing_mod.load_prices(), candidate=CANDIDATE_LABEL,
            eval_doc=eval_doc, check_record=record, gpu_hourly_rate=GPU_HOURLY_RATE, gpus=GPUS,
            traffic_scale=TRAFFIC_SCALE, demo=True,
        )
        report["demo_seed"] = seed
        report["demo_requests"] = requests
        json_path, html_path = report_mod.write_outputs(report, out)
        return DemoResult(report, json_path, html_path, site_json(report), time.perf_counter() - started,
                          judge.unmatched, captured)
    finally:
        if previous is None:
            os.environ.pop("THROTTLE_OWN_DIR", None)
        else:
            os.environ["THROTTLE_OWN_DIR"] = previous
        if work_dir is None:
            shutil.rmtree(temp, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "demo",
        help="run the whole loop (capture, shadow, eval, report) on synthetic data, offline",
        description=(
            "Run capture -> shadow -> eval -> report end to end on synthetic traffic with a fake API, "
            "a fake open-weight model and a fake judge, offline in a temporary folder. Every number "
            "is labelled DEMO: it shows how the report works, not what any real model saves."
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("throttle-demo"), metavar="DIR",
                        help="where report.json and report.html go (default ./throttle-demo/)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"seed (default {DEFAULT_SEED})")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS,
                        help=f"synthetic requests to capture (default {DEFAULT_REQUESTS}; at least 300)")
    parser.add_argument("--site-json", type=Path, default=None, metavar="PATH",
                        help="also write the landing page's demo-report.json here")


def handle(args: argparse.Namespace) -> int:
    from .. import style as style_mod

    if args.requests < 300:
        print("throttle demo: --requests must be at least 300 (each type needs 30+ judged requests)", file=sys.stderr)
        return 2
    print(f"Running the demo loop on {args.requests} synthetic requests (seed {args.seed}), offline...")
    result = run_demo(args.out.expanduser(), seed=args.seed, requests=args.requests)
    if args.site_json:
        path = args.site_json.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.site, indent=2) + "\n", encoding="utf-8")
    print(report_mod.render_panel(result.report, [result.json_path, result.html_path]))
    st = style_mod.get()
    print(f"Done in {result.seconds:.1f}s. Open {st.bold(str(result.html_path))}")
    if args.site_json:
        print(f"Site JSON: {args.site_json}")
    print("Run this on your own traffic: throttle capture --dir ~/.throttle/own   "
          "(then point OPENAI_BASE_URL / ANTHROPIC_BASE_URL at http://127.0.0.1:8790)")
    return 0
