"""Pairwise LLM judge for stage 4, EVALUATE.

The judge sees the original conversation and two answers, A and B, and must
end its reply with one line that is exactly ``A``, ``B`` or ``TIE``. Anything
else is a judge error (``None``), never a guess.

:func:`judge_both_orders` asks twice with the answers swapped. If the two
orders disagree the outcome is a TIE, which cancels the judge's position bias.

The judge endpoint is called with httpx directly (OpenAI chat completions or
Anthropic messages). The API key is read from an environment variable the
operator names and is never printed or stored.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import httpx

from . import records

PROVIDERS = (records.PROVIDER_OPENAI, records.PROVIDER_ANTHROPIC)
DEFAULT_URLS = {
    records.PROVIDER_OPENAI: "https://api.openai.com/v1/chat/completions",
    records.PROVIDER_ANTHROPIC: "https://api.anthropic.com/v1/messages",
}
ANTHROPIC_VERSION = "2023-06-01"
VERDICTS = ("A", "B", "TIE")
WIN, TIE, LOSS, ERROR = "WIN", "TIE", "LOSS", "ERROR"
MAX_PROMPT_CHARS = 24_000
MAX_ANSWER_CHARS = 8_000

_BASE_RUBRIC = (
    "Judge correctness and whether the answer follows the instructions. "
    "Ignore style, length and tone unless the instructions ask for them. "
    "If both answers are acceptable, answer TIE."
)

RUBRICS: dict[str, str] = {
    "classification": "Is the label correct for the input, and is it one of the allowed labels in the format asked for? "
                      "Two answers with the same label are a TIE.",
    "extraction": "Are the extracted fields correct and complete, with nothing invented, in the format or schema asked for? "
                  "Different field order or whitespace does not matter.",
    "summarization": "Does the summary keep the key facts, add nothing that is not in the source, and respect any length limit asked for?",
    "agent": "Does the answer take a correct next step: the right tool with valid arguments, or a correct reply when no tool is needed?",
    "code": "Is the code correct for the task and does it do what was asked? Prefer working code over explained code.",
    "chat": "Is the reply correct, helpful and does it follow the instructions? Do not reward length.",
}
DEFAULT_RUBRIC = "Is the answer correct and does it follow the instructions?"

JUDGE_SYSTEM = (
    "You compare two answers to the same request. "
    + _BASE_RUBRIC
    + " Think briefly, then end with a final line that is exactly one of: A, B, TIE."
)


def rubric_for(workload_type: str) -> str:
    return RUBRICS.get(workload_type, DEFAULT_RUBRIC)


class BudgetExhausted(Exception):
    """The judge call or dollar cap is reached; stop judging."""


@dataclass
class JudgeBudget:
    """Hard caps on judge spend. ``price`` turns (input, output tokens) into
    USD; without it only ``max_calls`` applies."""

    max_calls: int
    max_usd: float | None = None
    price: Callable[[records.Usage], float | None] | None = None
    calls: int = 0
    usd: float = 0.0
    unpriced_calls: int = 0

    def reserve(self) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExhausted(f"judge call cap reached ({self.max_calls} calls)")
        if self.max_usd is not None and self.price is not None and self.usd >= self.max_usd:
            raise BudgetExhausted(f"judge spend cap reached (${self.max_usd:.2f})")
        self.calls += 1

    def record(self, usage: records.Usage) -> None:
        if self.price is None:
            return
        cost = self.price(usage) if usage.complete else None
        if cost is None:
            self.unpriced_calls += 1
        else:
            self.usd += cost


@dataclass
class JudgeConfig:
    provider: str
    model: str
    url: str
    api_key: str | None = field(default=None, repr=False)
    max_tokens: int = 1024
    timeout: float = 120.0


def pricing_for(provider: str, model: str) -> Callable[[records.Usage], float | None] | None:
    """A USD price function from ``throttle.own.pricing`` when that module
    exists and knows the model; otherwise None (only the call cap applies)."""

    try:
        module = importlib.import_module("throttle.own.pricing")
        entry = module.load_prices().lookup(provider, model)
        cost_fn = module.cost_usd
    except Exception:  # missing module, unreadable table, or a different interface
        return None
    if entry is None:
        return None

    def price(usage: records.Usage) -> float | None:
        try:
            value = cost_fn(entry, usage)
        except Exception:
            return None
        return float(value) if isinstance(value, (int, float)) else None

    return price


def parse_verdict(text: str | None) -> str | None:
    """``A``, ``B`` or ``TIE`` when the last non-empty line is exactly that."""

    if not text:
        return None
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None
    last = lines[-1]
    return last if last in VERDICTS else None


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n[... cut ...]\n" + text[-half:]


def render_messages(messages: Sequence[records.Message]) -> str:
    parts = []
    for message in messages:
        body = message.text
        if message.tool_calls:
            calls = "; ".join(f"{c.get('name')}({c.get('arguments')})" for c in message.tool_calls)
            body = f"{body}\n[tool calls: {calls}]".strip()
        parts.append(f"[{message.role}]\n{body}")
    return _clip("\n\n".join(parts), MAX_PROMPT_CHARS)


def render_answer(text: str | None, tool_calls: Sequence[dict[str, Any]] = ()) -> str:
    body = text or ""
    if tool_calls:
        calls = "\n".join(f"tool call: {c.get('name')}({c.get('arguments')})" for c in tool_calls)
        body = f"{body}\n{calls}".strip()
    return _clip(body or "(empty answer)", MAX_ANSWER_CHARS)


def build_prompt(prompt_messages: Sequence[records.Message], answer_a: str, answer_b: str, rubric: str) -> str:
    return (
        f"Rubric: {rubric}\n\n"
        f"=== Request ===\n{render_messages(prompt_messages)}\n\n"
        f"=== Answer A ===\n{answer_a}\n\n"
        f"=== Answer B ===\n{answer_b}\n\n"
        "Which answer is better by the rubric? If both are acceptable, or both fail the same way, say TIE. "
        "End with a final line that is exactly A, B or TIE."
    )


async def _call(client: httpx.AsyncClient, config: JudgeConfig, prompt: str) -> tuple[str | None, records.Usage]:
    if config.provider == records.PROVIDER_ANTHROPIC:
        headers = {"content-type": "application/json", "anthropic-version": ANTHROPIC_VERSION}
        if config.api_key:
            headers["x-api-key"] = config.api_key
        body = {
            "model": config.model, "max_tokens": config.max_tokens, "temperature": 0,
            "system": JUDGE_SYSTEM, "messages": [{"role": "user", "content": prompt}],
        }
        response = await client.post(config.url, json=body, headers=headers, timeout=config.timeout)
        if response.status_code != 200:
            return None, records.Usage()
        payload = response.json()
        text, _, _ = records._text_from_anthropic_content(payload.get("content"))
        return text, records.usage_from_anthropic(payload)
    headers = {"content-type": "application/json"}
    if config.api_key:
        headers["authorization"] = f"Bearer {config.api_key}"
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
    }
    if "api.openai.com" in config.url:
        # Reasoning models reject max_tokens and any temperature but the default.
        body["max_completion_tokens"] = config.max_tokens
    else:
        body["max_tokens"] = config.max_tokens
        body["temperature"] = 0
    response = await client.post(config.url, json=body, headers=headers, timeout=config.timeout)
    if response.status_code != 200:
        return None, records.Usage()
    payload = response.json()
    view = records.normalize_openai(None, payload)
    return view.output_text, records.usage_from_openai(payload)


async def judge_pair(
    prompt_messages: Sequence[records.Message],
    answer_a: str,
    answer_b: str,
    *,
    client: httpx.AsyncClient,
    provider: str,
    model: str,
    rubric: str,
    url: str | None = None,
    api_key: str | None = None,
    budget: JudgeBudget | None = None,
    config: JudgeConfig | None = None,
) -> str | None:
    """One judge call: ``'A'``, ``'B'``, ``'TIE'`` or None (judge error).

    Raises :class:`BudgetExhausted` before the call when a cap is reached.
    """

    if config is None:
        if provider not in PROVIDERS:
            raise ValueError(f"judge provider must be one of {', '.join(PROVIDERS)}")
        config = JudgeConfig(provider=provider, model=model, url=url or DEFAULT_URLS[provider], api_key=api_key)
    if budget is not None:
        budget.reserve()
    prompt = build_prompt(prompt_messages, answer_a, answer_b, rubric)
    try:
        text, usage = await _call(client, config, prompt)
    except (httpx.HTTPError, ValueError, AttributeError, TypeError):
        return None
    if budget is not None:
        budget.record(usage)
    return parse_verdict(text)


async def judge_both_orders(
    prompt_messages: Sequence[records.Message],
    incumbent: str,
    candidate: str,
    *,
    client: httpx.AsyncClient,
    config: JudgeConfig,
    rubric: str,
    budget: JudgeBudget | None = None,
) -> str:
    """The candidate's outcome: WIN, TIE, LOSS or ERROR.

    Order 1 shows (incumbent, candidate), order 2 shows (candidate, incumbent).
    If either call fails it is ERROR; if the orders disagree it is TIE.
    """

    common = dict(client=client, provider=config.provider, model=config.model,
                  rubric=rubric, budget=budget, config=config)
    first = await judge_pair(prompt_messages, incumbent, candidate, **common)  # type: ignore[arg-type]
    if first is None:
        return ERROR
    second = await judge_pair(prompt_messages, candidate, incumbent, **common)  # type: ignore[arg-type]
    if second is None:
        return ERROR
    one = {"A": LOSS, "B": WIN, "TIE": TIE}[first]
    two = {"A": WIN, "B": LOSS, "TIE": TIE}[second]
    return one if one == two else TIE
