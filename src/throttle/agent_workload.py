"""Synthetic multi-turn agent sessions for ``throttle check --workload agent``.

The default check sends single-turn prompts with a unique tag each, so no
request can reuse another's prefix. Agent traffic is the opposite: one long
shared system prompt plus tool schema, then turns that each resend the whole
conversation so far (the model's previous reply, a tool result, a short
instruction). This module builds that traffic, deterministically from a seed:

* Every session of a run starts with the same system message: the run tag
  (``[run 482915] ``, so one run's prefix cache cannot help the next run),
  then about ``system_prompt_tokens`` words of instructions and tool schema.
* Session ``i``'s first user message is a task seeded by ``i``, so sessions
  share the system prefix and nothing after it.
* Turn ``k`` sends turn ``k-1``'s messages plus the model's reply to turn
  ``k-1`` and a user message carrying a synthetic tool result of about
  ``tool_output_tokens`` words. Turn ``k``'s prompt therefore starts with
  turn ``k-1``'s prompt, and a server's prefix cache can reuse it within the
  session, as it would for a real agent.

Sizes are estimates without a tokenizer: a plain common English word is about
one token, and JSON schema text about one token per 4 characters. The tokens
actually sent are MEASURED from the server's usage and recorded.
This module is pure: no I/O, no clock, no global random state.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

PROFILE_DEFAULT = "default"
PROFILE_AGENT = "agent"
PROFILES = (PROFILE_DEFAULT, PROFILE_AGENT)

# Bump when the generated text changes, so old and new agent checks get a
# different prompts_sha256 and are never compared as one workload.
GENERATOR_VERSION = 1
DEFAULT_SEED = 0
DEFAULT_TURNS = 6
DEFAULT_TOOL_OUTPUT_TOKENS = 300
DEFAULT_SYSTEM_PROMPT_TOKENS = 1500

NONCE_FORMAT = "[run {run_id}] "
_CHARS_PER_JSON_TOKEN = 4

# Workload shape parameters that are part of the workload identity.
SHAPE_IDENTITY_KEYS = (
    "profile",
    "turns",
    "sessions_concurrency",
    "system_prompt_tokens",
    "tool_output_tokens",
)

_TOOLS = (
    ("search_docs", "query"),
    ("read_file", "path"),
    ("run_tests", "target"),
    ("query_db", "sql"),
    ("http_get", "url"),
    ("write_file", "path"),
    ("list_dir", "path"),
    ("grep_repo", "pattern"),
)

_VOCAB = (
    "the", "a", "of", "to", "and", "in", "is", "for", "on", "with", "that",
    "this", "file", "line", "value", "error", "test", "build", "user", "data",
    "table", "row", "column", "request", "response", "server", "client",
    "result", "status", "code", "config", "update", "delete", "create", "read",
    "write", "check", "return", "list", "item", "name", "type", "string",
    "number", "field", "query", "index", "cache", "memory", "time", "date",
    "order", "price", "total", "count", "first", "last", "next", "new", "old",
    "open", "close", "start", "stop", "run", "step", "task", "plan", "note",
    "found", "missing", "failed", "passed", "warning", "info", "debug", "path",
    "module", "function", "class", "method", "call", "input", "output", "key",
    "map", "set", "root", "node", "tree", "page", "link", "text", "word",
    "small", "large", "high", "low", "fast", "slow", "good", "bad", "true",
    "false", "none", "empty", "full", "left", "right", "top", "bottom", "west",
    "east", "north", "south", "blue", "green", "red", "black", "white", "one",
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
)


@dataclass(frozen=True)
class AgentShape:
    """The parameters of one agent workload. Equal shapes send equal traffic
    (given equal model replies)."""

    turns: int = DEFAULT_TURNS
    sessions_concurrency: int = 2
    system_prompt_tokens: int = DEFAULT_SYSTEM_PROMPT_TOKENS
    tool_output_tokens: int = DEFAULT_TOOL_OUTPUT_TOKENS
    seed: int = DEFAULT_SEED

    def identity(self) -> dict[str, Any]:
        return {
            "profile": PROFILE_AGENT,
            "turns": self.turns,
            "sessions_concurrency": self.sessions_concurrency,
            "system_prompt_tokens": self.system_prompt_tokens,
            "tool_output_tokens": self.tool_output_tokens,
        }

    def generator_spec(self) -> dict[str, Any]:
        """What the prompt hash covers: the generator, its seed and shape."""
        return {"generator": "throttle.agent_workload", "version": GENERATOR_VERSION,
                "seed": self.seed, **self.identity()}


def _words(rng: random.Random, count: int) -> str:
    return " ".join(rng.choice(_VOCAB) for _ in range(max(count, 0)))


def system_prompt(shape: AgentShape) -> str:
    """Instructions plus tool schema, about ``system_prompt_tokens`` words.
    The same for every session and every run of one shape (the run tag is
    added in front by :func:`turn_messages`)."""

    rng = random.Random(f"{shape.seed}:system")
    parts = [
        "You are an autonomous software agent. Work in steps. At each step "
        "call one tool, read its result, then reply in one or two short "
        "sentences saying what you will do next.",
        "Tools (JSON schema):",
    ]
    used = sum(len(p.split()) for p in parts)
    index = 0
    while used < shape.system_prompt_tokens:
        name, arg = _TOOLS[index % len(_TOOLS)]
        description = _words(rng, 16)
        line = (
            f'{{"name": "{name}_{index}", "description": "{description}", '
            f'"parameters": {{"{arg}": "string"}}}}'
        )
        parts.append(line)
        used += len(line) // _CHARS_PER_JSON_TOKEN
        index += 1
    return "\n".join(parts)


def session_task(shape: AgentShape, session_index: int) -> str:
    rng = random.Random(f"{shape.seed}:task:{session_index}")
    return f"Task {session_index}: {_words(rng, 40)}. Start with the first step."


def tool_message(shape: AgentShape, session_index: int, turn: int) -> str:
    """The user message appended after the model's reply to ``turn``."""

    rng = random.Random(f"{shape.seed}:tool:{session_index}:{turn}")
    name, _ = _TOOLS[rng.randrange(len(_TOOLS))]
    return (
        f"Tool result ({name}): {_words(rng, shape.tool_output_tokens)}\n"
        "Continue with the next step."
    )


def system_message(shape: AgentShape, run_id: str) -> dict[str, str]:
    return {"role": "system", "content": NONCE_FORMAT.format(run_id=run_id) + system_prompt(shape)}


def turn_messages(
    shape: AgentShape,
    run_id: str,
    session_index: int,
    replies: Sequence[str],
    *,
    system: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """The messages of turn ``len(replies)`` of session ``session_index``.

    ``replies`` are the model's answers to the earlier turns of this session.
    ``system`` may pass a prebuilt :func:`system_message` to avoid rebuilding it.
    """

    messages = [
        dict(system or system_message(shape, run_id)),
        {"role": "user", "content": session_task(shape, session_index)},
    ]
    for turn, reply in enumerate(replies):
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": tool_message(shape, session_index, turn)})
    return messages


def warmup_messages(shape: AgentShape, run_id: str, index: int) -> list[dict[str, str]]:
    """An unmeasured request that puts this run's system prefix in the cache,
    so block 1 does not pay a prefill that later blocks skip."""

    return [
        system_message(shape, run_id),
        {"role": "user", "content": f"Warm-up {index}: reply with OK."},
    ]
