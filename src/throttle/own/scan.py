"""Find the AI calls in a code base: one :class:`CallSite` per call.

A call site is one place in the code that calls OpenAI or Anthropic. Throttle
calls it a *job*. Its identity is stable while the code changes:

* the name the team gave it, when the call is tagged (see
  :mod:`throttle.own.tag`): an ``X-Throttle-Job`` header literal, a
  ``throttle_job("name")`` helper call, or a ``# throttle-job: name`` /
  ``// throttle-job: name`` comment on the call line or the line above;
* otherwise ``<path>::<function>``, plus ``#2``, ``#3`` ... for the second and
  later calls in the same function.

The prompt is not part of the identity. It is tracked separately as a
template (literal text with ``{slots}`` where the code fills in values) and a
fingerprint, so an edited prompt shows up as the same job, changed.

Python is parsed with the standard library ``ast``. These are found:

* ``*.chat.completions.create/parse/stream`` (OpenAI chat, incl. ``beta.``)
* ``*.responses.create/parse/stream`` (OpenAI Responses)
* ``*.completions.create`` (OpenAI legacy completions, ``prompt=``)
* ``*.messages.create/stream`` (Anthropic) when it passes ``model=`` and
  ``max_tokens=`` or ``messages=`` (this keeps the OpenAI Assistants
  ``threads.messages.create`` out)
* ``openai.ChatCompletion.create`` (old OpenAI SDK)
* ``litellm.completion/acompletion`` (provider read from the model name)
* LangChain ``ChatOpenAI(...)`` / ``AzureChatOpenAI(...)`` / ``ChatAnthropic(...)``
  constructors (model only; LangChain prompts are built elsewhere)

Prompt text is read when it is a literal: a string, an f-string, ``+`` of
strings, ``"...".format(...)``, ``textwrap.dedent(...)``, ``.strip()``, or a
name assigned one of those in the same function or at module level.

JavaScript / TypeScript (``.js .jsx .mjs .cjs .ts .tsx .mts .cts``) uses a
small tokenizer, not a parser. It knows strings, template literals (with
``${...}``), comments and brackets, and finds the same SDK calls plus the
Vercel AI SDK (``generateText`` / ``streamText`` / ``generateObject`` /
``streamObject`` with ``model: openai("...")`` or ``anthropic("...")``).
Limits (v0): a regular expression literal that contains a quote or a bracket
can confuse it; prompts are read only from string literals in the call or
from a ``const``/``let``/``var`` assigned a string literal in the same file;
a function's name is found for declarations, methods, and arrow functions
assigned to a name.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_UNKNOWN = "unknown"

PY_SUFFIXES = (".py",)
JS_SUFFIXES = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts")
SKIP_DIRS = frozenset({
    "node_modules", ".git", ".venv", "venv", "env", "site-packages", "dist", "build", "__pycache__",
    ".next", ".nuxt", "coverage", ".tox", ".mypy_cache", "vendor", "third_party",
})
MAX_FILE_BYTES = 1_000_000
# A template needs this many literal characters before Throttle uses it to
# recognise traffic; "{text}" alone would match every request.
MIN_MATCH_CHARS = 12
TAG_HEADER = "x-throttle-job"
_TAG_COMMENT = re.compile(r"(?:#|//)\s*throttle-job:\s*([A-Za-z0-9._:/@-]+)")
_GREP_PATTERN = r"completions|responses\.|messages\.|ChatCompletion|litellm|ChatOpenAI|ChatAnthropic|generateText|streamText|generateObject|streamObject"


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Template:
    """Prompt text with slots. ``parts`` alternates literal, slot name,
    literal ... and always has odd length."""

    parts: tuple[str, ...]

    @classmethod
    def literal(cls, text: str) -> "Template":
        return cls((text,))

    @classmethod
    def slot(cls, name: str) -> "Template":
        return cls(("", name, ""))

    def __add__(self, other: "Template") -> "Template":
        head, tail = list(self.parts), list(other.parts)
        merged = head[:-1] + [head[-1] + tail[0]] + tail[1:]
        return Template(tuple(merged))

    @property
    def literals(self) -> list[str]:
        return [_collapse(p) for p in self.parts[0::2]]

    @property
    def slots(self) -> list[str]:
        return list(self.parts[1::2])

    @property
    def literal_chars(self) -> int:
        return sum(len(p) for p in self.literals)

    @property
    def text(self) -> str:
        out = []
        for i, part in enumerate(self.parts):
            out.append(part if i % 2 == 0 else "{" + part + "}")
        return "".join(out)

    def key(self) -> str:
        """What the fingerprint hashes: literal text with whitespace collapsed
        and every slot as one marker, so renaming a variable is not a change."""
        return "\x00".join(self.literals).strip()

    def fill(self, values: Mapping[str, str]) -> str:
        out = []
        for i, part in enumerate(self.parts):
            out.append(part if i % 2 == 0 else str(values.get(part, "")))
        return "".join(out)

    def regex(self) -> re.Pattern[str] | None:
        if self.literal_chars < MIN_MATCH_CHARS:
            return None
        pieces = [re.escape(lit) for lit in self.literals]
        return re.compile(r"\s*" + r".*?".join(pieces).strip() + r"\s*", re.DOTALL)

    def matches(self, text: str, *, min_chars: int = MIN_MATCH_CHARS) -> bool:
        """Does ``text`` fit this template, with any text in the slots?
        Whitespace is ignored. Linear time: the first literal must start the
        text, the last must end it, and the ones between must appear in
        order (leftmost first, which is exact for wildcard slots).
        ``min_chars=0`` checks a short template too (for confirming a job
        that is already known, not for recognising one)."""
        if self.literal_chars < min_chars:
            return False
        lits = [_squash(p) for p in self.parts[0::2]]
        hay = _squash(text)
        if len(lits) == 1:
            return hay == lits[0]
        first, middle, last = lits[0], lits[1:-1], lits[-1]
        if not hay.startswith(first) or not hay.endswith(last) or len(first) + len(last) > len(hay):
            return False
        pos, stop = len(first), len(hay) - len(last)
        for lit in middle:
            found = hay.find(lit, pos, stop)
            if found < 0:
                return False
            pos = found + len(lit)
        return True

    def to_json(self) -> list[str]:
        return list(self.parts)

    @classmethod
    def from_json(cls, data: Any) -> "Template | None":
        if not isinstance(data, list) or len(data) % 2 != 1 or not all(isinstance(p, str) for p in data):
            return None
        return cls(tuple(data))


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text)


_WS = re.compile(r"\s+")


def _squash(text: str) -> str:
    return _WS.sub("", text)


# ---------------------------------------------------------------------------
# Call sites
# ---------------------------------------------------------------------------


@dataclass
class CallSite:
    path: str                       # repo-relative, forward slashes
    line: int
    function: str                   # qualified name, or "<module>"
    language: str                   # python | javascript
    provider: str                   # openai | anthropic | unknown
    api: str                        # chat.completions | responses | completions | messages | litellm | langchain | ai-sdk
    model: str | None = None        # literal model name, None when set at run time
    model_expr: str | None = None   # the code that sets it, when not a literal
    system: Template | None = None
    user: Template | None = None    # the first user message
    prompt_dynamic: bool = False    # part of the prompt is built at run time
    has_tools: bool = False
    wants_json: bool = False
    tag: str | None = None
    ordinal: int = 1
    # sha256 (first 16 hex) of the call's arguments as written (tools, schemas,
    # few-shot messages, temperature ...); None when not known.
    call_digest: str | None = None

    @property
    def job(self) -> str:
        if self.tag:
            return self.tag
        base = f"{self.path}::{self.function}"
        return base if self.ordinal == 1 else f"{base}#{self.ordinal}"

    @property
    def prompt_fingerprint(self) -> str | None:
        """sha256 (first 16 hex) of the prompt templates, or None when no
        prompt text could be read."""
        known = [t for t in (self.system, self.user) if t is not None]
        if not known or sum(t.literal_chars for t in known) == 0:
            return None
        payload = {"system": self.system.key() if self.system else None,
                   "user": self.user.key() if self.user else None}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def approval_key(self) -> str:
        """What a job eval and an approval are tied to: the prompt fingerprint
        plus the model, tools, JSON mode and provider. A change to any of them
        needs a new eval before the job can move again."""
        payload = {"prompt": self.prompt_fingerprint, "model": self.model or self.model_expr,
                   "tools": self.has_tools, "json": self.wants_json, "provider": self.provider,
                   "call": self.call_digest}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def shape(self) -> dict[str, Any]:
        return {"model": self.model or self.model_expr, "tools": self.has_tools, "json": self.wants_json}

    @property
    def match_template(self) -> tuple[str, Template] | None:
        """Which message recognises this job's traffic: the system prompt when
        it has enough literal text, else the first user message."""
        if self.system is not None and self.system.literal_chars >= MIN_MATCH_CHARS:
            return "system", self.system
        if self.user is not None and self.user.literal_chars >= MIN_MATCH_CHARS:
            return "user", self.user
        return None

    @property
    def location(self) -> str:
        return f"{self.path}:{self.line}"

    def to_json(self) -> dict[str, Any]:
        return {
            "job": self.job, "path": self.path, "line": self.line, "function": self.function,
            "language": self.language, "provider": self.provider, "api": self.api,
            "model": self.model, "model_expr": self.model_expr,
            "system": self.system.to_json() if self.system else None,
            "user": self.user.to_json() if self.user else None,
            "system_text": self.system.text if self.system else None,
            "user_text": self.user.text if self.user else None,
            "prompt_dynamic": self.prompt_dynamic, "has_tools": self.has_tools,
            "wants_json": self.wants_json, "tag": self.tag, "ordinal": self.ordinal,
            "call_digest": self.call_digest, "prompt_fingerprint": self.prompt_fingerprint, "approval_key": self.approval_key,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "CallSite":
        return cls(
            path=str(data["path"]), line=int(data.get("line") or 0), function=str(data.get("function") or "<module>"),
            language=str(data.get("language") or "python"), provider=str(data.get("provider") or PROVIDER_UNKNOWN),
            api=str(data.get("api") or ""), model=data.get("model"), model_expr=data.get("model_expr"),
            system=Template.from_json(data.get("system")), user=Template.from_json(data.get("user")),
            prompt_dynamic=bool(data.get("prompt_dynamic")), has_tools=bool(data.get("has_tools")),
            wants_json=bool(data.get("wants_json")), tag=data.get("tag"), ordinal=int(data.get("ordinal") or 1),
            call_digest=data.get("call_digest"),
        )


def _number(sites: list[CallSite]) -> list[CallSite]:
    seen: dict[tuple[str, str], int] = {}
    for site in sorted(sites, key=lambda s: (s.path, s.line)):
        if site.tag:
            continue
        key = (site.path, site.function)
        seen[key] = seen.get(key, 0) + 1
        site.ordinal = seen[key]
    return sorted(sites, key=lambda s: (s.path, s.line))


def _comment_tag(lines: Sequence[str], line: int) -> str | None:
    for index in (line - 1, line - 2):
        if 0 <= index < len(lines):
            found = _TAG_COMMENT.search(lines[index])
            if found:
                return found.group(1)
    return None


def _provider_from_model(model: str | None) -> str:
    if not model:
        return PROVIDER_UNKNOWN
    lowered = model.lower()
    if lowered.startswith("anthropic/") or lowered.startswith("claude"):
        return PROVIDER_ANTHROPIC
    return PROVIDER_OPENAI


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


_CHAT_ENDINGS = {"create", "parse", "stream", "acreate"}
_LANGCHAIN = {"ChatOpenAI": PROVIDER_OPENAI, "AzureChatOpenAI": PROVIDER_OPENAI, "ChatAnthropic": PROVIDER_ANTHROPIC}


def _chain(node: ast.expr) -> list[str]:
    names: list[str] = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
    elif isinstance(node, ast.Call):
        names.extend(reversed(_chain(node.func)))
    return list(reversed(names))


def _classify_py(chain: list[str], kwargs: Mapping[str, ast.expr], litellm_names: set[str]) -> tuple[str, str] | None:
    if not chain:
        return None
    last = chain[-1]
    tail = chain[-3:]
    if len(chain) >= 3 and tail[0] == "chat" and tail[1] == "completions" and last in _CHAT_ENDINGS:
        return PROVIDER_OPENAI, "chat.completions"
    if len(chain) >= 2 and chain[-2] == "responses" and last in _CHAT_ENDINGS:
        return PROVIDER_OPENAI, "responses"
    if len(chain) >= 2 and chain[-2] == "ChatCompletion" and last in _CHAT_ENDINGS:
        return PROVIDER_OPENAI, "chat.completions"
    if len(chain) >= 2 and chain[-2] == "messages" and last in _CHAT_ENDINGS:
        if "model" in kwargs and ("max_tokens" in kwargs or "messages" in kwargs):
            return PROVIDER_ANTHROPIC, "messages"
        return None
    if len(chain) >= 2 and chain[-2] == "completions" and last in _CHAT_ENDINGS and "prompt" in kwargs:
        return PROVIDER_OPENAI, "completions"
    if last in ("completion", "acompletion") and (chain[0] == "litellm" or (len(chain) == 1 and last in litellm_names)):
        return PROVIDER_UNKNOWN, "litellm"
    if len(chain) >= 1 and last in _LANGCHAIN:
        return _LANGCHAIN[last], "langchain"
    return None


class _PyEnv:
    """Name -> assigned expression, for one module and its functions."""

    def __init__(self, tree: ast.Module) -> None:
        self.module = self._assignments(tree.body, deep=False)

    @staticmethod
    def _assignments(body: Iterable[ast.stmt], *, deep: bool) -> dict[str, ast.expr]:
        found: dict[str, ast.expr] = {}
        nodes: Iterable[ast.AST] = body
        if deep:
            nodes = [n for stmt in body for n in ast.walk(stmt)]
        for node in nodes:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                found[node.targets[0].id] = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                found[node.target.id] = node.value
        return found

    def for_function(self, func: ast.AST | None) -> dict[str, ast.expr]:
        scope = dict(self.module)
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope.update(self._assignments(func.body, deep=True))
        return scope


def _py_template(node: ast.expr | None, env: Mapping[str, ast.expr], depth: int = 0) -> Template | None:
    """The template for an expression, or None when it is not text we can read."""
    if node is None or depth > 8:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return Template.literal(node.value)
    if isinstance(node, ast.JoinedStr):
        out = Template.literal("")
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                out = out + Template.literal(value.value)
            elif isinstance(value, ast.FormattedValue):
                out = out + Template.slot(ast.unparse(value.value))
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _py_template(node.left, env, depth + 1)
        right = _py_template(node.right, env, depth + 1)
        if left is None and right is None:
            return None
        left = left or Template.slot(ast.unparse(node.left))
        right = right or Template.slot(ast.unparse(node.right))
        return left + right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        left = _py_template(node.left, env, depth + 1)
        if left is None or len(left.parts) != 1:
            return left
        pieces = re.split(r"%(?:\([^)]*\))?[sdrfi]", left.parts[0])
        out = Template.literal(pieces[0])
        for i, piece in enumerate(pieces[1:], start=1):
            out = out + Template.slot(f"arg{i}") + Template.literal(piece)
        return out
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "format":
            base = _py_template(func.value, env, depth + 1)
            if base is not None and len(base.parts) == 1:
                return _brace_template(base.parts[0])
            return base
        if isinstance(func, ast.Attribute) and func.attr in ("strip", "lstrip", "rstrip"):
            return _py_template(func.value, env, depth + 1)
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
        if name == "dedent" and node.args:
            return _py_template(node.args[0], env, depth + 1)
        return None
    if isinstance(node, ast.Name) and node.id in env:
        target = env[node.id]
        if target is node:
            return None
        return _py_template(target, env, depth + 1)
    return None


def _brace_template(text: str) -> Template:
    """``"Hi {name}".format(...)`` style: ``{x}`` become slots, ``{{`` stays literal."""
    parts: list[str] = [""]
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "{" and text.startswith("{{", i):
            parts[-1] += "{"
            i += 2
            continue
        if ch == "}" and text.startswith("}}", i):
            parts[-1] += "}"
            i += 2
            continue
        if ch == "{":
            end = text.find("}", i)
            if end == -1:
                parts[-1] += text[i:]
                break
            parts.append(text[i + 1:end].split(":")[0].split("!")[0] or "arg")
            parts.append("")
            i = end + 1
            continue
        parts[-1] += ch
        i += 1
    return Template(tuple(parts))


def _content_template(node: ast.expr | None, env: Mapping[str, ast.expr]) -> tuple[Template | None, bool]:
    """(template, dynamic) for a message ``content`` (a string or a list of
    ``{"type": "text", "text": ...}`` parts)."""
    if node is None:
        return None, False
    if isinstance(node, ast.List):
        out: Template | None = None
        dynamic = False
        for element in node.elts:
            if isinstance(element, ast.Dict):
                fields = _dict_fields(element)
                piece = _py_template(fields.get("text"), env) if "text" in fields else None
                if piece is None:
                    dynamic = True
                    continue
                out = piece if out is None else out + Template.literal("\n") + piece
        return out, dynamic
    template = _py_template(node, env)
    if template is None:
        return Template.slot(ast.unparse(node)), True
    return template, False


def _dict_fields(node: ast.Dict) -> dict[str, ast.expr]:
    out: dict[str, ast.expr] = {}
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            out[key.value] = value
    return out


def _py_messages(node: ast.expr | None, env: Mapping[str, ast.expr]) -> tuple[Template | None, Template | None, bool]:
    """(system, first user, dynamic) from a ``messages=[...]`` literal."""
    if isinstance(node, ast.Name) and node.id in env:
        node = env[node.id]
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None, None, node is not None
    system = user = None
    dynamic = False
    for element in node.elts:
        if not isinstance(element, ast.Dict):
            dynamic = True
            continue
        fields = _dict_fields(element)
        role = fields.get("role")
        role_name = role.value if isinstance(role, ast.Constant) and isinstance(role.value, str) else None
        template, dyn = _content_template(fields.get("content"), env)
        if role_name in ("system", "developer") and system is None:
            dynamic = dynamic or dyn
            system = template
        elif role_name == "user" and user is None:
            user = template
    return system, user, dynamic


def _py_tag(kwargs: Mapping[str, ast.expr]) -> str | None:
    for name in ("extra_headers", "headers", "default_headers"):
        node = kwargs.get(name)
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value.lower() == TAG_HEADER
                        and isinstance(value, ast.Constant) and isinstance(value.value, str)):
                    return value.value
        if isinstance(node, ast.Call):
            func = node.func
            fname = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            if fname in ("throttle_job", "job_headers") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    return arg.value
    return None


def _uses_json(kwargs: Mapping[str, ast.expr], api: str) -> bool:
    if "response_format" in kwargs or "output_format" in kwargs:
        return True
    if api == "messages" and "output_config" in kwargs:
        return True
    if api == "responses" and "text" in kwargs:
        return True
    return False


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


# Arguments that do not change the answer (the tag, transport settings).
_NOT_ANSWER_ARGS = frozenset({"extra_headers", "timeout", "headers", "max_retries", "user", "metadata", "store"})


def _py_call_digest(call: ast.Call, names: Mapping[str, ast.expr]) -> str:
    """The call's arguments as written; a name is replaced by what it is
    assigned in the function or module (one level), so editing a ``TOOLS``
    list defined elsewhere in the file is a change too."""
    parts = [_py_shape(a) for a in call.args]
    for kw in sorted(call.keywords, key=lambda k: k.arg or ""):
        if kw.arg in _NOT_ANSWER_ARGS:
            continue
        value = kw.value
        if isinstance(value, ast.Name) and value.id in names:
            value = names[value.id]
        parts.append(f"{kw.arg}={_py_shape(value)}")
    return _digest("\n".join(parts))


class _Anonymize(ast.NodeTransformer):
    """Variable names and f-string holes become placeholders, so renaming a
    variable is not a change (the same rule as the prompt fingerprint)."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="_", ctx=node.ctx), node)

    def visit_FormattedValue(self, node: ast.FormattedValue) -> ast.AST:
        return ast.copy_location(ast.Constant(value="{}"), node)


def _py_shape(node: ast.expr) -> str:
    import copy

    return ast.unparse(_Anonymize().visit(copy.deepcopy(node)))


def scan_python(path: str, source: str) -> list[CallSite]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    try:
        return _scan_python_tree(path, source, tree)
    except RecursionError:
        # very deeply nested code (generated files): skip the file, never crash the scan
        return []


def _scan_python_tree(path: str, source: str, tree: ast.Module) -> list[CallSite]:
    env = _PyEnv(tree)
    lines = source.splitlines()
    litellm_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "litellm":
            litellm_names.update(alias.asname or alias.name for alias in node.names)
    sites: list[CallSite] = []

    def visit(node: ast.AST, scope: list[str], func: ast.AST | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, scope + [child.name], child)
                continue
            if isinstance(child, ast.ClassDef):
                visit(child, scope + [child.name], func)
                continue
            if isinstance(child, ast.Call):
                site = _py_call(child, scope, func)
                if site is not None:
                    sites.append(site)
            visit(child, scope, func)

    def _py_call(call: ast.Call, scope: list[str], func: ast.AST | None) -> CallSite | None:
        kwargs = {k.arg: k.value for k in call.keywords if k.arg}
        found = _classify_py(_chain(call.func), kwargs, litellm_names)
        if found is None:
            return None
        provider, api = found
        names = env.for_function(func)
        model_node = kwargs.get("model")
        if model_node is None and api == "langchain":
            model_node = kwargs.get("model_name")
        model_t = _py_template(model_node, names) if model_node is not None else None
        model = model_t.parts[0] if model_t is not None and len(model_t.parts) == 1 else None
        model_expr = None if model or model_node is None else ast.unparse(model_node)
        system = user = None
        dynamic = False
        if api in ("chat.completions", "litellm"):
            system, user, dynamic = _py_messages(kwargs.get("messages"), names)
        elif api == "messages":
            if "system" in kwargs:
                system, dyn = _content_template(kwargs["system"], names)
                dynamic = dynamic or dyn
            _, user, _ = _py_messages(kwargs.get("messages"), names)
        elif api == "responses":
            if "instructions" in kwargs:
                system, dyn = _content_template(kwargs["instructions"], names)
                dynamic = dynamic or dyn
            inp = kwargs.get("input")
            if isinstance(inp, (ast.List, ast.Tuple)) or (isinstance(inp, ast.Name) and isinstance(names.get(inp.id), ast.List)):
                sys2, user, dyn = _py_messages(inp, names)
                system = system or sys2
                dynamic = dynamic or dyn
            else:
                user, _ = _content_template(inp, names)
        elif api == "completions":
            user, _ = _content_template(kwargs.get("prompt"), names)
        elif api == "langchain":
            dynamic = True
        if api == "litellm":
            provider = _provider_from_model(model)
        if any(k.arg is None for k in call.keywords):
            dynamic = True
        return CallSite(
            path=path, line=call.lineno, function=".".join(scope) or "<module>", language="python",
            provider=provider, api=api, model=model, model_expr=model_expr, system=system, user=user,
            prompt_dynamic=dynamic, has_tools="tools" in kwargs or "functions" in kwargs,
            wants_json=_uses_json(kwargs, api), tag=_py_tag(kwargs) or _comment_tag(lines, call.lineno),
            call_digest=_py_call_digest(call, names),
        )

    visit(tree, [], None)
    return sites


# ---------------------------------------------------------------------------
# JavaScript / TypeScript
# ---------------------------------------------------------------------------


@dataclass
class _Tok:
    kind: str          # ident | str | tmpl | punct | num | other
    text: str
    start: int
    value: Any = None  # str: the string value; tmpl: Template


_IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_NUM = re.compile(r"\d[\d_.xXa-fA-FeEn]*")
_PUNCT2 = ("?.", "=>", "...", "==", "!=", "<=", ">=", "&&", "||", "??", "++", "--", "+=", "-=")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}


def _js_string(src: str, i: int) -> tuple[str, int]:
    quote = src[i]
    out: list[str] = []
    i += 1
    while i < len(src):
        ch = src[i]
        if ch == "\\" and i + 1 < len(src):
            nxt = src[i + 1]
            out.append(_ESCAPES.get(nxt, nxt))
            i += 2
            continue
        if ch == quote or ch == "\n":
            return "".join(out), i + 1
        out.append(ch)
        i += 1
    return "".join(out), i


def _js_template(src: str, i: int) -> tuple[Template, int]:
    """Parse a template literal starting at the backtick at ``i``."""
    parts: list[str] = [""]
    i += 1
    while i < len(src):
        ch = src[i]
        if ch == "\\" and i + 1 < len(src):
            nxt = src[i + 1]
            parts[-1] += _ESCAPES.get(nxt, nxt)
            i += 2
            continue
        if ch == "`":
            return Template(tuple(parts)), i + 1
        if ch == "$" and src.startswith("${", i):
            end = _skip_balanced(src, i + 1)
            parts.append(src[i + 2:end - 1].strip() or "expr")
            parts.append("")
            i = end
            continue
        parts[-1] += ch
        i += 1
    return Template(tuple(parts)), i


def _skip_balanced(src: str, i: int) -> int:
    """From an opening ``{`` at ``i``, the index just past its match."""
    depth = 0
    while i < len(src):
        ch = src[i]
        if ch in "'\"":
            _, i = _js_string(src, i)
            continue
        if ch == "`":
            _, i = _js_template(src, i)
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def js_tokens(src: str) -> list[_Tok]:
    tokens: list[_Tok] = []
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        if ch.isspace():
            i += 1
            continue
        if src.startswith("//", i):
            end = src.find("\n", i)
            i = n if end == -1 else end
            continue
        if src.startswith("/*", i):
            end = src.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if ch in "'\"":
            value, end = _js_string(src, i)
            tokens.append(_Tok("str", src[i:end], i, value))
            i = end
            continue
        if ch == "`":
            value, end = _js_template(src, i)
            tokens.append(_Tok("tmpl", src[i:end], i, value))
            i = end
            continue
        match = _IDENT.match(src, i)
        if match:
            tokens.append(_Tok("ident", match.group(), i))
            i = match.end()
            continue
        match = _NUM.match(src, i)
        if match:
            tokens.append(_Tok("num", match.group(), i))
            i = match.end()
            continue
        two = next((p for p in _PUNCT2 if src.startswith(p, i)), None)
        if two:
            tokens.append(_Tok("punct", two, i))
            i += len(two)
            continue
        tokens.append(_Tok("punct", ch, i))
        i += 1
    return tokens


_OPEN = {"(": ")", "[": "]", "{": "}"}
_JS_KEYWORDS = {"if", "for", "while", "switch", "catch", "with", "return", "function", "async", "await", "new", "typeof"}
_AI_SDK = {"generateText", "streamText", "generateObject", "streamObject"}


def _match_close(tokens: Sequence[_Tok], i: int) -> int:
    """Index of the token closing the bracket at ``i`` (or len(tokens))."""
    depth = 0
    for j in range(i, len(tokens)):
        text = tokens[j].text if tokens[j].kind == "punct" else ""
        if text in _OPEN:
            depth += 1
        elif text in (")", "]", "}"):
            depth -= 1
            if depth == 0:
                return j
    return len(tokens)


def _match_open(tokens: Sequence[_Tok], j: int) -> int:
    depth = 0
    for i in range(j, -1, -1):
        text = tokens[i].text if tokens[i].kind == "punct" else ""
        if text in (")", "]", "}"):
            depth += 1
        elif text in _OPEN:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _object_fields(tokens: Sequence[_Tok], start: int, end: int) -> dict[str, tuple[int, int]]:
    """Top-level ``key: value`` spans of the object literal ``tokens[start:end+1]``
    (``start`` is ``{``). Shorthand ``{ model }`` maps to the identifier."""
    fields: dict[str, tuple[int, int]] = {}
    i = start + 1
    while i < end:
        tok = tokens[i]
        if tok.kind in ("ident", "str") and i + 1 < end and tokens[i + 1].text == ":":
            key = tok.value if tok.kind == "str" else tok.text
            j = i + 2
            value_start = j
            while j < end and tokens[j].text != ",":
                if tokens[j].kind == "punct" and tokens[j].text in _OPEN:
                    j = _match_close(tokens, j)
                j += 1
            fields[str(key)] = (value_start, j - 1)
            i = j + 1
            continue
        if tok.kind == "ident" and i + 1 <= end and tokens[i + 1].text in (",", "}"):
            fields[tok.text] = (i, i)
            i += 2
            continue
        if tok.kind == "punct" and tok.text in _OPEN:
            i = _match_close(tokens, i) + 1
            continue
        i += 1
    return fields


class _JsFile:
    def __init__(self, path: str, src: str) -> None:
        self.path = path
        self.src = src
        self.tokens = js_tokens(src)
        self.lines = src.splitlines()
        self.consts = self._consts()
        self.scopes = self._scopes()

    def _consts(self) -> dict[str, tuple[int, int]]:
        found: dict[str, tuple[int, int]] = {}
        toks = self.tokens
        for i, tok in enumerate(toks[:-3]):
            if tok.kind == "ident" and tok.text in ("const", "let", "var") and toks[i + 1].kind == "ident":
                k = i + 2
                if toks[k].text == ":":   # TypeScript annotation: skip to '='
                    while k < len(toks) and toks[k].text not in ("=", ";"):
                        k += 1
                if k < len(toks) and toks[k].text == "=":
                    end = k + 1
                    while end < len(toks) and toks[end].text not in (";",) and not (
                        toks[end].kind == "ident" and toks[end].text in ("const", "let", "var", "function", "export")
                    ):
                        if toks[end].kind == "punct" and toks[end].text in _OPEN:
                            end = _match_close(toks, end)
                        end += 1
                    found.setdefault(toks[i + 1].text, (k + 1, end - 1))
        return found

    def _scopes(self) -> list[tuple[int, int, str]]:
        """(open index, close index, name) for every named function or class body."""
        toks = self.tokens
        scopes: list[tuple[int, int, str]] = []
        for i, tok in enumerate(toks):
            if tok.kind != "punct" or tok.text != "{":
                continue
            name = self._body_name(i)
            if name:
                scopes.append((i, _match_close(toks, i), name))
        return scopes

    def _body_name(self, i: int) -> str | None:
        toks = self.tokens
        if i == 0:
            return None
        prev = toks[i - 1]
        # class Foo {  /  class Foo extends Bar {
        for back in range(i - 1, max(-1, i - 6), -1):
            if toks[back].kind == "ident" and toks[back].text == "class" and back + 1 < i and toks[back + 1].kind == "ident":
                return toks[back + 1].text
            if toks[back].kind == "punct":
                break
        j = i - 1
        if prev.kind == "punct" and prev.text == "=>":
            j = i - 2
            if j >= 0 and toks[j].kind == "ident":          # x => {
                open_at = j
            elif j >= 0 and toks[j].text == ")":
                open_at = _match_open(toks, j)
            else:
                return None
            k = open_at - 1
            if k >= 0 and toks[k].text == "async":
                k -= 1
            if k >= 1 and toks[k].text == "=" and toks[k - 1].kind == "ident":
                return toks[k - 1].text
            if k >= 1 and toks[k].text == ":" and toks[k - 1].kind in ("ident", "str"):
                return str(toks[k - 1].value or toks[k - 1].text)
            return None
        # skip a TypeScript return type: ): Promise<string> {
        while j >= 0 and toks[j].text != ")" and not (toks[j].kind == "punct" and toks[j].text in (";", "{", "}", "=")):
            j -= 1
        if j < 0 or toks[j].text != ")":
            return None
        open_at = _match_open(toks, j)
        if open_at <= 0:
            return None
        before = toks[open_at - 1]
        if before.kind == "ident" and before.text not in _JS_KEYWORDS:
            return before.text
        if before.kind == "ident" and before.text == "function":
            k = open_at - 2
            if k >= 0 and toks[k].text == "async":
                k -= 1
            if k >= 1 and toks[k].text == "=" and toks[k - 1].kind == "ident":
                return toks[k - 1].text
        return None

    def function_at(self, index: int) -> str:
        names = [name for start, end, name in self.scopes if start < index < end]
        return ".".join(names) or "<module>"

    def line_of(self, index: int) -> int:
        return self.src.count("\n", 0, self.tokens[index].start) + 1

    # -- values ----------------------------------------------------------

    def template(self, start: int, end: int, depth: int = 0) -> Template | None:
        """The template for ``tokens[start:end+1]``: string pieces joined by
        ``+``; any other piece becomes a slot. None when no piece is text."""
        toks = self.tokens
        if depth > 6 or start > end:
            return None
        pieces: list[Template] = []
        has_text = False
        i = start
        while i <= end:
            j = i
            while j <= end and not (toks[j].kind == "punct" and toks[j].text == "+"):
                if toks[j].kind == "punct" and toks[j].text in _OPEN:
                    j = _match_close(toks, j)
                j += 1
            piece, text = self._piece(i, min(j, end + 1) - 1, depth)
            if piece is not None:
                pieces.append(piece)
                has_text = has_text or text
            i = j + 1
        if not pieces or not has_text:
            return None
        out = pieces[0]
        for piece in pieces[1:]:
            out = out + piece
        return out

    def _piece(self, start: int, end: int, depth: int) -> tuple[Template | None, bool]:
        toks = self.tokens
        if start > end:
            return None, False
        while start < end and toks[start].text == "(" and _match_close(toks, start) == end:
            start, end = start + 1, end - 1
        first = toks[start]
        # "text".trim() and friends: a string followed only by method calls
        tail_ok = all(t.kind == "ident" or (t.kind == "punct" and t.text in (".", "(", ")")) for t in toks[start + 1:end + 1])
        if first.kind == "str" and tail_ok:
            return Template.literal(first.value), True
        if first.kind == "tmpl" and tail_ok:
            return first.value, True
        if start == end and first.kind == "ident" and first.text in self.consts:
            cs, ce = self.consts[first.text]
            inner = self.template(cs, ce, depth + 1)
            if inner is not None:
                return inner, True
        expr = self.src[first.start: toks[end].start + len(toks[end].text)].strip()
        return Template.slot(expr or "expr"), False

    def literal(self, start: int, end: int) -> str | None:
        t = self.template(start, end)
        if t is not None and len(t.parts) == 1:
            return t.parts[0]
        return None

    def resolve_span(self, start: int, end: int) -> tuple[int, int]:
        """Follow a lone identifier to its ``const`` value."""
        toks = self.tokens
        seen = 0
        while start == end and toks[start].kind == "ident" and toks[start].text in self.consts and seen < 4:
            start, end = self.consts[toks[start].text]
            seen += 1
        return start, end

    def content(self, start: int, end: int) -> tuple[Template | None, bool]:
        start, end = self.resolve_span(start, end)
        toks = self.tokens
        if toks[start].text == "[":
            out: Template | None = None
            dynamic = False
            close = _match_close(toks, start)
            i = start + 1
            while i < close:
                if toks[i].text == "{":
                    obj_end = _match_close(toks, i)
                    fields = _object_fields(toks, i, obj_end)
                    if "text" in fields:
                        piece = self.template(*fields["text"])
                        if piece is None:
                            dynamic = True
                        else:
                            out = piece if out is None else out + Template.literal("\n") + piece
                    i = obj_end + 1
                    continue
                i += 1
            return out, dynamic
        template = self.template(start, end)
        if template is None:
            expr = self.src[toks[start].start: toks[end].start + len(toks[end].text)].strip()
            return Template.slot(expr or "expr"), True
        return template, False

    def messages(self, start: int, end: int) -> tuple[Template | None, Template | None, bool]:
        start, end = self.resolve_span(start, end)
        toks = self.tokens
        if toks[start].text != "[":
            return None, None, True
        close = _match_close(toks, start)
        system = user = None
        dynamic = False
        i = start + 1
        while i < close:
            if toks[i].text == "{":
                obj_end = _match_close(toks, i)
                fields = _object_fields(toks, i, obj_end)
                role = self.literal(*fields["role"]) if "role" in fields else None
                if "content" in fields:
                    template, dyn = self.content(*fields["content"])
                else:
                    template, dyn = None, True
                if role in ("system", "developer") and system is None:
                    dynamic = dynamic or dyn
                    system = template
                elif role == "user" and user is None:
                    user = template
                i = obj_end + 1
                continue
            if toks[i].kind not in ("punct",):
                dynamic = True
            i += 1
        return system, user, dynamic


def _js_classify(chain: list[str], fields: Mapping[str, Any]) -> tuple[str, str] | None:
    last = chain[-1]
    if len(chain) >= 3 and chain[-3] == "chat" and chain[-2] == "completions" and last in _CHAT_ENDINGS:
        return PROVIDER_OPENAI, "chat.completions"
    if len(chain) >= 2 and chain[-2] == "responses" and last in _CHAT_ENDINGS:
        return PROVIDER_OPENAI, "responses"
    if len(chain) >= 2 and chain[-2] == "messages" and last in _CHAT_ENDINGS:
        if "model" in fields and ("max_tokens" in fields or "messages" in fields):
            return PROVIDER_ANTHROPIC, "messages"
        return None
    if len(chain) >= 2 and chain[-2] == "completions" and last in _CHAT_ENDINGS and "prompt" in fields:
        return PROVIDER_OPENAI, "completions"
    if len(chain) == 1 and last in _AI_SDK:
        return PROVIDER_UNKNOWN, "ai-sdk"
    return None


def scan_javascript(path: str, source: str) -> list[CallSite]:
    js = _JsFile(path, source)
    toks = js.tokens
    sites: list[CallSite] = []
    for i, tok in enumerate(toks):
        if tok.kind != "punct" or tok.text != "(" or i == 0 or toks[i - 1].kind != "ident":
            continue
        chain = [toks[i - 1].text]
        j = i - 2
        while j >= 1 and toks[j].text in (".", "?.") and toks[j - 1].kind == "ident":
            chain.insert(0, toks[j - 1].text)
            j -= 2
        if j >= 0 and toks[j].text in (".", "?."):
            chain.insert(0, "?")   # e.g. this.client.chat... or getClient().chat...
        if len(chain) == 1 and chain[0] not in _AI_SDK:
            continue
        if j >= 0 and toks[j].kind == "ident" and toks[j].text == "function":
            continue
        close = _match_close(toks, i)
        fields: dict[str, tuple[int, int]] = {}
        if i + 1 < close and toks[i + 1].text == "{":
            fields = _object_fields(toks, i + 1, _match_close(toks, i + 1))
        found = _js_classify(chain, fields)
        if found is None:
            continue
        provider, api = found
        model = model_expr = None
        if "model" in fields:
            ms, me = js.resolve_span(*fields["model"])
            model = js.literal(ms, me)
            if model is None and api == "ai-sdk" and toks[ms].kind == "ident" and ms + 2 <= me and toks[ms + 1].text == "(":
                fn = toks[ms].text
                provider = {"openai": PROVIDER_OPENAI, "anthropic": PROVIDER_ANTHROPIC}.get(fn, PROVIDER_UNKNOWN)
                if toks[ms + 2].kind == "str":
                    model = toks[ms + 2].value
            if model is None:
                model_expr = source[toks[ms].start: toks[me].start + len(toks[me].text)]
        system = user = None
        dynamic = False
        if api in ("chat.completions",) or (api in ("messages", "ai-sdk") and "messages" in fields):
            sys2, user, dynamic = js.messages(*fields["messages"]) if "messages" in fields else (None, None, True)
            system = sys2
        if api in ("messages", "ai-sdk") and "system" in fields:
            system, dyn = js.content(*fields["system"])
            dynamic = dynamic or dyn
        if api == "ai-sdk" and "prompt" in fields:
            user, _ = js.content(*fields["prompt"])
        if api == "responses":
            if "instructions" in fields:
                system, dyn = js.content(*fields["instructions"])
                dynamic = dynamic or dyn
            if "input" in fields:
                s, e = js.resolve_span(*fields["input"])
                if toks[s].text == "[":
                    sys2, user, dyn = js.messages(s, e)
                    system = system or sys2
                    dynamic = dynamic or dyn
                else:
                    user, _ = js.content(s, e)
        if api == "completions" and "prompt" in fields:
            user, _ = js.content(*fields["prompt"])
        if not fields:
            dynamic = True
        tag = None
        for key in ("headers", "extra_headers"):
            if key in fields:
                s, e = fields[key]
                if toks[s].text == "{":
                    inner = _object_fields(toks, s, _match_close(toks, s))
                    for name, span in inner.items():
                        if name.lower() == TAG_HEADER:
                            tag = js.literal(*span)
        # second argument, the SDK's request options:
        #   { headers: { "X-Throttle-Job": "name" } }  or  throttleJob("name")
        if tag is None and i + 1 < close and toks[i + 1].text == "{":
            k = _match_close(toks, i + 1) + 1
            if k + 1 < close and toks[k].text == ",":
                nxt = toks[k + 1]
                if nxt.text == "{":
                    opts = _object_fields(toks, k + 1, _match_close(toks, k + 1))
                    if "headers" in opts and toks[opts["headers"][0]].text == "{":
                        s = opts["headers"][0]
                        for name, span in _object_fields(toks, s, _match_close(toks, s)).items():
                            if name.lower() == TAG_HEADER:
                                tag = js.literal(*span)
                elif nxt.kind == "ident" and nxt.text in ("throttleJob", "throttle_job"):
                    if k + 3 < close and toks[k + 3].kind == "str":
                        tag = toks[k + 3].value
        line = js.line_of(i)
        if api == "ai-sdk" and provider == PROVIDER_UNKNOWN:
            provider = _provider_from_model(model)
        sites.append(CallSite(
            path=path, line=line, function=js.function_at(i), language="javascript", provider=provider, api=api,
            model=model, model_expr=model_expr, system=system, user=user, prompt_dynamic=bool(dynamic),
            has_tools="tools" in fields or "functions" in fields,
            wants_json="response_format" in fields or (api == "ai-sdk" and chain[-1].endswith("Object")),
            tag=tag or _comment_tag(js.lines, line),
            call_digest=_digest(" ".join(t.text for t in toks[i + 1:close])),
        ))
    return sites


# ---------------------------------------------------------------------------
# Files and git refs
# ---------------------------------------------------------------------------


def wanted(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if any(p in SKIP_DIRS for p in parts[:-1]):
        return False
    return path.endswith(PY_SUFFIXES) or path.endswith(JS_SUFFIXES)


def scan_source(path: str, source: str) -> list[CallSite]:
    if path.endswith(PY_SUFFIXES):
        return scan_python(path, source)
    if path.endswith(JS_SUFFIXES):
        return scan_javascript(path, source)
    return []


def scan_files(files: Mapping[str, str]) -> list[CallSite]:
    """Scan ``{repo-relative path: source}``; returns sites with job ids set."""
    sites: list[CallSite] = []
    for path in sorted(files):
        if wanted(path):
            sites.extend(scan_source(path, files[path]))
    return _number(sites)


def scan_dir(root: Path) -> list[CallSite]:
    root = Path(root)
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if not path.is_file() or not wanted(rel):
            continue
        if path.is_symlink() and not path.resolve().is_relative_to(root.resolve()):
            continue  # never read a file outside the repo
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            files[rel] = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return scan_files(files)


class GitError(RuntimeError):
    pass


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    try:
        done = subprocess.run(["git", "-C", str(repo), *args], input=input_bytes, capture_output=True, check=False)
    except FileNotFoundError:
        raise GitError("git is not installed") from None
    if done.returncode not in (0, 1):
        raise GitError(done.stderr.decode("utf-8", "replace").strip() or f"git {' '.join(args)} failed")
    if done.returncode == 1 and "grep" not in args[:3]:
        raise GitError(done.stderr.decode("utf-8", "replace").strip() or f"git {' '.join(args)} failed")
    return done.stdout


def resolve_ref(repo: Path, ref: str) -> str:
    out = _git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    sha = out.decode().strip()
    if not sha:
        raise GitError(f"unknown git ref {ref!r}")
    return sha


def read_ref(repo: Path, ref: str) -> dict[str, str]:
    """Sources at ``ref`` of the files that might call an AI API."""
    sha = resolve_ref(repo, ref)
    # -z and quotePath=false: paths with non-ASCII letters come back as they are
    listed = _git(repo, "-c", "core.quotePath=false", "grep", "-z", "-l", "-I", "-E", _GREP_PATTERN, sha, "--")
    paths = []
    for item in listed.decode("utf-8", "replace").split("\0"):
        path = item[len(sha) + 1:] if item.startswith(sha + ":") else item
        if path and "\n" not in path and wanted(path):
            paths.append(path)
    if not paths:
        return {}
    batch = "".join(f"{sha}:{p}\n" for p in paths).encode()
    raw = _git(repo, "cat-file", "--batch", input_bytes=batch)
    files: dict[str, str] = {}
    pos = 0
    for path in paths:
        header_end = raw.index(b"\n", pos)
        header = raw[pos:header_end].decode()
        pos = header_end + 1
        if header.endswith("missing"):
            continue
        size = int(header.split()[-1])
        body = raw[pos:pos + size]
        pos += size + 1
        if size <= MAX_FILE_BYTES:
            files[path] = body.decode("utf-8", "replace")
    return files


def scan_ref(repo: Path, ref: str) -> list[CallSite]:
    return scan_files(read_ref(Path(repo), ref))
