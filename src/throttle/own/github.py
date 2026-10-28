"""``throttle pr-comment``: post the pr-check comment on a GitHub pull request.

It keeps ONE comment per pull request: the first run creates it, later runs
edit it in place (found by a hidden marker line). It talks to the GitHub REST
API with the token in ``$GITHUB_TOKEN`` (the workflow's own token is enough
with ``pull-requests: write``). It needs no Throttle server and sends nothing
anywhere else. In a GitHub Action, the repository and the pull request number
are read from ``$GITHUB_REPOSITORY`` and ``$GITHUB_EVENT_PATH``.

When a pull request has no AI changes and no Throttle comment yet, nothing
is posted. If it had a comment before (the AI change was later reverted), the
comment is updated to say so.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .prcheck import MARKER

API = "https://api.github.com"
NO_CHANGES = "### Throttle: no AI calls changed"
EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2

# (method, url, headers, body bytes or None) -> (status, parsed JSON)
Opener = Callable[[str, str, dict[str, str], bytes | None], tuple[int, Any]]


class GitHubError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def urllib_opener(method: str, url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, Any]:
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed https API host
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw) if raw else None
        except ValueError:
            payload = None
        return exc.code, payload


class Client:
    def __init__(self, token: str, *, api: str = API, opener: Opener | None = None) -> None:
        self._token = token
        self.api = api.rstrip("/")
        self.opener = opener or urllib_opener

    def call(self, method: str, path: str, payload: Any = None) -> Any:
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "throttle-pr-check",
        }
        body = None
        if payload is not None:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        status, data = self.opener(method, self.api + path, headers, body)
        if not 200 <= status < 300:
            message = data.get("message") if isinstance(data, dict) else None
            hint = " (does the workflow have `permissions: pull-requests: write`?)" if status in (401, 403, 404) else ""
            raise GitHubError(f"GitHub said {status} to {method} {path}: {message or 'no message'}{hint}", status)
        return data


def is_ours(comment: Any, author: str | None = None) -> bool:
    """A Throttle comment: it has the marker AND a bot wrote it (by default
    the workflow token's ``github-actions[bot]``, or any app bot). Anyone can
    type the marker into a comment; Throttle must never take over, rewrite or
    put its report under a person's comment."""
    if not isinstance(comment, dict) or MARKER not in str(comment.get("body") or ""):
        return False
    user = comment.get("user") or {}
    if author:
        return user.get("login") == author
    return user.get("type") == "Bot"


def find_comment(client: Client, repo: str, pr: int, author: str | None = None) -> dict[str, Any] | None:
    page = 1
    while page <= 30:
        batch = client.call("GET", f"/repos/{repo}/issues/{pr}/comments?per_page=100&page={page}")
        if not isinstance(batch, list) or not batch:
            return None
        for comment in batch:
            if is_ours(comment, author):
                return comment
        if len(batch) < 100:
            return None
        page += 1
    return None


def post_sticky(client: Client, repo: str, pr: int, body: str, *, author: str | None = None) -> tuple[str, str | None]:
    """("created" | "updated" | "skipped", comment URL)."""
    if MARKER not in body:
        body = MARKER + "\n" + body
    existing = find_comment(client, repo, pr, author)
    if existing is None:
        if NO_CHANGES in body:
            return "skipped", None
        made = client.call("POST", f"/repos/{repo}/issues/{pr}/comments", {"body": body})
        return "created", (made or {}).get("html_url")
    if (existing.get("body") or "").strip() == body.strip():
        return "unchanged", existing.get("html_url")
    done = client.call("PATCH", f"/repos/{repo}/issues/comments/{existing['id']}", {"body": body})
    return "updated", (done or {}).get("html_url")


def _event(path: str | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    try:
        event = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return event if isinstance(event, dict) else {}


def from_fork(path: str | None) -> bool:
    """Is the pull request in this event from another repository? Its
    workflow token is read-only, so a comment cannot be posted."""
    pr = _event(path).get("pull_request") or {}
    head = ((pr.get("head") or {}).get("repo") or {}).get("full_name")
    base = ((pr.get("base") or {}).get("repo") or {}).get("full_name")
    return bool(head and base and head != base)


def pr_from_event(path: str | None) -> int | None:
    if not path or not Path(path).exists():
        return None
    try:
        event = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    number = (event.get("pull_request") or {}).get("number") or event.get("number")
    return int(number) if isinstance(number, int) else None


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "pr-comment",
        help="post the pr-check comment on a GitHub pull request (one comment, edited in place)",
        description="Create or update one Throttle comment on a pull request, using $GITHUB_TOKEN.",
    )
    parser.add_argument("--body-file", type=Path, required=True, help="the Markdown from `throttle pr-check --markdown`")
    parser.add_argument("--repo", default=None, help="owner/name (default: $GITHUB_REPOSITORY)")
    parser.add_argument("--pr", type=int, default=None, help="pull request number (default: from $GITHUB_EVENT_PATH)")
    parser.add_argument("--token-env", default="GITHUB_TOKEN", metavar="VAR", help="variable with the token (default GITHUB_TOKEN)")
    parser.add_argument("--api", default=os.environ.get("GITHUB_API_URL", API), help="GitHub API URL (for GitHub Enterprise)")
    parser.add_argument("--author", default=None, metavar="LOGIN",
                        help="login that owns Throttle's comment, e.g. my-app[bot] (default: any bot account)")


def handle(args: argparse.Namespace, *, opener: Opener | None = None) -> int:
    repo = args.repo or os.environ.get("GITHUB_REPOSITORY")
    pr = args.pr or pr_from_event(os.environ.get("GITHUB_EVENT_PATH"))
    token = os.environ.get(args.token_env)
    if not repo or not pr:
        print("throttle pr-comment: pass --repo owner/name and --pr N (they are read from the GitHub Action "
              "environment when it runs on a pull_request event)", file=sys.stderr)
        return EXIT_USAGE
    if not token:
        print(f"throttle pr-comment: ${args.token_env} is not set", file=sys.stderr)
        return EXIT_USAGE
    try:
        body = args.body_file.read_text(encoding="utf-8")
        action, url = post_sticky(Client(token, api=args.api, opener=opener), repo, pr, body,
                                  author=getattr(args, "author", None))
    except GitHubError as exc:
        if exc.status in (401, 403) and from_fork(os.environ.get("GITHUB_EVENT_PATH")):
            print("throttle pr-comment: this pull request comes from a fork, so the workflow token cannot comment. "
                  "The check is in the job summary instead.", file=sys.stderr)
            return EXIT_OK
        print(f"throttle pr-comment: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except OSError as exc:
        print(f"throttle pr-comment: {exc}", file=sys.stderr)
        return EXIT_FAILED
    if action == "skipped":
        print("No AI calls changed and no earlier Throttle comment: nothing posted.")
    else:
        print(f"Throttle comment {action}: {url or '(no URL returned)'}")
    return EXIT_OK
