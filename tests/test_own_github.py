"""throttle pr-comment: one sticky comment per pull request. Offline: the
GitHub API is a fake opener."""

from __future__ import annotations

import json

import pytest

from throttle.cli import build_parser
from throttle.own import github as gh
from throttle.own.prcheck import MARKER


class FakeGitHub:
    def __init__(self, comments=None, status=200):
        self.comments = list(comments or [])
        self.status = status
        self.calls = []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, headers, json.loads(body) if body else None))
        if self.status != 200:
            return self.status, {"message": "Resource not accessible by integration"}
        path = url.split("api.test", 1)[1]
        if method == "GET":
            page = int(path.rsplit("page=", 1)[1])
            return 200, self.comments[(page - 1) * 100: page * 100]
        if method == "POST":
            comment = {"id": 900 + len(self.comments), "body": json.loads(body)["body"],
                       "html_url": f"https://gh/c/{900 + len(self.comments)}",
                       "user": {"login": "github-actions[bot]", "type": "Bot"}}
            self.comments.append(comment)
            return 201, comment
        if method == "PATCH":
            cid = int(path.rsplit("/", 1)[1])
            for comment in self.comments:
                if comment["id"] == cid:
                    comment["body"] = json.loads(body)["body"]
                    return 200, comment
        return 404, {"message": "not found"}


def client(fake):
    return gh.Client("t0ken", api="https://api.test", opener=fake)


def test_creates_then_updates_one_comment():
    fake = FakeGitHub([{"id": i, "body": "someone else"} for i in range(150)])
    action, url = gh.post_sticky(client(fake), "o/r", 7, MARKER + "\nfirst")
    assert action == "created" and url == "https://gh/c/1050"
    assert [c[0] for c in fake.calls] == ["GET", "GET", "POST"]          # paged through 150 comments
    assert fake.calls[0][2]["Authorization"] == "Bearer t0ken"
    action, _ = gh.post_sticky(client(fake), "o/r", 7, MARKER + "\nsecond")
    assert action == "updated" and fake.comments[-1]["body"].endswith("second")
    assert sum(1 for c in fake.comments if MARKER in c["body"]) == 1
    action, _ = gh.post_sticky(client(fake), "o/r", 7, MARKER + "\nsecond")
    assert action == "unchanged"


def test_no_changes_posts_nothing_new_but_updates_an_old_comment():
    fake = FakeGitHub()
    body = MARKER + "\n" + gh.NO_CHANGES + "\n"
    assert gh.post_sticky(client(fake), "o/r", 1, body)[0] == "skipped"
    fake = FakeGitHub([{"id": 5, "body": MARKER + "\nold", "user": {"login": "github-actions[bot]", "type": "Bot"}}])
    assert gh.post_sticky(client(fake), "o/r", 1, body)[0] == "updated"


def test_permission_error_explains_the_fix():
    with pytest.raises(gh.GitHubError, match="pull-requests: write"):
        gh.post_sticky(client(FakeGitHub(status=403)), "o/r", 1, "x")


def test_cli_reads_the_action_environment(tmp_path, monkeypatch, capsys):
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"number": 42}}))
    body = tmp_path / "c.md"
    body.write_text(MARKER + "\nhello")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    fake = FakeGitHub()
    args = build_parser().parse_args(["pr-comment", "--body-file", str(body), "--api", "https://api.test"])
    assert gh.handle(args, opener=fake) == 0
    assert "/repos/o/r/issues/42/comments" in fake.calls[-1][1]
    assert "created" in capsys.readouterr().out
    monkeypatch.delenv("GITHUB_TOKEN")
    assert gh.handle(args, opener=fake) == 2
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    assert gh.handle(args, opener=FakeGitHub(status=403)) == 1


def test_never_edits_a_persons_comment_that_contains_the_marker():
    # Anyone can paste the hidden marker. Throttle must not rewrite that
    # comment (its report would appear under the person's name, and they
    # could later edit "the Throttle report").
    planted = {"id": 5, "body": MARKER + "\nlooks good, Throttle says ready to move",
               "user": {"login": "mallory", "type": "User"}}
    fake = FakeGitHub([planted])
    action, _ = gh.post_sticky(client(fake), "o/r", 7, MARKER + "\nreal report")
    assert action == "created"
    assert not [c for c in fake.calls if c[0] == "PATCH"]
    assert fake.comments[0]["body"].startswith(MARKER + "\nlooks good")
    # with --author, only that login's comment is Throttle's
    fake = FakeGitHub([{"id": 6, "body": MARKER, "user": {"login": "other-bot[bot]", "type": "Bot"}}])
    action, _ = gh.post_sticky(client(fake), "o/r", 7, MARKER + "\nx", author="my-app[bot]")
    assert action == "created"


def test_fork_pull_request_without_write_token_is_not_a_failure(tmp_path, monkeypatch):
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"number": 3, "pull_request": {
        "number": 3, "head": {"repo": {"full_name": "fork/r"}}, "base": {"repo": {"full_name": "o/r"}}}}))
    body = tmp_path / "c.md"
    body.write_text(MARKER + "\nhi")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    args = build_parser().parse_args(["pr-comment", "--body-file", str(body), "--api", "https://api.test"])
    assert gh.handle(args, opener=FakeGitHub(status=403)) == 0
    event.write_text(json.dumps({"number": 3, "pull_request": {
        "number": 3, "head": {"repo": {"full_name": "o/r"}}, "base": {"repo": {"full_name": "o/r"}}}}))
    assert gh.handle(args, opener=FakeGitHub(status=403)) == 1
