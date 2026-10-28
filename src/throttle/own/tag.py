"""Tag each AI call with the job it belongs to. Optional, no dependencies.

Throttle recognises most calls by their prompt. A tag makes it exact, and
keeps the job's name the same when the code moves to another file::

    from throttle.own.tag import throttle_job

    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[...],
        extra_headers=throttle_job("ticket-summary"),
    )

The header only goes as far as ``throttle capture``, which reads it and does
not forward it. Without Throttle in front, the API ignores it. Any code can
set the header itself; this module is only a shortcut. In JavaScript::

    client.chat.completions.create({...}, { headers: { "X-Throttle-Job": "ticket-summary" } })

A comment works too, for the name in pull request checks (not at run time)::

    # throttle-job: ticket-summary
"""

from __future__ import annotations

import re

HEADER = "X-Throttle-Job"
_VALID = re.compile(r"^[A-Za-z0-9._:/@-]{1,120}$")


def throttle_job(name: str) -> dict[str, str]:
    """``{"X-Throttle-Job": name}``, ready for ``extra_headers=``."""
    if not _VALID.match(name):
        raise ValueError("a job name is 1-120 letters, digits or . _ : / @ -")
    return {HEADER: name}


job_headers = throttle_job
