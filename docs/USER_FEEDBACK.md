# User feedback process

Throttle does not collect automatic telemetry. Feedback is manual, opt-in, and
kept deliberately lightweight: public bugs and feature requests use the GitHub
issue forms, while private conversations are summarized in a ledger stored
outside this Git worktree.

The private ledger is the working record. This public document defines its
fields and redaction rules; real tester identities and private messages must
never be added here.

## Entry fields

Each entry records:

- `id`: stable local identifier, such as `2026-08-18-001`.
- `date`: ISO date when the feedback was received.
- `source`: `reddit`, `github`, `discord`, `dm`, `email`,
  `internal_testing`, or `other`.
- `person`: handle/name only after explicit permission; otherwise `null`.
- `person_reference_consent`: whether public attribution was explicitly
  approved.
- `what_tried`: mode/protocol plus a redacted command or short description.
- `feedback_summary`: a paraphrase of what broke or what the person said.
- `signal_stage`: the observed usage stage from the table below.
- `status`: `open`, `resolved`, or `wontfix`.
- `related_public_url`: public issue/PR/thread URL, or `null` for private
  sources.
- `resolution`: what changed or why no change is planned.
- `follow_up_on`: ISO follow-up date, or `null`.

## Signal stages

| Stage | Counts as |
| --- | --- |
| `attention` | Star, fork, reaction, or generic comment only |
| `intent` | Asked how to try it or described a real pending decision |
| `activation` | Installed it or ran plan/smoke |
| `valid_use` | Completed benchmark/compare/Golden, including an honest inconclusive result |
| `action` | Acted on the evidence or explicitly requested a rerun |
| `outcome_pull` | Returned unprompted with a result, problem, or follow-up decision |
| `unclassified` | Not enough evidence yet |

The stage records observed behavior, not enthusiasm. A “cool tool” comment is
`attention`; it is not evidence of use or an operator decision.

## Quick capture checklist

1. Paraphrase by default; do not retain a private message verbatim unless there
   is a clear reason and permission.
2. Leave `person` as `null` unless the person explicitly agreed to be
   referenced. Consent to a private conversation is not consent to public
   attribution.
3. Redact API keys, credentials, authorization headers, endpoint URLs,
   hostnames/IPs, raw prompts/responses, unreviewed logs/reports, private local
   paths, invoices, and employer/customer identities.
4. Record what they actually attempted and the first broken or confusing step.
5. Classify signal from behavior, set its status, and add a follow-up date when
   one is genuinely needed.
6. Link only public issues, PRs, or threads. Do not put private-message URLs in
   the ledger.

The canonical private ledger and its copy-ready entry template live outside the
repository. `/.feedback-private/` and `/feedback.jsonl` are ignored as a second
line of defense if a copy is ever placed in a checkout.
