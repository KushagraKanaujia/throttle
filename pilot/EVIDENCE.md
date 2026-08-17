# Evidence rubric and YC-worthiness roadmap

The north-star metric is **verified operational decisions**, not signups,
reports, or compliments. "YC-worthy" is not a polish checklist or a guarantee
about selection. At this stage it means showing that a
specific operator has an urgent problem, trusts this method, changes behavior,
and pulls the product back into their workflow. The work below follows
[YC's essential startup advice](https://www.ycombinator.com/blog/ycs-essential-startup-advice/):
launch, stay close to users, do manual work, and build only from observed
behavior. It also follows YC's warning to [build for customers, not
VCs](https://www.ycombinator.com/blog/build-for-customers-not-vcs/).

## Count each operator once at the highest observed stage

| Stage | Required evidence | Interpretation |
| --- | --- | --- |
| 0 — Attention | View, upvote, star, "cool tool," generic idea | Noise |
| 1 — Qualified intent | Owns staging vLLM, SLO, pending decision, agrees to run card | Recruiting signal only |
| 2 — Activation | Runs `plan` and starts real endpoint traffic | Useful activation |
| 3 — Valid use | Completes a technically valid real-endpoint benchmark/comparison | Strong workflow evidence |
| 4 — Verified decision | Deploys, rejects, or deliberately defers a configuration based partly on the result, with evidence | Very strong |
| 5 — Pull | After a verified decision, returns unprompted with another real decision, makes an unsolicited qualified introduction, or pays for another run | Strongest |

A named blocker, rerun request, internal share, or scheduled follow-up is useful
diagnostic/engagement evidence, but it does not by itself advance an operator to
Stage 3 or 4 and cannot unlock a willingness-to-pay claim.

Negative behavior can be strong signal: retaining the baseline because the
candidate misses SLO, rejecting an untrustworthy artifact for a named reason,
or refusing endpoint access because of a concrete security boundary. Record the
behavior, not whether it flatters Throttle.

## Noise that must not enter traction claims

- likes, views, replies, stars, installs, or compliments without a real run;
- a hobby endpoint with no operational SLO or decision;
- smoke, synthetic, internal, invalid, partial, or contaminated results;
- feature wish lists given before the person attempts the current workflow;
- "I would pay" without a budget owner, transaction, or procurement step;
- founder-prompted agreement or a decision the operator had already made; and
- the existing internal golden artifact, which is engineering proof only.

## Evidence retained for each real pilot

With consent, retain:

- anonymous pilot ID and source-channel code;
- qualification facts and the decision/SLO declared before results;
- signed-off run card and zero-traffic plan approval;
- sanitized artifact and SHA-256, or fixed failure/stop reason;
- technical validity, decision eligibility, and exact uncertainty state;
- teardown confirmation and actual provider charge when available;
- the operator's immediate stated action;
- what they actually did 48–72 hours later; and
- whether they returned unprompted, referred someone, initiated a paid step, or
  requested another run within 14–30 days.

Never retain credentials, authorization headers, endpoint hosts/paths, raw
prompts, outputs, or raw server/client logs in pilot evidence, even with
consent. Store any necessary company identity or billing document outside the
de-identified ledger, under separate written purpose/consent and access control.

The CSV is deliberately code-only, with no free-text notes or URLs:

- `org_id`, `decision_proof_id`, and `evidence_packet_id` are random opaque IDs,
  never names, paths, or links;
- all event fields use ISO-8601 UTC timestamps, so decision, return, offer,
  payment, procurement, and renewal order can be recomputed;
- `segment_code` is `online_serving`, `async_batch`, or `hobby_ux`;
- `source_channel` is one of `vllm_forum`, `reddit`, `referral`, or
  `owner_authorized_direct`;
- `decision_kind` is `max_num_seqs` or `other_supported_engine_flag`;
- `incident_or_blocker_code` comes from the fixed run/report reason or one of
  `none`, `authorization_unavailable`, `security_boundary`, `slo_risk`,
  `billing_unbounded`, or `teardown_failed`;
- `same_day_action_code` is `none`, `deploy_planned`, `reject_planned`,
  `defer_planned`, `rerun_requested`, or `internal_share`;
- `verified_decision_code` is only `none`, `deployed`, `rejected`, or
  `deferred`; and
- `unprompted_return_for_real_decision` is true only when the timestamped return
  names a new operational decision; `introduction_kind` is `none`,
  `unsolicited_peer`, `unsolicited_internal`, or `solicited`; and
- payment columns are separate typed facts. `offer_id` refers to a versioned
  standard offer, not a custom description. Event timestamps prove ordering;
  return/introduction kind fields distinguish real pull from solicited contact.

If a value does not fit these fields, keep it in the separately protected run
card or do not retain it; never improvise a sensitive CSV note.

## Two-week operator sprint

### Days 0–3: recruit one qualified operator

- After explicit owner approval, the owner human-posts the single forum message.
- The owner replies to every substantive response within one day.
- Record counts for views/replies separately from qualified leads.
- Only after a second explicit owner approval, if no qualified lead appears
  after three business days, the owner may use the one Reddit fallback after
  checking account eligibility and current rules.

Gate: one person satisfies all qualification criteria and books a controlled
session. Until then there is no pilot traction.

### Days 3–10: run and observe

- Complete the run card and safety contract.
- Let the operator stay at the keyboard.
- Produce a valid, inconclusive, invalid, or stopped artifact honestly.
- Ask the neutral same-day behavior question and do not suggest the answer.
- Follow up once at 48–72 hours.

Gate: one external endpoint was genuinely exercised and its outcome plus
teardown is recorded. A technically valid report is valuable; an operator
action or concrete rejection is stronger.

### Days 10–14: decide, do not automatically build

Classify the highest behavior stage and the first irreversible point of
friction. A single request is not a product roadmap. Only prepare the next
pilot. Repeated behavior is logged for the post-phase decision; it does not
authorize code changes during this no-feature sprint.

## Evidence gates beyond the first pilot

### Gate A — Repeatable activation

Across at most 20 qualified, contextual invitations:

- at least 5 start against a real endpoint;
- at least 3 complete a valid run; and
- median time from approved run card to first valid result is under 30 minutes,
  excluding operator-controlled restart time.

If five start but fewer than three finish because of the same current-product
blocker, record it as the leading post-pilot build hypothesis. Do not implement
it during this no-feature evidence phase.

### Gate B — Decision value

Among the first five qualified completions:

- at least 2 make a verified deploy/reject/defer decision based partly on the
  evidence; and
- after a verified decision, at least 1 returns unprompted with another real
  decision within 14 days or makes an unsolicited introduction that produces a
  qualified operator.

If five complete and fewer than two act, the result is not changing decisions.
Revisit the problem/ICP before improving the tool.

### Gate C — Willingness to pay

Ask about payment only after at least three independent verified Stage-4
decisions. Offer one versioned, standardized paid engagement at a recorded
price, tied to the operator's next real decision. Evidence is a paid invoice or
a purchase/procurement process with a named budget owner and target date—not a
survey answer, internal share, rerun request, or casual meeting.

Gate: two independent accounts pay the standardized offer, or one account pays
and renews while a second begins a dated procurement/payment step. Record offer
ID, price, collected amount, procurement state/date, and renewal separately.
Track founder labor separately so services revenue is not mislabeled product
revenue.

### Gate D — Pull and wedge convergence

Before expanding scope, require:

- 5 qualified teams with valid evidence;
- 3 unprompted returns for another real decision within 30 days;
- 2 unsolicited peer/internal introductions that produce qualified operators;
  and
- one repeated decision pattern across at least 3 teams.

That repeated pattern chooses the wedge. Repeated SLO-aware config comparisons
support the autotuner/experiment-runner direction. Repeated async cost/job
decisions support the async direction. Mixed compliments support neither.

### Internal YC-evidence bar

Start one ten-week evidence clock when the first qualified external session
begins. Call the project **YC-evidence-ready** only if, by the earlier of that
ten-week deadline or completion of 10 qualified pilots, every threshold below
is met:

- at least 7 usable real-endpoint runs;
- at least 5 operator-confirmed deployment, rejection, or defer decisions;
- at least 4 operators returning unprompted for another real decision within
  30 days;
- at least 3 independent accounts with nonzero collected payment;
- at least 2 unsolicited teammate or peer introductions that become qualified
  operators;
- median synchronous founder support at or below 60 minutes; and
- one segment/decision pattern producing at least 60% of verified decisions and
  at least 2 of the 3 paying accounts.

These are internal evidence thresholds, not YC admissions criteria. If any is
missing at the deadline, the project is not YC-evidence-ready; apply the stop or
pivot rules below rather than softening the bar.

## Stop, narrow, or pivot rules

- Halt the pilot immediately for credential/sensitive-data exposure,
  unauthorized traffic, contaminated or misleading evidence, an unbounded
  spend state, or teardown failure. Record only a fixed incident code and do
  not continue recruitment until the boundary is safe.
- No qualified lead after the forum, Reddit fallback, and 20 contextual
  owner-authorized invitations: change the ICP/problem framing or stop; do not
  add features.
- Three or more qualified security refusals at the same boundary: treat that
  boundary as product evidence; do not dismiss it as sales friction.
- If more than half of qualified operators cannot obtain an authorized test
  environment, change the target workflow or customer segment.
- Five starts but fewer than three valid completions: record the repeated
  activation blocker and finish the no-feature phase before any implementation.
- Five valid completions but fewer than two verified Stage-4 decisions: the
  measurement is not valuable enough; narrow or reconsider the wedge.
- If fewer than 30% have another relevant decision recorded within 60 days,
  treat use as episodic; do not claim SaaS retention.
- If three of five pilots require more than two hours of bespoke founder
  analysis, do not call the workflow repeatable.
- If zero of five value-proven accounts pays or enters dated procurement within
  30 days of the standardized offer, revisit the buyer/packaging or stop the
  paid hypothesis.
- Ten valid qualified completions with no unprompted return, referral, or paid
  step: stop or pivot rather than polishing.
- Strong hobby usage without production-operator action: count it as usability
  evidence, not market validation.

## Weekly proof review

Every Friday, publish an internal one-page facts-only review:

```text
qualified leads / sessions booked / real starts / valid completions
verified decisions / rerun requests / unprompted returns / referrals
standardized offers / paid invoices / renewals / dated procurement events
median minutes to valid result / repeated blocker / evidence packet IDs
what was learned / what will not be built / next week's single experiment
```

Never report percentages without raw denominators. Never replace zero with a
story. The credible future YC narrative is the evidence trail: a narrow urgent
operator problem, trustworthy technical proof, fast founder-led learning,
operators acting on results, repeat usage, and eventually paid pull.
