# User testing and feedback guide

> **Current phase:** the first external operator pilot now uses the focused
> [operator-pilot packet](../pilot/README.md). This broader ten-operator guide is
> retained as historical planning context; do not use its legacy command flow
> as the current execution contract.

The goal of v1 testing is to learn whether real self-hosters can obtain a useful,
trustworthy measurement in minutes. It is not to collect opinions about a future
product or to manufacture a savings claim.

## Evidence status

There is no external usage behavior to summarize yet. The earlier 4.22-second
sequential vLLM baseline is engineering evidence, not demand evidence. The two
discarded runs (truncated output and HTTP 404 responses) are not product or
performance signal.

Start with a balanced pilot of ten qualified operators:

- four operating online/chat-style vLLM traffic;
- four operating async inference such as enrichment, embeddings, or evals; and
- two local or hobby operators for usability feedback only.

A qualified operator currently owns a self-hosted endpoint or recurring
inference job, knows its approximate engine/GPU/workload, and can run a small
test with approved inputs. Someone who is merely interested in self-hosting is
not a qualified operator.

## Where to recruit

Use one substantive technical thread first, then link to it from a small number
of relevant conversations. Do not duplicate-post or mass-DM.

1. [vLLM Forum — Benchmarking](https://discuss.vllm.ai/c/benchmarking/18) is
   the strongest first venue because the topic and operator audience match.
   Follow the [forum guidelines](https://discuss.vllm.ai/guidelines).
2. [vLLM Slack](https://slack.vllm.ai/) is useful for a short request that links
   back to the methodology thread. The
   [official vLLM project](https://github.com/vllm-project/vllm#contact-us)
   directs users to its forum and Slack; GitHub Discussions has been
   [retired in favor of the forum](https://github.com/vllm-project/vllm/discussions/15229).
3. [MLOps Community](https://mlops.community/blog/mlops-community-2-0) can reach
   production practitioners. Ask moderators where a hands-on inference test
   belongs and keep it free of sales or signup language.
4. [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) is relevant after
   genuine participation. Check its current rules, disclose "I built this," and
   have the human poster rewrite the post personally rather than pasting
   AI-written copy.
5. Use the [SGLang community](https://www.sglang.io/#community) or
   [TensorRT-LLM Discussions](https://github.com/NVIDIA/TensorRT-LLM/discussions)
   only after Throttle has actually been verified against that engine.

[r/selfhosted](https://www.reddit.com/r/selfhosted/) and
[r/MachineLearning](https://www.reddit.com/r/MachineLearning/) are later
channels: check their current promotion rules or megathreads before posting,
and do not treat broad attention there as operator validation.

## Non-salesy recruiting copy

Long forum version:

> **Looking for vLLM/OpenAI-compatible endpoint operators to test a small benchmark CLI**
>
> I built Throttle, an early CLI for benchmarking an endpoint you already
> operate. It sends the same fixed prompt set at concurrency 1, 4, and 8,
> rejects invalid levels, and reports observed throughput, latency, and cost.
> It can also compare two endpoints under identical test conditions.
>
> This is not a launch or an optimization claim. I am looking for actual runs,
> including failures, rather than opinions on the idea.
>
> Credentials remain local. Please do not share keys, endpoint URLs, prompts,
> responses, or employer details. Setup and methodology: [repository link].
>
> The most useful feedback is the engine/version and coarse GPU/model class,
> which levels were valid or rejected, the first confusing or blocked step,
> and whether the report caused a rerun or changed a deployment decision. A
> redacted receipt is welcome but optional.
>
> Failed requests are never performance data, and A/B comparisons use the same
> prompts and fixed `max_tokens`.

Short Slack version:

> I am looking for five operators willing to run a small CLI against an
> OpenAI-compatible endpoint they already manage. It measures the same fixed
> workload at concurrency 1/4/8, rejects invalid levels, and optionally compares
> two endpoints. I need observed runs or exact failure points, not feature
> ideas. Credentials stay local; method and limitations: [forum link].

Avoid waitlists, pricing surveys, speedup claims, "would you use this?", or any
number from an invalid benchmark. For r/LocalLLaMA, keep the facts but rewrite
the post in the poster's own voice and disclose the affiliation.

## Before a session

Ask the tester to choose a non-production endpoint or a controlled traffic
window, confirm the expected request count and cost, and use prompts they are
allowed to send. With defaults, warn them that each endpoint receives 27 real
requests (24 measured plus 3 warmups) and up to 3,456 requested output tokens.
A/B doubles those figures. Their server may log request content even though
Throttle's report does not.

Record only the context needed to interpret the result, with permission:

- serving engine/version and model identifier;
- GPU type/count and total hourly cost supplied to Throttle;
- exact Throttle command with URLs and key environment names redacted;
- whether traffic or cold starts could interfere; and
- the generated JSON report after checking it for local policy compliance.

Do not collect API keys, endpoint URLs, raw prompts, generated text, or unrelated
production logs.

## Suggested session

1. Start with one endpoint and the bundled prompts to verify connectivity.
2. Check that all levels are valid. Treat an invalid level as a diagnostic, not
   a performance result.
3. Repeat with a small, approved, representative JSONL sample.
4. If a real alternative server configuration already exists, use A/B mode. Do
   not change `max_tokens`, prompts, or the matrix between endpoints.
5. Ask what decision the result changed, if any, and what was confusing or
   missing. Do not prompt the tester toward a positive answer.
6. Follow up after 48--72 hours: ask what they actually changed or tested and,
   if nothing changed, what stopped them. Ask for aggregate before/after
   evidence only when a change was genuinely applied.

Never present stopping/truncating output as an optimization. Never cite latency,
throughput, cost, or deltas from an invalid level. Confirm that A/B deltas are
available before interpreting them.

## Signal rubric

Count each tester once at the highest stage reached and retain the underlying
evidence. Do not turn the stages into one vanity score.

| Stage | Evidence | Interpretation |
| --- | --- | --- |
| 0 — Attention | View, upvote, "cool," generic feature idea | Noise for product decisions |
| 1 — Intent | Click, install, repo star, setup question | Mild interest |
| 2 — Activation | Starts against a real endpoint/configuration | Useful |
| 3 — Valid use | Completes a technically valid run and receipt | Strong |
| 4 — Action | Reruns, compares, changes a deployment, or rejects a result for a concrete operational reason | Very strong |
| 5 — Outcome/pull | Supplies valid before/after evidence, returns unprompted, or asks for recurring automation | Strongest |

Concrete negative evidence from a qualified operator is stronger than praise.
"This violates our p95 SLO" or a repeated setup failure can direct the product;
"seems useful" cannot.

Initial review gates use raw counts:

- If fewer than five of 20 qualified, contextual invitations start a run,
  revisit the framing or recruiting channel.
- If at least five start but fewer than three complete because setup or inputs
  block them, fix activation before adding features.
- If at least five complete but fewer than two act because they distrust the
  result, add provenance, calibration, and reproducible comparisons.
- If ten qualified operators complete and nobody acts, returns, or supplies a
  concrete operational rejection, narrow or stop instead of broadening.
- Three independent action attempts plus two verified or repeated runs justify
  one more focused iteration; they do not prove a market.

## Minimal, opt-in tracking

Use a team-owned CSV or SQLite file plus the local redacted JSON receipt. Do not
add an analytics platform or automatic telemetry in v1. Suggested session fields:

```text
participant_id,source,cohort,tool_version,engine_version,gpu_class,
model_size_class,workload_type,benchmark_mode,started_at,
minutes_to_first_valid_result,blocked_step,decision_after_report,
decision_reason,reran_within_72h,returned_within_14d,evidence_link,
contact_permission
```

For benchmark validity, use one row per endpoint label and concurrency level:

```text
endpoint_label,prompt_set_hash,max_tokens,concurrency_level,level_valid,
invalid_reason,request_count,successful_request_count,completion_token_summary,
duration_seconds,requests_per_second,output_tokens_per_second,latency_p50_ms,
latency_p95_ms,observed_cost,comparison_valid,comparison_invalid_reason
```

Never collect API keys, authorization headers, endpoint hostnames/IPs, raw
prompts, generated text, company identity without consent, private invoices,
browser fingerprints, heatmaps, or session replay. A `?src=vllm_forum` label is
enough attribution. Receipts remain local unless the tester explicitly shares
one.

## Decide what to build from behavior

There are no observed behavior patterns yet, so no direction has earned the
next build. Once the pilot produces evidence, use this mapping:

| Observed behavior | Next build |
| --- | --- |
| Online-serving users repeatedly A/B configs, enforce TTFT/TPOT/p95 constraints, and want recurring runs | Option A: SLO-aware autotuner/experiment runner |
| Async users optimize cost/job, tolerate queueing/interruption, and ask for retries, checkpoints, or spot capacity | Option B: spot-native async plane |
| Qualified users refuse endpoint access for security/setup reasons | Local config or metrics import; do not host their credentials |
| Users fail while entering configs or discovering model/engine details | Config parsing and local autodiscovery |
| Users cannot reproduce or trust a result | Calibration, provenance, and a reproducible benchmark harness |
| Users apply one result but repeated testing is tedious | An experiment runner, the first credible wedge toward Option A |
| Reports are trusted but change control blocks action | Dry-run diffs, rollback guidance, and exportable evidence |
| Hobbyists engage but production operators do not act | Treat hobbyists as UX testers, not customer validation |
| Views/stars are high but real runs are rare | Distribution attention only; do not broaden features |

Only choose A or B from behavior in the relevant cohort. Require at least three
independent users to take an action and at least two to produce a verified
result, repeat run, or concrete operational rejection. Until then, report the
direction as **not yet observed**.
