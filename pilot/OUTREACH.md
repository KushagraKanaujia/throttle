# Operator outreach

This file prepares copy; it does not authorize an external post or message.

## Venue

Post first in the [vLLM Forum Benchmarking
category](https://discuss.vllm.ai/c/benchmarking/18). It is the best match:
vLLM's [current contact guidance](https://docs.vllm.ai/en/stable/community/contact_us/)
directs fellow-user discussion to the forum, and the category has current,
highly relevant configuration-specific threads. Follow the [forum
guidelines](https://discuss.vllm.ai/guidelines), use one topic, and do not
duplicate it across forum categories.

Do not use GitHub Discussions: vLLM's pinned notice says it is [no longer
used](https://github.com/vllm-project/vllm/discussions). Slack is positioned
for contribution and development coordination, not cold pilot recruitment.
Do not use Discord: it appears in older material but is absent from current
official vLLM contact guidance, so its current operator audience is unverified.

If the forum yields no qualified reply after three business days, use
[r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) once as the fallback. Read
its [current rule update](https://www.reddit.com/r/LocalLLaMA/comments/1su3ao4/rlocalllama_rule_updates/)
first, disclose that you built Throttle, meet its
self-promotion/participation requirements (including the current 1-in-10
guideline), disclose AI assistance in preparing the draft, remove any link if
moderators prefer that, and rewrite the post personally rather than pasting
generated copy.

## Ready-to-post forum copy

**Title: Looking for one vLLM operator with a real config decision to benchmark**

> I built Throttle, a local benchmark CLI, and am looking for one person who
> operates a self-hosted vLLM staging endpoint and is currently deciding
> between two settings that the same fixed workload can exercise—ideally a
> batching or `max_num_seqs` choice—against an existing latency, TTFT,
> throughput, or error target. I built the tool and will personally guide the
> run.
>
> The ask is concrete: let me guide one bounded benchmark against your setup
> and give you the sanitized baseline/candidate result. You run it on a machine
> you control, so the endpoint and credential stay with you. Before traffic we
> agree on the prompts, request/token/time/error/spend limits, and inspect a
> zero-traffic plan. The tool sends inference requests only; it does not
> provision, SSH into, restart, or reconfigure your server. You make and revert
> any config change. A failed or malformed response invalidates the result.
>
> This is not a signup or a request for opinions about the tool. I need one real
> pending decision and permission to follow up on what, if anything, you do
> after seeing the result. If that matches your setup, reply or message me with
> your vLLM version, coarse model/GPU class, SLO, and the two configurations
> you are considering. Please do not send an endpoint URL, API key, prompts, or
> employer-sensitive details.

Before posting, replace any phrasing that does not sound like you. Do not add a
speedup, savings, "AI optimizer," waitlist, or YC claim.

## Qualification reply

Send only after someone expresses concrete interest:

> Thanks. Before we schedule anything, can you confirm five things without
> sharing secrets?
>
> 1. You operate or are authorized to load-test a non-production vLLM endpoint.
> 2. The endpoint has a concrete, measurable latency, TTFT, throughput, or
>    error target.
> 3. You have two real configurations you expect to choose between; this is not
>    a hypothetical benchmark.
> 4. You can use approved, non-sensitive measured and warm-up prompts during a
>    controlled window.
> 5. For metered infrastructure, you know the billing rate and can approve a
>    hard session cap. For owned/non-metered hardware, you can approve a hard
>    traffic and wall-time window.
>
> If yes, I will send a short operator summary, the detailed safety contract,
> and the run card first. You keep the endpoint and key local throughout.

Do not qualify someone solely because they own a GPU. If they lack a current
decision or SLO, thank them without turning them into a pilot success.
The current decision-grade comparison also requires immutable model/image
provenance and the same GPU/load shape across baseline and candidate. A pure
GPU-sizing or concurrency-only decision is out of scope for this first pilot;
do not imply otherwise to win a reply.

## Posting cadence

1. After explicit owner approval, the owner publishes one vLLM Forum topic.
2. Answer substantive replies within one day; do not mass-DM forum members.
3. Wait three business days.
4. If there is no qualified lead, obtain a second owner approval before using
   the human-rewritten Reddit fallback once.
5. If both fail, record the response counts before changing copy or channel.
