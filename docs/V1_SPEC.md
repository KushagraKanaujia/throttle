# Throttle v1 specification (historical)

> This document records the superseded schema-1 smoke prototype. Its
> “recommendation” terminology and simultaneous endpoint comparison are not
> present in Throttle 0.2. Current behavior is documented in the repository
> README and `REPORT_SCHEMA.md`; schema-1 artifacts are never decision-grade.

## Decision

Throttle v1 is a local, bring-your-own-endpoint benchmark CLI. The user supplies
an existing OpenAI-compatible inference endpoint; Throttle provisions no GPU.

This was chosen over a calculator because the project has only one valid
historical baseline and no validated optimization multiplier. Measuring the
user's actual model, hardware, server configuration, and workload is useful;
projecting savings from invented or borrowed multipliers is not.

## Five-minute value

One command accepts an endpoint URL, model, API key environment variable, GPU
count and hourly price, optional p95 latency SLO, and optional JSONL prompts.
Defaults are a bundled prompt set, fixed `max_tokens=128`, concurrency 1/4/8,
eight measured requests per level, and one unmeasured warmup per level.

Every level uses the same prompt sequence and request parameters:

- `temperature=0`;
- the same fixed `max_tokens` cap; and
- no stop tokens.

The report includes status/error counts, success rate, prompt/completion token
totals, wall duration, requests/second, output tokens/second, p50/p95 latency,
and observed GPU cost per one million output tokens. A level is valid only when
every measured request is HTTP 200 and reports a positive integer
`usage.completion_tokens`. Invalid levels never produce performance metrics,
recommendations, or A/B deltas.

The recommended concurrency is the valid level with the highest observed
output-token throughput that satisfies the optional p95 SLO. Without an SLO,
it is simply the highest-throughput valid level.

## Guarded A/B mode

An optional candidate endpoint runs the identical matrix. A concurrency-level
throughput/cost delta is available only when both sides are valid and their
completion-token totals differ by at most 5%, calculated as
`abs(a - b) / max(a, b)`. Otherwise the comparison is explicitly unavailable
with a structured reason.

## Deliverable

The concrete v1 output is a terminal report plus sanitized machine-readable
JSON. It describes the exact tested workload, validity failures, per-level
metrics, observed concurrency recommendation, and guarded A/B results. It is
not a universal performance result, realized-savings claim, or monthly
projection. Credentials, endpoint URLs, prompts, and generated text remain
local and are not written to the report.

## Stack

- Python 3.11+
- `argparse` and `asyncio`
- `httpx` as the only runtime dependency
- standard-library JSON, statistics/math, timing, dataclasses, and packaging
- local fake HTTP server/mock transport for tests

There is no web server, frontend, database, cloud SDK, container requirement,
account system, or telemetry.

## Explicit v1 deferrals

- automatic vLLM or TensorRT-LLM reconfiguration;
- GPU/pod provisioning and GPU/instance selection;
- replica autoscaling and spot orchestration;
- semantic/prefix caching;
- production traffic proxying;
- async job queues;
- non-OpenAI-compatible backends;
- distributed multi-host load generation;
- accounts, authentication, and teams;
- hosted dashboard, history database, or telemetry;
- production-log traffic-shape discovery;
- projected monthly-savings claims; and
- polished UI.

These are deferred until observed behavior from real operators justifies one
specific next step.
