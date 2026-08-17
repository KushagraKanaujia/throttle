# Throttle external-pilot safety contract

This is the shareable agreement for a founder-guided benchmark against an
operator's endpoint. Both sides complete the run card before traffic.

## Default execution model

The operator runs Throttle on a machine they control while the founder guides
the session. The API key is placed in a local environment variable and is never
pasted into chat, screen-shared, sent to the founder, included in the command,
or retained in an artifact.

Founder-operated execution is an exception. It requires written permission, a
least-privilege endpoint-specific short-lived key, an expiry/revocation owner,
and a source-IP allowlist. The first pilot should use the operator-run default.

## Authorization and scope

No traffic starts until the run card records:

- the authorized operator, controlled window, time zone, emergency stop
  channel, and explicit permission to generate load;
- a staging/non-production endpoint, or separately written approval for a
  production maintenance window;
- the real decision declared before results, including baseline and candidate;
- at least one predeclared measurable latency, TTFT, throughput/goodput, or
  error target, with exact p95/TTFT thresholds when those apply;
- approved measured prompts and disjoint warm-up prompts, `max_tokens`, load
  shape, and cache policy;
- model/full revision, vLLM version, GPU/count, immutable image digest, and
  runtime-verified effective flags;
- exactly one billing basis: a known metered rate, or an explicit
  operator-owned/non-metered attestation with hard traffic/time limits; and
- artifact ownership, recipients, retention date, and whether the founder may
  retain the sanitized result.

No external pilot uses plaintext HTTP, `--allow-insecure-http`, unapproved
production prompts, or the cross-check-only GuideLLM backend. Unknown cost is
never accepted for metered infrastructure. On operator-owned, non-metered
hardware it may be acknowledged explicitly, but internal opportunity/energy
cost remains unavailable and is not reported as zero.

## Hard traffic and budget envelope

The default authorization for metered infrastructure is **no more than USD
$5.00 from session start through teardown**, or a lower cap chosen by the
operator. Raising it requires a new written approval. This is a hard session
cap only when the operator enforces it with a provider-side limit/auto-stop or
a continuous wall-clock cutoff based on the full billable rate. Without one of
those mechanisms, do not describe the $5 value as an absolute provider billing
cap and do not run a metered pilot under this contract.

Before traffic, divide the authorization into a smoke allocation, one identical
per-benchmark allocation used by both variants, and a setup/restart/in-flight
reserve:

```text
smoke allocation + 2 × benchmark allocation + reserve <= session cap
```

The baseline and candidate must use the exact same
`--max-estimated-spend`, `--max-elapsed-seconds`, and all other safety limits;
changing a "remaining" cap between them makes their manifests incompatible.

For a dedicated hourly endpoint, set both per-run limits such that:

```text
declared total hourly price × max elapsed seconds / 3600 <= per-run allocation
```

The separate session clock starts before smoke and ends after baseline restore
and teardown. It includes idle/config-restart time and the worst case of all
already accepted in-flight requests reaching their timeout. Throttle's CLI
bound covers only the declared model and client measurement interval; it does
not discover storage, CPU, network, queue, idle, or delayed provider charges.
For serverless billing, require a provider-side active-spend guard. If the
billing model cannot support a conservative bound, do not run a metered pilot.

Default traffic ceiling for one two-configuration pilot at `max_tokens=128`:

| Phase | Requests | Reserved output tokens | Other hard limits |
| --- | ---: | ---: | --- |
| Smoke | 27 | 3,456 | 5 minutes, concurrency 8, first error stops, 1 MiB/response |
| Baseline | 204 | 26,112 | 10 minutes, agreed concurrency, first error stops, 1 MiB/response |
| Candidate | 204 | 26,112 | Same as baseline |
| Traffic maximum | 435 | 55,680 | Operator stop and the session clock always win |

The 204 benchmark requests are three warm-ups plus three measured blocks of 67,
for 201 measured requests. Any different envelope must be written into the run
card and approved before `plan`; it is never increased mid-run silently.
One flag/configuration decision on the same GPU is in scope. Pure GPU-sizing or
concurrency-only comparison is not decision-compatible in the current tool and
is rejected for this first pilot.

## Control-plane read-only guarantee

Use this precise statement:

> Throttle sends only the authorized OpenAI-compatible chat-completion requests
> and writes a local sanitized report. It does not provision, SSH, deploy,
> restart, scale, reconfigure, delete, or call control-plane/admin APIs. The
> operator performs every configuration change and rollback.

Do not claim that inference causes no server state change. Requests consume
capacity and may affect caches, logs, rate-limit counters, billing counters,
and other workload traffic. Those effects are disclosed and accepted in the
run card.

Native HTTPS traffic disables ambient proxy inheritance and redirects. The
operator still verifies the exact destination in `throttle plan` before saying
"go."

## Immediate stop conditions

Stop scheduling and treat the artifact as diagnostic if any of these occurs:

- the operator says stop;
- the first HTTP, transport, malformed-completion, stream, usage, or response
  size error occurs;
- the destination, model, credential scope, workload approval, or effective
  configuration is uncertain;
- production traffic or an SLO shows unexpected impact;
- background traffic or cache state breaks the agreed comparison;
- a request, token, elapsed, concurrency, error, response-size, or spend ceiling
  is reached; or
- the candidate cannot be verified or the baseline cannot be restored.

Client cancellation stops new scheduling but cannot recall an inference request
the server already accepted. Those requests may finish and bill, so the session
reserve covers the full in-flight ceiling through one request timeout.

A stopped, invalid, partial, smoke, incompatible, or inconclusive run is never
presented as a recommendation. It is retained only as an honest diagnostic.

## Teardown

1. Stop scheduling, cancel controlled in-flight work, and allow one request
   timeout for traffic to drain.
2. Confirm the endpoint's traffic rate and service health have returned to the
   starting baseline. Previously disclosed log, cache, usage-counter, and
   billing side effects may persist.
3. The operator restores the baseline configuration and verifies its SLO;
   Throttle never performs this step.
4. Revoke/expire the temporary key, remove any temporary IP rule, unset the
   secret environment variable, and confirm no benchmark process remains.
5. Review every artifact locally before sharing it. A partial artifact remains
   labeled partial.
6. Record the end time, stop reason, actual provider charge when it posts, and
   teardown confirmation.
7. Delete only pilot-created temporary prompt copies and reports on the agreed
   date. The operator decides retention of their original workload/evidence.

## Data and publication

Throttle reports omit URLs/hosts, credentials, prompts, completions, and raw
logs, but retain workload fingerprints and operational metadata. Fingerprints
can reveal equality or confirm a guessed low-entropy workload. The operator
reviews the report before sharing it.

Private by default. With explicit consent, retain only the redacted run card,
sanitized JSON and SHA-256, declared SLO/decision, validity outcome, teardown,
and observed follow-up behavior for at most 30 days. Company attribution,
quotes, raw configuration evidence, billing documents, or public results each
require separate approval. Never retain API keys, endpoint hosts, raw prompts,
outputs, or unrelated logs. Retention beyond 30 days requires a separate
written renewal.
