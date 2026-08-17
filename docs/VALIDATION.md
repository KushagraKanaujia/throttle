# Throttle schema-1 validation (historical)

> This is preserved evidence from the earlier smoke prototype. It did not
> produce a live Throttle matrix or A/B report and must not be read as current
> benchmark or golden evidence. Throttle 0.2 rejects these schema-1 artifacts
> for saved-run decisions. See `GOLDEN_PROTOCOL.md` for the current gate.

Validation date: 2026-08-16/17 UTC

## Verdict

The local v1 package and its validity guards passed independent software
validation. A live RunPod attempt proved that one externally hosted
OpenAI-compatible Qwen3-8B endpoint could return a real HTTP 200 chat response
with positive token usage. It did **not** produce a live Throttle concurrency
matrix or a valid live A/B comparison.

Accordingly, none of the local fixture throughput/cost numbers are LLM or GPU
performance results, and there is no live optimization/speedup claim from this
session.

## Package and test checks

The following checks passed against the completed v1:

- Python 3.14.6, `throttle-bench` 0.1.0, and `httpx` 0.28.1.
- 12/12 `unittest` cases passed.
- `compileall` passed for `src` and `tests`.
- `pip check` reported no broken requirements.
- `throttle --help` loaded and exposed the documented arguments.
- A PEP 517 wheel built successfully, and inspection confirmed that it included
  `throttle/prompts.jsonl` and the console entry point.

The package declares Python 3.11 or newer, but this session did not separately
run the suite under Python 3.11, 3.12, or 3.13.

## Installed-CLI loopback validation

`validation/fake_openai_server.py` is a loopback-only deterministic HTTP
fixture, not an inference engine. It rejects an incorrect model, credential,
temperature, token cap, message shape, or any request containing `stop`. It
stores only prompt hashes for its audit endpoint.

The installed `.venv/bin/throttle` executable was exercised end to end:

| Case | Observed behavior | Result |
| --- | --- | --- |
| Valid baseline | 24/24 measured requests were HTTP 200 with positive completion usage across concurrency 1/4/8. | Exit 0; all levels valid and a recommendation was produced. |
| Valid guarded A/B | Both endpoints had 24/24 usable measured requests and identical completion-token totals at every level. | Exit 0; guarded comparisons were available. |
| 404/all invalid | 8/8 measured requests returned 404 across concurrency 1/4. | Exit 1; both levels invalid, throughput/latency/cost metrics null, no recommendation. |
| Token mismatch | Both sides returned HTTP 200 with positive usage, but candidate completion-token totals were 50% lower. | Exit 0; every A/B delta was suppressed with `completion_token_totals_outside_5_percent_tolerance`. |
| SLO miss | All measured requests were usable, but every valid level exceeded the 10 ms fixture SLO. | Exit 0; recommendation unavailable with `no_valid_level_meets_p95_slo`. |

The fixture audit additionally confirmed:

- every request used `max_tokens=40`;
- every request used `temperature=0`;
- no request contained `stop`;
- the A/B prompt multiset matched, and the sequential level's arrival order
  matched; concurrent network arrival order is intentionally not treated as a
  stable ordering signal; and
- generated JSON contained no endpoint URL, key, prompt text, response text, or
  `content` field.

These reports are sanitized but their timings and cost figures are
**synthetic/software-only**:

- `validation/local-baseline.json`
- `validation/local-ab-valid.json`
- `validation/local-all-invalid-404.json`
- `validation/local-ab-token-mismatch.json`
- `validation/local-slo-no-match.json`

They validate client behavior; they must not be quoted as model throughput,
latency, GPU cost, or savings.

## Live RunPod validation

### Safety and starting state

Before provisioning, the account had one stopped pod, no serverless endpoints,
no user templates, no network volumes, and an existing spend rate of
$0.007/hour. The stopped pod was never modified. Validation-session incremental
GPU spend began at zero, with a hard $3 cap.

Pod attempts used Community RTX 4090 at the observed $0.34/GPU-hour rate and
automatic termination guards. Serverless attempts used scale-to-zero,
one-worker maximums, 60-second idle timeouts, sequential activation, and a
30-minute A100 active-window cutoff. The pod vLLM bearer secret was generated
only for this session and removed after teardown. Serverless calls used the
pre-existing RunPod credential from the process environment without copying it
into any artifact.

### Pod attempts: invalid

The public RunPod `vLLM Latest` template referenced the mutable
`vllm/vllm-openai:latest` image.

1. The first two-pod attempt supplied `--model` based on the template's
   convention. The current mutable image instead exposes `vllm serve`, whose
   model argument is positional. Neither container published a runtime or port
   mapping; external model-route checks returned 404. The pods were deleted and
   no request was counted.
2. Two replacement pods used the corrected positional model argument and
   otherwise identical baseline/candidate settings. After an eight-minute
   bounded wait, both still had null control-plane runtime objects, no port
   mappings, and only external 404 responses. There was no evidence that a
   container or model download had begun. Both pods were deleted and no request
   was counted.

The mutable image never reached a state where its vLLM version, image digest,
GPU assignment, startup arguments, or inference behavior could be observed.

### First-party serverless worker: partial success only

The fallback used the first-party `runpod-workers/worker-vllm` release v2.25.1,
whose release metadata declares vLLM 0.27.1. Both generated templates referenced
the same versioned worker image and host-cached `Qwen/Qwen3-8B`.

The generated template configuration was verified directly:

| Setting | Baseline | Candidate |
| --- | ---: | ---: |
| `MODEL_NAME` | `Qwen/Qwen3-8B` | `Qwen/Qwen3-8B` |
| `MAX_MODEL_LEN` | 4096 | 4096 |
| `GPU_MEMORY_UTILIZATION` | 0.90 | 0.95 |
| `MAX_NUM_SEQS` | 256 | 2048 |
| `ENABLE_CHUNKED_PREFILL` | true | true |
| `MAX_CONCURRENCY` | 16 | 16 |
| `RAW_OPENAI_OUTPUT` | true | true |

Chunked prefill was deliberately identical and receives no optimization credit.
The worker release enables it by default, and current vLLM V1 behavior may also
do so. Merely adding `--enable-chunked-prefill` would not establish a new
optimization.

The initial ADA_24/RTX 4090 attempt produced an authenticated HTTP 200 model
listing from the baseline after a 116-second cold resume. This proved route and
model visibility only. It was not a generation result. The candidate pool was
throttled, so both endpoints were moved together to the exact configured type
`NVIDIA A100 80GB PCIe` for hardware comparability.

On A100:

- The baseline native warm trigger completed after 140,348 ms of queue delay
  and 249 ms of worker execution. A separate external OpenAI-compatible chat
  request then returned HTTP 200 in 1,149 ms, with one choice,
  `finish_reason=length`, 27 prompt tokens, 64 completion tokens, and 91 total
  tokens. The request used `temperature=0`, `max_tokens=64`, and no stop field.
  This is a legitimate single-request success, not a throughput benchmark.
- The candidate native warm trigger completed after 180,468 ms of queue delay
  and 226 ms of worker execution, just after the enforced three-minute cutoff.
  Workers were drained at the cutoff. The candidate never received the required
  separate external OpenAI chat request, so its native completion does not make
  it eligible for a Throttle A/B run.
- A pre-warm baseline chat attempt returned HTTP 500 with no usage and was
  discarded. Queue time, cold-start time, 404s, 500s, and requests without
  positive usage were never used as performance data.

No runtime `/version` response, container startup log, image digest, or parsed
vLLM engine configuration was captured. Therefore vLLM 0.27.1 is a release
metadata declaration, not an independently queried runtime version, and the
table above is verified template input rather than a captured engine log. The
runtime API did not expose a GPU subtype; the exact A100 type was verified as
control-plane configuration only.

### Why there is no live Throttle report

Throttle requires every measured request in a level to be HTTP 200 with positive
completion-token usage. The baseline endpoint met that condition for one manual
chat request. The candidate did not pass the separate external chat preflight
before cutoff, and both endpoints were not simultaneously warm and proven.

Running the matrix anyway would have repeated the project's prior invalid
pattern: timings from queueing, unavailable workers, or failed requests could be
mistaken for inference performance. The live concurrent matrix and A/B report
were therefore correctly not run.

## Cost accounting

The session stayed far below the $3 cap. RunPod billing-history queries had not
populated by teardown, so no exact incremental charge is available. Based on
observed resource rates and elapsed active/allocation windows, the conservative
session estimate is $0.30-$0.40. This is explicitly an estimate, not a billing
record.

Throttle's own wall-time-times-hourly-rate metric was not used as exact RunPod
serverless billing. Serverless cold queue time, worker-active time, and request
execution time are different intervals.

After teardown, the account spend rate returned to the exact starting rate of
$0.007/hour. Final resource checks found:

- the one pre-existing stopped pod still present and untouched;
- zero validation pods;
- zero serverless endpoints;
- zero user templates (Hub-created inline templates were removed with their
  endpoints); and
- zero network volumes.

The ephemeral pod vLLM key, job identifiers, and temporary raw HTTP responses
were removed. Attempts to separately delete the generated inline templates
returned not-found because endpoint deletion had already removed them; the
final user template count confirmed zero.

## Bugs and external tooling issues found

No Throttle implementation defect was identified by this validation pass.

External issues encountered:

- A mutable public vLLM template/image pairing had drifted from the documented
  model-argument convention. A pinned image digest and verified startup command
  are required for reproducible pod validation.
- `runpodctl serverless update --workers-max 0` did not apply zero in this
  session; the control REST API was needed to drain workers.
- The control REST API expected an exact GPU type name rather than the pool name
  for its GPU-type list.
- Billing data lagged past teardown, preventing exact same-session cost
  reconciliation.

These are deployment/tooling findings, not evidence of a Throttle client bug.

## What remains before any external claim

At minimum:

1. Pin the worker/pod image by immutable digest and capture the runtime vLLM
   version, image digest, CUDA/driver details, actual GPU model, and startup log
   showing the parsed effective engine configuration.
2. Prove both baseline and candidate with separate external authenticated chat
   HTTP 200 responses containing positive prompt and completion usage.
3. Keep both on the same GPU subtype and worker count. Use identical prompt
   order, `temperature=0`, fixed `max_tokens`, no stop/truncation, and the same
   request/concurrency matrix.
4. Run Throttle at concurrency 1/4/8 (or a justified larger matrix) with at
   least eight measured requests per level. Accept a level only if every request
   is HTTP 200 with positive completion usage; require the existing token-total
   tolerance before any A/B delta.
5. Repeat runs, reverse endpoint order, separate cold-start from warm inference,
   and reconcile actual provider billing after it posts.
6. Choose a comparison that can actually exercise the changed limit.
   `MAX_NUM_SEQS=256` versus 2048 cannot explain a gain at only eight concurrent
   requests unless another interaction is demonstrated. vLLM already performs
   continuous batching. A controlled `MAX_NUM_SEQS=1` versus 8 experiment could
   isolate batching, but it must be labeled as batching-disabled versus
   batching-enabled, not "vanilla" versus "optimized."

Until those steps succeed, the only legitimate live conclusion is that one
baseline Qwen3-8B chat request succeeded. There is no validated live throughput,
cost-per-token, batching gain, A/B delta, or savings percentage.
