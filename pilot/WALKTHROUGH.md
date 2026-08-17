# What the operator pilot looks like

The operator remains at the keyboard. A normal session needs a controlled
45–60 minute window, mostly for reviewing scope and changing the server between
baseline and candidate. The exact duration depends on the endpoint.

## 1. Fifteen-minute scope check

We fill in the run card, including the existing SLO, the decision already
pending, approved workload, cache policy, exact traffic envelope, and hard
spend cap. We do not invent an optimization for the demo. If the current tool
cannot exercise the operator's real decision, we do not run.

The preferred first decision is a batching/`max_num_seqs` change tested at a
load high enough to exercise the lower value. GPU-sizing and concurrency-only
comparisons are incompatible with the current decision-grade saved-run
contract, so they are not accepted for this first pilot rather than worked
around.

Decision eligibility requires an immutable image digest and full model commit.
If the operator cannot provide them, the session is labeled diagnostic before
traffic and does not satisfy the decision-grade pilot gate.

## 2. Operator installs locally

```sh
python3 -m venv .venv
. .venv/bin/activate
sha256sum /path/to/throttle_bench-0.2.0-py3-none-any.whl
python -m pip install /path/to/throttle_bench-0.2.0-py3-none-any.whl
throttle --version
```

Record the distributed package SHA-256 and version in the run card before
continuing. This identifies the artifact used; a version string alone does not.
On macOS, use `shasum -a 256` in place of `sha256sum`.

The operator prepares approved measured and disjoint warm-up JSONL files on
their own machine. Neither file is sent to the founder.

## 3. Credential stays local

Use the operator's secret manager if available. Otherwise stop screen sharing
and use a hidden shell prompt; for Bash:

```sh
read -rsp 'API key: ' THROTTLE_API_KEY
printf '\n'
export THROTTLE_API_KEY
```

Set `THROTTLE_ENDPOINT` locally while screen sharing is off. Do not type the key
or endpoint literal into a shared command/history, and do not send the shell
history or terminal transcript. Prefer a short-lived endpoint-scoped key and
revoke it at closeout.

## 4. Zero-traffic plan

The final command is built from the run card. A representative baseline plan
looks like this; bracketed values are filled in before the session:

```sh
throttle plan --run-mode benchmark \
  --model '[MODEL]' \
  --url "$THROTTLE_ENDPOINT" \
  --api-key-env THROTTLE_API_KEY \
  --prompts '[MEASURED.jsonl]' \
  --warmup-prompts '[WARMUP.jsonl]' \
  --concurrency '[LOAD_THAT_EXERCISES_THE_DECISION]' \
  --blocks 3 \
  --requests-per-block 67 \
  --warmup-requests 3 \
  --max-tokens 128 \
  --timeout-seconds 60 \
  --stream \
  --p95-slo-ms '[P95_SLO]' \
  --ttft-slo-ms '[TTFT_SLO]' \
  --seed 17 \
  --cache-policy '[CACHE_POLICY]' \
  --model-revision '[FULL_MODEL_COMMIT]' \
  --image-digest '[IMAGE@SHA256]' \
  --gpu '[GPU_CLASS]' \
  --gpu-fingerprint '[NON_SECRET_STABLE_DEVICE_LABEL]' \
  --cuda-version '[CUDA]' \
  --driver-version '[DRIVER]' \
  --server-version '[VLLM_VERSION]' \
  --engine-flag 'max_num_seqs=[BASELINE_VALUE]' \
  --engine-flag 'enable_chunked_prefill=[COMMON_EFFECTIVE_VALUE]' \
  --engine-flag 'enable_prefix_caching=[COMMON_EFFECTIVE_VALUE]' \
  --engine-flag 'max_model_len=[COMMON_EFFECTIVE_VALUE]' \
  --engine-flag 'gpu_memory_utilization=[COMMON_EFFECTIVE_VALUE]' \
  --engine-flags-provenance runtime_verified \
  --evidence-source live_inference \
  --cost-model dedicated-hourly \
  --total-hourly-price '[RATE]' \
  --max-requests 204 \
  --max-tokens-per-request 128 \
  --max-total-requested-tokens 26112 \
  --max-elapsed-seconds '[DERIVED_LIMIT]' \
  --max-errors 1 \
  --max-concurrency '[LOAD_CEILING]' \
  --max-response-bytes 1048576 \
  --max-estimated-spend '[IDENTICAL_PER_BENCHMARK_CAP]' \
  --variant baseline
```

This cost block is for a genuinely dedicated-hourly endpoint. A different
billing model uses its matching CLI flags and preflight, held identical across
baseline and candidate. The stable GPU label is non-secret and may appear
transiently in local process arguments; Throttle stores only its hash. The
operator separately retains proof that both runs used the same device.
If p95 or TTFT is not part of the approved run card, omit that optional SLO
flag from both variants; never populate a threshold merely to fill the template.
Record every decision-relevant effective serving flag, not only the treatment;
the four common examples above are a minimum when applicable. Add identical
entries for tensor parallelism, dtype/quantization, scheduler behavior, or
other effective controls that could affect the measurement. Verify them from
runtime evidence and change only `max_num_seqs` between variants. Chunked
prefill remains common and receives no optimization credit.

`plan` sends no traffic and does not read the credential. The operator checks
the visible destination, maximum calls/tokens, time, spend, and privacy text,
then gives an explicit go or stops.

If the planned command does not fit the whole-session envelope, reduce it or
cancel. Never raise the cap merely to make the session finish.

## 5. Connectivity smoke

Run a separate exact smoke plan, then the 27-request smoke. It is only a route,
credential, response-shape, and short-load check. Its output is never a
deployment recommendation. Any failure stops the session for diagnosis.

For the dedicated-hourly example, the smoke plan includes these explicit
traffic limits in addition to the same endpoint/model/cost/workload fields:

```sh
throttle plan --run-mode smoke \
  --model '[MODEL]' \
  --url "$THROTTLE_ENDPOINT" \
  --api-key-env THROTTLE_API_KEY \
  --prompts '[MEASURED.jsonl]' \
  --warmup-prompts '[WARMUP.jsonl]' \
  --concurrency 1 4 8 \
  --blocks 1 \
  --requests-per-block 8 \
  --warmup-requests 1 \
  --max-tokens 128 \
  --timeout-seconds 60 \
  --stream \
  --seed 17 \
  --cache-policy '[CACHE_POLICY]' \
  --cost-model dedicated-hourly \
  --total-hourly-price '[RATE]' \
  --max-requests 27 \
  --max-tokens-per-request 128 \
  --max-total-requested-tokens 3456 \
  --max-elapsed-seconds '[SMOKE_TIME_LIMIT]' \
  --max-errors 1 \
  --max-concurrency 8 \
  --max-response-bytes 1048576 \
  --max-estimated-spend '[SMOKE_ALLOCATION]'
```

After approval, change only `plan --run-mode smoke` to `smoke` and add
`--output smoke.json`. Deduct the preallocated smoke amount from the session
budget; do not change the identical baseline/candidate allocations.

## 6. Baseline measurement

After approval, replace `plan --run-mode benchmark` with `benchmark` and add
`--output baseline.json`. The operator confirms the server is actually using
the recorded baseline flag. Throttle must finish complete with 201 valid
measured requests, three complete blocks, zero errors, and no safety stop.

This fixed-load position normally exits **3** because its only tested load is a
search boundary; that means valid but individually inconclusive, not execution
failure. Do not use shell `set -e` or the process code alone as the gate. Continue
only after inspecting `baseline.json` and confirming top-level `status` is
`complete`, the condition is `valid` and `decision_grade`, all 201 measured
responses are valid, and errors/cancellations are zero.

## 7. Candidate measurement

The operator—not Throttle—changes and verifies only the predeclared candidate
setting. All workload, SLO, load, token, cache, safety, and environment controls
remain identical. In the saved baseline plan, change only the declared
`max_num_seqs` value and `--variant candidate`, then run that candidate plan.
After approval, replace `plan --run-mode benchmark` with `benchmark` and add
`--output candidate.json`. Apart from the engine treatment, variant label, and
output filename, every argument remains byte-for-byte identical.

The candidate position has the same expected exit-3 boundary behavior. Apply
the identical JSON checks to `candidate.json`; stop before comparison if any
condition or accounting check fails.

If background load, cache state, or any uncontrolled setting changes, stop and
reschedule rather than calling the pair comparable.

## 8. Offline comparison

```sh
throttle compare baseline.json candidate.json --output comparison.json
```

This step sends no network traffic and needs no key. We inspect status,
compatibility reasons, `decision_eligible`, confidence interval, SLO outcome,
token parity, and boundary state together. Supported, inconclusive, and invalid
are all legitimate outcomes. We do not translate an inconclusive result into a
positive recommendation.

## 9. Behavior question and teardown

Ask once, neutrally:

> What, if anything, will you do because of this result?

Then complete every teardown item in the safety contract. The operator reviews
the sanitized artifact before deciding whether to share it.

After 48–72 hours ask:

> What, if anything, did you actually do after seeing the result? If nothing,
> what stopped you?

The action, non-action, or concrete rejection is the pilot evidence—not praise
for the CLI.
