# Golden live run audit — 2026-08-17 UTC

## Verdict

The six-position artifact is complete and passes Throttle's own golden gate:

- `golden_protocol_eligible: true`
- `decision_eligible: true`
- `decision_state: supported`
- `overall_outcome: candidate_higher_throughput`
- order-balanced throughput delta: 217.85%
- 95% interval: 189.47% to 246.22%
- eligibility reasons: none

This is evidence for this pinned model, engine, GPU, and workload only. It is
not a universal optimization, projected savings, or an optimum claim.

## Runtime controls

- GPU: NVIDIA A100 80GB PCIe, one physical GPU for all six positions
- image: `runpod/pytorch@sha256:60baa36d3fb6b98fd4f4ece6b96776c83c01a8b7c540e54460ab4d496816141f`
- model: `Qwen/Qwen2.5-0.5B-Instruct`
- model revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- vLLM: 0.16.0
- Torch/CUDA: 2.9.1+cu128 / CUDA 12.8
- NVIDIA driver: 550.127.05
- cache policy: disabled; runtime logs showed `enable_prefix_caching=False`
- chunked prefill: enabled identically in both variants and not credited
- load: closed-loop concurrency 8
- position shape: three blocks of 67 measured requests, plus three separate
  warm-up requests
- request controls: streaming, temperature 0, `max_tokens=128`, no stop tokens
- treatment: baseline `max_num_seqs=1`; candidate `max_num_seqs=8`

Before every position, the operator verified the same private GPU fingerprint,
the expected live process argument, an authenticated external `/v1/models` 200,
and a correctly shaped chat completion with positive usage. Raw GPU identity,
credentials, prompts, responses, endpoint URLs, and server logs are not stored
in the sanitized artifacts.

## Six positions

| Position | Variant | max_num_seqs | Valid measured | Invalid | Blocks | Completion tokens | Block-mean output tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | baseline | 1 | 201/201 | 0 | 3 | 25,728 | 516.55 |
| C1 | candidate | 8 | 201/201 | 0 | 3 | 25,728 | 1,632.41 |
| B2 | baseline | 1 | 201/201 | 0 | 3 | 25,728 | 517.89 |
| C2 | candidate | 8 | 201/201 | 0 | 3 | 25,728 | 1,688.40 |
| B3 | baseline | 1 | 201/201 | 0 | 3 | 25,728 | 518.55 |
| C3 | candidate | 8 | 201/201 | 0 | 3 | 25,728 | 1,631.16 |

Totals: 1,206 valid measured requests, 18 measured blocks, 18 valid warm-up
requests, 154,368 completion tokens, zero errors, zero malformed responses,
zero cancellations, and zero token-length difference between variants.

## Capacity attempt and cost

The requested RTX 4090 was tried first at $0.74/hour. It remained
`initializing/awaiting_container` for the declared 12-minute cutoff, never
published SSH, and served no request. It was deleted before the permitted A100
PCIe fallback was created.

The successful A100 PCIe pod cost $1.39/hour and was deleted immediately after
validation. The six Throttle measurement windows account for $0.08023 of
client-wall-time GPU cost. Including image startup, package/model setup,
restarts, validation, and the discarded 4090 allocation, the conservative
elapsed-rate session estimate is $0.74 total (approximately $0.155 for the 4090
upper-bound window plus $0.581 for the A100 window).

RunPod billing history still returned no finalized rows for either freshly
deleted pod at the final query. Therefore $0.74 is explicitly an elapsed-rate
upper estimate, not a posted invoice amount. It is below the $3 session cap.

## Cleanup proof

- validation-created pods remaining: 0
- account pod state after cleanup: one pre-existing pod, stopped
- account spend rate restored to the pre-run $0.007/hour baseline
- validation SSH key removed
- ephemeral vLLM bearer key, SSH private key, raw GPU fingerprint, and local
  host-key record removed

All seven JSON artifacts are strict JSON, mode 0600, and passed the report
sanitization scan.
