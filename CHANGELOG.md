# Changelog

All notable changes to Throttle will be documented in this file.

## [0.3.0] - Unreleased

### Removed
- Claude Code cost proxy extracted to separate repository: https://github.com/KushagraKanaujia/claude-cost-proxy
  - Removed `throttle-proxy`, `throttle-setup`, `throttle-summary` commands
  - Removed `aiohttp` dependency
  - Main throttle repo now focuses exclusively on vLLM optimization

### Added
- One-command, operator-mediated `throttle golden` orchestration for the
  B1/C1/B2/C2/B3/C3 counterbalanced protocol, including a zero-traffic dry run
  and sanitized partial-session evidence
- A workload-scoped Golden decision summary that is emitted only when every
  protocol and statistical eligibility gate passes
- Warning-strict Python 3.11-3.14 CI, process-wide offline-network guards, and
  clean-wheel/source-byte package verification
- Deterministic adversarial boundary coverage for Unicode spoofing, credentials,
  paths, digests, duplicate JSON keys, cyclic/deep structures, and payload
  non-reflection
- Isolated, bounded vLLM Prometheus metric collection and suggestion-only
  `max_num_seqs` bottleneck analysis components. The only presentation path is
  the explicit `experimental-tuning` subcommand; standard commands and saved
  run reports remain unchanged, and the analysis is decision-ineligible by
  construction. Its create-only supplementary envelope binds the detached
  safety projection to the sanitized smoke report by canonical SHA-256.
- An isolated safety-validation boundary that pins its own reviewed policy,
  independently replays and binds collector/analyzer evidence, and returns a
  detached, non-actionable projection. The artifact cannot self-authorize
  routing into another CLI/report path, apply configuration, or bypass Golden.
- A pinned, deterministic vLLM exposition compatibility fixture and connected
  loopback orchestration test. This is software evidence only, not a live GPU,
  performance, scheduler-saturation, or savings result.
- Sealed native request profiles: typed sampling controls (`--temperature`,
  `--top-p`, `--request-seed`) and safe scalar `--request-field` extensions for
  model-specific controls such as Qwen thinking, sealed by a canonical SHA-256
  that saved-run comparison and Golden must match exactly

### Changed
- Golden now accepts any two canonical positive, distinct `max_num_seqs`
  values, preserves one declared closed-loop load at or above the larger value,
  and infers the treatment independently from all six saved reports. Historical
  1-versus-8 evidence remains valid. The exercise claim is explicitly limited
  to offered client demand, not direct server-scheduler saturation.
- Multi-load benchmark sweeps now warn before key resolution or traffic that
  their condition-major results are exploratory and cannot be decision-eligible
- Smoke sessions default to a 120-second ceiling, sustained benchmarks retain
  900 seconds, and Golden sessions use an explicit 5,400-second session ceiling
- Golden live preflight is platform-neutral across CUDA, Metal, ROCm, and CPU
  while preserving the stricter CUDA image/runtime requirements
- Saved and in-memory reports now share bounded depth, node, numeric, and string
  validation; Golden run fingerprints cover only validated evidence consumed by
  the decision gate

### Security
- Runtime and engine metadata reject normalized Unicode lookalikes,
  credential/userinfo shapes, URLs, absolute or traversal paths, and unsafe
  control characters without reflecting rejected values
- Report parsing rejects duplicate keys, non-finite or oversized numbers,
  non-JSON containers, cycles, and over-limit trees before comparison or Golden
  aggregation
- Request extensions cannot replace the model, messages, token cap, streaming
  controls, standard sampling fields, tools, credentials, or response shape;
  structured values, secrets, URLs, and headers are rejected before traffic,
  and a tampered profile fails saved-report preflight

## [0.2.1] - 2026-08-18

### Added
- Platform-aware accelerator provenance for CUDA, Metal, ROCm, and CPU runs
- Immutable software-environment pins for decision-grade direct-host benchmarks
- `--accelerator` and `--accelerator-fingerprint` aliases for existing GPU fields

### Changed
- Runtime manifest 1.1 supports non-CUDA comparisons while preserving manifest
  1.0 CUDA report compatibility and CUDA's existing image/driver requirements
- Generated and loaded runtime metadata now share one fail-closed sanitizer;
  manifest 1.1 legacy GPU aliases must reconcile with accelerator fields

## [0.2.0] - 2026-08-17

### Added
- Four explicit modes: plan, smoke, benchmark, and compare
- Decision-grade benchmark validation with strict statistical criteria
- Golden protocol for counterbalanced six-position testing
- GuideLLM 0.7.3 integration as optional cross-check backend
- Comprehensive cost models (unknown, dedicated-hourly, serverless-active-seconds, user-supplied)
- Hard safety limits for requests, tokens, time, errors, and spend
- Immutable runtime provenance tracking (model revision, image digest, engine flags)
- Native streaming protocol with TTFT, TPOT, and inter-chunk latency metrics
- Confidence intervals using Student-t for blocks and bootstrap for requests
- SLO goodput tracking with p95 E2E and TTFT thresholds
- Saved-run comparison with matched repeated blocks
- Privacy-first sanitized reports (no URLs, keys, or raw responses)

### Security
- Loopback-only plain HTTP, HTTPS required for non-loopback
- No proxy variable inheritance in native mode
- Isolated GuideLLM subprocess with cleaned environment
- Response byte size limits and completion validation
- Explicit acknowledgements for unknown cost and GuideLLM gaps

### Documentation
- Complete README with installation and usage examples
- Golden protocol specification
- Known gaps and validation documentation
- Operator pilot walkthrough
- User testing guide

## [0.1.0] - 2026-08-01

### Added
- Initial proof-of-concept release
- Basic smoke testing functionality
- Local validation artifacts
