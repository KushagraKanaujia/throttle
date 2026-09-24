# Changelog

All notable changes to Throttle will be documented in this file.

## [Unreleased]

### Added
- `throttle check`: measures $/M tokens against a live endpoint in repeated
  blocks (95% Student-t CI across blocks), fingerprints the serving config
  (`/v1/models`, vLLM `/metrics` `*_info` labels, `--label`, `--config
  KEY=VALUE`, the ASSUMED GPU rate), saves each check to a local history
  (`~/.throttle/checks/`, `--history-dir`, `$THROTTLE_CHECK_HISTORY_DIR`) and
  compares it with the previous check of the same endpoint (or `--against
  ID`). Also `--history`, `--json`, `--monthly-tokens` (labelled PROJECTED),
  and pre-traffic caps on requests, tokens, concurrency, time and worst-case
  GPU spend.
- Run-to-run noise bound for `check` verdicts. A within-run CI cannot see
  drift between runs, so CHEAPER / MORE EXPENSIVE now needs a measured change
  larger than `t(0.975, df) x sqrt(2) x SD`, where SD is the pooled relative
  SD of $/M among earlier repeat checks (last 24 h, not counting the check
  being judged) of the baseline's config and of this check's config (same
  endpoint, fingerprint, workload and Throttle version), **and**
  non-overlapping CIs. Otherwise NO WINNER, naming every failed condition.
  With fewer than 2 df (e.g. under 3 earlier repeats of one config), or a
  baseline more than 24 h old, the verdict is NOT CALIBRATED and no monthly
  figure is projected. A hint is printed when two same-config checks have
  non-overlapping CIs.
- `check` judges only the MEASURED part of a change: when the ASSUMED GPU
  rate differs from the baseline's, the new check is put on the baseline's
  rate and the rate-driven part is printed separately. A Throttle upgrade is
  listed as a change; history lines without an `id` are skipped and counted.
- `check` exit codes: 4 for a calibrated MORE EXPENSIVE whose measured
  change is at or above `--fail-if-costlier PCT`; 5 for NOT CALIBRATED when `--fail-if-costlier` is
  set (0 without it).
- Running `throttle` with no arguments prints a three-step getting-started
  list (demo, cost, check).
- `cost` and `demo` print a blended $/M line (total cost / all tokens) next to
  the input and output figures, with a note that those two are not additive.

### Changed
- `--url` and `--endpoint-url` are accepted interchangeably, and
  `http://host:port`, `.../v1` or the full chat-completions route all work in
  `cost`, `measure`, `check`, `plan`, `smoke` and `benchmark`; `proxy` accepts
  `--url` for `--backend-url`.
- `--gpu-hourly-rate`, `--gpu-rate-per-hour` and `--total-hourly-price` are
  aliases. On `plan`/`smoke`/`benchmark`, an hourly price implies
  `--cost-model dedicated-hourly` (printed as inferred); `--gpu-hourly-rate`
  with `--gpus` > 1 is refused as ambiguous.
- `cost`, `measure` and `check` default `--model` to the server's only model
  from `/v1/models` and name it; with several models the error lists them.
- A localhost endpoint with no `OPENAI_API_KEY` set sends no Authorization
  header (and says so) instead of refusing; remote endpoints still need a key.
- Clearer errors: a non-200 from the chat route says whether the URL or the
  model is wrong (listing served models), usage errors name the subcommand
  that was run, and the unknown-cost refusal suggests `--gpu-hourly-rate`.
- The `plan` output calls the spend limit a ceiling, not an estimate.
- README and QUICKSTART now lead with $/M tokens and re-checking after each
  config change; caching is documented as one opt-in lever (off by default,
  semantic tier opt-in with its false-match risk). `throttle check` is
  documented with recorded local-Ollama output for each verdict (NO WINNER,
  MORE EXPENSIVE, CHEAPER, NOT CALIBRATED) and its exit codes. Install
  instructions point at source, since PyPI still has 0.3.0; 0.4.0 is the next
  PyPI release.

### Fixed
- README commands that did not run as written: the `diagnose` example had no
  price and exited with a usage error, and the similarity-cache example used
  `https://...` placeholders (now a local `smoke` run). The agent-profiler
  steps now say that the proxy writes sessions to disk every 5 to 10 seconds,
  so `throttle sessions` run immediately after a request can show nothing.
- The README credited the semantic-cache false-match example ("Is it safe /
  dangerous to use eval in Python?") with a 0.9804 score that belongs to a
  different pair; the recorded score is 0.9874 (`data/negation_pairs.json`).

## [0.4.0] - 2026-09-23

### Added
- Agent session profiling (opt-in). `throttle proxy --enable-session-tracking`
  records per-turn timing (TTFT, total latency, gap since the previous turn)
  and token counts for multi-turn agent sessions to `~/.throttle/sessions.db`.
  Prompt and completion text are never stored: turns keep content hashes and
  token counts, and sessions keep the client IP used for session grouping.
- `throttle sessions` lists recorded sessions (`--since`, `--limit`) and, given
  a session id, shows a per-session breakdown of where wall-clock time went
  plus rule-based findings with suggested configuration changes.

### Changed
- Internal design notes, audit reports, and pilot/outreach material moved from
  the repository root into `docs/internal/`. The proxy guide now lives at
  `docs/PROXY_DEMO.md`.

## [0.3.0] - 2026-08-22

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
- Opt-in similarity cache for bypassing inference on semantically similar
  prompts. Cache hits are excluded from GPU latency percentiles to preserve
  decision-grade measurements. Telemetry (`cache_enabled`, `cache_hits`,
  `cache_misses`, `cache_hit_rate`) flows through experimental tuning
  validation and saved-run comparison. Thread-safe implementation uses Jaccard
  similarity on tokenized prompts.

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
