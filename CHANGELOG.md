# Changelog

All notable changes to Throttle will be documented in this file.

## [Unreleased]

### Changed
- Throttle Pilot terms: the pilot is now 3 months, free, for a small group of
  teams running agents on their own GPUs. We set Throttle up on your stack,
  find the config changes that lower your cost per token, and prove each one.
  After the pilot, pricing is based on the savings we verify together
  (conservative end of the 95% interval). Updated in `throttle upgrade`, the
  post-verdict nudge (same once-a-day rules), the first-run screen, the
  console Pilot page and `/api/upgrade`, and the README. The earlier 2-week
  terms are retired from public copy, and the retired-strings test now
  rejects them.

## [0.6.0] - 2026-10-06

### Added
- `throttle savings`: a conservative, auditable verified-savings statement for a
  baseline -> candidate change. Refuses unless re-judging the two recorded
  checks gives a calibrated CHEAPER on the same model and workload. Uses the
  conservative bound (baseline CI low - candidate CI high) x production tokens,
  with the point estimate labelled. Tokens are `--tokens` (REPORTED BY OPERATOR)
  or a vLLM generation-token counter delta since `throttle savings snapshot`
  (MEASURED; a counter reset is refused). `--json` emits a `savings_statement`
  record (version 1).
- `throttle check --workload agent`: measures multi-turn agent traffic. Each
  session is a seeded synthetic conversation: a shared system prompt and tool
  schema, then sequential turns that resend the conversation plus a scripted
  assistant turn and a synthetic tool result. The live reply is measured but
  never fed back, so every run sends the same prompts. Sessions run
  concurrently. Options: `--turns` (6), `--sessions-concurrency`
  (`--concurrency`), `--system-prompt-tokens` (~1500), `--tool-output-tokens`
  (~300), `--assistant-turn-tokens` (~150). A per-run tag in the system prompt
  stops one run's prefix cache helping the next. The record adds
  `workload.workload_shape` with the measured prompt and completion tokens per
  turn, and the cached prompt tokens per turn when the server reports them. The shape is part of the workload identity, so
  agent and default checks are never compared. Default checks and their
  records are unchanged.

### Changed
- Human CLI output is restyled; machine-readable output is not. `throttle check`
  opens with a one-line header (Throttle, version, endpoint, model), prints
  compact block progress lines and ends with a **verdict panel**: $/M with its
  95% CI, the coloured verdict and its reason, the delta vs the baseline in $/M
  and %, the projected monthly $ when computed, the saved check id and one
  next step (e.g. `throttle savings --baseline X --candidate Y --tokens N`, or
  how many unchanged checks calibrate noise). Every earlier line (MEASURED /
  ASSUMED / PROJECTED labels, CI method, noise bound, warnings) is still
  printed above the panel. `throttle savings` puts the statement in a panel
  with the conservative $ figure as the headline and the point estimate dim.
  `--json`, `--share`, record formats, exit codes and verdicts are unchanged,
  and a golden test pins them byte for byte.
- New `throttle.style` module (plain ANSI, no new dependencies): colour only on
  a TTY with `NO_COLOR` unset and `TERM` not `dumb`; `FORCE_COLOR` forces it;
  box-drawing panels with an ASCII fallback when the output encoding is not
  UTF-8; width follows the terminal, 60 to 80 columns.
- `throttle` with no arguments shows a short first-run screen: what Throttle is,
  three steps (start Ollama or point at vLLM / SGLang, `throttle check`,
  `throttle check --workload agent`) and the pilot line. The full 3-command
  guide stays in `throttle --help`.
- The Throttle Pilot replaces the Pro plan and the Cost Audit everywhere a user
  sees an offer: `throttle upgrade`, the post-verdict nudge text, the console
  Upgrade page (now **Pilot**) and its `/api/upgrade` copy, the bundled docs
  (`more/pilot` replaces `more/pro` and `more/cost-audit`) and the README. We
  find savings on your agent workload and prove them: free for 2 weeks, then 20%
  of verified monthly savings (conservative end of the 95% interval), $500/month
  floor, cancel anytime, at most 5 design partners per round. Contact:
  https://throttle-pro.com, kushthrottle@gmail.com (subject "Throttle pilot").
  The QR code now encodes https://throttle-pro.com. `throttle upgrade --audit`
  is a hidden alias that prints the same page. The nudge rules are unchanged
  (once per 24 h; off with `THROTTLE_NO_NUDGE`, in CI, and with `--json`,
  `--share` and `--fail-if-costlier`).

## [0.5.1] - 2026-09-29

### Added
- `throttle upgrade` (and `--audit`): what Throttle Pro and the $500 Cost Audit
  include, the link, and a QR code drawn in the terminal. Generated offline with
  the vendored qrcodegen library (MIT, `throttle/_vendor/`).
- Console **Upgrade** page with the same plans and locally generated QR codes.
- After a calibrated CHEAPER or MORE EXPENSIVE verdict, `throttle check` prints
  one line suggesting `throttle upgrade`, at most once per 24 hours. Off with
  `THROTTLE_NO_NUDGE=1`, in CI, and with `--json`, `--share` or
  `--fail-if-costlier`. No telemetry.

## [0.5.0] - 2026-09-27

### Added
- `notebooks/throttle-quickstart.ipynb`: measure $/M tokens on a free Colab or
  Kaggle T4 (Ollama or vLLM), change one setting and get a verdict.
- README "Try it in 2 minutes" section (Colab, local Ollama, `throttle demo`).
- `throttle check` prints the `--share-id` command for the check it just saved.
- `throttle ui`: the Throttle Console, a local web app over your check history
  (quickstart progress, checks with 95% CI bars and verdicts, compare with the
  noise band, endpoint trends, CI setup, share commands, bundled docs and
  search). Binds 127.0.0.1, refuses non-loopback Host headers, strict CSP, no
  network requests. Verdicts reuse `throttle check`'s comparison code.
- `throttle.check.judge_recorded()`: re-judges a recorded check the way it was
  judged when it ran (shared by `--share-id` and the console).

### Changed
- GitHub Action renamed to "Throttle LLM cost check" with a Marketplace-length
  description. Inputs, outputs and behavior are unchanged.

## [0.4.2] - 2026-09-25

0.4.1 was never published to PyPI; 0.4.2 is the first release that ships everything listed under 0.4.1 below, plus these fixes.

### Fixed
- A request rejected because its prompt plus `max_tokens` exceeds the
  model's context window (vLLM's HTTP 400 when `--max-model-len` is too
  small) now stops `validate-sim`, `measure`, `cost` and `check` with an
  error that names the limit, the request and its size, quotes the server,
  and says how to fit it, instead of "Request 86 failed with status 400"
  (#22). vLLM, SGLang, TGI, llama.cpp and OpenAI wordings are recognised.
- When one request fails, `validate-sim`, `measure` and `check` stop sending
  and let requests already in flight finish. They used to leave those
  requests running, and `asyncio.run` cancelled them mid-connect at shutdown,
  which printed an anyio traceback ("ValueError: second argument (exceptions)
  must be a non-empty sequence") or leaked sockets (#22).

## [0.4.1] - 2026-09-24

### Changed
- `throttle check` is cold-cache by default. It used to resend identical
  prompts every block and every run, so a server with prefix caching (vLLM V1
  `--enable-prefix-caching`, on by default; SGLang RadixAttention; Ollama KV
  reuse) served warm prefixes and $/M looked cheaper than real traffic. Now
  the first user message of every measured (and warm-up) request starts with
  a unique tag, `[run <id> req <n>] `, from a random per-check run id stored
  in the record with the tag format and a sha256 of the exact prompts sent.
  `--warm-cache` resends identical prompts on purpose. The header prints the
  mode. The run id is 6 decimal digits, so the tag costs the same number of
  prompt tokens in every run (9 with `llama3.2:3b`), and those tokens are not
  counted: a cold check sends each distinct base prompt once untagged
  (unmeasured, 1 output token) and input and total $/M use those token
  counts. Blocks record the tag tokens left out (`tag_input_tokens`).
- The prompt cache mode is part of a check's workload identity: a cold and a
  warm check are NO WINNER ("the prompt cache mode changed") and never share
  a calibration group. Checks recorded by 0.4.0 count as warm, so the first
  0.4.1 check against them is NO WINNER; record three cold checks of the
  unchanged config to recalibrate. `prompts_sha256` is still the canonical
  hash of the prompt file, the same hash `smoke`, `benchmark` and `golden`
  record.
- `cost` and `measure` tag their prompts the same way by default and accept
  `--warm-cache`. `measure` records `prompt_cache_mode` and the run id in its
  JSON, and `throttle compare` refuses to rank measure files with different
  modes (files without the field count as warm).
- `smoke`, `benchmark` and `golden` traffic is unchanged: they resend a fixed
  prompt set whose sha256 their reports and golden decisions are built on.
  `--cache-policy` help now says so and explains which value to declare.
- A comparison of different workloads now prints the baseline's age, and
  "What changed in the config" lists the workload fields that differ (e.g.
  `workload.prompt_cache_mode: cold -> warm`), as the `--share` summary does.
- With `--fail-if-costlier`, a check whose baseline ran a different workload
  (NO WINNER, "the workload differs") now exits 5 with a warning instead of 0:
  the gate judged nothing. **Upgrade note:** the first 0.4.1 check against a
  0.4.0 history is such a check (0.4.0 checks count as warm-cache, 0.4.1 is
  cold by default), so a CI gate reports 5 until three cold checks of the
  unchanged config rebuild the baseline. The GitHub Action applies its
  `not-calibrated` policy (warn by default) to it.
- `throttle compare` checks the cache mode of every measure file, not of
  every label, so two files that share a label are still refused when one is
  cold and the other warm.

### Added
- `throttle check --share`: prints a sanitized markdown summary (engine,
  model, GPU, ASSUMED GPU rate, workload and cache mode, $/M before and after
  with CIs, verdict, noise-floor status, Throttle version, what changed) with
  no URLs, hostnames, IPs, local paths or keys, and config values that look
  like secrets masked (also removed: `user:password@host` in any form,
  IPv4-mapped IPv6 addresses, dotted hostnames with any suffix, `host:port`
  with a 2 to 5 digit port, and `token=...`-style pairs or payment-style
  tokens inside labels and values). It also prints a link to a pre-filled GitHub issue
  (kept under 7,000 characters; a longer summary is cut with a note). Nothing
  is uploaded. `--history --share` shares the latest recorded check and
  `--share-id ID` a chosen one, without sending traffic. Checks now record
  the id of the baseline they were compared with, so a shared check is judged
  against the same baseline.
- `.github/ISSUE_TEMPLATE/share-results.yml`: a "Share your results" issue
  form (engine, GPU, model, verdict, summary, optional case-study consent),
  labelled `results`. The label must exist in the repository (GitHub drops
  unknown labels silently), and the form only works from the default branch.

## [0.4.0] - 2026-09-23

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
- Agent session profiling (opt-in). `throttle proxy --enable-session-tracking`
  records per-turn timing (TTFT, total latency, gap since the previous turn)
  and token counts for multi-turn agent sessions to `~/.throttle/sessions.db`.
  Prompt and completion text are never stored: turns keep content hashes and
  token counts, and sessions keep the client IP used for session grouping.
- `throttle sessions` lists recorded sessions (`--since`, `--limit`) and, given
  a session id, shows a per-session breakdown of where wall-clock time went
  plus rule-based findings with suggested configuration changes.

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
- Internal design notes, audit reports, and pilot/outreach material moved from
  the repository root into `docs/internal/`. The proxy guide now lives at
  `docs/PROXY_DEMO.md`.

### Fixed
- README commands that did not run as written: the `diagnose` example had no
  price and exited with a usage error, and the similarity-cache example used
  `https://...` placeholders (now a local `smoke` run). The agent-profiler
  steps now say that the proxy writes sessions to disk every 5 to 10 seconds,
  so `throttle sessions` run immediately after a request can show nothing.
- The README credited the semantic-cache false-match example ("Is it safe /
  dangerous to use eval in Python?") with a 0.9804 score that belongs to a
  different pair; the recorded score is 0.9874 (`data/negation_pairs.json`).

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
