# Internal notes (archive)

This folder holds working notes from Throttle's development: design drafts,
audit passes, progress logs, RunPod runbooks, and operator-pilot material.
They are kept for history and transparency. They are **not** user
documentation, and some of them describe states of the code that have since
changed.

For current, user-facing material start with:

- [README.md](../../README.md): install and usage
- [RESULTS.md](../../RESULTS.md): measured results and the evidence files behind them
- [docs/KNOWN_GAPS.md](../KNOWN_GAPS.md): known limitations
- [validation/README.md](../../validation/README.md): index of validation artifacts

| Path | What it is |
| --- | --- |
| `AGENT_SESSION_PROFILER_DESIGN.md` | Design notes for `throttle proxy --enable-session-tracking` / `throttle sessions` |
| `adversarial_audit_report.md`, `COMMAND_AUDIT.md`, `CREDIBILITY_PASS_COMPLETE.md` | Audit passes over claims and commands |
| `DECISIONS.md`, `FINDINGS.md`, `PROGRESS.md`, `PROJECT_STATE.md`, `FINAL_REPORT.md` | Development logs |
| `RUNPOD.md`, `RUNPOD_DEPLOY.md` | RunPod runbooks used for GPU validation runs |
| `validation_20260824_235357.json` | Raw local simulator output (Ollama, `llama3.2:1b`), not cited by RESULTS.md |
| `notebooks/` | Kaggle dual-T4 validation notebook |
| `pilot/` | Operator-pilot packet: outreach, safety contract, walkthrough, evidence ledger |
