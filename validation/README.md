# Validation Evidence Index

This directory contains controlled validation artifacts for Throttle, organized by type and date.

## Decision-eligible evidence

- **`golden-live-20260817/`**: Six-position counterbalanced Golden protocol run on Qwen2.5-0.5B-Instruct / vLLM 0.16.0 / A100 80GB. Measured +189.5% to +246.2% throughput increase (95% CI) for max_num_seqs 1→8 at concurrency 8. This is the only decision-eligible result in the repository. See `golden-live-20260817/RUN_AUDIT.md` for full details.

## Cross-stack compatibility evidence

- **`runpod-five-stack-20260819/`**: Throttle tested against vLLM, SGLang, Ollama, and LMDeploy on the same A100 80GB GPU. Four of five engines completed successfully (TGI container failed to start, unrelated to Throttle). Demonstrates protocol compatibility but not decision-grade performance claims.

## Experimental/supplementary evidence


## Historical schema-1 artifacts (0.1.x)

The JSON files in the root of this directory are sanitized local/fake-server artifacts from Throttle 0.1. They are preserved as historical validation evidence and may use superseded `recommendation` field names. They are smoke-only, are not current production guidance, and are rejected by Throttle 0.2+ saved-run comparison.

`fake_openai_server.py` is a local fixture, not a live-model benchmark. It is maintained against the current strict schema-2 response/stream contract; the historical JSON files were not regenerated or rewritten.
