# RunPod five-stack validation evidence — 2026-08-19

Created 2026-08-19 (America/Los_Angeles) for live validation of Throttle against
vLLM, TGI, SGLang, Ollama, and LMDeploy. Files in this directory preserve raw
command transcripts and generated Throttle JSON artifacts.

Pod identifier: `o4zw3lkzv9yr69`

## Summary

Throttle was tested against five different open-source inference engines on the same RunPod RTX 4090 24GB pod (per `REPORT.md`) to validate cross-stack compatibility. Four of the five engines completed successfully; one (TGI) failed to start due to a container issue unrelated to Throttle.

## Results

All four successful runs used:
- Model (per engine, from each JSON manifest): vLLM `Qwen/Qwen3-8B`; SGLang and LMDeploy `Qwen/Qwen2.5-3B-Instruct`; Ollama `qwen2.5:3b`
- GPU: NVIDIA GeForce RTX 4090 24GB (RunPod, per `REPORT.md`; the manifests record `runtime.gpu` as `unknown`)
- Workload: closed-loop concurrency 1 and 4
- Cost model: $0.74/hour dedicated
- Throttle version: 0.2.0

| Engine | Version | Status | Best throughput (tok/s) | Decision-eligible |
| --- | ---: | --- | ---: | --- |
| vLLM | 0.27.1 | Success | 143.92 | No (exploratory sweep) |
| SGLang | unknown | Success | 230.49 | No (exploratory sweep) |
| Ollama | 0.32.14 | Success | 172.52 | No (exploratory sweep) |
| LMDeploy | unknown | Success | 238.32 | No (exploratory sweep) |
| TGI | N/A | Container failed to start | N/A | N/A |

Throughput is not comparable across engines: the models differ (vLLM ran an 8B model; the others ran 3B models).

## Purpose

These runs demonstrate that Throttle's native OpenAI-compatible protocol works across multiple inference backends without modification. The throughput numbers are **descriptive only** — they reflect the specific default configurations of each container and are not decision-grade evidence (multi-condition sweeps cannot reach `decision_eligible: true` due to non-counterbalanced ordering).

## Evidence

- `vllm-throttle-c1-c4.json`: vLLM 0.27.1 benchmark
- `sglang-throttle-c1-c4.json`: SGLang benchmark
- `ollama-throttle-c1-c4.json`: Ollama 0.32.14 benchmark
- `lmdeploy-throttle-c1-c4.json`: LMDeploy benchmark
- `02-tgi-*.log`: TGI container startup failure logs

All JSON artifacts are mode 0600 and passed report sanitization.

