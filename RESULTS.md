# Throttle Validation Results

This document summarizes measured results from controlled Throttle validation runs. All claims trace to specific JSON artifacts in the `validation/` directory.

## Proven Decision-Eligible Result

**Configuration:** `max_num_seqs` 1 → 8 on vLLM 0.16.0  
**Model:** Qwen/Qwen2.5-0.5B-Instruct (revision `7ae557604adf67be50417f59c2c2f167def9a775`)  
**Hardware:** NVIDIA A100 80GB PCIe, $1.39/hour  
**Protocol:** Six-position counterbalanced Golden protocol (B1/C1/B2/C2/B3/C3)  
**Workload:** Closed-loop concurrency 8, 201 measured requests per position (3 blocks × 67 requests)  

**Measured throughput increase:** **+189.5% to +246.2%** (95% confidence interval)  
Point estimate: +217.8%

This result:
- Passed Throttle's strict six-run counterbalanced protocol ([docs/GOLDEN_PROTOCOL.md](docs/GOLDEN_PROTOCOL.md))
- Is `decision_eligible: true` and `decision_state: supported`
- Measured 1,206 valid requests across 6 positions with zero errors
- Used order-balanced B-C-B / C-B-C phase contrasts
- Is the **only decision-eligible result** in this repository

**Evidence:** [`validation/golden-live-20260817/`](validation/golden-live-20260817/)  
**Full audit:** [`validation/golden-live-20260817/RUN_AUDIT.md`](validation/golden-live-20260817/RUN_AUDIT.md)

### Important limitations

This result applies **only** to:
- The exact pinned model revision tested
- vLLM 0.16.0 on A100 80GB PCIe
- The specific workload tested (closed-loop concurrency 8, 128 max tokens)
- Configuration disabled prefix caching

It is **not**:
- A universal optimization claim
- A cost savings projection
- An optimum claim (other values were not tested)
- A production recommendation

## Cross-Stack Compatibility Evidence

Throttle was tested against four open-source inference engines on the same RunPod RTX 4090 24GB pod (per the run's `REPORT.md`) to validate protocol compatibility. A fifth engine (TGI) failed to start due to a container issue unrelated to Throttle.

**Configuration:** closed-loop concurrency 1 and 4, $0.74/hour. Models differed per engine: vLLM `Qwen/Qwen3-8B`; SGLang and LMDeploy `Qwen/Qwen2.5-3B-Instruct`; Ollama `qwen2.5:3b`  
**Purpose:** Demonstrate OpenAI-compatible protocol works across backends  
**Status:** Descriptive only — not decision-grade

| Engine | Version | Status | Best measured throughput | Decision-eligible |
| --- | --- | --- | --- | --- |
| **vLLM** (Qwen3-8B) | 0.27.1 | ✓ Success | 143.92 tok/s | No (exploratory sweep) |
| **SGLang** (Qwen2.5-3B-Instruct) | unknown | ✓ Success | 230.49 tok/s | No (exploratory sweep) |
| **Ollama** (qwen2.5:3b) | 0.32.14 | ✓ Success | 172.52 tok/s | No (exploratory sweep) |
| **LMDeploy** (Qwen2.5-3B-Instruct) | unknown | ✓ Success | 238.32 tok/s | No (exploratory sweep) |
| **TGI** | N/A | ✗ Container failed | N/A | N/A |

**Evidence:** [`validation/runpod-five-stack-20260819/`](validation/runpod-five-stack-20260819/)

### Why these results are not decision-grade

Multi-condition sweeps run conditions in sequence (condition-major order) and cannot counterbalance time drift. Throttle correctly marks them as `decision_eligible: false`. The throughput numbers reflect each engine's default container configuration and are not controlled comparisons. They are also not comparable across engines because the models differ (vLLM ran an 8B model; the others ran 3B models).

To make a decision-grade claim about any of these engines:
1. Pin all runtime variables (model revision, image digest, GPU fingerprint, engine flags)
2. Use `throttle golden` with two configurations (baseline and candidate)
3. Follow the six-position counterbalanced protocol

## What's Next

The next milestone is **independent replication** by an external operator. The Golden protocol is fully documented in [docs/GOLDEN_PROTOCOL.md](docs/GOLDEN_PROTOCOL.md) and designed to be reproduced on any compatible vLLM deployment.

For production claims, an operator would:
1. Run the Golden protocol on their staging environment
2. Validate the candidate configuration with the separate six-position protocol
3. Publish sanitized evidence (model, hardware, workload, outcome, limitations)
4. Show pre/post capacity bills if claiming realized cost savings

---

**All validation artifacts** are indexed in [`validation/README.md`](validation/README.md).
