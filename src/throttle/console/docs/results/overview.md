# Results

*Real runs with the raw numbers. Nothing here is a projection.*

![](../../illustrations/results.svg)

Every number below comes from a saved artifact in the Throttle repo or a recorded run.

- **[One vLLM flag on an A100](a100-max-num-seqs.md)**: `max_num_seqs` 1 → 8: $0.746 → $0.234 per million output tokens, six-position counterbalanced.
- **[Nothing changed: +27%, then −11%](nothing-changed.md)**: Four checks of an untouched server, and why the verdict is NO WINNER.
- **[A real change on a laptop](model-swap.md)**: `llama3.2:3b` → `1b`: −37.2%, outside a 24.5% noise bound.
- **[Community results](community.md)**: Shared by people running Throttle on their own GPUs.

## Receipts

| What | Result | Source |
| --- | --- | --- |
| Config decision on a real GPU | `max_num_seqs` 1 → 8 on vLLM 0.16.0: **+189.5% to +246.2% throughput** (95% CI, point estimate +217.8%) | A100 80GB, Qwen2.5-0.5B-Instruct, six-position golden protocol, `decision_eligible: true` |
| Cross-engine compatibility | vLLM, SGLang, Ollama and LMDeploy all measured end to end | RunPod GPUs, descriptive only (not decision-grade) |
| Agent session profile | Simulated 4-turn agent: 41.9% of wall-clock waiting on the client, an estimated 54.3% of prompt tokens redundant prefill | Local Ollama `llama3.2:3b` on a MacBook; tool pauses scripted at 1.5 s |
| Proxy cache hit | Cold call 1.75 s, same prompt from cache 1.2 ms, reworded prompt from cache 9.5 ms | Local Ollama on a MacBook, opt-in semantic tier; 4 requests, 2 backend calls |

The golden A100 result is the only decision-eligible result. It applies to that exact model, engine, GPU and workload, and it isn't a savings projection. The laptop numbers show how the mechanisms work; they don't predict results on your traffic.

Raw artifacts: [github.com/KushagraKanaujia/throttle/tree/main/validation](https://github.com/KushagraKanaujia/throttle/tree/main/validation)
