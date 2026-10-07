# Hot Aisle MI300X runs, 2026-10-01

These are the raw `throttle check` history records and answer samples behind the MI300X numbers in the README, on the Cost Index and in Throttle's posts.

**Setup**
- One rented AMD MI300X VM on Hot Aisle, priced at its $2.99/GPU-hr list price (ASSUMED).
- vLLM on ROCm, version `0.23.1.dev1+g9ddef7117.d20260901.rocm714` (see `mi300x-vllm-version.txt`).
- Qwen2.5-Instruct 7B, 32B and 72B.
- Default `throttle check` workload: cold cache, concurrency 32, max 256 output tokens.
- Each config was checked 3 or 4 times in a row, which is what the noise bound is calibrated from.
- Endpoints in the records are `localhost` on the VM.

**$ per million output tokens (mean of the repeat checks)**

| Model | Change | Before | After | Note |
|---|---|---|---|---|
| 72B | vLLM on-the-fly FP8 (`--quantization fp8`) | $1.67 | $0.88 | Looked 47% cheaper. Answers changed shape (see below), so it is not a saving. |
| 72B | Pre-quantized FP8 checkpoint (RedHat dynamic) | $1.67 | $1.13 | 32% cheaper, answers correct |
| 72B | 1 GPU → 2 GPUs, tensor parallel 2 (`mi300x-72b-tp1-vs-tp2.ndjson`) | $1.67 | $2.14 | 1.56x the throughput, but 28% more per token, because you pay for 2 GPUs |
| 32B | vLLM on-the-fly FP8 | $0.77 | $0.45 | Looked 41% cheaper. Broken output: every request hit the 256-token cap |
| 7B | vLLM on-the-fly FP8 | $0.23 | $0.19 | Output length +6–9%, under the 10% OUTPUT CHANGED line |
| 7B | Pre-quantized FP8 | $0.23 | $0.19 | Answers correct |

**Answer samples (`answers/`)**
- These are 6 fixed prompts at temperature 0.
- `qwen32b-bf16-vs-fp8-onthefly.json` is the broken case. Asked to summarize the causes of World War I in about 80 words, the on-the-fly FP8 server returned 256 tokens of `!!!!` and finish_reason `length`. BF16 answered normally.
- The other files are the correct runs that the "answers correct" notes above refer to.

**Caveats**
- One workload, one session.
- For the 72B FP8 comparison, the FP8 server ran on GPU 1 and the BF16 baseline on GPU 0 of the same VM.
- These were recorded with Throttle 0.5.x. Rows recorded before OUTPUT CHANGED existed were judged by reading the answer samples.
- These results apply to this model, engine, GPU and workload only. They are not a savings projection.
