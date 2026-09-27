# FAQ

*Common questions about Throttle.*

<details><summary>Does Throttle change my server?</summary>

    No. It provisions nothing and never changes your server. `--config` only records what you changed.

</details>

<details><summary>Why does it say NOT CALIBRATED?</summary>

    It needs recent repeat checks of an unchanged config (at least 2 degrees of freedom, within 24 hours) to measure run-to-run noise. Run the same check a few more times without changing anything. See [Noise and calibration](../concepts/noise-and-calibration.md).

</details>

<details><summary>Why is the GPU rate 'ASSUMED'?</summary>

    Throttle can't see your invoice. You supply the hourly rate; tokens and time are measured. A change in the rate alone is never reported as a verdict.

</details>

<details><summary>Which servers work?</summary>

    Any OpenAI-compatible chat completions API that returns token `usage`. vLLM, SGLang, Ollama and LMDeploy were measured end to end.

</details>

<details><summary>Does it measure quality?</summary>

    No. It measures cost. Validate quality separately when a cheaper config uses a smaller model or lower precision.

</details>

<details><summary>How many requests does a check send?</summary>

    By default 5 blocks × 4 measured requests plus one warm-up (in the default cold-cache mode, each distinct prompt is also sent once untagged and unmeasured), all capped by safety limits checked before any traffic (requests, tokens, concurrency, time and estimated spend).

</details>
