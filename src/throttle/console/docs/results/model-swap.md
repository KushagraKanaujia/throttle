# A real change on a laptop

*llama3.2:3b → 1b on local Ollama: −37.2%, outside the noise bound.*

Throttle 0.4.2, local Ollama on a MacBook, GPU rate assumed at $1.50/hr.

1. Four checks of the unchanged `llama3.2:3b` to calibrate noise.
2. Switched the model to `llama3.2:1b` and checked again.

| | $/M output tokens |
| --- | ---: |
| `llama3.2:3b` | $8.71 |
| `llama3.2:1b` | $5.47 (95% CI $5.34 to $5.60) |
| Change | **−37.2%** |
| Noise bound | 24.5% (3 df) |
| Verdict | **CHEAPER** (exit code 0) |

> **Warning:** A smaller model is cheaper to serve. Whether its answers are good enough is a separate question; Throttle measures cost, not quality.
