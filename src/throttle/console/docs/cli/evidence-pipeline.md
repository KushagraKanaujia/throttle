# plan, smoke, benchmark, golden

*Capped, evidence-labelled runs for decision-grade comparisons.*

For a decision you want to defend (for example, a config change before a big rollout), Throttle has a staged pipeline. Each step is capped before any traffic goes out.

### 1. plan: zero traffic

Shows the destination, request count, token ceiling and cost of a run without sending anything. It doesn't read your API key, resolve DNS or open a connection.

### 2. smoke: a short sanity run

By default 24 measured calls plus three warm-ups. The report is marked `decision_eligible: false` with a short-sample warning: it confirms the setup, not a decision.

### 3. benchmark: sustained blocks

Repeated measurement blocks with 95% intervals at each load (`--concurrency 1 2 4 8 ...`). The sweep is exploratory because its order isn't counterbalanced; use it to pick what to test with `golden`.

### 4. golden: the decision protocol

Six positions in counterbalanced order (B1 C1 B2 C2 B3 C3) so warm-up and drift can't favour either side. This is how the [A100 result](../results/a100-max-num-seqs.md) was produced.

`diagnose` classifies the likely bottleneck before you start (not decision-grade), and `experimental-tuning` is an opt-in, suggestion-only analysis of server metrics.

These commands have many options (cost models, safety limits, provenance fields). The exact 0.4.2 help for each is below.

<details><summary>Full `throttle plan --help` output (0.4.2)</summary>

```text
usage: throttle plan [-h] [--run-mode {smoke,benchmark}] --model MODEL
                     --url URL [--api-key-env NAME] [--allow-insecure-http]
                     [--backend {native,guidellm}]
                     [--guidellm-prompt-tokens GUIDELLM_PROMPT_TOKENS]
                     [--guidellm-executable GUIDELLM_EXECUTABLE]
                     [--allow-guidellm-validation-gaps]
                     [--cost-model {unknown,dedicated-hourly,serverless-active-seconds,user-supplied}]
                     [--gpus GPUS] [--total-hourly-price USD |
                     --per-gpu-hourly-price PER_GPU_HOURLY_PRICE]
                     [--active-second-price ACTIVE_SECOND_PRICE]
                     [--max-active-workers MAX_ACTIVE_WORKERS]
                     [--billed-active-seconds BILLED_ACTIVE_SECONDS]
                     [--user-supplied-total USER_SUPPLIED_TOTAL]
                     [--allow-unknown-cost] [--prompts PROMPTS]
                     [--warmup-prompts WARMUP_PROMPTS]
                     [--concurrency N [N ...] | --request-rate RPS [RPS ...]]
                     [--open-loop-max-in-flight OPEN_LOOP_MAX_IN_FLIGHT]
                     [--max-tokens MAX_TOKENS] [--blocks BLOCKS]
                     [--requests-per-block REQUESTS_PER_BLOCK |
                     --block-seconds BLOCK_SECONDS]
                     [--warmup-requests WARMUP_REQUESTS]
                     [--timeout-seconds TIMEOUT_SECONDS]
                     [--stream | --no-stream] [--p95-slo-ms P95_SLO_MS]
                     [--ttft-slo-ms TTFT_SLO_MS] [--seed SEED]
                     [--max-requests MAX_REQUESTS]
                     [--max-tokens-per-request MAX_TOKENS_PER_REQUEST]
                     [--max-total-requested-tokens MAX_TOTAL_REQUESTED_TOKENS]
                     [--max-elapsed-seconds MAX_ELAPSED_SECONDS]
                     [--max-errors MAX_ERRORS]
                     [--max-concurrency MAX_CONCURRENCY]
                     [--max-response-bytes MAX_RESPONSE_BYTES]
                     [--max-estimated-spend MAX_ESTIMATED_SPEND]
                     [--cache-policy {unknown,disabled,cold,warm,representative}]
                     [--model-revision MODEL_REVISION]
                     [--image-digest IMAGE_DIGEST] [--gpu GPU]
                     [--gpu-fingerprint GPU_FINGERPRINT]
                     [--cuda-version CUDA_VERSION]
                     [--driver-version DRIVER_VERSION]
                     [--accelerator-backend {cuda,metal,rocm,cpu}]
                     [--accelerator-runtime-version ACCELERATOR_RUNTIME_VERSION]
                     [--host-os-version HOST_OS_VERSION]
                     [--software-environment-digest SOFTWARE_ENVIRONMENT_DIGEST]
                     [--server-name SERVER_NAME]
                     [--server-version SERVER_VERSION]
                     [--engine-flag NAME=VALUE]
                     [--engine-flags-provenance {operator_attested,runtime_verified}]
                     [--variant {baseline,candidate,unspecified}]
                     [--sequence-position SEQUENCE_POSITION]
                     [--evidence-source {unverified_endpoint,live_inference,synthetic_validation}]
                     [--enable-cache] [--cache-ttl-seconds CACHE_TTL_SECONDS]
                     [--cache-max-size CACHE_MAX_SIZE]
                     [--cache-similarity-threshold CACHE_SIMILARITY_THRESHOLD]

options:
  -h, --help            show this help message and exit
  --run-mode {smoke,benchmark}
  --model MODEL         model identifier sent to the API
  --url, --endpoint-url URL
                        server URL: http://host:port, http://host:port/v1, or
                        the exact chat-completions route (--endpoint-url is
                        accepted too)
  --api-key-env NAME    environment variable containing the bearer key
                        (default: OPENAI_API_KEY; for a localhost endpoint
                        with no key set, no key is sent)
  --allow-insecure-http
                        allow plaintext HTTP away from loopback (unsafe;
                        recorded in manifest)
  --backend {native,guidellm}
  --guidellm-prompt-tokens GUIDELLM_PROMPT_TOKENS
                        required with --backend guidellm; synthetic_text
                        prompt length
  --guidellm-executable GUIDELLM_EXECUTABLE
                        GuideLLM 0.7.3 executable (version is verified before
                        traffic)
  --allow-guidellm-validation-gaps
                        acknowledge cross-check-only completion/response-size
                        limitations
  --cost-model {unknown,dedicated-hourly,serverless-active-seconds,user-supplied}
                        how you pay for the endpoint (default: dedicated-
                        hourly when an hourly price is given, otherwise
                        unknown)
  --gpus GPUS
  --total-hourly-price, --gpu-hourly-rate, --gpu-rate-per-hour USD
                        what ALL GPUs behind the endpoint cost per hour
                        together, in dollars (--gpu-hourly-rate is accepted
                        too; implies --cost-model dedicated-hourly). For a
                        per-GPU price use --per-gpu-hourly-price with --gpus
  --per-gpu-hourly-price PER_GPU_HOURLY_PRICE
  --active-second-price ACTIVE_SECOND_PRICE
  --max-active-workers MAX_ACTIVE_WORKERS
  --billed-active-seconds BILLED_ACTIVE_SECONDS
  --user-supplied-total USER_SUPPLIED_TOTAL
  --allow-unknown-cost  explicitly acknowledge that the spend ceiling cannot
                        be enforced
  --prompts PROMPTS     measured JSONL workload
  --warmup-prompts WARMUP_PROMPTS
                        separate warm-up JSONL workload (default: bundled
                        separate set)
  --concurrency N [N ...]
  --request-rate RPS [RPS ...]
  --open-loop-max-in-flight OPEN_LOOP_MAX_IN_FLIGHT
                        in-flight ceiling for constant-rate traffic
  --max-tokens MAX_TOKENS
  --blocks BLOCKS
  --requests-per-block, --requests REQUESTS_PER_BLOCK
  --block-seconds BLOCK_SECONDS
  --warmup-requests WARMUP_REQUESTS
  --timeout-seconds TIMEOUT_SECONDS
  --stream, --no-stream
  --p95-slo-ms P95_SLO_MS
  --ttft-slo-ms TTFT_SLO_MS
  --seed SEED
  --max-requests MAX_REQUESTS
  --max-tokens-per-request MAX_TOKENS_PER_REQUEST
  --max-total-requested-tokens MAX_TOTAL_REQUESTED_TOKENS
  --max-elapsed-seconds MAX_ELAPSED_SECONDS
                        whole-run ceiling (default: 120 for smoke, 900 for
                        benchmark, 5400 for the six-position golden session,
                        60 for diagnose)
  --max-errors MAX_ERRORS
  --max-concurrency MAX_CONCURRENCY
  --max-response-bytes MAX_RESPONSE_BYTES
  --max-estimated-spend MAX_ESTIMATED_SPEND
  --cache-policy {unknown,disabled,cold,warm,representative}
                        declare the server's prefix-cache state for the report
                        (default: unknown; golden requires an explicit value).
                        This flag does not change the traffic: smoke,
                        benchmark and golden resend the same measured prompts
                        in every block and position (that fixed set is what
                        the recorded workload sha256 covers), so a server with
                        prefix caching on (vLLM V1 by default, SGLang, Ollama)
                        serves repeats warm. Use 'disabled' only if caching is
                        off on the server, 'warm' if it is on. 'throttle
                        check' sends unique prompts by default (see its
                        --warm-cache)
  --model-revision MODEL_REVISION
  --image-digest IMAGE_DIGEST
  --gpu, --accelerator GPU
  --gpu-fingerprint, --accelerator-fingerprint GPU_FINGERPRINT
  --cuda-version CUDA_VERSION
  --driver-version DRIVER_VERSION
  --accelerator-backend {cuda,metal,rocm,cpu}
  --accelerator-runtime-version ACCELERATOR_RUNTIME_VERSION
  --host-os-version HOST_OS_VERSION
  --software-environment-digest SOFTWARE_ENVIRONMENT_DIGEST
  --server-name SERVER_NAME
  --server-version SERVER_VERSION
  --engine-flag NAME=VALUE
                        effective, non-secret engine flag; repeat as needed
  --engine-flags-provenance {operator_attested,runtime_verified}
  --variant {baseline,candidate,unspecified}
  --sequence-position SEQUENCE_POSITION
  --evidence-source {unverified_endpoint,live_inference,synthetic_validation}
  --enable-cache        Enable similarity-based semantic caching to bypass API
                        requests
  --cache-ttl-seconds CACHE_TTL_SECONDS
                        Time-to-live for cached responses in seconds
  --cache-max-size CACHE_MAX_SIZE
                        Maximum number of items to hold in the cache (FIFO
                        eviction)
  --cache-similarity-threshold CACHE_SIMILARITY_THRESHOLD
                        Jaccard similarity threshold [0.0 - 1.0] for a cache
                        hit
```

</details>

<details><summary>Full `throttle smoke --help` output (0.4.2)</summary>

```text
usage: throttle smoke [-h] --model MODEL --url URL [--api-key-env NAME]
                      [--allow-insecure-http] [--backend {native,guidellm}]
                      [--guidellm-prompt-tokens GUIDELLM_PROMPT_TOKENS]
                      [--guidellm-executable GUIDELLM_EXECUTABLE]
                      [--allow-guidellm-validation-gaps]
                      [--cost-model {unknown,dedicated-hourly,serverless-active-seconds,user-supplied}]
                      [--gpus GPUS] [--total-hourly-price USD |
                      --per-gpu-hourly-price PER_GPU_HOURLY_PRICE]
                      [--active-second-price ACTIVE_SECOND_PRICE]
                      [--max-active-workers MAX_ACTIVE_WORKERS]
                      [--billed-active-seconds BILLED_ACTIVE_SECONDS]
                      [--user-supplied-total USER_SUPPLIED_TOTAL]
                      [--allow-unknown-cost] [--prompts PROMPTS]
                      [--warmup-prompts WARMUP_PROMPTS]
                      [--concurrency N [N ...] | --request-rate RPS [RPS ...]]
                      [--open-loop-max-in-flight OPEN_LOOP_MAX_IN_FLIGHT]
                      [--max-tokens MAX_TOKENS] [--blocks BLOCKS]
                      [--requests-per-block REQUESTS_PER_BLOCK |
                      --block-seconds BLOCK_SECONDS]
                      [--warmup-requests WARMUP_REQUESTS]
                      [--timeout-seconds TIMEOUT_SECONDS]
                      [--stream | --no-stream] [--p95-slo-ms P95_SLO_MS]
                      [--ttft-slo-ms TTFT_SLO_MS] [--seed SEED]
                      [--max-requests MAX_REQUESTS]
                      [--max-tokens-per-request MAX_TOKENS_PER_REQUEST]
                      [--max-total-requested-tokens MAX_TOTAL_REQUESTED_TOKENS]
                      [--max-elapsed-seconds MAX_ELAPSED_SECONDS]
                      [--max-errors MAX_ERRORS]
                      [--max-concurrency MAX_CONCURRENCY]
                      [--max-response-bytes MAX_RESPONSE_BYTES]
                      [--max-estimated-spend MAX_ESTIMATED_SPEND]
                      [--cache-policy {unknown,disabled,cold,warm,representative}]
                      [--model-revision MODEL_REVISION]
                      [--image-digest IMAGE_DIGEST] [--gpu GPU]
                      [--gpu-fingerprint GPU_FINGERPRINT]
                      [--cuda-version CUDA_VERSION]
                      [--driver-version DRIVER_VERSION]
                      [--accelerator-backend {cuda,metal,rocm,cpu}]
                      [--accelerator-runtime-version ACCELERATOR_RUNTIME_VERSION]
                      [--host-os-version HOST_OS_VERSION]
                      [--software-environment-digest SOFTWARE_ENVIRONMENT_DIGEST]
                      [--server-name SERVER_NAME]
                      [--server-version SERVER_VERSION]
                      [--engine-flag NAME=VALUE]
                      [--engine-flags-provenance {operator_attested,runtime_verified}]
                      [--variant {baseline,candidate,unspecified}]
                      [--sequence-position SEQUENCE_POSITION]
                      [--evidence-source {unverified_endpoint,live_inference,synthetic_validation}]
                      [--enable-cache] [--cache-ttl-seconds CACHE_TTL_SECONDS]
                      [--cache-max-size CACHE_MAX_SIZE]
                      [--cache-similarity-threshold CACHE_SIMILARITY_THRESHOLD]
                      [--output OUTPUT]

options:
  -h, --help            show this help message and exit
  --model MODEL         model identifier sent to the API
  --url, --endpoint-url URL
                        server URL: http://host:port, http://host:port/v1, or
                        the exact chat-completions route (--endpoint-url is
                        accepted too)
  --api-key-env NAME    environment variable containing the bearer key
                        (default: OPENAI_API_KEY; for a localhost endpoint
                        with no key set, no key is sent)
  --allow-insecure-http
                        allow plaintext HTTP away from loopback (unsafe;
                        recorded in manifest)
  --backend {native,guidellm}
  --guidellm-prompt-tokens GUIDELLM_PROMPT_TOKENS
                        required with --backend guidellm; synthetic_text
                        prompt length
  --guidellm-executable GUIDELLM_EXECUTABLE
                        GuideLLM 0.7.3 executable (version is verified before
                        traffic)
  --allow-guidellm-validation-gaps
                        acknowledge cross-check-only completion/response-size
                        limitations
  --cost-model {unknown,dedicated-hourly,serverless-active-seconds,user-supplied}
                        how you pay for the endpoint (default: dedicated-
                        hourly when an hourly price is given, otherwise
                        unknown)
  --gpus GPUS
  --total-hourly-price, --gpu-hourly-rate, --gpu-rate-per-hour USD
                        what ALL GPUs behind the endpoint cost per hour
                        together, in dollars (--gpu-hourly-rate is accepted
                        too; implies --cost-model dedicated-hourly). For a
                        per-GPU price use --per-gpu-hourly-price with --gpus
  --per-gpu-hourly-price PER_GPU_HOURLY_PRICE
  --active-second-price ACTIVE_SECOND_PRICE
  --max-active-workers MAX_ACTIVE_WORKERS
  --billed-active-seconds BILLED_ACTIVE_SECONDS
  --user-supplied-total USER_SUPPLIED_TOTAL
  --allow-unknown-cost  explicitly acknowledge that the spend ceiling cannot
                        be enforced
  --prompts PROMPTS     measured JSONL workload
  --warmup-prompts WARMUP_PROMPTS
                        separate warm-up JSONL workload (default: bundled
                        separate set)
  --concurrency N [N ...]
  --request-rate RPS [RPS ...]
  --open-loop-max-in-flight OPEN_LOOP_MAX_IN_FLIGHT
                        in-flight ceiling for constant-rate traffic
  --max-tokens MAX_TOKENS
  --blocks BLOCKS
  --requests-per-block, --requests REQUESTS_PER_BLOCK
  --block-seconds BLOCK_SECONDS
  --warmup-requests WARMUP_REQUESTS
  --timeout-seconds TIMEOUT_SECONDS
  --stream, --no-stream
  --p95-slo-ms P95_SLO_MS
  --ttft-slo-ms TTFT_SLO_MS
  --seed SEED
  --max-requests MAX_REQUESTS
  --max-tokens-per-request MAX_TOKENS_PER_REQUEST
  --max-total-requested-tokens MAX_TOTAL_REQUESTED_TOKENS
  --max-elapsed-seconds MAX_ELAPSED_SECONDS
                        whole-run ceiling (default: 120 for smoke, 900 for
                        benchmark, 5400 for the six-position golden session,
                        60 for diagnose)
  --max-errors MAX_ERRORS
  --max-concurrency MAX_CONCURRENCY
  --max-response-bytes MAX_RESPONSE_BYTES
  --max-estimated-spend MAX_ESTIMATED_SPEND
  --cache-policy {unknown,disabled,cold,warm,representative}
                        declare the server's prefix-cache state for the report
                        (default: unknown; golden requires an explicit value).
                        This flag does not change the traffic: smoke,
                        benchmark and golden resend the same measured prompts
                        in every block and position (that fixed set is what
                        the recorded workload sha256 covers), so a server with
                        prefix caching on (vLLM V1 by default, SGLang, Ollama)
                        serves repeats warm. Use 'disabled' only if caching is
                        off on the server, 'warm' if it is on. 'throttle
                        check' sends unique prompts by default (see its
                        --warm-cache)
  --model-revision MODEL_REVISION
  --image-digest IMAGE_DIGEST
  --gpu, --accelerator GPU
  --gpu-fingerprint, --accelerator-fingerprint GPU_FINGERPRINT
  --cuda-version CUDA_VERSION
  --driver-version DRIVER_VERSION
  --accelerator-backend {cuda,metal,rocm,cpu}
  --accelerator-runtime-version ACCELERATOR_RUNTIME_VERSION
  --host-os-version HOST_OS_VERSION
  --software-environment-digest SOFTWARE_ENVIRONMENT_DIGEST
  --server-name SERVER_NAME
  --server-version SERVER_VERSION
  --engine-flag NAME=VALUE
                        effective, non-secret engine flag; repeat as needed
  --engine-flags-provenance {operator_attested,runtime_verified}
  --variant {baseline,candidate,unspecified}
  --sequence-position SEQUENCE_POSITION
  --evidence-source {unverified_endpoint,live_inference,synthetic_validation}
  --enable-cache        Enable similarity-based semantic caching to bypass API
                        requests
  --cache-ttl-seconds CACHE_TTL_SECONDS
                        Time-to-live for cached responses in seconds
  --cache-max-size CACHE_MAX_SIZE
                        Maximum number of items to hold in the cache (FIFO
                        eviction)
  --cache-similarity-threshold CACHE_SIMILARITY_THRESHOLD
                        Jaccard similarity threshold [0.0 - 1.0] for a cache
                        hit
  --output OUTPUT
```

</details>

<details><summary>Full `throttle benchmark --help` output (0.4.2)</summary>

```text
usage: throttle benchmark [-h] --model MODEL --url URL [--api-key-env NAME]
                          [--allow-insecure-http]
                          [--backend {native,guidellm}]
                          [--guidellm-prompt-tokens GUIDELLM_PROMPT_TOKENS]
                          [--guidellm-executable GUIDELLM_EXECUTABLE]
                          [--allow-guidellm-validation-gaps]
                          [--cost-model {unknown,dedicated-hourly,serverless-active-seconds,user-supplied}]
                          [--gpus GPUS] [--total-hourly-price USD |
                          --per-gpu-hourly-price PER_GPU_HOURLY_PRICE]
                          [--active-second-price ACTIVE_SECOND_PRICE]
                          [--max-active-workers MAX_ACTIVE_WORKERS]
                          [--billed-active-seconds BILLED_ACTIVE_SECONDS]
                          [--user-supplied-total USER_SUPPLIED_TOTAL]
                          [--allow-unknown-cost] [--prompts PROMPTS]
                          [--warmup-prompts WARMUP_PROMPTS]
                          [--concurrency N [N ...] |
                          --request-rate RPS [RPS ...]]
                          [--open-loop-max-in-flight OPEN_LOOP_MAX_IN_FLIGHT]
                          [--max-tokens MAX_TOKENS] [--blocks BLOCKS]
                          [--requests-per-block REQUESTS_PER_BLOCK |
                          --block-seconds BLOCK_SECONDS]
                          [--warmup-requests WARMUP_REQUESTS]
                          [--timeout-seconds TIMEOUT_SECONDS]
                          [--stream | --no-stream] [--p95-slo-ms P95_SLO_MS]
                          [--ttft-slo-ms TTFT_SLO_MS] [--seed SEED]
                          [--max-requests MAX_REQUESTS]
                          [--max-tokens-per-request MAX_TOKENS_PER_REQUEST]
                          [--max-total-requested-tokens MAX_TOTAL_REQUESTED_TOKENS]
                          [--max-elapsed-seconds MAX_ELAPSED_SECONDS]
                          [--max-errors MAX_ERRORS]
                          [--max-concurrency MAX_CONCURRENCY]
                          [--max-response-bytes MAX_RESPONSE_BYTES]
                          [--max-estimated-spend MAX_ESTIMATED_SPEND]
                          [--cache-policy {unknown,disabled,cold,warm,representative}]
                          [--model-revision MODEL_REVISION]
                          [--image-digest IMAGE_DIGEST] [--gpu GPU]
                          [--gpu-fingerprint GPU_FINGERPRINT]
                          [--cuda-version CUDA_VERSION]
                          [--driver-version DRIVER_VERSION]
                          [--accelerator-backend {cuda,metal,rocm,cpu}]
                          [--accelerator-runtime-version ACCELERATOR_RUNTIME_VERSION]
                          [--host-os-version HOST_OS_VERSION]
                          [--software-environment-digest SOFTWARE_ENVIRONMENT_DIGEST]
                          [--server-name SERVER_NAME]
                          [--server-version SERVER_VERSION]
                          [--engine-flag NAME=VALUE]
                          [--engine-flags-provenance {operator_attested,runtime_verified}]
                          [--variant {baseline,candidate,unspecified}]
                          [--sequence-position SEQUENCE_POSITION]
                          [--evidence-source {unverified_endpoint,live_inference,synthetic_validation}]
                          [--enable-cache]
                          [--cache-ttl-seconds CACHE_TTL_SECONDS]
                          [--cache-max-size CACHE_MAX_SIZE]
                          [--cache-similarity-threshold CACHE_SIMILARITY_THRESHOLD]
                          [--output OUTPUT]

options:
  -h, --help            show this help message and exit
  --model MODEL         model identifier sent to the API
  --url, --endpoint-url URL
                        server URL: http://host:port, http://host:port/v1, or
                        the exact chat-completions route (--endpoint-url is
                        accepted too)
  --api-key-env NAME    environment variable containing the bearer key
                        (default: OPENAI_API_KEY; for a localhost endpoint
                        with no key set, no key is sent)
  --allow-insecure-http
                        allow plaintext HTTP away from loopback (unsafe;
                        recorded in manifest)
  --backend {native,guidellm}
  --guidellm-prompt-tokens GUIDELLM_PROMPT_TOKENS
                        required with --backend guidellm; synthetic_text
                        prompt length
  --guidellm-executable GUIDELLM_EXECUTABLE
                        GuideLLM 0.7.3 executable (version is verified before
                        traffic)
  --allow-guidellm-validation-gaps
                        acknowledge cross-check-only completion/response-size
                        limitations
  --cost-model {unknown,dedicated-hourly,serverless-active-seconds,user-supplied}
                        how you pay for the endpoint (default: dedicated-
                        hourly when an hourly price is given, otherwise
                        unknown)
  --gpus GPUS
  --total-hourly-price, --gpu-hourly-rate, --gpu-rate-per-hour USD
                        what ALL GPUs behind the endpoint cost per hour
                        together, in dollars (--gpu-hourly-rate is accepted
                        too; implies --cost-model dedicated-hourly). For a
                        per-GPU price use --per-gpu-hourly-price with --gpus
  --per-gpu-hourly-price PER_GPU_HOURLY_PRICE
  --active-second-price ACTIVE_SECOND_PRICE
  --max-active-workers MAX_ACTIVE_WORKERS
  --billed-active-seconds BILLED_ACTIVE_SECONDS
  --user-supplied-total USER_SUPPLIED_TOTAL
  --allow-unknown-cost  explicitly acknowledge that the spend ceiling cannot
                        be enforced
  --prompts PROMPTS     measured JSONL workload
  --warmup-prompts WARMUP_PROMPTS
                        separate warm-up JSONL workload (default: bundled
                        separate set)
  --concurrency N [N ...]
  --request-rate RPS [RPS ...]
  --open-loop-max-in-flight OPEN_LOOP_MAX_IN_FLIGHT
                        in-flight ceiling for constant-rate traffic
  --max-tokens MAX_TOKENS
  --blocks BLOCKS
  --requests-per-block, --requests REQUESTS_PER_BLOCK
  --block-seconds BLOCK_SECONDS
  --warmup-requests WARMUP_REQUESTS
  --timeout-seconds TIMEOUT_SECONDS
  --stream, --no-stream
  --p95-slo-ms P95_SLO_MS
  --ttft-slo-ms TTFT_SLO_MS
  --seed SEED
  --max-requests MAX_REQUESTS
  --max-tokens-per-request MAX_TOKENS_PER_REQUEST
  --max-total-requested-tokens MAX_TOTAL_REQUESTED_TOKENS
  --max-elapsed-seconds MAX_ELAPSED_SECONDS
                        whole-run ceiling (default: 120 for smoke, 900 for
                        benchmark, 5400 for the six-position golden session,
                        60 for diagnose)
  --max-errors MAX_ERRORS
  --max-concurrency MAX_CONCURRENCY
  --max-response-bytes MAX_RESPONSE_BYTES
  --max-estimated-spend MAX_ESTIMATED_SPEND
  --cache-policy {unknown,disabled,cold,warm,representative}
                        declare the server's prefix-cache state for the report
                        (default: unknown; golden requires an explicit value).
                        This flag does not change the traffic: smoke,
                        benchmark and golden resend the same measured prompts
                        in every block and position (that fixed set is what
                        the recorded workload sha256 covers), so a server with
                        prefix caching on (vLLM V1 by default, SGLang, Ollama)
                        serves repeats warm. Use 'disabled' only if caching is
                        off on the server, 'warm' if it is on. 'throttle
                        check' sends unique prompts by default (see its
                        --warm-cache)
  --model-revision MODEL_REVISION
  --image-digest IMAGE_DIGEST
  --gpu, --accelerator GPU
  --gpu-fingerprint, --accelerator-fingerprint GPU_FINGERPRINT
  --cuda-version CUDA_VERSION
  --driver-version DRIVER_VERSION
  --accelerator-backend {cuda,metal,rocm,cpu}
  --accelerator-runtime-version ACCELERATOR_RUNTIME_VERSION
  --host-os-version HOST_OS_VERSION
  --software-environment-digest SOFTWARE_ENVIRONMENT_DIGEST
  --server-name SERVER_NAME
  --server-version SERVER_VERSION
  --engine-flag NAME=VALUE
                        effective, non-secret engine flag; repeat as needed
  --engine-flags-provenance {operator_attested,runtime_verified}
  --variant {baseline,candidate,unspecified}
  --sequence-position SEQUENCE_POSITION
  --evidence-source {unverified_endpoint,live_inference,synthetic_validation}
  --enable-cache        Enable similarity-based semantic caching to bypass API
                        requests
  --cache-ttl-seconds CACHE_TTL_SECONDS
                        Time-to-live for cached responses in seconds
  --cache-max-size CACHE_MAX_SIZE
                        Maximum number of items to hold in the cache (FIFO
                        eviction)
  --cache-similarity-threshold CACHE_SIMILARITY_THRESHOLD
                        Jaccard similarity threshold [0.0 - 1.0] for a cache
                        hit
  --output OUTPUT
```

</details>

<details><summary>Full `throttle golden --help` output (0.4.2)</summary>

```text
usage: throttle golden [-h] --model MODEL --url URL [--api-key-env NAME]
                       [--allow-insecure-http] [--backend {native,guidellm}]
                       [--guidellm-prompt-tokens GUIDELLM_PROMPT_TOKENS]
                       [--guidellm-executable GUIDELLM_EXECUTABLE]
                       [--allow-guidellm-validation-gaps]
                       [--cost-model {unknown,dedicated-hourly,serverless-active-seconds,user-supplied}]
                       [--gpus GPUS] [--total-hourly-price USD |
                       --per-gpu-hourly-price PER_GPU_HOURLY_PRICE]
                       [--active-second-price ACTIVE_SECOND_PRICE]
                       [--max-active-workers MAX_ACTIVE_WORKERS]
                       [--billed-active-seconds BILLED_ACTIVE_SECONDS]
                       [--user-supplied-total USER_SUPPLIED_TOTAL]
                       [--allow-unknown-cost] [--prompts PROMPTS]
                       [--warmup-prompts WARMUP_PROMPTS]
                       [--concurrency N [N ...] |
                       --request-rate RPS [RPS ...]]
                       [--open-loop-max-in-flight OPEN_LOOP_MAX_IN_FLIGHT]
                       [--max-tokens MAX_TOKENS] [--blocks BLOCKS]
                       [--requests-per-block REQUESTS_PER_BLOCK |
                       --block-seconds BLOCK_SECONDS]
                       [--warmup-requests WARMUP_REQUESTS]
                       [--timeout-seconds TIMEOUT_SECONDS]
                       [--stream | --no-stream] [--p95-slo-ms P95_SLO_MS]
                       [--ttft-slo-ms TTFT_SLO_MS] [--seed SEED]
                       [--max-requests MAX_REQUESTS]
                       [--max-tokens-per-request MAX_TOKENS_PER_REQUEST]
                       [--max-total-requested-tokens MAX_TOTAL_REQUESTED_TOKENS]
                       [--max-elapsed-seconds MAX_ELAPSED_SECONDS]
                       [--max-errors MAX_ERRORS]
                       [--max-concurrency MAX_CONCURRENCY]
                       [--max-response-bytes MAX_RESPONSE_BYTES]
                       [--max-estimated-spend MAX_ESTIMATED_SPEND]
                       [--cache-policy {unknown,disabled,cold,warm,representative}]
                       [--model-revision MODEL_REVISION]
                       [--image-digest IMAGE_DIGEST] [--gpu GPU]
                       [--gpu-fingerprint GPU_FINGERPRINT]
                       [--cuda-version CUDA_VERSION]
                       [--driver-version DRIVER_VERSION]
                       [--accelerator-backend {cuda,metal,rocm,cpu}]
                       [--accelerator-runtime-version ACCELERATOR_RUNTIME_VERSION]
                       [--host-os-version HOST_OS_VERSION]
                       [--software-environment-digest SOFTWARE_ENVIRONMENT_DIGEST]
                       [--server-name SERVER_NAME]
                       [--server-version SERVER_VERSION]
                       [--engine-flag NAME=VALUE]
                       [--engine-flags-provenance {operator_attested,runtime_verified}]
                       [--variant {baseline,candidate,unspecified}]
                       [--sequence-position SEQUENCE_POSITION]
                       [--evidence-source {unverified_endpoint,live_inference,synthetic_validation}]
                       [--enable-cache]
                       [--cache-ttl-seconds CACHE_TTL_SECONDS]
                       [--cache-max-size CACHE_MAX_SIZE]
                       [--cache-similarity-threshold CACHE_SIMILARITY_THRESHOLD]
                       --baseline-config NAME=VALUE
                       --candidate-config NAME=VALUE [--dry-run]
                       [--output-dir OUTPUT_DIR] [--operator OPERATOR]
                       [--hardware-ownership {owned,rented}]
                       [--hardware-provider HARDWARE_PROVIDER]
                       [--hardware-rate-usd-per-hour HARDWARE_RATE_USD_PER_HOUR]
                       [--environment-note ENVIRONMENT_NOTE]
                       [--no-result-store]

Run B1/C1/B2/C2/B3/C3 against one endpoint/accelerator. Throttle pauses for
the operator to apply and verify each server configuration; it never
reconfigures the server itself.

options:
  -h, --help            show this help message and exit
  --model MODEL         model identifier sent to the API
  --url, --endpoint-url URL
                        server URL: http://host:port, http://host:port/v1, or
                        the exact chat-completions route (--endpoint-url is
                        accepted too)
  --api-key-env NAME    environment variable containing the bearer key
                        (default: OPENAI_API_KEY; for a localhost endpoint
                        with no key set, no key is sent)
  --allow-insecure-http
                        allow plaintext HTTP away from loopback (unsafe;
                        recorded in manifest)
  --backend {native,guidellm}
  --guidellm-prompt-tokens GUIDELLM_PROMPT_TOKENS
                        required with --backend guidellm; synthetic_text
                        prompt length
  --guidellm-executable GUIDELLM_EXECUTABLE
                        GuideLLM 0.7.3 executable (version is verified before
                        traffic)
  --allow-guidellm-validation-gaps
                        acknowledge cross-check-only completion/response-size
                        limitations
  --cost-model {unknown,dedicated-hourly,serverless-active-seconds,user-supplied}
                        how you pay for the endpoint (default: dedicated-
                        hourly when an hourly price is given, otherwise
                        unknown)
  --gpus GPUS
  --total-hourly-price, --gpu-hourly-rate, --gpu-rate-per-hour USD
                        what ALL GPUs behind the endpoint cost per hour
                        together, in dollars (--gpu-hourly-rate is accepted
                        too; implies --cost-model dedicated-hourly). For a
                        per-GPU price use --per-gpu-hourly-price with --gpus
  --per-gpu-hourly-price PER_GPU_HOURLY_PRICE
  --active-second-price ACTIVE_SECOND_PRICE
  --max-active-workers MAX_ACTIVE_WORKERS
  --billed-active-seconds BILLED_ACTIVE_SECONDS
  --user-supplied-total USER_SUPPLIED_TOTAL
  --allow-unknown-cost  explicitly acknowledge that the spend ceiling cannot
                        be enforced
  --prompts PROMPTS     measured JSONL workload
  --warmup-prompts WARMUP_PROMPTS
                        separate warm-up JSONL workload (default: bundled
                        separate set)
  --concurrency N [N ...]
  --request-rate RPS [RPS ...]
  --open-loop-max-in-flight OPEN_LOOP_MAX_IN_FLIGHT
                        in-flight ceiling for constant-rate traffic
  --max-tokens MAX_TOKENS
  --blocks BLOCKS
  --requests-per-block, --requests REQUESTS_PER_BLOCK
  --block-seconds BLOCK_SECONDS
  --warmup-requests WARMUP_REQUESTS
  --timeout-seconds TIMEOUT_SECONDS
  --stream, --no-stream
  --p95-slo-ms P95_SLO_MS
  --ttft-slo-ms TTFT_SLO_MS
  --seed SEED
  --max-requests MAX_REQUESTS
  --max-tokens-per-request MAX_TOKENS_PER_REQUEST
  --max-total-requested-tokens MAX_TOTAL_REQUESTED_TOKENS
  --max-elapsed-seconds MAX_ELAPSED_SECONDS
                        whole-run ceiling (default: 120 for smoke, 900 for
                        benchmark, 5400 for the six-position golden session,
                        60 for diagnose)
  --max-errors MAX_ERRORS
  --max-concurrency MAX_CONCURRENCY
  --max-response-bytes MAX_RESPONSE_BYTES
  --max-estimated-spend MAX_ESTIMATED_SPEND
  --cache-policy {unknown,disabled,cold,warm,representative}
                        declare the server's prefix-cache state for the report
                        (default: unknown; golden requires an explicit value).
                        This flag does not change the traffic: smoke,
                        benchmark and golden resend the same measured prompts
                        in every block and position (that fixed set is what
                        the recorded workload sha256 covers), so a server with
                        prefix caching on (vLLM V1 by default, SGLang, Ollama)
                        serves repeats warm. Use 'disabled' only if caching is
                        off on the server, 'warm' if it is on. 'throttle
                        check' sends unique prompts by default (see its
                        --warm-cache)
  --model-revision MODEL_REVISION
  --image-digest IMAGE_DIGEST
  --gpu, --accelerator GPU
  --gpu-fingerprint, --accelerator-fingerprint GPU_FINGERPRINT
  --cuda-version CUDA_VERSION
  --driver-version DRIVER_VERSION
  --accelerator-backend {cuda,metal,rocm,cpu}
  --accelerator-runtime-version ACCELERATOR_RUNTIME_VERSION
  --host-os-version HOST_OS_VERSION
  --software-environment-digest SOFTWARE_ENVIRONMENT_DIGEST
  --server-name SERVER_NAME
  --server-version SERVER_VERSION
  --engine-flag NAME=VALUE
                        effective, non-secret engine flag; repeat as needed
  --engine-flags-provenance {operator_attested,runtime_verified}
  --variant {baseline,candidate,unspecified}
  --sequence-position SEQUENCE_POSITION
  --evidence-source {unverified_endpoint,live_inference,synthetic_validation}
  --enable-cache        Enable similarity-based semantic caching to bypass API
                        requests
  --cache-ttl-seconds CACHE_TTL_SECONDS
                        Time-to-live for cached responses in seconds
  --cache-max-size CACHE_MAX_SIZE
                        Maximum number of items to hold in the cache (FIFO
                        eviction)
  --cache-similarity-threshold CACHE_SIMILARITY_THRESHOLD
                        Jaccard similarity threshold [0.0 - 1.0] for a cache
                        hit
  --baseline-config, --baseline-engine-flag NAME=VALUE
                        verified baseline treatment as canonical
                        max_num_seqs=INTEGER (1..2147483647)
  --candidate-config, --candidate-engine-flag NAME=VALUE
                        verified candidate treatment as a distinct canonical
                        max_num_seqs=INTEGER (1..2147483647)
  --dry-run, --plan     show all six positions and session ceilings without
                        reading a key or sending traffic
  --output-dir OUTPUT_DIR
                        new directory for B1.json through C3.json and
                        golden.json
  --operator OPERATOR   who is running this, for the result store's provenance
                        record (defaults to $USER@hostname if not set)
  --hardware-ownership {owned,rented}
                        required to persist this run to the result store; a
                        decision-eligible result whose ownership can't be
                        determined is not stored
  --hardware-provider HARDWARE_PROVIDER
                        e.g. runpod, lambda; only meaningful when --hardware-
                        ownership rented
  --hardware-rate-usd-per-hour HARDWARE_RATE_USD_PER_HOUR
  --environment-note ENVIRONMENT_NOTE
                        free text, e.g. 'RunPod pod, on-demand, deleted after
                        run'
  --no-result-store     don't check for or persist to the result store for
                        this run

Start with the same arguments plus --dry-run to inspect all six positions and
session ceilings without reading the API key or sending traffic. See the
Golden live protocol section in the project README for the complete pinned
example.
```

</details>

<details><summary>Full `throttle diagnose --help` output (0.4.2)</summary>

```text
usage: throttle diagnose [-h] --model MODEL --url URL [--api-key-env NAME]
                         [--allow-insecure-http] [--backend {native}]
                         [--cost-model {unknown,dedicated-hourly,serverless-active-seconds,user-supplied}]
                         [--gpus GPUS] [--total-hourly-price USD |
                         --per-gpu-hourly-price PER_GPU_HOURLY_PRICE]
                         [--active-second-price ACTIVE_SECOND_PRICE]
                         [--max-active-workers MAX_ACTIVE_WORKERS]
                         [--billed-active-seconds BILLED_ACTIVE_SECONDS]
                         [--user-supplied-total USER_SUPPLIED_TOTAL]
                         [--allow-unknown-cost] [--prompts PROMPTS]
                         [--warmup-prompts WARMUP_PROMPTS]
                         [--concurrency N [N ...] |
                         --request-rate RPS [RPS ...]]
                         [--open-loop-max-in-flight OPEN_LOOP_MAX_IN_FLIGHT]
                         [--max-tokens MAX_TOKENS] [--blocks BLOCKS]
                         [--requests-per-block REQUESTS_PER_BLOCK |
                         --block-seconds BLOCK_SECONDS]
                         [--warmup-requests WARMUP_REQUESTS]
                         [--timeout-seconds TIMEOUT_SECONDS]
                         [--stream | --no-stream] [--p95-slo-ms P95_SLO_MS]
                         [--ttft-slo-ms TTFT_SLO_MS] [--seed SEED]
                         [--max-requests MAX_REQUESTS]
                         [--max-tokens-per-request MAX_TOKENS_PER_REQUEST]
                         [--max-total-requested-tokens MAX_TOTAL_REQUESTED_TOKENS]
                         [--max-elapsed-seconds MAX_ELAPSED_SECONDS]
                         [--max-errors MAX_ERRORS]
                         [--max-concurrency MAX_CONCURRENCY]
                         [--max-response-bytes MAX_RESPONSE_BYTES]
                         [--max-estimated-spend MAX_ESTIMATED_SPEND]
                         [--output OUTPUT]

Run a short probe at multiple concurrency levels and classify the dominant
bottleneck regime. Use this before throttle golden to identify which config
dimensions are worth testing.

options:
  -h, --help            show this help message and exit
  --model MODEL         model identifier sent to the API
  --url, --endpoint-url URL
                        server URL: http://host:port, http://host:port/v1, or
                        the exact chat-completions route (--endpoint-url is
                        accepted too)
  --api-key-env NAME    environment variable containing the bearer key
                        (default: OPENAI_API_KEY; for a localhost endpoint
                        with no key set, no key is sent)
  --allow-insecure-http
                        allow plaintext HTTP away from loopback (unsafe;
                        recorded in manifest)
  --backend {native}
  --cost-model {unknown,dedicated-hourly,serverless-active-seconds,user-supplied}
                        how you pay for the endpoint (default: dedicated-
                        hourly when an hourly price is given, otherwise
                        unknown)
  --gpus GPUS
  --total-hourly-price, --gpu-hourly-rate, --gpu-rate-per-hour USD
                        what ALL GPUs behind the endpoint cost per hour
                        together, in dollars (--gpu-hourly-rate is accepted
                        too; implies --cost-model dedicated-hourly). For a
                        per-GPU price use --per-gpu-hourly-price with --gpus
  --per-gpu-hourly-price PER_GPU_HOURLY_PRICE
  --active-second-price ACTIVE_SECOND_PRICE
  --max-active-workers MAX_ACTIVE_WORKERS
  --billed-active-seconds BILLED_ACTIVE_SECONDS
  --user-supplied-total USER_SUPPLIED_TOTAL
  --allow-unknown-cost  explicitly acknowledge that the spend ceiling cannot
                        be enforced
  --prompts PROMPTS     measured JSONL workload
  --warmup-prompts WARMUP_PROMPTS
                        separate warm-up JSONL workload (default: bundled
                        separate set)
  --concurrency N [N ...]
  --request-rate RPS [RPS ...]
  --open-loop-max-in-flight OPEN_LOOP_MAX_IN_FLIGHT
                        in-flight ceiling for constant-rate traffic
  --max-tokens MAX_TOKENS
  --blocks BLOCKS
  --requests-per-block, --requests REQUESTS_PER_BLOCK
  --block-seconds BLOCK_SECONDS
  --warmup-requests WARMUP_REQUESTS
  --timeout-seconds TIMEOUT_SECONDS
  --stream, --no-stream
  --p95-slo-ms P95_SLO_MS
  --ttft-slo-ms TTFT_SLO_MS
  --seed SEED
  --max-requests MAX_REQUESTS
  --max-tokens-per-request MAX_TOKENS_PER_REQUEST
  --max-total-requested-tokens MAX_TOTAL_REQUESTED_TOKENS
  --max-elapsed-seconds MAX_ELAPSED_SECONDS
                        whole-run ceiling (default: 120 for smoke, 900 for
                        benchmark, 5400 for the six-position golden session,
                        60 for diagnose)
  --max-errors MAX_ERRORS
  --max-concurrency MAX_CONCURRENCY
  --max-response-bytes MAX_RESPONSE_BYTES
  --max-estimated-spend MAX_ESTIMATED_SPEND
  --output OUTPUT
```

</details>

<details><summary>Full `throttle experimental-tuning --help` output (0.4.2)</summary>

```text
usage: throttle experimental-tuning [-h] --model MODEL --url URL
                                    [--api-key-env NAME]
                                    [--allow-insecure-http]
                                    [--backend {native,guidellm}]
                                    [--guidellm-prompt-tokens GUIDELLM_PROMPT_TOKENS]
                                    [--guidellm-executable GUIDELLM_EXECUTABLE]
                                    [--allow-guidellm-validation-gaps]
                                    [--cost-model {unknown,dedicated-hourly,serverless-active-seconds,user-supplied}]
                                    [--gpus GPUS] [--total-hourly-price USD |
                                    --per-gpu-hourly-price PER_GPU_HOURLY_PRICE]
                                    [--active-second-price ACTIVE_SECOND_PRICE]
                                    [--max-active-workers MAX_ACTIVE_WORKERS]
                                    [--billed-active-seconds BILLED_ACTIVE_SECONDS]
                                    [--user-supplied-total USER_SUPPLIED_TOTAL]
                                    [--allow-unknown-cost] [--prompts PROMPTS]
                                    [--warmup-prompts WARMUP_PROMPTS]
                                    [--concurrency N [N ...] |
                                    --request-rate RPS [RPS ...]]
                                    [--open-loop-max-in-flight OPEN_LOOP_MAX_IN_FLIGHT]
                                    [--max-tokens MAX_TOKENS]
                                    [--blocks BLOCKS]
                                    [--requests-per-block REQUESTS_PER_BLOCK |
                                    --block-seconds BLOCK_SECONDS]
                                    [--warmup-requests WARMUP_REQUESTS]
                                    [--timeout-seconds TIMEOUT_SECONDS]
                                    [--stream | --no-stream]
                                    [--p95-slo-ms P95_SLO_MS]
                                    [--ttft-slo-ms TTFT_SLO_MS] [--seed SEED]
                                    [--max-requests MAX_REQUESTS]
                                    [--max-tokens-per-request MAX_TOKENS_PER_REQUEST]
                                    [--max-total-requested-tokens MAX_TOTAL_REQUESTED_TOKENS]
                                    [--max-elapsed-seconds MAX_ELAPSED_SECONDS]
                                    [--max-errors MAX_ERRORS]
                                    [--max-concurrency MAX_CONCURRENCY]
                                    [--max-response-bytes MAX_RESPONSE_BYTES]
                                    [--max-estimated-spend MAX_ESTIMATED_SPEND]
                                    [--cache-policy {unknown,disabled,cold,warm,representative}]
                                    [--model-revision MODEL_REVISION]
                                    [--image-digest IMAGE_DIGEST] [--gpu GPU]
                                    [--gpu-fingerprint GPU_FINGERPRINT]
                                    [--cuda-version CUDA_VERSION]
                                    [--driver-version DRIVER_VERSION]
                                    [--accelerator-backend {cuda,metal,rocm,cpu}]
                                    [--accelerator-runtime-version ACCELERATOR_RUNTIME_VERSION]
                                    [--host-os-version HOST_OS_VERSION]
                                    [--software-environment-digest SOFTWARE_ENVIRONMENT_DIGEST]
                                    [--server-name SERVER_NAME]
                                    [--server-version SERVER_VERSION]
                                    [--engine-flag NAME=VALUE]
                                    [--engine-flags-provenance {operator_attested,runtime_verified}]
                                    [--variant {baseline,candidate,unspecified}]
                                    [--sequence-position SEQUENCE_POSITION]
                                    [--evidence-source {unverified_endpoint,live_inference,synthetic_validation}]
                                    [--enable-cache]
                                    [--cache-ttl-seconds CACHE_TTL_SECONDS]
                                    [--cache-max-size CACHE_MAX_SIZE]
                                    [--cache-similarity-threshold CACHE_SIMILARITY_THRESHOLD]
                                    --metrics-url METRICS_URL
                                    [--attest-same-deployment-exclusive-metrics]
                                    [--output OUTPUT]
                                    [--experimental-output EXPERIMENTAL_OUTPUT]

Same inference deployment; no unrelated inference traffic. Both facts are
operator-attested, not independently proven. Run one native closed-loop smoke
workload while polling one explicit vLLM metrics exporter. Output is
suggestion-only: it never changes configuration, decision eligibility, or
Golden eligibility.

options:
  -h, --help            show this help message and exit
  --model MODEL         model identifier sent to the API
  --url, --endpoint-url URL
                        server URL: http://host:port, http://host:port/v1, or
                        the exact chat-completions route (--endpoint-url is
                        accepted too)
  --api-key-env NAME    environment variable containing the bearer key
                        (default: OPENAI_API_KEY; for a localhost endpoint
                        with no key set, no key is sent)
  --allow-insecure-http
                        allow plaintext HTTP away from loopback (unsafe;
                        recorded in manifest)
  --backend {native,guidellm}
  --guidellm-prompt-tokens GUIDELLM_PROMPT_TOKENS
                        required with --backend guidellm; synthetic_text
                        prompt length
  --guidellm-executable GUIDELLM_EXECUTABLE
                        GuideLLM 0.7.3 executable (version is verified before
                        traffic)
  --allow-guidellm-validation-gaps
                        acknowledge cross-check-only completion/response-size
                        limitations
  --cost-model {unknown,dedicated-hourly,serverless-active-seconds,user-supplied}
                        how you pay for the endpoint (default: dedicated-
                        hourly when an hourly price is given, otherwise
                        unknown)
  --gpus GPUS
  --total-hourly-price, --gpu-hourly-rate, --gpu-rate-per-hour USD
                        what ALL GPUs behind the endpoint cost per hour
                        together, in dollars (--gpu-hourly-rate is accepted
                        too; implies --cost-model dedicated-hourly). For a
                        per-GPU price use --per-gpu-hourly-price with --gpus
  --per-gpu-hourly-price PER_GPU_HOURLY_PRICE
  --active-second-price ACTIVE_SECOND_PRICE
  --max-active-workers MAX_ACTIVE_WORKERS
  --billed-active-seconds BILLED_ACTIVE_SECONDS
  --user-supplied-total USER_SUPPLIED_TOTAL
  --allow-unknown-cost  explicitly acknowledge that the spend ceiling cannot
                        be enforced
  --prompts PROMPTS     measured JSONL workload
  --warmup-prompts WARMUP_PROMPTS
                        separate warm-up JSONL workload (default: bundled
                        separate set)
  --concurrency N [N ...]
  --request-rate RPS [RPS ...]
  --open-loop-max-in-flight OPEN_LOOP_MAX_IN_FLIGHT
                        in-flight ceiling for constant-rate traffic
  --max-tokens MAX_TOKENS
  --blocks BLOCKS
  --requests-per-block, --requests REQUESTS_PER_BLOCK
  --block-seconds BLOCK_SECONDS
  --warmup-requests WARMUP_REQUESTS
  --timeout-seconds TIMEOUT_SECONDS
  --stream, --no-stream
  --p95-slo-ms P95_SLO_MS
  --ttft-slo-ms TTFT_SLO_MS
  --seed SEED
  --max-requests MAX_REQUESTS
  --max-tokens-per-request MAX_TOKENS_PER_REQUEST
  --max-total-requested-tokens MAX_TOTAL_REQUESTED_TOKENS
  --max-elapsed-seconds MAX_ELAPSED_SECONDS
                        whole-run ceiling (default: 120 for smoke, 900 for
                        benchmark, 5400 for the six-position golden session,
                        60 for diagnose)
  --max-errors MAX_ERRORS
  --max-concurrency MAX_CONCURRENCY
  --max-response-bytes MAX_RESPONSE_BYTES
  --max-estimated-spend MAX_ESTIMATED_SPEND
  --cache-policy {unknown,disabled,cold,warm,representative}
                        declare the server's prefix-cache state for the report
                        (default: unknown; golden requires an explicit value).
                        This flag does not change the traffic: smoke,
                        benchmark and golden resend the same measured prompts
                        in every block and position (that fixed set is what
                        the recorded workload sha256 covers), so a server with
                        prefix caching on (vLLM V1 by default, SGLang, Ollama)
                        serves repeats warm. Use 'disabled' only if caching is
                        off on the server, 'warm' if it is on. 'throttle
                        check' sends unique prompts by default (see its
                        --warm-cache)
  --model-revision MODEL_REVISION
  --image-digest IMAGE_DIGEST
  --gpu, --accelerator GPU
  --gpu-fingerprint, --accelerator-fingerprint GPU_FINGERPRINT
  --cuda-version CUDA_VERSION
  --driver-version DRIVER_VERSION
  --accelerator-backend {cuda,metal,rocm,cpu}
  --accelerator-runtime-version ACCELERATOR_RUNTIME_VERSION
  --host-os-version HOST_OS_VERSION
  --software-environment-digest SOFTWARE_ENVIRONMENT_DIGEST
  --server-name SERVER_NAME
  --server-version SERVER_VERSION
  --engine-flag NAME=VALUE
                        effective, non-secret engine flag; repeat as needed
  --engine-flags-provenance {operator_attested,runtime_verified}
  --variant {baseline,candidate,unspecified}
  --sequence-position SEQUENCE_POSITION
  --evidence-source {unverified_endpoint,live_inference,synthetic_validation}
  --enable-cache        Enable similarity-based semantic caching to bypass API
                        requests
  --cache-ttl-seconds CACHE_TTL_SECONDS
                        Time-to-live for cached responses in seconds
  --cache-max-size CACHE_MAX_SIZE
                        Maximum number of items to hold in the cache (FIFO
                        eviction)
  --cache-similarity-threshold CACHE_SIMILARITY_THRESHOLD
                        Jaccard similarity threshold [0.0 - 1.0] for a cache
                        hit
  --metrics-url METRICS_URL
                        explicit vLLM Prometheus endpoint; no credentials,
                        redirects, ambient proxies, or non-loopback plaintext
                        are allowed
  --attest-same-deployment-exclusive-metrics
                        attest that this exporter belongs to the same
                        inference deployment and that no unrelated inference
                        traffic reaches it during the sampled window; neither
                        fact is independently proven
  --output OUTPUT       ordinary schema-2.0 non-decision-grade smoke report;
                        parent directory must already exist and the create-
                        only file must not exist
  --experimental-output EXPERIMENTAL_OUTPUT
                        separate bound safety-validation envelope; parent
                        directory must already exist and the create-only file
                        must not exist

Defaults: one smoke block with 201 measured requests, 3 separate warm-ups, and
a 900-second traffic-run ceiling, plus bounded exporter-scrape and processing
overhead. Supply exactly one --concurrency and runtime-effective max_num_seqs
and max_num_batched_tokens engine flags.
```

</details>
