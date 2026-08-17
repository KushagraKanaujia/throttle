# External pilot run card

Complete this together before any traffic. Store it privately. Do not write an
endpoint URL, API key, raw prompt, output, or sensitive company identifier here.

## Qualification

- Pilot ID:
- Date, window, and time zone:
- Authorized operator role:
- Staging/non-production confirmed: yes / no
- Emergency stop channel:
- vLLM version:
- Coarse model and GPU class:
- Real pending decision:
- Baseline configuration:
- Candidate configuration:
- Why this decision is pending now:
- What would happen without this benchmark:

## Success criteria declared before results

- p95 end-to-end SLO (ms or not applicable):
- TTFT SLO (ms or not applicable):
- Throughput/goodput objective:
- Error tolerance: zero malformed/failed measured requests
- Primary predeclared SLO/objective used for the decision:
- Decision rule if candidate passes:
- Decision rule if result is inconclusive:

## Workload and cache

- Approved measured workload description (no raw text):
- Approved separate warm-up workload description:
- Operator confirms the two sets are disjoint: yes / no
- Prompts are authorized and non-sensitive: yes / no
- `max_tokens`:
- Temperature: 0
- Stop tokens: none
- Native streaming enabled: yes / no
- Deterministic seed:
- Cache policy: disabled / cold / warm / representative
- Known background traffic/cold-start caveats:

## Immutable/runtime evidence

- Throttle version and package SHA-256:
- Full model revision:
- Immutable image digest (required for decision eligibility):
- Non-secret stable GPU label plus operator-controlled device proof: yes / no
- CUDA and driver versions:
- Effective engine flags verified at runtime: yes / no
- Evidence remains operator-controlled: yes / no

## Hard envelope

- Billing basis and declared rate, or non-metered attestation:
- Metered session cap (default maximum USD $5.00):
- Provider-side guard or full-session wall-clock cutoff:
- Smoke allocation:
- Identical per-benchmark allocation (used for both variants):
- Setup/restart/in-flight reserve:
- Allocation equation is within session cap: yes / no
- Smoke / per-benchmark max requests, including warm-ups:
- Smoke / per-benchmark max output tokens/request:
- Smoke / per-benchmark max total requested output tokens:
- Smoke / per-benchmark max elapsed seconds:
- Max errors: 1 (first error stops)
- Max concurrency/in-flight:
- Max response bytes:
- Request timeout:
- Source machine is controlled by operator: yes / no
- HTTPS, no insecure override: yes / no

## Data agreement

- Operator runs locally and retains the key: yes / no
- Founder may receive sanitized report: yes / no
- Founder may retain de-identified behavior event: yes / no
- Retention/deletion date (within 30 days unless separately renewed):
- Public result/company/quote permission: no unless separately written

## Go/no-go

- Exact `throttle plan` inspected by operator: yes / no
- Destination shown in terminal is correct: yes / no
- Request/token/time/spend/privacy disclosures accepted: yes / no
- Baseline health checked: yes / no
- Operator's explicit approval words and timestamp:

If any required item is blank or "no," the session is no-go.

## Closeout

- Final status: complete / inconclusive / invalid / stopped / cancelled
- Sanitized artifact SHA-256:
- Operator action stated immediately after result:
- Baseline restored and healthy: yes / no
- Temporary key/IP rule removed: yes / no
- No benchmark process or traffic remains: yes / no
- Provider charge when available:
- Teardown confirmed by operator and timestamp:
