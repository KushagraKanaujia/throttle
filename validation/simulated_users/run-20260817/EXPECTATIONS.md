# Predeclared simulated-user run expectations

These expectations were written before viewing any run result. All evidence is
local fixture smoke evidence and is permanently non-decision-grade.

Common envelope: closed-loop concurrency `1, 2, 4, 8, 16`; one measured block;
16 measured requests and one separate warm-up per condition; 85 maximum calls;
64 maximum output tokens per request; 5,440 reserved output tokens; 30 seconds;
one error; concurrency 16; 65,536 response bytes; unknown billing explicitly
acknowledged; synthetic-validation provenance; streaming; cache disabled.

| Case | Expected artifact | Expected status | Expected traffic result |
|---|---|---|---|
| short chat | `short_chat.json` | complete / exit 0 | 85/85 valid; five valid conditions |
| support ticket | `support_ticket.json` | complete / exit 0 | 85/85 valid; five valid conditions |
| code assistant | `code_assistant.json` | complete / exit 0 | 85/85 valid; five valid conditions |
| retrieval QA | `retrieval_qa.json` | complete / exit 0 | 85/85 valid; five valid conditions |
| document summary | `document_summary.json` | complete / exit 0 | 85/85 valid; five valid conditions |
| mixed workload | `mixed_workload.json` | complete / exit 0 | 85/85 valid; five valid conditions |
| large-prompt pressure | `stress_large.json` | stopped / exit 1 | levels 1/2/4/8 valid; level 16 invalid after fixed HTTP 503; stop reason `max_errors` |

Every report must use schema 2, mode `smoke`, evidence source
`synthetic_validation`, and `decision_eligible=false`. No report may support an
inference, GPU, capacity, production, recommendation, or savings claim.
