# Throttle: stop renting your AI.

Throttle cuts your AI bill. It finds the AI calls in your app that a cheaper
model can do just as well, proves it on your own traffic, and moves them.

It lives in your code: a GitHub Action, plus one change to your API base URL.
When a pull request adds or changes an AI call, Throttle tests that job against
other models, closed or open, on the requests your app really sends. If one
does the job as well for less, your team approves and the job moves. If the
new model errors or gives a broken answer, that request goes back to the API on
its own. Every miss is kept as a lesson.

Setup: one base-URL change to a proxy you run, a model server or cheaper API
model to test against, and a judge key you choose. The GitHub Action comments
on PRs.

Throttle is open source (MIT). We are looking for design partners: teams with a
real AI bill who want to cut it. It has not run on a company's production
traffic yet. Write to kush@throttle-pro.com.

**Why the proof matters** (real runs; the FP8 results are on a rented AMD
MI300X with vLLM, [raw data](validation/hotaisle-mi300x-20261001/)):

- An FP8 checkpoint of Qwen2.5-72B cost $1.13 per million tokens instead of
  $1.67, with correct answers.
- vLLM's on-the-fly FP8 looked 47% cheaper, but the answers broke. The 32B
  model replied "!!!!".
- On a MacBook running Ollama, the same untouched server moved 27% one way
  and 11% the other across four checks. A change has to beat that noise before
  Throttle calls it a win.

## Try it in a minute

```sh
pip install "throttle-pro==0.8.0"
throttle scan .
throttle demo
```

Pin the version so what you install never changes under you.

`throttle scan .` reads the code in the current
folder and lists the OpenAI and Anthropic SDK calls it can read: where each
one is, which model it uses, what it costs per 1,000 calls at list price, and
up to two cheaper models to test. It needs no API key, no network and no
traffic, and it writes nothing. It takes seconds on a normal repo. Every
dollar figure is an ESTIMATE: list price times an assumed 1,000 tokens in and
300 out per call (change it with `--tokens-in` and `--tokens-out`; add
`--calls-per-day N` for a monthly figure). The cheaper models are UNTESTED.
Scan says they cost less, not that they do the job as well. Tests, examples
and evals are left out of the totals (`--all` puts them back). A model set at
runtime is priced from a default written in the code, or from `--model NAME`.
Calls through your own wrapper or an agent framework are not found; `throttle
capture` sees them on real traffic. Scan ends with the commands that prove it
on your own traffic ([The loop](#the-loop)). `--json` gives the same data for
tools (schema `throttle.own.scan/1`).

`throttle demo` runs the whole loop offline on made-up traffic: capture,
workloads, shadow, eval and report. It needs no API key, no GPU and no
network. It writes `throttle-demo/report.html`. Every number in it is
labelled DEMO, because none of it is a measurement of a real company or a
real model.

Already serving your own model? `throttle check --url http://your-server:8000
--model <name> --total-hourly-price <usd>` measures what it really costs per
1M tokens, with a 95% interval, and refuses to call a change a win inside the
noise. See [Serving your own model](#serving-your-own-model).

## How Throttle is different

- **AI jobs are tracked in pull requests.** Each AI call in your code is a
  job. A pull request that adds or changes one gets a comment with its cost
  and its proof. A changed prompt is never moved silently: it goes back to
  the API until the new version passes.
- **It refuses to call a winner inside the noise.** vLLM's on-the-fly FP8
  "looked 47% cheaper", but the answers broke (the 32B model replied "!!!!").
  Throttle flags that as OUTPUT CHANGED, not a saving. The pre-quantized FP8
  checkpoint of Qwen2.5-72B on an MI300X went from $1.67 to $1.13 per 1M
  tokens (32% cheaper) with correct answers. That came from a serving change,
  measured against the same model in BF16, not against an API.
- **Fallback per request.** When a moved job's answer is an error, cut off,
  bad JSON or a bad tool call, that one request goes to the API instead.
- **Nothing hosted, nothing to shut down.** Your data, your models, your CI.
- **Flat fee.** No cut of each request.

Throttle's default path tests existing models and does not train on provider
outputs.

## The loop

```text
capture ──► workloads ──► shadow ──► eval ──► report ──► route ──► refresh
```

| Stage | Question it answers |
| --- | --- |
| Capture | What does your app send to the API? |
| Workloads | What kinds of work is it, and what does each cost? |
| Shadow | How does your own model answer the same requests? |
| Eval | Is it as good, per kind of work? |
| Report | What should move, and what does that save? |
| Route | Serve it from your model, with fallback. What did it save? |
| Refresh | Is the next open model better than the one you run? |

Every command keeps its data in `~/.throttle/own` (or `$THROTTLE_OWN_DIR`).
Every command takes `--dir` to use another folder.

### 1. Capture

Run a local pass-through proxy and point your clients at it.

```sh
throttle capture                   # listens on 127.0.0.1:8790, redaction "mask"
export OPENAI_BASE_URL=http://127.0.0.1:8790/v1
export ANTHROPIC_BASE_URL=http://127.0.0.1:8790
```

Every call goes to the API you already use, with the same headers, and the
answer comes back unchanged. Streaming is passed through chunk by chunk. A
redacted copy of each call is written to `~/.throttle/own/capture/`. Nothing
is cached. Your API keys are never written down.

- `--redact off|mask|hash|metadata` picks how much of each body is kept (see
  [Security model and data flow](#security-model-and-data-flow)).
- `--openai-upstream` and `--anthropic-upstream` point at a different
  upstream, such as a gateway you already run.
- `throttle capture stats` shows what you have: calls, errors, models, and how
  many calls reported token usage.

### 2. Workloads

```sh
throttle workloads
throttle workloads --since 7d --json
```

Groups captured calls by kind of work: classification, extraction,
summarization, agent, code or chat. A fixed list of rules decides, and the
first rule that matches wins. For example, a request that offers tools is
agent work, and a request that asks for JSON is extraction. Calls captured in
`metadata` mode have no bodies, so they show up as `unknown`.

For each kind it shows requests, tokens and API spend. Spend is the measured
token counts times the published list price. The price table is bundled,
with the date each price was checked and the page it came from. `--prices
FILE` adds or replaces prices; those are labelled ASSUMED. The monthly figure
scales the capture window to 30 days and is refused for a window shorter than
a day. `--embed` also splits each kind into clusters by prompt similarity
(needs `pip install "throttle-pro[embeddings]==0.8.0"`).

### 3. Shadow

Start an open-weight model on your own server (vLLM, SGLang, Ollama, LMDeploy
or anything OpenAI-compatible), then replay a sample of captured requests to
it.

```sh
throttle shadow --candidate-url http://gpu-box:8000/v1 \
    --candidate-model Qwen/Qwen2.5-72B-Instruct --label qwen72b@vllm
```

It samples up to `--per-type` requests of each kind (default 50) and at most
`--max-requests` in total (default 500). Anthropic requests are translated to
the OpenAI chat format. The answers are stored for `throttle eval` and are
never sent to any of your users. A timeout or connection failure is retried
once; a candidate that keeps failing counts as a loss in eval.

### 4. Eval

```sh
export OPENAI_API_KEY=...
throttle eval --candidate qwen72b@vllm --judge-model gpt-4.1 --judge-key-env OPENAI_API_KEY
```

For each kind of work, Throttle compares your model's answers with the API
model's captured answers in two ways:

- **Automatic checks**, scored the same way for both sides: valid JSON when
  JSON was asked for, matches the JSON schema when one was given, valid tool
  calls when tools were offered, same label for classification, and not cut
  off by the token limit.
- **A pairwise LLM judge** that sees the conversation and both answers, and
  is asked twice with the answers swapped.

The verdict per kind is BETTER, NO MATERIAL DIFFERENCE, WORSE, INCONCLUSIVE or
INSUFFICIENT DATA. The rules are in
[How we keep numbers honest](#how-we-keep-numbers-honest). The judge is capped
by `--max-judge-calls` (default 600) and, when its price is known,
`--judge-max-usd`. Use `--judge-provider anthropic` for a Claude judge,
`--judge-url` for a judge you host, or `--no-judge` for automatic checks only
(verdict CHECKS ONLY).

### 5. Report

Quality is half the answer. The other half is what your own model costs to
run. Measure that with `throttle check` on the candidate server, then build
the report:

```sh
throttle check --url http://gpu-box:8000 --model Qwen/Qwen2.5-72B-Instruct \
    --total-hourly-price 4.98 --gpus 2 --label qwen72b@vllm
throttle report --candidate qwen72b@vllm --gpu-hourly-rate 2.49 --gpus 2
```

`check` takes the hourly price of all the GPUs behind the server.
`report --gpu-hourly-rate` is the price of one GPU, and `--gpus` is how many
an always-on deployment needs.

Testing a cheaper hosted model instead (for example a smaller OpenAI or
Anthropic model, or an open model on a per-token provider)? Skip the GPU
flags and price it per token:

```sh
throttle report --candidate-model gpt-4.1-mini            # list price from the bundled price table
throttle report --candidate-price-in 0.20 --candidate-price-out 0.80   # your price per 1M tokens (sample values)
```

The cost always uses the candidate's own measured token counts, so a model
with a cheaper list price but longer answers can still lose.

The report gives one recommendation per kind of work:

| Recommendation | Means |
| --- | --- |
| **MOVE** | Quality is BETTER or NO MATERIAL DIFFERENCE, the owned cost is measured, and it saves money even at the expensive end of its confidence interval. |
| **KEEP ON API** | It costs more to own, or quality is worse and the answers break format (bad JSON, bad tool calls). |
| **NEEDS FINETUNE** | Quality is worse, but the format checks hold. A fine-tune may close the gap. This is a recommendation only; Throttle does not fine-tune. |
| **NEEDS MORE DATA** | Too few samples, an inconclusive eval, or no measured owned cost yet. |

It writes `report.json` and a self-contained `report.html` (to
`./throttle-report/`, or `--out DIR`). Without a `throttle check` record for
the candidate, the report still shows quality, but every cost and savings
cell says "not measured" and has no number. A dedicated GPU is paid for every
hour, so the report never prices your model below `--gpu-hourly-rate` x
`--gpus` x 730 hours a month.

### 6. Route

When eval says a kind of work is as good or better, approve it and turn on
routing:

```sh
throttle route approve --type extraction --candidate qwen72b@vllm \
    --url http://gpu-box:8000/v1 --model Qwen/Qwen2.5-72B-Instruct
throttle capture --route
```

`approve` is refused unless the latest eval for that candidate said BETTER or
NO MATERIAL DIFFERENCE for that kind. `--canary 0.1` serves only 10% of it.
`throttle route list` shows the routes and `throttle route revoke --type
extraction` stops one on the next request.

With `--route`, capture types each request and sends approved kinds to your
model. The answer comes back in the format your client expects (OpenAI or
Anthropic), with the header `x-throttle-served-by: owned`. Anything doubtful
falls back to the API with the original request: a connection error, a
timeout, an HTTP error, an empty or cut-off answer, invalid JSON, a schema
failure, or a tool call that is unknown or has bad arguments.

```sh
throttle route savings --month 2026-11 --gpu-hourly-rate 2.49 --gpus 2
```

This is a realized-savings statement you can audit (good for showing your
boss). It prices the requests your model actually served at the API price
they avoided, then subtracts what your GPUs cost for every hour of that
calendar month (720 hours in November). If your invoice is the better number,
pass `--owned-cost-usd` instead. If the net is zero or less, it says "No
savings this month." The report's 730 hours is an average month, used only
for its projection.

### 7. Refresh

When a new open model comes out, rerun the same saved eval sets on it:

```sh
throttle refresh --current qwen72b@vllm --new-label qwen3-72b@vllm \
    --new-url http://gpu-box:8001/v1 --new-model Qwen/Qwen3-72B \
    --judge-model gpt-4.1 --judge-key-env OPENAI_API_KEY
```

It compares the new model with the API model and with the model you run now,
on the same items. Per kind of work it says UPGRADE (not worse than either,
shown with confidence), STAY (worse than either) or NEEDS MORE DATA. Refresh
does not measure cost, so run `throttle check` on the new model before you
switch. `throttle refresh --watch-hf ORG/MODEL` only checks Hugging Face for a
newer revision or a sibling model.

## Throttle for GitHub

Throttle also watches your code. Each place in your code that calls OpenAI or
Anthropic is a **job**. On every pull request that adds, edits or removes a
job, Throttle leaves one comment: what that job costs on the API, whether your
own model passed the tests for it, and what happens next.

### Set up

Add `.github/workflows/throttle.yml`:

```yaml
on: pull_request
permissions: { contents: read, pull-requests: write }
jobs:
  throttle:
    runs-on: ubuntu-latest   # or a runner in your cloud that can read ~/.throttle/own
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
      - uses: KushagraKanaujia/throttle/.github/actions/throttle-pr@c817ceae8e0e70d8a1ea389df837f51f91a991d4 # v0.7.0
```

Pin the action by its full commit SHA, as above, not by `@main` or a tag: a
SHA cannot be moved under you. The action installs Throttle from the same
commit, so one pin covers both.

That is all. It runs in your CI with your workflow token, and there is no
Throttle server. On a GitHub-hosted runner it sees only the code, so the
comment lists the AI changes but no traffic costs. Run it on a runner that can
read your Throttle data folder (input `data-dir`) to get costs and proof. Add
`candidate-url`, `candidate-model` and `judge-model` to get a first look at
new jobs. Keys go in as the *names* of env vars (`api-key-env`,
`judge-key-env`), set from your secrets.

Keep it on `pull_request`, never `pull_request_target`. On a public
repository the comment and the job summary are public: anything in your data
folder that the comment shows (costs, call counts) is visible to everyone, and
a self-hosted runner that can read your data folder should not run workflows
from forks. A pull request from a fork gets a read-only token, so the check
goes to the job summary instead of a comment. Throttle only ever edits its own
bot comment, never a comment someone else wrote. At most 5 new jobs per pull
request get a first look (`max-first-looks`), and model answers in the comment
are shown as plain code, so links and @mentions in them do nothing.

Without GitHub, the same check runs anywhere:

```sh
throttle pr-check --base origin/main --head HEAD --markdown comment.md --json check.json
```

### What the comment says

Money first, then proof, then the action. For each job:

| What happens | When |
| --- | --- |
| **Ready to move** | Its own job eval passed, and your model costs less per call. Approve it with `throttle jobs approve`. |
| **Passed, measure cost** | Its job eval passed, but there is no `throttle check` of your model yet. |
| **Keep on the API** | Your model breaks the output format, or costs more per call. |
| **Keep on the API for now** | Your model loses on quality but keeps the format. Its misses are saved as lessons for a later fine-tune. |
| **Test it more** | New, or not enough judged calls yet. Stays on the API. |
| **Back to testing** | It runs on your model today and this pull request changes its prompt, model, tools or JSON mode. When it merges, it goes back to the API until the new version passes. |
| **Route retired** | The call is gone. When this merges, its route is removed. |

The cost comes from captured calls that match the job, priced at list
price. A monthly figure needs at least a day of captured traffic; before
that it is per 1,000 calls. When the prompt changed, traffic from the old
version is labelled "old version" and is not added to the monthly total.

**First look.** A brand-new job has no traffic. If you set a candidate and a
judge, Throttle has the judge write a few made-up inputs from the prompt
(default 8), sends each to the API model and to your model, and has the judge
compare the answers in both orders. It is labelled "first look, not enough
data yet". It is not a verdict and it never moves anything. It costs one API
call, one call to your model and two judge calls per input, on your keys.

### Jobs follow the code

```sh
throttle jobs scan                       # list the AI calls in the code
throttle jobs sync --repo .              # apply the code to the job list (run on merge to main)
throttle jobs eval --repo . --candidate qwen72b@vllm --judge-model gpt-4.1 --judge-key-env OPENAI_API_KEY
throttle jobs approve JOB --candidate qwen72b@vllm --url http://gpu-box:8000/v1 --model Qwen/Qwen2.5-72B-Instruct
throttle jobs list | log JOB | undo JOB | pin JOB | unpin JOB | lessons
```

- A job's id is `path::function` (plus `#2` for a second call in one
  function), or the name you tag it with. Its prompt is tracked separately,
  so an edited prompt is the same job, changed.
- `sync` retires removed jobs and their routes. When a moved job's prompt,
  model, tools or JSON mode changes, its move is suspended and it goes back
  to testing. A job that moved to another file with the same prompt keeps
  its state.
- `approve` needs a passing job eval of exactly the current version of the
  call: same prompt, model, tools, JSON mode and call settings (few-shot
  messages, tool schemas, temperature ...). The eval only counts captured
  calls that match that version. A job whose prompt Throttle cannot read
  (built at run time or imported from another module) is never moved, since
  Throttle could not see it change.
- Even before a sync, `throttle capture --route` serves a moved job from
  your model only while the request still matches the version it was
  approved on: every readable prompt template, the model, tools and JSON
  mode. Anything else goes to the API. A changed call is never moved silently.
- Once a job is tracked, only its own state decides where it runs. Workload
  type routes apply to traffic that is not a tracked job.
- Every fallback and every judged loss of a job is saved to
  `~/.throttle/own/jobs/lessons/` with the API's answer and your model's
  answer. Nothing trains on them yet; they are kept for the fine-tune loop.
  First looks in pull requests never add lessons: made-up inputs from a pull
  request that may never merge are not your traffic.

**Tagging (optional).** Throttle matches traffic to jobs by prompt. To make
it exact, and to keep a job's name when code moves, send a header:

```python
from throttle.own.tag import throttle_job
client.chat.completions.create(model="gpt-4o-mini", messages=msgs, extra_headers=throttle_job("ticket-summary"))
```

In JavaScript, pass `{ headers: { "X-Throttle-Job": "ticket-summary" } }` as
the request options. `throttle capture` reads the header and does not forward
it. A `# throttle-job: ticket-summary` comment on the line above a call names
it for pull request checks only.

**What the scanner finds.** Python (parsed with the standard library):
`chat.completions.create/parse/stream`, `responses.create`,
`completions.create`, Anthropic `messages.create/stream`, `litellm`
`completion`, and LangChain `ChatOpenAI`/`ChatAnthropic` (model only).
JavaScript and TypeScript (a small tokenizer): the same SDK calls plus the
Vercel AI SDK (`generateText`, `streamText`, `generateObject`,
`streamObject`). It reads prompts that are literals, f-strings, template
literals, `+` of strings, `.format()`, or a name set to one of those in the
same function or file.

## How we keep numbers honest

**Every number has a label.**

| Label | Means |
| --- | --- |
| MEASURED | Token counts and times we saw, and $/M tokens from a `throttle check` run with a 95% confidence interval. |
| LIST PRICE | A published API price, with the date it was checked and its source page. |
| ASSUMED | A number you gave us: your GPU hourly rate, or a price from `--prices`. |
| DEMO | Made up, from `throttle demo`. Not a measurement of anything. |
| ESTIMATE | From `throttle scan`: list price times a token count per call that is ASSUMED or that you gave. Not measured. |
| UNTESTED | A cheaper model `throttle scan` names. It costs less at list price. Nobody has checked yet that it does the job as well. |

`route savings --owned-cost-usd` is labelled OPERATOR-REPORTED.

**Missing usage is left out, never estimated.** If the API did not report
token counts for a call, that call is excluded from cost math and counted as
excluded. The same goes for a model with no known price.

**The quality verdict, in plain words.**

- Each item is judged twice, with the two answers swapped. It is a win or a
  loss only when both orders agree. It is a tie only when both say tie.
- When the two orders disagree, the judge is unsure. The item scores 0 like a
  tie, but it is shown on its own as "judge unsure" and never counted as a
  tie. If more than 30% of items disagree, the judge is mostly picking a
  position, not an answer, and the verdict is INSUFFICIENT DATA. More than 20%
  judge errors also means INSUFFICIENT DATA.
- Fewer than 30 judged items (`--min-n`) is INSUFFICIENT DATA.
- WORSE: we are confident your model loses by more than the margin (default 5
  points of win minus loss), or one of its check pass rates is clearly below
  the API model's.
- BETTER: the low end of the win rate beats the high end of the loss rate by
  more than the margin.
- NO MATERIAL DIFFERENCE: the 95% interval for win minus loss sits inside
  plus or minus 5 points, AND the loss rate is shown to be at most 5%. That
  second bound needs about 75 loss-free items. So a small sample of all ties
  is INCONCLUSIVE, not "no difference". A small sample cannot prove that two
  models are equal, so we do not say it does.
- Anything else is INCONCLUSIVE. In the report, INSUFFICIENT DATA and
  INCONCLUSIVE both become NEEDS MORE DATA.

**Savings only from measured runs.** The report shows a savings number only
when there is a measured `throttle check` of your model. It uses the
expensive end of that check's confidence interval. The realized-savings
number comes only from requests your model actually served. Requests that fell back to the
API are already on your API bill and are not counted as savings.

**Forced routes are never counted.** `route approve --force` skips the
passing eval. Those routes are marked FORCED everywhere, shown on their own
line in `route savings`, and never counted in the net.

**Prompt caching is counted against us.** When we price what the API would
have charged, we use the share of input the API served from its prompt cache
for that kind of work, priced at the cheaper cached rate. With no cache
history, all input is priced at the cached rate, which is the lowest the API
could have charged.

## Security model and data flow

**What runs where.** Everything Throttle runs is yours:

- **A proxy you run.** `throttle capture` is a pass-through proxy on your
  machine or server. It binds to `127.0.0.1` by default and warns you if you
  bind it anywhere else.
- **Your CI.** The GitHub Action runs on your runner with your workflow token.
  There is no Throttle server.
- **The CLI.** Workloads, shadow, eval, report, route and refresh run where
  you run them. Data stays in `~/.throttle/own` (or `$THROTTLE_OWN_DIR`).

**What leaves your infra.**

- The API calls your app already makes, unchanged.
- **Judge calls** go to the provider and key you choose. If the judge is a
  public API, the captured prompts and answers it judges go to that provider.
  Use your current provider, or a judge you host.
- **Candidate API models** see the prompts you send them. A model you host
  sees them only on your own hardware.
- **Throttle the company receives nothing.** No telemetry: the CLI does not
  phone home, log in or meter anything.

**Keys.** API keys are read from environment variables you name. They are
never printed or written to disk, and capture never stores auth headers.

**Redaction modes** for captured bodies (`throttle capture --redact`):

- `mask` (default): emails, phone numbers, US SSNs, card numbers, IP
  addresses and secrets (API keys, tokens, passwords) become placeholders
  like `<EMAIL>`. Inline images and audio become `<IMAGE>` or `<AUDIO>`.
- `hash`: the same, plus a salted hash, so repeats of one value can be
  matched. Set the salt with `THROTTLE_OWN_SALT`.
- `metadata`: no bodies at all, only model, token counts and timing.
  Workloads still shows spend, but shadow and eval need bodies.
- `off`: bodies kept as sent.

Pattern redaction does not catch names, street addresses, dates of birth, or
free-text medical or financial details. It matches patterns; it does not
understand text. If that content cannot sit on your disk, capture with
`--redact metadata`.

**Supply chain.** Releases go to PyPI through trusted publishing (OIDC), so no
upload token is stored anywhere, and the release workflow pins every action it
uses by SHA. Pin your install and the GitHub Action to a full commit SHA, as
shown above. This matters most for the capture proxy: it sees your API keys in
flight, so run only code you have pinned.

To report a vulnerability, see [SECURITY.md](SECURITY.md).

## Serving your own model

Throttle also measures the server you run your model on. These commands
started the project and still work on vLLM, SGLang, Ollama, LMDeploy and any
OpenAI-compatible server.

- **`throttle check`**: dollars per 1M tokens with a 95% confidence interval.
  It calibrates against your machine's run-to-run noise and calls CHEAPER or
  MORE EXPENSIVE only when a change beats that noise. Its record is what
  `throttle report` prices your model with.

  ```sh
  throttle check --url http://localhost:8000 --model Qwen/Qwen2.5-7B-Instruct --gpu-hourly-rate 2.49
  ```

- **`throttle autotune`**: tries serving configs and keeps the cheapest one
  that is cheaper beyond the noise and whose output did not change shape.
  `throttle autotune --dry-run` shows it on a simulated server.
- **`throttle coldstart`**: launches your server command and times how long it
  takes to answer and to send its first token. Repeat `--cmd` to compare
  engines.
- **`throttle compare`**: runs your own prompts against two live endpoints
  and says whether to switch.

Docs: `throttle ui` opens a local console with the full docs, or read them in
[`src/throttle/console/docs`](src/throttle/console/docs). Methodology:
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Receipts

Real runs only. Each one links to its raw data.

| Change | Setup | Result |
| --- | --- | --- |
| One vLLM flag: `max_num_seqs` 1 to 8 | A100 PCIe 80GB, vLLM 0.16.0, Qwen2.5-0.5B-Instruct, $1.39/hr (RunPod rate) | $0.746 to $0.234 per 1M output tokens (-68.6%), six-position counterbalanced protocol, decision-eligible. [Raw data](https://github.com/KushagraKanaujia/throttle/tree/main/validation/golden-live-20260817) |
| 1 GPU to 2 GPUs (tensor parallel 2) | AMD MI300X (Hot Aisle), vLLM 0.23.1 ROCm, Qwen2.5-72B-Instruct BF16, $2.99/GPU-hr ASSUMED | $1.67 to $2.14 per 1M output tokens (+28%). 1.56x the throughput, but two GPUs per hour. MORE EXPENSIVE in 3 of 3 checks. [Raw checks](https://github.com/KushagraKanaujia/throttle/tree/main/validation/hotaisle-mi300x-20261001) |
| vLLM on-the-fly FP8 | Same MI300X, Qwen2.5-72B and 32B | Looked 47% and 41% cheaper, but the answers broke: the 72B ran to the token cap and the 32B answered "!!!!". OUTPUT CHANGED, so not a saving. A pre-quantized FP8 checkpoint was 32% cheaper on 72B with correct answers. [Raw checks and answers](https://github.com/KushagraKanaujia/throttle/tree/main/validation/hotaisle-mi300x-20261001) |

The A100 baseline was deliberately bad (one sequence at a time), so it shows
the measurement, not a saving to expect.

**The loop has not yet run on a real company's traffic.** Its tests use
fakes, and `throttle demo` uses made-up data. When it runs on real traffic,
the results go here.

## Limits

- **Owned routes do not truly stream.** When a client asks for a stream and
  your model serves it, Throttle waits for the full answer and replays it as
  a short stream. Time to first token is the full answer time.
- **Routing types a request before the answer exists.** That can differ from
  what `throttle workloads` says later. For example, classification needs a
  short answer, which before the call can only be read from a small
  `max_tokens`.
- **A failed owned answer adds latency.** Each routed request tries your
  model first. A fallback waits for that try (up to the timeout) before the
  API call.
- **Requests with images, `n` > 1, or no messages are never routed.**
  Anthropic `thinking`, `cache_control` and `tool_choice` are not sent to
  your model.
- **An LLM judge is not ground truth.** We run it both ways round and
  refuse a verdict when it is position-biased, but read some answers
  yourself before you move traffic.
- **No fine-tuning pipeline yet.** NEEDS FINETUNE is a recommendation only.
- **Pattern redaction misses names and addresses.** Use `metadata` mode if
  that matters.
- **The workload rules are simple.** They are a fixed list of patterns, not a
  model. `--embed` adds similarity clusters.
- **The code scanner reads what it can see.** Prompts built at run time (from
  a database, a template file, another module) cannot be read: the job is
  still found and diffed by model, but there is no first look, and its
  traffic is matched only by the `X-Throttle-Job` header. A JavaScript regular
  expression that contains a quote or a bracket can confuse the tokenizer.
  OpenAI Responses API traffic is found in code but not yet captured.
- **Template matching needs fixed text.** A prompt needs at least 12 fixed
  characters for Throttle to recognise its traffic. Two jobs with the same
  prompt need a tag to tell them apart.

## Development

```sh
git clone https://github.com/KushagraKanaujia/throttle && cd throttle
pip install -e . pytest pytest-asyncio
python -m pytest tests -q
```

Tests run offline. See [CONTRIBUTING.md](CONTRIBUTING.md) and
[CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
