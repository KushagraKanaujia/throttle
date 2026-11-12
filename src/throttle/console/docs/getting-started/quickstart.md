# Quickstart

*See the whole loop in a minute, then run it on your own traffic.*

## 1. Install

```bash
pip install "throttle-pro==0.7.0"
```

This pins v0.7.0. To install from the source tag instead, see Installation.

## 2. Scan your code

```bash
throttle scan .
```

It finds every OpenAI and Anthropic call in the folder, prices each one at list price, and names up to two cheaper models to test. Offline, no keys, in seconds. Every dollar is an ESTIMATE and every cheaper model is UNTESTED. See [Scan your code](scan.md).

## 3. Run the demo

```bash
throttle demo
```

It runs capture, workloads, shadow, eval and report on made-up traffic, offline, with no keys and no GPU. Open `throttle-demo/report.html`. Every number is labelled DEMO. See [Demo](../loop/demo.md).

## 4. Run it on your traffic

```bash
# capture: a local pass-through proxy
throttle capture
export OPENAI_BASE_URL=http://127.0.0.1:8790/v1
export ANTHROPIC_BASE_URL=http://127.0.0.1:8790

# after a day or more of traffic
throttle workloads

# your own model on your own server
throttle shadow --candidate-url http://gpu-box:8000/v1 \
    --candidate-model Qwen/Qwen2.5-72B-Instruct --label qwen72b@vllm
throttle eval --candidate qwen72b@vllm --judge-model gpt-4.1 --judge-key-env OPENAI_API_KEY

# measure what it costs, then the report
throttle check --url http://gpu-box:8000 --model Qwen/Qwen2.5-72B-Instruct \
    --total-hourly-price 4.98 --gpus 2 --label qwen72b@vllm
throttle report --candidate qwen72b@vllm --gpu-hourly-rate 2.49 --gpus 2
```

Each step has its own page under [The loop](../loop/overview.md).

## What happens next

When a kind of work says MOVE, approve it and serve it from your model with [Route](../loop/route.md). When a new open model ships, test it with [Refresh](../loop/refresh.md).

- **[Your first cost check](first-check.md)**: More on `throttle check`, the measurement the report prices your model with.
