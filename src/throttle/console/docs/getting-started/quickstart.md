# Quickstart

*Get a real cost-per-million-tokens number in about two minutes.*

Pick whichever path fits what you have.

### Free GPU (Colab / Kaggle)

No local GPU needed. The notebook starts a small model on a free T4, runs `throttle check`, changes one setting and prints a verdict.

- **[Open in Colab](https://colab.research.google.com/github/KushagraKanaujia/throttle/blob/main/notebooks/throttle-quickstart.ipynb)**: Runtime → Change runtime type → T4 GPU, then Run all.
- **[Open in Kaggle](https://kaggle.com/kernels/welcome?src=https://github.com/KushagraKanaujia/throttle/blob/main/notebooks/throttle-quickstart.ipynb)**: Turn on a GPU accelerator in the notebook settings first.

### Local Ollama

#### 1. Pull a small model

```bash
ollama pull llama3.2:3b
```

#### 2. Install Throttle

```bash
pipx install throttle-pro
```

#### 3. Measure it

```bash
throttle check --url http://localhost:11434 --model llama3.2:3b --gpu-hourly-rate 1.50
```

A laptop has no real GPU price, so `1.50` is an assumed rate. The dollar figure scales with it; the tokens and time are measured.

### No GPU at all

```bash
pipx install throttle-pro
throttle demo
```

`throttle demo` simulates a baseline and a tuned configuration and prints a side-by-side cost comparison in seconds. Every number it prints is labelled SIMULATED.

## What happens next

The first check prints your cost per million output tokens with a 95% confidence interval and saves it. Run the same command two more times without changing anything: those repeats measure your machine's run-to-run noise. Then change one setting and run it again to get a verdict.

- **[Your first cost check](first-check.md)**: Walk through a full before/after comparison, step by step.
