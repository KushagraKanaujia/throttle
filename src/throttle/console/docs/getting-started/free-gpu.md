# Try on a free GPU

*Run Throttle on a free Colab or Kaggle T4 without owning a GPU.*

The quickstart notebook lives in the repo at `notebooks/throttle-quickstart.ipynb`.

- **[Open in Colab](https://colab.research.google.com/github/KushagraKanaujia/throttle/blob/main/notebooks/throttle-quickstart.ipynb)**
- **[Open in Kaggle](https://kaggle.com/kernels/welcome?src=https://github.com/KushagraKanaujia/throttle/blob/main/notebooks/throttle-quickstart.ipynb)**

What it does:

1. Starts a small model on the free T4 with an OpenAI-compatible server (Ollama by default; vLLM optional).
2. Installs `throttle-pro` and runs `throttle check` four times to calibrate noise.
3. Changes one setting and checks again for a verdict.
4. Ends with `throttle check --history` and `--share`, so you can post your result.

> **Note:** Free notebook GPUs are shared and their hourly price is not real. Treat the dollar figures as relative (the GPU rate is labelled ASSUMED); the tokens and time are measured.
