# Your first cost check

*Measure, change one setting, and get a verdict you can trust.*

This walkthrough uses a local Ollama server. Any OpenAI-compatible endpoint works the same way: swap the URL and model.

### 1. Record today's config three times

```bash
throttle check --url http://localhost:11434 --model llama3.2:3b \
--gpu-hourly-rate 1.50 --label before
```

Run it **three times** without changing anything. Throttle compares each check with the last one for this endpoint. Until it has three recent repeats of an unchanged config, the verdict is **NOT CALIBRATED**. That's expected: the repeats are how it measures run-to-run noise.

### 2. Change one serving setting

For example, restart Ollama with more parallel slots:
```bash
OLLAMA_NUM_PARALLEL=4 ollama serve
```

### 3. Check again

```bash
throttle check --url http://localhost:11434 --model llama3.2:3b \
--gpu-hourly-rate 1.50 --label after --config OLLAMA_NUM_PARALLEL=4
```

`--config` only *records* what you changed, so it shows up under "what changed". Throttle never changes your server.

### 4. Read the verdict

- **CHEAPER / MORE EXPENSIVE**: the change is bigger than the measured noise bound and the 95% intervals don't overlap.
- **NO WINNER**: the difference is inside the noise, or the intervals overlap.
- **NOT CALIBRATED**: not enough recent repeats to know the noise yet.

## Keep the workload identical

Every check of the same comparison must use the same `--blocks`, `--requests-per-block`, `--concurrency`, `--max-tokens`, prompts and cache mode. Throttle refuses to call a winner between different workloads.

> **Warning:** Match `--concurrency` to your production load. Cost per token depends heavily on how many requests are in flight at once.

## See your history

```bash
throttle check --history
```

Lists past checks (sends no traffic). Share one with `throttle check --share-id CHECK_ID`.
