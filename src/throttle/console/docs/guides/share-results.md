# Share your results

*Post a sanitized summary of a check. Nothing is uploaded.*

![](../../illustrations/share.svg)

```bash
throttle check ... --share              # measure, then print a shareable summary
throttle check --history --share        # share the latest recorded check, no traffic
throttle check --share-id CHECK_ID      # share a chosen check, no traffic
```

`--share` prints a markdown summary: engine, model, GPU (from `--config gpu=...`), the GPU hourly rate (ASSUMED), the workload and cache mode, $/M before and after with their confidence intervals, the verdict and noise-floor status, the Throttle version, and what changed in the config.

It **leaves out** the endpoint URL, hostnames, IPs, local paths and keys, and masks config values that look like secrets (`sk-...`, `hf_...`, long random tokens).

It also prints a link to a pre-filled GitHub issue ("Share your results"). You open the link, read it, and submit it yourself.

- **[Community results](../results/community.md)**: See what other people measured, and add yours.
