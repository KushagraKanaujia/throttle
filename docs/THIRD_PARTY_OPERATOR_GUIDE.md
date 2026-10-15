# Recording a run as someone other than the maintainer

This is written from an actual attempt, not from the design docs: a fresh
operator identity (not Kushagra, not the result store's author), a
non-GPU machine, a fresh clone, no shortcuts. Two real points of friction
came out of it, both fixed or documented below; the rest confirms the
happy path works as designed.

## What you need to supply

Beyond the usual `golden` flags, six are specific to the result store:

| Flag | Required? | What happens if you skip it |
|---|---|---|
| `--operator` | No, but you should set it | Falls back to `$USER@$(hostname)`. Fine on your own machine, meaningless in a container or CI runner, set it explicitly there. |
| `--hardware-ownership {owned,rented}` | Yes, to persist anything | If unset, the run still completes normally, but the result is **not stored**, and you'll see: `Result not persisted to the result store: --hardware-ownership wasn't set, so provenance can't be determined.` Nothing fails loudly, it fails quietly and tells you why. Watch for that line, it's easy to miss if you're not looking for it. |
| `--hardware-provider` | No | Defaults to `"unknown"`. Only meaningful when rented. |
| `--hardware-rate-usd-per-hour` | No | Defaults to none recorded. |
| `--environment-note` | No | Defaults to `"unknown"`. Free text, put anything useful (`"personal laptop"`, `"RunPod pod, on-demand"`). |
| `--no-result-store` | No | Opts out of the store entirely, no check before the run, nothing persisted after. |

**Friction #1**: these six flags sit at the very end of `throttle golden
--help`'s output, after 50+ other flags, with no section header
separating them from the rest. Nothing marks them as "the result store
options." If you're not specifically looking for them, you can complete
a full run and never realize it wasn't recorded anywhere.

## Friction #2: the actual blocking error tells you nothing about which field is wrong

Running as a genuine newcomer, with no GPU, produced this:

```
Decision-grade preflight: BLOCKED — golden_runtime_unverified_manifest_metadata
```

That's the whole message. No field name, no hint. The real rule, only
discoverable by reading `provenance.py`: on a non-CUDA
`--accelerator-backend`, three fields must be real text, not the
`"unknown"` default: `--gpu`, `--accelerator-runtime-version`, and
`--host-os-version`. (`--cuda-version` and `--driver-version` are
allowed to stay `"unknown"`, they're CUDA-specific and don't apply
off CUDA.) If you're on a CPU-only box and honestly report `"unknown"`
for everything you don't have real hardware info for, exactly like a
first-timer would, you'll hit this and have no way to know why without
reading source.

## What actually worked, end to end

This is the exact invocation that cleared every preflight check and
completed a real six-position run, CPU-only, no GPU, as an operator
named `alex@thirdparty-test`, not the maintainer:

```bash
throttle golden \
  --model test-model \
  --model-revision 0123456789abcdef0123456789abcdef01234567 \
  --url http://<your-endpoint>/v1 \
  --api-key-env YOUR_KEY_ENV \
  --cache-policy disabled \
  --evidence-source live_inference \
  --engine-flags-provenance runtime_verified \
  --server-name <serving-engine-name> \
  --server-version <serving-engine-version> \
  --allow-unknown-cost \
  --gpu "CPU only, no GPU" \
  --gpu-fingerprint "no-gpu-cpu-only" \
  --cuda-version unknown \
  --driver-version unknown \
  --accelerator-backend cpu \
  --accelerator-runtime-version "Python 3.12 / glibc" \
  --host-os-version "Ubuntu 24.04" \
  --image-digest unknown \
  --software-environment-digest "local-cpu-env@sha256:<any 64-hex digest>" \
  --baseline-config max_num_seqs=1 \
  --candidate-config max_num_seqs=8 \
  --operator "alex@thirdparty-test" \
  --hardware-ownership owned
```

Once past preflight, it pauses before each of the six positions
(`B1`, `C1`, `B2`, `C2`, `B3`, `C3`) and waits for you to type e.g.
`B1 verified` after actually applying that position's config on your
server. This part matches the design exactly, and worked without
surprises.

## What a record from a stranger actually looks like

The point of the store is that someone who's never seen this
conversation can read a record and know where it came from. Confirmed
by checking the actual record fields after a real run:

- `provenance.operator` is exactly the string you passed, verbatim,
  no validation against any identity system. It's an honor-system
  field: if you lie, the record lies. The git-commit corroboration
  (if these records are shared via a repo) is the only external check.
- `provenance.hardware_ownership` is `"owned"` or `"rented"`, nothing
  else is accepted, `Provenance.__post_init__` raises immediately on
  anything else.
- If the run isn't decision-eligible (mine came back "inconclusive"
  since a mock backend has no real difference to measure), nothing
  gets stored at all, correctly. The store only ever holds
  decision-eligible results.

## What's still just from testing, not from a real deployment

Everything above was run against a hand-built mock server standing in
for a real inference backend, on a CPU-only box, so the actual
throughput numbers are meaningless (identical canned output regardless
of config, hence "inconclusive"). What's proven is the mechanical
path end to end: preflight, position-by-position confirmation, and
decision-eligibility gating all work correctly for an operator who
isn't the maintainer, on hardware that isn't the maintainer's. What's
not proven here is anything about real GPU behavior, since there
wasn't any.
