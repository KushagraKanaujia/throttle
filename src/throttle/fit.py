"""``throttle fit``: the fewest GPUs your model needs, before you rent any.

Give a model, the context length your users need and how many of them talk to
it at once. Throttle works out, for each GPU type, the fewest cards that hold
the weights plus the KV cache for that many full-length chats, how many such
chats that many cards can hold, and (if you give prices) what each option
costs a month.

It is memory math only: offline, no GPU, no keys, nothing sent anywhere.
Every number is an ESTIMATE. It does not predict speed: more cards can be
slower, not faster, because they have to talk to each other. So it ends with
the command that measures the shortlist on your real server.

Pass ``--current GPU=COUNT`` (for example ``--current h100=4``) and it tells
you if fewer cards look like enough, which is where the money usually is.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import style as style_module

EXIT_OK = 0
EXIT_DOES_NOT_FIT = 1
EXIT_USAGE = 2

HOURS_PER_MONTH = 730
GIB = 1024 ** 3
# What vLLM lets itself use by default (--gpu-memory-utilization 0.9), and a
# flat allowance per card for the CUDA context, activations and graphs.
DEFAULT_MEMORY_FRACTION = 0.90
DEFAULT_OVERHEAD_GB = 1.5
MAX_GPUS = 8

FIT_DESCRIPTION = __doc__.split("\n\n", 1)[1] if __doc__ else ""
FIT_EPILOG = """\
examples:
  # which GPUs, and how many, for a 70B model, 32K context, 16 users at once
  throttle fit --model llama-3.3-70b --quant fp8 --context 32k --users 16

  # with your prices, and what you run today
  throttle fit --model qwen2.5-72b --quant fp8 --context 16k --users 32 \\
      --gpu h100 --gpu a100-80 --price h100=2.49 --price a100-80=1.79 --current h100=4

  # any model: point at its config.json and give the parameter count
  throttle fit --config ./config.json --params 32.8 --quant awq --context 8k --users 8

  # llama.cpp / GGUF on your own cards (any card count works there)
  throttle fit --model qwen2.5-32b --quant q4_k_m --engine llamacpp --gpu rtx3090 --gpu 1060=6

Built-in models: see --list-models. Built-in GPUs: see --list-gpus.
"""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_b: float          # total parameters, billions (all experts for MoE)
    layers: int
    kv_heads: int
    head_dim: int
    attention_heads: int
    hf_id: str = ""           # name to serve it by


# Published architecture numbers (each model's config.json).
MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec("llama-3.1-8b", 8.03, 32, 8, 128, 32, "meta-llama/Llama-3.1-8B-Instruct"),
        ModelSpec("llama-3.1-70b", 70.6, 80, 8, 128, 64, "meta-llama/Llama-3.1-70B-Instruct"),
        ModelSpec("llama-3.3-70b", 70.6, 80, 8, 128, 64, "meta-llama/Llama-3.3-70B-Instruct"),
        ModelSpec("qwen2.5-7b", 7.62, 28, 4, 128, 28, "Qwen/Qwen2.5-7B-Instruct"),
        ModelSpec("qwen2.5-14b", 14.8, 48, 8, 128, 40, "Qwen/Qwen2.5-14B-Instruct"),
        ModelSpec("qwen2.5-32b", 32.8, 64, 8, 128, 40, "Qwen/Qwen2.5-32B-Instruct"),
        ModelSpec("qwen2.5-72b", 72.7, 80, 8, 128, 64, "Qwen/Qwen2.5-72B-Instruct"),
        ModelSpec("qwen3-32b", 32.8, 64, 8, 128, 64, "Qwen/Qwen3-32B"),
        ModelSpec("qwen3-30b-a3b", 30.5, 48, 4, 128, 32, "Qwen/Qwen3-30B-A3B"),
        ModelSpec("mistral-7b", 7.25, 32, 8, 128, 32, "mistralai/Mistral-7B-Instruct-v0.3"),
        ModelSpec("mixtral-8x7b", 46.7, 32, 8, 128, 32, "mistralai/Mixtral-8x7B-Instruct-v0.1"),
    )
}

# Memory on the card, GB as sold.
GPUS: dict[str, float] = {
    "h200": 141, "h100": 80, "a100-80": 80, "a100-40": 40, "mi300x": 192,
    "l40s": 48, "rtx6000-ada": 48, "a10g": 24, "l4": 24,
    "rtx5090": 32, "rtx4090": 24, "rtx3090": 24,
}
DEFAULT_GPUS = ("h100", "a100-80", "l40s", "l4")

# Approximate bytes per weight, including quantization scales.
QUANT_BYTES: dict[str, float] = {
    "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0,
    "awq": 0.56, "gptq": 0.56, "int4": 0.56,
    "q8_0": 1.0625, "q6_k": 0.82, "q5_k_m": 0.71, "q4_k_m": 0.60,
}
KV_BYTES: dict[str, float] = {"auto": 2.0, "fp16": 2.0, "fp8": 1.0, "q8_0": 1.0625, "q4_0": 0.5625}


@dataclass
class Option:
    gpu: str
    vram_gb: float
    gpus: int | None              # fewest that fit; None if not within MAX_GPUS
    max_users: int | None         # full-context chats that many cards can hold
    monthly_cost: float | None    # only when the user gave a price
    hourly_price: float | None


def parse_size(text: str) -> int:
    """'32k' -> 32768, '128K' -> 131072, '8192' -> 8192."""
    raw = text.strip().lower()
    mult = 1
    if raw.endswith("k"):
        raw, mult = raw[:-1], 1024
    elif raw.endswith("m"):
        raw, mult = raw[:-1], 1024 * 1024
    try:
        value = int(float(raw) * mult)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a size: {text!r} (try 8192, 32k)") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be above 0")
    return value


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be above 0")
    return value


def add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    which = parser.add_argument_group("the model")
    which.add_argument("--model", help="a built-in model name (see --list-models)")
    which.add_argument("--config", type=Path,
                       help="a Hugging Face config.json to read layers and heads from")
    which.add_argument("--params", type=float,
                       help="total parameters in billions (needed with --config, "
                       "or to override a built-in)")
    which.add_argument("--quant", default="bf16", choices=sorted(QUANT_BYTES),
                       help="weight format (default bf16)")
    which.add_argument("--kv-dtype", default="auto", choices=sorted(KV_BYTES),
                       help="KV cache format (default auto = 16 bit)")

    load = parser.add_argument_group("your traffic")
    load.add_argument("--context", type=parse_size, default=8192,
                      help="tokens each chat may grow to, prompt plus answer (default 8k)")
    load.add_argument("--users", type=positive_int, default=1,
                      help="chats running at the same time at that length (default 1)")

    hw = parser.add_argument_group("hardware and prices")
    hw.add_argument("--gpu", action="append", metavar="NAME[=GB]",
                    help="GPU to consider; repeat for more. A custom card: name=GB "
                    f"(default: {', '.join(DEFAULT_GPUS)})")
    hw.add_argument("--price", action="append", metavar="NAME=$/HOUR",
                    help="your hourly price for one card, e.g. h100=2.49; repeat per GPU")
    hw.add_argument("--current", metavar="NAME=COUNT",
                    help="what you run today, e.g. h100=4, to see if fewer cards look enough")
    hw.add_argument("--engine", default="vllm", choices=("vllm", "sglang", "llamacpp"),
                    help="vllm/sglang split across 1, 2, 4 or 8 cards that divide the "
                    "attention heads; llamacpp can use any count (default vllm)")
    hw.add_argument("--memory-fraction", type=float, default=DEFAULT_MEMORY_FRACTION,
                    help=f"share of each card the engine may use (default {DEFAULT_MEMORY_FRACTION})")
    hw.add_argument("--overhead-gb", type=float, default=DEFAULT_OVERHEAD_GB,
                    help=f"GB per card kept for the runtime itself (default {DEFAULT_OVERHEAD_GB})")

    out = parser.add_argument_group("output")
    out.add_argument("--json", action="store_true", help="print JSON instead of a table")
    out.add_argument("--list-models", action="store_true", help="list built-in models and exit")
    out.add_argument("--list-gpus", action="store_true", help="list built-in GPUs and exit")


class UsageError(Exception):
    pass


def resolve_model(args: argparse.Namespace) -> ModelSpec:
    if args.model and args.config:
        raise UsageError("give --model or --config, not both")
    if args.config:
        try:
            cfg = json.loads(args.config.read_text())
        except (OSError, ValueError) as exc:
            raise UsageError(f"could not read {args.config}: {exc}") from exc
        cfg = cfg.get("text_config", cfg)
        try:
            layers = int(cfg["num_hidden_layers"])
            heads = int(cfg["num_attention_heads"])
            kv_heads = int(cfg.get("num_key_value_heads") or heads)
            head_dim = int(cfg.get("head_dim") or int(cfg["hidden_size"]) // heads)
        except (KeyError, TypeError, ValueError) as exc:
            raise UsageError(f"{args.config} is missing {exc}; is it a transformer config.json?") from exc
        if not args.params:
            raise UsageError("--config needs --params (total parameters in billions)")
        return ModelSpec(args.config.name, args.params, layers, kv_heads, head_dim, heads)
    if not args.model:
        raise UsageError("give --model (see --list-models) or --config path/to/config.json")
    key = args.model.strip().lower()
    if key not in MODELS:
        raise UsageError(f"unknown model {args.model!r}. Built in: {', '.join(MODELS)}. "
                         "For anything else use --config and --params.")
    spec = MODELS[key]
    if args.params:
        spec = ModelSpec(spec.name, args.params, spec.layers, spec.kv_heads, spec.head_dim,
                         spec.attention_heads, spec.hf_id)
    return spec


def parse_pairs(values: Sequence[str] | None, flag: str) -> dict[str, float]:
    pairs: dict[str, float] = {}
    for item in values or ():
        name, sep, number = item.partition("=")
        try:
            value = float(number)
        except ValueError:
            value = -1.0
        if not sep or not name.strip() or value <= 0:
            raise UsageError(f"{flag} wants NAME=NUMBER above 0, got {item!r}")
        pairs[name.strip().lower()] = value
    return pairs


def resolve_gpus(values: Sequence[str] | None) -> dict[str, float]:
    gpus: dict[str, float] = {}
    for item in values or DEFAULT_GPUS:
        name, sep, gb = item.partition("=")
        key = name.strip().lower()
        if sep:
            gpus.update(parse_pairs([f"{key}={gb}"], "--gpu"))
        elif key in GPUS:
            gpus[key] = GPUS[key]
        else:
            raise UsageError(f"unknown GPU {name!r}. Built in: {', '.join(GPUS)}. "
                             "For another card use name=GB, e.g. --gpu 1060=6")
    return gpus


def gpu_counts(engine: str, spec: ModelSpec) -> list[int]:
    if engine == "llamacpp":
        return list(range(1, MAX_GPUS + 1))
    return [n for n in (1, 2, 4, 8)
            if spec.attention_heads % n == 0 and (spec.kv_heads % n == 0 or n % spec.kv_heads == 0)]


def weights_gb(spec: ModelSpec, quant: str) -> float:
    return spec.params_b * 1e9 * QUANT_BYTES[quant] / GIB


def kv_gb_per_token(spec: ModelSpec, kv_dtype: str) -> float:
    # keys and values, every layer, every KV head
    return 2 * spec.layers * spec.kv_heads * spec.head_dim * KV_BYTES[kv_dtype] / GIB


def usable_gb(vram_gb: float, fraction: float, overhead: float) -> float:
    # cards are sold in GB (10^9); work in GiB like the engines report
    return vram_gb * 1e9 / GIB * fraction - overhead


def max_users(n: int, per_card: float, weights: float, kv_chat: float,
              kv_heads: int = 0) -> int:
    room = n * per_card - weights
    if room <= 0 or n * per_card <= 0:
        return 0
    # split across more cards than KV heads, each head is copied to several cards
    copies = max(1.0, n / kv_heads) if kv_heads else 1.0
    return int(math.floor(room / (kv_chat * copies)))


def plan(spec: ModelSpec, args: argparse.Namespace, gpus: Mapping[str, float],
         prices: Mapping[str, float]) -> list[Option]:
    weights = weights_gb(spec, args.quant)
    kv_chat = kv_gb_per_token(spec, args.kv_dtype) * args.context
    counts = gpu_counts(args.engine, spec)
    options = []
    for name, vram in gpus.items():
        per_card = usable_gb(vram, args.memory_fraction, args.overhead_gb)
        heads = 0 if args.engine == "llamacpp" else spec.kv_heads
        fewest = next((n for n in counts
                       if max_users(n, per_card, weights, kv_chat, heads) >= args.users), None)
        price = prices.get(name)
        options.append(Option(
            gpu=name, vram_gb=vram, gpus=fewest,
            max_users=max_users(fewest, per_card, weights, kv_chat, heads) if fewest else None,
            monthly_cost=round(fewest * price * HOURS_PER_MONTH, 2) if fewest and price else None,
            hourly_price=price,
        ))
    return options


def pick(options: Sequence[Option]) -> Option | None:
    fits = [o for o in options if o.gpus]
    if not fits:
        return None
    priced = [o for o in fits if o.monthly_cost is not None]
    if priced:
        return min(priced, key=lambda o: o.monthly_cost)  # type: ignore[arg-type,return-value]
    return min(fits, key=lambda o: (o.gpus, o.vram_gb))


def current_finding(current: tuple[str, int] | None, options: Sequence[Option]) -> dict[str, Any] | None:
    if not current:
        return None
    name, count = current
    match = next((o for o in options if o.gpu == name), None)
    if match is None:
        return {"gpu": name, "count": count, "status": "not_considered"}
    if not match.gpus:
        return {"gpu": name, "count": count, "status": "does_not_fit_estimate"}
    spare = count - match.gpus
    finding: dict[str, Any] = {"gpu": name, "count": count, "fewest_estimate": match.gpus,
                               "spare_cards": max(spare, 0),
                               "status": "maybe_fewer" if spare > 0 else "no_fewer"}
    if spare > 0 and match.hourly_price:
        finding["monthly_saving_if_confirmed"] = round(spare * match.hourly_price * HOURS_PER_MONTH, 2)
    return finding


def measure_command(spec: ModelSpec, best: Option, current: Mapping[str, Any] | None,
                    engine: str) -> str:
    counts = {best.gpus or 1}
    if current and current.get("gpu") == best.gpu and current.get("count"):
        counts.add(int(current["count"]))
    model = shlex.quote(spec.hf_id or "YOUR/MODEL")
    rate = f" --gpu-hourly-rate {best.hourly_price}" if best.hourly_price else ""
    if engine == "llamacpp":
        return (f"throttle check --url http://localhost:8000 --model {model} "
                f"--gpus {best.gpus}{rate}   # once per card layout you try")
    launch = ("vllm serve {model} --tensor-parallel-size {tp}" if engine == "vllm"
              else "python -m sglang.launch_server --model-path {model} --tp {tp}")
    return (f"throttle autotune --url http://localhost:8000 --model {model}{rate} \\\n"
            f"      --launch \"{launch}\" \\\n"
            f"      --space tp={','.join(str(c) for c in sorted(counts))}")


def fmt_money(value: float | None) -> str:
    return "-" if value is None else f"${value:,.0f}"


def handle_fit(args: argparse.Namespace) -> int:
    if args.list_models:
        for spec in MODELS.values():
            print(f"{spec.name:16} {spec.params_b:>6.1f}B  {spec.layers} layers, "
                  f"{spec.kv_heads} KV heads x {spec.head_dim}")
        return EXIT_OK
    if args.list_gpus:
        for name, gb in GPUS.items():
            print(f"{name:12} {gb:>4.0f} GB")
        return EXIT_OK
    try:
        if not 0 < args.memory_fraction <= 1:
            raise UsageError("--memory-fraction must be above 0 and at most 1")
        if args.overhead_gb < 0:
            raise UsageError("--overhead-gb cannot be negative")
        spec = resolve_model(args)
        gpus = resolve_gpus(args.gpu)
        prices = parse_pairs(args.price, "--price")
        unknown = sorted(set(prices) - set(gpus))
        if unknown:
            raise UsageError(f"--price for {', '.join(unknown)} but that GPU is not in the list; "
                             "add it with --gpu")
        current = None
        if args.current:
            cur = parse_pairs([args.current], "--current")
            (cur_name, cur_count), = cur.items()
            if cur_count != int(cur_count):
                raise UsageError("--current wants a whole card count, e.g. h100=4")
            current = (cur_name, int(cur_count))
            if cur_name not in gpus:
                gpus = {**gpus, **resolve_gpus([cur_name])}
    except UsageError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    options = plan(spec, args, gpus, prices)
    best = pick(options)
    finding = current_finding(current, options)
    weights = weights_gb(spec, args.quant)
    kv_chat = kv_gb_per_token(spec, args.kv_dtype) * args.context
    command = measure_command(spec, best, finding, args.engine) if best else None

    if args.json:
        print(json.dumps({
            "estimate": True,
            "model": asdict(spec), "quant": args.quant, "kv_dtype": args.kv_dtype,
            "context": args.context, "users": args.users, "engine": args.engine,
            "weights_gib": round(weights, 2), "kv_gib_per_chat": round(kv_chat, 3),
            "options": [asdict(o) for o in options],
            "pick": best.gpu if best else None, "current": finding,
            "measure_command": command,
        }, indent=2))
        return EXIT_OK if best else EXIT_DOES_NOT_FIT

    st = style_module.Style()
    print(st.rule("throttle fit  (ESTIMATE: memory math, not measured)"))
    print(st.dim("Assumes every chat at the full --context at once (the safe side)."))
    print(f"{spec.name}, {args.quant} weights, {args.context:,} token context, "
          f"{args.users} chat{'s' if args.users != 1 else ''} at once, {args.engine}")
    print(st.dim(f"weights about {weights:,.1f} GiB, KV cache about {kv_chat:,.2f} GiB per full chat"))
    print()
    header = f"  {'GPU':12} {'memory':>7} {'fewest cards':>13} {'chats they hold':>16} {'per month':>10}"
    print(st.bold(header))
    for o in options:
        cards = str(o.gpus) if o.gpus else f">{MAX_GPUS}" if args.engine == "llamacpp" else "no fit"
        users = str(o.max_users) if o.max_users is not None else "-"
        line = (f"  {o.gpu:12} {o.vram_gb:>5.0f}GB {cards:>13} {users:>16} "
                f"{fmt_money(o.monthly_cost):>10}")
        print(st.good(line) if best and o is best else line)
    print()

    if not best:
        print(st.bad(f"Nothing here fits on up to {MAX_GPUS} cards."))
        print("Try a smaller weight format (--quant fp8 or awq), a fp8 KV cache "
              "(--kv-dtype fp8), a shorter --context, fewer --users, or a bigger GPU.")
        return EXIT_DOES_NOT_FIT

    why = "cheapest at your prices" if best.monthly_cost is not None else "fewest cards"
    print(f"Best looking: {st.bold(f'{best.gpus} x {best.gpu}')} ({why}).")
    if best.monthly_cost is None:
        print(st.dim("Add --price NAME=$/HOUR for the monthly cost of each option."))
    if finding:
        status = finding["status"]
        if status == "maybe_fewer":
            line = (f"You run {finding['count']} x {finding['gpu']}; the estimate says "
                    f"{finding['fewest_estimate']} may be enough.")
            if "monthly_saving_if_confirmed" in finding:
                line += (f" If a measurement agrees, that is about "
                         f"{fmt_money(finding['monthly_saving_if_confirmed'])} a month.")
            print(st.warn(line))
        elif status == "no_fewer":
            print(f"You run {finding['count']} x {finding['gpu']}; the estimate does not "
                  "find room to drop a card at this load.")
        elif status == "does_not_fit_estimate":
            print(st.warn(f"The estimate says {finding['gpu']} cannot hold this load on up to "
                          f"{MAX_GPUS} cards, yet you run it: your real chats are probably "
                          "shorter than --context, or the numbers above need your config.json."))
    print()
    print("This does not tell you speed. Extra cards add traffic between them and")
    print("can be slower. Measure on your server before you change anything:")
    print(st.accent(f"  {command}"))
    return EXIT_OK
