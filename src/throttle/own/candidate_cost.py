"""Per-token owned cost: price a candidate that you do not run on your own GPU.

``throttle report`` has two ways to put a dollar figure on the candidate:

* GPU mode (``--gpu-hourly-rate`` and ``--gpus``): a measured ``throttle
  check`` of a self-hosted server gives $/M tokens, plus an always-on GPU
  floor. That lives in :mod:`throttle.own.report`.
* Per-token mode (this module): the candidate is billed per token, either a
  cheaper closed API model or an open model on a hosted endpoint. No GPU is
  run and no floor applies.

Price source, first match wins:

1. ``--candidate-price-in`` and ``--candidate-price-out`` (USD per 1M tokens).
   The operator supplies them, so they are labelled ``ASSUMED``.
2. ``--candidate-model NAME`` found in the bundled ``prices.json`` (via
   :mod:`throttle.own.pricing`), labelled ``LIST PRICE (<as_of>)``. A match
   from an operator ``--prices`` file is ``ASSUMED`` instead.
3. Nothing: per-token mode is not available.

Tokens always come from the candidate's OWN shadow answers, one request at a
time. List price x the API model's token counts is never used: a model with a
cheaper list price can write longer answers (or think longer) and cost more in
real use.

Math, per workload type:

* each shadow answer that reported complete usage is priced with
  :func:`throttle.own.pricing.cost_usd` (cached input tokens at the cached
  price when the entry has one, otherwise at the input price);
* the mean cost per request gets a 95% percentile-bootstrap CI over
  requests (the same resampler the eval uses for its net score);
* owned $/mo = requests per month x that mean, as point, CI low and CI high.

Requests per month is the same projection the GPU mode uses; it has no CI of
its own. A type with fewer than two priced answers has no CI and so no owned
number. If any answer is over a long-context threshold the price page does not
price, the whole type is left unpriced rather than guessed.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .. import statistics as stats
from . import pricing as pricing_mod
from . import records

MODE_GPU = "gpu"
MODE_PER_TOKEN = "per_token"
ASSUMED = "ASSUMED"
LIST_PRICE = "LIST PRICE"
FLAG_SOURCE = "--candidate-price-in / --candidate-price-out"
MIN_PRICED = 2          # a CI needs at least two priced answers


@dataclass(frozen=True)
class CandidatePrice:
    """A per-token price for the candidate and where it came from."""

    price: pricing_mod.Price
    label: str            # "ASSUMED" | "LIST PRICE"
    label_detail: str     # e.g. "LIST PRICE (2026-10-27)" or "ASSUMED (operator price)"
    model: str | None     # the model name the price is for, when known

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "label_detail": self.label_detail,
            "model": self.model,
            "input_per_m": self.price.input_per_m,
            "cached_input_per_m": self.price.cached_input_per_m,
            "output_per_m": self.price.output_per_m,
            "as_of": self.price.as_of if self.label == LIST_PRICE else None,
            "source": self.price.source,
        }


def assumed_price(input_per_m: float, output_per_m: float, *, model: str | None = None) -> CandidatePrice:
    """An operator-supplied price (``--candidate-price-in/--candidate-price-out``)."""

    for name, value in (("--candidate-price-in", input_per_m), ("--candidate-price-out", output_per_m)):
        if value is None or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a non-negative number of USD per 1M tokens")
    if input_per_m == 0 and output_per_m == 0:
        raise ValueError("--candidate-price-in and --candidate-price-out cannot both be 0")
    price = pricing_mod.Price(
        provider="operator", model=model or "candidate", input_per_m=float(input_per_m),
        output_per_m=float(output_per_m), cached_input_per_m=None, as_of="operator-supplied",
        source=FLAG_SOURCE, origin=pricing_mod.ORIGIN_OVERRIDE,
    )
    return CandidatePrice(price=price, label=ASSUMED, label_detail="ASSUMED (operator price)", model=model)


def lookup_price(table: pricing_mod.PriceTable, model: str | None) -> CandidatePrice | None:
    """``model`` in the price table (any provider), or ``None``."""

    if not model:
        return None
    for provider in records.PROVIDERS:
        price = table.lookup(provider, model)
        if price is not None:
            label = ASSUMED if price.origin == pricing_mod.ORIGIN_OVERRIDE else LIST_PRICE
            return CandidatePrice(price=price, label=label, label_detail=price.label, model=model)
    return None


def resolve_price(
    table: pricing_mod.PriceTable,
    *,
    price_in: float | None,
    price_out: float | None,
    model: str | None,
) -> CandidatePrice | None:
    """Explicit flags first (ASSUMED), then a table lookup of ``model``.
    Raises ``ValueError`` when only one of the two price flags is given."""

    if (price_in is None) != (price_out is None):
        raise ValueError("give both --candidate-price-in and --candidate-price-out (USD per 1M tokens)")
    if price_in is not None and price_out is not None:
        return assumed_price(price_in, price_out, model=model)
    return lookup_price(table, model)


def model_matches(price: CandidatePrice, measured_model: str | None) -> bool | None:
    """Whether the priced model is the one that wrote the shadow answers.
    ``None`` when either name is unknown (nothing to compare)."""

    if not measured_model or not price.model:
        return None
    if price.label == LIST_PRICE or price.price.origin == pricing_mod.ORIGIN_BUNDLED:
        return pricing_mod.prefix_matches(price.price.model, measured_model)
    return price.model.strip().lower() == measured_model.strip().lower()


# ---------------------------------------------------------------------------
# Measured candidate usage
# ---------------------------------------------------------------------------


def load_samples(root: Path, candidate: str, eval_doc: Mapping[str, Any] | None,
                 *, classify=None) -> dict[str, list[records.Usage]]:
    """Per workload type, the usage the candidate reported on each shadow
    answer the eval used. When the eval saved its eval sets, exactly those
    requests are used; otherwise every shadow answer, typed as eval would."""

    from . import evaluate as ev

    only_ids: dict[str, list[str]] | None = {}
    types = (eval_doc or {}).get("types") or {}
    for kind, type_eval in types.items():
        path = (type_eval or {}).get("evalset_path")
        try:
            _, owner, ids = ev.load_evalset(Path(path)) if path else ("", "", [])
        except (OSError, ValueError, KeyError, TypeError):
            owner, ids = "", []
        if owner != candidate or not ids:
            only_ids = None          # one type without its saved set: use every shadow answer
            break
        only_ids[kind] = list(ids)   # type: ignore[index]
    sets = ev.build_eval_sets(root, candidate, classify=classify, only_ids=only_ids or None)
    return {kind: [p.shadow.usage for p in pairs if p.shadow.ok] for kind, pairs in sets.items()}


@dataclass(frozen=True)
class TypeCost:
    owned: dict[str, float] | None      # {"point", "low", "high"} $/mo, or None
    note: str                           # what the number is, or why there is none
    per_request: dict[str, Any] | None  # {"mean", "low", "high", "n", "excluded"} $ per request


def _seed(kind: str, values: Sequence[float]) -> int:
    digest = hashlib.sha256((kind + "|" + ",".join(f"{v:.12g}" for v in values)).encode()).hexdigest()
    return int(digest[:8], 16)


def per_request_cost(usages: Iterable[records.Usage], price: CandidatePrice, *, kind: str = "") -> tuple[
        dict[str, Any] | None, str | None]:
    """Mean $ per request with a 95% bootstrap CI, from the candidate's own
    usage. Returns (stats, why not)."""

    costs: list[float] = []
    excluded = 0
    unpriced = 0
    for usage in usages:
        if not usage.complete:
            excluded += 1
            continue
        cost = pricing_mod.cost_usd(price.price, usage)
        if cost is None:
            unpriced += 1
            continue
        costs.append(cost)
    if unpriced:
        return None, (f"{unpriced} shadow answer(s) cannot be priced at {price.label_detail} "
                      "(over a long-context threshold with no published price, or bad cached-token counts)")
    if len(costs) < MIN_PRICED:
        return None, (f"only {len(costs)} shadow answer(s) reported token usage; "
                      f"need at least {MIN_PRICED} for a 95% CI")
    ci = stats.paired_bootstrap_interval_95(costs, seed=_seed(kind, costs))
    return {"mean": float(ci["estimate"]), "low": float(ci["low"]), "high": float(ci["high"]),
            "n": len(costs), "excluded": excluded}, None


def monthly_cost(usages: Sequence[records.Usage] | None, price: CandidatePrice, requests_per_month: float | None,
                 *, kind: str = "") -> TypeCost:
    """Owned $/mo for one workload type in per-token mode."""

    if requests_per_month is None:
        return TypeCost(None, "not projected: capture window under 1 day", None)
    if not usages:
        return TypeCost(None, "not measured: no shadow answers with token usage for this type", None)
    per_request, why = per_request_cost(usages, price, kind=kind)
    if per_request is None:
        return TypeCost(None, f"not measured: {why}", None)
    owned = {k: requests_per_month * per_request[src]
             for k, src in (("point", "mean"), ("low", "low"), ("high", "high"))}
    note = (f"candidate's own tokens on {per_request['n']} shadow answers x {price.label_detail} "
            f"${price.price.input_per_m:g}/M in, ${price.price.output_per_m:g}/M out")
    return TypeCost(owned, note, per_request)
