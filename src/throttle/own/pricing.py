"""Published API list prices, so captured usage can be turned into dollars.

The bundled table (``prices.json`` next to this module) holds one entry per
model id prefix with the USD price per 1M input, cached-input and output
tokens, the date it was checked and the official page it came from. Every
dollar figure built from it is labelled ``LIST PRICE (<as_of>)``; entries
from an operator override file (``--prices FILE``) are labelled ``ASSUMED``.

Out of scope, on purpose:

* Batch, flex, fast and priority tiers. Prices are the standard tier only.
* Data-residency and regional uplifts.
* Cache *writes*. The shared usage record has no separate cache-write count,
  so cache-write tokens are billed at the base input price. That understates
  API spend slightly for models that charge extra for writes, which is the
  conservative direction for a savings claim.

Long-context pricing: an entry can name an input-token threshold above which a
different price applies. If the page publishes no price above the threshold,
requests over it are left unpriced rather than guessed.

Nothing here estimates tokens: incomplete usage gives ``None``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from .records import PROVIDERS, Usage

BUNDLED = "prices.json"
ORIGIN_BUNDLED = "bundled"
ORIGIN_OVERRIDE = "override"

# What may follow a price prefix in a real model id: a dated snapshot
# (gpt-4o-2024-08-06, claude-opus-4-5-20251101, Vertex's @20251101) and
# nothing else. "gpt-4o-mini" therefore never matches "gpt-4o".
_SNAPSHOT_SUFFIX = re.compile(r"^[-@](\d{4}-\d{2}-\d{2}|\d{8}|\d{4})$")


@dataclass(frozen=True)
class LongContext:
    above_input_tokens: int
    input_per_m: float | None = None
    cached_input_per_m: float | None = None
    output_per_m: float | None = None

    @property
    def priced(self) -> bool:
        return self.input_per_m is not None and self.output_per_m is not None


@dataclass(frozen=True)
class Price:
    provider: str
    model: str                       # model id prefix
    input_per_m: float
    output_per_m: float
    cached_input_per_m: float | None
    as_of: str
    source: str
    long_context: LongContext | None = None
    note: str | None = None
    origin: str = ORIGIN_BUNDLED

    @property
    def label(self) -> str:
        if self.origin == ORIGIN_OVERRIDE:
            return f"ASSUMED (operator price, {self.as_of})"
        return f"LIST PRICE ({self.as_of})"

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "input_per_m": self.input_per_m,
            "cached_input_per_m": self.cached_input_per_m,
            "output_per_m": self.output_per_m,
            "as_of": self.as_of,
            "source": self.source,
            "label": self.label,
        }
        if self.long_context is not None:
            data["long_context"] = {
                "above_input_tokens": self.long_context.above_input_tokens,
                "input_per_m": self.long_context.input_per_m,
                "cached_input_per_m": self.long_context.cached_input_per_m,
                "output_per_m": self.long_context.output_per_m,
            }
        if self.note:
            data["note"] = self.note
        return data


def prefix_matches(prefix: str, model: str) -> bool:
    prefix, model = prefix.lower(), model.strip().lower()
    if model == prefix:
        return True
    return model.startswith(prefix) and bool(_SNAPSHOT_SUFFIX.match(model[len(prefix):]))


class PriceTable:
    def __init__(self, prices: list[Price]) -> None:
        self._prices: dict[tuple[str, str], Price] = {}
        for price in prices:
            self._prices[(price.provider, price.model.lower())] = price

    def __len__(self) -> int:
        return len(self._prices)

    def __iter__(self):
        return iter(self._prices.values())

    def lookup(self, provider: str, model: str | None) -> Price | None:
        """The entry with the longest prefix that matches ``model`` exactly
        or as a dated snapshot of it."""
        if not model:
            return None
        best: Price | None = None
        for (entry_provider, prefix), price in self._prices.items():
            if entry_provider != provider or not prefix_matches(prefix, model):
                continue
            if best is None or len(prefix) > len(best.model):
                best = price
        return best

    @property
    def labels(self) -> list[str]:
        return sorted({p.label for p in self})


def cost_usd(price: Price | None, usage: Usage) -> float | None:
    """API cost of one request at ``price``, or ``None`` when it cannot be
    computed from what the upstream reported (never estimated)."""
    if price is None or not usage.complete:
        return None
    input_tokens = usage.input_tokens or 0
    output_tokens = usage.output_tokens or 0
    cached = usage.cached_input_tokens or 0
    if cached > input_tokens:
        return None
    input_rate, cached_rate, output_rate = price.input_per_m, price.cached_input_per_m, price.output_per_m
    long = price.long_context
    if long is not None and input_tokens > long.above_input_tokens:
        if not long.priced:
            return None
        input_rate = long.input_per_m  # type: ignore[assignment]
        output_rate = long.output_per_m  # type: ignore[assignment]
        cached_rate = long.cached_input_per_m
    if cached_rate is None:
        cached_rate = input_rate
    total = (input_tokens - cached) * input_rate + cached * cached_rate + output_tokens * output_rate
    return total / 1_000_000


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _number(entry: Mapping[str, Any], key: str, *, required: bool, where: str) -> float | None:
    value = entry.get(key)
    if value is None:
        if required:
            raise ValueError(f"{where}: missing {key}")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{where}: {key} must be a non-negative number")
    return float(value)


def _parse_entry(entry: Any, *, origin: str, where: str, default_source: str) -> Price:
    if not isinstance(entry, Mapping):
        raise ValueError(f"{where}: each price must be an object")
    provider = entry.get("provider")
    if provider not in PROVIDERS:
        raise ValueError(f"{where}: provider must be one of {', '.join(PROVIDERS)}")
    model = entry.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{where}: missing model")
    long = None
    raw_long = entry.get("long_context")
    if raw_long is not None:
        if not isinstance(raw_long, Mapping) or not isinstance(raw_long.get("above_input_tokens"), int):
            raise ValueError(f"{where}: long_context needs an integer above_input_tokens")
        long = LongContext(
            above_input_tokens=raw_long["above_input_tokens"],
            input_per_m=_number(raw_long, "input_per_m", required=False, where=where),
            cached_input_per_m=_number(raw_long, "cached_input_per_m", required=False, where=where),
            output_per_m=_number(raw_long, "output_per_m", required=False, where=where),
        )
    as_of = entry.get("as_of")
    if origin == ORIGIN_BUNDLED and not as_of:
        raise ValueError(f"{where}: missing as_of")
    return Price(
        provider=provider,
        model=model.strip(),
        input_per_m=_number(entry, "input_per_m", required=True, where=where),  # type: ignore[arg-type]
        output_per_m=_number(entry, "output_per_m", required=True, where=where),  # type: ignore[arg-type]
        cached_input_per_m=_number(entry, "cached_input_per_m", required=False, where=where),
        as_of=str(as_of or "undated"),
        source=str(entry.get("source") or default_source),
        long_context=long,
        note=entry.get("note") if isinstance(entry.get("note"), str) else None,
        origin=origin,
    )


def _parse_document(data: Any, *, origin: str, name: str) -> list[Price]:
    entries = data.get("prices") if isinstance(data, Mapping) else data
    if not isinstance(entries, list):
        raise ValueError(f"{name}: expected a list of prices or an object with a 'prices' list")
    return [
        _parse_entry(entry, origin=origin, where=f"{name} entry {i + 1}", default_source=f"operator file {name}")
        for i, entry in enumerate(entries)
    ]


def bundled_prices() -> list[Price]:
    text = resources.files("throttle.own").joinpath(BUNDLED).read_text(encoding="utf-8")
    return _parse_document(json.loads(text), origin=ORIGIN_BUNDLED, name=BUNDLED)


def load_prices(override_path: str | Path | None = None) -> PriceTable:
    """The bundled table, with entries from ``override_path`` replacing those
    with the same provider and model prefix and adding the rest."""
    prices = bundled_prices()
    if override_path is not None:
        path = Path(override_path).expanduser()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"cannot read price file {path}: {exc.strerror or exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"price file {path} is not valid JSON: {exc}") from exc
        prices.extend(_parse_document(data, origin=ORIGIN_OVERRIDE, name=str(path)))
    return PriceTable(prices)
