"""Canonical model-token accounting, pricing, and budget errors.

Provider SDKs use different field names, but persisted run records deliberately
use one small contract.  Prices live in :mod:`config`; this module only applies
the selected table and never embeds vendor rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from .config import settings


USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)

_PERSISTED_FIELDS = {
    "input_tokens": "inputTokens",
    "output_tokens": "outputTokens",
    "cache_creation_input_tokens": "cacheCreationInputTokens",
    "cache_read_input_tokens": "cacheReadInputTokens",
}


def empty_usage() -> dict[str, int]:
    return {field: 0 for field in USAGE_FIELDS}


def _value(source: Any, *names: str) -> int:
    for name in names:
        if isinstance(source, Mapping) and name in source:
            value = source.get(name)
        else:
            value = getattr(source, name, None)
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return 0
    return 0


def normalize_usage(source: Any, provider: str = "") -> dict[str, int]:
    """Return the four canonical counters from an SDK object or JSON block."""
    if source is None:
        return empty_usage()
    # Anthropic/OpenAI snake_case plus Gemini's usageMetadata names. OpenAI
    # cached tokens are nested under prompt_tokens_details.
    prompt_details = (
        source.get("prompt_tokens_details", {}) if isinstance(source, Mapping) else {}
    )
    result = {
        "input_tokens": _value(source, "input_tokens", "prompt_tokens", "promptTokenCount"),
        "output_tokens": _value(
            source, "output_tokens", "completion_tokens", "candidatesTokenCount"
        ),
        "cache_creation_input_tokens": _value(
            source, "cache_creation_input_tokens", "cacheCreationInputTokens"
        ),
        "cache_read_input_tokens": _value(
            source, "cache_read_input_tokens", "cacheReadInputTokens"
        ),
    }
    if not result["cache_read_input_tokens"]:
        result["cache_read_input_tokens"] = _value(prompt_details, "cached_tokens")
    return result


def add_usage(total: dict[str, int], increment: Mapping[str, Any]) -> dict[str, int]:
    for field in USAGE_FIELDS:
        total[field] = int(total.get(field, 0)) + _value(increment, field)
    return total


def persisted_usage(usage: Mapping[str, Any]) -> dict[str, int]:
    return {_PERSISTED_FIELDS[field]: _value(usage, field) for field in USAGE_FIELDS}


def internal_usage(usage: Mapping[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    return {
        field: _value(usage, field, _PERSISTED_FIELDS[field]) for field in USAGE_FIELDS
    }


def price_for_model(model: str, table: Mapping[str, Mapping[str, float]] | None = None):
    table = table if table is not None else settings.llm_price_table
    if model in table:
        return table[model]
    # Version suffixes are common. Config may intentionally use a trailing '*'
    # for a stable family price without requiring a deploy for each dated id.
    matches = [
        (key[:-1], value) for key, value in table.items()
        if key.endswith("*") and model.startswith(key[:-1])
    ]
    if not matches:
        return None
    return max(matches, key=lambda pair: len(pair[0]))[1]


def cost_usd(
    usage: Mapping[str, Any],
    model: str,
    table: Mapping[str, Mapping[str, float]] | None = None,
) -> float:
    price = price_for_model(model, table)
    if price is None:
        return 0.0
    tokens = internal_usage(usage)
    total = (
        tokens["input_tokens"] * float(price.get("input", 0))
        + tokens["output_tokens"] * float(price.get("output", 0))
        + tokens["cache_creation_input_tokens"] * float(price.get("cacheWrite", 0))
        + tokens["cache_read_input_tokens"] * float(price.get("cacheRead", 0))
    ) / 1_000_000
    return round(total, 8)


class BudgetExceededError(RuntimeError):
    error_code = "budget_exceeded"

    def __init__(self, scope: str, current_spend: float, cap: float, *, refused: str):
        self.scope = scope
        self.current_spend = round(float(current_spend), 8)
        self.cap = round(float(cap), 8)
        self.refused = refused
        super().__init__(
            f"{scope} budget exceeded: ${self.current_spend:.6f} spent of "
            f"${self.cap:.6f}; refused {refused}"
        )

    def to_dict(self) -> dict:
        return {
            "code": self.error_code,
            "scope": self.scope,
            "currentSpendUSD": self.current_spend,
            "capUSD": self.cap,
            "refused": self.refused,
        }


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def window_start(window: str, *, now: datetime | None = None) -> datetime:
    if not window.endswith("d") or not window[:-1].isdigit() or int(window[:-1]) < 1:
        raise ValueError("window must be a positive number of days, e.g. 7d")
    return (now or datetime.now(timezone.utc)) - timedelta(days=int(window[:-1]))
