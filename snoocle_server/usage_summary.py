"""Compact usage rollups and admission-time spend enforcement."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from .config import settings
from .reconcile.agent_config import AgentConfig
from .usage import BudgetExceededError, internal_usage, parse_timestamp, window_start

log = logging.getLogger(__name__)


def effective_caps(cfg: AgentConfig | None = None) -> dict[str, float]:
    return {
        "run": (
            cfg.run_cost_cap_usd
            if cfg is not None and cfg.run_cost_cap_usd is not None
            else settings.run_cost_cap_usd
        ),
        "batch": (
            cfg.batch_cost_cap_usd
            if cfg is not None and cfg.batch_cost_cap_usd is not None
            else settings.batch_cost_cap_usd
        ),
        "daily": (
            cfg.daily_cost_cap_usd
            if cfg is not None and cfg.daily_cost_cap_usd is not None
            else settings.daily_cost_cap_usd
        ),
    }


def _reliable_cost(run: dict) -> float:
    if not run.get("usageReliable", False):
        return 0.0
    try:
        return max(0.0, float(run.get("costUSD") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def current_spend(runs: list[dict], *, batch_id: str | None = None,
                  now: datetime | None = None) -> dict[str, float]:
    now = now or datetime.now(timezone.utc)
    day_start = now - timedelta(hours=24)
    daily = 0.0
    batch = 0.0
    for run in runs:
        cost = _reliable_cost(run)
        started = parse_timestamp(run.get("startedAt"))
        if started is not None and started >= day_start:
            daily += cost
        if batch_id and run.get("batchId") == batch_id:
            batch += cost
    return {"daily": round(daily, 8), "batch": round(batch, 8)}


def enforce_admission_budgets(run_store, cfg: AgentConfig | None = None,
                              *, batch_id: str | None = None) -> dict[str, float]:
    caps = effective_caps(cfg)
    spend = current_spend(run_store.list_all_runs(), batch_id=batch_id)
    if spend["daily"] >= caps["daily"]:
        error = BudgetExceededError(
            "daily", spend["daily"], caps["daily"], refused="run admission"
        )
        log.warning("budget admission refused scope=daily spend=%s cap=%s",
                    spend["daily"], caps["daily"])
        raise error
    if batch_id and spend["batch"] >= caps["batch"]:
        error = BudgetExceededError(
            "batch", spend["batch"], caps["batch"], refused="run admission"
        )
        log.warning("budget admission refused scope=batch batch=%s spend=%s cap=%s",
                    batch_id, spend["batch"], caps["batch"])
        raise error
    return spend


def build_usage_summary(run_store, *, window: str = "7d",
                        cfg: AgentConfig | None = None,
                        now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    start = window_start(window, now=now)
    selected = [
        run for run in run_store.list_all_runs()
        if (parse_timestamp(run.get("startedAt")) or datetime.min.replace(tzinfo=timezone.utc))
        >= start
    ]
    per_day: dict[str, dict] = {}
    per_song: dict[str, dict] = {}
    per_model: dict[str, dict] = {}
    per_batch: dict[str, dict] = {}

    def add(bucket: dict, key: str, run: dict) -> None:
        item = bucket.setdefault(key, {
            "runs": 0,
            "usage": {
                "inputTokens": 0, "outputTokens": 0,
                "cacheCreationInputTokens": 0, "cacheReadInputTokens": 0,
            },
            "costUSD": 0.0,
        })
        item["runs"] += 1
        usage = internal_usage(run.get("usage"))
        item["usage"]["inputTokens"] += usage["input_tokens"]
        item["usage"]["outputTokens"] += usage["output_tokens"]
        item["usage"]["cacheCreationInputTokens"] += usage["cache_creation_input_tokens"]
        item["usage"]["cacheReadInputTokens"] += usage["cache_read_input_tokens"]
        item["costUSD"] = round(item["costUSD"] + _reliable_cost(run), 8)

    reliable_runs = 0
    unreliable_runs = 0
    for run in selected:
        if run.get("usageReliable", False):
            reliable_runs += 1
        else:
            unreliable_runs += 1
        started = parse_timestamp(run.get("startedAt"))
        add(per_day, started.date().isoformat() if started else "unknown", run)
        add(per_song, str(run.get("songId") or "unknown"), run)
        add(per_model, str(run.get("model") or "unknown"), run)
        if run.get("batchId"):
            add(per_batch, str(run["batchId"]), run)

    caps = effective_caps(cfg)
    spend = current_spend(run_store.list_all_runs(), now=now)
    total = round(sum(_reliable_cost(run) for run in selected), 8)
    return {
        "window": window,
        "from": start.isoformat(),
        "to": now.isoformat(),
        "runs": len(selected),
        "reliableRuns": reliable_runs,
        "unreliableRuns": unreliable_runs,
        "costUSD": total,
        "perDay": per_day,
        "perSong": per_song,
        "perModel": per_model,
        "perBatch": per_batch,
        "budget": {
            "daily": {
                "currentSpendUSD": spend["daily"],
                "capUSD": caps["daily"],
                "remainingUSD": round(max(0.0, caps["daily"] - spend["daily"]), 8),
                "exceeded": spend["daily"] >= caps["daily"],
            },
            "perRunCapUSD": caps["run"],
            "perBatchCapUSD": caps["batch"],
        },
        "priceTableVersion": settings.llm_price_table_version,
    }
