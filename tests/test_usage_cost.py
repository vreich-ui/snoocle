"""Trustworthy provider usage, cost budgets, and compact rollups."""

from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from snoocle_server import pipeline as pipeline_mod
from snoocle_server.api import app
from snoocle_server.config import settings
from snoocle_server.mir.base import MirAnalysis
from snoocle_server.reconcile.admission import reconcile_admitted
from snoocle_server.reconcile.anthropic_agent import AnthropicAgentProvider
from snoocle_server.reconcile.providers import AnthropicProvider
from snoocle_server.reconcile.trace import start_run
from snoocle_server.store.runs import InMemoryRunRepository
from snoocle_server.store.jobs import InMemoryJobRepository
from snoocle_server.usage import BudgetExceededError, cost_usd
from snoocle_server.usage_summary import build_usage_summary, enforce_admission_budgets


def _usage(input_tokens=0, output_tokens=0, cache_write=0, cache_read=0):
    return types.SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_write,
        cache_read_input_tokens=cache_read,
    )


def test_anthropic_provider_captures_exact_usage_including_cache(monkeypatch):
    response = types.SimpleNamespace(
        stop_reason="end_turn",
        content=[types.SimpleNamespace(type="text", text="{}")],
        model="claude-opus-4-8",
        usage=_usage(1234, 567, 8901, 2345),
    )

    class Client:
        def __init__(self, **kwargs):
            self.messages = self

        def create(self, **kwargs):
            return response

    fake_anthropic = types.SimpleNamespace(
        Anthropic=Client,
        AnthropicError=RuntimeError,
        APIStatusError=RuntimeError,
        APIConnectionError=RuntimeError,
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    result = AnthropicProvider().complete("system", [{"role": "user", "text": "x"}])

    assert result.usage == {
        "input_tokens": 1234,
        "output_tokens": 567,
        "cache_creation_input_tokens": 8901,
        "cache_read_input_tokens": 2345,
    }


def test_step_and_run_cost_use_configured_price_table(monkeypatch):
    table = {
        "test-model": {"input": 5, "output": 25, "cacheWrite": 6.25, "cacheRead": 0.5}
    }
    monkeypatch.setattr(settings, "llm_price_table", table)
    usage = {
        "input_tokens": 1000, "output_tokens": 200,
        "cache_creation_input_tokens": 3000, "cache_read_input_tokens": 4000,
    }
    recorder = start_run("artist--song", "anthropic", "standard")

    step_cost = recorder.record_model_usage("test-model", usage)

    assert step_cost == 0.03075
    assert recorder.trace.cost_usd == 0.03075
    assert recorder.trace.to_dict()["usage"] == {
        "inputTokens": 1000,
        "outputTokens": 200,
        "cacheCreationInputTokens": 3000,
        "cacheReadInputTokens": 4000,
    }
    assert cost_usd(usage, "test-model") == 0.03075


def test_run_halts_mid_flight_and_persists_budget_exceeded(monkeypatch):
    runs = InMemoryRunRepository()
    calls = 0

    def create(self, **kwargs):
        nonlocal calls
        calls += 1
        return types.SimpleNamespace(
            stop_reason="tool_use",
            content=[],
            usage=_usage(input_tokens=10, output_tokens=100),
            container=None,
        )

    monkeypatch.setattr(settings, "anthropic_api_key", "test")
    monkeypatch.setattr(settings, "run_cost_cap_usd", 0.001)
    monkeypatch.setattr(AnthropicAgentProvider, "_create_client", lambda self: types.SimpleNamespace(
        messages=types.SimpleNamespace(create=types.MethodType(create, self))
    ))
    monkeypatch.setattr("snoocle_server.reconcile.admission.get_run_store", lambda: runs)

    with pytest.raises(BudgetExceededError):
        reconcile_admitted(
            "Budget Song", "Tester", [],
            MirAnalysis(engines={"chords": "test"}, duration_seconds=10, key="C major"),
            provider="anthropic-agent",
        )

    assert calls == 1
    saved = runs.list_all_runs()
    assert len(saved) == 1
    assert saved[0]["status"] == "budget_exceeded"
    assert saved[0]["costUSD"] > 0.001
    assert saved[0]["usage"]["outputTokens"] == 100


def test_daily_cap_refuses_admission_with_current_spend(monkeypatch):
    runs = InMemoryRunRepository()
    runs.save_run({
        "runId": "paid", "songId": "a--b", "model": "claude-opus-4-8",
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "usageReliable": True, "usage": {}, "costUSD": 25.0,
    })
    monkeypatch.setattr(settings, "daily_cost_cap_usd", 25.0)

    with pytest.raises(BudgetExceededError) as caught:
        enforce_admission_budgets(runs)

    assert caught.value.scope == "daily"
    assert caught.value.current_spend == 25.0
    assert caught.value.to_dict()["code"] == "budget_exceeded"


def test_daily_cap_refuses_http_run_admission(monkeypatch):
    runs = InMemoryRunRepository()
    runs.save_run({
        "runId": "paid", "songId": "a--b", "model": "claude-opus-4-8",
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "usageReliable": True, "usage": {}, "costUSD": 25.0,
    })
    monkeypatch.setattr(settings, "daily_cost_cap_usd", 25.0)
    monkeypatch.setattr(pipeline_mod, "get_run_store", lambda: runs)

    response = TestClient(app).post("/v1/songs/analyze", json={
        "title": "No New Bill", "artist": "Tester", "provider": "mock",
        "skipAudio": True,
    })

    assert response.status_code == 429
    assert response.json()["errorCode"] == "budget_exceeded"
    assert response.json()["scope"] == "daily"
    assert response.json()["currentSpendUSD"] == 25.0


def test_batch_cap_refuses_next_run_and_queue_propagates_batch_id(monkeypatch):
    runs = InMemoryRunRepository()
    runs.save_run({
        "runId": "batch-paid", "songId": "a--b", "model": "claude-opus-4-8",
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "usageReliable": True, "usage": {}, "costUSD": 10.0, "batchId": "batch-a",
    })
    monkeypatch.setattr(settings, "batch_cost_cap_usd", 10.0)
    with pytest.raises(BudgetExceededError) as caught:
        enforce_admission_budgets(runs, batch_id="batch-a")
    assert caught.value.scope == "batch"

    jobs = InMemoryJobRepository().submit([{
        "title": "Queued", "artist": "Tester", "batchId": "batch-a",
    }])
    assert jobs[0].batch_id == "batch-a"
    assert jobs[0].to_worker_json()["batchId"] == "batch-a"


def test_usage_summary_endpoint_rolls_up_fixture_runs(monkeypatch):
    runs = InMemoryRunRepository()
    now = datetime.now(timezone.utc)
    fixtures = [
        {
            "runId": "r1", "songId": "a--one", "model": "claude-opus-4-8",
            "startedAt": (now - timedelta(hours=2)).isoformat(),
            "usageReliable": True,
            "usage": {"inputTokens": 100, "outputTokens": 20,
                      "cacheCreationInputTokens": 30, "cacheReadInputTokens": 40},
            "costUSD": 0.01, "batchId": "batch-a",
        },
        {
            "runId": "r2", "songId": "a--one", "model": "claude-opus-4-8",
            "startedAt": (now - timedelta(days=1, hours=2)).isoformat(),
            "usageReliable": True,
            "usage": {"inputTokens": 200, "outputTokens": 30,
                      "cacheCreationInputTokens": 50, "cacheReadInputTokens": 60},
            "costUSD": 0.02, "batchId": "batch-a",
        },
        {
            "runId": "legacy", "songId": "b--two", "model": "claude-opus-4-8",
            "startedAt": (now - timedelta(hours=1)).isoformat(),
            "costUSD": 999,
        },
    ]
    for run in fixtures:
        runs.save_run(run)
    monkeypatch.setattr("snoocle_server.store.runs.get_run_store", lambda: runs)

    response = TestClient(app).get("/v1/usage/summary?window=7d")

    assert response.status_code == 200
    body = response.json()
    assert body["runs"] == 3
    assert body["reliableRuns"] == 2
    assert body["unreliableRuns"] == 1
    assert body["costUSD"] == 0.03
    assert body["perSong"]["a--one"]["usage"]["inputTokens"] == 300
    assert body["perModel"]["claude-opus-4-8"]["costUSD"] == 0.03
    assert body["perBatch"]["batch-a"]["costUSD"] == 0.03
    assert body["perSong"]["b--two"]["costUSD"] == 0.0


def test_old_run_is_marked_usage_unreliable_without_backfill():
    runs = InMemoryRunRepository()
    runs.save_run({"runId": "old", "songId": "a--b", "startedAt": "2025-01-01T00:00:00Z"})
    assert runs.get_run("old")["usageReliable"] is False
    assert "usageReliable" not in runs._runs["old"]
