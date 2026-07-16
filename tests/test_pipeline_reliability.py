"""Analyze-pipeline reliability: never hang, fail loudly with the step name.

Covers the §3/§4 contract: a fatal step failure or timeout becomes a
502 whose detail names the step; best-effort steps (discover/acquire/mir) are
recorded as failed but don't sink the request; and the mock path runs the whole
analyze -> persist -> fetch -> versions flow with no external calls.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server import pipeline as pipeline_mod
from snoocle_server.api import app
from snoocle_server.config import settings
from snoocle_server.discovery.search import SearchError
from snoocle_server.reconcile.engine import ReconcileError, ReconcileResult
from snoocle_server.schema import Song
from snoocle_server.store.memory import InMemorySongRepository

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch):
    store = InMemorySongRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: store)
    monkeypatch.setattr("snoocle_server.pipeline.get_store", lambda: store)
    return store


def _fake_song(song_id="anon--x") -> Song:
    return Song.model_validate(
        {
            "id": song_id,
            "metadata": {"title": "X", "artist": "Anon"},
            "lines": [{"lineIndex": 0, "lyrics": "la", "chordPlacements": [{"charIndex": 0, "chord": "C"}]}],
            "provenance": [{"timestamp": "2026-07-09T00:00:00Z", "actor": "reconcile:test/fake", "action": "reconciled"}],
        }
    )


def _fake_result(song_id="anon--x") -> ReconcileResult:
    return ReconcileResult(
        song=_fake_song(song_id), provider="anthropic", model="fake", attempts=1,
        audio_attached=False, usage={},
    )


def test_mock_analyze_is_fully_offline_and_persists(monkeypatch):
    """provider=mock must make ZERO external calls: any attempt to discover,
    acquire, or reconcile-over-network would raise here."""
    def boom(*a, **k):  # noqa: ANN001
        raise AssertionError("external call attempted in the mock path")

    monkeypatch.setattr(pipeline_mod, "discover_sources", boom)
    monkeypatch.setattr(pipeline_mod, "acquire", boom)

    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Offline", "artist": "Tester", "provider": "mock", "skipAudio": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["steps"]["discover"].startswith("skipped")
    assert body["storedVersion"]
    # persisted + fetchable + versioned
    sid = body["songId"]
    assert sid in client.get("/v1/songs").json()["songs"]
    assert client.get(f"/v1/songs/{sid}").status_code == 200
    assert client.get(f"/v1/songs/{sid}/versions").json()["versions"]


def test_fatal_reconcile_failure_returns_502_naming_step(monkeypatch):
    def fail(*a, **k):  # noqa: ANN001
        raise ReconcileError("provider exploded")

    monkeypatch.setattr(pipeline_mod, "reconcile", fail)
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Boom", "artist": "Tester", "provider": "mock", "skipAudio": True},
    )
    assert r.status_code == 502
    assert r.json()["detail"].startswith("reconcile: ")
    assert "provider exploded" in r.json()["detail"]


def test_fatal_502_detail_includes_step_outcomes(monkeypatch):
    """The 502 detail carries the per-step outcomes so the client can see WHY
    the fatal step had nothing to work with (not just that it failed)."""
    def fail(*a, **k):  # noqa: ANN001
        raise ReconcileError("nothing to reconcile: no candidate sources and no MIR analysis")

    monkeypatch.setattr(pipeline_mod, "reconcile", fail)
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Boom", "artist": "Tester", "provider": "mock", "skipAudio": True},
    )
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail.startswith("reconcile: nothing to reconcile")
    assert "[steps:" in detail
    assert "discover=" in detail and "mir=skipped" in detail


def test_reconcile_timeout_returns_502(monkeypatch):
    monkeypatch.setattr(settings, "reconcile_timeout_seconds", 0.2)

    def slow(*a, **k):  # noqa: ANN001
        time.sleep(1.0)
        return _fake_result()

    monkeypatch.setattr(pipeline_mod, "reconcile", slow)
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Slow", "artist": "Tester", "provider": "mock", "skipAudio": True},
    )
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail.startswith("reconcile: ") and "timed out" in detail


def test_best_effort_discover_failure_does_not_sink_request(monkeypatch):
    """A non-mock provider whose discovery fails still succeeds if reconcile
    can proceed — the failure is recorded, not fatal."""
    def fail_discover(*a, **k):  # noqa: ANN001
        raise SearchError("all search backends down")

    monkeypatch.setattr(pipeline_mod, "discover_sources", fail_discover)
    monkeypatch.setattr(pipeline_mod, "reconcile", lambda *a, **k: _fake_result("anon--x"))

    r = client.post(
        "/v1/songs/analyze",
        json={"title": "X", "artist": "Anon", "provider": "anthropic", "skipAudio": True},
    )
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert steps["discover"].startswith("failed")
    assert steps["reconcile"].startswith("ok")
    assert r.json()["storedVersion"]
