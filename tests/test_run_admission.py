"""Duplicate reconciliation admission: concurrency, TTL, evidence, and force."""

from __future__ import annotations

import threading
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server import pipeline as pipeline_mod
from snoocle_server.api import app
from snoocle_server.config import settings
from snoocle_server.store.memory import InMemorySongRepository
from snoocle_server.store.run_admission import (
    DuplicateRunError,
    InMemoryRunAdmissionRepository,
    idempotency_key,
)
from snoocle_server.store.runs import InMemoryRunRepository


@pytest.fixture
def stores(monkeypatch):
    songs = InMemorySongRepository()
    runs = InMemoryRunRepository()
    locks = InMemoryRunAdmissionRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: songs)
    monkeypatch.setattr(pipeline_mod, "get_store", lambda: songs)
    monkeypatch.setattr(pipeline_mod, "get_run_store", lambda: runs)
    monkeypatch.setattr(pipeline_mod, "get_run_admission_store", lambda: locks)
    monkeypatch.setattr("snoocle_server.store.runs.get_run_store", lambda: runs)
    monkeypatch.setattr(settings, "quality_enabled", False)
    return songs, runs, locks


def _request(**extra) -> dict:
    return {
        "title": "Admission Test",
        "artist": "Tester",
        "provider": "mock",
        "skipAudio": True,
        **extra,
    }


def test_two_concurrent_identical_requests_admit_one_and_return_one_409(
    monkeypatch, stores
):
    _, runs, _ = stores
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    calls_lock = threading.Lock()
    real_step = pipeline_mod._step_reconcile

    def slow_reconcile(*args, **kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
        entered.set()
        assert release.wait(5), "test did not release the admitted run"
        return real_step(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "_step_reconcile", slow_reconcile)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(lambda: TestClient(app).post("/v1/songs/analyze", json=_request()))
        assert entered.wait(5), "first request never reached reconciliation"
        duplicate = TestClient(app).post("/v1/songs/analyze", json=_request())
        assert duplicate.status_code == 409, duplicate.text
        assert duplicate.json()["errorCode"] == "duplicate_run"
        assert duplicate.json()["blockingRunId"]
        release.set()
        admitted = first.result(timeout=5)

    assert admitted.status_code == 200, admitted.text
    assert duplicate.json()["blockingRunId"] == admitted.json()["runId"]
    assert calls == 1
    assert len(runs.list_runs(admitted.json()["songId"])) == 1


def test_completed_retry_false_run_is_refused_with_existing_summary(stores):
    client = TestClient(app)
    first = client.post("/v1/songs/analyze", json=_request())
    duplicate = client.post("/v1/songs/analyze", json=_request())

    assert first.status_code == 200, first.text
    assert duplicate.status_code == 409, duplicate.text
    body = duplicate.json()
    assert body["blockingRunId"] == first.json()["runId"]
    assert body["existingRun"]["runId"] == first.json()["runId"]
    assert body["existingRun"]["storedVersion"] == first.json()["storedVersion"]


def test_duplicate_guidance_does_not_reopen_an_applied_correction(stores):
    from snoocle_server.store.song_notes import get_song_notes_store, reset_song_notes_store

    reset_song_notes_store()
    client = TestClient(app)
    request = _request(
        guidance="keep the existing formatting",
        scope={"listen": True, "reconcile": True},
    )
    first = client.post("/v1/songs/analyze", json=request)
    duplicate = client.post("/v1/songs/analyze", json=request)

    assert first.status_code == 200, first.text
    assert duplicate.status_code == 409, duplicate.text
    correction = get_song_notes_store().get_record(first.json()["songId"])["correction"]
    assert correction["applied_to_version"] == first.json()["storedVersion"]
    reset_song_notes_store()


def test_changed_evidence_hash_is_admitted():
    store = InMemoryRunAdmissionRepository()
    key_a, hash_a = idempotency_key(
        "artist--song",
        "cfg",
        {"sources": {"ids": ["source-a"], "gatheredAt": "2026-07-31T07:00:00Z"}},
    )
    key_b, hash_b = idempotency_key(
        "artist--song",
        "cfg",
        {"sources": {"ids": ["source-b"], "gatheredAt": "2026-07-31T07:00:00Z"}},
    )
    assert key_a != key_b and hash_a != hash_b
    store.admit(
        key=key_a, evidence_hash=hash_a, run_id="run-a", lease_seconds=60,
        completed_ttl_seconds=86400,
    )
    admitted = store.admit(
        key=key_b, evidence_hash=hash_b, run_id="run-b", lease_seconds=60,
        completed_ttl_seconds=86400,
    )
    assert admitted.run_id == "run-b"


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("mir", "analyzedAt"), "2026-07-31T08:00:00Z"),
        (("mir", "engines"), {"chords": "new-engine"}),
        (("sources", "ids"), ["source-b"]),
        (("sources", "gatheredAt"), "2026-07-31T08:00:00Z"),
        (("lrcAlign", "fingerprint"), "lrc-b"),
        (("request", "scope"), {"listen": False, "reconcile": True}),
        (("request", "notes"), "different guidance"),
    ],
)
def test_every_evidence_content_identifier_changes_the_key(path, replacement):
    manifest = {
        "mir": {"analyzedAt": "2026-07-31T07:00:00Z", "engines": {"chords": "v1"}},
        "sources": {"ids": ["source-a"], "gatheredAt": "2026-07-31T07:00:00Z"},
        "lrcAlign": {"fingerprint": "lrc-a"},
        "request": {
            "scope": {"listen": True, "reconcile": True},
            "notes": "original guidance",
        },
    }
    changed = deepcopy(manifest)
    changed[path[0]][path[1]] = replacement
    assert idempotency_key("artist--song", "cfg", manifest) != idempotency_key(
        "artist--song", "cfg", changed
    )


def test_force_admits_and_marks_the_run(stores):
    _, runs, _ = stores
    client = TestClient(app)
    first = client.post("/v1/songs/analyze", json=_request())
    forced = client.post(
        "/v1/songs/analyze",
        json=_request(force=True, forceReason="operator corrected the source selection"),
    )

    assert first.status_code == 200, first.text
    assert forced.status_code == 200, forced.text
    assert forced.json()["runId"] != first.json()["runId"]
    trace = runs.get_run(forced.json()["runId"])
    assert trace["forced"] is True
    assert trace["forceReason"] == "operator corrected the source selection"


def test_force_requires_a_reason(stores):
    response = TestClient(app).post(
        "/v1/songs/analyze", json=_request(force=True)
    )
    assert response.status_code == 422


def test_stale_crashed_lock_expires_and_admits():
    store = InMemoryRunAdmissionRepository()
    now = datetime(2026, 7, 31, 7, 3, 13, tzinfo=timezone.utc)
    first = store.admit(
        key="same", evidence_hash="evidence", run_id="crashed", lease_seconds=30,
        completed_ttl_seconds=86400, now=now,
    )
    with pytest.raises(DuplicateRunError) as blocked:
        store.admit(
            key="same", evidence_hash="evidence", run_id="too-soon", lease_seconds=30,
            completed_ttl_seconds=86400, now=now + timedelta(seconds=29),
        )
    assert blocked.value.blocking_run_id == first.run_id

    admitted = store.admit(
        key="same", evidence_hash="evidence", run_id="replacement", lease_seconds=30,
        completed_ttl_seconds=86400, now=now + timedelta(seconds=31),
    )
    assert admitted.run_id == "replacement"


def test_mcp_duplicate_is_a_machine_readable_error(monkeypatch):
    from snoocle_server import mcp_server as mcp_mod
    from snoocle_server.reconcile import admission as direct_mod

    locks = InMemoryRunAdmissionRepository()
    runs = InMemoryRunRepository()
    monkeypatch.setattr(direct_mod, "get_run_admission_store", lambda: locks)
    monkeypatch.setattr(direct_mod, "get_run_store", lambda: runs)

    first = mcp_mod.reconcile_song(
        "MCP Admission", "Tester", candidates_json="[]", provider="mock"
    )
    with pytest.raises(DuplicateRunError) as duplicate:
        mcp_mod.reconcile_song(
            "MCP Admission", "Tester", candidates_json="[]", provider="mock"
        )

    assert duplicate.value.code == "duplicate_run"
    assert duplicate.value.blocking_run_id == first["runId"]
    assert '"code":"duplicate_run"' in str(duplicate.value)
    assert duplicate.value.summary["runId"] == first["runId"]


def test_standalone_reconcile_rest_surface_uses_the_same_gate(monkeypatch):
    from snoocle_server.reconcile import admission as direct_mod

    locks = InMemoryRunAdmissionRepository()
    runs = InMemoryRunRepository()
    monkeypatch.setattr(direct_mod, "get_run_admission_store", lambda: locks)
    monkeypatch.setattr(direct_mod, "get_run_store", lambda: runs)
    body = {
        "title": "REST Admission",
        "artist": "Tester",
        "provider": "mock",
        "candidates": [],
    }
    client = TestClient(app)
    first = client.post("/v1/reconcile", json=body)
    duplicate = client.post("/v1/reconcile", json=body)

    assert first.status_code == 200, first.text
    assert first.json()["runId"]
    assert duplicate.status_code == 409, duplicate.text
    assert duplicate.json()["errorCode"] == "duplicate_run"
    assert duplicate.json()["blockingRunId"] == first.json()["runId"]
