"""Per-song reconciliation notes: the store, the REST surface, and the replay.

The load-bearing behaviour is replay — a note typed once must shape EVERY later
analysis of that song, keyed by the same song id the pipeline resolves, and it
must be visible in provenance rather than applied silently.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server import pipeline as pipeline_mod
from snoocle_server.api import app
from snoocle_server.reconcile.engine import ReconcileError
from snoocle_server.schema.song import slugify_song_id
from snoocle_server.store.memory import InMemorySongRepository
from snoocle_server.store.runs import InMemoryRunRepository, reset_run_store
from snoocle_server.store.song_notes import (
    InMemorySongNotesStore,
    build_song_notes_store,
    get_song_notes_store,
    reset_song_notes_store,
)

SONG_ID = slugify_song_id("Tester", "Fixme")  # what analyze below resolves to


# --- store contract (both backends) -----------------------------------------


@pytest.fixture(params=["memory", "firestore"])
def notes_store(request):
    if request.param == "memory":
        yield InMemorySongNotesStore()
        return
    if not os.environ.get("FIRESTORE_EMULATOR_HOST"):
        pytest.skip("Firestore emulator not running (set FIRESTORE_EMULATOR_HOST)")
    pytest.importorskip("google.cloud.firestore")
    from snoocle_server.store.song_notes import FirestoreSongNotesStore

    store = FirestoreSongNotesStore(project=os.environ.get("GOOGLE_CLOUD_PROJECT", "snoocle-test"))
    store._COLLECTION = "song_notes_test_" + uuid.uuid4().hex[:8]
    try:
        yield store
    finally:
        for doc in store._client.collection(store._COLLECTION).list_documents():
            doc.delete()


def test_notes_store_roundtrip(notes_store):
    assert notes_store.get("nobody--nothing") == ""
    assert notes_store.get_record("nobody--nothing") is None

    assert notes_store.set(SONG_ID, "  the bridge is Bm  ") == "the bridge is Bm"
    assert notes_store.get(SONG_ID) == "the bridge is Bm"  # stored trimmed
    rec = notes_store.get_record(SONG_ID)
    assert rec["notes"] == "the bridge is Bm" and rec["updated_at"]

    notes_store.set(SONG_ID, "second take")
    assert notes_store.get(SONG_ID) == "second take"


def test_notes_store_empty_deletes(notes_store):
    notes_store.set(SONG_ID, "temporary")
    assert notes_store.set(SONG_ID, "   ") == ""
    assert notes_store.get(SONG_ID) == ""
    assert notes_store.get_record(SONG_ID) is None


def test_notes_store_delete_reports_whether_anything_was_there(notes_store):
    assert notes_store.delete(SONG_ID) is False
    notes_store.set(SONG_ID, "kill me")
    assert notes_store.delete(SONG_ID) is True
    assert notes_store.get(SONG_ID) == ""


def test_notes_store_singleton_resets(monkeypatch):
    reset_song_notes_store()
    first = get_song_notes_store()
    assert first is get_song_notes_store()
    assert isinstance(build_song_notes_store(), InMemorySongNotesStore)  # default backend
    reset_song_notes_store()
    assert get_song_notes_store() is not first


# --- REST surface ------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    songs = InMemorySongRepository()
    runs = InMemoryRunRepository()
    notes = InMemorySongNotesStore()
    monkeypatch.setattr(api_mod, "get_store", lambda: songs)
    monkeypatch.setattr("snoocle_server.pipeline.get_store", lambda: songs)
    monkeypatch.setattr("snoocle_server.pipeline.get_run_store", lambda: runs)
    monkeypatch.setattr("snoocle_server.store.runs.get_run_store", lambda: runs)
    monkeypatch.setattr("snoocle_server.store.song_notes.get_song_notes_store", lambda: notes)
    reset_run_store()
    reset_song_notes_store()
    return TestClient(app)


def test_get_notes_of_unknown_song_is_an_empty_document_not_404(client):
    r = client.get("/v1/songs/never--seen/notes")
    assert r.status_code == 200
    assert r.json() == {"songId": "never--seen", "notes": "", "updatedAt": None}


def test_put_get_delete_notes(client):
    put = client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "capo-free voicings please"})
    assert put.status_code == 200
    body = put.json()
    assert body["songId"] == SONG_ID
    assert body["notes"] == "capo-free voicings please"
    assert body["updatedAt"]

    assert client.get(f"/v1/songs/{SONG_ID}/notes").json()["notes"] == "capo-free voicings please"

    assert client.delete(f"/v1/songs/{SONG_ID}/notes").json() == {"deleted": True}
    assert client.delete(f"/v1/songs/{SONG_ID}/notes").json() == {"deleted": False}
    assert client.get(f"/v1/songs/{SONG_ID}/notes").json()["notes"] == ""


def test_put_empty_notes_deletes_the_record(client):
    client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "something"})
    cleared = client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "   "})
    assert cleared.status_code == 200
    assert cleared.json() == {"songId": SONG_ID, "notes": "", "updatedAt": None}


def test_put_notes_rejects_over_the_length_cap(client):
    r = client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "x" * 8001})
    assert r.status_code == 400
    assert "8000" in r.json()["detail"]
    # and nothing was stored
    assert client.get(f"/v1/songs/{SONG_ID}/notes").json()["notes"] == ""
    # the cap itself is inclusive
    assert client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "y" * 8000}).status_code == 200


# --- replay on analyze -------------------------------------------------------


@pytest.fixture()
def reconcile_calls(monkeypatch):
    """Capture the kwargs the reconciler is actually called with, while still
    running the real engine (so provenance is the real thing)."""
    calls: list[dict] = []
    real = pipeline_mod.reconcile

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "reconcile", spy)
    return calls


def _analyze(client, **extra) -> dict:
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Fixme", "artist": "Tester", "provider": "mock",
              "skipAudio": True, **extra},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _reconciled_notes(body: dict) -> str:
    entry = next(p for p in body["song"]["provenance"] if p["action"] == "reconciled")
    return entry["notes"]


def test_stored_notes_replay_as_guidance_when_the_request_omits_it(client, reconcile_calls):
    client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "the bridge is Bm, not D"})
    body = _analyze(client)
    assert body["songId"] == SONG_ID  # the id the notes were stored under
    assert reconcile_calls[-1]["guidance"] == "the bridge is Bm, not D"
    assert "stored notes" in _reconciled_notes(body)


def test_no_notes_means_no_guidance(client, reconcile_calls):
    body = _analyze(client)
    assert reconcile_calls[-1]["guidance"] is None
    assert "guidance" not in _reconciled_notes(body)


def test_request_guidance_wins_and_is_persisted_for_next_time(client, reconcile_calls):
    client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "OLD note"})
    body = _analyze(client, guidance="NEW instruction")
    assert reconcile_calls[-1]["guidance"] == "NEW instruction"
    assert "this request" in _reconciled_notes(body)
    # persisted, so the *next* run replays it without the client resending
    assert client.get(f"/v1/songs/{SONG_ID}/notes").json()["notes"] == "NEW instruction"
    _analyze(client)
    assert reconcile_calls[-1]["guidance"] == "NEW instruction"


def test_notes_survive_a_failing_analysis(client, monkeypatch):
    def boom(*args, **kwargs):
        raise ReconcileError("provider exploded")

    monkeypatch.setattr(pipeline_mod, "reconcile", boom)
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Fixme", "artist": "Tester", "provider": "mock",
              "skipAudio": True, "guidance": "keep me"},
    )
    assert r.status_code == 502
    # the typed instruction is not lost with the run that failed
    assert client.get(f"/v1/songs/{SONG_ID}/notes").json()["notes"] == "keep me"


def test_notes_store_outage_does_not_fail_an_analysis(client, monkeypatch, reconcile_calls):
    class _Broken(InMemorySongNotesStore):
        def get(self, song_id: str) -> str:
            raise RuntimeError("firestore down")

    monkeypatch.setattr("snoocle_server.store.song_notes.get_song_notes_store", lambda: _Broken())
    body = _analyze(client)
    assert body["songId"] == SONG_ID
    assert reconcile_calls[-1]["guidance"] is None
