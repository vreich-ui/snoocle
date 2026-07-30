"""Pipeline-level routing: a targeted correction gets notes-only scope
inferred, skips discovery, and (when it produced a patch) skips the
timing/LRC/confidence passes that exist to guard a REGENERATED document —
none of which apply when nothing was regenerated.

Uses a stand-in `reconcile()` (matching tests/test_timing_carry_forward.py's
`_StaticReconciler` pattern) so these tests are about pipeline.py's OWN
wiring — scope inference, the discovery skip, threading patch_ops_eligible
through, reading patch_ops_applied back — independent of the LLM call
mechanics, which tests/test_patch_reconcile.py covers directly against
reconcile/engine.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server import pipeline as pipeline_mod
from snoocle_server.api import app
from snoocle_server.config import settings
from snoocle_server.reconcile.engine import ReconcileResult
from snoocle_server.schema import Song
from snoocle_server.store.memory import InMemorySongRepository
from snoocle_server.store.song_notes import reset_song_notes_store

client = TestClient(app)

SONG_ID = "the-rolling-stones--paint-it-black"


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch):
    store = InMemorySongRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: store)
    monkeypatch.setattr(pipeline_mod, "get_store", lambda: store)
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(pipeline_mod, "fetch_lrc", lambda *a, **k: None)
    # Every test here re-analyzes the SAME song id — without resetting this
    # (a separate, process-wide singleton) a guidance note set by one test
    # replays as a later test's "no guidance given" case, per
    # _resolve_guidance's stored-notes-replay contract.
    reset_song_notes_store()
    yield store
    reset_song_notes_store()


def _gold_song() -> Song:
    return Song.model_validate(
        {
            "id": SONG_ID,
            "metadata": {"title": "Paint It Black", "artist": "The Rolling Stones", "bpm": 128.0},
            "audio": {
                "youtubeVideoId": "flSmiIne4k1",
                "beats": [{"time": 0.0, "measure": 1, "beatInMeasure": 1}],
                "syncMap": [{"lineIndex": 0, "time": 0.0}],
            },
            "sections": [
                {"sectionIndex": 0, "name": "Verse 1", "kind": "verse",
                 "startLineIndex": 0, "endLineIndex": 0, "startTime": 0.0, "endTime": 4.0},
            ],
            "lines": [
                {"lineIndex": 0, "lyrics": "I see a red door", "timeSeconds": 0.0, "confidence": 0.9,
                 "chordPlacements": [{"charIndex": 0, "chord": "C", "timeSeconds": 0.0, "confidence": 0.9}]},
            ],
            "provenance": [
                {"timestamp": "2026-07-01T00:00:00Z", "actor": "reconcile:test/gold", "action": "reconciled"},
            ],
        }
    )


class _Recorder:
    """Stands in for `reconcile`; returns `song` unchanged and records every
    keyword it was called with, including the new `patch_ops_eligible`."""

    def __init__(self, song: Song, patch_ops_applied: int = 0):
        self.song = song
        self.patch_ops_applied = patch_ops_applied
        self.calls: list[dict] = []

    def __call__(self, title, artist, candidates, mir, **kwargs):
        self.calls.append({"candidates": candidates, "mir": mir, **kwargs})
        return ReconcileResult(
            song=self.song.model_copy(deep=True), provider="test-patch", model="fake",
            attempts=1, audio_attached=False, usage={},
            patch_ops_applied=self.patch_ops_applied,
        )

    @property
    def last(self) -> dict:
        assert self.calls, "reconcile was never called"
        return self.calls[-1]


def _analyze(**extra):
    body = {"title": "Paint It Black", "artist": "The Rolling Stones", "provider": "anthropic"}
    body.update(extra)
    return client.post("/v1/songs/analyze", json=body)


def _no_discovery(monkeypatch) -> None:
    def boom(*a, **k):  # noqa: ANN001
        raise AssertionError("discovery ran despite an inferred notes-only scope")

    monkeypatch.setattr(pipeline_mod, "discover_sources", boom)


def _no_audio(monkeypatch) -> None:
    def boom(*a, **k):  # noqa: ANN001
        raise AssertionError("audio was acquired despite an inferred notes-only scope")

    monkeypatch.setattr(pipeline_mod, "acquire", boom)
    monkeypatch.setattr(pipeline_mod, "analyze_audio", boom)


# --- the primary reported scenario: scope inference + no rediscovery -------------


def test_a_targeted_correction_infers_notes_only_and_skips_discovery(monkeypatch, isolated_store):
    isolated_store.save(_gold_song(), message="gold")
    _no_discovery(monkeypatch)
    _no_audio(monkeypatch)
    recorder = _Recorder(_gold_song(), patch_ops_applied=1)
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(guidance="change the C to a B in line 12")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["steps"]["discover"].startswith("skipped (scope: notes only)")
    assert body["steps"]["acquire"].startswith("skipped")
    assert "inferred" in body["steps"]["scope"]
    assert "chord-symbol" in body["steps"]["scope"]
    assert "listen=off, reconcile=off" in body["steps"]["scope"]

    call = recorder.last
    assert call["scope"].notes_only
    assert call["patch_ops_eligible"] is True
    # the stored version stood in as the prior since none was attached
    assert call["prior_song"]["id"] == SONG_ID


def test_a_lyric_targeting_correction_infers_notes_only_but_is_not_patch_eligible(
    monkeypatch, isolated_store
):
    isolated_store.save(_gold_song(), message="gold")
    _no_discovery(monkeypatch)
    _no_audio(monkeypatch)
    recorder = _Recorder(_gold_song())
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(guidance="line 0's lyrics should say 'window' not 'door'")
    assert r.status_code == 200, r.text
    assert "inferred" in r.json()["steps"]["scope"]

    call = recorder.last
    assert call["scope"].notes_only
    assert call["patch_ops_eligible"] is False


# --- explicit caller scope always wins ---------------------------------------


def test_an_explicit_scope_is_never_overridden_by_inference(monkeypatch, isolated_store):
    isolated_store.save(_gold_song(), message="gold")
    ran: list[str] = []
    monkeypatch.setattr(
        pipeline_mod, "discover_sources", lambda *a, **k: ran.append("discover") or []
    )
    recorder = _Recorder(_gold_song())
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(
        guidance="change the C to a B in line 12",  # would otherwise infer notes-only
        scope={"listen": True, "reconcile": True},
        skipAudio=True,
    )
    assert r.status_code == 200, r.text
    assert ran == ["discover"], "the caller's explicit scope must still run discovery"
    assert "inferred" not in r.json()["steps"]["scope"]
    assert r.json()["steps"]["scope"] == "listen=on, reconcile=on"
    assert recorder.last["patch_ops_eligible"] is False, (
        "patch eligibility is meaningless outside notes-only scope"
    )


# --- no prior document, nothing to infer against ------------------------------


def test_scope_inference_never_fires_with_no_prior_to_correct(monkeypatch):
    """"Notes naming a chord" has nothing to apply itself to on a song's
    first-ever analysis — inference must not fire, and the run proceeds as
    an ordinary full analysis."""
    ran: list[str] = []
    monkeypatch.setattr(
        pipeline_mod, "discover_sources", lambda *a, **k: ran.append("discover") or []
    )
    recorder = _Recorder(_gold_song())
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(guidance="change the C to a B in line 12", skipAudio=True)
    assert r.status_code == 200, r.text
    assert ran == ["discover"]
    assert "scope" not in r.json()["steps"]
    assert recorder.last["scope"] is None
    assert recorder.last["patch_ops_eligible"] is False


def test_no_guidance_at_all_is_unaffected(monkeypatch, isolated_store):
    isolated_store.save(_gold_song(), message="gold")
    ran: list[str] = []
    monkeypatch.setattr(
        pipeline_mod, "discover_sources", lambda *a, **k: ran.append("discover") or []
    )
    recorder = _Recorder(_gold_song())
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(skipAudio=True)
    assert r.status_code == 200, r.text
    assert ran == ["discover"]
    assert "scope" not in r.json()["steps"]


# --- a patched result skips the passes that guard a REGENERATED document ---------


def test_a_patched_result_skips_timing_lrc_and_confidence(monkeypatch, isolated_store):
    isolated_store.save(_gold_song(), message="gold")
    _no_discovery(monkeypatch)
    _no_audio(monkeypatch)

    def boom_lrc(*a, **k):  # noqa: ANN001
        raise AssertionError("LRC ran on a patched result")

    monkeypatch.setattr(pipeline_mod, "fetch_lrc", boom_lrc)
    recorder = _Recorder(_gold_song(), patch_ops_applied=1)
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(guidance="change the C to a B in line 12")
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert steps["timing"].startswith("skipped (patch:")
    assert steps["lrc"].startswith("skipped (patch:")
    assert steps["confidence"].startswith("skipped (patch:")


def test_an_unpatched_notes_only_result_still_runs_the_guarding_passes(
    monkeypatch, isolated_store
):
    """The model declining the patch (full-reconcile fallback within notes-
    only scope) must still go through carry-forward — patch_ops_applied=0
    there, so this is the pre-existing path, unchanged."""
    isolated_store.save(_gold_song(), message="gold")
    _no_discovery(monkeypatch)
    _no_audio(monkeypatch)
    recorder = _Recorder(_gold_song(), patch_ops_applied=0)
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(guidance="change the C to a B in line 12")
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert steps["timing"].startswith("ok:")
    assert "preserved" in steps["timing"] or "carried" in steps["timing"].lower()
