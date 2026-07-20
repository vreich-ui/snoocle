"""Eval harness: content metrics + the gold/score/scorecard API.

The metric functions are pure (dict in, numbers out) so they're pinned directly;
the API path exercises marking a stored version as gold and scoring against it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server.api import app
from snoocle_server.eval import score_song
from snoocle_server.store.evals import InMemoryEvalStore, reset_eval_store
from snoocle_server.store.memory import InMemorySongRepository

client = TestClient(app)


def _song(lines, sections=None, sync=None):
    return {
        "lines": lines,
        "sections": sections or [],
        "audio": {"syncMap": sync or []},
    }


def _line(i, lyrics, chords):
    return {
        "lineIndex": i,
        "lyrics": lyrics,
        "chordPlacements": [{"charIndex": c[0], "chord": c[1]} for c in chords],
    }


# --- pure metrics -----------------------------------------------------------


def test_identical_songs_score_perfect():
    s = _song([_line(0, "hello world", [(0, "C"), (6, "G")])],
              sections=[{"kind": "verse", "name": "Verse 1"}])
    m = score_song(s, s)
    assert m["chordSimilarity"] == 1.0
    assert m["lyricSimilarity"] == 1.0
    assert m["lyricWER"] == 0.0
    assert m["sectionSimilarity"] == 1.0
    assert m["overall"] == 1.0


def test_chord_difference_lowers_chord_similarity_only():
    gold = _song([_line(0, "hello world", [(0, "C"), (6, "G")])])
    cand = _song([_line(0, "hello world", [(0, "C"), (6, "Am")])])  # G -> Am
    m = score_song(cand, gold)
    assert 0.0 < m["chordSimilarity"] < 1.0
    assert m["lyricSimilarity"] == 1.0  # lyrics untouched
    # root similarity also drops (G root vs A root)
    assert m["chordRootSimilarity"] < 1.0


def test_extension_only_diff_keeps_root_similarity_high():
    gold = _song([_line(0, "x", [(0, "Cmaj7")])])
    cand = _song([_line(0, "x", [(0, "C")])])  # same root, lost the maj7
    m = score_song(cand, gold)
    assert m["chordSimilarity"] < 1.0
    assert m["chordRootSimilarity"] == 1.0  # root C == C


def test_lyric_error_rate():
    gold = _song([_line(0, "the quick brown fox", [])])
    cand = _song([_line(0, "the slow brown fox", [])])  # 1 of 4 words wrong
    m = score_song(cand, gold)
    assert m["lyricWER"] == pytest.approx(0.25, abs=1e-6)
    assert m["lyricSimilarity"] == pytest.approx(0.75, abs=1e-6)


def test_timing_mae_when_both_have_syncmap():
    gold = _song([_line(0, "a", [])], sync=[{"lineIndex": 0, "time": 10.0}])
    cand = _song([_line(0, "a", [])], sync=[{"lineIndex": 0, "time": 12.0}])
    m = score_song(cand, gold)
    assert m["timingMAE"] == pytest.approx(2.0, abs=1e-6)


def test_timing_mae_none_without_syncmap():
    s = _song([_line(0, "a", [])])
    assert score_song(s, s)["timingMAE"] is None


# --- API: gold + score + scorecard ------------------------------------------


@pytest.fixture(autouse=True)
def isolated_stores(monkeypatch):
    store = InMemorySongRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: store)
    monkeypatch.setattr("snoocle_server.pipeline.get_store", lambda: store)
    evals = InMemoryEvalStore()
    monkeypatch.setattr("snoocle_server.store.evals.get_eval_store", lambda: evals)
    reset_eval_store()
    return store


def _make_song_via_mock() -> tuple[str, str]:
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Eval", "artist": "Tester", "provider": "mock", "skipAudio": True},
    )
    assert r.status_code == 200, r.text
    return r.json()["songId"], r.json()["storedVersion"]


def test_mark_gold_then_score_current_is_perfect():
    song_id, version = _make_song_via_mock()

    g = client.put(f"/v1/songs/{song_id}/gold", json={"version": version})
    assert g.status_code == 200, g.text
    assert g.json()["goldVersion"] == version

    s = client.get(f"/v1/songs/{song_id}/score")
    assert s.status_code == 200, s.text
    # current == gold (no edits) -> perfect overall
    assert s.json()["metrics"]["overall"] == 1.0


def test_gold_rejects_unknown_version():
    song_id, _ = _make_song_via_mock()
    r = client.put(f"/v1/songs/{song_id}/gold", json={"version": "deadbeef"})
    assert r.status_code == 404


def test_score_requires_gold():
    song_id, _ = _make_song_via_mock()
    r = client.get(f"/v1/songs/{song_id}/score")
    assert r.status_code == 400
    assert "no gold" in r.json()["detail"]


def test_scorecard_lists_gold_songs_with_aggregate():
    song_id, version = _make_song_via_mock()
    client.put(f"/v1/songs/{song_id}/gold", json={"version": version})

    r = client.get("/v1/eval/scorecard")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 1
    assert body["songs"][0]["songId"] == song_id
    assert body["aggregate"]["overall"] == 1.0
