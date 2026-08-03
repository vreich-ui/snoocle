"""Suggesting a better recording — and never analyzing one on its own.

An AUDIO verdict means the document is as good as its recording allows. The fix
is a different recording, and finding one costs a search; USING one costs a
download, a full MIR pass, and possibly a model call. So the split is strict:
this reports, the operator spends.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server import pipeline as pipeline_mod
from snoocle_server import recordings as recordings_mod
from snoocle_server.api import app
from snoocle_server.config import settings
from snoocle_server.recordings import realign_action, suggest_recordings
from snoocle_server.schema import Song
from snoocle_server.store.memory import InMemorySongRepository

client = TestClient(app)

SONG_ID = "the-rolling-stones--paint-it-black"
CURRENT_VIDEO = "livevideo12"


def _song() -> Song:
    return Song.model_validate(
        {
            "id": SONG_ID,
            "metadata": {"title": "Paint It Black", "artist": "The Rolling Stones"},
            "audio": {"youtubeVideoId": CURRENT_VIDEO, "analyzedVideoId": CURRENT_VIDEO},
            "lines": [
                {"lineIndex": 0, "lyrics": "I see a red door",
                 "chordPlacements": [{"charIndex": 0, "chord": "Em"}]}
            ],
        }
    )


def _entry(video_id: str, title: str, *, channel: str = "SomeChannel", duration: int = 202):
    return {"id": video_id, "title": title, "channel": channel, "duration": duration,
            "url": f"https://www.youtube.com/watch?v={video_id}"}


@pytest.fixture
def store(monkeypatch):
    repo = InMemorySongRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: repo)
    monkeypatch.setattr(pipeline_mod, "get_store", lambda: repo)
    return repo


# --- ranking -----------------------------------------------------------------


def test_official_studio_audio_outranks_covers_and_lessons(monkeypatch):
    monkeypatch.setattr(
        recordings_mod, "search_video",
        lambda title, artist, max_results=8: [
            _entry("cover123456", "Paint It Black cover by SomeGuy"),
            _entry("lesson123456"[:11], "How to play Paint It Black - guitar lesson"),
            _entry("official123", "The Rolling Stones - Paint It Black (Official Audio)",
                   channel="TheRollingStonesVEVO"),
            _entry("remaster123", "Paint It Black (Remastered 2002)"),
        ],
    )

    report = suggest_recordings(_song())

    assert [s.video_id for s in report.suggestions] == [
        "official123",  # official audio, artist channel
        "remaster123",  # a studio master
        "cover123456",  # a cover: penalized
        "lesson12345",  # a lesson: penalized harder
    ]
    assert report.suggestions[0].score > report.suggestions[-1].score


def test_the_recording_the_song_already_uses_is_never_suggested(monkeypatch):
    monkeypatch.setattr(
        recordings_mod, "search_video",
        lambda title, artist, max_results=8: [
            _entry(CURRENT_VIDEO, "Paint It Black (Live 1966)"),
            _entry("official123", "Paint It Black (Official Audio)"),
        ],
    )

    report = suggest_recordings(_song())

    assert [s.video_id for s in report.suggestions] == ["official123"]
    assert CURRENT_VIDEO in report.excluded


def test_every_suggestion_carries_the_action_that_would_spend(monkeypatch):
    monkeypatch.setattr(
        recordings_mod, "search_video",
        lambda title, artist, max_results=8: [_entry("official123", "Paint It Black")],
    )

    report = suggest_recordings(_song())
    payload = report.to_dict()

    assert payload["analyzed"] is False
    assert payload["suggestions"][0]["action"] == realign_action(SONG_ID, "official123")
    assert payload["suggestions"][0]["action"] == (
        f"analyze official123 as the timing reference for {SONG_ID}"
    )
    assert "Nothing has been analyzed" in payload["howToApply"]


def test_a_failed_search_is_reported_not_raised(monkeypatch):
    """This hangs off a run that already produced a storable document; it must
    never be able to turn that run into a failure."""

    def boom(*a, **k):
        raise RuntimeError("YouTube said no")

    monkeypatch.setattr(recordings_mod, "search_video", boom)

    report = suggest_recordings(_song())

    assert report.suggestions == []
    assert report.error and "YouTube said no" in report.error
    assert "failed" in report.describe()
    assert report.to_dict()["analyzed"] is False


def test_the_ranking_is_acquisitions_own_judgement_not_a_second_one():
    from snoocle_server.audio import acquire

    assert recordings_mod.score_video is acquire.score_video
    # And picking the single best still goes through it.
    entries = [
        _entry("cover123456", "Paint It Black cover"),
        _entry("official123", "Paint It Black (Official Audio)"),
    ]
    assert acquire.pick_best_video(entries, "Paint It Black", "The Rolling Stones")[
        "id"
    ] == "official123"


# --- the HTTP surface --------------------------------------------------------


def test_the_endpoint_reports_and_changes_nothing(monkeypatch, store):
    store.save(_song(), message="gold")
    monkeypatch.setattr(
        recordings_mod, "search_video",
        lambda title, artist, max_results=8: [
            _entry("official123", "Paint It Black (Official Audio)")
        ],
    )

    before = store.current_version(SONG_ID)
    r = client.get(f"/v1/songs/{SONG_ID}/recording-suggestions")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["songId"] == SONG_ID
    assert body["analyzed"] is False
    assert body["suggestions"][0]["videoId"] == "official123"
    assert store.current_version(SONG_ID) == before, "a GET must not store anything"


def test_suggestions_for_an_unknown_song_are_a_404(store):
    r = client.get("/v1/songs/no-such--song/recording-suggestions")
    assert r.status_code == 404


# --- the analyze pipeline's own audio-fault hook ------------------------------


def _audio_fault_run(monkeypatch, store):
    """An analyze run that grades as an AUDIO fault, reusing the fixtures that
    pin that verdict in tests/test_quality_pipeline.py rather than inventing a
    second set that might drift from it."""
    from test_quality_pipeline import (  # noqa: PLC0415
        OTHER_SONG,
        TRUE_PROGRESSION,
        _Reconciler,
        _candidate,
        _mir,
        _song,
        _wire,
    )

    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(pipeline_mod, "fetch_lrc", lambda *a, **k: None)
    # Two sheets agreeing with each other, and a recording that contradicts
    # them both: the audio is what is wrong here.
    _wire(monkeypatch, mir=_mir(OTHER_SONG),
          candidates=[_candidate(TRUE_PROGRESSION, "web-1"),
                      _candidate(TRUE_PROGRESSION, "web-2")])
    monkeypatch.setattr(
        pipeline_mod, "reconcile",
        _Reconciler(_song(TRUE_PROGRESSION, timed=False, section_times=False)),
    )
    return pipeline_mod.run_pipeline(
        "Quality Gate", "Test", provider="anthropic", agent_policy="always"
    )


def test_an_audio_fault_analyze_run_reports_alternatives_without_analyzing_them(
    monkeypatch, store
):
    """The wiring the brief asks for: when a version comes out
    timing-unreliable, the run looks for a better recording and reports it."""
    calls: list[dict] = []

    def fake_suggest(song, **kwargs):
        calls.append({"songId": song.id, **kwargs})
        return recordings_mod.RecordingSuggestions(
            song_id=song.id,
            reason=kwargs.get("reason", ""),
            suggestions=[
                recordings_mod.RecordingSuggestion(
                    video_id="official123", title="Paint It Black (Official Audio)",
                    channel="VEVO", duration_seconds=202.0, score=4.5,
                    url="https://youtu.be/official123",
                    action=realign_action(song.id, "official123"),
                )
            ],
        )

    monkeypatch.setattr(pipeline_mod, "suggest_recordings", fake_suggest)
    report = _audio_fault_run(monkeypatch, store)

    assert report.steps["timing-reliability"] == "marked unreliable (audio fault)"
    assert calls, "an audio fault must look for a better recording"
    assert "audio fault" in calls[0]["reason"]
    assert report.recording_suggestions is not None
    assert report.recording_suggestions.suggestions[0].video_id == "official123"
    # Reported, not acted on: still exactly one stored version, from this run.
    assert len(store.versions(report.song_id)) == 1


def test_the_search_can_be_switched_off(monkeypatch, store):
    monkeypatch.setattr(settings, "quality_suggest_recordings", False)

    def boom(*a, **k):
        raise AssertionError("the recording search ran despite being switched off")

    monkeypatch.setattr(pipeline_mod, "suggest_recordings", boom)
    report = _audio_fault_run(monkeypatch, store)

    assert report.recording_suggestions is None
    assert "recording-suggestions" not in report.steps
