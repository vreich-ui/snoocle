"""Timing must survive a re-analysis that doesn't listen to the audio.

The regression this file pins down: a run with `scope.listen=false` hands
`mir=None` to `snap_chords`, which is a documented no-op, so the reconciler's
freshly-emitted document — correctly timing-less, because a post-pass normally
fills it — went to the store with `audio.beats` emptied, `metadata.bpm` nulled,
every placement's `beat`/`timeSeconds` gone, every section time gone, and the
intro lines collapsed. It validated and it stored: the loss was silent.

Three things are covered here, deliberately at different levels:
  - the carry-forward pass itself (pure, unit-level): what matches, what
    doesn't, and that a placement the reconciler ADDED never steals a
    neighbour's time;
  - the pipeline wiring: listen=false with a prior carries forward, listen=false
    with NO prior fails loudly, listen=true is untouched;
  - the guard: independent of both, no run may store a version that drops
    audio-derived data the prior version had.
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
from snoocle_server.timing.carry_forward import (
    ACTION,
    audio_data_lost,
    carry_forward_timing,
)

client = TestClient(app)

SONG_ID = "ccr--rain"
VIDEO_ID = "Ylo1hjRnTFI"
BPM = 117.5


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch):
    store = InMemorySongRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: store)
    monkeypatch.setattr(pipeline_mod, "get_store", lambda: store)
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    # Nothing in this file may touch the network.
    monkeypatch.setattr(pipeline_mod, "fetch_lrc", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod, "discover_sources", lambda *a, **k: [])
    return store


def _beats(count: int = 16) -> list[dict]:
    step = 60.0 / BPM
    return [
        {"time": round(i * step, 4), "measure": i // 4 + 1, "beatInMeasure": i % 4 + 1}
        for i in range(count)
    ]


def _gold_song() -> Song:
    """The version that HAD timing: two instrumental intro lines, three sung
    lines, two timed sections, a beat grid, a bpm and an analyzed video id."""
    return Song.model_validate(
        {
            "id": SONG_ID,
            "metadata": {"title": "Rain", "artist": "CCR", "bpm": BPM},
            "audio": {
                "youtubeVideoId": VIDEO_ID,
                "analyzedVideoId": VIDEO_ID,
                "beats": _beats(),
                "syncMap": [
                    {"lineIndex": 0, "time": 0.0},
                    {"lineIndex": 1, "time": 4.0},
                    {"lineIndex": 2, "time": 8.0},
                    {"lineIndex": 3, "time": 16.0},
                    {"lineIndex": 4, "time": 24.0},
                ],
            },
            "sections": [
                {
                    "sectionIndex": 0, "name": "Intro", "kind": "intro",
                    "startLineIndex": 0, "endLineIndex": 1,
                    "startTime": 0.0, "endTime": 8.0,
                },
                {
                    "sectionIndex": 1, "name": "Verse 1", "kind": "verse",
                    "startLineIndex": 2, "endLineIndex": 4,
                    "startTime": 8.0, "endTime": 30.0,
                },
            ],
            "lines": [
                {
                    "lineIndex": 0, "lyrics": "", "timeSeconds": 0.0, "confidence": 0.9,
                    "chordPlacements": [
                        {"charIndex": 0, "chord": "C", "timeSeconds": 0.0,
                         "confidence": 0.9, "beat": {"measure": 1, "beat": 1}},
                        {"charIndex": 1, "chord": "G", "timeSeconds": 2.0,
                         "confidence": 0.9, "beat": {"measure": 2, "beat": 1}},
                    ],
                },
                {
                    "lineIndex": 1, "lyrics": "", "timeSeconds": 4.0, "confidence": 0.9,
                    "chordPlacements": [
                        {"charIndex": 0, "chord": "Am", "timeSeconds": 4.0,
                         "confidence": 0.9, "beat": {"measure": 3, "beat": 1}},
                        {"charIndex": 1, "chord": "F", "timeSeconds": 6.0,
                         "confidence": 0.7},
                    ],
                },
                {
                    "lineIndex": 2, "lyrics": "Someone told me long ago",
                    "timeSeconds": 8.0, "confidence": 0.9,
                    "chordPlacements": [
                        {"charIndex": 0, "chord": "C", "timeSeconds": 8.0, "confidence": 0.9},
                        {"charIndex": 12, "chord": "Am", "timeSeconds": 12.0, "confidence": 0.9},
                    ],
                },
                {
                    "lineIndex": 3, "lyrics": "There's a calm before the storm",
                    "timeSeconds": 16.0, "confidence": 0.8,
                    "chordPlacements": [
                        {"charIndex": 0, "chord": "F", "timeSeconds": 16.0, "confidence": 0.9},
                        {"charIndex": 10, "chord": "C", "timeSeconds": 20.0, "confidence": 0.9},
                    ],
                },
                {
                    "lineIndex": 4, "lyrics": "I know it's been comin for some time",
                    "timeSeconds": 24.0, "confidence": 0.8,
                    "chordPlacements": [
                        {"charIndex": 0, "chord": "C", "timeSeconds": 24.0, "confidence": 0.9},
                        {"charIndex": 20, "chord": "G", "timeSeconds": 28.0, "confidence": 0.9},
                    ],
                },
            ],
            "provenance": [
                {"timestamp": "2026-07-01T00:00:00Z", "actor": "reconcile:test/gold",
                 "action": "reconciled"},
            ],
        }
    )


def _reconciled_without_timing() -> Song:
    """What the reconciler emits on a listen=false run: the same document,
    improved — a walkdown added on line 2 (Am/G), another on line 4 (C/B), a
    charIndex corrected on line 3, a new line appended — and every timing
    field empty, because filling those is a post-pass's job."""
    return Song.model_validate(
        {
            "id": SONG_ID,
            "metadata": {"title": "Rain", "artist": "CCR"},  # bpm: gone
            "audio": {"youtubeVideoId": VIDEO_ID},           # beats/syncMap: gone
            "sections": [
                {"sectionIndex": 0, "name": "Intro", "kind": "intro",
                 "startLineIndex": 0, "endLineIndex": 1},
                {"sectionIndex": 1, "name": "Verse 1", "kind": "verse",
                 "startLineIndex": 2, "endLineIndex": 5},
            ],
            "lines": [
                {"lineIndex": 0, "lyrics": "", "chordPlacements": [
                    {"charIndex": 0, "chord": "C"}, {"charIndex": 1, "chord": "G"}]},
                {"lineIndex": 1, "lyrics": "", "chordPlacements": [
                    {"charIndex": 0, "chord": "Am"}, {"charIndex": 1, "chord": "F"}]},
                {"lineIndex": 2, "lyrics": "Someone told me long ago",
                 "chordPlacements": [
                     {"charIndex": 0, "chord": "C"},
                     {"charIndex": 12, "chord": "Am"},
                     {"charIndex": 18, "chord": "Am/G"}]},   # ADDED
                {"lineIndex": 3, "lyrics": "There's a calm before the storm",
                 "chordPlacements": [
                     {"charIndex": 0, "chord": "F"},
                     {"charIndex": 12, "chord": "C"}]},      # charIndex CORRECTED
                {"lineIndex": 4, "lyrics": "I know it's been comin for some time",
                 "chordPlacements": [
                     {"charIndex": 0, "chord": "C"},
                     {"charIndex": 10, "chord": "C/B"},      # ADDED, between two matches
                     {"charIndex": 20, "chord": "G"}]},
                {"lineIndex": 5, "lyrics": "Yesterday and days before",  # ADDED line
                 "chordPlacements": [{"charIndex": 0, "chord": "G"}]},
            ],
            "provenance": [
                {"timestamp": "2026-07-29T00:00:00Z", "actor": "reconcile:test/fake",
                 "action": "reconciled"},
            ],
        }
    )


def _placement(song: Song, line_index: int, chord: str):
    return next(p for p in song.lines[line_index].chordPlacements if p.chord == chord)


# --- the pass itself -------------------------------------------------------------


def test_carry_forward_restores_every_kind_of_timing():
    carried, stats = carry_forward_timing(_reconciled_without_timing(), _gold_song())

    # audio-derived, copied wholesale
    assert len(carried.audio.beats) == 16
    assert carried.metadata.bpm == BPM
    assert carried.audio.analyzedVideoId == VIDEO_ID
    assert stats.beats_carried and stats.bpm_carried and stats.analyzed_video_id_carried

    # lines — including the two empty-lyric intro lines, which must keep their
    # OWN times rather than collapsing onto one.
    assert [line.timeSeconds for line in carried.lines] == [0.0, 4.0, 8.0, 16.0, 24.0, None]
    assert carried.lines[0].confidence == 0.9

    # placements: matched exactly, matched by reading order, and beat refs back
    assert _placement(carried, 0, "C").timeSeconds == 0.0
    assert _placement(carried, 0, "C").beat.measure == 1
    assert _placement(carried, 1, "F").timeSeconds == 6.0
    assert _placement(carried, 1, "F").confidence == 0.7
    # line 3's C moved from charIndex 10 to 12 — the reading-order fallback
    # still recognizes it as the same placement.
    assert _placement(carried, 3, "C").timeSeconds == 20.0

    # sections, matched on index + line range (verse 1 now ends on line 5)
    assert (carried.sections[0].startTime, carried.sections[0].endTime) == (0.0, 8.0)
    assert (carried.sections[1].startTime, carried.sections[1].endTime) == (8.0, 30.0)
    assert stats.sections_carried == 2


def test_an_added_placement_keeps_empty_timing_rather_than_stealing_one():
    """The Am/G and C/B walkdowns this run added did not exist in the prior
    version. Giving them a neighbour's timestamp would be inventing data."""
    carried, stats = carry_forward_timing(_reconciled_without_timing(), _gold_song())

    for line_index, chord in ((2, "Am/G"), (4, "C/B")):
        added = _placement(carried, line_index, chord)
        assert added.timeSeconds is None
        assert added.beat is None

    # ...and the placements around them kept their own times, unshifted.
    assert _placement(carried, 2, "Am").timeSeconds == 12.0
    assert _placement(carried, 4, "C").timeSeconds == 24.0
    assert _placement(carried, 4, "G").timeSeconds == 28.0

    # the appended line's placement is new too
    assert carried.lines[5].chordPlacements[0].timeSeconds is None
    assert stats.placements_empty == 3
    assert stats.placements_carried == 10


def test_sync_map_is_regenerated_from_the_line_times():
    carried, _ = carry_forward_timing(_reconciled_without_timing(), _gold_song())
    assert [(p.lineIndex, p.time) for p in carried.audio.syncMap] == [
        (line.lineIndex, line.timeSeconds)
        for line in carried.lines
        if line.timeSeconds is not None
    ]


def test_a_rewritten_line_does_not_inherit_a_time_but_a_typo_fix_does():
    prior = _gold_song()
    song = _reconciled_without_timing().model_copy()
    lines = [line.model_dump() for line in song.lines]
    lines[2]["lyrics"] = "Someone told me long agoo"          # typo-level edit
    lines[3]["lyrics"] = "Completely different words here"    # a different line
    lines[3]["chordPlacements"] = [{"charIndex": 0, "chord": "F"}]
    song = Song.model_validate({**song.model_dump(), "lines": lines})

    carried, _ = carry_forward_timing(song, prior)
    assert carried.lines[2].timeSeconds == 8.0
    assert carried.lines[3].timeSeconds is None
    assert carried.lines[3].chordPlacements[0].timeSeconds is None


def test_the_pass_records_its_own_provenance_in_the_snap_notes_shape():
    """The acceptance script and the evidence manifest read `timing-snap`'s
    notes/confidence; this pass must be readable by exactly the same code."""
    import re

    carried, stats = carry_forward_timing(
        _reconciled_without_timing(), _gold_song(), prior_version="cfb45e1f69e6"
    )
    entry = carried.provenance[-1]
    assert entry.action == ACTION == "timing-carry-forward"
    assert entry.sources == ["prior-version:cfb45e1f69e6"]
    assert entry.confidence == pytest.approx(stats.match_ratio)

    m = re.search(r"matched (\d+)/(\d+) chord placement", entry.notes)
    assert m and (int(m.group(1)), int(m.group(2))) == (10, 13)
    assert "3 left empty" in entry.notes
    assert "5/6 line(s) timed" in entry.notes
    assert "beats=carried" in entry.notes and "bpm=carried" in entry.notes


# (that the acceptance script's check_snap_match actually parses this entry is
# asserted in tests/test_acceptance_audio_fixtures.py, next to the script's own
# tests and using its module loader.)


# --- the pipeline wiring ---------------------------------------------------------


class _StaticReconciler:
    """Stands in for `reconcile`, always returning the timing-less document."""

    def __init__(self, song: Song):
        self.song = song
        self.calls: list[dict] = []

    def __call__(self, title, artist, candidates, mir, **kwargs):
        self.calls.append({"candidates": candidates, "mir": mir, **kwargs})
        return ReconcileResult(
            song=self.song.model_copy(deep=True), provider="anthropic", model="fake",
            attempts=1, audio_attached=False, usage={},
        )


def _analyze(**extra):
    body = {
        "title": "Rain",
        "artist": "CCR",
        "provider": "anthropic",
        "agentPolicy": "always",
    }
    body.update(extra)
    return client.post("/v1/songs/analyze", json=body)


def _no_audio(monkeypatch) -> None:
    def boom(*a, **k):  # noqa: ANN001
        raise AssertionError("audio was acquired/analyzed despite listen=false")

    monkeypatch.setattr(pipeline_mod, "acquire", boom)
    monkeypatch.setattr(pipeline_mod, "analyze_audio", boom)


def test_listen_off_carries_the_stored_versions_timing_into_the_new_one(
    monkeypatch, isolated_store
):
    """The end-to-end regression: re-analyze with listen=false and the stored
    version's timing must be in the version that comes out."""
    isolated_store.save(_gold_song(), message="gold")
    _no_audio(monkeypatch)
    monkeypatch.setattr(
        pipeline_mod, "reconcile", _StaticReconciler(_reconciled_without_timing())
    )

    r = _analyze(scope={"listen": False, "reconcile": True})
    assert r.status_code == 200, r.text
    assert r.json()["steps"]["timing"].startswith("ok: 10/13 placement time(s)")

    stored = isolated_store.get(SONG_ID)
    assert len(stored.audio.beats) == 16
    assert stored.metadata.bpm == BPM
    assert stored.audio.analyzedVideoId == VIDEO_ID
    assert [line.timeSeconds for line in stored.lines] == [0.0, 4.0, 8.0, 16.0, 24.0, None]
    assert [(s.startTime, s.endTime) for s in stored.sections] == [(0.0, 8.0), (8.0, 30.0)]
    assert _placement(stored, 3, "C").timeSeconds == 20.0
    assert _placement(stored, 2, "Am/G").timeSeconds is None
    assert [(p.lineIndex, p.time) for p in stored.audio.syncMap] == [
        (line.lineIndex, line.timeSeconds)
        for line in stored.lines
        if line.timeSeconds is not None
    ]
    assert [p.action for p in stored.provenance].count(ACTION) == 1


def test_listen_off_can_carry_forward_from_a_request_supplied_prior(monkeypatch):
    """The UI's "re-run with my fixes" flow sends the edited document as
    `priorSong` and there may be nothing else — that document is the timing
    source too."""
    _no_audio(monkeypatch)
    monkeypatch.setattr(
        pipeline_mod, "reconcile", _StaticReconciler(_reconciled_without_timing())
    )

    r = _analyze(
        scope={"listen": False, "reconcile": True},
        priorSong=_gold_song().model_dump(mode="json"),
    )
    assert r.status_code == 200, r.text
    song = Song.model_validate(r.json()["song"])
    assert len(song.audio.beats) == 16 and song.metadata.bpm == BPM
    assert song.lines[2].timeSeconds == 8.0


def test_listen_off_with_no_prior_version_fails_instead_of_storing(monkeypatch):
    """Nothing to reuse means the request is incoherent, not "store it empty".
    It fails BEFORE reconciliation — paying for a run whose result could not
    be stored would be the expensive way to say the same thing."""
    _no_audio(monkeypatch)

    def never(*a, **k):  # noqa: ANN001
        raise AssertionError("reconcile ran for a request that cannot be stored")

    monkeypatch.setattr(pipeline_mod, "reconcile", never)

    r = _analyze(scope={"listen": False, "reconcile": True})
    assert r.status_code == 502, r.text
    detail = r.json()["detail"]
    assert detail.startswith("timing: ")
    assert "no prior version" in detail
    assert "listen=true" in detail and "allowTimingLoss" in detail
    assert r.json()["errorCode"] == "no_prior_timing_to_carry_forward"
    assert SONG_ID not in client.get("/v1/songs").json()["songs"]


def test_listen_off_with_no_prior_version_is_allowed_with_the_override(monkeypatch):
    _no_audio(monkeypatch)
    monkeypatch.setattr(
        pipeline_mod, "reconcile", _StaticReconciler(_reconciled_without_timing())
    )

    r = _analyze(scope={"listen": False, "reconcile": True}, allowTimingLoss=True)
    assert r.status_code == 200, r.text
    assert r.json()["steps"]["timing"] == "skipped (no MIR)"


def test_listen_off_does_not_let_lrclib_overwrite_the_carried_timing(monkeypatch):
    """apply_lrc replaces line times and re-derives every chord time by
    proportional redistribution — with no MIR it cannot even snap them back to
    a beat grid. "Reuse the existing analysis" means reuse it."""
    _no_audio(monkeypatch)
    monkeypatch.setattr(
        pipeline_mod, "reconcile", _StaticReconciler(_reconciled_without_timing())
    )

    def boom(*a, **k):  # noqa: ANN001
        raise AssertionError("LRCLIB was consulted on a carry-forward run")

    monkeypatch.setattr(pipeline_mod, "fetch_lrc", boom)

    r = _analyze(
        scope={"listen": False, "reconcile": True},
        priorSong=_gold_song().model_dump(mode="json"),
    )
    assert r.status_code == 200, r.text
    assert r.json()["steps"]["lrc"].startswith("skipped (scope: timing carried forward")


def test_listen_on_still_takes_the_snap_chords_path(monkeypatch, isolated_store):
    """The carry-forward pass must be invisible to a run that listens: the
    snap path, its step text, and its provenance action are unchanged."""
    isolated_store.save(_gold_song(), message="gold")
    monkeypatch.setattr(
        pipeline_mod, "reconcile", _StaticReconciler(_reconciled_without_timing())
    )

    snapped: list[str] = []

    def fake_snap(song, mir):
        snapped.append("ran")
        return song.model_copy(
            update={
                "metadata": song.metadata.model_copy(update={"bpm": BPM}),
                "audio": song.audio.model_copy(
                    update={"beats": _gold_song().audio.beats}
                ),
            }
        )

    monkeypatch.setattr(pipeline_mod, "snap_chords", fake_snap)

    r = _analyze(skipAudio=True)
    assert r.status_code == 200, r.text
    assert snapped == ["ran"]
    assert ACTION not in [p["action"] for p in r.json()["song"]["provenance"]]


# --- the guard -------------------------------------------------------------------


def test_audio_data_lost_names_only_what_actually_regressed():
    gold = _gold_song()
    assert audio_data_lost(gold, gold) == []

    stripped = Song.model_validate(
        {
            **gold.model_dump(),
            "metadata": {**gold.metadata.model_dump(), "bpm": None},
            "audio": {**gold.audio.model_dump(), "beats": []},
        }
    )
    lost = audio_data_lost(gold, stripped)
    assert len(lost) == 2
    assert "audio.beats (16 entries -> 0)" in lost
    assert "metadata.bpm (117.5 -> null)" in lost
    # A song that never had them can't lose them.
    assert audio_data_lost(stripped, stripped) == []


def test_the_guard_fires_on_a_regression_reached_by_some_other_path(
    monkeypatch, isolated_store
):
    """listen=true, so no carry-forward runs — this is the "whatever path got
    there" case: MIR was unavailable, snap_chords was a no-op, and the run is
    about to commit a version with the beat grid and bpm gone."""
    isolated_store.save(_gold_song(), message="gold")
    monkeypatch.setattr(
        pipeline_mod, "reconcile", _StaticReconciler(_reconciled_without_timing())
    )

    r = _analyze(skipAudio=True)
    assert r.status_code == 502, r.text
    detail = r.json()["detail"]
    assert detail.startswith("timing-guard: ")
    assert "audio.beats (16 entries -> 0)" in detail
    assert "metadata.bpm (117.5 -> null)" in detail
    assert r.json()["errorCode"] == "timing_data_loss"
    # nothing was written: the stored version is still the gold one
    assert len(isolated_store.get(SONG_ID).audio.beats) == 16
    assert len(isolated_store.versions(SONG_ID)) == 1


def test_the_guard_is_bypassable_by_the_override_flag(monkeypatch, isolated_store):
    isolated_store.save(_gold_song(), message="gold")
    monkeypatch.setattr(
        pipeline_mod, "reconcile", _StaticReconciler(_reconciled_without_timing())
    )

    r = _analyze(skipAudio=True, allowTimingLoss=True)
    assert r.status_code == 200, r.text
    assert r.json()["steps"]["timing-guard"].startswith("overridden: dropping audio.beats")
    assert isolated_store.get(SONG_ID).audio.beats == []


def test_the_guard_reports_when_it_had_something_to_protect(monkeypatch, isolated_store):
    """A passing guard is worth saying out loud — "the beat grid survived this
    run" is the whole point, and silence reads the same as not running."""
    isolated_store.save(_gold_song(), message="gold")
    monkeypatch.setattr(pipeline_mod, "acquire", lambda *a, **k: None)
    monkeypatch.setattr(
        pipeline_mod, "reconcile", _StaticReconciler(_reconciled_without_timing())
    )

    r = _analyze(
        scope={"listen": False, "reconcile": True},
        priorSong=_gold_song().model_dump(mode="json"),
    )
    assert r.status_code == 200, r.text
    assert r.json()["steps"]["timing-guard"] == "ok: audio.beats and metadata.bpm preserved"


def test_a_first_analysis_has_nothing_to_guard(monkeypatch):
    """No prior version, no baseline, no guard step — a song analyzed for the
    first time must not be told it lost data it never had."""
    monkeypatch.setattr(
        pipeline_mod, "reconcile", _StaticReconciler(_reconciled_without_timing())
    )

    r = _analyze(skipAudio=True)
    assert r.status_code == 200, r.text
    assert "timing-guard" not in r.json()["steps"]
