"""The model is never the authority on timing.

The invariant this file pins down: every deterministic timing pass guards its
writes with "is this field empty?" (``timing/snap.py:303,308``,
``timing/carry_forward.py:272,313,324``) — and model-supplied timing is never
empty. So the reconciler's OWN output silently outranked every measured or
carried-forward value, and nothing downstream could tell.

The reported symptom, exactly: on a notes-only run the model is handed
``priorHumanEditedSong`` and told to "return it with the notes applied and
nothing else changed" (``reconcile/anthropic_agent.py``), so it re-emits every
``timeSeconds``. Carry-forward then matched **0/N** placements — its guard is
``placement.timeSeconds is None`` — and because confidence is only written
inside that same skipped block, every placement came out with no confidence at
all, which ``timing/confidence.py`` then stamped with the neutral 0.5. The
operator read "matched 0/59" next to 59 placements timed at 0.5 and could not
tell that one caused the other.

``reconcile/engine.py`` strips model-supplied timing in ``_finalize``, before
any pass runs. What it strips depends on what will refill it — the matrix is
asserted field by field below, including the fields deliberately KEPT because
nothing else on that path can write them.

WHO will refill it is the CALLER's declaration (``TimingAuthority``), never
inferred from ``mir is not None``: a MIR in the inputs is a fact about this
run's evidence and says nothing about whether a pass is going to run. Two
public entry points — ``POST /v1/reconcile`` and MCP ``reconcile_song`` — hand
a MIR in and then return the document with no timing pass at all, so the
default is ``NONE`` (strip nothing) and only a caller that OWNS a pass may ask
for the fields that pass writes to be cleared. A declared pass that then skips
or raises is undone by the caller (``TimingStrip.restore``), so a best-effort
failure can never become a fatal ``timing_data_loss``.
"""

from __future__ import annotations

import json

import pytest

from snoocle_server import pipeline as pipeline_mod
from snoocle_server.audio.acquire import AcquiredAudio
from snoocle_server.config import settings
from snoocle_server.discovery.cache import DiscoveryCacheInfo
from snoocle_server.discovery.models import CandidateSource
from snoocle_server.mir.base import Beat, ChordSegment, MirAnalysis
from snoocle_server.mir.cache import MirCacheInfo
from snoocle_server.reconcile import TimingAuthority, reconcile
from snoocle_server.reconcile.engine import (
    TIMING_RESTORE_ACTION,
    TIMING_STRIP_ACTION,
)
from snoocle_server.reconcile.providers import LLMProvider, LLMResponse
from snoocle_server.reconcile.trace import start_run
from snoocle_server.schema import Song
from snoocle_server.scope import AnalysisScope
from snoocle_server.store.memory import InMemorySongRepository
from snoocle_server.store.song_notes import reset_song_notes_store
from snoocle_server.timing.carry_forward import ACTION as CARRY_FORWARD_ACTION
from snoocle_server.timing.carry_forward import carry_forward_timing
from snoocle_server.timing.confidence import score_song
from snoocle_server.timing.snap import snap_chords

TITLE = "Rain"
ARTIST = "CCR"
SONG_ID = "ccr--rain"
VIDEO_ID = "Ylo1hjRnTFI"
BPM = 117.5

NOTES_ONLY = AnalysisScope(listen=False, reconcile=False)
FULL = AnalysisScope(listen=True, reconcile=True)


def _beats(count: int = 16) -> list[dict]:
    step = 60.0 / BPM
    return [
        {"time": round(i * step, 4), "measure": i // 4 + 1, "beatInMeasure": i % 4 + 1}
        for i in range(count)
    ]


def _timed_song() -> dict:
    """The prior version: fully timed, 6 placements, 3 lines, 2 sections, a
    beat grid, a bpm and a syncMap. This is also — verbatim — what the model
    echoes back on a notes-only run."""
    return {
        "id": SONG_ID,
        "metadata": {"title": TITLE, "artist": ARTIST, "bpm": BPM},
        "audio": {
            "youtubeVideoId": VIDEO_ID,
            "analyzedVideoId": VIDEO_ID,
            "beats": _beats(),
            "syncMap": [
                {"lineIndex": 0, "time": 0.0},
                {"lineIndex": 1, "time": 8.0},
                {"lineIndex": 2, "time": 16.0},
            ],
        },
        "sections": [
            {"sectionIndex": 0, "name": "Verse 1", "kind": "verse",
             "startLineIndex": 0, "endLineIndex": 1, "startTime": 0.0, "endTime": 16.0},
            {"sectionIndex": 1, "name": "Chorus", "kind": "chorus",
             "startLineIndex": 2, "endLineIndex": 2, "startTime": 16.0, "endTime": 24.0},
        ],
        "lines": [
            {"lineIndex": 0, "lyrics": "Someone told me long ago",
             "timeSeconds": 0.0, "confidence": 0.9,
             "chordPlacements": [
                 {"charIndex": 0, "chord": "C", "timeSeconds": 0.0, "confidence": 0.9,
                  "beat": {"measure": 1, "beat": 1}},
                 {"charIndex": 12, "chord": "Am", "timeSeconds": 4.0, "confidence": 0.9}]},
            {"lineIndex": 1, "lyrics": "There's a calm before the storm",
             "timeSeconds": 8.0, "confidence": 0.8,
             "chordPlacements": [
                 {"charIndex": 0, "chord": "F", "timeSeconds": 8.0, "confidence": 0.9},
                 {"charIndex": 10, "chord": "C", "timeSeconds": 12.0, "confidence": 0.7}]},
            {"lineIndex": 2, "lyrics": "I know it's been comin for some time",
             "timeSeconds": 16.0, "confidence": 0.8,
             "chordPlacements": [
                 {"charIndex": 0, "chord": "C", "timeSeconds": 16.0, "confidence": 0.9},
                 {"charIndex": 20, "chord": "G", "timeSeconds": 20.0, "confidence": 0.9}]},
        ],
        "provenance": [
            {"timestamp": "2026-07-01T00:00:00Z", "actor": "reconcile:test/gold",
             "action": "reconciled"},
        ],
    }


def _mir(*, beats: bool = True, bpm: float | None = 90.0) -> MirAnalysis:
    """A measured analysis that DISAGREES with the model's numbers, so which
    one won is never ambiguous."""
    return MirAnalysis(
        engines={"beats": "madmom", "chords": "chord-cnn-lstm"},
        duration_seconds=200.0,
        bpm=bpm,
        time_signature="4/4",
        key="C major",
        beats=[Beat(time=i * 0.5, position=(i % 4) + 1) for i in range(64)] if beats else [],
        chords=[
            ChordSegment(start=1.0, end=5.0, chord="C"),
            ChordSegment(start=5.0, end=9.0, chord="Am"),
            ChordSegment(start=9.0, end=13.0, chord="F"),
            ChordSegment(start=13.0, end=17.0, chord="C"),
            ChordSegment(start=17.0, end=21.0, chord="C"),
            ChordSegment(start=21.0, end=25.0, chord="G"),
        ],
    )


class _Echo(LLMProvider):
    """The reported model behaviour: hand it ``priorHumanEditedSong`` and it
    returns that document verbatim — every ``timeSeconds`` included."""

    name = "test-echo"
    default_model = "test-echo-1"
    wants_context = True
    context: dict | None = None
    trace = None

    def complete(self, system, turns, model=None, max_tokens=None, audio=None):
        prior = (self.context or {}).get("prior_song")
        assert prior is not None, "the engine did not hand the prior document to the model"
        return LLMResponse(
            text=json.dumps(prior), provider=self.name, model=self.default_model
        )


class _Static(LLMProvider):
    """Returns one fixed document, whatever it was asked."""

    name = "test-static"
    default_model = "test-static-1"

    def __init__(self, document: dict):
        self.document = document

    def complete(self, system, turns, model=None, max_tokens=None, audio=None):
        return LLMResponse(
            text=json.dumps(self.document), provider=self.name, model=self.default_model
        )


def _use(monkeypatch, provider: LLMProvider) -> LLMProvider:
    monkeypatch.setattr(
        "snoocle_server.reconcile.engine.get_provider", lambda name=None: provider
    )
    return provider


def _strip_entry(song: Song):
    return next((p for p in song.provenance if p.action == TIMING_STRIP_ACTION), None)


def _placements(song: Song):
    return [p for line in song.lines for p in line.chordPlacements]


# --- the headline fix: a notes-only run that echoes the prior's timing -----------


@pytest.fixture()
def store(monkeypatch):
    repo = InMemorySongRepository()
    monkeypatch.setattr(pipeline_mod, "get_store", lambda: repo)
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(pipeline_mod, "fetch_lrc", lambda *a, **k: None)
    for name in ("discover_sources", "acquire", "analyze_audio"):
        monkeypatch.setattr(
            pipeline_mod, name,
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("a notes-only run gathered evidence")
            ),
        )
    reset_song_notes_store()
    yield repo
    reset_song_notes_store()


def test_a_notes_only_run_that_echoes_the_priors_timing_now_carries_forward_n_of_n(
    monkeypatch, store
):
    """The reported run, end to end: 6 placements, the model re-emits all 6
    times, and carry-forward must match 6/6 — not 0/6 — with nothing left on
    the neutral 0.5 that a missing confidence turns into."""
    store.save(Song.model_validate(_timed_song()), message="gold")
    _use(monkeypatch, _Echo())

    report = pipeline_mod.run_pipeline(
        TITLE, ARTIST, provider="anthropic", scope=NOTES_ONLY,
        guidance="the Am in line 1 should be an Am7",
    )

    assert report.steps["timing"].startswith(
        "ok: 6/6 placement time(s), 3/3 line time(s), 2/2 section time(s) carried forward"
    )
    stored = store.get(SONG_ID)

    # ...and the times are the PRIOR's, carried, not the model's echo passed
    # through: same values, but now written by the pass that owns them, which
    # is what the provenance and the match count report.
    assert [line.timeSeconds for line in stored.lines] == [0.0, 8.0, 16.0]
    assert [p.timeSeconds for p in _placements(stored)] == [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    assert [p.confidence for p in _placements(stored)] == [0.9, 0.9, 0.9, 0.7, 0.9, 0.9]
    assert 0.5 not in [p.confidence for p in _placements(stored)]
    assert _placements(stored)[0].beat.measure == 1

    # section spans: stripping them is what ENABLES the carry (see
    # timing/carry_forward.py::_carry_sections, which needs both times None)
    assert [(s.startTime, s.endTime) for s in stored.sections] == [(0.0, 16.0), (16.0, 24.0)]

    # the recording-level fields came back from the prior too
    assert stored.metadata.bpm == BPM
    assert len(stored.audio.beats) == 16
    assert [(p.lineIndex, p.time) for p in stored.audio.syncMap] == [
        (0, 0.0), (1, 8.0), (2, 16.0)
    ]

    actions = [p.action for p in stored.provenance]
    assert actions.index(TIMING_STRIP_ACTION) < actions.index(CARRY_FORWARD_ACTION)
    notes = _strip_entry(stored).notes
    for field in (
        "chordPlacement.timeSeconds/confidence/beat (6)",
        "line.timeSeconds/confidence (3)",
        "section.startTime/endTime (2)",
        "audio.beats (16 entries)",
        "audio.syncMap (3 points)",
        f"metadata.bpm ({BPM})",
    ):
        assert field in notes, notes
    assert "timing.carry_forward" in notes


def test_without_the_strip_the_same_echo_matches_nothing_and_lands_on_0_5():
    """The regression itself, reproduced against the passes directly: feed
    carry-forward a document that still carries the model's echoed times and
    it matches 0/6, leaves every confidence empty, and the confidence pass
    stamps the neutral 0.5 — the exact pair of symptoms from the report."""
    echoed = Song.model_validate(_timed_song())
    # the model's echo, with confidence dropped the way a re-emitted document
    # loses it, but timeSeconds kept — which is all the guard looks at
    echoed = Song.model_validate(
        {
            **echoed.model_dump(),
            "lines": [
                {
                    **line.model_dump(),
                    "confidence": None,
                    "chordPlacements": [
                        {**p.model_dump(), "confidence": None}
                        for p in line.chordPlacements
                    ],
                }
                for line in echoed.lines
            ],
        }
    )

    carried, stats = carry_forward_timing(echoed, Song.model_validate(_timed_song()))
    assert (stats.placements_carried, stats.placements_total) == (0, 6)
    assert stats.sections_carried == 0
    assert all(p.confidence is None for p in _placements(carried))

    scored, _ = score_song(carried, [], None)
    assert [p.confidence for p in _placements(scored)] == [0.5] * 6


# --- the matrix, field by field, per branch --------------------------------------


def test_notes_only_strips_every_timing_field_the_carry_forward_pass_owns(monkeypatch):
    _use(monkeypatch, _Static(_timed_song()))
    song = reconcile(
        TITLE, ARTIST, [], None, provider_name="test-static", song_id=SONG_ID,
        prior_song=_timed_song(), scope=NOTES_ONLY,
        timing_authority=TimingAuthority.CARRY_FORWARD,
    ).song

    assert [line.timeSeconds for line in song.lines] == [None] * 3
    assert [line.confidence for line in song.lines] == [None] * 3
    assert all(
        p.timeSeconds is None and p.confidence is None and p.beat is None
        for p in _placements(song)
    )
    assert [(s.startTime, s.endTime) for s in song.sections] == [(None, None)] * 2
    assert song.audio.beats == [] and song.audio.syncMap == []
    assert song.metadata.bpm is None
    # not timing: which upload the times were measured against is still true
    assert song.audio.analyzedVideoId == VIDEO_ID
    assert _strip_entry(song) is not None


def test_a_declared_snap_strips_only_the_recording_level_fields_it_will_refill(monkeypatch):
    """A caller that DECLARES ``timing.snap`` and runs it: the model's junk grid
    must not block the measured one. But per-element times and section spans
    stay — ``snap_chords`` overwrites placement/line times only when SOMETHING
    matched (nothing at all is written on a zero-match run), and on Mode A
    nothing can write section spans at all
    (``timing/realign.retime_sections`` is reachable only from ``realign.py``),
    so stripping those would just lower coverage.

    The declaration and the pass are one unit: the snap call below is not a
    convenience, it is the second half of what
    ``timing_authority=TimingAuthority.SNAP`` promised. A caller that cannot
    make that call must not make that declaration — see
    ``test_a_reconcile_that_runs_no_timing_pass_strips_nothing_even_with_a_mir``.
    """
    junk = _timed_song()
    junk["metadata"]["bpm"] = 1.0                                     # junk bpm
    junk["audio"]["beats"] = [{"time": 0.0, "measure": 1, "beatInMeasure": 1}]  # junk grid
    mir = _mir()
    _use(monkeypatch, _Static(junk))

    song = reconcile(
        TITLE, ARTIST, [], mir, provider_name="test-static", song_id=SONG_ID,
        scope=FULL, timing_authority=TimingAuthority.SNAP,
    ).song

    # stripped: the three the model's values would have BLOCKED
    assert song.metadata.bpm is None
    assert song.audio.beats == []
    assert song.audio.syncMap == []
    # kept: nothing else on this path writes them
    assert [(s.startTime, s.endTime) for s in song.sections] == [(0.0, 16.0), (16.0, 24.0)]
    assert [line.timeSeconds for line in song.lines] == [0.0, 8.0, 16.0]
    assert [p.timeSeconds for p in _placements(song)] == [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]

    # ...and the declared pass, run as the declaring caller runs it, actually
    # lands the measured grid — which is the point of the strip
    snapped = snap_chords(song, mir)
    assert snapped.metadata.bpm == 90.0
    assert len(snapped.audio.beats) == 64
    assert [p.timeSeconds for p in _placements(snapped)] == [1.0, 5.0, 9.0, 13.0, 17.0, 21.0]
    assert snapped.audio.syncMap


def test_a_declared_snap_keeps_beats_and_bpm_this_runs_mir_cannot_supply(monkeypatch):
    """Strip only what something will refill: a MIR with no beat grid and no
    bpm gives ``snap_chords`` nothing to write there (``build_beat_grid``
    returns [] and the bpm write is ``if ... and mir.bpm``), so emptying those
    fields would be a pure loss — and would trip the pre-store audio-data
    guard against a prior version that had them."""
    _use(monkeypatch, _Static(_timed_song()))
    song = reconcile(
        TITLE, ARTIST, [], _mir(beats=False, bpm=None), provider_name="test-static",
        song_id=SONG_ID, scope=FULL, timing_authority=TimingAuthority.SNAP,
    ).song

    assert len(song.audio.beats) == 16
    assert song.metadata.bpm == BPM
    # the syncMap still goes: snap_chords regenerates it unconditionally
    assert song.audio.syncMap == []
    assert "audio.beats" not in _strip_entry(song).notes


def test_a_reconcile_that_runs_no_timing_pass_strips_nothing_even_with_a_mir(monkeypatch):
    """The default is the SAFE answer. A caller that declares no authority runs
    no pass by definition, so its document keeps every field the model sent —
    whatever its inputs looked like, MIR and scope included. This is the engine
    half of the ``POST /v1/reconcile`` / MCP ``reconcile_song`` regression; the
    entry points themselves are asserted further down."""
    _use(monkeypatch, _Static(_timed_song()))
    for mir in (None, _mir()):
        for scope in (None, FULL, NOTES_ONLY, AnalysisScope(listen=True, reconcile=False)):
            song = reconcile(
                TITLE, ARTIST, [], mir, provider_name="test-static", song_id=SONG_ID,
                prior_song=_timed_song(), scope=scope,
            ).song
            assert song.metadata.bpm == BPM
            assert len(song.audio.beats) == 16
            assert song.audio.syncMap
            assert [line.timeSeconds for line in song.lines] == [0.0, 8.0, 16.0]
            assert [(s.startTime, s.endTime) for s in song.sections] == [
                (0.0, 16.0), (16.0, 24.0)
            ]
            assert _strip_entry(song) is None


def test_a_declared_snap_with_no_mir_strips_nothing(monkeypatch):
    """``snap_chords`` returns the document untouched when ``mir is None``
    (``timing/snap.py:250``), so a declaration it cannot honour is not an
    authority — this is the ``listen=off`` + no prior + ``allowTimingLoss``
    shape at the engine level, and nothing may be taken from it."""
    _use(monkeypatch, _Static(_timed_song()))
    song = reconcile(
        TITLE, ARTIST, [], None, provider_name="test-static", song_id=SONG_ID,
        prior_song=_timed_song(), scope=FULL, timing_authority=TimingAuthority.SNAP,
    ).song
    assert song.metadata.bpm == BPM
    assert len(song.audio.beats) == 16
    assert song.audio.syncMap
    assert _strip_entry(song) is None


def test_the_patch_path_strips_nothing(monkeypatch):
    """A patch copies the prior document through verbatim and the pipeline
    skips every timing pass for it — a strip here would empty timing with
    nothing left to refill it, so the patch path overrides even an explicit
    declaration."""

    class _Patcher(LLMProvider):
        name = "test-patch"
        default_model = "test-patch-1"
        supports_patch_ops = True

        def complete(self, system, turns, model=None, max_tokens=None, audio=None):
            return LLMResponse(
                text=json.dumps({"ops": [{"op": "replace_chord", "lineIndex": 1,
                                          "charIndex": 10, "from": "C", "to": "Am"}]}),
                provider=self.name, model=self.default_model,
            )

    _use(monkeypatch, _Patcher())
    result = reconcile(
        TITLE, ARTIST, [], None, provider_name="test-patch", song_id=SONG_ID,
        prior_song=_timed_song(), scope=NOTES_ONLY, patch_ops_eligible=True,
        guidance="line 1's second chord is an Am",
        timing_authority=TimingAuthority.CARRY_FORWARD,
    )
    assert result.patch_ops_applied == 1
    song = result.song
    assert song.metadata.bpm == BPM and len(song.audio.beats) == 16
    assert [line.timeSeconds for line in song.lines] == [0.0, 8.0, 16.0]
    assert [p.timeSeconds for p in _placements(song)] == [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    assert [(s.startTime, s.endTime) for s in song.sections] == [(0.0, 16.0), (16.0, 24.0)]
    assert _strip_entry(song) is None


# --- observability: a silent strip would only move the invisible write -----------


def test_the_strip_is_recorded_on_the_run_trace_as_well_as_in_provenance(monkeypatch):
    _use(monkeypatch, _Static(_timed_song()))
    recorder = start_run(SONG_ID, "test-static", "standard")
    song = reconcile(
        TITLE, ARTIST, [], None, provider_name="test-static", song_id=SONG_ID,
        prior_song=_timed_song(), scope=NOTES_ONLY, trace=recorder,
        timing_authority=TimingAuthority.CARRY_FORWARD,
    ).song

    step = next(s for s in recorder.trace.steps if s.label == TIMING_STRIP_ACTION)
    assert "timing.carry_forward" in step.summary
    assert "metadata.bpm (117.5)" in step.detail["fields"]
    entry = _strip_entry(song)
    assert entry.actor.startswith("snoocle-server/")  # the SERVER did this, not the model
    assert entry.confidence is None
    assert "the model is not the authority on timing" in entry.notes


# --- defect 1: the entry points that run NO timing pass --------------------------
#
# The strip's precondition ("a deterministic pass is about to time this") is a
# property of the CALLER. These two hand a MIR in and then return the document,
# so nothing may be taken from it — and nothing may be written into provenance
# naming an authority that never ran.


def test_post_v1_reconcile_preserves_the_bpm_and_the_sync_map(monkeypatch):
    """``POST /v1/reconcile`` runs no timing pass: it validates, finalizes and
    returns. Inferring an authority from the request's MIR emptied
    ``metadata.bpm`` and ``audio.syncMap`` for a ``timing.snap`` that never
    executed — and ``audio.syncMap`` is what the player scrolls from."""
    from fastapi.testclient import TestClient

    from snoocle_server.api import app

    _use(monkeypatch, _Static(_timed_song()))
    response = TestClient(app).post(
        "/v1/reconcile",
        json={
            "title": TITLE,
            "artist": ARTIST,
            "candidates": [],
            "mir": _mir().model_dump(mode="json"),
            "provider": "test-static",
        },
    )
    assert response.status_code == 200, response.text
    song = response.json()["song"]
    assert song["metadata"]["bpm"] == BPM
    assert [(p["lineIndex"], p["time"]) for p in song["audio"]["syncMap"]] == [
        (0, 0.0), (1, 8.0), (2, 16.0)
    ]
    assert len(song["audio"]["beats"]) == 16
    assert [line["timeSeconds"] for line in song["lines"]] == [0.0, 8.0, 16.0]
    assert TIMING_STRIP_ACTION not in [p["action"] for p in song["provenance"]]


def test_mcp_reconcile_song_preserves_the_bpm_and_the_sync_map(monkeypatch):
    """Same for the MCP tool, which its own docstring says does not persist —
    the agent workflow is ``discover_song -> acquire_audio -> analyze_audio ->
    reconcile_song``, and ``save_song`` applies no timing guard, so a stripped
    document here is what gets stored."""
    from snoocle_server import mcp_server

    _use(monkeypatch, _Static(_timed_song()))
    out = mcp_server.reconcile_song(
        TITLE, ARTIST,
        candidates_json="[]",
        mir_json=json.dumps(_mir().model_dump(mode="json")),
        provider="test-static",
    )
    song = out["song"]
    assert song["metadata"]["bpm"] == BPM
    assert len(song["audio"]["syncMap"]) == 3
    assert len(song["audio"]["beats"]) == 16
    assert TIMING_STRIP_ACTION not in [p["action"] for p in song["provenance"]]


# --- defect 2: the branch condition must be the one the pipeline branches on -----


def _candidate() -> CandidateSource:
    """One text source, so a run with no MIR and no prior still has something to
    reconcile (reconcile/engine.py refuses an empty run for a real provider)."""
    return CandidateSource(
        sourceId="web-1",
        url="https://example.test/web-1",
        lines=[
            Song.model_validate(_timed_song()).lines[i].model_copy(
                update={"timeSeconds": None, "confidence": None}
            )
            for i in range(3)
        ],
    )


def _no_gathering(monkeypatch, *, candidates: list[CandidateSource] | None = None) -> None:
    """Discovery answers without the network; acquire/MIR must not be reached."""
    monkeypatch.setattr(
        pipeline_mod, "_step_discover",
        lambda *a, **k: (
            list(candidates or []),
            DiscoveryCacheInfo(status="miss", gathered_at="2026-07-30T00:00:00Z"),
        ),
    )


def _with_audio(monkeypatch, mir: MirAnalysis) -> None:
    """Give the run an acquired recording and an analysis of it, offline."""
    monkeypatch.setattr(
        pipeline_mod, "_step_acquire",
        lambda *a, **k: AcquiredAudio(
            path="/dev/null", video_id=VIDEO_ID, video_title="X", duration_seconds=200.0,
        ),
    )
    monkeypatch.setattr(
        pipeline_mod, "_step_mir",
        lambda *a, **k: (
            mir, MirCacheInfo(status="miss", analyzed_at="2026-07-30T00:00:00Z")
        ),
    )


def test_listen_off_with_no_prior_and_allow_timing_loss_strips_nothing(
    monkeypatch, store
):
    """The exact combination the ``no_prior_timing_to_carry_forward`` guard
    tells callers to use (``pipeline.py``'s "set allowTimingLoss=true"). It
    falls through to the SNAP branch — where it has no MIR either, because
    ``listen=off`` skipped the analysis — so ``timing: skipped (no MIR)`` and
    NOTHING times the document. Branching the strip on ``not scope.listen``
    alone emptied every timing field here and credited
    ``timing.carry_forward`` with deciding them."""
    monkeypatch.setattr(settings, "quality_enabled", False)
    _no_gathering(monkeypatch, candidates=[_candidate()])
    _use(monkeypatch, _Static(_timed_song()))

    report = pipeline_mod.run_pipeline(
        TITLE, ARTIST, provider="anthropic",
        scope=AnalysisScope(listen=False, reconcile=True),
        allow_timing_loss=True,
    )

    assert report.steps["timing"] == "skipped (no MIR)"
    stored = store.get(SONG_ID)
    assert stored.metadata.bpm == BPM
    assert len(stored.audio.beats) == 16
    assert len(stored.audio.syncMap) == 3
    assert [line.timeSeconds for line in stored.lines] == [0.0, 8.0, 16.0]
    assert [p.timeSeconds for p in _placements(stored)] == [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    assert [(s.startTime, s.endTime) for s in stored.sections] == [(0.0, 16.0), (16.0, 24.0)]
    # and no provenance entry naming an authority that never executed
    assert _strip_entry(stored) is None


def test_a_snap_failure_stays_non_fatal_instead_of_becoming_timing_data_loss(
    monkeypatch, store
):
    """``snap_chords`` is best-effort by design ("a failure here must never
    block storing an otherwise-good song"). After the strip it is the only thing
    that can put ``audio.beats``/``metadata.bpm`` back, so its failure would
    empty fields the prior version had and trip the FATAL pre-store guard. The
    pass that owed them didn't run, so the strip is undone instead."""
    monkeypatch.setattr(settings, "quality_enabled", False)
    store.save(Song.model_validate(_timed_song()), message="gold")
    _no_gathering(monkeypatch)
    _with_audio(monkeypatch, _mir())
    _use(monkeypatch, _Static(_timed_song()))

    def _boom(song, mir):
        raise RuntimeError("beat grid blew up")

    monkeypatch.setattr(pipeline_mod, "snap_chords", _boom)

    report = pipeline_mod.run_pipeline(TITLE, ARTIST, provider="anthropic")

    assert report.steps["timing"].startswith("failed: beat grid blew up")
    assert "restored audio.beats (16 entries)" in report.steps["timing"]
    assert "metadata.bpm (117.5)" in report.steps["timing"]
    # the guard the strip used to weaponize now has nothing to complain about
    assert report.steps["timing-guard"] == "ok: audio.beats and metadata.bpm preserved"
    assert report.stored_version

    stored = store.get(SONG_ID)
    assert stored.metadata.bpm == BPM
    assert len(stored.audio.beats) == 16
    assert len(stored.audio.syncMap) == 3
    # the history says both halves: what was taken, and that nobody measured it
    restore = next(p for p in stored.provenance if p.action == TIMING_RESTORE_ACTION)
    assert "timing.snap" in restore.notes and "did not time this document" in restore.notes
    assert "nothing measured them on this run" in restore.notes


def test_a_snap_failure_with_no_prior_version_still_keeps_what_the_model_sent(
    monkeypatch, store
):
    """No prior means no pre-store guard to trip — and the fields must still
    come back, or a best-effort failure would silently produce a bpm-less,
    syncMap-less first version."""
    monkeypatch.setattr(settings, "quality_enabled", False)
    _no_gathering(monkeypatch)
    _with_audio(monkeypatch, _mir())
    _use(monkeypatch, _Static(_timed_song()))
    monkeypatch.setattr(
        pipeline_mod, "snap_chords",
        lambda song, mir: (_ for _ in ()).throw(RuntimeError("nope")),
    )

    pipeline_mod.run_pipeline(TITLE, ARTIST, provider="anthropic")

    stored = store.get(SONG_ID)
    assert stored.metadata.bpm == BPM
    assert len(stored.audio.beats) == 16
    assert len(stored.audio.syncMap) == 3


def test_a_successful_snap_is_not_second_guessed(monkeypatch, store):
    """The undo is for a pass that did NOT run. When snap runs, the measured
    grid stands and the model's numbers stay gone — otherwise the strip would
    achieve nothing."""
    monkeypatch.setattr(settings, "quality_enabled", False)
    store.save(Song.model_validate(_timed_song()), message="gold")
    _no_gathering(monkeypatch)
    _with_audio(monkeypatch, _mir())
    _use(monkeypatch, _Static(_timed_song()))

    report = pipeline_mod.run_pipeline(TITLE, ARTIST, provider="anthropic")

    assert report.steps["timing"] == "ok"
    stored = store.get(SONG_ID)
    assert stored.metadata.bpm == 90.0                 # the MIR's, not the model's
    assert len(stored.audio.beats) == 64               # the measured grid
    assert TIMING_RESTORE_ACTION not in [p.action for p in stored.provenance]


# --- observability: what the strip costs a listen=off run ------------------------


def _timed_song_with_a_new_line() -> dict:
    """What a notes-only reconciliation legitimately returns: the prior document
    plus a line the model added, which has no partner in the prior version and
    therefore no time to carry."""
    document = _timed_song()
    document["lines"].append(
        {
            "lineIndex": 3,
            "lyrics": "And a brand new line the model added for us",
            "chordPlacements": [{"charIndex": 0, "chord": "Am"}],
        }
    )
    return document


def test_the_lines_the_strip_leaves_untimed_are_reported_not_just_graded(
    monkeypatch, store
):
    """On a ``listen=off`` run the model's new or reworded lines end with no
    time and nothing refills them: the strip took the model's numbers,
    carry-forward has no match to copy, and LRC is skipped whenever timing was
    carried forward. That is the right answer (quality/escalation.py: "could not
    time this region" beats fabricated spacing) — but the operator has to read
    it in the step output rather than infer it from a timingCoverage drop."""
    monkeypatch.setattr(settings, "quality_enabled", False)
    store.save(Song.model_validate(_timed_song()), message="gold")
    _use(monkeypatch, _Static(_timed_song_with_a_new_line()))

    report = pipeline_mod.run_pipeline(
        TITLE, ARTIST, provider="anthropic", scope=NOTES_ONLY,
        guidance="add the tag line at the end",
    )

    assert "1 line(s) and 1 placement(s) left untimed" in report.steps["timing"]
    assert "nothing else times them on this path" in report.steps["timing"]
    stored = store.get(SONG_ID)
    assert [line.timeSeconds for line in stored.lines] == [0.0, 8.0, 16.0, None]
    assert _placements(stored)[-1].timeSeconds is None

    run = next(
        s for s in pipeline_mod.get_run_store().get_run(report.run_id)["steps"]
        if s["label"] == "left-untimed"
    )
    assert run["detail"]["linesUntimed"] == 1
    assert run["detail"]["placementsUntimed"] == 1
