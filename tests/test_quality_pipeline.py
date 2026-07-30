"""The quality gate as the pipeline wires it: grade, attribute, act — once.

The behaviour these tests pin is the one the production evidence asks for:

- a MODEL-fault run retries exactly ONCE, and the retry is handed the specific
  grade and the specific failures (metric names, line indexes) rather than a
  vague "try harder";
- an AUDIO-fault run does NOT retry — it stores and marks the version's timing
  unreliable;
- the grade is recorded in provenance on every run, whatever the verdict, and
  whether or not anything was escalated.

`reconcile` is stood in for (the `_Reconciler` below, same pattern as
tests/test_correction_pipeline.py) so these are about pipeline.py's OWN wiring
and never make a model call.
"""

from __future__ import annotations

import pytest

from snoocle_server import pipeline as pipeline_mod
from snoocle_server.audio.acquire import AcquiredAudio
from snoocle_server.config import settings
from snoocle_server.discovery.cache import DiscoveryCacheInfo
from snoocle_server.discovery.models import CandidateSource
from snoocle_server.mir.base import ChordSegment, MirAnalysis
from snoocle_server.mir.cache import MirCacheInfo
from snoocle_server.reconcile.engine import ReconcileResult
from snoocle_server.schema import Song
from snoocle_server.schema.song import ChordPlacement, Line
from snoocle_server.store.memory import InMemorySongRepository
from snoocle_server.store.song_notes import reset_song_notes_store

SONG_ID = "test--quality-gate"
DURATION = 60.0
TRUE_PROGRESSION = ["C", "G", "Am", "F", "Dm", "G", "Em", "Am", "F", "C", "Bb", "F"]
INVENTED = ["D", "E", "D", "E", "D", "E", "D", "E"]
OTHER_SONG = ["Dm", "Eb", "Dm", "Bb", "Gm", "Db", "Dm", "Ab", "Dm", "Eb", "Cm", "Dm"]


def _mir(chords: list[str], *, duration: float = DURATION) -> MirAnalysis:
    step = duration / len(chords)
    return MirAnalysis(
        engines={"chords": "chord-cnn-lstm"},
        duration_seconds=duration,
        key="C major",
        chords=[
            ChordSegment(start=i * step, end=(i + 1) * step, chord=c)
            for i, c in enumerate(chords)
        ],
    )


def _candidate(chords: list[str], source_id: str) -> CandidateSource:
    return CandidateSource(
        sourceId=source_id,
        url=f"https://example.test/{source_id}",
        lines=[
            Line(lineIndex=i, lyrics=f"line {i} has real words in it",
                 chordPlacements=[ChordPlacement(charIndex=0, chord=chord)])
            for i, chord in enumerate(chords)
        ],
    )


def _song(chords: list[str], *, timed: bool, section_times: bool = True) -> Song:
    lines = []
    for i, chord in enumerate(chords):
        timing = {"timeSeconds": i * 5.0, "confidence": 0.9} if timed else {}
        lines.append(
            {
                "lineIndex": i,
                "lyrics": f"line {i} has real words in it",
                **timing,
                "chordPlacements": [{"charIndex": 0, "chord": chord, **timing}],
            }
        )
    section = {
        "sectionIndex": 0, "name": "Verse 1", "kind": "verse",
        "startLineIndex": 0, "endLineIndex": len(chords) - 1,
    }
    if section_times:
        section |= {"startTime": 0.0, "endTime": DURATION}
    return Song.model_validate(
        {
            "id": SONG_ID,
            "metadata": {"title": "Quality Gate", "artist": "Test", "key": "C major"},
            "audio": {"durationSeconds": DURATION},
            "sections": [section],
            "lines": lines,
            "provenance": [
                {"timestamp": "2026-07-30T00:00:00Z", "actor": "reconcile:test/fake",
                 "action": "reconciled"}
            ],
        }
    )


class _Reconciler:
    """Stands in for `reconcile`, returning a queued song per call and recording
    the `quality_feedback` each call was given."""

    def __init__(self, *songs: Song):
        self.songs = list(songs)
        self.calls: list[dict] = []

    def __call__(self, title, artist, candidates, mir, **kwargs):
        self.calls.append({"candidates": candidates, "mir": mir, **kwargs})
        song = self.songs[min(len(self.calls) - 1, len(self.songs) - 1)]
        return ReconcileResult(
            song=song.model_copy(deep=True), provider="test", model="fake",
            attempts=1, audio_attached=False, usage={},
        )

    @property
    def feedbacks(self) -> list[str | None]:
        return [c.get("quality_feedback") for c in self.calls]


@pytest.fixture
def store(monkeypatch):
    repo = InMemorySongRepository()
    monkeypatch.setattr(pipeline_mod, "get_store", lambda: repo)
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(pipeline_mod, "fetch_lrc", lambda *a, **k: None)
    reset_song_notes_store()
    yield repo
    reset_song_notes_store()


def _wire(monkeypatch, *, mir: MirAnalysis, candidates: list[CandidateSource]) -> None:
    """Give the run an audio analysis and a source set without touching the network."""
    monkeypatch.setattr(
        pipeline_mod, "_step_acquire",
        lambda *a, **k: AcquiredAudio(
            path="/dev/null", video_id="abcdefghijk", video_title="X",
            duration_seconds=DURATION,
        ),
    )
    monkeypatch.setattr(
        pipeline_mod, "_step_mir",
        lambda *a, **k: (mir, MirCacheInfo(status="miss", analyzed_at="2026-07-30T00:00:00Z")),
    )
    monkeypatch.setattr(
        pipeline_mod, "_step_discover",
        lambda *a, **k: (
            list(candidates),
            DiscoveryCacheInfo(status="miss", gathered_at="2026-07-30T00:00:00Z"),
        ),
    )


def _analyze(**extra):
    """The synchronous entry point -- the suite has no async plugin, and
    run_pipeline is the same orchestration behind an asyncio.run."""
    return pipeline_mod.run_pipeline("Quality Gate", "Test", provider="anthropic", **extra)


def _actions(song: Song) -> list[str]:
    return [p.action for p in song.provenance]


# --- MODEL fault: one retry, and only one -----------------------------------


def test_a_model_fault_run_retries_once_with_the_specific_failures(monkeypatch, store):
    """Sources agree with each other and with the audio; the first document
    agrees with neither. That is actionable, so one retry happens — carrying
    the metric names and the offending indexes."""
    _wire(monkeypatch, mir=_mir(TRUE_PROGRESSION),
          candidates=[_candidate(TRUE_PROGRESSION, "web-1"),
                      _candidate(TRUE_PROGRESSION, "web-2")])
    reconciler = _Reconciler(
        _song(INVENTED, timed=False),  # first attempt: ignores the evidence
        _song(TRUE_PROGRESSION, timed=True),  # retry: uses it
    )
    monkeypatch.setattr(pipeline_mod, "reconcile", reconciler)

    report = _analyze()

    assert len(reconciler.calls) == 2, "a MODEL fault should buy exactly one retry"
    assert reconciler.feedbacks[0] is None
    feedback = reconciler.feedbacks[1]
    assert feedback is not None
    # Specific, not vague: the failing metric is named, and so are the offenders.
    assert "chordMatchRatio" in feedback
    assert "line 0, charIndex 0" in feedback
    assert "This is the only retry." in feedback
    assert "try harder" not in feedback.lower()

    # The stored document is the retry's, and the run reports a passing grade.
    stored = store.get(SONG_ID)
    assert [p.chord for line in stored.lines for p in line.chordPlacements] == TRUE_PROGRESSION
    assert report.steps["quality"].startswith("pass")
    assert "after 1 retry" in report.steps["quality"]


def test_a_model_fault_retry_that_is_no_better_is_never_retried_again(monkeypatch, store):
    """The Marley case: repeated attempts on the same inputs do not converge.
    The second failing grade must not buy a third attempt."""
    _wire(monkeypatch, mir=_mir(TRUE_PROGRESSION),
          candidates=[_candidate(TRUE_PROGRESSION, "web-1"),
                      _candidate(TRUE_PROGRESSION, "web-2")])
    reconciler = _Reconciler(_song(INVENTED, timed=False))  # every call returns the same
    monkeypatch.setattr(pipeline_mod, "reconcile", reconciler)

    report = _analyze()

    assert len(reconciler.calls) == 2, "one retry, never two"
    assert report.steps["quality"].startswith("fail")
    run = pipeline_mod.get_run_store().get_run(report.run_id)
    assert run["quality"]["retriesSpent"] == 1
    assert run["quality"]["escalation"]["retry"] is False
    assert "already spent" in run["quality"]["escalation"]["reason"]
    assert "do not converge" in run["quality"]["escalation"]["reason"]

    stored = store.get(SONG_ID)
    assert _actions(stored).count("quality-grade") == 1, "one grade per stored version"
    grade_entry = next(p for p in stored.provenance if p.action == "quality-grade")
    assert "fail" in grade_entry.notes
    assert "model" in grade_entry.notes


def test_retrying_can_be_switched_off_without_switching_off_grading(monkeypatch, store):
    monkeypatch.setattr(settings, "quality_retry_enabled", False)
    _wire(monkeypatch, mir=_mir(TRUE_PROGRESSION),
          candidates=[_candidate(TRUE_PROGRESSION, "web-1"),
                      _candidate(TRUE_PROGRESSION, "web-2")])
    reconciler = _Reconciler(_song(INVENTED, timed=False))
    monkeypatch.setattr(pipeline_mod, "reconcile", reconciler)

    report = _analyze()

    assert len(reconciler.calls) == 1
    assert report.steps["quality"].startswith("fail")
    assert "quality-grade" in _actions(store.get(SONG_ID))


def test_a_failed_retry_does_not_sink_a_storable_first_attempt(monkeypatch, store):
    """The retry is optional; the first document is not. Losing a storable song
    to a failed extra attempt would be strictly worse than storing the graded
    original."""
    _wire(monkeypatch, mir=_mir(TRUE_PROGRESSION),
          candidates=[_candidate(TRUE_PROGRESSION, "web-1"),
                      _candidate(TRUE_PROGRESSION, "web-2")])
    first = _song(INVENTED, timed=False)
    calls: list[str | None] = []

    def reconcile(title, artist, candidates, mir, **kwargs):
        calls.append(kwargs.get("quality_feedback"))
        if kwargs.get("quality_feedback") is not None:
            raise RuntimeError("the model fell over on the retry")
        return ReconcileResult(
            song=first.model_copy(deep=True), provider="test", model="fake",
            attempts=1, audio_attached=False, usage={},
        )

    monkeypatch.setattr(pipeline_mod, "reconcile", reconcile)

    report = _analyze()

    assert len(calls) == 2
    assert report.stored_version, "the first attempt must still have been stored"
    assert "failed" in report.steps["quality-retry"]
    assert "quality-grade" in _actions(store.get(SONG_ID))


# --- AUDIO fault: store, mark, never retry ----------------------------------


def test_an_audio_fault_run_does_not_retry_and_marks_the_timing_unreliable(
    monkeypatch, store
):
    """Both sheets agree; the recording is what disagrees. A retry would fit the
    same unreliable timeline again, at full price."""
    _wire(monkeypatch, mir=_mir(OTHER_SONG),
          candidates=[_candidate(TRUE_PROGRESSION, "web-1"),
                      _candidate(TRUE_PROGRESSION, "web-2")])
    reconciler = _Reconciler(_song(TRUE_PROGRESSION, timed=False, section_times=False))
    monkeypatch.setattr(pipeline_mod, "reconcile", reconciler)

    report = _analyze()

    assert len(reconciler.calls) == 1, "an AUDIO fault must not be retried"
    assert report.steps["timing-reliability"] == "marked unreliable (audio fault)"
    assert "audio (not actionable)" in report.steps["quality"]

    stored = store.get(SONG_ID)
    actions = _actions(stored)
    assert actions.count("quality-grade") == 1
    assert actions.count("timing-unreliable") == 1
    marker = next(p for p in stored.provenance if p.action == "timing-unreliable")
    assert "NOT reliable and was not retried" in marker.notes


# --- SOURCE fault: one targeted search --------------------------------------


def test_a_source_fault_searches_once_and_only_retries_on_better_evidence(
    monkeypatch, store
):
    """The sheets contradict each other, so a retry against them buys nothing —
    but a search that turns up sheets agreeing with the audio does."""
    _wire(monkeypatch, mir=_mir(TRUE_PROGRESSION),
          candidates=[_candidate(TRUE_PROGRESSION, "web-1"), _candidate(OTHER_SONG, "web-2")])
    searches: list[bool] = []

    better = [_candidate(TRUE_PROGRESSION, "web-3"), _candidate(TRUE_PROGRESSION, "web-4")]

    def discover(title, artist, max_candidates, refresh=False):
        searches.append(refresh)
        if refresh:  # the escalation's own search bypasses the cache
            return list(better), DiscoveryCacheInfo(status="refresh", gathered_at="x")
        return (
            [_candidate(TRUE_PROGRESSION, "web-1"), _candidate(OTHER_SONG, "web-2")],
            DiscoveryCacheInfo(status="miss", gathered_at="x"),
        )

    monkeypatch.setattr(pipeline_mod, "_step_discover", discover)
    reconciler = _Reconciler(
        _song(INVENTED, timed=False), _song(TRUE_PROGRESSION, timed=True)
    )
    monkeypatch.setattr(pipeline_mod, "reconcile", reconciler)

    report = _analyze()

    assert searches == [False, True], "exactly one extra, cache-bypassing search"
    assert len(reconciler.calls) == 2
    assert [c.sourceId for c in reconciler.calls[1]["candidates"]] == ["web-3", "web-4"]
    assert "reconciling once more" in report.steps["quality-search"]
    assert "freshly gathered sheets" in reconciler.feedbacks[1]


def test_a_source_fault_search_that_finds_nothing_better_stores_with_the_grade(
    monkeypatch, store
):
    contradictory = [_candidate(TRUE_PROGRESSION, "web-1"), _candidate(OTHER_SONG, "web-2")]
    _wire(monkeypatch, mir=_mir(TRUE_PROGRESSION), candidates=contradictory)
    searches: list[bool] = []

    def discover(title, artist, max_candidates, refresh=False):
        searches.append(refresh)
        return list(contradictory), DiscoveryCacheInfo(status="miss", gathered_at="x")

    monkeypatch.setattr(pipeline_mod, "_step_discover", discover)
    reconciler = _Reconciler(_song(INVENTED, timed=False))
    monkeypatch.setattr(pipeline_mod, "reconcile", reconciler)

    report = _analyze()

    assert searches == [False, True]
    assert len(reconciler.calls) == 1, "no better evidence, so no full-price retry"
    assert "none agreeing with the audio better" in report.steps["quality-search"]
    assert "quality-grade" in _actions(store.get(SONG_ID))


# --- the grade is recorded whatever happens ---------------------------------


def test_a_passing_run_still_records_its_grade(monkeypatch, store):
    """The history is the point: ten runs at ~0.5 are only visible as
    non-convergence if every run wrote down what it scored."""
    _wire(monkeypatch, mir=_mir(TRUE_PROGRESSION),
          candidates=[_candidate(TRUE_PROGRESSION, "web-1")])
    reconciler = _Reconciler(_song(TRUE_PROGRESSION, timed=True))
    monkeypatch.setattr(pipeline_mod, "reconcile", reconciler)

    report = _analyze()

    assert len(reconciler.calls) == 1
    assert report.steps["quality"].startswith("pass")
    entry = next(p for p in store.get(SONG_ID).provenance if p.action == "quality-grade")
    assert entry.confidence is not None and entry.confidence > 0.6
    assert "chordMatchRatio=1.00" in entry.notes


def test_the_grade_attribution_and_decision_land_on_the_run_trace(monkeypatch, store):
    _wire(monkeypatch, mir=_mir(TRUE_PROGRESSION),
          candidates=[_candidate(TRUE_PROGRESSION, "web-1")])
    monkeypatch.setattr(
        pipeline_mod, "reconcile", _Reconciler(_song(TRUE_PROGRESSION, timed=True))
    )

    report = _analyze()

    run = pipeline_mod.get_run_store().get_run(report.run_id)
    assert run["quality"]["grade"]["verdict"] == "pass"
    assert run["quality"]["attribution"]["fault"] == "none"
    assert run["quality"]["retriesSpent"] == 0
    assert run["quality"]["escalation"]["retry"] is False
    assert any(step["label"] == "quality-grade" for step in run["steps"])


def test_grading_can_be_switched_off_entirely(monkeypatch, store):
    monkeypatch.setattr(settings, "quality_enabled", False)
    _wire(monkeypatch, mir=_mir(TRUE_PROGRESSION),
          candidates=[_candidate(TRUE_PROGRESSION, "web-1")])
    monkeypatch.setattr(pipeline_mod, "reconcile", _Reconciler(_song(INVENTED, timed=False)))

    report = _analyze()

    assert report.steps["quality"] == "skipped (quality grading disabled)"
    assert "quality-grade" not in _actions(store.get(SONG_ID))


def test_a_grader_failure_never_fails_the_run(monkeypatch, store):
    _wire(monkeypatch, mir=_mir(TRUE_PROGRESSION),
          candidates=[_candidate(TRUE_PROGRESSION, "web-1")])
    monkeypatch.setattr(pipeline_mod, "reconcile", _Reconciler(_song(TRUE_PROGRESSION, timed=True)))

    def boom(*a, **k):
        raise RuntimeError("grader exploded")

    monkeypatch.setattr(pipeline_mod, "grade_song", boom)

    report = _analyze()

    assert report.stored_version, "the song must still be stored"
    assert report.steps["quality"].startswith("failed: ")


# --- the evidence manifest now carries the candidate scores -----------------


def test_the_manifest_hands_the_reconciler_each_sources_audio_agreement(
    monkeypatch, store
):
    _wire(monkeypatch, mir=_mir(TRUE_PROGRESSION),
          candidates=[_candidate(TRUE_PROGRESSION, "web-1"), _candidate(OTHER_SONG, "web-2")])
    monkeypatch.setattr(pipeline_mod, "reconcile", _Reconciler(_song(TRUE_PROGRESSION, timed=True)))

    report = _analyze()

    scores = report.evidence_manifest["sources"]["scores"]
    assert [s["sourceId"] for s in scores] == ["web-1", "web-2"]
    assert scores[0]["score"] == 1.0
    assert scores[1]["score"] < scores[0]["score"]
