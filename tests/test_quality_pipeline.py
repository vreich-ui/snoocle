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
from snoocle_server import recordings as recordings_mod
from snoocle_server.audio.acquire import AcquiredAudio
from snoocle_server.config import settings
from snoocle_server.discovery.cache import DiscoveryCacheInfo
from snoocle_server.discovery.models import CandidateSource
from snoocle_server.mir.base import AnalyzedWindow, ChordSegment, MirAnalysis
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


def _mir(
    chords: list[str], *, duration: float = DURATION, span_end: float | None = None
) -> MirAnalysis:
    """`chords` spread evenly across the track.

    `span_end` stops the chord timeline early while the track keeps its real
    length -- the lo-fi/live shape where the recognizer gives up partway through
    (the 1966 take: a chord timeline dying at 86.5s of 220.6s). Everything after
    it holds the last matched time, which is what produces a collapsed run at
    the tail with nothing later to spread toward.
    """
    end = span_end if span_end is not None else duration
    step = end / len(chords)
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
    return pipeline_mod.run_pipeline(
        "Quality Gate",
        "Test",
        provider="anthropic",
        agent_policy="always",
        **extra,
    )


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


# --- a surviving collapse run: mark, and spend nothing ----------------------


#: One more line than the MIR timeline's matched chords reach with a real
#: root, so `snap` holds the last matched time for all of them and the
#: collapse guard finds no later anchor to spread them toward. Deliberately
#: only 3 trailing lines (the guard's own threshold) rather than 4: enough of
#: the document still lands inside the MIR's full-track timeline that
#: `timingCoverage` clears its own threshold on its own -- the collapse really
#: is the ONLY thing wrong with this grade (`Grade.failing == (collapseRuns
#: metric,)`), not merely the only fail ROUTE (see quality/escalation.py and
#: the adversarial-review fix it names). A run with a genuinely low
#: `timingCoverage` alongside the collapse is exercised separately below
#: (`test_an_audio_fault_that_also_fails_its_own_coverage_still_searches_for_a_recording`),
#: where searching for a better recording is the CORRECT thing to do.
COLLAPSING_ONLY = TRUE_PROGRESSION[:9] + ["B", "B", "B"]


def test_a_surviving_collapse_run_marks_the_timing_and_searches_for_nothing(
    monkeypatch, store
):
    """A tail collapse that is the ONLY thing wrong with the grade, on a run
    whose fault attribution is AUDIO for a reason that has nothing to do with
    coverage: the candidate sheets agree with each other but not with what the
    MIR chord timeline actually heard (a mismatched recording), while the MIR
    timeline itself still spans the whole track and the document's own timing
    still covers most of it. `collapseRuns` is a hard gate, so this still
    grades `fail`, and the fault is still AUDIO -- which on its own would mean
    a live `suggest_recordings` search. But `timing.collapse_guard` already
    refused to invent spacing for the collapsed lines, and no other recording
    of the song supplies the span it looked for, so the collapse-only branch
    must win here regardless of what the fault attribution alone would do:
    mark the timing unreliable, and make no network call and no second
    attempt.
    """
    _wire(monkeypatch, mir=_mir(TRUE_PROGRESSION),
          candidates=[_candidate(OTHER_SONG, "web-1"), _candidate(OTHER_SONG, "web-2")])
    reconciler = _Reconciler(_song(COLLAPSING_ONLY, timed=False))
    monkeypatch.setattr(pipeline_mod, "reconcile", reconciler)

    # A spy, not an empty result: the point is that the CALL never happens.
    searched: list[dict] = []

    def spy_suggest(song, **kwargs):
        searched.append({"songId": song.id, **kwargs})
        return recordings_mod.RecordingSuggestions(song_id=song.id, reason="spy")

    monkeypatch.setattr(pipeline_mod, "suggest_recordings", spy_suggest)

    report = _analyze()

    assert searched == [], "a surviving collapse must not search for another recording"
    assert report.recording_suggestions is None
    assert "recording-suggestions" not in report.steps
    assert len(reconciler.calls) == 1, "and must not spend the retry either"
    assert reconciler.feedbacks == [None]

    # The collapse guard really did leave the run alone, and the grade really
    # did fail on that gate ALONE -- both as its only fail route and as its
    # only failing metric -- otherwise this test would be passing for the
    # wrong reason.
    assert "no later anchor" in report.steps["timing-collapse-guard"]
    assert report.steps["quality"].startswith("fail")
    run = pipeline_mod.get_run_store().get_run(report.run_id)
    assert run["quality"]["grade"]["failing"] == ["collapseRuns"]
    assert run["quality"]["grade"]["metrics"]["collapseRuns"]["ok"] is False
    assert run["quality"]["grade"]["metrics"]["timingCoverage"]["ok"] is True
    assert run["quality"]["attribution"]["fault"] == "audio"
    assert run["quality"]["escalation"] == {
        "retry": False,
        "search": False,
        "markTimingUnreliable": True,
        "suggestAlternativeRecording": False,
        "reanalyzeFullAccuracy": False,
        "reason": run["quality"]["escalation"]["reason"],
        "markCause": run["quality"]["escalation"]["markCause"],
    }
    assert "timing.collapse_guard" in run["quality"]["escalation"]["reason"]
    assert "are the only thing failing this grade" in run["quality"]["escalation"]["reason"]
    assert run["quality"]["retriesSpent"] == 0
    assert run["quality"]["searchesSpent"] == 0

    # The step text names the real cause instead of the hardcoded audio fault.
    assert "collapse" in report.steps["timing-reliability"]
    assert "audio fault" not in report.steps["timing-reliability"]

    # And the marker is still written -- it is what stops a later carry-forward
    # from inheriting this region's timing as if it were measured.
    stored = store.get(SONG_ID)
    assert _actions(stored).count("timing-unreliable") == 1
    marker = next(p for p in stored.provenance if p.action == "timing-unreliable")
    assert "survived timing.collapse_guard" in marker.notes


def test_an_audio_fault_that_also_fails_its_own_coverage_still_searches_for_a_recording(
    monkeypatch, store
):
    """The reported production shape, end to end: a lo-fi run whose MIR chord
    timeline dies at 37% of the track, four trailing lines held at one
    timestamp as a result. Unlike the collapse-only test above, `timingCoverage`
    ALSO genuinely fails here (0.34 << 0.6) -- the same MIR shortfall that
    triggers the AUDIO attribution also leaves most of the document untimed,
    not merely its collapsed tail. That is a second, independent reason this
    grade fails, so the collapse-only branch must NOT take over: this is an
    ordinary AUDIO fault, and an ordinary AUDIO fault searches for a better
    recording of the song (`suggest_alternative_recording`) precisely because
    the recording itself, not just one region of it, is what cannot be timed.
    """
    collapsing = TRUE_PROGRESSION + ["C", "G", "Am", "F"]
    _wire(monkeypatch, mir=_mir(TRUE_PROGRESSION, span_end=22.0),
          candidates=[_candidate(collapsing, "web-1"), _candidate(collapsing, "web-2")])
    reconciler = _Reconciler(_song(collapsing, timed=False))
    monkeypatch.setattr(pipeline_mod, "reconcile", reconciler)

    searched: list[dict] = []

    def spy_suggest(song, **kwargs):
        searched.append({"songId": song.id, **kwargs})
        return recordings_mod.RecordingSuggestions(song_id=song.id, reason="spy")

    monkeypatch.setattr(pipeline_mod, "suggest_recordings", spy_suggest)

    report = _analyze()

    run = pipeline_mod.get_run_store().get_run(report.run_id)
    assert set(run["quality"]["grade"]["failing"]) == {"timingCoverage", "collapseRuns"}
    assert run["quality"]["attribution"]["fault"] == "audio"

    assert searched != [], "a genuinely low-coverage audio fault must still search"
    assert report.recording_suggestions is not None
    assert "recording-suggestions" in report.steps
    assert len(reconciler.calls) == 1, "an AUDIO fault must never retry, regardless"

    assert run["quality"]["escalation"]["retry"] is False
    assert run["quality"]["escalation"]["markTimingUnreliable"] is True
    assert run["quality"]["escalation"]["suggestAlternativeRecording"] is True
    assert run["quality"]["escalation"]["markCause"] == "audio fault"
    assert report.steps["timing-reliability"] == "marked unreliable (audio fault)"


def test_a_collapse_only_fail_does_not_spend_the_model_retry(monkeypatch, store):
    """The other reported shape: a full-length MIR, sheets that match the audio
    better than the document (a MODEL fault), and a tail collapse that is the
    only fail ROUTE -- and, deliberately, also the only failing METRIC (see
    `COLLAPSING_ONLY` above): `timingCoverage` clears its own threshold here,
    so this really is a collapse-only fail, not a broader MODEL fault that
    happens to also carry a collapse (that case is
    `test_a_collapse_plus_a_genuine_second_failure_still_gets_the_model_retry`
    below). The hard gate used to route cases like this one into a full second
    reconciliation -- model plus tools -- carrying feedback that asked the
    model to space the very lines `timing.collapse_guard` had just refused to
    space.
    """
    _wire(monkeypatch, mir=_mir(TRUE_PROGRESSION),
          candidates=[_candidate(TRUE_PROGRESSION, "web-1"),
                      _candidate(TRUE_PROGRESSION, "web-2")])
    reconciler = _Reconciler(_song(COLLAPSING_ONLY, timed=False))
    monkeypatch.setattr(pipeline_mod, "reconcile", reconciler)

    report = _analyze()

    assert len(reconciler.calls) == 1, "a collapse-only fail must not buy a retry"
    assert "quality-retry" not in report.steps
    run = pipeline_mod.get_run_store().get_run(report.run_id)
    assert run["quality"]["attribution"]["fault"] == "model"
    assert run["quality"]["grade"]["failing"] == ["collapseRuns"], (
        "the collapse must be the grade's ONLY failing metric for this branch "
        "to be the right one to test"
    )
    assert run["quality"]["grade"]["metrics"]["collapseRuns"]["ok"] is False
    assert run["quality"]["escalation"]["retry"] is False
    assert run["quality"]["escalation"]["markTimingUnreliable"] is True
    assert run["quality"]["retriesSpent"] == 0
    assert "collapse" in report.steps["timing-reliability"]


def test_a_collapse_plus_a_genuine_second_failure_still_gets_the_model_retry(
    monkeypatch, store
):
    """A document whose tail collapses AND whose `timingCoverage` genuinely
    fails on its own (0.58 < 0.6 -- one line short of `COLLAPSING_ONLY` above):
    `collapseRuns`' hard gate is still this grade's only fail ROUTE (neither
    `overall` nor the failing-majority route trips), but it is no longer the
    grade's only failing METRIC. A document that is broken for its own reasons
    besides the collapse is an ordinary MODEL fault and still buys its one
    retry -- suppressing that would silently drop a real, actionable defect
    behind an incidental collapse, and would write a false "the only thing
    failing this grade" claim into the stored timing-unreliable provenance
    that this branch does NOT even reach.
    """
    document = TRUE_PROGRESSION[:8] + ["B", "B", "B", "B"]
    _wire(monkeypatch, mir=_mir(TRUE_PROGRESSION),
          candidates=[_candidate(TRUE_PROGRESSION, "web-1"),
                      _candidate(TRUE_PROGRESSION, "web-2")])
    reconciler = _Reconciler(
        _song(document, timed=False), _song(TRUE_PROGRESSION, timed=True),
    )
    monkeypatch.setattr(pipeline_mod, "reconcile", reconciler)

    report = _analyze()

    assert len(reconciler.calls) == 2, "a genuine second failure must still buy its retry"
    feedback = reconciler.feedbacks[1]
    assert feedback is not None
    assert "timingCoverage" in feedback
    assert "collapseRuns" in feedback

    run = pipeline_mod.get_run_store().get_run(report.run_id)
    # `run["quality"]` only ever carries the FINAL grade (the retry's, which
    # passes) -- the first attempt's own grade+attribution is what the
    # "quality-grade" step recorded before the retry ran.
    first_grade_step = next(s for s in run["steps"] if s["label"] == "quality-grade")
    assert set(first_grade_step["detail"]["grade"]["failing"]) == {
        "timingCoverage", "collapseRuns",
    }
    assert first_grade_step["detail"]["attribution"]["fault"] == "model"
    assert run["quality"]["retriesSpent"] == 1
    assert report.steps["quality"].startswith("pass")
    assert "after 1 retry" in report.steps["quality"]

    # No timing-unreliable marker: the fault branch took over, not the
    # collapse-only one, so nothing here claims the collapse was the whole
    # story.
    stored = store.get(SONG_ID)
    assert "timing-unreliable" not in _actions(stored)


# --- partial-accuracy timing coverage: recommend, don't retry (issue #59) --


def test_a_fast_accuracy_low_coverage_run_recommends_reanalysis_not_a_retry(
    monkeypatch, store
):
    """A fast-accuracy MIR (a few sampled windows, not the whole track) whose
    document ends up with poor timing coverage gets the full-accuracy
    reanalysis recommendation -- NOT a model retry, even though the sources
    here agree with the audio and the document better than it (what would
    otherwise be a MODEL fault)."""
    partial_mir = _mir(TRUE_PROGRESSION).model_copy(
        update={
            "analyzed_windows": [
                AnalyzedWindow(start=6.0, end=16.0),
                AnalyzedWindow(start=26.0, end=36.0),
                AnalyzedWindow(start=46.0, end=56.0),
            ]
        }
    )
    _wire(monkeypatch, mir=partial_mir,
          candidates=[_candidate(TRUE_PROGRESSION, "web-1"),
                      _candidate(TRUE_PROGRESSION, "web-2")])
    reconciler = _Reconciler(_song(INVENTED, timed=False))
    monkeypatch.setattr(pipeline_mod, "reconcile", reconciler)

    report = _analyze()

    assert len(reconciler.calls) == 1, "the recommendation must not spend the retry budget"
    assert "recommended: re-analyze at full accuracy" in report.steps["accuracy-escalation"]
    assert "fast/windowed accuracy" in report.steps["accuracy-escalation"]

    run = pipeline_mod.get_run_store().get_run(report.run_id)
    assert run["quality"]["escalation"]["reanalyzeFullAccuracy"] is True
    assert run["quality"]["escalation"]["retry"] is False
    assert run["quality"]["retriesSpent"] == 0
    stored = store.get(SONG_ID)
    assert "quality-grade" in _actions(stored)
    assert "timing-unreliable" not in _actions(stored)


def test_a_full_accuracy_low_coverage_run_is_unaffected(monkeypatch, store):
    """The same document, the same everything, except the MIR analysis
    covered the whole track -- ordinary MODEL-fault handling applies, exactly
    as test_a_model_fault_run_retries_once_with_the_specific_failures pins."""
    _wire(monkeypatch, mir=_mir(TRUE_PROGRESSION),
          candidates=[_candidate(TRUE_PROGRESSION, "web-1"),
                      _candidate(TRUE_PROGRESSION, "web-2")])
    reconciler = _Reconciler(
        _song(INVENTED, timed=False), _song(TRUE_PROGRESSION, timed=True),
    )
    monkeypatch.setattr(pipeline_mod, "reconcile", reconciler)

    report = _analyze()

    assert len(reconciler.calls) == 2, "a MODEL fault still gets its one retry"
    assert "accuracy-escalation" not in report.steps
    run = pipeline_mod.get_run_store().get_run(report.run_id)
    assert run["quality"]["retriesSpent"] == 1


# --- SOURCE fault: one targeted search --------------------------------------


def test_a_source_fault_search_cannot_override_retry_false(
    monkeypatch, store
):
    """Search may improve evidence, but escalation.retry=false is authoritative:
    no internally generated path may turn that into another model call."""
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
    assert len(reconciler.calls) == 1
    assert "reconciling once more" in report.steps["quality-search"]
    assert report.steps["quality-retry"] == "skipped: escalation.retry=false"
    run = pipeline_mod.get_run_store().get_run(report.run_id)
    assert any(step["label"] == "quality-retry-suppressed" for step in run["steps"])


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

    # The gate is shared with Mode B (quality/gate.py) — patch it there, which
    # is also where a real bug would live.
    monkeypatch.setattr(pipeline_mod, "evaluate_quality", boom)

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
