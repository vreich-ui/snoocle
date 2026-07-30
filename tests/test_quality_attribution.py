"""Fault attribution: the part that must be right.

A failing grade is not automatically the model's fault, and retrying a run
whose evidence was bad costs full price for the same result. Three patterns,
each built here from the evidence that distinguishes it:

- sources agree with each other AND with the audio, document doesn't -> MODEL,
  actionable.
- sources agree with each other, the audio contradicts them -> AUDIO, not
  actionable by retry.
- sources contradict each other -> SOURCE, needs better evidence.

Plus the honest non-answers, which matter just as much: guessing "model" when
the evidence can't support it is how a retry budget gets burned on runs that
were never going to converge.
"""

from __future__ import annotations

from snoocle_server.chords import transpose_chord
from snoocle_server.discovery.models import CandidateSource
from snoocle_server.mir.base import ChordSegment, MirAnalysis
from snoocle_server.quality.attribution import (
    Fault,
    attribute_fault,
    candidate_agreement,
    mir_timeline_coverage,
)
from snoocle_server.quality.grader import grade_song
from snoocle_server.reconcile.match import score_candidates
from snoocle_server.schema import Song
from snoocle_server.schema.song import ChordPlacement, Line

_DURATION = 60.0
#: A I-V-vi-IV pop progression with a borrowed bVII, twelve chords long.
_TRUE_PROGRESSION = ["C", "G", "Am", "F", "Dm", "G", "Em", "Am", "F", "C", "Bb", "F"]
#: A DIFFERENT song, not a transposed copy: a modal progression whose interval
#: content shares little with the above. Fixtures matter here -- two pop
#: progressions built on the same I-V-vi-IV skeleton in different keys really
#: are near-identical evidence, and calling that "disagreement" would be the
#: test lying rather than the metric.
_OTHER_SONG = ["Dm", "Eb", "Dm", "Bb", "Gm", "Db", "Dm", "Ab", "Dm", "Eb", "Cm", "Dm"]
#: A third, also unrelated: a twelve-bar blues.
_THIRD_SONG = ["E", "E", "A", "E", "B", "A", "E", "E", "A", "E", "B", "E"]
#: Chords the model produced that appear in no source and in no timeline.
_INVENTED = ["D", "E", "D", "E", "D", "E", "D", "E"]


def _mir(chords: list[str], *, duration: float = _DURATION, span: float | None = None,
         key: str = "C major") -> MirAnalysis:
    """A chord timeline of `chords`, spread evenly over `span` (default: the
    whole track). A span shorter than the duration is the lo-fi signature: the
    recognizer stopped producing segments partway through."""
    span = span if span is not None else duration
    step = span / len(chords)
    return MirAnalysis(
        duration_seconds=duration,
        key=key,
        chords=[
            ChordSegment(start=i * step, end=(i + 1) * step, chord=c)
            for i, c in enumerate(chords)
        ],
    )


def _candidate(chords: list[str], source_id: str) -> CandidateSource:
    return CandidateSource(
        sourceId=source_id,
        lines=[
            Line(lineIndex=i, lyrics=f"line {i} has real words in it",
                 chordPlacements=[ChordPlacement(charIndex=0, chord=chord)])
            for i, chord in enumerate(chords)
        ],
    )


def _document(chords: list[str], *, timed: bool = True, key: str | None = "C major",
              duration: float | None = _DURATION) -> Song:
    """A Song carrying `chords`. Untimed is what the pipeline actually produces
    when nothing matched the audio: snap has no anchors to assign."""
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
    return Song.model_validate(
        {
            "id": "test--attributed-song",
            "metadata": {"title": "Attributed", "artist": "Test", "key": key},
            "audio": {"durationSeconds": duration},
            "lines": lines,
        }
    )


#: The 1966-live shape, to the extent a fixture can carry it: a chord timeline
#: that only reaches the first third of the track, so only the first third of
#: the document ever matched or got timed.
_LOFI_MATCHED_LINES = 9
_LOFI_DURATION = 220.6
_LOFI_TIMELINE_END = 86.5


def _lofi_document() -> Song:
    """What the pipeline really produces from an unusable recording.

    The lines the chord timeline reached are timed; every line after the last
    match holds that same timestamp, because snap.py never extrapolates past
    known data and the collapse guard leaves a run with no later anchor exactly
    as found. That is the honest output -- and it is exactly what a grade has to
    be able to say is not good enough to play along to.
    """
    chords = _TRUE_PROGRESSION * 2
    step = _LOFI_TIMELINE_END / _LOFI_MATCHED_LINES
    lines = []
    for i, chord in enumerate(chords):
        time = min(i, _LOFI_MATCHED_LINES - 1) * step
        confidence = 0.9 if i < _LOFI_MATCHED_LINES else 0.3
        lines.append(
            {
                "lineIndex": i,
                "lyrics": f"line {i} has real words in it",
                "timeSeconds": time,
                "confidence": confidence,
                "chordPlacements": [
                    {"charIndex": 0, "chord": chord, "timeSeconds": time,
                     "confidence": confidence}
                ],
            }
        )
    return Song.model_validate(
        {
            "id": "test--attributed-song",
            "metadata": {"title": "Attributed", "artist": "Test", "key": "C major"},
            "audio": {"durationSeconds": _LOFI_DURATION},
            "lines": lines,
        }
    )


def _attribute(document: Song, mir: MirAnalysis, candidates: list[CandidateSource]):
    grade = grade_song(document, mir, candidates)
    return grade, attribute_fault(grade, mir, candidates)


# --- the three patterns ------------------------------------------------------


def test_model_fault_when_the_sources_and_the_audio_agree_but_the_document_doesnt():
    mir = _mir(_TRUE_PROGRESSION)
    candidates = [_candidate(_TRUE_PROGRESSION, "web-1"), _candidate(_TRUE_PROGRESSION, "web-2")]
    # The document ignored both: chords that appear in neither, so nothing
    # snapped and nothing got timed.
    document = _document(_INVENTED, timed=False)

    grade, attribution = _attribute(document, mir, candidates)

    assert grade.verdict == "fail", grade.describe()
    assert attribution.fault is Fault.MODEL
    assert attribution.actionable is True
    assert attribution.candidate_agreement == 1.0
    assert attribution.candidate_mir_agreement == 1.0
    assert attribution.document_mir_agreement == 0.0
    assert "did not use it" in attribution.reason


def test_audio_fault_when_the_sources_agree_but_the_recording_contradicts_them():
    """Two independent sheets say the same thing and the chord recognizer heard
    something else entirely -- a lo-fi/live/broadcast recording. Retrying would
    fit the same unreliable timeline again."""
    mir = _mir(_OTHER_SONG)  # full-length timeline, but not this song's chords
    candidates = [_candidate(_TRUE_PROGRESSION, "web-1"), _candidate(_TRUE_PROGRESSION, "web-2")]
    document = _document(_TRUE_PROGRESSION, timed=False)  # faithful to the sheets

    grade, attribution = _attribute(document, mir, candidates)

    assert grade.verdict == "fail", grade.describe()
    assert attribution.fault is Fault.AUDIO
    assert attribution.actionable is False
    assert attribution.candidate_agreement == 1.0
    assert attribution.candidate_mir_agreement < 0.5
    assert "not fixable by a retry" in attribution.reason


def test_audio_fault_when_the_chord_timeline_dies_partway_through_the_track():
    """The 1966 live take: an MIR chord timeline that stops at 86.5s of 220.6s.
    Everything downstream is interpolating over silence."""
    mir = _mir(
        _TRUE_PROGRESSION[:_LOFI_MATCHED_LINES],
        duration=_LOFI_DURATION,
        span=_LOFI_TIMELINE_END,
    )
    candidates = [_candidate(_TRUE_PROGRESSION, "web-1"), _candidate(_TRUE_PROGRESSION, "web-2")]
    document = _lofi_document()

    grade, attribution = _attribute(document, mir, candidates)

    assert grade.verdict == "fail", grade.describe()
    assert attribution.fault is Fault.AUDIO
    assert attribution.actionable is False
    assert attribution.mir_timeline_coverage < 0.5
    assert "could not follow" in attribution.reason


def test_source_fault_when_the_candidates_contradict_each_other():
    mir = _mir(_TRUE_PROGRESSION)
    candidates = [
        _candidate(_TRUE_PROGRESSION, "web-1"),
        _candidate(_OTHER_SONG, "web-2"),
        _candidate(_THIRD_SONG, "web-3"),
    ]
    document = _document(_INVENTED, timed=False)

    grade, attribution = _attribute(document, mir, candidates)

    assert grade.verdict == "fail"
    assert attribution.fault is Fault.SOURCE
    assert attribution.actionable is True
    assert attribution.candidate_agreement < 0.6
    assert "contradict each" in attribution.reason


def test_no_candidate_carried_chords_is_a_source_fault_not_the_models():
    """The model had nothing to reconcile from, and a retry hands it the same
    nothing again."""
    mir = _mir(_TRUE_PROGRESSION)
    document = _document(_INVENTED, timed=False)
    grade, attribution = _attribute(document, mir, [])

    assert attribution.fault is Fault.SOURCE
    assert attribution.actionable is True
    assert "no text evidence" in attribution.reason


# --- transposition: a sheet in another key is GOOD evidence ------------------


def test_a_transposed_sheet_still_counts_as_agreeing_with_the_audio():
    """The distinction the whole scorer exists for. Both sheets are the same
    song, one written 3 semitones down; the audio is in the higher key. That
    is MODEL fault (good evidence, bad document), never SOURCE or AUDIO."""
    recording = [transpose_chord(c, 3) for c in _TRUE_PROGRESSION]
    mir = _mir(recording, key="Eb major")
    candidates = [_candidate(_TRUE_PROGRESSION, "web-1"), _candidate(recording, "web-2")]
    document = _document(_INVENTED, timed=False)

    scores = score_candidates(candidates, mir)
    assert [s.transposition for s in scores] == [3, 0]
    assert all(s.score == 1.0 for s in scores)

    _grade, attribution = _attribute(document, mir, candidates)
    assert attribution.fault is Fault.MODEL
    assert attribution.candidate_agreement == 1.0


# --- a warn is no longer "nothing to attribute" ------------------------------


def test_a_warn_with_a_failing_metric_gets_a_named_fault_not_none():
    """`sectionCoverage` failing (untimed sections) on an otherwise-healthy
    document holds the verdict at `warn` (that gate defaults OFF). Before
    this change, `warn` short-circuited straight to `Fault.NONE` -- "nothing
    to attribute" -- even though a metric measurably failed. It must now be
    routed through the same comparison logic a `fail` gets, and it must not
    come back NONE: sources, audio and document all agree here, which is a
    real, named, honest non-answer (UNKNOWN) rather than "nothing is
    wrong"."""
    mir = _mir(_TRUE_PROGRESSION)
    candidates = [_candidate(_TRUE_PROGRESSION, "web-1")]
    sections = [
        {"sectionIndex": 0, "name": "Verse 1", "kind": "verse",
         "startLineIndex": 0, "endLineIndex": 11},  # never timed
    ]
    document = Song.model_validate(
        _document(_TRUE_PROGRESSION, duration=90.0).model_dump() | {"sections": sections}
    )
    grade, attribution = _attribute(document, mir, candidates)

    assert grade.verdict == "warn", grade.describe()
    assert grade.metric("sectionCoverage").ok is False
    assert attribution.fault is not Fault.NONE
    assert attribution.fault is Fault.UNKNOWN
    assert "no retry" in attribution.reason


def test_a_warn_still_finds_a_model_fault_when_the_evidence_says_so():
    """The same routing, but where the comparison has something specific to
    say: sources and audio agree, the document doesn't -- MODEL fault, even
    though the overall verdict only ever reaches `warn` because nothing else
    failed hard enough to sink it below the fail line."""
    mir = _mir(_TRUE_PROGRESSION)
    candidates = [_candidate(_TRUE_PROGRESSION, "web-1"), _candidate(_TRUE_PROGRESSION, "web-2")]
    sections = [
        {"sectionIndex": 0, "name": "Verse 1", "kind": "verse",
         "startLineIndex": 0, "endLineIndex": 11},  # never timed -- sectionCoverage fails
    ]
    # A document that mostly ignores the evidence on JUST enough placements to
    # miss the chordMatchRatio threshold slightly while everything else (an
    # otherwise well-timed, well-covered document) keeps the overall above the
    # fail line.
    document = Song.model_validate(
        _document(_TRUE_PROGRESSION[:-6] + ["D", "E", "D", "E", "D", "E"], duration=90.0)
        .model_dump()
        | {"sections": sections}
    )
    grade, attribution = _attribute(document, mir, candidates)

    assert grade.verdict == "warn", grade.describe()
    assert attribution.fault is Fault.MODEL
    assert attribution.actionable is True


# --- the honest non-answers --------------------------------------------------


def test_a_passing_grade_attributes_nothing():
    mir = _mir(_TRUE_PROGRESSION)
    candidates = [_candidate(_TRUE_PROGRESSION, "web-1")]
    sections = [
        {"sectionIndex": 0, "name": "Verse 1", "kind": "verse", "startLineIndex": 0,
         "endLineIndex": 11, "startTime": 0.0, "endTime": _DURATION},
    ]
    document = Song.model_validate(
        _document(_TRUE_PROGRESSION).model_dump() | {"sections": sections}
    )
    grade, attribution = _attribute(document, mir, candidates)

    assert grade.verdict == "pass", grade.describe()
    assert attribution.fault is Fault.NONE
    assert attribution.actionable is False


def test_no_mir_means_nothing_can_be_attributed_and_nothing_is_retried():
    """A failing grade with no audio analysis cannot be compared to the
    recording, so it names no culprit -- and a retry with no new information
    would be exactly the non-convergence this module exists to prevent."""
    document = _document(_INVENTED, timed=False)
    grade = grade_song(document, None, [_candidate(_TRUE_PROGRESSION, "web-1")])
    attribution = attribute_fault(grade, None, [_candidate(_TRUE_PROGRESSION, "web-1")])

    assert attribution.fault is Fault.UNKNOWN
    assert attribution.actionable is False


def test_an_unknown_grade_attributes_nothing():
    # No MIR, no duration, no key: only lyric completeness is measurable, which
    # is not enough to grade on, let alone escalate.
    document = _document(["C", "G"], timed=False, key=None, duration=None)
    grade = grade_song(document)
    assert grade.verdict == "unknown", grade.describe()
    attribution = attribute_fault(grade, None, [])
    assert attribution.fault is Fault.UNKNOWN
    assert attribution.actionable is False


# --- the comparisons themselves ---------------------------------------------


def test_one_candidate_cannot_corroborate_anything():
    """Reporting 1.0 for a single source would read as "the sources agree" when
    nothing was compared."""
    assert candidate_agreement([_candidate(_TRUE_PROGRESSION, "web-1")]) is None


def test_candidate_agreement_is_transposition_invariant():
    same_song_two_keys = [
        _candidate(_TRUE_PROGRESSION, "web-1"),
        _candidate([transpose_chord(c, 2) for c in _TRUE_PROGRESSION], "web-2"),
    ]
    assert candidate_agreement(same_song_two_keys) == 1.0

    # Richer extensions and a missing passing chord are not disagreement either.
    variants = [
        _candidate(_TRUE_PROGRESSION, "web-1"),
        _candidate(["Cmaj7", "G", "Am7", "F", "Dm7", "G7", "Em", "Am", "F", "C", "Bb"], "web-2"),
    ]
    assert candidate_agreement(variants) > 0.9

    # Two different songs are.
    assert candidate_agreement(
        [_candidate(_TRUE_PROGRESSION, "web-1"), _candidate(_OTHER_SONG, "web-2")]
    ) < 0.6


def test_mir_timeline_coverage_reports_the_span_the_recognizer_produced():
    assert mir_timeline_coverage(_mir(_TRUE_PROGRESSION)) == 1.0
    assert mir_timeline_coverage(
        _mir(_TRUE_PROGRESSION, duration=_LOFI_DURATION, span=_LOFI_TIMELINE_END)
    ) < 0.4
    assert mir_timeline_coverage(MirAnalysis(duration_seconds=100.0, chords=[])) == 0.0
    assert mir_timeline_coverage(MirAnalysis(chords=[])) is None  # no duration to divide by


def test_the_coincidence_floor_the_audio_threshold_is_calibrated_against():
    """Why `mir_agreement` is 0.5 and not, say, 0.8.

    The matcher shared with timing.snap scans a 3-segment window on a 12-pitch
    alphabet, so unrelated sequences match in order surprisingly often. This
    pins the floor: a candidate set that cannot beat it is not describing the
    recording, and any threshold BELOW it would be unreachable noise.
    """
    import random

    from snoocle_server.chords import PITCH_CLASSES_SHARP
    from snoocle_server.quality.attribution import AttributionThresholds

    rng = random.Random(3)  # seeded: this is a measurement, not a flaky test
    scores = []
    for _ in range(50):
        sheet = [rng.choice(PITCH_CLASSES_SHARP) for _ in range(20)]
        heard = [rng.choice(PITCH_CLASSES_SHARP) for _ in range(20)]
        scores.append(score_candidates([_candidate(sheet, "web-1")], _mir(heard))[0].score)

    floor = sum(scores) / len(scores)
    assert 0.25 < floor < 0.5, floor
    assert AttributionThresholds().mir_agreement > floor


def test_attribution_serializes_every_comparison_it_made():
    mir = _mir(_TRUE_PROGRESSION)
    candidates = [_candidate(_TRUE_PROGRESSION, "web-1"), _candidate(_TRUE_PROGRESSION, "web-2")]
    _grade, attribution = _attribute(_document(_INVENTED, timed=False), mir, candidates)
    payload = attribution.to_dict()

    assert payload["fault"] == "model"
    assert payload["actionable"] is True
    assert payload["candidateAgreement"] == 1.0
    assert payload["documentMirAgreement"] == 0.0
    assert [s["sourceId"] for s in payload["candidateScores"]] == ["web-1", "web-2"]
