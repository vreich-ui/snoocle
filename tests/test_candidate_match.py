"""Scoring one candidate sheet against the audio it claims to describe.

The distinction this exists to make: a sheet in the "wrong" key is GOOD
evidence that needs transposing, and scoring it as garbage is what makes a
re-search look necessary when it isn't. So every score is the best of the 12
transpositions, and the shift it needed is reported alongside.
"""

from __future__ import annotations

from snoocle_server.discovery.models import CandidateSource
from snoocle_server.mir.base import ChordSegment, MirAnalysis
from snoocle_server.reconcile.match import score_candidate, score_candidates
from snoocle_server.schema.song import ChordPlacement, Line

# A plain I-V-vi-IV in G, four bars of five seconds each.
_PROGRESSION = ["G", "D", "Em", "C"]


def _mir(chords: list[str], *, start: float = 0.0, step: float = 5.0) -> MirAnalysis:
    return MirAnalysis(
        duration_seconds=start + step * len(chords),
        chords=[
            ChordSegment(start=start + i * step, end=start + (i + 1) * step, chord=c)
            for i, c in enumerate(chords)
        ],
    )


def _candidate(chords: list[str], source_id: str = "web-1") -> CandidateSource:
    """One chord per line, so lineIndex/charIndex in a conflict are meaningful."""
    return CandidateSource(
        sourceId=source_id,
        lines=[
            Line(
                lineIndex=i,
                lyrics=f"line {i} of the sheet",
                chordPlacements=[ChordPlacement(charIndex=0, chord=chord)],
            )
            for i, chord in enumerate(chords)
        ],
    )


def test_a_sheet_in_the_recordings_key_scores_high_at_zero_transposition():
    score = score_candidate(_candidate(_PROGRESSION), _mir(_PROGRESSION))
    assert (score.matched, score.total) == (4, 4)
    assert score.score == 1.0
    assert score.transposition == 0
    assert score.conflicts == []


def test_a_transposed_but_correct_sheet_scores_high_at_a_nonzero_transposition():
    """The whole point: a sheet in G against a recording in Bb is a good source
    needing +3, not a bad source."""
    recording_in_bb = ["Bb", "F", "Gm", "Eb"]  # the same progression, up 3
    score = score_candidate(_candidate(_PROGRESSION), _mir(recording_in_bb))

    assert score.score == 1.0
    assert score.transposition == 3
    assert (score.matched, score.total) == (4, 4)
    assert score.conflicts == []


def test_the_smallest_shift_wins_a_tie_and_zero_wins_outright():
    """A sheet already in the recording's key must never be reported as
    transposed -- a chromatic timeline matches at every shift."""
    chromatic = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    score = score_candidate(_candidate(["C"]), _mir(chromatic))
    assert score.score == 1.0
    assert score.transposition == 0


def test_a_wrong_song_sheet_scores_low_at_all_twelve_transpositions():
    """No shift rescues a sheet that is not this song: the progression walks
    the circle differently, so nothing lines up in reading order."""
    wrong_song = ["F#m", "A", "E", "B", "F#m", "A", "E", "B"]
    score = score_candidate(_candidate(wrong_song), _mir(_PROGRESSION))
    assert score.score < 0.5, score.describe()  # 3/8 at its best shift

    # And it is low because it is wrong, not because scoring is broken: the
    # right sheet against the same timeline scores 1.0.
    assert score_candidate(_candidate(_PROGRESSION), _mir(_PROGRESSION)).score == 1.0


def test_family_clashes_appear_in_conflicts_without_lowering_the_score():
    """The root matched, which is what carries the alignment. "The sheet says
    Em where the audio says E" is still exactly what a reviewer wants to see."""
    audio_heard_major = ["G", "D", "E", "C"]  # sheet says Em on the third chord
    score = score_candidate(_candidate(_PROGRESSION), _mir(audio_heard_major))

    assert score.score == 1.0  # every root matched
    assert len(score.conflicts) == 1
    conflict = score.conflicts[0]
    assert conflict["lineIndex"] == 2
    assert conflict["chord"] == "Em"
    assert conflict["audioChord"] == "E"
    assert conflict["timeSeconds"] == 10.0
    assert "family clash" in conflict["reason"]


def test_a_richer_extension_of_the_same_root_is_not_a_conflict():
    """MIR's chord vocabulary is coarser than a sheet's -- plain C where the
    sheet says Cmaj7 is agreement, per the reconcile prompt's own semantics."""
    score = score_candidate(_candidate(["Cmaj7", "G7", "Am7"]), _mir(["C", "G", "Am"]))
    assert score.score == 1.0
    assert score.conflicts == []


def test_nothing_to_compare_reports_zero_with_the_denominator_visible():
    """"no chords to compare" and "chords that match nothing" both score 0.0 --
    the counts are what tells them apart, so both are returned."""
    no_chords = score_candidate(_candidate([]), _mir(_PROGRESSION))
    assert (no_chords.score, no_chords.matched, no_chords.total) == (0.0, 0, 0)

    no_timeline = score_candidate(_candidate(_PROGRESSION), _mir([]))
    assert (no_timeline.score, no_timeline.matched, no_timeline.total) == (0.0, 0, 4)

    no_mir_at_all = score_candidate(_candidate(_PROGRESSION), None)
    assert (no_mir_at_all.score, no_mir_at_all.total) == (0.0, 4)


def test_no_chord_segments_are_ignored_like_everywhere_else():
    mir = MirAnalysis(
        duration_seconds=20.0,
        chords=[
            ChordSegment(start=0.0, end=5.0, chord="N"),
            ChordSegment(start=5.0, end=10.0, chord="G"),
            ChordSegment(start=10.0, end=15.0, chord="D"),
        ],
    )
    assert score_candidate(_candidate(["G", "D"]), mir).score == 1.0


def test_score_candidates_preserves_order_and_labels_each_source():
    scores = score_candidates(
        [_candidate(_PROGRESSION, "web-1"), _candidate(["F#m"], "web-2")],
        _mir(_PROGRESSION),
    )
    assert [s.source_id for s in scores] == ["web-1", "web-2"]
    assert scores[0].to_dict()["sourceId"] == "web-1"


def test_the_scorer_and_snap_use_the_same_matcher():
    """Not a style point: a grade computed from a second, subtly different
    matcher would measure something the pipeline never did."""
    from snoocle_server.timing import root_match, snap

    assert snap.match_chord_roots is root_match.match_chord_roots
    from snoocle_server.reconcile import match

    assert match.match_chord_roots is root_match.match_chord_roots
