"""Mode B's deterministic core: transposition, timing clearance, structure.

These are the parts that must be exactly right for a re-alignment to be worth
having: the chords must land in the new recording's key, the OLD recording's
times must be gone rather than lingering as plausible-looking fiction, and the
structural comparison must only cry "different" when something really is —
since it is the only thing that can spend model tokens.
"""

from __future__ import annotations

import pytest

from snoocle_server.chords import transpose_chord
from snoocle_server.mir.base import ChordSegment, MirAnalysis, StructureSegment
from snoocle_server.schema import Song
from snoocle_server.timing.realign import (
    MIN_TRANSPOSITION_SCORE,
    apply_transposition,
    clear_recording_timing,
    compare_structure,
    derive_transposition,
    song_as_candidate,
    stored_recording_duration,
    transpose_key_name,
)

PROGRESSION = ["C", "G", "Am", "F", "Dm", "G", "Em", "Am", "F", "C", "Bb", "F"]
OTHER_SONG = ["Dm", "Eb", "Dm", "Bb", "Gm", "Db", "Dm", "Ab", "Dm", "Eb", "Cm", "Dm"]
DURATION = 60.0
LYRIC = "line {i} has words that must survive verbatim"


def _mir(
    chords: list[str],
    *,
    duration: float = DURATION,
    sections: int = 4,
    key: str = "C major",
) -> MirAnalysis:
    step = duration / len(chords)
    seg = duration / max(sections, 1)
    return MirAnalysis(
        engines={"chords": "chord-cnn-lstm", "structure": "songformer"},
        duration_seconds=duration,
        key=key,
        chords=[
            ChordSegment(start=i * step, end=(i + 1) * step, chord=c)
            for i, c in enumerate(chords)
        ],
        sections=[
            StructureSegment(start=i * seg, end=(i + 1) * seg, label="verse")
            for i in range(sections)
        ],
    )


def _song(
    chords: list[str] = tuple(PROGRESSION),
    *,
    timed: bool = True,
    sections: int = 4,
    duration: float | None = DURATION,
    key: str | None = "C major",
) -> Song:
    lines = []
    for i, chord in enumerate(chords):
        timing = {"timeSeconds": i * 5.0, "confidence": 0.9} if timed else {}
        lines.append(
            {
                "lineIndex": i,
                "lyrics": LYRIC.format(i=i),
                **timing,
                "chordPlacements": [
                    {
                        "charIndex": 5,
                        "chord": chord,
                        **({"timeSeconds": i * 5.0, "confidence": 0.9,
                            "beat": {"measure": i + 1, "beat": 1}} if timed else {}),
                    }
                ],
            }
        )
    per = max(1, len(chords) // max(sections, 1))
    section_list = []
    for s in range(sections):
        start = s * per
        end = (s + 1) * per - 1 if s < sections - 1 else len(chords) - 1
        if start > end:
            continue
        section_list.append(
            {
                "sectionIndex": s,
                "name": f"Section {s}",
                "kind": "verse",
                "startLineIndex": start,
                "endLineIndex": end,
                **({"startTime": start * 5.0, "endTime": (end + 1) * 5.0} if timed else {}),
            }
        )
    return Song.model_validate(
        {
            "id": "test--mode-b",
            "metadata": {"title": "Mode B", "artist": "Test", "key": key, "bpm": 120.0},
            "audio": {
                "youtubeVideoId": "aaaaaaaaaaa",
                "analyzedVideoId": "aaaaaaaaaaa",
                "durationSeconds": duration,
                "videoOffsets": {"bbbbbbbbbbb": 2.5},
                "beats": [{"time": 0.0, "measure": 1, "beatInMeasure": 1}] if timed else [],
                "syncMap": [{"lineIndex": i, "time": i * 5.0} for i in range(len(chords))]
                if timed
                else [],
            },
            "sections": section_list,
            "lines": lines,
            "provenance": [
                {"timestamp": "2026-07-30T00:00:00Z", "actor": "reconcile:test/x",
                 "action": "reconciled"}
            ],
        }
    )


# --- transposition ----------------------------------------------------------


def test_the_transposition_search_is_the_graders_not_a_second_one():
    """The brief's requirement, and a real one: two searches would drift."""
    from snoocle_server.reconcile import match
    from snoocle_server.timing import realign

    assert realign.score_candidate is match.score_candidate

    # And the document goes into it as a candidate source without conversion,
    # because CandidateSource.lines IS schema.song.Line.
    song = _song()
    assert song_as_candidate(song).lines == list(song.lines)


def test_a_recording_three_semitones_up_derives_plus_three():
    recording = [transpose_chord(c, 3) for c in PROGRESSION]
    estimate = derive_transposition(_song(), _mir(recording, key="Eb major"))

    assert estimate.semitones == 3
    assert estimate.trustworthy is True
    assert estimate.applies is True
    assert estimate.score.score == 1.0
    assert estimate.key_semitones == 3  # the key strings corroborate
    assert "+3 semitone(s)" in estimate.reason


def test_the_same_key_derives_zero_and_does_not_apply():
    estimate = derive_transposition(_song(), _mir(PROGRESSION))
    assert (estimate.semitones, estimate.trustworthy, estimate.applies) == (0, True, False)


def test_a_recording_of_a_different_song_is_not_trusted_to_transpose():
    """The coincidence floor: at ~0.4 the best shift says nothing, and moving
    the document's chords on that basis would be pure damage."""
    estimate = derive_transposition(_song(), _mir(OTHER_SONG))

    assert estimate.trustworthy is False
    assert estimate.semitones == 0
    assert estimate.score.score < MIN_TRANSPOSITION_SCORE
    assert "indistinguishable from coincidence" in estimate.reason


def test_a_disagreeing_key_string_is_recorded_and_loses_to_the_chord_search():
    """One key guess per recording vs a whole sequence compared: the sequence
    wins, and the disagreement is written down rather than hidden."""
    recording = [transpose_chord(c, 3) for c in PROGRESSION]
    estimate = derive_transposition(_song(), _mir(recording, key="F major"))

    assert estimate.semitones == 3
    assert estimate.key_semitones == 5
    assert "suggest +5 instead" in estimate.reason
    assert "chord-sequence search wins" in estimate.reason


def test_apply_transposition_moves_chords_and_the_key_and_nothing_else():
    song = _song()
    moved = apply_transposition(song, 3)

    assert [p.chord for line in moved.lines for p in line.chordPlacements] == [
        transpose_chord(c, 3) for c in PROGRESSION
    ]
    assert moved.metadata.key == "Eb major"
    # Byte-identical lyrics and untouched placement positions.
    assert [line.lyrics for line in moved.lines] == [line.lyrics for line in song.lines]
    assert [p.charIndex for line in moved.lines for p in line.chordPlacements] == [
        p.charIndex for line in song.lines for p in line.chordPlacements
    ]
    assert [(s.startLineIndex, s.endLineIndex) for s in moved.sections] == [
        (s.startLineIndex, s.endLineIndex) for s in song.sections
    ]
    assert moved.provenance == song.provenance


def test_transposing_by_zero_is_the_same_document():
    song = _song()
    assert apply_transposition(song, 0) is song
    assert apply_transposition(song, 12) is song


@pytest.mark.parametrize(
    "key,semitones,expected",
    [
        ("C major", 3, "Eb major"),
        ("C major", 2, "D major"),
        ("A minor", -2, "G minor"),
        ("Bb major", 1, "B major"),
        ("F# minor", 3, "A minor"),
        ("not a key", 3, "not a key"),  # unparseable stays put
    ],
)
def test_key_names_transpose_and_keep_their_mode(key, semitones, expected):
    assert transpose_key_name(key, semitones) == expected


# --- clearing the old recording's timing -------------------------------------


def test_clearing_timing_removes_every_trace_of_the_old_recording():
    """The load-bearing step. A placement keeping its old time because nothing
    in the new timeline matched it is a time from a different performance,
    presented as measured."""
    cleared = clear_recording_timing(_song())

    assert all(line.timeSeconds is None and line.confidence is None for line in cleared.lines)
    assert all(
        p.timeSeconds is None and p.confidence is None and p.beat is None
        for line in cleared.lines
        for p in line.chordPlacements
    )
    assert all(s.startTime is None and s.endTime is None for s in cleared.sections)
    assert cleared.audio.syncMap == []
    assert cleared.audio.beats == []
    assert cleared.audio.analyzedVideoId is None
    assert cleared.audio.durationSeconds is None
    assert cleared.metadata.bpm is None
    # videoOffsets are corrections relative to the OLD reference: keeping one
    # would silently mis-shift playback anchored somewhere else.
    assert cleared.audio.videoOffsets == {}


def test_clearing_timing_preserves_everything_the_document_owns():
    song = _song()
    cleared = clear_recording_timing(song)

    assert [line.lyrics for line in cleared.lines] == [line.lyrics for line in song.lines]
    assert [p.chord for line in cleared.lines for p in line.chordPlacements] == PROGRESSION
    assert [p.charIndex for line in cleared.lines for p in line.chordPlacements] == [
        p.charIndex for line in song.lines for p in line.chordPlacements
    ]
    assert [(s.sectionIndex, s.name, s.startLineIndex) for s in cleared.sections] == [
        (s.sectionIndex, s.name, s.startLineIndex) for s in song.sections
    ]
    assert cleared.provenance == song.provenance
    assert cleared.metadata.key == song.metadata.key


# --- structural comparison ---------------------------------------------------


def test_a_structurally_identical_recording_is_explained():
    comparison = compare_structure(_song(), _mir(PROGRESSION))

    assert comparison.explained is True
    assert comparison.comparable is True
    assert comparison.duration_delta == 0.0
    assert comparison.describe() == "the stored document explains this recording's structure"
    assert comparison.reasons == ["duration and section count both fit the stored document"]


def test_a_small_length_difference_is_still_the_same_arrangement():
    """Two masters of the same song differ by a fade; that is not an arrangement
    change, and treating it as one would spend tokens on nothing."""
    comparison = compare_structure(_song(), _mir(PROGRESSION, duration=DURATION * 1.05))
    assert comparison.explained is True


def test_a_materially_longer_recording_is_a_structural_difference():
    comparison = compare_structure(_song(), _mir(PROGRESSION, duration=DURATION * 1.5))

    assert comparison.explained is False
    assert comparison.duration_delta == pytest.approx(0.5)
    assert "50% longer" in comparison.reasons[0]
    assert "repeats this document does not have" in comparison.reasons[0]


def test_a_truncated_recording_is_a_structural_difference():
    comparison = compare_structure(_song(), _mir(PROGRESSION, duration=DURATION * 0.7))

    assert comparison.explained is False
    assert "shorter" in comparison.reasons[0]
    assert "truncated" in comparison.reasons[0]


def test_a_gross_section_count_difference_is_a_structural_difference():
    comparison = compare_structure(_song(sections=2), _mir(PROGRESSION, sections=9))

    assert comparison.explained is False
    assert any("structure timeline has 9 segment(s)" in r for r in comparison.reasons)


def test_a_small_section_count_difference_is_tolerated():
    """The stored sections come from a human-named sheet and the segments from
    SongFormer or a librosa novelty fallback. They are allowed to disagree."""
    comparison = compare_structure(_song(sections=4), _mir(PROGRESSION, sections=6))
    assert comparison.explained is True


def test_an_unmeasurable_comparison_is_reported_and_is_not_a_difference():
    """"I cannot tell" must never be the reason a model gets called."""
    untimed = _song(timed=False, duration=None, sections=0)
    comparison = compare_structure(untimed, MirAnalysis(chords=[], sections=[]))

    assert comparison.comparable is False
    assert comparison.explained is True
    assert "not comparable" in comparison.describe()


def test_stored_duration_falls_back_to_the_span_the_document_covers():
    assert stored_recording_duration(_song()) == DURATION
    no_duration = _song(duration=None)
    assert stored_recording_duration(no_duration) == pytest.approx(60.0)
    assert stored_recording_duration(_song(timed=False, duration=None)) is None
