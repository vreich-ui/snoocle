from __future__ import annotations

import json

from snoocle_server.deterministic import (
    align_song_deterministically_service,
    build_song_from_candidate,
    select_candidate_deterministically,
)
from snoocle_server.discovery.models import CandidateSource, SectionStart
from snoocle_server.mir.base import Beat, ChordSegment, MirAnalysis
from snoocle_server.schema import ChordPlacement, Line


def _candidate(source_id: str = "sheet-a", chord: str = "C") -> CandidateSource:
    return CandidateSource(
        sourceId=source_id,
        retrievedAt="2026-07-31T00:00:00Z",
        confidence=0.9,
        parseConfidence=0.9,
        coverage="full-song",
        sectionStarts=[SectionStart(name="Verse 1", startLineIndex=0)],
        sectionsHint=["Verse 1"],
        lines=[
            Line(
                lineIndex=index,
                lyrics=text,
                chordPlacements=[ChordPlacement(charIndex=0, chord=chord)],
            )
            for index, text in enumerate(["one", "two", "three", "four"])
        ],
    )


def _mir(chord: str = "C") -> MirAnalysis:
    return MirAnalysis(
        engines={"beats": "test", "chords": "test", "structure": "test"},
        duration_seconds=16,
        bpm=60,
        time_signature="4/4",
        key="C major",
        beats=[Beat(time=float(index), position=(index % 4) + 1) for index in range(16)],
        chords=[
            ChordSegment(start=float(index * 4), end=float((index + 1) * 4), chord=chord)
            for index in range(4)
        ],
    )


def test_baseline_preserves_source_content_and_has_no_speculative_timing():
    candidate = _candidate()
    song = build_song_from_candidate(
        candidate,
        song_id="artist--title",
        title="Title",
        artist="Artist",
        youtube_video_id="abcdefghijk",
    )

    assert [line.lyrics for line in song.lines] == [line.lyrics for line in candidate.lines]
    assert [
        [(placement.charIndex, placement.chord) for placement in line.chordPlacements]
        for line in song.lines
    ] == [
        [(placement.charIndex, placement.chord) for placement in line.chordPlacements]
        for line in candidate.lines
    ]
    assert all(line.timeSeconds is None for line in song.lines)
    assert all(
        placement.timeSeconds is None
        for line in song.lines
        for placement in line.chordPlacements
    )
    assert song.displayPreferences.capo == 0
    assert song.sections[0].name == "Verse 1"


def test_same_inputs_produce_identical_song_and_zero_model_usage():
    candidate = _candidate()
    baseline = build_song_from_candidate(
        candidate,
        song_id="artist--title",
        title="Title",
        artist="Artist",
    )

    first = align_song_deterministically_service(baseline, _mir(), candidates=[candidate])
    second = align_song_deterministically_service(baseline, _mir(), candidates=[candidate])

    assert first.song.model_dump(mode="json") == second.song.model_dump(mode="json")
    assert first.report["modelCalls"] == 0
    assert first.report["modelCostUSD"] == 0
    assert all(stage.to_dict()["modelCalls"] == 0 for stage in first.observations)


def test_strict_selection_stops_on_close_but_different_sources():
    selection = select_candidate_deterministically(
        [_candidate("a", "C"), _candidate("b", "G")],
        strategy="strict",
    )

    assert selection.status == "needs_review"
    assert selection.reason == "candidate_sources_conflict"
    assert selection.selected is None


def test_conflict_packet_excludes_lyrics_song_beats_and_provenance():
    candidate = _candidate()
    baseline = build_song_from_candidate(
        candidate,
        song_id="artist--title",
        title="Title",
        artist="Artist",
    )

    result = align_song_deterministically_service(
        baseline,
        _mir("G"),
        candidates=[candidate],
    )
    serialized = json.dumps(result.conflict_packet).casefold()

    assert "lyrics" not in serialized
    assert "provenance" not in serialized
    assert '"beats"' not in serialized
    assert '"song"' not in serialized
    assert "one" not in serialized
