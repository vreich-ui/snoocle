"""A generic safety net for timings that collapse onto one timestamp,
independent of what upstream cause produced them.

Two real causes have already been seen: MIR beat data ending early on a
fade-out (Marley — addressed at its source in mir/beats.py), and an unusable
chord timeline on lo-fi source audio (a 1966 live recording where the last
line is timed at 86.5s of a 220.6s track and lines 17-20 all share
86.51755102040816 — 39% timing coverage, unflagged). This module is the guard
for the next cause nobody has found yet: it looks at the FINAL song, not at
why it got that way.
"""

from __future__ import annotations

from snoocle_server.schema.song import BeatMark, ChordPlacement, Line, Song
from snoocle_server.timing.collapse_guard import (
    COLLAPSE_RUN_THRESHOLD,
    guard_against_collapsed_timing,
    timing_coverage,
)

from test_schema import make_song


def _lined_song(times: list[float | None], *, one_placement_per_line: bool = False) -> Song:
    """A Song with one line per entry in `times`, lyrics long enough to hold
    a charIndex, and (optionally) a single chord placement per line carrying
    the same time as its line -- monotonic non-decreasing, as the schema
    requires, since duplicates are exactly what this module looks for."""
    data = make_song()
    data["audio"]["syncMap"] = []
    lines = []
    for i, t in enumerate(times):
        line = {"lineIndex": i, "lyrics": f"line number {i}", "chordPlacements": []}
        if t is not None:
            line["timeSeconds"] = t
        if one_placement_per_line:
            line["chordPlacements"] = [{"charIndex": 0, "chord": "C", "timeSeconds": t}]
        lines.append(line)
    data["lines"] = lines
    data["sections"] = [
        {
            "sectionIndex": 0,
            "name": "Verse 1",
            "kind": "verse",
            "startLineIndex": 0,
            "endLineIndex": len(times) - 1,
        }
    ]
    return Song.model_validate(data)


def _line_times(song: Song) -> list[float | None]:
    return [line.timeSeconds for line in song.lines]


# --- timing_coverage ---------------------------------------------------------


def test_coverage_is_full_when_lines_span_the_whole_duration():
    song = _lined_song([0.0, 30.0, 60.0, 100.0])
    assert timing_coverage(song.lines, 100.0) == 1.0


def test_coverage_is_partial_and_matches_the_reported_paint_it_black_figure():
    # The reported figure: last line at 86.5s of a 220.6s track -- ~39.2%,
    # unflagged before this guard existed.
    song = _lined_song([0.0, 40.0, 86.5])
    coverage = timing_coverage(song.lines, 220.6)
    assert round(coverage, 3) == round(86.5 / 220.6, 3)
    assert 0.39 < coverage < 0.40


def test_coverage_is_zero_for_an_entirely_untimed_document():
    song = _lined_song([None, None, None])
    assert timing_coverage(song.lines, 200.0) == 0.0


def test_coverage_is_unknown_without_a_duration():
    song = _lined_song([0.0, 50.0])
    assert timing_coverage(song.lines, None) is None
    assert timing_coverage(song.lines, 0.0) is None


def test_coverage_is_clamped_and_never_exceeds_one():
    # Defensive: times shouldn't validly exceed duration, but the figure a
    # human reads must never claim more than 100%.
    song = _lined_song([0.0, 50.0])
    assert timing_coverage(song.lines, 10.0) == 1.0


# --- the line-level collapse itself ------------------------------------------


def test_collapse_below_threshold_is_left_untouched():
    assert COLLAPSE_RUN_THRESHOLD == 3
    # exactly 2 identical times is unremarkable and must not be touched
    song = _lined_song([0.0, 10.0, 10.0, 20.0])
    result, entry = guard_against_collapsed_timing(song, duration=30.0)
    assert _line_times(result) == [0.0, 10.0, 10.0, 20.0]
    assert "no collapsed runs detected" in entry.notes


def test_a_synthetic_collapse_gets_spread_and_recorded():
    # lines 1-4 all collapse to 10.0, with a real anchor at 20.0 afterwards
    song = _lined_song([0.0, 10.0, 10.0, 10.0, 10.0, 20.0])
    result, entry = guard_against_collapsed_timing(song, duration=30.0)
    times = _line_times(result)

    assert times[0] == 0.0  # untouched before the run
    assert times[1] == 10.0  # the run's anchor -- left exactly as found
    spread = times[2:5]
    assert len(set(spread)) == 3  # no longer piled onto one timestamp
    assert spread == sorted(spread)
    assert all(10.0 < t < 20.0 for t in spread)
    assert times[5] == 20.0  # untouched after the run

    assert entry.action == "timing-collapse-guard"
    assert "spread" in entry.notes
    assert "3 line(s)" in entry.notes
    assert "10.00-20.00s" in entry.notes
    # syncMap regenerated to match the corrected line times
    assert [sp.time for sp in result.audio.syncMap] == [t for t in times if t is not None]


def test_a_collapse_with_no_later_anchor_is_left_alone_and_reported():
    # lines 2-5 collapse to 10.0 and NOTHING later breaks free of it (the
    # Paint It Black shape: the collapse reaches the end of the document).
    song = _lined_song([0.0, 10.0, 10.0, 10.0, 10.0])
    result, entry = guard_against_collapsed_timing(song, duration=100.0)

    assert _line_times(result) == [0.0, 10.0, 10.0, 10.0, 10.0]  # nothing invented
    assert "left alone" in entry.notes
    assert "no later anchor" in entry.notes
    assert "3 line(s)" in entry.notes  # lineIndex 2-4; index 1 is the run's own anchor
    assert "spread" not in entry.notes.split("left alone")[0]


def test_a_healthy_document_is_untouched_but_still_gets_a_provenance_entry():
    song = _lined_song([0.0, 5.0, 12.0, 19.5, 30.0])
    result, entry = guard_against_collapsed_timing(song, duration=30.0)

    assert _line_times(result) == _line_times(song)
    assert [l.chordPlacements for l in result.lines] == [l.chordPlacements for l in song.lines]
    assert len(result.provenance) == len(song.provenance) + 1
    assert entry.action == "timing-collapse-guard"
    assert "no collapsed runs detected" in entry.notes
    assert "timing coverage: 100.0%" in entry.notes


def test_the_provenance_entry_reports_coverage_alongside_the_intervention():
    song = _lined_song([0.0, 10.0, 10.0, 10.0, 20.0])
    _result, entry = guard_against_collapsed_timing(song, duration=40.0)
    assert "timing coverage: 50.0%" in entry.notes  # (20.0 - 0.0) / 40.0


def test_spread_uses_the_beat_grid_when_one_exists():
    song = _lined_song([0.0, 10.0, 10.0, 10.0, 20.0])
    grid = [
        BeatMark(time=t, measure=1 + i // 4, beatInMeasure=(i % 4) + 1)
        for i, t in enumerate([x * 0.5 for x in range(80)])  # every 0.5s, 0..39.5
    ]
    song = song.model_copy(update={"audio": song.audio.model_copy(update={"beats": grid})})
    result, _entry = guard_against_collapsed_timing(song, duration=40.0)
    spread = [t for t in _line_times(result)[2:4]]
    grid_times = {b.time for b in grid}
    assert all(any(abs(t - g) < 1e-6 for g in grid_times) for t in spread)


def test_spread_falls_back_to_even_division_without_a_beat_grid():
    song = _lined_song([0.0, 10.0, 10.0, 10.0, 20.0])  # no beats on this song
    result, _entry = guard_against_collapsed_timing(song, duration=40.0)
    spread = _line_times(result)[2:4]
    expected = [10.0 + (20.0 - 10.0) * (i + 1) / 3 for i in range(2)]
    assert spread == expected


def test_spread_times_are_strictly_increasing_and_the_song_stays_valid():
    # Song.model_validate enforces non-decreasing line times across the whole
    # document -- if the spread ever ties or inverts, this raises.
    song = _lined_song([0.0] + [5.0] * 6 + [6.0])
    result, _entry = guard_against_collapsed_timing(song, duration=10.0)
    times = [t for t in _line_times(result) if t is not None]
    assert times == sorted(times)
    assert len(set(times)) == len(times)


# --- placement-level collapses within a single line --------------------------


def _line_with_placement_times(times: list[float | None]) -> Line:
    placements = [
        ChordPlacement(charIndex=i * 5, chord="C", timeSeconds=t) for i, t in enumerate(times)
    ]
    return Line(lineIndex=0, lyrics="x " * (len(times) * 5), chordPlacements=placements)


def test_a_placement_level_collapse_within_a_line_gets_spread():
    line = _line_with_placement_times([0.0, 4.0, 4.0, 4.0, 8.0])
    song = make_song()
    song["audio"]["syncMap"] = []
    song["lines"] = [line.model_dump()]
    song["sections"] = [
        {"sectionIndex": 0, "name": "V", "kind": "verse", "startLineIndex": 0, "endLineIndex": 0}
    ]
    song = Song.model_validate(song)

    result, entry = guard_against_collapsed_timing(song, duration=20.0)
    times = [p.timeSeconds for p in result.lines[0].chordPlacements]

    assert times[0] == 0.0
    assert times[1] == 4.0  # the run's own anchor, untouched
    assert len(set(times[2:4])) == 2
    assert all(4.0 < t < 8.0 for t in times[2:4])
    assert times[4] == 8.0
    assert "placement(s) on line 0" in entry.notes
    # spread placements are marked interpolated-tier, not measured
    assert all(p.confidence == 0.3 for p in result.lines[0].chordPlacements[2:4])


def test_a_placement_level_collapse_with_no_later_in_line_anchor_is_left_alone():
    line = _line_with_placement_times([0.0, 4.0, 4.0, 4.0])  # collapse reaches the line's end
    song = make_song()
    song["audio"]["syncMap"] = []
    song["lines"] = [line.model_dump()]
    song["sections"] = [
        {"sectionIndex": 0, "name": "V", "kind": "verse", "startLineIndex": 0, "endLineIndex": 0}
    ]
    song = Song.model_validate(song)

    result, entry = guard_against_collapsed_timing(song, duration=20.0)
    times = [p.timeSeconds for p in result.lines[0].chordPlacements]

    assert times == [0.0, 4.0, 4.0, 4.0]
    assert "placement(s) on line 0" in entry.notes
    assert "no later in-line anchor" in entry.notes
