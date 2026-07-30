from snoocle_server.mir.base import Beat, ChordSegment, MirAnalysis
from snoocle_server.schema.song import Song
from snoocle_server.timing.snap import snap_chords

from test_schema import make_song


def _line(song: Song, idx: int):
    return next(l for l in song.lines if l.lineIndex == idx)


def test_mir_none_is_a_noop():
    song = Song.model_validate(make_song())
    result = snap_chords(song, None)
    assert result is song
    assert result.provenance == song.provenance


def test_full_match_sets_high_confidence_times_and_syncmap():
    song = Song.model_validate(make_song())
    mir = MirAnalysis(
        duration_seconds=12.0,
        engines={"chords": "chord-cnn-lstm"},
        chords=[
            ChordSegment(start=0.0, end=3.0, chord="C"),
            ChordSegment(start=3.0, end=6.0, chord="G"),
            ChordSegment(start=6.0, end=9.0, chord="Am"),
            ChordSegment(start=9.0, end=12.0, chord="F"),
        ],
    )
    result = snap_chords(song, mir)

    line0 = _line(result, 0)
    assert [p.chord for p in line0.chordPlacements] == ["C", "G", "Am"]
    assert [p.timeSeconds for p in line0.chordPlacements] == [0.0, 3.0, 6.0]
    assert all(p.confidence == 0.9 for p in line0.chordPlacements)
    assert line0.timeSeconds == 0.0

    line1 = _line(result, 1)
    assert line1.chordPlacements[0].timeSeconds == 9.0
    assert line1.timeSeconds == 9.0

    assert [(sp.lineIndex, sp.time) for sp in result.audio.syncMap] == [(0, 0.0), (1, 9.0)]

    # coverage acceptance: >=90% of placements timed
    total = sum(len(l.chordPlacements) for l in result.lines)
    timed = sum(1 for l in result.lines for p in l.chordPlacements if p.timeSeconds is not None)
    assert timed / total >= 0.9


def test_family_mismatch_gets_lower_confidence():
    song = Song.model_validate(make_song())
    # Only the first line's chords matter here; MIR heard C as major (agrees),
    # but heard the sheet's "Am" as a plain major "A" -- a genuine
    # major/minor clash on a matched root.
    mir = MirAnalysis(
        chords=[
            ChordSegment(start=0.0, end=3.0, chord="C"),
            ChordSegment(start=3.0, end=6.0, chord="G"),
            ChordSegment(start=6.0, end=9.0, chord="A"),  # sheet says Am
        ]
    )
    result = snap_chords(song, mir)
    line0 = _line(result, 0)
    am_placement = next(p for p in line0.chordPlacements if p.chord == "Am")
    assert am_placement.timeSeconds == 6.0
    assert am_placement.confidence == 0.7


def test_unmatched_placement_interpolates_between_neighbors():
    song = Song.model_validate(make_song())
    # MIR is missing the G segment entirely (dropped a passing chord).
    mir = MirAnalysis(
        chords=[
            ChordSegment(start=0.0, end=3.0, chord="C"),
            ChordSegment(start=6.0, end=9.0, chord="Am"),
        ]
    )
    result = snap_chords(song, mir)
    line0 = _line(result, 0)
    c, g, am = line0.chordPlacements
    assert (c.timeSeconds, c.confidence) == (0.0, 0.9)
    assert g.timeSeconds == 3.0  # interpolated halfway between C@0 and Am@6
    assert g.confidence == 0.3
    assert (am.timeSeconds, am.confidence) == (6.0, 0.9)

    # F (line 1) has nothing after it to interpolate toward -- holds the
    # last known time rather than extrapolating.
    line1 = _line(result, 1)
    assert line1.chordPlacements[0].timeSeconds == 6.0
    assert line1.chordPlacements[0].confidence == 0.3


def test_zero_mir_chord_matches_leaves_song_lines_unchanged_but_appends_provenance():
    song = Song.model_validate(make_song())
    mir = MirAnalysis(chords=[])  # MIR ran but found nothing
    result = snap_chords(song, mir)
    for line in result.lines:
        assert line.timeSeconds is None
        for p in line.chordPlacements:
            assert p.timeSeconds is None
    assert result.audio.syncMap == []
    assert result.provenance[-1].action == "timing-snap"
    assert result.provenance[-1].confidence == 0.0  # 0 matched of 4 total placements


def test_provenance_entry_records_match_stats():
    song = Song.model_validate(make_song())
    mir = MirAnalysis(
        engines={"chords": "chord-cnn-lstm", "beats": "madmom"},
        chords=[
            ChordSegment(start=0.0, end=3.0, chord="C"),
            ChordSegment(start=3.0, end=6.0, chord="G"),
            ChordSegment(start=6.0, end=9.0, chord="Am"),
            ChordSegment(start=9.0, end=12.0, chord="F"),
        ],
    )
    result = snap_chords(song, mir)
    entry = result.provenance[-1]
    assert entry.actor == "snoocle-server/timing"
    assert entry.action == "timing-snap"
    assert entry.confidence == 1.0
    assert "4/4" in entry.notes
    assert set(entry.sources) == {"beats:madmom", "chords:chord-cnn-lstm"}


def test_beat_snap_attaches_beatref_within_tolerance():
    song = Song.model_validate(make_song())
    mir = MirAnalysis(
        beats=[
            Beat(time=0.0, position=1),
            Beat(time=0.5, position=0),
            Beat(time=1.0, position=0),
        ],
        chords=[
            # 0.05s off the beat at 0.0 -- within the 0.35s snap tolerance.
            ChordSegment(start=0.05, end=3.0, chord="C"),
            ChordSegment(start=3.0, end=6.0, chord="G"),
            ChordSegment(start=6.0, end=9.0, chord="Am"),
        ],
    )
    result = snap_chords(song, mir)
    c = _line(result, 0).chordPlacements[0]
    assert c.timeSeconds == 0.0  # snapped onto the beat, not 0.05
    assert c.beat is not None
    assert c.beat.measure == 1 and c.beat.beat == 1.0
    # beats grid gets filled onto the song since it started empty
    assert len(result.audio.beats) == 3


def test_bpm_filled_from_mir_when_song_metadata_lacks_it():
    song = Song.model_validate(make_song())
    assert song.metadata.bpm is None
    mir = MirAnalysis(bpm=128.0, chords=[])
    result = snap_chords(song, mir)
    assert result.metadata.bpm == 128.0


def test_existing_bpm_and_beats_are_not_overwritten():
    data = make_song()
    data["metadata"]["bpm"] = 90.0
    data["audio"]["beats"] = [{"time": 0.0, "measure": 1, "beatInMeasure": 1}]
    song = Song.model_validate(data)
    mir = MirAnalysis(bpm=128.0, beats=[Beat(time=0.0, position=1), Beat(time=1.0, position=0)])
    result = snap_chords(song, mir)
    assert result.metadata.bpm == 90.0
    assert len(result.audio.beats) == 1  # kept the original, not MIR's 2-beat grid


def test_song_with_no_lines_does_not_crash():
    data = make_song()
    data["lines"] = []
    data["sections"] = []
    song = Song.model_validate(data)
    mir = MirAnalysis(chords=[ChordSegment(start=0.0, end=1.0, chord="C")])
    result = snap_chords(song, mir)
    assert result.lines == []
    assert result.audio.syncMap == []
    assert result.provenance[-1].action == "timing-snap"
    assert result.provenance[-1].confidence is None  # 0/0 placements -- undefined, not 0%


# --- fade-out endings ---------------------------------------------------------
#
# The failure these cover, from the stored "Three Little Birds" document: beat
# tracking lost lock 15s before the end of the track, so every placement past
# the last matched chord piled onto one timestamp -- the outro stopped
# scrolling. The beat grid now runs to the end (tempo-continued through the
# fade), which gives the tail something to spread across.


def _fade_out_song():
    """3 lines x 4 placements; MIR only heard the first line's worth."""
    data = make_song()
    data["lines"] = [
        {
            "lineIndex": i,
            "lyrics": "Don't worry about a thing, every little thing",
            "chordPlacements": [
                {"charIndex": 0, "chord": "C"},
                {"charIndex": 10, "chord": "G"},
                {"charIndex": 20, "chord": "Am"},
                {"charIndex": 30, "chord": "F"},
            ],
        }
        for i in range(3)
    ]
    data["sections"] = [
        {
            "sectionIndex": 0,
            "name": "Chorus",
            "kind": "chorus",
            "startLineIndex": 0,
            "endLineIndex": 2,
        }
    ]
    data["audio"]["syncMap"] = []
    return Song.model_validate(data)


def _fade_out_mir(beats_reach_end: bool) -> MirAnalysis:
    detected_until = 8.0
    beats = [
        Beat(time=i * 0.5, position=(i % 4) + 1, detected=(i * 0.5 <= detected_until))
        for i in range(60)  # 0.0 .. 29.5s
    ]
    if not beats_reach_end:
        beats = [b for b in beats if b.detected]
    return MirAnalysis(
        engines={"beats": "librosa-fallback", "chords": "chroma-template-fallback"},
        duration_seconds=30.0,
        beats=beats,
        chords=[
            ChordSegment(start=0.0, end=2.0, chord="C"),
            ChordSegment(start=2.0, end=4.0, chord="G"),
            ChordSegment(start=4.0, end=6.0, chord="Am"),
            ChordSegment(start=6.0, end=8.0, chord="F"),
        ],
    )


def _times(song: Song) -> list[float]:
    return [p.timeSeconds for line in song.lines for p in line.chordPlacements]


def test_placements_past_the_last_match_no_longer_collapse_onto_one_time():
    result = snap_chords(_fade_out_song(), _fade_out_mir(beats_reach_end=True))
    times = _times(result)

    assert times[:4] == [0.0, 2.0, 4.0, 6.0]  # matched against MIR
    tail = times[4:]
    assert len(set(tail)) == len(tail), f"tail collapsed onto shared timestamps: {tail}"
    assert tail == sorted(tail)
    assert tail[0] > 6.0
    assert tail[-1] < 30.0  # never past the end of the track
    # and they spread across the previously dead window, not just past it
    assert tail[-1] > 20.0


def test_line_times_and_syncmap_follow_the_spread_out_tail():
    result = snap_chords(_fade_out_song(), _fade_out_mir(beats_reach_end=True))
    line_times = [line.timeSeconds for line in result.lines]
    assert line_times == sorted(line_times)
    assert len(set(line_times)) == 3  # every line gets its own start, not one shared
    assert [sp.lineIndex for sp in result.audio.syncMap] == [0, 1, 2]


def test_a_grid_that_stops_at_the_fade_still_spreads_across_the_duration():
    """A MIR result whose beats stop short anyway — one analyzed before grids
    were tempo-continued, replayed from cache. The analyzed duration is still
    a real anchor, so the tail spreads rather than collapsing; it just has no
    beats to snap onto out there."""
    result = snap_chords(_fade_out_song(), _fade_out_mir(beats_reach_end=False))
    tail = _times(result)[4:]
    assert len(set(tail)) == len(tail)
    assert tail == sorted(tail)
    assert tail[-1] < 30.0
    past_the_grid = [
        p for line in result.lines for p in line.chordPlacements if p.timeSeconds > 8.4
    ]
    assert past_the_grid and all(p.beat is None for p in past_the_grid)


def test_with_neither_beats_nor_a_duration_the_tail_still_holds_constant():
    """No grid and no duration means nothing is known past the last match --
    inventing a spread there would be fabrication, not continuation."""
    song = _fade_out_song()
    mir = _fade_out_mir(beats_reach_end=False).model_copy(
        update={"beats": [], "duration_seconds": 0.0}
    )
    tail = _times(snap_chords(song, mir))[4:]
    assert set(tail) == {6.0}  # the last matched chord's time, held


def test_provenance_splits_measured_from_inferred_beats():
    result = snap_chords(_fade_out_song(), _fade_out_mir(beats_reach_end=True))
    notes = result.provenance[-1].notes
    assert "beats=filled (grid: 17 measured + 43 inferred)" in notes
    assert [b.detected for b in result.audio.beats].count(False) == 43


def test_provenance_reports_an_all_measured_grid_as_such():
    song = Song.model_validate(make_song())
    mir = MirAnalysis(
        beats=[Beat(time=0.0, position=1), Beat(time=0.5, position=2)],
        chords=[ChordSegment(start=0.0, end=3.0, chord="C")],
    )
    notes = snap_chords(song, mir).provenance[-1].notes
    assert "beats=filled (grid: 2 measured + 0 inferred)" in notes


def test_a_song_whose_beats_already_reach_the_end_is_unaffected():
    """Nothing here changes for the ordinary case: every placement matches, so
    there is no tail to spread and the times are exactly the MIR segments'."""
    song = Song.model_validate(make_song())
    mir = MirAnalysis(
        duration_seconds=12.0,
        beats=[Beat(time=i * 0.5, position=(i % 4) + 1) for i in range(24)],
        chords=[
            ChordSegment(start=0.0, end=3.0, chord="C"),
            ChordSegment(start=3.0, end=6.0, chord="G"),
            ChordSegment(start=6.0, end=9.0, chord="Am"),
            ChordSegment(start=9.0, end=12.0, chord="F"),
        ],
    )
    result = snap_chords(song, mir)
    assert _times(result) == [0.0, 3.0, 6.0, 9.0]
    assert all(p.confidence == 0.9 for line in result.lines for p in line.chordPlacements)
    assert all(b.detected for b in result.audio.beats)
