from snoocle_server.mir.base import Beat, MirAnalysis
from snoocle_server.timing.quantize import build_beat_grid, snap_time


def _mir(beats, time_signature=None):
    return MirAnalysis(beats=beats, time_signature=time_signature)


def test_build_beat_grid_with_downbeats_4_4_at_120bpm():
    # 120bpm -> 0.5s per beat; downbeat every 4th beat.
    beats = [
        Beat(time=0.0, position=1),
        Beat(time=0.5, position=0),
        Beat(time=1.0, position=0),
        Beat(time=1.5, position=0),
        Beat(time=2.0, position=1),
        Beat(time=2.5, position=0),
        Beat(time=3.0, position=0),
        Beat(time=3.5, position=0),
    ]
    grid = build_beat_grid(_mir(beats))
    assert [(b.measure, b.beatInMeasure) for b in grid] == [
        (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 1), (2, 2), (2, 3), (2, 4),
    ]
    assert [b.time for b in grid] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]


def test_build_beat_grid_without_downbeats_falls_back_to_signature():
    # No downbeat info (librosa fallback) -- synthesize measures of 3 (3/4).
    beats = [Beat(time=float(i) * 0.5, position=0) for i in range(6)]
    grid = build_beat_grid(_mir(beats, time_signature="3/4"))
    assert [(b.measure, b.beatInMeasure) for b in grid] == [
        (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)
    ]


def test_build_beat_grid_defaults_to_4_4_without_signature():
    beats = [Beat(time=float(i) * 0.5, position=0) for i in range(8)]
    grid = build_beat_grid(_mir(beats))
    assert [b.measure for b in grid] == [1, 1, 1, 1, 2, 2, 2, 2]


def test_build_beat_grid_empty_when_no_mir_or_no_beats():
    assert build_beat_grid(None) == []
    assert build_beat_grid(_mir([])) == []


def test_snap_time_within_tolerance():
    grid = build_beat_grid(_mir([Beat(time=0.0, position=1), Beat(time=0.5, position=0)]))
    t, ref = snap_time(0.44, grid, tolerance=0.1)
    assert t == 0.5
    assert ref is not None and ref.measure == 1 and ref.beat == 2.0


def test_snap_time_outside_tolerance_returns_unchanged():
    grid = build_beat_grid(_mir([Beat(time=0.0, position=1), Beat(time=0.5, position=0)]))
    t, ref = snap_time(0.2, grid, tolerance=0.1)
    assert t == 0.2
    assert ref is None


def test_snap_time_empty_grid_is_noop():
    t, ref = snap_time(1.23, [])
    assert t == 1.23
    assert ref is None


def test_build_beat_grid_carries_the_detected_flag_through():
    beats = [Beat(time=0.0, position=1), Beat(time=0.5, position=2)]
    beats += [Beat(time=1.0, position=3, detected=False)]
    grid = build_beat_grid(_mir(beats))
    assert [b.detected for b in grid] == [True, True, False]
    # measure/beat numbering does not care where the beat came from
    assert [(b.measure, b.beatInMeasure) for b in grid] == [(1, 1), (1, 2), (1, 3)]
