"""Tempo continuation across spans where beat tracking lost lock.

The incident these guard: ``bob-marley--three-little-birds`` stored 440 beats
ending at 177.31s of a 192.18s track — 15 seconds, 8% of the song, with no
beat data — while the structure engine had section times through the full
192s. The audio was fully analyzed; the onset-driven beat tracker
specifically lost lock on the fade-out. Pure tests here; the end-to-end
reproduction through real audio lives in test_beat_fill_fade_out.py.
"""

from __future__ import annotations

from snoocle_server.mir.base import Beat, MirAnalysis
from snoocle_server.mir.beat_fill import (
    MIN_DETECTED_BEATS,
    beats_per_bar,
    fill_beat_gaps,
)
from snoocle_server.timing.quantize import build_beat_grid


def _grid(count, interval=0.5, start=0.0, per_bar=4, first_position=1):
    """`count` detected beats at a fixed interval, bar phase cycling."""
    return [
        Beat(
            time=start + i * interval,
            position=((first_position - 1 + i) % per_bar) + 1,
        )
        for i in range(count)
    ]


# --- the reported failure -----------------------------------------------------


def test_fade_out_tail_is_continued_at_the_established_tempo():
    # The live-store shape: 440 beats stopping 14.87s short of the track end.
    interval = 177.31 / 439
    beats = _grid(440, interval=interval)
    filled, fill = fill_beat_gaps(beats, duration=192.18, time_signature="4/4")

    assert fill.detected == 440
    assert fill.inferred_head == 0
    assert fill.inferred_tail > 0
    # coverage: no more than one beat of the track is left un-gridded
    assert 192.18 - filled[-1].time <= interval + 1e-6
    assert filled[-1].time < 192.18
    # the detected beats are returned untouched, in order, at the front
    assert [b.time for b in filled[:440]] == [b.time for b in beats]
    assert all(b.detected for b in filled[:440])


def test_inferred_beats_are_marked_and_keep_the_tempo_and_bar_phase():
    beats = _grid(64, interval=0.5)  # 120bpm, last beat at 31.5s, position 4
    filled, fill = fill_beat_gaps(beats, duration=48.0, time_signature="4/4")

    inferred = [b for b in filled if not b.detected]
    assert len(inferred) == fill.inferred_tail == 32
    assert inferred[0].time == 32.0
    # spacing holds across the seam and throughout
    spacings = [round(b.time - a.time, 6) for a, b in zip(filled, filled[1:])]
    assert set(spacings) == {0.5}
    # phase continues: beat 64 was position 4, so the next is a downbeat
    assert [b.position for b in inferred[:5]] == [1, 2, 3, 4, 1]


def test_extrapolated_beats_survive_into_the_song_beat_grid():
    beats = _grid(64, interval=0.5)
    filled, _ = fill_beat_gaps(beats, duration=48.0, time_signature="4/4")
    grid = build_beat_grid(MirAnalysis(beats=filled, time_signature="4/4"))

    assert len(grid) == len(filled)
    assert [b.detected for b in grid] == [b.detected for b in filled]
    assert sum(1 for b in grid if not b.detected) == 32
    # measure/beat numbering continues across the seam rather than restarting
    seam = grid[63:66]
    assert [(b.measure, b.beatInMeasure) for b in seam] == [(16, 4), (17, 1), (17, 2)]


# --- the same gap at the START ------------------------------------------------


def test_quiet_intro_is_continued_backwards_to_the_track_start():
    # Beats begin 14.5s in (a near-silent intro the tracker can't follow).
    beats = _grid(64, interval=0.5, start=14.5)
    filled, fill = fill_beat_gaps(beats, duration=46.0, time_signature="4/4")

    assert fill.inferred_head == 29  # 14.5s / 0.5s
    assert filled[0].time == 0.0
    assert all(not b.detected for b in filled[:29])
    assert all(b.detected for b in filled[29:93])
    spacings = {round(b.time - a.time, 6) for a, b in zip(filled, filled[1:])}
    assert spacings == {0.5}


def test_head_fill_never_produces_a_negative_time():
    beats = _grid(64, interval=0.5, start=14.3)  # not a whole number of beats
    filled, fill = fill_beat_gaps(beats, duration=46.0)
    assert fill.inferred_head == 28
    assert filled[0].time >= 0.0


def test_head_phase_runs_backwards_from_the_first_detected_beat():
    # First detected beat is a downbeat, so the beat before it is a bar's last.
    beats = _grid(64, interval=0.5, start=10.0, first_position=1)
    filled, _ = fill_beat_gaps(beats, duration=42.0, time_signature="4/4")
    head = [b for b in filled if not b.detected and b.time < 10.0]
    assert [b.position for b in head[-4:]] == [1, 2, 3, 4]


def test_both_ends_are_filled_when_both_are_short():
    beats = _grid(64, interval=0.5, start=12.0)
    filled, fill = fill_beat_gaps(beats, duration=60.0, time_signature="4/4")
    assert fill.inferred_head == 24
    assert fill.inferred_tail > 0
    assert filled[0].time == 0.0
    assert 60.0 - filled[-1].time <= 0.5


# --- when NOT to fill ---------------------------------------------------------


def test_grid_that_already_reaches_the_end_is_untouched():
    beats = _grid(120, interval=0.5)  # 0.0 .. 59.5
    filled, fill = fill_beat_gaps(beats, duration=60.0, time_signature="4/4")
    assert fill.inferred == 0
    assert filled == beats
    assert all(b.detected for b in filled)
    assert "already covers" in fill.reason


def test_a_gap_under_two_bars_is_left_alone():
    # 3.4s short at 120bpm 4/4 -- under the 4s (two bar) threshold. An ordinary
    # ring-out, not a lost lock.
    beats = _grid(120, interval=0.5)
    _filled, fill = fill_beat_gaps(beats, duration=62.9, time_signature="4/4")
    assert fill.inferred == 0

    _filled, fill = fill_beat_gaps(beats, duration=63.6, time_signature="4/4")
    assert fill.inferred_tail > 0  # just over two bars: fill


def test_too_few_detected_beats_to_trust_the_tempo():
    beats = _grid(MIN_DETECTED_BEATS - 1, interval=0.5)
    filled, fill = fill_beat_gaps(beats, duration=180.0)
    assert fill.inferred == 0
    assert filled == beats
    assert "need 16" in fill.reason


def test_implausible_spacing_is_not_continued():
    # 6s between "beats" is 10bpm -- an artifact, not a tempo.
    beats = _grid(32, interval=6.0)
    _filled, fill = fill_beat_gaps(beats, duration=600.0)
    assert fill.inferred == 0
    assert "no plausible tempo" in fill.reason


def test_unknown_duration_is_not_guessed_at():
    beats = _grid(64, interval=0.5)
    filled, fill = fill_beat_gaps(beats, duration=0.0)
    assert fill.inferred == 0
    assert filled == beats


# --- robustness ---------------------------------------------------------------


def test_a_dropped_beat_does_not_drag_the_continuation_off_tempo():
    beats = _grid(64, interval=0.5)
    # drop one beat near the end: one 1.0s gap among 0.5s ones
    beats = [b for i, b in enumerate(beats) if i != 58]
    filled, fill = fill_beat_gaps(beats, duration=60.0, time_signature="4/4")
    assert fill.interval_seconds == 0.5
    tail = [b for b in filled if not b.detected]
    assert round(tail[1].time - tail[0].time, 6) == 0.5


def test_filling_is_idempotent():
    beats = _grid(64, interval=0.5, start=12.0)
    once, fill_1 = fill_beat_gaps(beats, duration=60.0, time_signature="4/4")
    twice, fill_2 = fill_beat_gaps(once, duration=60.0, time_signature="4/4")
    assert [(b.time, b.detected) for b in twice] == [(b.time, b.detected) for b in once]
    assert (fill_2.detected, fill_2.inferred) == (fill_1.detected, fill_1.inferred)


def test_beats_without_bar_phase_stay_without_bar_phase():
    # position 0 means the engine reported no downbeats; there is no phase to
    # continue and inventing one would fake downbeats nobody heard.
    beats = [Beat(time=i * 0.5, position=0) for i in range(64)]
    filled, fill = fill_beat_gaps(beats, duration=48.0)
    assert fill.inferred_tail == 32
    assert all(b.position == 0 for b in filled)


def test_meter_comes_from_the_engine_before_the_time_signature():
    three_four = [Beat(time=i * 0.5, position=(i % 3) + 1) for i in range(64)]
    assert beats_per_bar(three_four, None) == 3
    # a declared signature only fills in when positions say nothing
    assert beats_per_bar([Beat(time=0.0, position=0)], "3/4") == 3
    assert beats_per_bar([Beat(time=0.0, position=0)], None) == 4

    # and the 2-bar threshold follows the meter: 3/4 at 120bpm is 1.5s a bar,
    # so two bars is 3s -- a 2.9s gap stays, a 3.2s one gets filled. (In 4/4
    # the same beats would need 4s, and neither would.)
    _f, under = fill_beat_gaps(three_four, duration=34.4)
    _f, over = fill_beat_gaps(three_four, duration=34.7)
    assert under.inferred == 0 and over.inferred_tail > 0


def test_describe_reports_the_measured_inferred_split():
    beats = _grid(64, interval=0.5, start=12.0)
    _filled, fill = fill_beat_gaps(beats, duration=60.0, time_signature="4/4")
    text = fill.describe()
    assert "64 detected" in text
    assert f"{fill.inferred} inferred" in text
    assert "120.0bpm" in text
