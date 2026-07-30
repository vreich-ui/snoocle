"""Beat grids must reach the end of the track, even through a fade-out.

The librosa fallback (the engine the production image actually uses for
beats) is onset-strength driven, so it loses lock exactly where onsets stop
being crisp. On the stored "Three Little Birds" document that meant beats
ending at 177.31s of a 192.18s track — 8% of the song with no beat data, and
every chord placement past the last beat collapsing onto one timestamp
because nothing later existed to interpolate toward.

The fix is tempo continuation, not re-detection: the faded signal is
precisely where onset tracking is unreliable, so re-running it there would
only produce confident nonsense. Beats added this way are marked
``detected=False`` and must stay distinguishable everywhere they travel.
"""

from __future__ import annotations

import shutil

import pytest

from snoocle_server.mir.base import Beat
from snoocle_server.mir.beats import MIN_BEATS_FOR_TEMPO, extend_to_duration


def _grid(count: int, interval: float = 0.5, start: float = 0.0, per_bar: int = 4) -> list[Beat]:
    return [
        Beat(time=start + i * interval, position=(i % per_bar) + 1) for i in range(count)
    ]


# --- the tail: fade-outs ------------------------------------------------------


def test_tail_gap_is_filled_at_the_established_tempo():
    beats = _grid(40)  # 0.0 .. 19.5s at 120bpm
    extended = extend_to_duration(beats, duration=30.0)

    assert extended[:40] == beats  # detected beats are untouched
    added = extended[40:]
    assert added, "a 10.5s tail on a 120bpm track is >2 bars — must be filled"
    assert all(not b.detected for b in added)
    assert added[0].time == pytest.approx(20.0)
    intervals = [b.time - a.time for a, b in zip(extended, extended[1:])]
    assert all(iv == pytest.approx(0.5) for iv in intervals)
    # covers the track: the last beat is within one interval of the end
    assert 30.0 - extended[-1].time <= 0.5
    assert extended[-1].time < 30.0


def test_extrapolated_beats_continue_the_bar_phase():
    # 39 beats: the last detected one is position 3 of its bar.
    beats = _grid(39)
    assert beats[-1].position == 3
    extended = extend_to_duration(beats, duration=30.0)
    added = [b for b in extended if not b.detected]
    assert [b.position for b in added[:5]] == [4, 1, 2, 3, 4]


def test_a_grid_that_already_reaches_the_end_is_untouched():
    beats = _grid(60)  # 0.0 .. 29.5s
    assert extend_to_duration(beats, duration=30.0) == beats


def test_a_gap_shorter_than_two_bars_is_left_alone():
    # 3.5s short of the end at 120bpm 4/4 = 1.75 bars: a track ending between
    # beats, not a tracker that lost lock.
    beats = _grid(40)  # ends at 19.5s
    assert extend_to_duration(beats, duration=23.0) == beats


def test_too_few_beats_to_trust_the_tempo_leaves_the_grid_alone():
    beats = _grid(MIN_BEATS_FOR_TEMPO - 1)
    assert extend_to_duration(beats, duration=300.0) == beats


def test_three_four_grids_use_a_three_beat_bar_for_the_threshold_and_phase():
    beats = _grid(40, per_bar=3)
    assert beats[-1].position == 1
    extended = extend_to_duration(beats, duration=30.0)
    added = [b for b in extended if not b.detected]
    assert [b.position for b in added[:4]] == [2, 3, 1, 2]


def test_unknown_positions_stay_unknown():
    beats = [Beat(time=i * 0.5, position=0) for i in range(40)]
    extended = extend_to_duration(beats, duration=30.0)
    assert all(b.position == 0 for b in extended)
    assert any(not b.detected for b in extended)


def test_the_input_list_is_never_mutated():
    beats = _grid(40)
    before = list(beats)
    extend_to_duration(beats, duration=30.0)
    assert beats == before


def test_a_nonsense_duration_cannot_grow_the_grid_without_bound():
    beats = _grid(40)
    extended = extend_to_duration(beats, duration=10_000.0)  # bad probe
    assert len(extended) - len(beats) <= 2000


def test_zero_duration_is_a_no_op():
    beats = _grid(40)
    assert extend_to_duration(beats, duration=0.0) == beats


def test_median_tempo_survives_a_dropped_beat_near_the_end():
    beats = _grid(40)
    # the tracker skipped a beat just before losing lock: one 1.0s interval
    beats = beats[:38] + [Beat(time=19.5, position=3)]
    extended = extend_to_duration(beats, duration=30.0)
    added = [b for b in extended if not b.detected]
    assert added[0].time == pytest.approx(20.0)  # median, not the 1.0s outlier


# --- the head: quiet intros ---------------------------------------------------


def test_leading_gap_is_filled_backwards_from_the_first_detected_beat():
    beats = _grid(40, start=12.0)  # 12.0 .. 31.5s
    extended = extend_to_duration(beats, duration=32.0)
    lead = [b for b in extended if b.time < 12.0]
    assert lead, "12s before the first beat at 120bpm is >2 bars — must be filled"
    assert all(not b.detected for b in lead)
    assert lead == sorted(lead, key=lambda b: b.time)
    assert lead[-1].time == pytest.approx(11.5)
    assert all(b.time > 0 for b in lead)
    # phase walks backwards from the first detected beat (position 1)
    assert [b.position for b in lead[-3:]] == [2, 3, 4]


def test_a_short_lead_in_is_left_alone():
    beats = _grid(40, start=1.0)  # one bar short of t=0 would be 2.0s
    extended = extend_to_duration(beats, duration=20.5)
    assert all(b.detected for b in extended)


# --- the shape of the real regression ----------------------------------------


def test_three_little_birds_shape_gets_full_coverage():
    """The stored document's numbers: 439 beats running 0.46s .. 177.31s of a
    192.18s track — an 8% dead tail across the whole "Chorus 5 (fade)"
    section. Evenly spaced here; the stored grid wobbles by a few tens of ms,
    which is what the median interval is for."""
    first, last, duration = 0.46439909, 177.30757, 192.18
    interval = (last - first) / 438
    beats = [Beat(time=first + i * interval, position=(i % 4) + 1) for i in range(439)]
    assert beats[-1].time == pytest.approx(177.31, abs=0.01)

    extended = extend_to_duration(beats, duration=duration)
    measured = [b for b in extended if b.detected]
    inferred = [b for b in extended if not b.detected]

    assert len(measured) == 439
    assert len(inferred) == 36  # ~14.9s of fade at ~149bpm
    assert duration - extended[-1].time <= interval
    assert extended[-1].time < duration
    # nothing in the previously dead window is passed off as measured
    assert all(b.time > last for b in inferred)


# --- against real audio -------------------------------------------------------

librosa = pytest.importorskip("librosa")
np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")


@pytest.fixture(scope="module")
def fade_out_click_track(tmp_path_factory):
    """60s of 120bpm clicks over a sustained triad, fading to silence over the
    final 20s — the shape of a fade-out ending."""
    sr = 22050
    duration = 60.0
    t = np.arange(int(sr * duration)) / sr
    triad = 0.2 * (
        np.sin(2 * np.pi * 261.63 * t)
        + np.sin(2 * np.pi * 329.63 * t)
        + np.sin(2 * np.pi * 392.0 * t)
    )
    clicks = np.zeros_like(t)
    for beat in np.arange(0.0, duration, 0.5):
        i = int(beat * sr)
        clicks[i : i + 200] = 0.8
    fade_start = 40.0
    envelope = np.ones_like(t)
    tail = t >= fade_start
    envelope[tail] = (
        np.maximum(1.0 - (t[tail] - fade_start) / (duration - fade_start), 0.0) ** 5
    )
    path = tmp_path_factory.mktemp("fade") / "fade_out.wav"
    sf.write(path, ((triad + clicks) * envelope).astype(np.float32), sr, subtype="PCM_16")
    return str(path), duration


def test_the_tracker_really_does_lose_lock_on_a_fade_out(fade_out_click_track):
    """The reproduction. If this ever stops failing on its own, the engine
    changed — and the assertion below is how we find out."""
    from snoocle_server.mir.beats import track_beats_librosa

    path, duration = fade_out_click_track
    beats, bpm, _ts = track_beats_librosa(path)
    gap = duration - beats[-1].time
    assert gap > 4 * 0.5, f"expected the fade to kill beat tracking; gap was only {gap:.2f}s"
    assert bpm and 100 <= bpm <= 140


def test_track_beats_covers_the_full_duration_of_a_fade_out(fade_out_click_track):
    from snoocle_server.mir.beats import track_beats

    path, duration = fade_out_click_track
    beats, _bpm, _ts, engine = track_beats(path, duration=duration)

    assert engine == "librosa-fallback"
    inferred = [b for b in beats if not b.detected]
    assert inferred, "the faded tail must be covered by tempo continuation"
    assert duration - beats[-1].time <= 0.6  # ~one beat at 120bpm
    assert beats == sorted(beats, key=lambda b: b.time)
    # continuation only — no beat is invented before the tracker's last lock
    assert all(b.time > beats[len(beats) - len(inferred) - 1].time for b in inferred)


def test_without_a_duration_nothing_is_inferred(fade_out_click_track):
    from snoocle_server.mir.beats import track_beats

    path, _duration = fade_out_click_track
    beats, _bpm, _ts, _engine = track_beats(path)
    assert all(b.detected for b in beats)


# --- the whole chain ----------------------------------------------------------


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_analyze_audio_end_to_end_leaves_no_dead_window(fade_out_click_track):
    """What the bug actually cost, start to finish: beats stop early, so the
    beat-synchronous chord timeline stops with them, so every placement past
    the last matched chord lands on one timestamp and the outro stops
    scrolling. One pass over a real fade-out, through the real pipeline."""
    from snoocle_server.mir import analyze_audio
    from snoocle_server.schema.song import Song
    from snoocle_server.timing.snap import snap_chords

    path, duration = fade_out_click_track
    analysis = analyze_audio(path, accuracy="thorough")

    measured, inferred = analysis.beat_counts()
    assert inferred > 0
    assert duration - analysis.beats[-1].time <= 1.0
    assert analysis.chords[-1].end == pytest.approx(duration, abs=1.0)

    song = Song.model_validate(
        {
            "id": "test--fade-out",
            "metadata": {"title": "Fade Out", "artist": "Test"},
            "audio": {},
            "sections": [],
            "lines": [
                {
                    "lineIndex": i,
                    "lyrics": "one little thing and another little thing",
                    "chordPlacements": [
                        {"charIndex": 0, "chord": "C"},
                        {"charIndex": 20, "chord": "G"},
                    ],
                }
                for i in range(8)
            ],
            "provenance": [],
        }
    )
    result = snap_chords(song, analysis)

    times = [p.timeSeconds for line in result.lines for p in line.chordPlacements]
    assert all(t is not None for t in times)
    assert times == sorted(times)
    late = [t for t in times if t > 40.0]  # the faded stretch
    assert late, "nothing was placed in the fade at all"
    assert len(set(late)) == len(late), f"the fade collapsed onto shared times: {late}"
    assert f"grid: {measured} measured + {inferred} inferred" in result.provenance[-1].notes
