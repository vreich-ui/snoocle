"""Beat-grid continuation over the dead space a beat tracker leaves behind.

The bug this pins: librosa's tracker (the engine the production image
actually uses — madmom is excluded from the Docker build) backtracks from the
last local maximum of its DP cumulative score, so a fade-out silently
truncates the beat list well before the audio ends. Live evidence:
bob-marley--three-little-birds stored 440 beats ending at 177.31s of a 192.18s
track — 15 seconds, 8% of the song, with no beat data at all, while the
structure engine had section times through the full 192s.

The pure-logic tests below pin the continuation rules; the librosa-backed
ones reproduce the loss of lock on synthesized audio (a fade-out and, since
the same mechanism runs mirrored at the head, a fade-in) and show the grid
covering the track afterwards.
"""

from __future__ import annotations

import pytest

from snoocle_server.mir.base import Beat
from snoocle_server.mir.beats import MIN_CONFIRMED_BEATS, extend_beat_grid


def _grid(count: int, interval: float = 0.5, start: float = 0.0, meter: int = 4) -> list[Beat]:
    """`count` detected beats at a steady tempo, phase cycling 1..meter."""
    return [
        Beat(time=round(start + i * interval, 4), position=(i % meter) + 1)
        for i in range(count)
    ]


def _intervals(beats: list[Beat]) -> list[float]:
    return [round(b.time - a.time, 4) for a, b in zip(beats, beats[1:])]


# --- the tail: a fade-out --------------------------------------------------


def test_tail_gap_is_continued_to_the_track_end():
    detected = _grid(60)  # 120bpm, 0.0 .. 29.5s
    extended = extend_beat_grid(detected, duration=45.0, time_signature="4/4")

    assert extended[:60] == detected  # measured beats are never touched
    inferred = extended[60:]
    assert inferred, "a 15.5s tail (nearly 8 bars) must be continued"
    assert all(b.detected is False for b in inferred)
    assert all(b.detected is True for b in extended[:60])

    # the grid now reaches the end (within one beat) at the same tempo
    assert 45.0 - extended[-1].time <= 0.5
    assert extended[-1].time <= 45.0
    assert set(_intervals(extended)) == {0.5}

    # and the beat-in-bar phase just keeps running: ...3, 4, 1, 2...
    positions = [b.position for b in extended]
    assert positions[:8] == [1, 2, 3, 4, 1, 2, 3, 4]
    assert positions[58:64] == [3, 4, 1, 2, 3, 4]


def test_a_grid_that_already_reaches_the_end_is_untouched():
    detected = _grid(120)  # 0.0 .. 59.5s
    assert extend_beat_grid(detected, duration=60.0) == detected
    # a beat or two short is the tracker stopping early, not losing lock
    assert extend_beat_grid(detected, duration=61.0) == detected


def test_a_gap_under_two_bars_is_left_alone():
    detected = _grid(60)  # ends 29.5s
    two_bars = 2 * 4 * 0.5
    assert extend_beat_grid(detected, duration=29.5 + two_bars) == detected
    assert extend_beat_grid(detected, duration=29.5 + two_bars + 0.6) != detected


def test_too_few_confirmed_beats_is_never_extrapolated_from():
    short = _grid(MIN_CONFIRMED_BEATS - 1)
    assert extend_beat_grid(short, duration=300.0) == short
    just_enough = _grid(MIN_CONFIRMED_BEATS)
    assert len(extend_beat_grid(just_enough, duration=300.0)) > MIN_CONFIRMED_BEATS


def test_an_implausible_tempo_is_never_extrapolated_from():
    # 5s between "beats" is not a tempo, it is a dropout — refuse to continue.
    sparse = [Beat(time=float(i) * 5.0, position=(i % 4) + 1) for i in range(40)]
    assert extend_beat_grid(sparse, duration=600.0) == sparse


def test_empty_and_degenerate_inputs_are_no_ops():
    assert extend_beat_grid([], duration=100.0) == []
    assert extend_beat_grid(_grid(60), duration=0.0) == _grid(60)
    assert extend_beat_grid(_grid(60), duration=-1.0) == _grid(60)


def test_continuation_is_idempotent():
    once = extend_beat_grid(_grid(60), duration=45.0)
    assert extend_beat_grid(once, duration=45.0) == once


def test_meter_follows_the_engines_own_phase():
    # madmom on a 3/4 tune reports positions 1..3; the continuation must not
    # silently switch the song to 4/4.
    detected = _grid(60, meter=3)
    extended = extend_beat_grid(detected, duration=45.0, time_signature="3/4")
    assert [b.position for b in extended[59:65]] == [3, 1, 2, 3, 1, 2]


def test_position_unknown_stays_unknown():
    detected = [Beat(time=float(i) * 0.5, position=0) for i in range(60)]
    extended = extend_beat_grid(detected, duration=45.0)
    assert len(extended) > 60
    assert all(b.position == 0 for b in extended)


# --- the head: a long quiet intro ------------------------------------------


def test_head_gap_is_backfilled_with_the_phase_running_backwards():
    detected = _grid(60, start=12.0)  # 12.0 .. 41.5s
    extended = extend_beat_grid(detected, duration=42.0, time_signature="4/4")

    inferred = [b for b in extended if not b.detected]
    assert inferred and all(b.time < 12.0 for b in inferred)
    assert extended[0].time == pytest.approx(0.0, abs=0.5)
    assert extended[0].time >= 0.0
    assert set(_intervals(extended)) == {0.5}
    # the beat before a "1" is the meter's last beat
    first_detected = next(i for i, b in enumerate(extended) if b.detected)
    assert extended[first_detected].position == 1
    assert extended[first_detected - 1].position == 4


def test_a_track_that_starts_on_time_gets_no_backfill():
    detected = _grid(120, start=0.6)
    assert extend_beat_grid(detected, duration=60.6) == detected


# --- the real engine -------------------------------------------------------

librosa = pytest.importorskip("librosa")
np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from snoocle_server.mir.beats import track_beats_librosa  # noqa: E402

_SR = 22050


def _click_track(duration: float, bpm: float = 120.0) -> "np.ndarray":
    """A steady click track over a sustained triad — unambiguously beat-tracked."""
    t = np.arange(int(_SR * duration)) / _SR
    y = 0.15 * (
        np.sin(2 * np.pi * 220.0 * t)
        + np.sin(2 * np.pi * 277.18 * t)
        + np.sin(2 * np.pi * 329.63 * t)
    )
    n = int(0.02 * _SR)
    click = np.exp(-np.linspace(0.0, 12.0, n)) * np.sin(2 * np.pi * 2000.0 * np.arange(n) / _SR)
    for beat in np.arange(0.0, duration, 60.0 / bpm):
        i = int(beat * _SR)
        y[i : i + n] += 0.8 * click
    return y


def _db_ramp(y: "np.ndarray", *, span: tuple[float, float], depth_db: float) -> "np.ndarray":
    """Apply a dB-linear gain ramp over `span` — what a real fade is. The dB
    domain matters: librosa's onset strength is spectral flux over a log-mel
    spectrogram, so a LINEAR fade barely dents it while a dB-linear one walks
    the music down into the noise floor, which is where lock is lost."""
    t = np.arange(len(y)) / _SR
    env = np.ones_like(t)
    mask = (t >= span[0]) & (t <= span[1])
    env[mask] = 10 ** (np.linspace(0.0, depth_db, int(mask.sum())) / 20.0)
    if span[1] < t[-1]:
        env[t > span[1]] = 10 ** (depth_db / 20.0)
    return y * env


def _write(tmp_path, name: str, y: "np.ndarray") -> str:
    path = tmp_path / name
    sf.write(path, y.astype(np.float32), _SR, subtype="PCM_16")
    return str(path)


@pytest.fixture(scope="module")
def fade_out_track(tmp_path_factory):
    """60s at 120bpm, fading 80dB across the last 20s."""
    y = _db_ramp(_click_track(60.0), span=(40.0, 60.0), depth_db=-80.0)
    return _write(tmp_path_factory.mktemp("fade"), "fade_out.wav", y), 60.0


@pytest.fixture(scope="module")
def fade_in_track(tmp_path_factory):
    """60s at 120bpm, rising 80dB across the first 20s."""
    y = _click_track(60.0)
    t = np.arange(len(y)) / _SR
    env = np.ones_like(t)
    mask = t <= 20.0
    env[mask] = 10 ** (np.linspace(-80.0, 0.0, int(mask.sum())) / 20.0)
    return _write(tmp_path_factory.mktemp("fade"), "fade_in.wav", y * env), 60.0


def test_the_tracker_really_does_lose_lock_in_a_fade_out(fade_out_track):
    path, duration = fade_out_track
    beats, bpm, _ts = track_beats_librosa(path)
    times = [t for t, _ in beats]
    interval = 60.0 / bpm
    # the reported failure, reproduced: many seconds of track with no beats
    assert duration - times[-1] > 2 * 4 * interval
    assert duration - times[-1] > 5.0


def test_the_continued_grid_covers_a_fade_out_to_the_end(fade_out_track):
    path, duration = fade_out_track
    raw, bpm, ts = track_beats_librosa(path)
    detected = [Beat(time=t, position=p) for t, p in raw]

    extended = extend_beat_grid(detected, duration, ts)
    inferred = [b for b in extended if not b.detected]

    assert len(inferred) > 8
    assert duration - extended[-1].time <= 60.0 / bpm
    assert [b.time for b in extended] == sorted(b.time for b in extended)
    # the inferred stretch keeps the tempo the audio established
    tail_intervals = _intervals(extended[-len(inferred) - 1 :])
    assert all(abs(iv - 60.0 / bpm) < 0.05 for iv in tail_intervals)
    # and every measured beat survives verbatim
    assert [b.time for b in extended if b.detected] == [t for t, _ in raw]


def test_a_quiet_intro_loses_lock_the_same_way_and_is_backfilled(fade_in_track):
    """The head case, checked rather than assumed: librosa's tracker starts
    late on a fade-in for the same reason it stops early on a fade-out."""
    path, duration = fade_in_track
    raw, bpm, ts = track_beats_librosa(path)
    detected = [Beat(time=t, position=p) for t, p in raw]
    assert detected[0].time > 2 * 4 * (60.0 / bpm)  # lock lost at the head too

    extended = extend_beat_grid(detected, duration, ts)
    assert extended[0].time < detected[0].time
    assert extended[0].time < 60.0 / bpm  # covered back to the top of the track
    assert extended[0].detected is False
    assert [b.time for b in extended if b.detected] == [t for t, _ in raw]
