"""End-to-end reproduction of the fade-out beat-tracking failure.

Synthesizes a track shaped like the incident: a click-driven body, then an
outro where the percussion fades out under a sustained pad that still changes
chord. The librosa fallback (the engine the production image actually uses)
loses lock the moment the transients go, and because chord recognition is
beat-synchronous the whole outro collapses into a single chord segment — so
every chord placement in it lands on the same timestamp at interpolation
confidence. That is exactly what ``bob-marley--three-little-birds`` looks
like in the store: 440 beats ending at 177.31s of a 192.18s track, with every
placement past the last beat sharing one time.

Needs only numpy+soundfile+librosa (no ffmpeg), so it runs wherever the
fallback engine itself can run.
"""

from __future__ import annotations

import pytest

librosa = pytest.importorskip("librosa")
np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from snoocle_server.mir.base import Beat, MirAnalysis
from snoocle_server.mir.beat_fill import fill_beat_gaps
from snoocle_server.mir.beats import track_beats_librosa
from snoocle_server.mir.chordrec import recognize_chords_chroma
from snoocle_server.schema.song import Song
from snoocle_server.timing.snap import snap_chords

SR = 22050
DURATION = 60.0
BPM = 120.0
TRIADS = {"C": [0, 4, 7], "G": [7, 11, 2], "Am": [9, 0, 4], "F": [5, 9, 0]}

# The body's chord changes, then the outro's — the outro's second change is
# the one a dead beat grid cannot see.
BODY = [(0.0, 16.0, "C"), (16.0, 32.0, "G"), (32.0, 46.0, "Am")]
OUTRO_CHANGE = 53.0
FADE_START = 42.0
CLICKS_END = 46.0


def _pad(t, name, amp=0.12):
    y = np.zeros(len(t))
    for semitone in TRIADS[name]:
        y += amp * np.sin(2 * np.pi * (261.63 * 2 ** (semitone / 12)) * t)
    return y


def _render(path, quiet_intro_until=0.0):
    n = int(SR * DURATION)
    t = np.arange(n) / SR

    pad = np.zeros(n)
    for start, end, name in BODY:
        span = (t >= start) & (t < end)
        pad[span] += _pad(t[span], name)
    # Outro: F crossfading into C, slowly enough to leave no transient for an
    # onset tracker to latch onto — a fading vamp, not a hit.
    outro = t >= BODY[-1][1]
    to_c = np.clip((t - (OUTRO_CHANGE - 1.0)) / 2.0, 0.0, 1.0)
    pad[outro] += _pad(t[outro], "F") * (1 - to_c[outro]) + _pad(t[outro], "C") * to_c[outro]

    clicks = np.zeros(n)
    click_n = int(0.02 * SR)
    click = 0.9 * np.exp(-np.linspace(0, 12, click_n)) * np.sin(
        2 * np.pi * 2000 * np.arange(click_n) / SR
    )
    for beat in np.arange(0.0, DURATION, 60.0 / BPM):
        i = int(beat * SR)
        clicks[i : i + click_n] += click[: n - i]

    # percussion out over FADE_START..CLICKS_END, everything fading under it
    click_gain = np.clip((CLICKS_END - t) / (CLICKS_END - FADE_START), 0.0, 1.0)
    click_gain[t < FADE_START] = 1.0
    fade = np.ones(n)
    tail = t >= FADE_START
    fade[tail] = 0.25 + 0.75 * np.linspace(1.0, 0.0, int(tail.sum())) ** 2

    y = pad * fade + clicks * click_gain * fade
    if quiet_intro_until:
        y[t < quiet_intro_until] *= 0.0005
    sf.write(path, y.astype(np.float32), SR, subtype="PCM_16")
    return str(path)


@pytest.fixture(scope="module")
def fade_out_wav(tmp_path_factory):
    return _render(tmp_path_factory.mktemp("fade") / "fade_out.wav")


@pytest.fixture(scope="module")
def quiet_intro_wav(tmp_path_factory):
    return _render(tmp_path_factory.mktemp("fade") / "quiet_intro.wav", quiet_intro_until=14.0)


def _tracked(wav) -> list[Beat]:
    beats, _bpm, _ts = track_beats_librosa(wav)
    return [Beat(time=t, position=p) for t, p in beats]


def _mir(wav, beats: list[Beat]) -> MirAnalysis:
    chords = recognize_chords_chroma(wav, [b.time for b in beats])
    return MirAnalysis(
        engines={"beats": "librosa-fallback", "chords": "chroma-template-fallback"},
        duration_seconds=DURATION,
        time_signature="4/4",
        beats=beats,
        chords=chords,
    )


def _song() -> Song:
    """One placement per real chord change, in reading order — the last two
    live in the window the dead beat grid cannot resolve."""
    return Song.model_validate(
        {
            "id": "test--fade-out",
            "metadata": {"title": "Fade Out", "artist": "Test"},
            "audio": {"syncMap": []},
            "sections": [
                {
                    "sectionIndex": 0,
                    "name": "Verse",
                    "kind": "verse",
                    "startLineIndex": 0,
                    "endLineIndex": 4,
                }
            ],
            "lines": [
                {"lineIndex": 0, "lyrics": "one", "chordPlacements": [{"charIndex": 0, "chord": "C"}]},
                {"lineIndex": 1, "lyrics": "two", "chordPlacements": [{"charIndex": 0, "chord": "G"}]},
                {"lineIndex": 2, "lyrics": "three", "chordPlacements": [{"charIndex": 0, "chord": "Am"}]},
                {"lineIndex": 3, "lyrics": "four", "chordPlacements": [{"charIndex": 0, "chord": "F"}]},
                {"lineIndex": 4, "lyrics": "five", "chordPlacements": [{"charIndex": 0, "chord": "C"}]},
            ],
            "provenance": [
                {"timestamp": "2026-07-30T00:00:00Z", "actor": "test", "action": "created"}
            ],
        }
    )


# --- the bug ------------------------------------------------------------------


def test_beat_tracking_loses_lock_on_the_fade_out(fade_out_wav):
    detected = _tracked(fade_out_wav)
    gap = DURATION - detected[-1].time
    # more than two bars (4s at 120bpm) of the track with no beat data at all
    assert gap > 4.0
    assert gap / DURATION > 0.05  # the reported song lost 8% of its duration


def test_extrapolation_covers_the_track_to_the_end(fade_out_wav):
    detected = _tracked(fade_out_wav)
    filled, fill = fill_beat_gaps(detected, DURATION, "4/4")

    assert fill.detected == len(detected)
    assert fill.inferred_tail > 0
    assert DURATION - filled[-1].time <= fill.interval_seconds + 1e-6
    assert filled[-1].time < DURATION
    # the inferred beats carry the detected tempo, and say what they are
    inferred = [b for b in filled if not b.detected]
    assert len(inferred) == fill.inferred_tail
    spacing = [round(b.time - a.time, 3) for a, b in zip(inferred, inferred[1:])]
    assert len(set(spacing)) == 1
    assert abs(60.0 / spacing[0] - BPM) < 6.0


def test_detected_beats_are_never_re_detected_or_moved(fade_out_wav):
    detected = _tracked(fade_out_wav)
    filled, _fill = fill_beat_gaps(detected, DURATION, "4/4")
    kept = [b for b in filled if b.detected]
    assert [b.time for b in kept] == [b.time for b in detected]


# --- the consequence, and the fix ---------------------------------------------


def test_placements_in_the_dead_window_collapse_onto_one_timestamp(fade_out_wav):
    """Pre-fix behaviour, pinned: the outro's two chords share a time."""
    detected = _tracked(fade_out_wav)
    result = snap_chords(_song(), _mir(fade_out_wav, detected))
    times = [line.chordPlacements[0].timeSeconds for line in result.lines]
    assert times[3] == times[4]  # F and C, both in the fade, same instant
    assert result.lines[4].chordPlacements[0].confidence == 0.3


def test_extrapolated_grid_gives_the_dead_window_its_own_timestamps(fade_out_wav):
    detected = _tracked(fade_out_wav)
    filled, fill = fill_beat_gaps(detected, DURATION, "4/4")
    result = snap_chords(_song(), _mir(fade_out_wav, filled))

    times = [line.chordPlacements[0].timeSeconds for line in result.lines]
    assert all(t is not None for t in times)
    assert times == sorted(times)
    # the outro's two chords are now distinct, and land near their real onsets
    assert times[4] - times[3] > 4.0
    assert abs(times[3] - BODY[-1][1]) < 2.0
    assert abs(times[4] - OUTRO_CHANGE) < 2.0
    # and both are audio-matched now, not interpolated
    assert result.lines[4].chordPlacements[0].confidence >= 0.7

    # provenance states the split rather than presenting inferred beats as measured
    notes = result.provenance[-1].notes
    assert f"{fill.detected} measured" in notes
    assert f"{fill.inferred} tempo-inferred" in notes
    assert sum(1 for b in result.audio.beats if not b.detected) == fill.inferred


# --- the same gap at the START ------------------------------------------------


def test_a_near_silent_intro_leaves_the_same_hole_and_is_filled_the_same_way(
    quiet_intro_wav,
):
    detected = _tracked(quiet_intro_wav)
    assert detected[0].time > 4.0  # nothing tracked through the quiet lead-in

    filled, fill = fill_beat_gaps(detected, DURATION, "4/4")
    assert fill.inferred_head > 0
    assert filled[0].time < detected[0].time
    assert filled[0].time >= 0.0
    assert filled[0].time < fill.interval_seconds
    assert all(not b.detected for b in filled[: fill.inferred_head])
