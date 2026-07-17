"""Regression tests for the librosa beat-tracking fallback — the engine the
production image actually uses for beats (madmom is intentionally excluded
from the Docker build).

Needs only numpy+soundfile to synthesize a wav, so unlike test_mir.py it runs
without ffmpeg — this exact path once crashed in production (librosa >= 0.10
returns tempo as a 1-element ndarray; float() on it is a TypeError under
numpy 2, sinking the whole MIR analysis) while every environment with the
test-suite lacked ffmpeg and skipped the MIR tests.
"""

from __future__ import annotations

import pytest

librosa = pytest.importorskip("librosa")
np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from snoocle_server.mir.beats import track_beats_librosa


@pytest.fixture(scope="module")
def click_track_wav(tmp_path_factory):
    """20s of 120bpm click pulses + a sustained triad."""
    sr = 22050
    t = np.arange(sr * 20) / sr
    triad = 0.2 * (np.sin(2 * np.pi * 261.63 * t) + np.sin(2 * np.pi * 329.63 * t) + np.sin(2 * np.pi * 392.0 * t))
    clicks = np.zeros_like(t)
    for beat in np.arange(0.0, 20.0, 0.5):  # 120bpm
        i = int(beat * sr)
        clicks[i : i + 200] = 0.8
    path = tmp_path_factory.mktemp("beats") / "clicks.wav"
    sf.write(path, (triad + clicks).astype(np.float32), sr, subtype="PCM_16")
    return str(path)


def test_librosa_fallback_returns_plain_floats(click_track_wav):
    beats, bpm, time_signature = track_beats_librosa(click_track_wav)
    assert beats and all(isinstance(t, float) for t, _ in beats)
    # bpm must be a plain float (not an ndarray) or None — float(ndarray) was
    # the production crash
    assert bpm is None or isinstance(bpm, float)
    assert bpm and 100 <= bpm <= 140  # 120bpm click track
    assert time_signature == "4/4"
