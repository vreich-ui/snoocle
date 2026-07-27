"""Calibration tests for the cross-correlation video-offset estimator
(timing/offset.py, master plan B3).

These pin the confidence heuristic against synthetic fixtures rather than
real audio (no network, fully deterministic via fixed RNG seeds) — see the
module's own docstring for why the confidence is a calibrated heuristic, not
a rigorous statistical guarantee. The exact numbers here were validated by
running the real `estimate_offset` code path (not just the bare cross-
correlation math) during development; if this ever needs re-tuning, re-run
that same sweep before touching the threshold rather than just loosening the
assertions.

Needs ffmpeg (estimate_offset shells out to it via to_analysis_wav) —
skipped automatically when it's missing.
"""

from __future__ import annotations

import shutil

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
pytest.importorskip("librosa")

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

from snoocle_server.config import settings
from snoocle_server.timing.offset import estimate_offset

_SR = 22050
_DURATION = 20.0


def _make_song(duration: float, sr: int, seed: int) -> "np.ndarray":
    """A synthetic 'song': irregular decaying clicks (not a rigid metronome
    grid, which would make different-tempo tracks alias into false
    correlation) plus a tonal hum and a little broadband noise."""
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    y = np.zeros(n, dtype=np.float64)
    click = np.exp(-np.linspace(0, 1, int(0.03 * sr)) * 30)
    t = 0.0
    while t < duration:
        idx = int(t * sr)
        end = min(n, idx + len(click))
        y[idx:end] += click[: end - idx] * rng.uniform(0.6, 1.0)
        t += rng.uniform(0.35, 0.65)
    y += 0.05 * np.sin(2 * np.pi * 220 * np.arange(n) / sr) + rng.normal(0, 0.02, n)
    return y.astype(np.float32)


def _write(path, y, sr=_SR):
    sf.write(path, y, sr, subtype="PCM_16")
    return str(path)


@pytest.fixture(scope="module")
def ref_song_wav(tmp_path_factory):
    y = _make_song(_DURATION, _SR, seed=7)
    return _write(tmp_path_factory.mktemp("offset") / "ref.wav", y)


@pytest.fixture(scope="module")
def aligned_other_wav(tmp_path_factory):
    """The reference song's audio, but padded with 3.2s of leading silence —
    a stand-in for 'the same song, uploaded with a longer intro'."""
    y = _make_song(_DURATION, _SR, seed=7)
    pad = np.zeros(int(3.2 * _SR), dtype=np.float32)
    padded = np.concatenate([pad, y])[: len(y) + len(pad)]
    return _write(tmp_path_factory.mktemp("offset") / "aligned_other.wav", padded)


@pytest.fixture(scope="module")
def unrelated_song_wav(tmp_path_factory):
    y = _make_song(_DURATION, _SR, seed=99)
    return _write(tmp_path_factory.mktemp("offset") / "unrelated_song.wav", y)


@pytest.fixture(scope="module")
def noise_wav_pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("offset")
    n1 = np.random.default_rng(11).normal(0, 1, int(_DURATION * _SR)).astype(np.float32)
    n2 = np.random.default_rng(22).normal(0, 1, int(_DURATION * _SR)).astype(np.float32)
    return _write(d / "noise1.wav", n1), _write(d / "noise2.wav", n2)


def test_aligned_recording_gives_high_confidence_and_correct_offset(ref_song_wav, aligned_other_wav):
    estimate = estimate_offset(ref_song_wav, aligned_other_wav)
    assert estimate.offset_seconds == pytest.approx(3.2, abs=0.1)
    assert estimate.confidence > settings.offset_min_confidence
    # comfortable margin above the gate, not just barely over it
    assert estimate.confidence > 0.6


def test_unrelated_song_gives_confidence_below_the_store_gate(ref_song_wav, unrelated_song_wav):
    estimate = estimate_offset(ref_song_wav, unrelated_song_wav)
    assert estimate.confidence < settings.offset_min_confidence


def test_unrelated_noise_gives_confidence_below_the_store_gate(noise_wav_pair):
    ref, other = noise_wav_pair
    estimate = estimate_offset(ref, other)
    assert estimate.confidence < settings.offset_min_confidence


def test_offset_sign_convention_is_seconds_to_add_to_reference(ref_song_wav, aligned_other_wav):
    """`other` here happens 3.2s LATER than `ref` (it was padded with leading
    silence) — offset_seconds must be the positive amount to ADD to
    ref-based times to land correctly on `other`, matching
    AudioInfo.videoOffsets' documented semantics exactly."""
    estimate = estimate_offset(ref_song_wav, aligned_other_wav)
    assert estimate.offset_seconds > 0


def test_too_short_overlap_returns_zero_confidence_not_a_crash(tmp_path, ref_song_wav):
    """A file far shorter than the minimum overlap window can't be trusted at
    any lag in the search range — must degrade to 'I don't know' (0.0
    confidence), never raise."""
    tiny = np.random.default_rng(1).normal(0, 1, int(0.5 * _SR)).astype(np.float32)
    tiny_path = _write(tmp_path / "tiny.wav", tiny)
    estimate = estimate_offset(ref_song_wav, tiny_path, max_offset_seconds=30.0)
    assert estimate.offset_seconds == 0.0
    assert estimate.confidence == 0.0


def test_max_offset_seconds_bounds_the_search(ref_song_wav, aligned_other_wav):
    """A search window narrower than the true 3.2s offset can't find it —
    confirms the bound is actually enforced, not just accepted and ignored."""
    estimate = estimate_offset(ref_song_wav, aligned_other_wav, max_offset_seconds=1.0)
    assert abs(estimate.offset_seconds) <= 1.0 + 1e-6
    assert estimate.offset_seconds != pytest.approx(3.2, abs=0.1)
