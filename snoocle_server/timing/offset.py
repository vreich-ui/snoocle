"""Cross-correlation video-offset estimation (master plan B3).

When the same song exists as two different video uploads (the analyzed
lyric/chord video, plus e.g. a live performance or a different lyric video),
their audio is essentially never aligned at t=0 -- the second video might
start a few seconds earlier or later. Rather than re-running MIR +
reconciliation against every extra upload, this module finds the constant
time shift that makes the second video's rhythmic onsets line up with the
already-analyzed reference audio, using onset-strength cross-correlation --
free, deterministic, no model weights.

Algorithm: librosa onset-strength envelopes (hop 512) for both files, then a
bounded-lag search (default +/- 30s -- offsets between two uploads of the
same song are essentially always this small) over the per-lag Normalized
Cross-Correlation (NCC). The best lag's NCC doubles as the confidence.

IMPORTANT — this confidence is a documented heuristic, not a statistical
guarantee. Empirically (see tests/test_offset.py, which pins the calibration
against synthetic fixtures): a genuinely aligned pair scores roughly 0.6-0.97
depending on how rhythmically distinctive the audio is, while two unrelated
signals cluster below ~0.46. There is real spread and the populations sit
closer together than a hand-wavy "just check it's > 0.9" would suggest --
which is exactly why the confidence is always returned to the caller rather
than collapsed into a silent accept/reject. The API layer
(POST /v1/songs/{id}/video-offset, see api.py) is what turns a low value into
a 409 refusal using a configurable threshold (Settings.offset_min_confidence),
and a human can always override by supplying an offset directly.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio.utils import to_analysis_wav

log = logging.getLogger(__name__)

# Offsets between two uploads of the same song are essentially always small
# (a few seconds of intro/outro padding difference); 30s comfortably covers
# even a long alternate intro without letting the search wander into
# spurious far-lag matches.
DEFAULT_MAX_OFFSET_SECONDS = 30.0

_ANALYSIS_SR = 22050
_HOP_LENGTH = 512
# Candidate lags with less than this much overlapping audio are skipped --
# a short overlap window can show a spuriously high correlation by chance,
# which the plain "biggest raw cross-correlation" approach fell for during
# development (see the module's calibration notes in tests/test_offset.py).
_MIN_OVERLAP_SECONDS = 2.0


@dataclass
class OffsetEstimate:
    # Seconds to ADD to the reference audio's stored times to get correct
    # times for the OTHER video (matches AudioInfo.videoOffsets' semantics).
    # Negative means the other video's audio starts earlier.
    offset_seconds: float
    # Heuristic confidence in [0, 1] -- see module docstring.
    confidence: float


def _onset_envelope(wav_path: str | Path) -> "np.ndarray":
    import librosa

    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP_LENGTH)
    return env - env.mean()


def _ncc_at_lag(ref: "np.ndarray", other: "np.ndarray", lag: int) -> float | None:
    """Normalized cross-correlation of `ref` against `other` shifted by
    `lag` frames (positive lag = other's events happen LATER). None when the
    overlap at this lag is too short to trust."""
    if lag >= 0:
        o = other[lag : lag + len(ref)]
        r = ref[: len(o)]
    else:
        r = ref[-lag : -lag + len(other)]
        o = other[: len(r)]
    n = min(len(r), len(o))
    min_frames = int(_MIN_OVERLAP_SECONDS * _ANALYSIS_SR / _HOP_LENGTH)
    if n < min_frames:
        return None
    r = r[:n]
    o = o[:n]
    denom = float(np.sqrt(np.sum(r * r) * np.sum(o * o)))
    if denom < 1e-9:
        return None
    return float(np.dot(r, o) / denom)


def estimate_offset(
    ref_path: str | Path,
    other_path: str | Path,
    max_offset_seconds: float = DEFAULT_MAX_OFFSET_SECONDS,
) -> OffsetEstimate:
    """How many seconds to ADD to `ref_path`-based stored times so they line
    up with `other_path`. Both files are converted to the standard analysis
    wav first (any ffmpeg-readable container works, mirroring
    mir/pipeline.py's own convention), so this accepts the same raw
    downloads/uploads the rest of the pipeline does -- no pre-conversion
    needed by callers.

    Never raises for "no good match": a search window with no lag having
    enough overlap returns offset 0.0 at confidence 0.0 rather than an
    exception, since "I don't know" is itself a valid, expected outcome here
    (see the confidence-gated 409 in api.py).
    """
    with tempfile.TemporaryDirectory(prefix="snoocle-offset-") as td:
        ref_wav = Path(td) / "ref.wav"
        other_wav = Path(td) / "other.wav"
        to_analysis_wav(ref_path, ref_wav, sample_rate=_ANALYSIS_SR)
        to_analysis_wav(other_path, other_wav, sample_rate=_ANALYSIS_SR)
        ref_env = _onset_envelope(ref_wav)
        other_env = _onset_envelope(other_wav)

    max_lag_frames = int(max_offset_seconds * _ANALYSIS_SR / _HOP_LENGTH)
    best_lag = 0
    best_ncc: float | None = None
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        ncc = _ncc_at_lag(ref_env, other_env, lag)
        if ncc is not None and (best_ncc is None or ncc > best_ncc):
            best_ncc = ncc
            best_lag = lag

    if best_ncc is None:
        # No lag in the search window had enough overlap (e.g. one file is
        # shorter than _MIN_OVERLAP_SECONDS) -- report zero offset, zero trust.
        return OffsetEstimate(offset_seconds=0.0, confidence=0.0)

    offset_seconds = best_lag * _HOP_LENGTH / _ANALYSIS_SR
    confidence = max(0.0, min(1.0, best_ncc))
    return OffsetEstimate(offset_seconds=offset_seconds, confidence=confidence)
