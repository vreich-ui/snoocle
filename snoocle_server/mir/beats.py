"""Beat/downbeat tracking: madmom primary, librosa fallback.

madmom (CPJKU) is the reference engine (RNN downbeat processor, as used by
ChordMiniApp). Its native build is fussy outside Docker, so import is probed
at call time; when unavailable we fall back to librosa's onset-based tracker
(good tempo/beat grid, no reliable downbeats — positions assume 4/4 then).
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _madmom_available() -> bool:
    try:
        import madmom  # noqa: F401

        return True
    except Exception:  # pragma: no cover - import-environment specific
        return False


def track_beats_madmom(wav_path: str) -> tuple[list[tuple[float, int]], float | None, str | None]:
    """Returns ([(time, position)], bpm, time_signature)."""
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

    act = RNNDownBeatProcessor()(wav_path)
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    raw = proc(act)  # array of [time, beat_position]
    beats = [(float(t), int(p)) for t, p in raw]
    bpm = None
    if len(beats) > 8:
        import numpy as np

        intervals = np.diff([t for t, _ in beats])
        bpm = float(round(60.0 / float(np.median(intervals)), 1))
    meter = max((p for _, p in beats), default=0)
    time_signature = f"{meter}/4" if meter in (3, 4) else None
    return beats, bpm, time_signature


def track_beats_librosa(wav_path: str) -> tuple[list[tuple[float, int]], float | None, str | None]:
    import librosa

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    times = librosa.frames_to_time(beat_frames, sr=sr)
    # no downbeat information from librosa: assume 4/4 starting on beat 1
    beats = [(float(t), (i % 4) + 1) for i, t in enumerate(times)]
    bpm = float(tempo) if tempo else None
    return beats, (round(bpm, 1) if bpm else None), "4/4"


def track_beats(wav_path: str) -> tuple[list[tuple[float, int]], float | None, str | None, str]:
    """Returns (beats, bpm, time_signature, engine_id)."""
    if _madmom_available():
        try:
            beats, bpm, ts = track_beats_madmom(wav_path)
            return beats, bpm, ts, "madmom"
        except Exception as e:  # noqa: BLE001
            log.warning("madmom beat tracking failed, falling back to librosa: %s", e)
    beats, bpm, ts = track_beats_librosa(wav_path)
    return beats, bpm, ts, "librosa-fallback"
