"""Beat/downbeat tracking: madmom primary, librosa fallback.

madmom (CPJKU) is the reference engine (RNN downbeat processor, as used by
ChordMiniApp). Its native build is fussy outside Docker, so import is probed
at call time; when unavailable we fall back to librosa's onset-based tracker
(good tempo/beat grid, no reliable downbeats — positions assume 4/4 then).

Both engines are driven by onset strength, so both lose lock exactly where
onsets stop being crisp: a fade-out ending is the classic case (the librosa
fallback on "Three Little Birds" stopped at 177.3s of a 192.2s track, leaving
8% of the song with no beat data at all). Re-detecting in that span is
hopeless — the signal is the problem — so :func:`extend_to_duration` continues
the grid there at the tempo the tracker already established, and marks every
beat it adds ``detected=False``.
"""

from __future__ import annotations

import logging
from statistics import median

from .base import Beat

log = logging.getLogger(__name__)

# How many beats the tracker must have locked before its tempo is worth
# continuing. 16 is four bars of 4/4 — enough that the median interval is a
# real tempo estimate and not two lucky onsets.
MIN_BEATS_FOR_TEMPO = 16
# How long an unbeated span has to be before it counts as lost lock rather
# than a track simply ending between beats. Two bars.
MIN_GAP_BARS = 2.0
# Pathological guard, not a working limit: a bad duration probe must not be
# able to synthesize an unbounded grid. 2000 beats is ~16 minutes at 120bpm.
_MAX_INFERRED_BEATS = 2000


def _madmom_available() -> bool:
    try:
        import madmom  # noqa: F401

        return True
    except Exception:  # pragma: no cover - import-environment specific
        return False


def _beats_per_bar(beats: list[Beat]) -> int:
    """Bar length implied by the reported beat positions (madmom picks 3 or 4;
    the librosa fallback assumes 4). 4 when positions are unknown."""
    positions = [b.position for b in beats if b.position > 0]
    return max(positions) if positions else 4


def extend_to_duration(beats: list[Beat], duration: float) -> list[Beat]:
    """Continue the beat grid at the established tempo across any span at the
    head or tail of the track that the tracker never locked onto.

    Returns a new list; the input is never mutated. Added beats carry
    ``detected=False`` and continue the existing ``position`` phase, so the
    measure/beatInMeasure grid built from them stays coherent with the part
    that was actually heard.

    Conservative by construction — the grid is left exactly as detected unless
    BOTH hold:

    * at least ``MIN_BEATS_FOR_TEMPO`` beats were detected (the tempo estimate
      is trustworthy), and
    * the unbeated span is at least ``MIN_GAP_BARS`` bars long (a track that
      merely ends between beats is not a tracking failure).

    The tempo used is the median detected inter-beat interval — robust to the
    occasional doubled/halved interval near the point where lock was lost.
    """
    detected = [b for b in beats if b.detected]
    if duration <= 0 or len(detected) < MIN_BEATS_FOR_TEMPO:
        return list(beats)

    intervals = [b.time - a.time for a, b in zip(detected, detected[1:]) if b.time > a.time]
    if not intervals:
        return list(beats)
    interval = median(intervals)
    if interval <= 0:
        return list(beats)

    per_bar = _beats_per_bar(detected)
    min_gap = MIN_GAP_BARS * per_bar * interval

    lead: list[Beat] = []
    first = detected[0]
    if first.time >= min_gap:
        position = first.position
        t = first.time - interval
        while t > 0 and len(lead) < _MAX_INFERRED_BEATS:
            # walk the bar phase backwards: 1 -> per_bar, 3 -> 2, ...
            position = ((position - 2) % per_bar) + 1 if position else 0
            lead.append(Beat(time=t, position=position, detected=False))
            t -= interval
        lead.reverse()

    tail: list[Beat] = []
    last = detected[-1]
    if duration - last.time >= min_gap:
        position = last.position
        t = last.time + interval
        while t < duration and len(tail) < _MAX_INFERRED_BEATS:
            position = (position % per_bar) + 1 if position else 0
            tail.append(Beat(time=t, position=position, detected=False))
            t += interval

    if lead or tail:
        log.info(
            "beat grid continued at %.1f bpm: %d beat(s) before %.2fs, %d after %.2fs "
            "(track duration %.2fs)",
            60.0 / interval, len(lead), first.time, len(tail), last.time, duration,
        )
    return lead + list(beats) + tail


def track_beats_madmom(wav_path: str) -> tuple[list[Beat], float | None, str | None]:
    """Returns (beats, bpm, time_signature)."""
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

    act = RNNDownBeatProcessor()(wav_path)
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    raw = proc(act)  # array of [time, beat_position]
    beats = [Beat(time=float(t), position=int(p)) for t, p in raw]
    bpm = None
    if len(beats) > 8:
        import numpy as np

        intervals = np.diff([b.time for b in beats])
        bpm = float(round(60.0 / float(np.median(intervals)), 1))
    meter = max((b.position for b in beats), default=0)
    time_signature = f"{meter}/4" if meter in (3, 4) else None
    return beats, bpm, time_signature


def track_beats_librosa(wav_path: str) -> tuple[list[Beat], float | None, str | None]:
    import librosa
    import numpy as np

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    times = librosa.frames_to_time(beat_frames, sr=sr)
    # no downbeat information from librosa: assume 4/4 starting on beat 1
    beats = [Beat(time=float(t), position=(i % 4) + 1) for i, t in enumerate(times)]
    # librosa >= 0.10 returns tempo as a 1-element ndarray; float() on that is
    # a TypeError under numpy 2, which would sink the whole MIR analysis.
    bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    return beats, (round(bpm, 1) if bpm else None), "4/4"


def track_beats(
    wav_path: str, duration: float | None = None
) -> tuple[list[Beat], float | None, str | None, str]:
    """Returns (beats, bpm, time_signature, engine_id).

    When `duration` (of the analyzed audio, in its own time coordinates) is
    given, the grid is continued at the established tempo across any span the
    tracker lost lock on — see :func:`extend_to_duration`.
    """
    if _madmom_available():
        try:
            beats, bpm, ts = track_beats_madmom(wav_path)
            engine = "madmom"
        except Exception as e:  # noqa: BLE001
            log.warning("madmom beat tracking failed, falling back to librosa: %s", e)
            beats, bpm, ts = track_beats_librosa(wav_path)
            engine = "librosa-fallback"
    else:
        beats, bpm, ts = track_beats_librosa(wav_path)
        engine = "librosa-fallback"

    if duration is not None:
        beats = extend_to_duration(beats, duration)
    return beats, bpm, ts, engine
