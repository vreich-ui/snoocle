"""Beat/downbeat tracking: madmom primary, librosa fallback.

madmom (CPJKU) is the reference engine (RNN downbeat processor, as used by
ChordMiniApp). Its native build is fussy outside Docker, so import is probed
at call time; when unavailable we fall back to librosa's onset-based tracker
(good tempo/beat grid, no reliable downbeats — positions assume 4/4 then).

``extend_beat_grid`` covers this module's known blind spot: the librosa
tracker is onset-strength based and its backtrace ends at the last local
maximum of the DP's cumulative score, so a decaying fade-out silently
truncates the beat list well before the audio ends (measured here: a 60s
click track fading 80 dB over its last 20s stops producing beats at 52s).
The same happens mirrored at the head of a track that fades in from near
silence. Re-detecting in that region is exactly where onset detection is
unreliable, so the grid is instead continued at the established tempo and
every beat produced that way is flagged ``detected=False``.
"""

from __future__ import annotations

import logging
import statistics

from .base import Beat, beats_per_measure

log = logging.getLogger(__name__)

#: Dead space (at either end) that has to be exceeded before the grid is
#: continued: two bars at the established tempo. Below that the tracker
#: merely stopped a beat or two early, which is not worth inventing over.
GAP_BARS = 2.0

#: Enough confirmed beats that the tempo is trustworthy to continue from.
MIN_CONFIRMED_BEATS = 16

#: Beats used to establish the local tempo at each end of the grid. Local,
#: not global: what matters is the tempo where the tracker lost lock.
_TEMPO_WINDOW_BEATS = 32

#: Sanity bounds on the beat interval (300bpm .. 30bpm). Outside these the
#: "tempo" is an artifact (a half/double-time flip, a dropout) and nothing
#: should be extrapolated from it.
_MIN_INTERVAL_SECONDS = 0.2
_MAX_INTERVAL_SECONDS = 2.0


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
    import numpy as np

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    times = librosa.frames_to_time(beat_frames, sr=sr)
    # no downbeat information from librosa: assume 4/4 starting on beat 1
    beats = [(float(t), (i % 4) + 1) for i, t in enumerate(times)]
    # librosa >= 0.10 returns tempo as a 1-element ndarray; float() on that is
    # a TypeError under numpy 2, which would sink the whole MIR analysis.
    bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
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


def _meter(beats: list[Beat], time_signature: str | None) -> int:
    """Beats per bar to continue the phase with: the engine's own highest
    reported position when it reports any (madmom's 3 or 4, the librosa
    fallback's synthesized 4), else the time signature's numerator."""
    positions = [b.position for b in beats if b.position >= 1]
    if positions:
        top = max(positions)
        if top >= 2:
            return top
    return beats_per_measure(time_signature)


def _interval(times: list[float], *, at_end: bool) -> float | None:
    """Median beat interval over the last (or first) ``_TEMPO_WINDOW_BEATS``
    beats, or None when the grid is too short or the tempo implausible."""
    window = times[-_TEMPO_WINDOW_BEATS:] if at_end else times[:_TEMPO_WINDOW_BEATS]
    if len(window) < 2:
        return None
    gaps = [b - a for a, b in zip(window, window[1:]) if b > a]
    if not gaps:
        return None
    median = float(statistics.median(gaps))
    if not _MIN_INTERVAL_SECONDS <= median <= _MAX_INTERVAL_SECONDS:
        return None
    return median


def extend_beat_grid(
    beats: list[Beat],
    duration: float,
    time_signature: str | None = None,
    *,
    gap_bars: float = GAP_BARS,
    min_confirmed_beats: int = MIN_CONFIRMED_BEATS,
) -> list[Beat]:
    """Continue a truncated beat grid across dead space at either end.

    Returns a NEW list covering ``[0, duration]`` when the detected beats fall
    more than ``gap_bars`` bars short of an end AND at least
    ``min_confirmed_beats`` beats were detected (enough to trust the tempo).
    Continued beats carry the established interval and keep cycling the
    engine's beat-in-bar phase, and every one of them is ``detected=False``.

    A grid that already reaches both ends, a grid too short to trust, or an
    implausible tempo all return the input unchanged (a copy). Nothing is
    re-detected: the faded region is precisely where onset detection is
    unreliable, so tempo continuation is the more accurate answer there, not
    a compromise.
    """
    grid = list(beats)
    if duration <= 0 or len(grid) < min_confirmed_beats:
        return grid

    times = [b.time for b in grid]
    meter = _meter(grid, time_signature)

    tail = _interval(times, at_end=True)
    if tail is not None and duration - times[-1] > gap_bars * meter * tail:
        last = grid[-1]
        step = 1
        while last.time + step * tail <= duration:
            position = ((last.position + step - 1) % meter) + 1 if last.position >= 1 else 0
            grid.append(
                Beat(time=round(last.time + step * tail, 4), position=position, detected=False)
            )
            step += 1

    head = _interval(times, at_end=False)
    if head is not None and times[0] > gap_bars * meter * head:
        first = grid[0]  # tail continuation only appended, so this is still beat 1
        prepended: list[Beat] = []
        step = 1
        while first.time - step * head >= 0.0:
            position = ((first.position - step - 1) % meter) + 1 if first.position >= 1 else 0
            prepended.append(
                Beat(time=round(first.time - step * head, 4), position=position, detected=False)
            )
            step += 1
        grid = prepended[::-1] + grid

    return grid
