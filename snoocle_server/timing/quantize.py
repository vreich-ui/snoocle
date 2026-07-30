"""Beat grid construction and time->beat snapping.

Turns the MIR engines' flat beat list (time + downbeat position) into the
Song schema's measure/beatInMeasure grid (``AudioInfo.beats``), and snaps an
arbitrary time (a chord or line's estimated start) onto that grid when it
falls close enough to a real beat -- the foundation for beat-level chord
placement and, eventually, per-beat voicing capture.

Pure and dependency-free: no audio I/O, no numpy. Input is whatever
``MirAnalysis.beats`` already produced (madmom when available, a librosa
fallback otherwise -- see ``mir/beats.py``).
"""

from __future__ import annotations

import bisect
from typing import Optional

from ..mir.base import Beat, MirAnalysis
from ..mir.base import beats_per_measure as _beats_per_measure_from_signature
from ..schema.song import BeatMark, BeatRef

# Default snap tolerance: half of a "no MIR beat is closer than this" pad --
# chosen so a chord recognized ~a few frames off its true onset still snaps,
# while genuinely off-grid material (rubato intros, freeform outros) doesn't
# get artificially forced onto the beat.
DEFAULT_SNAP_TOLERANCE_SECONDS = 0.35


def build_beat_grid(
    mir: Optional[MirAnalysis], beats_per_measure: Optional[int] = None
) -> list[BeatMark]:
    """MIR beats -> the schema's measure/beatInMeasure grid.

    When the beat engine reported downbeats (``Beat.position == 1`` on at
    least one beat -- madmom, the primary engine), measures are counted from
    those downbeats directly: exact, regardless of time signature.

    When no downbeat information is available (the librosa fallback, which
    reports ``position=0`` for everything), measures are synthesized by
    grouping every ``beats_per_measure`` beats -- inferred from
    ``metadata.timeSignature`` when given, else a 4/4 default. This is a
    best-effort approximation, not audio truth; it is still useful for a
    metronome click, just not authoritative for musical downbeat placement.

    ``Beat.detected`` rides through onto ``BeatMark.detected``: a beat the
    grid was continued through rather than heard (fade-out, quiet intro --
    see ``mir.beats.extend_beat_grid``) stays distinguishable from a measured
    one everywhere downstream.
    """
    beats: list[Beat] = list(mir.beats) if mir else []
    if not beats:
        return []

    bpm_count = beats_per_measure or _beats_per_measure_from_signature(
        mir.time_signature if mir else None
    )
    has_downbeats = any(b.position == 1 for b in beats)

    grid: list[BeatMark] = []
    measure = 1
    beat_in_measure = 0
    for i, b in enumerate(beats):
        if has_downbeats:
            if b.position == 1:
                if i > 0:
                    measure += 1
                beat_in_measure = 1
            else:
                beat_in_measure += 1
        else:
            beat_in_measure += 1
            if beat_in_measure > bpm_count:
                measure += 1
                beat_in_measure = 1
        grid.append(
            BeatMark(
                time=b.time,
                measure=measure,
                beatInMeasure=beat_in_measure,
                detected=b.detected,
            )
        )
    return grid


def snap_time(
    t: float,
    grid: list[BeatMark],
    tolerance: float = DEFAULT_SNAP_TOLERANCE_SECONDS,
) -> tuple[float, Optional[BeatRef]]:
    """Snap `t` to the nearest grid beat within `tolerance` seconds.

    Returns ``(t, None)`` unchanged when the grid is empty or nothing is
    close enough -- callers must not assume a BeatRef is always produced.
    """
    if not grid:
        return t, None

    times = [b.time for b in grid]
    idx = bisect.bisect_left(times, t)
    candidates = []
    if idx < len(grid):
        candidates.append(grid[idx])
    if idx > 0:
        candidates.append(grid[idx - 1])
    best = min(candidates, key=lambda b: abs(b.time - t))
    if abs(best.time - t) <= tolerance:
        return best.time, BeatRef(measure=best.measure, beat=float(best.beatInMeasure))
    return t, None
