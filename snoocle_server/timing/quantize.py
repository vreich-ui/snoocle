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
import re
from typing import Optional

from ..mir.base import Beat, MirAnalysis
from ..schema.song import BeatMark, BeatRef

_DEFAULT_BEATS_PER_MEASURE = 4
_TIME_SIGNATURE_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")

# Default snap tolerance: half of a "no MIR beat is closer than this" pad --
# chosen so a chord recognized ~a few frames off its true onset still snaps,
# while genuinely off-grid material (rubato intros, freeform outros) doesn't
# get artificially forced onto the beat.
DEFAULT_SNAP_TOLERANCE_SECONDS = 0.35


def _beats_per_measure_from_signature(time_signature: Optional[str]) -> int:
    if time_signature:
        m = _TIME_SIGNATURE_RE.match(time_signature)
        if m:
            numerator = int(m.group(1))
            if numerator > 0:
                return numerator
    return _DEFAULT_BEATS_PER_MEASURE


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

    ``Beat.detected`` rides through onto ``BeatMark.detected``: a grid may
    extend past where the tracker last locked (fade-outs), and the caller has
    to be able to tell those beats apart from measured ones.
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


def grid_counts(grid: list[BeatMark]) -> tuple[int, int]:
    """(measured, inferred) beats in a grid — inferred ones were continued
    from the established tempo, not heard. See ``mir.beats.extend_to_duration``."""
    measured = sum(1 for b in grid if b.detected)
    return measured, len(grid) - measured


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
