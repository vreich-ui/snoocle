"""A generic safety net for chord/line timings that collapse onto one
timestamp, independent of what upstream caused it.

Two known causes have already produced this:

- MIR beat data ending well before the track does on a fade-out (the
  Marley case: 440 beats stopping at 177.31s of a 192.18s track) --
  addressed at its source by ``mir.beats.extend_beat_grid``, which gives
  ``snap.py`` a later anchor to interpolate toward.
- An unusable chord timeline on lo-fi source audio (a 1966 live recording
  where the MIR chord timeline dies at 86.5s of a 220.6s track): lines
  17-20 all end up at 86.51755102040816, because ``snap.py``'s
  ``_derive_line_times`` correctly holds the last known time for every
  line after the final match rather than inventing one -- but four
  identical consecutive line times is a symptom worth catching on its own,
  whatever produced it.

Each of those gets (or will get) its own targeted fix. This module is the
guard for the NEXT unknown cause: after every timing pass has run, scan the
final song for runs of >= ``COLLAPSE_RUN_THRESHOLD`` consecutive lines, or
consecutive placements within one line, sharing an identical ``timeSeconds``.
A run that short is unremarkable (holding a value for one or two positions
happens legitimately); three or more in a row past that is the same
"nothing left to interpolate toward" signature the Marley bug exposed.

The FIRST entry of a collapsed run is left as-is -- it is the genuine
anchor the rest piled onto. The rest are spread across the open interval
between that anchor and the next DIFFERENT time found later in the same
sequence (the same beat grid `snap.py` already built when one exists,
otherwise even division). When no later time exists to spread toward --
the collapse sits at or past the last thing this song ever measured -- nothing
is invented; the entries are left exactly as found and the gap is reported
as such. This is a guard, not a fix: it must never manufacture confidence
the audio never gave it, and every intervention (or refusal to intervene)
is recorded in provenance loudly enough that a rising rate is visible.

``timing_coverage`` is reported alongside: the fraction of track duration
actually spanned by line timings, independent of whether any run collapsed
-- a document can have zero collapses and still leave most of the track
untimed (the lo-fi case above: 39% coverage, no runs long enough to trigger
the guard by itself if the reconciler only produced a few timed lines).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from ..schema.song import BeatMark, Line, ProvenanceEntry, Song, SyncPoint
from .quantize import DEFAULT_SNAP_TOLERANCE_SECONDS, snap_time

#: A run this short is unremarkable -- e.g. two adjacent lines legitimately
#: starting on the very same beat. Three or more consecutive identical
#: timestamps is the pattern actually seen in production.
COLLAPSE_RUN_THRESHOLD = 3


def timing_coverage(lines: list[Line], duration: Optional[float]) -> Optional[float]:
    """Fraction of `duration` spanned by timed lines: (last - first) / duration.

    None when duration is unknown -- there is nothing to divide by. 0.0 when
    no line has a time at all (distinct from "unknown": the audio WAS
    analyzed, nothing just got timed). Clamped to [0, 1] defensively; times
    are already validated <= any sane duration, but this must never report
    a coverage figure a human would read as impossible.
    """
    if duration is None or duration <= 0:
        return None
    times = [line.timeSeconds for line in lines if line.timeSeconds is not None]
    if not times:
        return 0.0
    span = max(times) - min(times)
    return max(0.0, min(1.0, span / duration))


def _find_collapsed_runs(
    times: list[Optional[float]], threshold: int = COLLAPSE_RUN_THRESHOLD
) -> list[tuple[int, int, float]]:
    """Maximal ``(start, end_exclusive, value)`` ranges of >= `threshold`
    consecutive entries sharing an identical, non-None time. None entries
    break a run without starting one of their own."""
    runs: list[tuple[int, int, float]] = []
    i, n = 0, len(times)
    while i < n:
        if times[i] is None:
            i += 1
            continue
        j = i + 1
        while j < n and times[j] == times[i]:
            j += 1
        if j - i >= threshold:
            runs.append((i, j, times[i]))
        i = j
    return runs


def _next_anchor(times: list[Optional[float]], from_idx: int, value: float) -> Optional[float]:
    """The first time at/after `from_idx` that differs from `value`, skipping
    untimed (None) entries. None when nothing later ever breaks free of it --
    the collapse reaches the end of the sequence (or only holds/None follow),
    which is exactly the "no span to spread into" case."""
    for t in times[from_idx:]:
        if t is not None and t != value:
            return t
    return None


def _spread_times(t0: float, t1: float, count: int, beat_grid: list[BeatMark]) -> list[float]:
    """`count` new times strictly between `t0` and `t1`. Snapped onto the
    beat grid when a nearby beat exists (the same grid and tolerance
    `snap.py` uses); even division otherwise, which is also the fallback for
    any division whose snap would land outside the open interval or collide
    with a previous pick."""
    steps = count + 1
    spread = [t0 + (t1 - t0) * (i + 1) / steps for i in range(count)]
    if beat_grid:
        for i, t in enumerate(spread):
            snapped, _ref = snap_time(t, beat_grid, DEFAULT_SNAP_TOLERANCE_SECONDS)
            if t0 < snapped < t1:
                spread[i] = snapped
    # Snapping each pick independently can, in principle, collide two picks
    # onto the same beat or invert their order; nudge forward rather than let
    # a "fix" reintroduce the very collapse it was meant to remove.
    for i in range(1, len(spread)):
        if spread[i] <= spread[i - 1]:
            spread[i] = min(spread[i - 1] + 1e-3, t1 - 1e-6)
    return spread


def guard_against_collapsed_timing(
    song: Song, duration: Optional[float] = None
) -> tuple[Song, ProvenanceEntry]:
    """Spread collapsed runs of line/placement timing, and report the
    resulting timing coverage. Always returns a NEW song and a
    ``ProvenanceEntry`` (action ``"timing-collapse-guard"``) -- even a
    perfectly healthy document gets one, so coverage and intervention rate
    are visible on every run, not just the runs that needed fixing.
    """
    beat_grid = list(song.audio.beats)
    lines = list(song.lines)
    line_times: list[Optional[float]] = [line.timeSeconds for line in lines]

    spread_notes: list[str] = []
    left_alone_notes: list[str] = []

    for start, end, value in _find_collapsed_runs(line_times):
        dup_count = end - start - 1  # the anchor (start) itself is left alone
        if dup_count <= 0:
            continue
        next_t = _next_anchor(line_times, end, value)
        span = f"lineIndex {lines[start + 1].lineIndex}-{lines[end - 1].lineIndex}"
        if next_t is None:
            left_alone_notes.append(
                f"{dup_count} line(s) ({span}) held at {value:.2f}s with no later "
                f"anchor to spread toward"
            )
            continue
        spread = _spread_times(value, next_t, dup_count, beat_grid)
        for offset, t in enumerate(spread):
            idx = start + 1 + offset
            lines[idx] = lines[idx].model_copy(update={"timeSeconds": t})
            line_times[idx] = t
        spread_notes.append(
            f"{dup_count} line(s) ({span}) spread across {value:.2f}-{next_t:.2f}s"
        )

    for li, line in enumerate(lines):
        placements = list(line.chordPlacements)
        if len(placements) < COLLAPSE_RUN_THRESHOLD:
            continue
        p_times: list[Optional[float]] = [p.timeSeconds for p in placements]
        changed = False
        for start, end, value in _find_collapsed_runs(p_times):
            dup_count = end - start - 1
            if dup_count <= 0:
                continue
            next_t = _next_anchor(p_times, end, value)
            if next_t is None:
                left_alone_notes.append(
                    f"{dup_count} placement(s) on line {line.lineIndex} held at "
                    f"{value:.2f}s with no later in-line anchor to spread toward"
                )
                continue
            spread = _spread_times(value, next_t, dup_count, beat_grid)
            for offset, t in enumerate(spread):
                idx = start + 1 + offset
                # Spread, not measured: confidence stays at (or drops to) the
                # interpolated-tier 0.3, same as snap.py's own interpolation --
                # a later confidence pass may still raise it on independent
                # agreement.
                existing = placements[idx].confidence
                placements[idx] = placements[idx].model_copy(
                    update={
                        "timeSeconds": t,
                        "confidence": min(existing, 0.3) if existing is not None else 0.3,
                    }
                )
                p_times[idx] = t
            changed = True
            spread_notes.append(
                f"{dup_count} placement(s) on line {line.lineIndex} spread across "
                f"{value:.2f}-{next_t:.2f}s"
            )
        if changed:
            lines[li] = line.model_copy(update={"chordPlacements": placements})

    sync_map = [
        SyncPoint(lineIndex=line.lineIndex, time=line.timeSeconds)
        for line in lines
        if line.timeSeconds is not None
    ]
    coverage = timing_coverage(lines, duration)

    notes_parts: list[str] = []
    if spread_notes:
        notes_parts.append("spread: " + "; ".join(spread_notes))
    if left_alone_notes:
        notes_parts.append("left alone (no later anchor): " + "; ".join(left_alone_notes))
    if not spread_notes and not left_alone_notes:
        notes_parts.append("no collapsed runs detected")
    notes_parts.append(
        f"timing coverage: {coverage:.1%}" if coverage is not None else "timing coverage: unknown (no duration)"
    )

    entry = ProvenanceEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        actor="snoocle-server/timing",
        action="timing-collapse-guard",
        confidence=coverage,
        notes="; ".join(notes_parts),
    )

    updated = song.model_copy(
        update={
            "lines": lines,
            "audio": song.audio.model_copy(update={"syncMap": sync_map}),
            "provenance": list(song.provenance) + [entry],
        }
    )
    # model_copy(update=...) skips validators (pydantic v2) -- re-validate so a
    # bug here raises loudly rather than silently persisting an invalid Song.
    return Song.model_validate(updated.model_dump()), entry
