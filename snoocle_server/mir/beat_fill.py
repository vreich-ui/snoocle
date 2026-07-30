"""Tempo continuation across spans where beat tracking lost lock.

Both beat engines are onset-driven (madmom's RNN, librosa's onset-strength
DP tracker), so they stop producing beats wherever the signal stops
producing onsets — which is exactly what a fade-out is. Measured on the
live store: ``bob-marley--three-little-birds`` has 440 beats ending at
177.31s of a 192.18s track, leaving 15s (8% of the song) with no beat data
at all, while the structure engine had section times through the full 192s.
The audio was fully analyzed; beat tracking specifically lost lock.

The damage is not confined to the beat grid. Chord recognition is
beat-synchronous (``chordrec.recognize_chords_chroma`` frames on the beat
times), so a dead tail collapses into a single trailing segment, so
``timing.snap`` finds no chord match past the last beat, so every placement
in that window falls back to interpolation and — with no later anchor to
interpolate toward — they all land on the same timestamp at confidence 0.3.
One bad beat grid, and the last 8% of the song is untimed.

What this module does about it: when detected beats stop short of (or start
late in) the analyzed span by more than ~2 bars, and enough beats were
detected to trust the tempo, it continues the grid at the established
interval, keeping the bar phase. It deliberately does NOT re-run detection
over the quiet region — that signal is precisely where onset detection is
unreliable, and a tempo continuation from 400 confident beats is the more
accurate answer there, not a compromise.

Every beat this module adds carries ``detected=False`` all the way through
``timing.quantize.build_beat_grid`` into the stored ``AudioInfo.beats``, and
``timing.snap`` reports the measured/inferred split in provenance. Inferred
beats must never be indistinguishable from measured ones.

Pure: no audio I/O, no numpy, no engine imports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median

from .base import Beat

#: Fill only when at least this many beats were actually detected. Below it
#: the tempo estimate is not worth trusting — a handful of beats found in an
#: otherwise unreadable track says nothing about what the rest of it does.
MIN_DETECTED_BEATS = 16

#: Fill only when the uncovered span is longer than this many bars. Ordinary
#: endings leave a beat or two of ring-out uncovered; that is not a lost lock
#: and synthesizing beats over it would be noise.
MIN_GAP_BARS = 2.0

#: How many beat intervals at the relevant end of the track define "the
#: established tempo". Local rather than global on purpose: a song that
#: shifts tempo should be continued at the tempo it was actually playing
#: where the grid died, not at its lifetime average.
_TEMPO_WINDOW_BEATS = 32

#: Plausibility bounds on a beat interval (400bpm .. 20bpm). Outside these
#: the "tempo" is an artifact and continuing it would be worse than a gap.
_MIN_INTERVAL_SECONDS = 0.15
_MAX_INTERVAL_SECONDS = 3.0

#: Intervals further than this fraction from the median are dropped before
#: the tempo is re-taken — a missed beat doubles one gap, and one doubled gap
#: must not drag the continuation off tempo.
_OUTLIER_TOLERANCE = 0.25

_TIME_SIGNATURE_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")

_DEFAULT_BEATS_PER_BAR = 4


@dataclass(frozen=True)
class BeatFill:
    """What :func:`fill_beat_gaps` did, for logging and provenance."""

    detected: int
    inferred_head: int
    inferred_tail: int
    interval_seconds: float | None = None
    beats_per_bar: int = _DEFAULT_BEATS_PER_BAR
    #: Why nothing was inferred, when nothing was. Empty when beats were added.
    reason: str = ""

    @property
    def inferred(self) -> int:
        return self.inferred_head + self.inferred_tail

    @property
    def total(self) -> int:
        return self.detected + self.inferred

    def describe(self) -> str:
        if not self.inferred:
            return f"{self.detected} detected, none inferred ({self.reason or 'grid already covers the span'})"
        parts = []
        if self.inferred_head:
            parts.append(f"{self.inferred_head} before the first")
        if self.inferred_tail:
            parts.append(f"{self.inferred_tail} after the last")
        bpm = 60.0 / self.interval_seconds if self.interval_seconds else 0.0
        return (
            f"{self.detected} detected, {self.inferred} inferred "
            f"({', '.join(parts)}) at {bpm:.1f}bpm"
        )


def beats_per_bar(beats: list[Beat], time_signature: str | None) -> int:
    """Bar length in beats, preferring what the engine actually reported.

    madmom numbers beats 1..N within the bar, so the highest position it ever
    emitted IS the meter. Only when positions are uninformative (the librosa
    fallback path, or a grid with no position data) do we fall back to the
    declared time signature, then to 4/4.
    """
    observed = max((b.position for b in beats), default=0)
    if observed >= 2:
        return observed
    if time_signature:
        m = _TIME_SIGNATURE_RE.match(time_signature)
        if m and int(m.group(1)) > 0:
            return int(m.group(1))
    return _DEFAULT_BEATS_PER_BAR


def _established_interval(times: list[float]) -> float | None:
    """Median beat interval over `times`, robust to a dropped beat, or None
    when the spacing is not a plausible tempo."""
    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not gaps:
        return None
    rough = median(gaps)
    if rough <= 0:
        return None
    tight = [g for g in gaps if abs(g - rough) <= _OUTLIER_TOLERANCE * rough]
    interval = median(tight) if tight else rough
    if not (_MIN_INTERVAL_SECONDS <= interval <= _MAX_INTERVAL_SECONDS):
        return None
    return interval


def _position_after(position: int, steps: int, per_bar: int) -> int:
    """Bar phase `steps` beats after `position` (negative steps go back).

    position 0 means the engine reported no bar phase at all; there is none to
    continue, and inventing one would fake downbeats that were never heard.
    """
    if position < 1:
        return 0
    return ((position - 1 + steps) % per_bar) + 1


def fill_beat_gaps(
    beats: list[Beat],
    duration: float,
    time_signature: str | None = None,
) -> tuple[list[Beat], BeatFill]:
    """Continue the beat grid across un-tracked spans at either end.

    `duration` is the length of the span that was ANALYZED (the clip, for a
    windowed run), in the same coordinates as the beat times. Returns a new
    beat list — detected beats untouched, inferred ones appended/prepended
    with ``detected=False`` — plus a :class:`BeatFill` describing the work.

    Idempotent: any beats already marked inferred are discarded and recomputed
    from the detected ones, so re-running can never stack extrapolations.
    """
    detected = sorted((b for b in beats if b.detected), key=lambda b: b.time)
    if len(detected) < MIN_DETECTED_BEATS:
        return detected, BeatFill(
            detected=len(detected),
            inferred_head=0,
            inferred_tail=0,
            reason=f"only {len(detected)} detected beat(s), need {MIN_DETECTED_BEATS}",
        )
    if duration <= 0:
        return detected, BeatFill(
            detected=len(detected), inferred_head=0, inferred_tail=0,
            reason="analyzed duration unknown",
        )

    per_bar = beats_per_bar(detected, time_signature)
    times = [b.time for b in detected]

    head: list[Beat] = []
    head_interval = _established_interval(times[: _TEMPO_WINDOW_BEATS + 1])
    if head_interval is not None:
        lead = detected[0].time
        if lead > MIN_GAP_BARS * per_bar * head_interval:
            steps = int(lead / head_interval)
            for k in range(1, steps + 1):
                t = detected[0].time - k * head_interval
                if t < 0:
                    break
                head.append(
                    Beat(
                        time=t,
                        position=_position_after(detected[0].position, -k, per_bar),
                        detected=False,
                    )
                )
            head.reverse()

    tail: list[Beat] = []
    tail_interval = _established_interval(times[-(_TEMPO_WINDOW_BEATS + 1):])
    if tail_interval is not None:
        remaining = duration - detected[-1].time
        if remaining > MIN_GAP_BARS * per_bar * tail_interval:
            steps = int(remaining / tail_interval)
            for k in range(1, steps + 1):
                t = detected[-1].time + k * tail_interval
                if t >= duration:
                    break
                tail.append(
                    Beat(
                        time=t,
                        position=_position_after(detected[-1].position, k, per_bar),
                        detected=False,
                    )
                )

    reason = ""
    if not head and not tail:
        if head_interval is None and tail_interval is None:
            reason = "no plausible tempo in the detected beats"
        else:
            reason = f"grid already covers the analyzed span within {MIN_GAP_BARS:g} bar(s)"

    return head + detected + tail, BeatFill(
        detected=len(detected),
        inferred_head=len(head),
        inferred_tail=len(tail),
        interval_seconds=tail_interval or head_interval,
        beats_per_bar=per_bar,
        reason=reason,
    )
