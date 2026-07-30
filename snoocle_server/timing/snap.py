"""Deterministic chord/line-time assignment from MIR audio analysis.

``snap_chords`` is the bridge between "the reconciler produced lyrics+chords
from text sources" and "the app can scroll in sync": it never invents a
timestamp itself, it only reads the audio-grounded MIR chord timeline
(``mir.chords``, already start/end timestamped) and assigns those times to
the placements the reconciler already decided on, by matching root pitch
class in reading order. No LLM call, no network -- pure and deterministic,
so the same (song, mir) pair always produces the same output.

Algorithm, in order:
1. Walk chordPlacements in reading order (line by line, ascending charIndex
   -- already schema-guaranteed). Walk MIR chord segments in start-time
   order, skipping "N" (no-chord).
2. Greedy forward matching: for each placement, look at the next unclaimed
   MIR segment(s) (a small forward window, since the reconciler occasionally
   keeps/drops a chord the MIR engine didn't) for a root pitch-class match.
   On a match: timeSeconds = segment.start; confidence 0.9 when the segment
   and placement also agree on major/minor family (matching the existing
   reconcile prompt's own definition of "family" -- MIR reporting a coarser
   extension, e.g. plain C where the sheet says Cmaj7, still counts as
   agreement), else 0.7 -- reserved for a genuine major/minor clash on a
   matched root (e.g. MIR heard major where the sheet says minor), which is
   worth flagging at lower confidence for the review queue (task A5).
3. Placements with no match interpolate linearly (by position in the
   reading-order sequence) between their nearest matched neighbors;
   confidence 0.3. A run with zero matches leaves timing untouched.
   Placements after the LAST match normally have nothing to interpolate
   toward and hold that match's time -- which is right when the timeline
   genuinely ends there, and wrong when the beat tracker lost lock in a
   fade-out and every trailing placement then piles onto one timestamp.
   So when (and only when) the beat grid was continued past the audio the
   tracker heard (``BeatMark.detected is False`` -- see
   ``mir.beats.extend_beat_grid``), the end of that continued grid is used
   as the closing anchor and the trailing placements spread across it.
   Still confidence 0.3: interpolated, not measured.
4. Every assigned time is snapped onto the MIR beat grid when within
   ``quantize.DEFAULT_SNAP_TOLERANCE_SECONDS`` (adds a BeatRef; never changes
   confidence).
5. Line.timeSeconds = the earliest chord time on that line; lines with no
   chords interpolate the same way, by lineIndex position. A final clamp
   pass guarantees non-decreasing line times (defensive -- the inputs are
   already monotonic by construction, see the module tests).
6. audio.syncMap is REGENERATED from the resulting line times (one entry per
   timed line) rather than merged, so it can never diverge from them.
   metadata.bpm and audio.beats are filled from MIR when the song doesn't
   already have them.

A ProvenanceEntry documents the pass (action="timing-snap") whenever `mir`
is given, even when nothing matched -- callers rely on this to distinguish
"MIR was unavailable" from "MIR ran and found nothing to snap to".
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import NamedTuple, Optional

from ..mir.base import ChordSegment, MirAnalysis
from ..schema.song import (
    AudioInfo,
    BeatMark,
    ChordPlacement,
    Line,
    ProvenanceEntry,
    Song,
    SyncPoint,
)
from .quantize import DEFAULT_SNAP_TOLERANCE_SECONDS, build_beat_grid, snap_time
from .root_match import match_chord_roots, sounding_segments

#: Confidence tiers this pass assigns. Named because other passes reason about
#: them: the quality grader counts placements sitting at the INTERPOLATED tier
#: (a time derived from neighbours, not heard in the audio), and the collapse
#: guard never raises a spread placement above it.
MATCHED_CONFIDENCE = 0.9
#: A matched root whose major/minor family clashes with the audio's.
FAMILY_CLASH_CONFIDENCE = 0.7
#: Interpolated between matched neighbours — not measured.
INTERPOLATED_CONFIDENCE = 0.3


class _MatchedTime(NamedTuple):
    position: int  # index into the flat reading-order placement list
    time: float


def _flatten_placements(song: Song) -> list[tuple[int, int, ChordPlacement]]:
    """(lineIndex, placementIndexWithinLine, placement) in reading order."""
    flat: list[tuple[int, int, ChordPlacement]] = []
    for line in song.lines:
        for pi, placement in enumerate(line.chordPlacements):
            flat.append((line.lineIndex, pi, placement))
    return flat


def _match_chords_to_mir(
    flat: list[tuple[int, int, ChordPlacement]], mir: MirAnalysis
) -> tuple[dict[int, tuple[float, float]], dict[int, ChordSegment]]:
    """Returns {flat_index: (time, confidence)} and {flat_index: matched_segment}.

    The matching itself lives in :mod:`timing.root_match` — shared with the
    candidate scorer (``reconcile.match``) rather than reimplemented there.
    This function is only the confidence mapping on top of it.
    """
    matched = match_chord_roots(
        [placement.chord for _li, _pi, placement in flat], sounding_segments(mir)
    )
    matches: dict[int, tuple[float, float]] = {}
    matched_segments: dict[int, ChordSegment] = {}
    for flat_idx, match in matched.items():
        matches[flat_idx] = (
            match.segment.start,
            MATCHED_CONFIDENCE if match.same_family else FAMILY_CLASH_CONFIDENCE,
        )
        matched_segments[flat_idx] = match.segment
    return matches, matched_segments


def _tail_anchor(grid: list[BeatMark], after: float) -> Optional[float]:
    """The end of a beat grid that was CONTINUED past what the tracker heard,
    when that end lies after `after`. None otherwise -- a grid whose last beat
    was detected offers no knowledge beyond it, and trailing placements must
    keep holding the last matched time rather than spreading over a guess."""
    if not grid or grid[-1].detected or grid[-1].time <= after:
        return None
    return grid[-1].time


def _interpolate_unmatched(
    flat_len: int,
    matches: dict[int, tuple[float, float]],
    tail_anchor: Optional[float] = None,
) -> dict[int, tuple[float, float]]:
    """Fill every unmatched position by linear interpolation (by position,
    not by time) between its nearest matched neighbors; confidence 0.3.
    Positions before the first match reuse it (held constant -- never
    extrapolated past known data), and so do positions after the last unless
    `tail_anchor` gives a later time to spread toward (see the module
    docstring: the continued beat grid over a fade-out)."""
    if not matches:
        return {}

    matched_positions = sorted(matches)
    result: dict[int, tuple[float, float]] = dict(matches)

    last_matched = matched_positions[-1]
    if tail_anchor is not None and last_matched < flat_len - 1:
        t0 = matches[last_matched][0]
        span = tail_anchor - t0
        steps = flat_len - last_matched  # keeps the final placement short of the end
        for i in range(last_matched + 1, flat_len):
            result[i] = (t0 + span * (i - last_matched) / steps, INTERPOLATED_CONFIDENCE)

    for i in range(flat_len):
        if i in result:
            continue
        # nearest matched neighbor at/before i, and at/after i
        before = None
        after = None
        for p in matched_positions:
            if p <= i:
                before = p
            if p >= i and after is None:
                after = p
        if before is not None and after is not None and before != after:
            t0 = matches[before][0]
            t1 = matches[after][0]
            frac = (i - before) / (after - before)
            result[i] = (t0 + frac * (t1 - t0), INTERPOLATED_CONFIDENCE)
        elif before is not None:
            result[i] = (matches[before][0], INTERPOLATED_CONFIDENCE)
        elif after is not None:
            result[i] = (matches[after][0], INTERPOLATED_CONFIDENCE)
        # else: no matches at all -- unreachable here since `matches` is non-empty
    return result


def _derive_line_times(song: Song, chord_times: dict[int, float]) -> list[Optional[float]]:
    """chord_times maps a FLAT placement index to its assigned time. Returns
    one entry per line: the earliest chord time on that line, or an
    interpolation across lineIndex for lines with no chords at all."""
    flat = _flatten_placements(song)
    line_first_chord_time: dict[int, float] = {}
    for flat_idx, (line_idx, pi, _placement) in enumerate(flat):
        if pi != 0:
            continue
        # first placement of this line in reading order == earliest, since
        # within-line placements are schema-guaranteed ascending by charIndex
        # and we assign non-decreasing times to them.
        if flat_idx in chord_times:
            line_first_chord_time[line_idx] = chord_times[flat_idx]
    # A line's true first chord might not be its list's [0]th entry if that
    # entry had no assigned time for some reason; fall back to a full scan.
    for line in song.lines:
        if line.lineIndex in line_first_chord_time:
            continue
        times = [
            chord_times[fi]
            for fi, (li, _pi, _p) in enumerate(flat)
            if li == line.lineIndex and fi in chord_times
        ]
        if times:
            line_first_chord_time[line.lineIndex] = min(times)

    n = len(song.lines)
    raw: list[Optional[float]] = [line_first_chord_time.get(i) for i in range(n)]

    known_positions = [i for i, t in enumerate(raw) if t is not None]
    if not known_positions:
        return raw

    result = list(raw)
    for i in range(n):
        if result[i] is not None:
            continue
        before = max((p for p in known_positions if p <= i), default=None)
        after = min((p for p in known_positions if p >= i), default=None)
        if before is not None and after is not None and before != after:
            t0, t1 = raw[before], raw[after]
            frac = (i - before) / (after - before)
            result[i] = t0 + frac * (t1 - t0)
        elif before is not None:
            result[i] = raw[before]
        elif after is not None:
            result[i] = raw[after]

    # Defensive monotonic clamp -- inputs are already non-decreasing by
    # construction (matched times come from a start-sorted MIR sequence and
    # a strictly-advancing search pointer), this just guards against any
    # future change to the matching logic silently breaking that.
    running_max = float("-inf")
    for i in range(n):
        if result[i] is None:
            continue
        if result[i] < running_max:
            result[i] = running_max
        else:
            running_max = result[i]
    return result


def snap_chords(song: Song, mir: Optional[MirAnalysis]) -> Song:
    """Pure: returns a NEW Song with chord/line timing populated from `mir`.

    A no-op (returns `song` unchanged, no provenance appended) when `mir` is
    None -- "MIR was unavailable for this run" must stay distinguishable
    from "MIR ran and matched nothing", which still appends provenance.
    """
    if mir is None:
        return song

    flat = _flatten_placements(song)
    beat_grid = build_beat_grid(mir)

    matches, _matched_segments = _match_chords_to_mir(flat, mir)
    tail_anchor = (
        _tail_anchor(beat_grid, after=max(t for t, _c in matches.values()))
        if matches
        else None
    )
    assigned = _interpolate_unmatched(len(flat), matches, tail_anchor)

    # Rebuild chordPlacements per line with times/confidence/beat filled in.
    new_lines: list[Line] = []
    flat_idx = 0
    chord_times_by_flat: dict[int, float] = {}
    for line in song.lines:
        new_placements: list[ChordPlacement] = []
        for placement in line.chordPlacements:
            update: dict = {}
            if flat_idx in assigned:
                t, confidence = assigned[flat_idx]
                snapped_t, beat_ref = snap_time(t, beat_grid, DEFAULT_SNAP_TOLERANCE_SECONDS)
                update["timeSeconds"] = snapped_t
                update["confidence"] = confidence
                if beat_ref is not None:
                    update["beat"] = beat_ref
                chord_times_by_flat[flat_idx] = snapped_t
            new_placements.append(
                placement.model_copy(update=update) if update else placement
            )
            flat_idx += 1
        new_lines.append(line.model_copy(update={"chordPlacements": new_placements}))

    line_times = _derive_line_times(
        song.model_copy(update={"lines": new_lines}), chord_times_by_flat
    )
    final_lines: list[Line] = []
    for line, t in zip(new_lines, line_times):
        update = {}
        if t is not None:
            update["timeSeconds"] = t
        final_lines.append(line.model_copy(update=update) if update else line)

    sync_map = [
        SyncPoint(lineIndex=line.lineIndex, time=line.timeSeconds)
        for line in final_lines
        if line.timeSeconds is not None
    ]

    audio_update: dict = {"syncMap": sync_map}
    if not song.audio.beats and beat_grid:
        audio_update["beats"] = beat_grid
    new_audio: AudioInfo = song.audio.model_copy(update=audio_update)

    metadata_update: dict = {}
    if song.metadata.bpm is None and mir.bpm:
        metadata_update["bpm"] = mir.bpm
    new_metadata = song.metadata.model_copy(update=metadata_update) if metadata_update else song.metadata

    total = len(flat)
    matched_count = len(matches)
    sources = [f"{slot}:{impl}" for slot, impl in sorted(mir.engines.items())]
    # The measured/inferred split is reported for the grid this pass BUILT,
    # whether or not it was written onto the song -- "the audio was only
    # heard through 92% of the track" is the fact worth recording either way.
    inferred_beats = sum(1 for b in beat_grid if not b.detected)
    beats_note = (
        f"beats={'filled' if 'beats' in audio_update else 'kept/empty'} "
        f"({len(beat_grid) - inferred_beats} measured, {inferred_beats} inferred)"
    )
    provenance_entry = ProvenanceEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        actor="snoocle-server/timing",
        action="timing-snap",
        sources=sources,
        confidence=(matched_count / total) if total else None,
        notes=(
            f"matched {matched_count}/{total} chord placement(s) to the MIR "
            f"timeline; {len(sync_map)}/{len(final_lines)} line(s) timed; "
            + beats_note
            + (
                f"; {total - 1 - max(matches)} trailing placement(s) spread to "
                f"the continued grid end at {tail_anchor:.2f}s"
                if tail_anchor is not None and matches and max(matches) < total - 1
                else ""
            )
        ),
    )

    updated = song.model_copy(
        update={
            "lines": final_lines,
            "audio": new_audio,
            "metadata": new_metadata,
            "provenance": list(song.provenance) + [provenance_entry],
        }
    )
    # model_copy(update=...) does not re-run validators (pydantic v2). This
    # pass must never hand the pipeline an invalid Song, so re-validate
    # explicitly -- a bug here should raise loudly, never persist silently.
    return Song.model_validate(updated.model_dump())
