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

from ..chords import ChordParseError, parse_chord
from ..mir.base import ChordSegment, MirAnalysis
from ..schema.song import AudioInfo, ChordPlacement, Line, ProvenanceEntry, Song, SyncPoint
from .quantize import DEFAULT_SNAP_TOLERANCE_SECONDS, build_beat_grid, snap_time

# How many MIR segments ahead of the current search pointer we're willing to
# scan for a root match before giving up on a placement. Small on purpose:
# the reconciler's chord sequence and MIR's should already be closely
# aligned in order; a large window would let a placement steal a segment
# that rightfully belongs to a later one.
_LOOKAHEAD = 2


class _MatchedTime(NamedTuple):
    position: int  # index into the flat reading-order placement list
    time: float


def _is_minor_family(quality: str) -> bool:
    return quality.startswith("m") and not quality.startswith("maj")


def _same_family(quality_a: str, quality_b: str) -> bool:
    return _is_minor_family(quality_a) == _is_minor_family(quality_b)


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
    """Returns {flat_index: (time, confidence)} and {flat_index: matched_segment}."""
    segments = sorted(
        (s for s in mir.chords if s.chord != "N"),
        key=lambda s: s.start,
    )
    matches: dict[int, tuple[float, float]] = {}
    matched_segments: dict[int, ChordSegment] = {}
    search_idx = 0
    for flat_idx, (_line_idx, _pi, placement) in enumerate(flat):
        try:
            want = parse_chord(placement.chord)
        except ChordParseError:
            continue  # schema already guarantees this can't happen; be defensive anyway

        found_at: Optional[int] = None
        scanned = 0
        j = search_idx
        while j < len(segments) and scanned <= _LOOKAHEAD:
            seg = segments[j]
            try:
                seg_chord = parse_chord(seg.chord)
            except ChordParseError:
                seg_chord = None
            if seg_chord is not None and seg_chord.root_pc == want.root_pc:
                found_at = j
                break
            j += 1
            scanned += 1

        if found_at is not None:
            seg = segments[found_at]
            seg_chord = parse_chord(seg.chord)
            confidence = 0.9 if _same_family(seg_chord.quality, want.quality) else 0.7
            matches[flat_idx] = (seg.start, confidence)
            matched_segments[flat_idx] = seg
            search_idx = found_at + 1

    return matches, matched_segments


def _interpolate_unmatched(
    flat_len: int, matches: dict[int, tuple[float, float]]
) -> dict[int, tuple[float, float]]:
    """Fill every unmatched position by linear interpolation (by position,
    not by time) between its nearest matched neighbors; confidence 0.3.
    Positions before the first match or after the last reuse that match's
    time (held constant -- never extrapolated past known data)."""
    if not matches:
        return {}

    matched_positions = sorted(matches)
    result: dict[int, tuple[float, float]] = dict(matches)

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
            result[i] = (t0 + frac * (t1 - t0), 0.3)
        elif before is not None:
            result[i] = (matches[before][0], 0.3)
        elif after is not None:
            result[i] = (matches[after][0], 0.3)
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
    assigned = _interpolate_unmatched(len(flat), matches)

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
    provenance_entry = ProvenanceEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        actor="snoocle-server/timing",
        action="timing-snap",
        sources=sources,
        confidence=(matched_count / total) if total else None,
        notes=(
            f"matched {matched_count}/{total} chord placement(s) to the MIR "
            f"timeline; {len(sync_map)}/{len(final_lines)} line(s) timed; "
            f"beats={'filled' if 'beats' in audio_update else 'kept/empty'}"
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
