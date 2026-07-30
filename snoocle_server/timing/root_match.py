"""The root-pitch-class matcher, shared by everything that asks whether a
chord sequence lines up with the audio's own chord timeline.

This is :mod:`timing.snap`'s matcher, extracted verbatim so a second caller
can use it without a second implementation. Two callers exist:

- ``timing.snap`` assigns MIR segment start times to the placements the
  reconciler decided on — the original use, whose behaviour and tests this
  extraction must leave bit-for-bit unchanged.
- ``reconcile.match`` scores a CANDIDATE sheet against the same timeline, at
  each of the 12 transpositions, to answer "is this source good evidence?".

The algorithm (unchanged): walk the chord sequence in reading order against
the no-chord-filtered, start-sorted MIR segments, matching on ROOT PITCH
CLASS with a small forward window. Small on purpose — the sequences should
already be closely aligned in order, and a large window would let one chord
steal a segment that rightfully belongs to a later one. The search pointer
only ever advances, so a match is never reused.

Root, not quality: per the reconcile prompt's own "family" semantics, MIR
reporting a coarser extension of the same root (plain C where the sheet says
Cmaj7) is agreement, not a clash. A genuine major/minor disagreement on a
matched root IS worth reporting, so every match carries ``same_family`` and
lets the caller decide what that is worth (snap lowers confidence; the
candidate scorer records it as a conflict).
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

from ..chords import Chord, ChordParseError, parse_chord
from ..mir.base import ChordSegment, MirAnalysis

#: How many MIR segments ahead of the current search pointer we're willing to
#: scan for a root match before giving up on a chord. See the module docstring
#: for why it is small.
DEFAULT_LOOKAHEAD = 2


def is_minor_family(quality: str) -> bool:
    return quality.startswith("m") and not quality.startswith("maj")


def same_family(quality_a: str, quality_b: str) -> bool:
    return is_minor_family(quality_a) == is_minor_family(quality_b)


def sounding_segments(mir: MirAnalysis) -> list[ChordSegment]:
    """The MIR chord timeline in start-time order, with "N" (no-chord) dropped."""
    return sorted((s for s in mir.chords if s.chord != "N"), key=lambda s: s.start)


class RootMatch(NamedTuple):
    """One chord matched to one MIR segment."""

    segment_index: int  # index into the segment list that was searched
    segment: ChordSegment
    segment_chord: Chord  # the segment's parsed symbol (already known to parse)
    same_family: bool  # major/minor agreement on the matched root


def match_chord_roots(
    chords: Sequence[str],
    segments: Sequence[ChordSegment],
    *,
    transposition: int = 0,
    lookahead: int = DEFAULT_LOOKAHEAD,
) -> dict[int, RootMatch]:
    """``{index into `chords`: RootMatch}`` for every chord that matched.

    `segments` must already be filtered/sorted (see :func:`sounding_segments`).
    `transposition` is added to each CHORD's root before comparing, which is
    how a sheet written in a different key than the recording still matches:
    a sheet in G against a timeline in Bb matches at +3.

    An unparseable chord symbol is skipped without advancing the search
    pointer; an unparseable segment simply never matches.
    """
    matches: dict[int, RootMatch] = {}
    search_idx = 0
    for idx, symbol in enumerate(chords):
        try:
            want = parse_chord(symbol)
        except ChordParseError:
            continue  # not a chord we can reason about; costs no segment
        want_root = (want.root_pc + transposition) % 12

        found_at: Optional[int] = None
        found_chord: Optional[Chord] = None
        scanned = 0
        j = search_idx
        while j < len(segments) and scanned <= lookahead:
            try:
                seg_chord = parse_chord(segments[j].chord)
            except ChordParseError:
                seg_chord = None
            if seg_chord is not None and seg_chord.root_pc == want_root:
                found_at = j
                found_chord = seg_chord
                break
            j += 1
            scanned += 1

        if found_at is not None and found_chord is not None:
            matches[idx] = RootMatch(
                segment_index=found_at,
                segment=segments[found_at],
                segment_chord=found_chord,
                same_family=same_family(found_chord.quality, want.quality),
            )
            search_idx = found_at + 1

    return matches
