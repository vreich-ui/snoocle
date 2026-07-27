"""Per-chord agreement scoring: how much independent evidence backs each
stored chord, and a compact "review this first" queue built from it.

Two independent signals, each in [0, 1] when available at all:

- **source agreement**: of the candidate text sources that have a line
  fuzzily matching this one (difflib ratio on the lyric text -- reconciled
  lyrics rarely match a source's spelling/punctuation exactly), what
  fraction agree on the chord ROOT nearest the same proportional position
  in that line? charIndex doesn't line up across sources either (different
  line lengths), so position is compared as a fraction of line length.
- **MIR agreement**: does the MIR chord timeline's root at this placement's
  `timeSeconds` (set by :mod:`timing.snap`, which must run BEFORE this
  module for that signal to exist) match the stored chord's root?

Per the reconcile prompt's own "family" semantics (docs/ARCHITECTURE.md),
this module only compares ROOT for agreement -- a source or MIR reporting a
coarser/richer extension of the same root+family is not a disagreement.

`combined = 0.5*source_agreement + 0.5*mir_agreement` REPLACES the coarser
confidence `timing.snap` assigned, but only when BOTH signals are available
-- a placement with only one signal (or neither) keeps its snap-time
confidence rather than being penalized for missing data it was never going
to have (e.g. no candidate source covers that line, or MIR has no coverage
there).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from ..chords import ChordParseError, parse_chord
from ..discovery.models import CandidateSource
from ..mir.base import MirAnalysis
from ..schema.song import Line, Song

# Below this, a candidate source's line is not considered "the same line" --
# reconciled lyrics can differ from a source's spelling, but not this much.
_LINE_MATCH_THRESHOLD = 0.5
# Below this, a MIR segment is too far from the placement's time to count as
# "covering" it (handles the last placement landing exactly on a segment's
# end due to float rounding).
_MIR_TIME_TOLERANCE = 0.5

REVIEW_QUEUE_THRESHOLD = 0.6


@dataclass
class PlacementScore:
    lineIndex: int
    charIndex: int
    chord: str
    confidence: float
    reasons: list[str] = field(default_factory=list)


def _best_matching_line(candidate_lines: list[Line], target_lyrics: str) -> Optional[Line]:
    best: Optional[Line] = None
    best_ratio = 0.0
    for cl in candidate_lines:
        if not cl.lyrics and not target_lyrics:
            continue
        ratio = SequenceMatcher(None, cl.lyrics.casefold(), target_lyrics.casefold()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = cl
    return best if best is not None and best_ratio >= _LINE_MATCH_THRESHOLD else None


def _nearest_root(cand_line: Line, frac: float) -> Optional[int]:
    if not cand_line.chordPlacements:
        return None
    target_pos = frac * max(len(cand_line.lyrics), 1)
    nearest = min(cand_line.chordPlacements, key=lambda p: abs(p.charIndex - target_pos))
    try:
        return parse_chord(nearest.chord).root_pc
    except ChordParseError:
        return None


def _source_agreement(
    candidates: list[CandidateSource], line: Line, placement_char_index: int, want_root_pc: int
) -> tuple[Optional[float], int, int]:
    frac = placement_char_index / len(line.lyrics) if line.lyrics else 0.0
    agreements = 0
    considered = 0
    for cand in candidates:
        cand_line = _best_matching_line(cand.lines, line.lyrics)
        if cand_line is None:
            continue
        root = _nearest_root(cand_line, frac)
        if root is None:
            continue
        considered += 1
        if root == want_root_pc:
            agreements += 1
    if considered == 0:
        return None, agreements, considered
    return agreements / considered, agreements, considered


def _mir_root_at(mir: MirAnalysis, t: float) -> Optional[int]:
    segments = [s for s in mir.chords if s.chord != "N"]
    if not segments:
        return None
    covering = [s for s in segments if s.start <= t < s.end]
    seg = covering[0] if covering else min(
        segments, key=lambda s: min(abs(s.start - t), abs(s.end - t))
    )
    if not covering and min(abs(seg.start - t), abs(seg.end - t)) > _MIR_TIME_TOLERANCE:
        return None
    try:
        return parse_chord(seg.chord).root_pc
    except ChordParseError:
        return None


def score_song(
    song: Song, candidates: list[CandidateSource], mir: Optional[MirAnalysis]
) -> tuple[Song, list[PlacementScore]]:
    """Returns (song-with-updated-confidences, every placement's score).

    Every placement is scored (not just low-confidence ones) so callers can
    build whatever view they need; :func:`build_review_queue` is the
    "show me what to check" filter on top of this.
    """
    scores: list[PlacementScore] = []
    new_lines = []
    for line in song.lines:
        new_placements = []
        for placement in line.chordPlacements:
            try:
                want = parse_chord(placement.chord)
            except ChordParseError:
                new_placements.append(placement)
                continue

            source_frac, agreements, considered = _source_agreement(
                candidates, line, placement.charIndex, want.root_pc
            )
            mir_frac: Optional[float] = None
            if mir is not None and placement.timeSeconds is not None:
                mir_root = _mir_root_at(mir, placement.timeSeconds)
                if mir_root is not None:
                    mir_frac = 1.0 if mir_root == want.root_pc else 0.0

            reasons: list[str] = []
            confidence = placement.confidence if placement.confidence is not None else 0.5
            if source_frac is not None and mir_frac is not None:
                confidence = 0.5 * source_frac + 0.5 * mir_frac
            if source_frac is not None and source_frac < 1.0:
                reasons.append(
                    f"only {agreements}/{considered} candidate source(s) agree on this chord"
                )
            if mir_frac is not None and mir_frac < 1.0:
                reasons.append("the audio (MIR) analysis heard a different root here")
            if source_frac is None and mir_frac is None:
                reasons.append("no independent signal available (no matching source line, no MIR coverage)")

            scores.append(
                PlacementScore(
                    lineIndex=line.lineIndex,
                    charIndex=placement.charIndex,
                    chord=placement.chord,
                    confidence=confidence,
                    reasons=reasons,
                )
            )
            if confidence != placement.confidence:
                new_placements.append(placement.model_copy(update={"confidence": confidence}))
            else:
                new_placements.append(placement)
        new_lines.append(line.model_copy(update={"chordPlacements": new_placements}))

    updated = song.model_copy(update={"lines": new_lines})
    return Song.model_validate(updated.model_dump()), scores


def build_review_queue(
    scores: list[PlacementScore], threshold: float = REVIEW_QUEUE_THRESHOLD
) -> list[dict]:
    """Compact, ascending-by-confidence list of placements worth a human
    look. Deliberately a list of plain dicts (not a schema type) -- this
    lives on the run trace, not the Song."""
    low = [s for s in scores if s.confidence < threshold]
    low.sort(key=lambda s: s.confidence)
    return [
        {
            "lineIndex": s.lineIndex,
            "charIndex": s.charIndex,
            "chord": s.chord,
            "confidence": round(s.confidence, 3),
            "reasons": s.reasons,
        }
        for s in low
    ]
