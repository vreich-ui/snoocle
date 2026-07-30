"""How well does one candidate sheet agree with the audio?

Reconciliation has always treated candidates as interchangeable evidence with
a confidence number attached by whoever found them (search rank, UG votes).
Nothing measured a candidate against the recording it claims to describe —
which is the only question that separates "this sheet is wrong" from "this
sheet is right in a different key" from "the audio is too lo-fi to tell".

``score_candidate`` answers it deterministically, with no model and no
network:

- Walk the candidate's chords in reading order against the MIR chord
  timeline, matching on root pitch class — the SAME matcher
  :mod:`timing.snap` uses (``timing.root_match``), not a second
  implementation of it.
- Do that at all TWELVE transpositions and keep the best. A sheet written in
  G against a recording in Bb matches ~nothing at 0 and ~everything at +3:
  it is a GOOD source that needs transposing, and scoring it as garbage is
  exactly the mistake that makes a re-search look necessary when it isn't.
  Ties go to the smallest shift (and 0 wins outright), so a sheet already in
  the recording's key is never reported as transposed.
- Report major/minor clashes on matched roots as ``conflicts``. A clash is
  not a scoring penalty — the root matched, which is what carries the
  alignment — but "the sheet says Am where the audio says A" is precisely
  the disagreement a human reviewer wants to see.

Used by :mod:`quality.attribution` to decide whether a bad document is the
model's fault or the evidence's, and by :mod:`manifest` to describe how good
this run's sources actually were.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..chords import ChordParseError, parse_chord
from ..discovery.models import CandidateSource
from ..mir.base import MirAnalysis
from ..timing.root_match import match_chord_roots, sounding_segments

#: Every transposition, as a SIGNED semitone count in the order we prefer to
#: report them: no shift first, then the smallest shifts, downward before
#: upward at equal distance (so a tritone reports as -6, never +6).
TRANSPOSITIONS: tuple[int, ...] = (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6)


@dataclass(frozen=True)
class CandidateScore:
    """One candidate's agreement with the MIR timeline at its best transposition."""

    score: float  # matched / total, in [0, 1]; 0.0 when there was nothing to compare
    transposition: int  # semitones to ADD to the sheet's chords to reach the audio
    matched: int
    total: int  # parseable chord symbols in the candidate
    conflicts: list[dict] = field(default_factory=list)
    source_id: Optional[str] = None

    def to_dict(self) -> dict:
        """The compact form recorded on a manifest / in provenance."""
        out: dict = {
            "score": round(self.score, 3),
            "transposition": self.transposition,
            "matched": self.matched,
            "total": self.total,
            "conflicts": len(self.conflicts),
        }
        if self.source_id is not None:
            out["sourceId"] = self.source_id
        return out

    def describe(self) -> str:
        shift = "" if self.transposition == 0 else f" at {self.transposition:+d} semitone(s)"
        return (
            f"{self.source_id or 'candidate'}: {self.matched}/{self.total} "
            f"chord(s) match the audio{shift} ({self.score:.0%})"
            + (f", {len(self.conflicts)} family clash(es)" if self.conflicts else "")
        )


def _candidate_chords(candidate: CandidateSource) -> list[tuple[int, int, str]]:
    """(lineIndex, charIndex, symbol) in reading order, parseable symbols only.

    Unparseable tokens are dropped rather than counted as misses: a sheet is
    not worse evidence about harmony because its parser choked on one cell,
    and the matcher skips them anyway.
    """
    out: list[tuple[int, int, str]] = []
    for line in candidate.lines:
        for placement in sorted(line.chordPlacements, key=lambda p: p.charIndex):
            try:
                parse_chord(placement.chord)
            except ChordParseError:
                continue
            out.append((line.lineIndex, placement.charIndex, placement.chord))
    return out


def score_candidate(candidate: CandidateSource, mir: Optional[MirAnalysis]) -> CandidateScore:
    """Score `candidate` against `mir`'s chord timeline over all 12 transpositions.

    A score of 0.0 with ``total == 0`` means "this candidate had no chords to
    compare" and a score of 0.0 with ``total > 0`` means "it has chords and
    none of them line up" — the caller must not conflate them, so both the
    numerator and the denominator are returned rather than the ratio alone.
    """
    chords = _candidate_chords(candidate)
    segments = sounding_segments(mir) if mir is not None else []
    total = len(chords)
    if not chords or not segments:
        return CandidateScore(
            score=0.0,
            transposition=0,
            matched=0,
            total=total,
            source_id=candidate.sourceId,
        )

    symbols = [symbol for _li, _ci, symbol in chords]
    best_t = 0
    best_matches: dict = {}
    for t in TRANSPOSITIONS:
        matches = match_chord_roots(symbols, segments, transposition=t)
        # Strictly greater: TRANSPOSITIONS is ordered by preference, so the
        # first shift to reach a given count keeps it.
        if len(matches) > len(best_matches):
            best_t, best_matches = t, matches

    conflicts = [
        {
            "lineIndex": chords[idx][0],
            "charIndex": chords[idx][1],
            "chord": chords[idx][2],
            "audioChord": match.segment.chord,
            "timeSeconds": round(match.segment.start, 2),
            "reason": "major/minor family clash on a matched root",
        }
        for idx, match in sorted(best_matches.items())
        if not match.same_family
    ]

    return CandidateScore(
        score=len(best_matches) / total,
        transposition=best_t,
        matched=len(best_matches),
        total=total,
        conflicts=conflicts,
        source_id=candidate.sourceId,
    )


def score_candidates(
    candidates: Sequence[CandidateSource], mir: Optional[MirAnalysis]
) -> list[CandidateScore]:
    """:func:`score_candidate` for each candidate, in the order given."""
    return [score_candidate(c, mir) for c in candidates]
