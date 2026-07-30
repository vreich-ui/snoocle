"""Whose fault is a bad grade? The part that has to be right.

Retrying is the expensive reflex, and most of the time it is the wrong one. A
run whose EVIDENCE was bad pays full price for the same result: the Marley
song was reconciled ten times in sixteen minutes at match ratios 0.41 0.51
0.42 0.53 0.68 0.55 0.48 0.47 0.63 0.49 — ten full-price attempts, no
convergence, because nothing ever asked whether a retry could help.

So before anything escalates, three patterns are separated by comparing the
document, the candidate sheets, and the audio timeline against each other.
Each candidate is scored against the MIR timeline at all 12 transpositions
(``reconcile.match.score_candidate``) so a sheet in the "wrong" key counts as
the good evidence it is:

- The candidates agree with each other AND with the MIR timeline, but the
  document agrees with neither -> **MODEL** fault. The evidence was there and
  the document did not use it. Actionable: one retry, handed the specific
  metrics and indexes.
- The candidates agree with each other and the MIR timeline contradicts them
  -> **AUDIO** fault. A lo-fi, live, or broadcast recording whose chord
  timeline is not trustworthy (the 1966 live Paint It Black: an MIR chord
  timeline that dies at 86.5s of a 220.6s track). NOT actionable by retry —
  the model would be asked to fit the same unreliable timeline again.
- The candidates disagree with each other -> **SOURCE** fault. Better sources
  are needed; a retry against sheets we already know contradict each other
  buys nothing.

And two honest non-answers, both non-actionable: **NONE** (nothing is wrong)
and **UNKNOWN** (there is not enough evidence to attribute anything — no MIR,
or a grade nothing could be measured on). Guessing "model" when the evidence
cannot support it is how a retry budget gets burned on runs that were never
going to converge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Optional, Sequence

from ..chords import ChordParseError, parse_chord
from ..discovery.models import CandidateSource
from ..mir.base import MirAnalysis
from ..reconcile.match import CandidateScore, score_candidates
from .grader import Grade


class Fault(str, Enum):
    NONE = "none"
    MODEL = "model"
    AUDIO = "audio"
    SOURCE = "source"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AttributionThresholds:
    """Where each comparison tips over. Configurable via ``SNOOCLE_QUALITY_*``."""

    #: Below this mean pairwise agreement, the candidate sheets are considered
    #: to disagree with each other -> SOURCE.
    source_agreement: float = 0.6
    #: Below this mean candidate-vs-MIR score, the audio is considered to
    #: contradict sources that otherwise agree -> AUDIO. Read it as "no better
    #: than coincidence": the greedy root matcher this shares with
    #: ``timing.snap`` scores ~0.38 on two UNRELATED chord sequences of equal
    #: length (measured, see tests/test_quality_attribution.py), because a
    #: 12-pitch alphabet and a 3-segment search window make accidental in-order
    #: matches cheap. A candidate set that cannot beat 0.5 against a timeline is
    #: not describing the same recording the recognizer heard.
    mir_agreement: float = 0.5
    #: Below this share of the track spanned by the MIR chord timeline, the
    #: audio analysis is too partial to blame anything else -> AUDIO.
    mir_timeline_coverage: float = 0.5
    #: How far the DOCUMENT must fall below what the candidates achieve
    #: against the same timeline before the gap is the model's fault.
    model_margin: float = 0.15

    @classmethod
    def from_settings(cls, settings=None) -> "AttributionThresholds":
        if settings is None:
            from ..config import settings as default_settings

            settings = default_settings
        values = {}
        for f in cls.__dataclass_fields__:
            value = getattr(settings, f"quality_{f}", None)
            if value is not None:
                values[f] = value
        try:
            return cls(**values)
        except Exception:  # noqa: BLE001 — bad config degrades to defaults
            return cls()


@dataclass(frozen=True)
class Attribution:
    """Who is responsible for the grade, and whether anything can be done."""

    fault: Fault
    actionable: bool  # can a retry / a better search plausibly change the outcome?
    reason: str
    candidate_agreement: Optional[float] = None  # sources vs each other
    candidate_mir_agreement: Optional[float] = None  # sources vs the audio timeline
    document_mir_agreement: Optional[float] = None  # the document vs the audio timeline
    mir_timeline_coverage: Optional[float] = None
    candidate_scores: tuple[CandidateScore, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        return f"{self.fault.value} ({'actionable' if self.actionable else 'not actionable'}) — {self.reason}"

    def to_dict(self) -> dict:
        def r(v):
            return round(v, 3) if v is not None else None

        return {
            "fault": self.fault.value,
            "actionable": self.actionable,
            "reason": self.reason,
            "candidateAgreement": r(self.candidate_agreement),
            "candidateMirAgreement": r(self.candidate_mir_agreement),
            "documentMirAgreement": r(self.document_mir_agreement),
            "mirTimelineCoverage": r(self.mir_timeline_coverage),
            "candidateScores": [s.to_dict() for s in self.candidate_scores],
        }


def _root_sequence(candidate: CandidateSource) -> list[int]:
    """The candidate's chord ROOTS in reading order.

    Roots only, for the same reason ``timing.root_match`` matches on roots: a
    sheet writing Cmaj7 where another writes C is not a disagreement about the
    song.
    """
    out: list[int] = []
    for line in candidate.lines:
        for p in sorted(line.chordPlacements, key=lambda p: p.charIndex):
            try:
                out.append(parse_chord(p.chord).root_pc)
            except ChordParseError:
                continue
    return out


def _interval_sequence(candidate: CandidateSource) -> list[int]:
    """Successive root INTERVALS — the shape of the progression, not its key.

    Transposition-invariance for free, which matters twice over. The obvious
    alternative (absolute roots, compared at whichever of the 12 shifts scores
    best) is both more work and less discriminating: taking a maximum over 12
    attempts inflates the similarity of unrelated pop progressions to ~0.75
    on a 12-pitch alphabet, where intervals put two transcriptions of the same
    song at 0.92-1.0 and two different songs at 0.3-0.6.
    """
    roots = _root_sequence(candidate)
    return [(roots[i + 1] - roots[i]) % 12 for i in range(len(roots) - 1)]


def candidate_agreement(candidates: Sequence[CandidateSource]) -> Optional[float]:
    """Mean pairwise agreement between the candidates' chord progressions.

    Compared as interval sequences (see :func:`_interval_sequence`), so two
    sheets of the same song written in different keys count as agreeing —
    which they do.

    ``None`` with fewer than two candidates carrying a progression: one source
    cannot corroborate or contradict anything, and reporting 1.0 there would
    read as "the sources agree" when nothing was compared.
    """
    sequences = [seq for c in candidates if (seq := _interval_sequence(c))]
    if len(sequences) < 2:
        return None
    ratios = [
        SequenceMatcher(None, a, b, autojunk=False).ratio()
        for i, a in enumerate(sequences)
        for b in sequences[i + 1 :]
    ]
    return sum(ratios) / len(ratios)


def mir_timeline_coverage(mir: MirAnalysis) -> Optional[float]:
    """Fraction of the track the MIR CHORD timeline actually spans.

    The signature of an unusable recording: the chord recognizer stops
    producing segments partway through (86.5s of 220.6s on the 1966 live
    take), so everything downstream is interpolating over silence.
    """
    if not mir.duration_seconds or mir.duration_seconds <= 0:
        return None
    sounding = [s for s in mir.chords if s.chord != "N"]
    if not sounding:
        return 0.0
    span = max(s.end for s in sounding) - min(s.start for s in sounding)
    return max(0.0, min(1.0, span / mir.duration_seconds))


def attribute_fault(
    grade: Grade,
    mir: Optional[MirAnalysis] = None,
    candidates: Sequence[CandidateSource] = (),
    *,
    thresholds: Optional[AttributionThresholds] = None,
    scores: Optional[Sequence[CandidateScore]] = None,
    sources_expected: bool = True,
) -> Attribution:
    """Attribute `grade` to the model, the audio, or the sources.

    Pure and deterministic. `scores` may be passed in when the caller already
    computed them (the pipeline does, for the manifest) — otherwise they are
    computed here.

    `sources_expected=False` says this run never gathered text sources and was
    not supposed to (Mode B re-times a document that already exists — see
    :mod:`snoocle_server.realign`). Without that, an empty candidate set reads
    as "the sources failed", which would blame the wrong thing AND mask the
    audio checks below — the ones that matter when a re-alignment lands on
    another unusable recording.
    """
    t = thresholds or AttributionThresholds.from_settings()

    if grade.verdict == "pass":
        return Attribution(
            fault=Fault.NONE,
            actionable=False,
            reason="the grade is pass; nothing to attribute",
            document_mir_agreement=grade.value("chordMatchRatio"),
        )
    # `warn` is deliberately NOT short-circuited here. A `warn` grade still
    # carries at least one failing metric (see `grade_song`) — it only means
    # nothing HARD-gated tripped and the weighted overall held. That failing
    # metric deserves a real, named fault (MODEL/AUDIO/SOURCE/UNKNOWN) from
    # the same comparison logic a `fail` gets, not a blanket NONE that reads
    # as "nothing is wrong" when something measurably is. Whatever this finds
    # falls through to `quality.escalation`, which only acts on
    # `grade.verdict == "fail"` — so a `warn`'s fault is recorded honestly
    # without ever spending a retry or a search on it.
    if grade.verdict == "unknown":
        return Attribution(
            fault=Fault.UNKNOWN,
            actionable=False,
            reason=(
                "too little measurable signal to grade, let alone attribute "
                "(no retry: there is no stated defect for a retry to fix)"
            ),
        )

    doc_vs_mir = grade.value("chordMatchRatio")

    if mir is None:
        return Attribution(
            fault=Fault.UNKNOWN,
            actionable=False,
            reason=(
                "no audio analysis in this run, so the document cannot be compared "
                "against the recording — nothing to attribute this grade to"
            ),
            document_mir_agreement=doc_vs_mir,
        )

    resolved_scores = tuple(scores if scores is not None else score_candidates(candidates, mir))
    coverage = mir_timeline_coverage(mir)
    scored = [s for s in resolved_scores if s.total > 0]
    cand_vs_mir = (
        sum(s.score for s in scored) / len(scored) if scored else None
    )
    agreement = candidate_agreement(list(candidates))

    common = {
        "candidate_agreement": agreement,
        "candidate_mir_agreement": cand_vs_mir,
        "document_mir_agreement": doc_vs_mir,
        "mir_timeline_coverage": coverage,
        "candidate_scores": resolved_scores,
    }

    if not scored and sources_expected:
        # No usable text evidence in a run that should have had some. Not the
        # model's fault — it had nothing to reconcile from — and a retry would
        # hand it the same nothing again.
        return Attribution(
            fault=Fault.SOURCE,
            actionable=True,
            reason=(
                "no candidate source carried chords this run, so the document had "
                "no text evidence to agree with; better sources are the only fix"
            ),
            **common,
        )

    if agreement is not None and agreement < t.source_agreement:
        return Attribution(
            fault=Fault.SOURCE,
            actionable=True,
            reason=(
                f"the {len(scored)} candidate sheets agree with each other only "
                f"{agreement:.0%} (< {t.source_agreement:.0%}) — they contradict each "
                f"other, so retrying against the same sheets cannot converge"
            ),
            **common,
        )

    if coverage is not None and coverage < t.mir_timeline_coverage:
        return Attribution(
            fault=Fault.AUDIO,
            actionable=False,
            reason=(
                f"the MIR chord timeline spans only {coverage:.0%} of the track "
                f"(< {t.mir_timeline_coverage:.0%}) — a lo-fi/live/broadcast source the "
                f"analysis could not follow; a retry would fit the same partial timeline"
            ),
            **common,
        )

    if not scored:
        # A run with no text sources by design (see `sources_expected`). The
        # audio checks above still applied; the source and model comparisons
        # below cannot, because both need something to compare the document
        # against.
        return Attribution(
            fault=Fault.UNKNOWN,
            actionable=False,
            reason=(
                "this run gathered no text sources by design, and the audio analysis "
                "covers the track, so nothing here identifies what a second attempt "
                "would do differently"
            ),
            **common,
        )

    if cand_vs_mir is not None and cand_vs_mir < t.mir_agreement:
        return Attribution(
            fault=Fault.AUDIO,
            actionable=False,
            reason=(
                f"the candidate sheets agree with each other "
                + (f"({agreement:.0%}) " if agreement is not None else "")
                + f"but only {cand_vs_mir:.0%} of their chords match the audio timeline "
                f"(< {t.mir_agreement:.0%}) — the recording, not the document, is what "
                f"disagrees; not fixable by a retry"
            ),
            **common,
        )

    if doc_vs_mir is not None and doc_vs_mir + t.model_margin < cand_vs_mir:
        return Attribution(
            fault=Fault.MODEL,
            actionable=True,
            reason=(
                f"the candidate sheets match the audio {cand_vs_mir:.0%} of the time and "
                f"the document only {doc_vs_mir:.0%} — the evidence agreed and the "
                f"document did not use it"
            ),
            **common,
        )

    return Attribution(
        fault=Fault.UNKNOWN,
        actionable=False,
        reason=(
            "this grade has failing metrics but sources, audio and document all agree "
            "with each other as well as they ever do — no retry, since nothing "
            "identifies what a second attempt would do differently"
        ),
        **common,
    )
