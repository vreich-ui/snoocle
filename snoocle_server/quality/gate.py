"""The three quality steps as one call: grade, attribute, decide.

Every caller wants all three and wants them in that order, because each reads
the one before it. Keeping the sequence here rather than in each orchestrator
is what stops a second caller (Mode B —
:mod:`snoocle_server.realign`) from grading with slightly different inputs than
the analyze pipeline does, which would make the two songs' grade histories
incomparable for no reason anyone would notice.

Pure and deterministic: a function of (document, MIR, candidates) plus the
budget already spent, so the same run trace always re-derives the same
decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..mir.base import MirAnalysis
from ..schema.song import Song
from .attribution import Attribution, attribute_fault
from .escalation import Escalation, plan_escalation
from .grader import Grade, GradeThresholds, grade_song


@dataclass(frozen=True)
class QualityDecision:
    """What the gate concluded, in the order it concluded it."""

    grade: Grade
    attribution: Attribution
    escalation: Escalation

    def describe(self) -> str:
        return f"{self.grade.describe()} | fault: {self.attribution.describe()}"

    def to_dict(self) -> dict:
        return {
            "grade": self.grade.to_dict(),
            "attribution": self.attribution.to_dict(),
            "escalation": self.escalation.to_dict(),
        }


def evaluate(
    song: Song,
    mir: Optional[MirAnalysis] = None,
    candidates: Sequence = (),
    *,
    can_search: bool = True,
    can_retry: bool = True,
    retries_spent: int = 0,
    searches_spent: int = 0,
    sources_expected: bool = True,
    thresholds: Optional[GradeThresholds] = None,
) -> QualityDecision:
    """Grade `song`, attribute the fault, and plan the escalation.

    `retries_spent`/`searches_spent` are what the calling run has ALREADY
    used — passing the real counts is what makes the one-retry ceiling
    structural rather than a promise. `sources_expected=False` marks a run that
    gathered no text sources on purpose (Mode B); see
    :func:`quality.attribution.attribute_fault`.
    """
    grade = grade_song(song, mir, candidates, thresholds=thresholds)
    attribution = attribute_fault(
        grade, mir, candidates, sources_expected=sources_expected
    )
    escalation = plan_escalation(
        grade,
        attribution,
        retries_spent=retries_spent,
        searches_spent=searches_spent,
        can_search=can_search,
        can_retry=can_retry,
        used_partial_accuracy=mir.analyzed_partially if mir is not None else False,
    )
    return QualityDecision(grade=grade, attribution=attribution, escalation=escalation)
