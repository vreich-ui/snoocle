"""Quality grading: judge a reconciled document, attribute the fault, and
escalate only when escalation can actually help.

Three modules, in the order the pipeline uses them:

- :mod:`quality.grader` — the deterministic grade (per-metric values plus an
  overall verdict). No model, no network. Recorded in provenance on every run.
- :mod:`quality.attribution` — MODEL vs AUDIO vs SOURCE, by comparing the
  document, the candidate sheets and the MIR timeline against each other.
- :mod:`quality.escalation` — the decision, with a hard ceiling of one retry
  per grade.

``quality.theory`` holds the music21-backed key-explainability check, which is
optional (extra ``theory``) and reports "not measured" when absent.
"""

from __future__ import annotations

from .attribution import Attribution, AttributionThresholds, Fault, attribute_fault
from .escalation import MAX_RETRIES_PER_GRADE, Escalation, build_retry_feedback, plan_escalation
from .grader import (
    Grade,
    GradeThresholds,
    Metric,
    grade_provenance_entry,
    grade_song,
)
from .theory import TheoryReport, theory_validity

__all__ = [
    "Attribution",
    "AttributionThresholds",
    "Escalation",
    "Fault",
    "Grade",
    "GradeThresholds",
    "MAX_RETRIES_PER_GRADE",
    "Metric",
    "TheoryReport",
    "attribute_fault",
    "build_retry_feedback",
    "grade_provenance_entry",
    "grade_song",
    "plan_escalation",
    "theory_validity",
]
