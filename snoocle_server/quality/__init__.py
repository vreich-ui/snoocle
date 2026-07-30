"""Quality grading: judge a reconciled document, attribute the fault, and
escalate only when escalation can actually help.

Four modules. :mod:`quality.gate` is the entry point — it runs the other three
in the order they depend on each other, so every caller grades the same way:

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
from .gate import QualityDecision, evaluate
from .grader import (
    Grade,
    GradeThresholds,
    Metric,
    grade_provenance_entry,
    grade_song,
    timing_unreliable_provenance_entry,
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
    "QualityDecision",
    "TheoryReport",
    "attribute_fault",
    "build_retry_feedback",
    "evaluate",
    "grade_provenance_entry",
    "grade_song",
    "plan_escalation",
    "theory_validity",
    "timing_unreliable_provenance_entry",
]
