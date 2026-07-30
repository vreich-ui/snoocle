"""quality.escalation.plan_escalation: the decision, function-level.

test_quality_pipeline.py already pins the end-to-end MODEL/AUDIO/SOURCE
behaviour through the real pipeline; these tests exercise `plan_escalation`
directly, including the accuracy-escalation branch from issue #59: a failing
`timingCoverage` metric on a run that only analyzed part of the track
(fast/windowed accuracy) recommends a full-accuracy re-analysis instead of
retrying the model -- and that recommendation must never touch the model
retry budget.
"""

from __future__ import annotations

from snoocle_server.quality.attribution import Attribution, Fault
from snoocle_server.quality.escalation import plan_escalation
from snoocle_server.quality.grader import Grade, GradeThresholds, Metric

THRESHOLDS = GradeThresholds()


def _grade(*, coverage_ok: bool | None, coverage_value: float = 0.3, verdict: str = "fail") -> Grade:
    """A minimal failing grade with a `timingCoverage` metric in the given
    state. `coverage_ok=None` means the metric wasn't measured at all."""
    coverage = Metric(
        name="timingCoverage",
        value=coverage_value if coverage_ok is not None else None,
        score=coverage_value if coverage_ok is not None else None,
        ok=coverage_ok,
        threshold=THRESHOLDS.timing_coverage,
        weight=0.2,
        detail="test metric",
    )
    chord_match = Metric(
        name="chordMatchRatio", value=0.9, score=0.9, ok=True,
        threshold=THRESHOLDS.chord_match_ratio, weight=0.25, detail="test metric",
    )
    return Grade(verdict=verdict, overall=0.4, metrics=(coverage, chord_match), thresholds=THRESHOLDS)


_MODEL_ATTRIBUTION = Attribution(fault=Fault.MODEL, actionable=True, reason="model ignored the evidence")
_AUDIO_ATTRIBUTION = Attribution(fault=Fault.AUDIO, actionable=False, reason="the recording is unreliable")
_NONE_ATTRIBUTION = Attribution(fault=Fault.NONE, actionable=False, reason="nothing to attribute")


# --- the new branch: partial accuracy + failing coverage ---------------------


def test_partial_accuracy_with_failing_coverage_recommends_full_reanalysis():
    escalation = plan_escalation(
        _grade(coverage_ok=False), _MODEL_ATTRIBUTION, used_partial_accuracy=True,
    )
    assert escalation.reanalyze_full_accuracy is True
    assert escalation.retry is False
    assert escalation.search is False
    assert escalation.mark_timing_unreliable is False
    assert "timing coverage" in escalation.reason
    assert "fast/windowed accuracy" in escalation.reason
    assert "recommend-full-accuracy-reanalysis" in escalation.describe()
    assert escalation.to_dict()["reanalyzeFullAccuracy"] is True


def test_the_recommendation_overrides_whatever_fault_attribution_says():
    """The gap is indistinguishable from a real fade in this data (issue #59)
    -- the recommendation applies whether attribution called it MODEL, AUDIO,
    or anything else, because the fix is the same either way."""
    for attribution in (_MODEL_ATTRIBUTION, _AUDIO_ATTRIBUTION, _NONE_ATTRIBUTION):
        escalation = plan_escalation(
            _grade(coverage_ok=False), attribution, used_partial_accuracy=True,
        )
        assert escalation.reanalyze_full_accuracy is True
        assert escalation.retry is False


def test_full_accuracy_with_failing_coverage_falls_through_to_existing_verdict():
    """Nothing better to escalate to on a run that already saw the whole
    track -- ordinary MODEL/AUDIO/SOURCE attribution applies unchanged."""
    escalation = plan_escalation(
        _grade(coverage_ok=False), _MODEL_ATTRIBUTION, used_partial_accuracy=False,
    )
    assert escalation.reanalyze_full_accuracy is False
    assert escalation.retry is True
    assert "model fault" in escalation.reason


def test_partial_accuracy_with_passing_coverage_does_not_recommend_reanalysis():
    """Only a FAILING timingCoverage triggers the recommendation -- a partial
    analysis whose coverage happens to be fine has nothing to fix."""
    escalation = plan_escalation(
        _grade(coverage_ok=True, verdict="warn"), _NONE_ATTRIBUTION, used_partial_accuracy=True,
    )
    assert escalation.reanalyze_full_accuracy is False


def test_partial_accuracy_with_unmeasured_coverage_falls_through():
    """No `timingCoverage` metric at all (e.g. duration unknown) -- nothing to
    check the recommendation's own condition against, so it can't fire."""
    escalation = plan_escalation(
        _grade(coverage_ok=None), _MODEL_ATTRIBUTION, used_partial_accuracy=True,
    )
    assert escalation.reanalyze_full_accuracy is False
    assert escalation.retry is True


def test_a_passing_grade_never_recommends_reanalysis_even_when_partial():
    escalation = plan_escalation(
        _grade(coverage_ok=True, verdict="pass"), _NONE_ATTRIBUTION, used_partial_accuracy=True,
    )
    assert escalation.acts is False
    assert escalation.reanalyze_full_accuracy is False


# --- the recommendation must never touch the model retry budget -------------


def test_the_recommendation_never_sets_retry_regardless_of_retries_spent():
    for retries_spent in (0, 1, 2):
        escalation = plan_escalation(
            _grade(coverage_ok=False), _MODEL_ATTRIBUTION,
            used_partial_accuracy=True, retries_spent=retries_spent,
        )
        assert escalation.retry is False
        assert escalation.reanalyze_full_accuracy is True


def test_the_recommendation_fires_the_same_whether_or_not_retrying_is_allowed():
    for can_retry in (True, False):
        escalation = plan_escalation(
            _grade(coverage_ok=False), _MODEL_ATTRIBUTION,
            used_partial_accuracy=True, can_retry=can_retry,
        )
        assert escalation.reanalyze_full_accuracy is True
        assert escalation.retry is False


def test_a_spent_model_retry_still_falls_through_normally_on_full_accuracy():
    """The ceiling that guards the model retry budget is completely
    unaffected by this feature: a full-accuracy run whose one retry is
    already spent gets exactly the pre-existing "stored as-is" outcome."""
    escalation = plan_escalation(
        _grade(coverage_ok=False), _MODEL_ATTRIBUTION,
        used_partial_accuracy=False, retries_spent=1,
    )
    assert escalation.retry is False
    assert escalation.reanalyze_full_accuracy is False
    assert "already spent" in escalation.reason


# --- existing behaviour, unchanged -------------------------------------------


def test_a_passing_grade_still_does_nothing():
    escalation = plan_escalation(_grade(coverage_ok=True, verdict="pass"), _NONE_ATTRIBUTION)
    assert escalation.acts is False
    assert "nothing to escalate" in escalation.reason


def test_audio_fault_still_marks_timing_unreliable_and_never_retries():
    escalation = plan_escalation(_grade(coverage_ok=True), _AUDIO_ATTRIBUTION)
    assert escalation.mark_timing_unreliable is True
    assert escalation.retry is False
    assert escalation.reanalyze_full_accuracy is False
