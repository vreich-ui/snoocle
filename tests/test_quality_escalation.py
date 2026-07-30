"""quality.escalation.plan_escalation: the decision, function-level.

test_quality_pipeline.py already pins the end-to-end MODEL/AUDIO/SOURCE
behaviour through the real pipeline; these tests exercise `plan_escalation`
directly, including the accuracy-escalation branch from issue #59: a failing
`timingCoverage` metric on a run that only analyzed part of the track
(fast/windowed accuracy) recommends a full-accuracy re-analysis instead of
retrying the model -- and that recommendation must never touch the model
retry budget.

Plus the collapse branch: `collapseRuns` became a hard gate that fails a grade
on its own, which put collapse-only failures in front of the fault branches --
where a MODEL attribution would spend the one retry asking the model to invent
the spacing `timing.collapse_guard` deliberately refused to invent, and an
AUDIO one would spend a live recording search on a region no other recording
of the song can time either. The branch keeps the gate and makes the ACTION
honest: mark the timing unreliable, spend nothing.
"""

from __future__ import annotations

from snoocle_server.quality.attribution import Attribution, Fault
from snoocle_server.quality.escalation import build_retry_feedback, plan_escalation
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
_SOURCE_ATTRIBUTION = Attribution(fault=Fault.SOURCE, actionable=True, reason="the sheets contradict each other")
_NONE_ATTRIBUTION = Attribution(fault=Fault.NONE, actionable=False, reason="nothing to attribute")


def _ok(name: str, value: float, threshold: float, weight: float) -> Metric:
    return Metric(
        name=name, value=value, score=value, ok=True, threshold=threshold,
        weight=weight, detail="test metric",
    )


def _collapse(*, runs: float = 1.0, ok: bool = False) -> Metric:
    """`collapseRuns` as the grader builds it: a hard gate, a MAXIMUM whose
    threshold is literally zero, and a density `score` a single short run
    barely dents."""
    return Metric(
        name="collapseRuns", value=runs, score=0.96, ok=ok, threshold=0.0, weight=0.1,
        hard_gate=True, maximum=True,
        detail=(
            "1 run(s) of >=3 consecutive entries sharing one timestamp survive the "
            "collapse guard (3/40 timed entr(y/ies) piled onto an anchor)"
        ),
        offenders=[
            {"kind": "lines", "fromLineIndex": 16, "toLineIndex": 19,
             "timeSeconds": 200.0, "count": 4}
        ],
    )


def _collapse_only_fail(*, overall: float = 0.897) -> Grade:
    """The reported shape: a full-length MIR, a tail collapse, and every other
    metric comfortably fine -- so `collapseRuns`' hard gate is the ONLY route
    to `fail` (the overall holds and one failure of three measured metrics is
    nowhere near the majority route)."""
    return Grade(
        verdict="fail",
        overall=overall,
        metrics=(
            _ok("chordMatchRatio", 0.9, THRESHOLDS.chord_match_ratio, 0.25),
            _ok("timingCoverage", 0.95, THRESHOLDS.timing_coverage, 0.2),
            _collapse(),
        ),
        thresholds=THRESHOLDS,
    )


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


# --- a collapse-only fail: mark the timing, spend nothing -------------------


def test_a_collapse_only_fail_spends_neither_a_retry_nor_a_search():
    """Whatever the fault attribution says. `timing.collapse_guard` already
    spread every run it had a span and a beat grid to spread into; a run that
    reaches the grader had neither, so no model attempt and no other set of
    sheets can supply the times -- the only honest action left is to say the
    timing of that region is not to be trusted."""
    for attribution in (
        _MODEL_ATTRIBUTION, _AUDIO_ATTRIBUTION, _SOURCE_ATTRIBUTION, _NONE_ATTRIBUTION,
    ):
        escalation = plan_escalation(_collapse_only_fail(), attribution)
        assert escalation.retry is False, attribution.fault
        assert escalation.search is False, attribution.fault
        assert escalation.feedback is None, attribution.fault
        assert escalation.mark_timing_unreliable is True, attribution.fault
        assert escalation.reanalyze_full_accuracy is False, attribution.fault
        assert escalation.acts is True, attribution.fault
        # The reason cites the deterministic guard, not a new rationale.
        assert "timing.collapse_guard" in escalation.reason
        assert "invent" in escalation.reason


def test_a_collapse_only_fails_claim_of_being_the_only_failure_is_true():
    """The reason text says the collapse is "the only thing failing this
    grade" -- that has to be a true statement about the grade it is written
    for, not just about which fail ROUTE tripped. `_collapse_only_fail`'s
    other two metrics genuinely pass (see its own docstring), so `grade.failing`
    really is exactly the collapse metric here, and the claim the reason
    makes is one this fixture actually backs up."""
    grade = _collapse_only_fail()
    collapse = grade.metric("collapseRuns")
    assert grade.failing == (collapse,), "the fixture must be genuinely collapse-only"

    escalation = plan_escalation(grade, _MODEL_ATTRIBUTION)
    assert "are the only thing failing this grade" in escalation.reason


def test_a_collapse_only_fail_makes_no_recording_search_but_still_marks():
    """The two used to be one flag. A surviving collapse marks the timing (a
    later carry-forward has to be able to see it) and must NOT ask for another
    recording: the guard found no span to spread into, and no other recording
    of the song supplies one."""
    escalation = plan_escalation(_collapse_only_fail(), _AUDIO_ATTRIBUTION)

    assert escalation.mark_timing_unreliable is True
    assert escalation.suggest_alternative_recording is False
    assert escalation.to_dict()["suggestAlternativeRecording"] is False
    assert escalation.to_dict()["markTimingUnreliable"] is True
    # The step text and the provenance marker read off this, so it has to name
    # the real cause rather than the AUDIO fault it used to hardcode.
    assert "collapse" in escalation.mark_cause
    assert "audio" not in escalation.mark_cause
    assert escalation.to_dict()["markCause"] == escalation.mark_cause
    assert "mark-timing-unreliable" in escalation.describe()
    assert "suggest-alternative-recording" not in escalation.describe()


def test_a_collapse_only_fail_leaves_the_whole_retry_budget_unspent():
    """Not "the retry is already spent" -- never requested. A collapse must
    not be able to consume the one retry a genuine MODEL fault later in the
    same run would need, nor be blocked by it."""
    for retries_spent in (0, 1):
        escalation = plan_escalation(
            _collapse_only_fail(), _MODEL_ATTRIBUTION,
            retries_spent=retries_spent, can_retry=True,
        )
        assert escalation.retry is False
        assert "already spent" not in escalation.reason


def test_a_collapse_only_fail_leaves_the_search_budget_unspent():
    escalation = plan_escalation(
        _collapse_only_fail(), _SOURCE_ATTRIBUTION, searches_spent=0, can_search=True,
    )
    assert escalation.search is False
    assert "targeted search" not in escalation.describe()


def test_a_fail_the_collapse_did_not_cause_still_gets_its_fault_branch():
    """The branch is narrow on purpose: it fires only when the collapse gate is
    the ONE thing failing the grade. A document that is broadly bad AND happens
    to carry a collapse is a MODEL fault like any other, and still buys its one
    retry -- suppressing that would hide real regressions behind an incidental
    collapse."""
    broadly_bad = Grade(
        verdict="fail",
        overall=0.41,  # below the 0.6 fail line all by itself
        metrics=(
            Metric(name="chordMatchRatio", value=0.2, score=0.2, ok=False,
                   threshold=THRESHOLDS.chord_match_ratio, weight=0.25, detail="test metric"),
            _ok("timingCoverage", 0.95, THRESHOLDS.timing_coverage, 0.2),
            _collapse(),
        ),
        thresholds=THRESHOLDS,
    )
    assert broadly_bad.fails_only_on_hard_gate("collapseRuns") is False

    escalation = plan_escalation(broadly_bad, _MODEL_ATTRIBUTION)
    assert escalation.retry is True
    assert "model fault" in escalation.reason


def test_a_collapse_that_only_warned_is_still_not_escalated_at_all():
    """Belt and braces on the short-circuit: nothing about the new branch may
    make a non-`fail` verdict act."""
    warned = Grade(
        verdict="warn", overall=0.9,
        metrics=(_ok("chordMatchRatio", 0.9, THRESHOLDS.chord_match_ratio, 0.25), _collapse()),
        thresholds=THRESHOLDS,
    )
    escalation = plan_escalation(warned, _MODEL_ATTRIBUTION)
    assert escalation.acts is False
    assert "nothing to escalate" in escalation.reason


def test_the_full_accuracy_recommendation_still_comes_before_the_collapse_branch():
    """Ordering, deliberately: a fast/windowed analysis has no beat grid at the
    edges the guard needed to spread into, so the full-accuracy re-analysis is
    the one remedy that can make the collapse itself go away -- and it is still
    a recommendation that spends nothing."""
    partial = Grade(
        verdict="fail", overall=0.7,
        metrics=(
            _ok("chordMatchRatio", 0.9, THRESHOLDS.chord_match_ratio, 0.25),
            Metric(name="timingCoverage", value=0.3, score=0.3, ok=False,
                   threshold=THRESHOLDS.timing_coverage, weight=0.2, detail="test metric"),
            _ok("interpolationShare", 0.1, THRESHOLDS.interpolation_share, 0.1),
            _collapse(),
            _ok("lyricCompleteness", 1.0, THRESHOLDS.lyric_completeness, 0.1),
        ),
        thresholds=THRESHOLDS,
    )
    # `timingCoverage` and `collapseRuns` both genuinely fail here, so this is
    # NOT a collapse-only fail by the grade's failing METRICS -- only its fail
    # ROUTES read that way (`collapseRuns`' hard gate is the only ROUTE:
    # `overall` holds and 2 failures of 5 measured metrics is below
    # `half_of_measured`). That distinction is exactly what
    # `plan_escalation`'s collapse branch now also requires (see
    # `test_a_fail_route_of_hard_gate_alone_is_not_enough_with_a_second_failing_metric`
    # below) -- it does not matter for THIS test either way, though: the
    # partial-accuracy recommendation is checked before the collapse branch
    # gets a look at all, so it must win regardless of how the collapse
    # branch's own condition would have read.
    assert partial.fails_only_on_hard_gate("collapseRuns") is True
    assert partial.failing != (partial.metric("collapseRuns"),)

    escalation = plan_escalation(partial, _MODEL_ATTRIBUTION, used_partial_accuracy=True)
    assert escalation.reanalyze_full_accuracy is True
    assert escalation.retry is False
    assert escalation.search is False
    assert escalation.mark_timing_unreliable is False


def test_a_fail_route_of_hard_gate_alone_is_not_enough_with_a_second_failing_metric():
    """The gap the adversarial review found: `fails_only_on_hard_gate` answers
    "is the gate the only fail ROUTE", not "is the gate the only failing
    METRIC" -- `_fail_routes` only adds the failing-majority route once HALF
    of the measured metrics fail, so a second metric can fail on its own
    defect without ever reaching that route or dragging `overall` under its
    line. Five metrics measured, `half_of_measured` is 3: `timingCoverage`
    genuinely fails alongside `collapseRuns`, giving only 2 failures -- nowhere
    near the majority route -- so `collapseRuns`' hard gate is STILL this
    grade's only fail ROUTE even though it is not its only failing metric.
    The collapse branch must not fire here: this is an ordinary MODEL fault
    that also happens to carry a collapse, and it still buys its one retry."""
    grade = Grade(
        verdict="fail", overall=0.83,
        metrics=(
            _ok("chordMatchRatio", 0.9, THRESHOLDS.chord_match_ratio, 0.25),
            Metric(name="timingCoverage", value=0.55, score=0.55, ok=False,
                   threshold=THRESHOLDS.timing_coverage, weight=0.2, detail="test metric"),
            _ok("interpolationShare", 0.1, THRESHOLDS.interpolation_share, 0.1),
            _collapse(),
            _ok("lyricCompleteness", 1.0, THRESHOLDS.lyric_completeness, 0.1),
        ),
        thresholds=THRESHOLDS,
    )
    assert grade.fail_routes == ("hard-gate:collapseRuns",)
    assert grade.fails_only_on_hard_gate("collapseRuns") is True
    assert len(grade.failing) == 2, "the gap: two failing metrics, one fail route"

    escalation = plan_escalation(grade, _MODEL_ATTRIBUTION)
    assert escalation.retry is True
    assert escalation.mark_timing_unreliable is False
    assert "model fault" in escalation.reason
    feedback = escalation.feedback
    assert feedback is not None
    assert "timingCoverage" in feedback
    assert "collapseRuns" in feedback


# --- rendering a threshold that points the other way ------------------------


def test_the_retry_feedback_prints_maximum_thresholds_with_the_right_comparator():
    """`collapseRuns = 1.00 (needs >= 0.0)` states a SATISFIED condition for a
    metric that just failed -- worse than saying nothing, because the model is
    being told the thing it is asked to fix is already fine. Every
    MAXIMUM-style metric (see `Metric.maximum`) renders `<=`."""
    grade = Grade(
        verdict="fail", overall=0.3,
        metrics=(
            Metric(name="chordMatchRatio", value=0.2, score=0.2, ok=False,
                   threshold=THRESHOLDS.chord_match_ratio, weight=0.25,
                   detail="2/10 placement(s) match"),
            Metric(name="interpolationShare", value=0.8, score=0.2, ok=False,
                   threshold=THRESHOLDS.interpolation_share, weight=0.1, maximum=True,
                   detail="8/10 timed placement(s) are interpolated"),
            _collapse(),
        ),
        thresholds=THRESHOLDS,
    )
    text = build_retry_feedback(grade, _MODEL_ATTRIBUTION)

    assert "**collapseRuns** = 1.00 (needs <= 0.0)" in text
    assert "**interpolationShare** = 0.80 (needs <= 0.5)" in text
    # The minimum-style metrics are untouched.
    assert "**chordMatchRatio** = 0.20 (needs >= 0.5)" in text
    assert "needs >= 0.0" not in text


def test_a_compound_ok_metric_renders_its_real_requirement_not_a_satisfied_clause():
    """The exact shape the adversarial review reproduced: `sectionCoverage`'s
    coverage NUMBER is satisfied (1.00 >= its 0.75 threshold) while `ok` is
    still `False` because of an untimed section -- a renderer that only knows
    `comparator`/`threshold` would print "needs >= 0.75", stating a condition
    that reads as already met for a metric that just failed. `requirement`
    must be what gets rendered instead."""
    section_coverage = Metric(
        name="sectionCoverage", value=1.0, score=1.0, ok=False,
        threshold=0.75, weight=0.1,
        requirement=">= 0.75 coverage, no untimed section, <= 0.5s overlap",
        detail=(
            "sections span 100.0% of the 60.0s track; 1/2 untimed, 0 gap(s), "
            "0.0s of overlap"
        ),
    )
    grade = Grade(
        verdict="fail", overall=0.7,
        metrics=(
            _ok("chordMatchRatio", 0.9, THRESHOLDS.chord_match_ratio, 0.25),
            section_coverage,
        ),
        thresholds=THRESHOLDS,
    )
    text = build_retry_feedback(grade, _MODEL_ATTRIBUTION)

    assert (
        "**sectionCoverage** = 1.00 (needs >= 0.75 coverage, no untimed section, "
        "<= 0.5s overlap): sections span 100.0%" in text
    )
    # The self-contradicting reconstruction this bug used to print.
    assert "needs >= 0.75):" not in text


def test_a_compound_ok_metric_without_a_declared_requirement_never_renders_a_lie():
    """The safety net for any FUTURE compound metric that forgets to declare
    `requirement`: if the reconstructed (comparator, threshold) clause would
    read as satisfied for a metric whose `ok` is `False`, the renderer must
    print no threshold clause at all -- `metric.detail` alone still tells the
    model something true, and a false "already fine" claim is worse than
    printing less."""
    mystery = Metric(
        name="mysteryMetric", value=1.0, score=1.0, ok=False,
        threshold=0.75, weight=0.1,
        detail="the number looks fine but a different condition failed",
    )
    grade = Grade(
        verdict="fail", overall=0.7,
        metrics=(
            _ok("chordMatchRatio", 0.9, THRESHOLDS.chord_match_ratio, 0.25),
            mystery,
        ),
        thresholds=THRESHOLDS,
    )
    text = build_retry_feedback(grade, _MODEL_ATTRIBUTION)

    assert "**mysteryMetric**: the number looks fine but a different condition failed" in text
    assert "needs >=" not in text.split("mysteryMetric")[1].split("\n")[0]
    assert "1.00 (needs" not in text


def test_an_unmeasured_failing_metric_prints_its_detail_instead_of_crashing():
    """A `value is not None` conjunct used to sit in the reconstruction test,
    reading as None-safety while giving none: a `value=None, ok=False` metric
    evaluated it False and fell straight through to `f"{metric.value:.2f}"` and
    a TypeError, and the `requirement` branch above it never checked at all.

    Unreachable today -- `grade_song` gives every unmeasured metric `ok=None`
    and `Grade.failing` selects `ok is False` -- so this pins the guard as a
    REAL one rather than a comment that implies protection it does not give.
    """
    for extra in ({}, {"requirement": ">= 0.75 coverage, no untimed section"}):
        unmeasured = Metric(
            name="unmeasurable", value=None, score=None, ok=False,
            threshold=0.75, weight=0.1,
            detail="nothing to measure, and it still failed",
            **extra,
        )
        grade = Grade(
            verdict="fail", overall=0.7,
            metrics=(_ok("chordMatchRatio", 0.9, THRESHOLDS.chord_match_ratio, 0.25), unmeasured),
            thresholds=THRESHOLDS,
        )
        text = build_retry_feedback(grade, _MODEL_ATTRIBUTION)  # no TypeError
        assert "**unmeasurable**: nothing to measure, and it still failed" in text
        assert "needs" not in text.split("unmeasurable")[1].split("\n")[0]


# --- Escalation.to_dict names the cause, not just the flags ------------------


def test_to_dict_carries_mark_cause_alongside_the_boolean_flags():
    """`markTimingUnreliable: true` alone cannot tell a consumer WHY -- a
    collapse and an unreliable recording both set it. `report.steps` and the
    provenance note both state the cause in prose; `to_dict()` (what
    `pipeline.py`'s `recorder.set_quality` and `RealignReport.to_dict` both
    serialise) has to carry the same fact structurally, not just in `reason`
    free text a consumer would otherwise have to substring-match."""
    collapse_escalation = plan_escalation(_collapse_only_fail(), _AUDIO_ATTRIBUTION)
    payload = collapse_escalation.to_dict()
    assert payload["markCause"] == "collapsed timing the guard could not spread"
    assert payload["markTimingUnreliable"] is True

    audio_escalation = plan_escalation(_grade(coverage_ok=True), _AUDIO_ATTRIBUTION)
    payload = audio_escalation.to_dict()
    assert payload["markCause"] == "audio fault"
    assert payload["markTimingUnreliable"] is True

    # A non-marking escalation carries the field too, just empty -- the key is
    # always present, so a consumer never has to guess whether it was omitted.
    passing = plan_escalation(_grade(coverage_ok=True, verdict="pass"), _NONE_ATTRIBUTION)
    assert passing.to_dict()["markCause"] == ""


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


def test_audio_fault_still_asks_for_an_alternative_recording():
    """The other half of the decoupling: THIS recording is what cannot be
    timed, so a different recording is the remedy, and the flag that spends the
    search says so on its own."""
    escalation = plan_escalation(_grade(coverage_ok=True), _AUDIO_ATTRIBUTION)
    assert escalation.suggest_alternative_recording is True
    assert escalation.to_dict()["suggestAlternativeRecording"] is True
    assert escalation.mark_cause == "audio fault"
    assert "suggest-alternative-recording" in escalation.describe()
