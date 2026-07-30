"""What to DO about a grade — and, mostly, what not to do.

The rule that shapes this module: **never retry more than once on a single
grade.** The Marley evidence (ten reconciles in sixteen minutes, match ratios
0.41 0.51 0.42 0.53 0.68 0.55 0.48 0.47 0.63 0.49) is what repeated attempts on
the same inputs look like — variance around the same mean, at full price each
time. One retry, given something specific to act on, or none.

Given a grade and its :mod:`quality.attribution`:

- **MODEL fault, failing grade** -> one retry, carrying
  :func:`build_retry_feedback`: the metric names, the measured values, and the
  offending line/placement indexes. Not "try harder".
- **AUDIO fault** -> store it, mark the version timing-unreliable, report an
  alternative recording (``suggest_alternative_recording``), do not retry. The
  document is as good as this recording allows.
- **SOURCE fault** -> allow one targeted search. If it turns up materially
  better evidence, that one retry may use it; if it doesn't, store with the
  grade. The retry budget is one either way.
- **Anything else** (pass, warn, unknown) -> store with the grade.

Checked BEFORE any of the above: a failing ``timingCoverage`` metric on a run
that only analyzed PART of the track (fast/windowed accuracy — see
``MirAnalysis.analyzed_partially``) is not a model problem or an audio
problem in the sense the fault attribution above means. A fast-accuracy
window's edge and a genuine in-window fade produce an identical gap in the
data (see issue #59) — telling them apart needs an amplitude/onset-decay
test, not a boundary check, and #53 deliberately didn't build one. The fix
that already exists is simpler: analyze the same recording again at full
accuracy, where the beat grid reaches every edge it needs to (#53). This is
reported as a RECOMMENDATION (``Escalation.reanalyze_full_accuracy``), the
same way an AUDIO fault's alternative-recording suggestion is reported and
not auto-run — a second MIR pass is real cost, and choosing to pay it is an
operator decision. It is a distinct remedy from a model retry: it never sets
``retry`` and never reads or increments the retry budget, so it can neither
consume it nor be blocked by it having already been spent.

Collapse runs are deliberately NOT a retry or a search path. They are already
handled deterministically, before grading, by
``timing.collapse_guard``: a run gets spread over the span to the next distinct
time using the beat grid where one exists, and a run with no span to spread
into is left exactly as found and recorded as such. A run that reaches the
grader is therefore one nothing could honestly fix — "could not time this
region" beats fabricated spacing, and asking a model to invent the spacing
instead is the worst of both.

That used to need no code here, because a surviving collapse could only ever
grade ``warn`` and this module short-circuits on any verdict but ``fail``.
``collapseRuns`` is now a hard gate (``Metric.hard_gate``) and fails a grade on
its own, which is right — a surviving collapse IS a defect and grading it
``warn`` hid a production incident — but it puts collapse-only failures in
front of the fault branches below, where a MODEL attribution would spend the
retry asking for exactly the invented spacing the guard refused to invent, and
an AUDIO one would spend a live recording search. So the rule above is now
stated as a branch: a grade whose ONLY reason to fail is the collapse gate
takes it, spends no retry and no search, and marks the timing unreliable
instead — the one honest action, and the one a later carry-forward can read to
keep from inheriting that region's times.

That branch requires TWO things, not one. :meth:`Grade.fails_only_on_hard_gate`
answers "is the gate the only fail ROUTE" — which stays useful on its own (a
caller asking "would this have graded warn without this one gate" gets an
honest answer from it, and other code may still want exactly that). But a fail
ROUTE is not the same thing as a failing METRIC: ``_fail_routes`` only adds
``FAIL_ROUTE_FAILING_MAJORITY`` once HALF of the measured metrics fail, so a
document with, say, seven measured metrics can have a second one genuinely
failing — below its own threshold, on its own defect — without that reaching
``half_of_measured`` or dragging ``overall`` under its line either. Take the
gate route alone as license to mark-and-stop and that second failure vanishes
silently: no retry plans for it (the correct remedy for, say, a genuine
``timingCoverage`` shortfall), and the mark-timing-unreliable reason claims the
collapse is "the only thing failing this grade" when it demonstrably is not —
a false claim written into stored provenance. So the branch below additionally
requires the collapse to be the grade's ONLY failing metric
(:attr:`Grade.failing`), not merely its only fail route. A document that fails
for its own reasons and also carries a collapse takes its fault branch
instead, and gets that branch's one retry (or search, or mark) — which is
correct: the module docstring above forbids retrying *for* the collapse
itself, not retrying a document that is broken elsewhere too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .attribution import Attribution, Fault
from .grader import MAX_OFFENDERS, Grade

#: Hard ceiling, stated once. Referenced by the pipeline so the number and the
#: reason live in the same place.
MAX_RETRIES_PER_GRADE = 1

#: How much better a targeted search's evidence must score against the audio
#: before it earns the one retry. A search that comes back with sheets no
#: better than the ones already judged contradictory has not changed the
#: situation, and reconciling again would pay full price for the same result.
SEARCH_IMPROVEMENT_MARGIN = 0.05


@dataclass(frozen=True)
class Escalation:
    """The decision: retry, search, mark, recommend, or accept — and why."""

    retry: bool
    search: bool  # one targeted re-search for better sources
    mark_timing_unreliable: bool
    reason: str
    feedback: Optional[str] = None  # the specific text handed to the model on a retry
    # A RECOMMENDATION, never auto-run: this run only analyzed part of the
    # track (fast/windowed accuracy) and its timing coverage failed. See the
    # module docstring — reported the same way an AUDIO fault's alternative-
    # recording suggestion is, and never combined with `retry`.
    reanalyze_full_accuracy: bool = False
    #: Look for a DIFFERENT recording of this song and report what it finds
    #: (``recordings.suggest_recordings``: one search, no download, no
    #: analysis). Deliberately its OWN flag rather than something a caller
    #: infers from `mark_timing_unreliable`: "this recording cannot be timed"
    #: (AUDIO fault) does imply looking for another one, but "this one region
    #: could not be timed on any recording this run had" (a collapse run that
    #: survived ``timing.collapse_guard``) does not — a different recording
    #: cannot supply a span the guard found no span for, and searching for one
    #: spends a network call on a defect it cannot address.
    suggest_alternative_recording: bool = False
    #: A few words naming what made the timing unreliable, for the step text
    #: and the provenance marker. Empty unless `mark_timing_unreliable` is set.
    mark_cause: str = ""

    @property
    def acts(self) -> bool:
        return (
            self.retry
            or self.search
            or self.mark_timing_unreliable
            or self.reanalyze_full_accuracy
            or self.suggest_alternative_recording
        )

    def describe(self) -> str:
        actions = [
            name
            for name, on in (
                ("retry", self.retry),
                ("targeted-search", self.search),
                ("mark-timing-unreliable", self.mark_timing_unreliable),
                ("suggest-alternative-recording", self.suggest_alternative_recording),
                ("recommend-full-accuracy-reanalysis", self.reanalyze_full_accuracy),
            )
            if on
        ]
        return (", ".join(actions) if actions else "no action") + f" — {self.reason}"

    def to_dict(self) -> dict:
        return {
            "retry": self.retry,
            "search": self.search,
            "markTimingUnreliable": self.mark_timing_unreliable,
            "suggestAlternativeRecording": self.suggest_alternative_recording,
            "reanalyzeFullAccuracy": self.reanalyze_full_accuracy,
            "reason": self.reason,
            "markCause": self.mark_cause,
        }


def plan_escalation(
    grade: Grade,
    attribution: Attribution,
    *,
    retries_spent: int = 0,
    searches_spent: int = 0,
    can_search: bool = True,
    can_retry: bool = True,
    used_partial_accuracy: bool = False,
) -> Escalation:
    """Decide what this run should do about `grade`. Pure and deterministic.

    `retries_spent`/`searches_spent` are what this run has ALREADY used, which
    is what makes the one-retry ceiling a fact rather than a hope: pass the
    real counts and a second escalation cannot happen.

    `used_partial_accuracy` says this run's MIR analysis only covered part of
    the track (``MirAnalysis.analyzed_partially`` — fast accuracy's sampled
    windows). Checked first, before fault attribution even matters: a failing
    ``timingCoverage`` metric on a partial analysis gets the full-accuracy
    reanalysis recommendation instead of whatever the fault-specific branches
    below would have planned — see the module docstring and issue #59.

    Checked second, and also before fault attribution: a grade whose only
    reason to fail is the ``collapseRuns`` hard gate AND whose only failing
    metric is ``collapseRuns`` (see the module docstring — a fail ROUTE and a
    failing METRIC are not the same test, and the branch needs both). Marks
    the timing unreliable and does nothing else — no retry, no search — for
    the reason the module docstring gives.
    """
    if grade.verdict != "fail":
        return Escalation(
            retry=False, search=False, mark_timing_unreliable=False,
            reason=f"grade {grade.verdict}: nothing to escalate",
        )

    coverage = grade.metric("timingCoverage")
    if used_partial_accuracy and coverage is not None and coverage.ok is False:
        return Escalation(
            retry=False, search=False, mark_timing_unreliable=False,
            reanalyze_full_accuracy=True,
            reason=(
                f"timing coverage {coverage.value:.0%} (< {coverage.threshold:.0%}) on a "
                "run that only analyzed part of the track (fast/windowed accuracy) — a "
                "window's edge and a genuine in-window fade look identical in this data "
                "(issue #59), so the recommended fix is a full-accuracy re-analysis of "
                "this same recording, not another model attempt at the same partial "
                "evidence"
            ),
        )

    # Checked before the fault branches for the reason the module docstring
    # gives: a collapse run that survived `timing.collapse_guard` is not a
    # defect any fault-specific remedy addresses, so it must not reach a branch
    # that would spend one. Deliberately AFTER the partial-accuracy
    # recommendation above, which supersedes it: a fast/windowed analysis has no
    # beat grid at the edges the guard needed to spread into, so the
    # full-accuracy re-analysis it recommends is the one remedy that can make
    # the collapse itself go away — deterministically, with no model involved.
    #
    # `fails_only_on_hard_gate` alone is too broad: it answers "is the gate the
    # only fail ROUTE", but `_fail_routes` only adds the failing-majority route
    # once HALF of the measured metrics fail, so a second metric can fail on
    # its own defect -- below its own threshold -- without ever reaching that
    # route or dragging `overall` under its line. Requiring `grade.failing ==
    # (collapse,)` on top closes that gap: this branch fires only when the
    # collapse is the grade's one and only failing metric, not merely its one
    # and only failing ROUTE. See the module docstring.
    collapse = grade.metric("collapseRuns")
    collapse_is_the_whole_defect = (
        collapse is not None
        and collapse.measured
        and grade.fails_only_on_hard_gate("collapseRuns")
        and grade.failing == (collapse,)
    )
    if collapse_is_the_whole_defect:
        return Escalation(
            retry=False, search=False, mark_timing_unreliable=True,
            mark_cause="collapsed timing the guard could not spread",
            reason=(
                f"{collapse.value:.0f} collapsed timing run(s) survived "
                "timing.collapse_guard and are the only thing failing this grade. The "
                "guard already spread every run it had a later distinct time to spread "
                "toward, using the beat grid where one existed; a run that reaches the "
                "grader had none, so nothing could honestly fix it. No retry (asking a "
                "model to space those entries is asking it to invent the spacing the "
                "guard refused to invent) and no search (different sheets cannot supply "
                "times the recording never gave) — the timing of that region is marked "
                f"unreliable instead: {collapse.detail}"
            ),
        )

    if attribution.fault is Fault.AUDIO:
        return Escalation(
            retry=False, search=False, mark_timing_unreliable=True,
            # The recording itself is what cannot be timed, so a DIFFERENT
            # recording is the remedy — reported, never auto-run.
            suggest_alternative_recording=True,
            mark_cause="audio fault",
            reason=(
                "audio fault: storing the document and marking its timing "
                f"unreliable rather than retrying — {attribution.reason}"
            ),
        )

    if attribution.fault is Fault.MODEL:
        if not can_retry:
            return Escalation(
                retry=False, search=False, mark_timing_unreliable=False,
                reason=(
                    "model fault, but retrying is switched off for this run "
                    "(SNOOCLE_QUALITY_RETRY_ENABLED=0, or nothing was regenerated to "
                    "retry); storing with the grade"
                ),
            )
        if retries_spent >= MAX_RETRIES_PER_GRADE:
            return Escalation(
                retry=False, search=False, mark_timing_unreliable=False,
                reason=(
                    f"model fault, but {retries_spent} retry(ies) already spent this "
                    f"run (ceiling {MAX_RETRIES_PER_GRADE}): repeated attempts on the "
                    f"same inputs do not converge, so this grade is stored as-is"
                ),
            )
        return Escalation(
            retry=True, search=False, mark_timing_unreliable=False,
            reason=f"model fault, one retry with the specific failures — {attribution.reason}",
            feedback=build_retry_feedback(grade, attribution),
        )

    if attribution.fault is Fault.SOURCE:
        if not can_search:
            return Escalation(
                retry=False, search=False, mark_timing_unreliable=False,
                reason=(
                    "source fault, but this run is not allowed to gather sources "
                    "(scope); storing with the grade"
                ),
            )
        if searches_spent > 0:
            return Escalation(
                retry=False, search=False, mark_timing_unreliable=False,
                reason=(
                    "source fault, and the one targeted search this run allows is "
                    "already spent; storing with the grade"
                ),
            )
        return Escalation(
            retry=False, search=True, mark_timing_unreliable=False,
            reason=f"source fault, one targeted search for better evidence — {attribution.reason}",
        )

    return Escalation(
        retry=False, search=False, mark_timing_unreliable=False,
        reason=f"fault {attribution.fault.value}: {attribution.reason}",
    )


def _metric_headline(metric) -> str:
    """The one line naming a failing metric: its value, what it needed, and
    its detail.

    `Metric.comparator`, not a list of names kept here: a MAXIMUM-style
    threshold rendered with `>=` states a SATISFIED condition for the metric
    that just failed ("collapseRuns = 1.00 (needs >= 0.0)"), which is worse
    than saying nothing — the model would be asked to fix something the text
    says is already fine. The same trap exists for any metric whose `ok` is a
    COMPOUND test a single (comparator, threshold) pair cannot state truthfully
    (`sectionCoverage`: coverage AND no untimed section AND no excess overlap).

    `Metric.requirement` is where such a metric states its real, compound
    requirement — used here in preference to reconstructing one. When it is
    absent (most metrics: their `ok` really is just `value` against
    `threshold`) this reconstructs the simple clause, but only after checking
    it does not itself contradict `ok is False` — a metric whose `ok` turns
    out to be compound WITHOUT declaring `requirement` must never have this
    render a threshold that reads as already satisfied; it prints no
    value/threshold at all in that case, `metric.detail` alone, rather than a
    lie in either direction.

    An UNMEASURED metric (`value is None`) takes that same detail-alone form,
    for the same reason and one more: there is no number to print, and both
    clause-building branches below format `value` with `:.2f`. Unreachable
    today — `grade_song` gives every unmeasured metric `ok=None`, and
    `Grade.failing` selects `ok is False` — so this is a guard against a future
    metric that fails without a value, not a live bug. It is a REAL guard
    rather than a `value is not None` conjunct in the reconstruction test:
    that conjunct read as None-safety while routing `value=None, ok=False`
    into `f"{metric.value:.2f}"` and a `TypeError`.
    """
    if metric.value is None:
        return f"- **{metric.name}**: {metric.detail}"
    if metric.requirement:
        return (
            f"- **{metric.name}** = {metric.value:.2f} (needs {metric.requirement}): "
            f"{metric.detail}"
        )
    reconstructed_satisfied = (
        metric.value <= metric.threshold if metric.maximum else metric.value >= metric.threshold
    )
    if reconstructed_satisfied:
        return f"- **{metric.name}**: {metric.detail}"
    return (
        f"- **{metric.name}** = {metric.value:.2f} (needs "
        f"{metric.comparator} {metric.threshold}): "
        f"{metric.detail}"
    )


def build_retry_feedback(grade: Grade, attribution: Attribution) -> str:
    """The text a retry is given: named metrics, named indexes.

    A vague "the last attempt was low quality, try harder" invites the model
    to re-roll the same document. This says which measurement failed, what it
    measured, and exactly which lines and placements produced it — and it
    states WHICH situation the retry is (the document ignored good evidence,
    or the evidence itself has been replaced), because those call for
    different work.
    """
    situation = (
        "The candidate sheets in this request agree with the audio analysis better "
        "than your document does, so the evidence for a correct answer is already "
        "in front of you. Fix these specific measurements:"
        if attribution.fault is Fault.MODEL
        else "The sources the previous attempt had contradicted each other. This "
        "request carries freshly gathered sheets that agree with the audio better — "
        "reconcile from those. Fix these specific measurements:"
    )
    lines = [
        "## Quality grade of your previous attempt: FAIL",
        (
            f"A deterministic grader scored the document you just produced at "
            f"{grade.overall:.2f} overall (threshold {grade.thresholds.overall:.2f}). "
            if grade.overall is not None
            else "A deterministic grader failed the document you just produced. "
        )
        + attribution.reason
        + ".",
        "",
        situation,
    ]
    for metric in grade.failing:
        lines.append("")
        lines.append(_metric_headline(metric))
        for offender in metric.offenders[:MAX_OFFENDERS]:
            lines.append(f"    - {_offender_text(offender)}")
        if len(metric.offenders) > MAX_OFFENDERS:
            lines.append(
                f"    - ... and {len(metric.offenders) - MAX_OFFENDERS} more of the same kind"
            )
    lines += [
        "",
        "Correct these against the evidence you were given — the candidate sheets and "
        "the MIR chord timeline in this same request. Do not invent chords, lyrics or "
        "times to satisfy a number, and do not change anything the grade did not flag: "
        "a document that trades a real reading for a better score is worse than the one "
        "you just produced. This is the only retry.",
    ]
    return "\n".join(lines)


def _offender_text(offender: dict) -> str:
    """One offender as a line of prose. Unknown shapes are printed verbatim
    rather than dropped — a grade must never hide what it measured."""
    if "gap" in offender:
        a, b = offender["gap"]
        return f"no section covers {a}s-{b}s"
    if offender.get("kind") == "lines":
        return (
            f"lines {offender['fromLineIndex']}-{offender['toLineIndex']} all share "
            f"timeSeconds {offender['timeSeconds']}"
        )
    if offender.get("kind") == "placements":
        return (
            f"line {offender['lineIndex']}: placements at charIndex "
            f"{offender['fromCharIndex']}-{offender['toCharIndex']} all share "
            f"timeSeconds {offender['timeSeconds']}"
        )
    if "chord" in offender:
        where = f"line {offender.get('lineIndex')}, charIndex {offender.get('charIndex')}"
        reason = offender.get("reason")
        return f"{where}: {offender['chord']}" + (f" — {reason}" if reason else "")
    if "sectionIndex" in offender:
        return (
            f"section {offender['sectionIndex']} ({offender.get('name')}): "
            f"{offender.get('reason')}"
        )
    if "lineIndex" in offender:
        return f"line {offender['lineIndex']}: {offender.get('reason', 'flagged')}"
    return "; ".join(f"{k}={v}" for k, v in offender.items())


def search_found_better(
    old_scores, new_scores, *, margin: float = SEARCH_IMPROVEMENT_MARGIN
) -> bool:
    """Did a targeted search actually improve the evidence?

    Compares the MEAN agreement with the audio timeline
    (``reconcile.match.score_candidate``) across candidates that carry chords
    at all. Empty new evidence never wins; equal evidence never wins — see
    :data:`SEARCH_IMPROVEMENT_MARGIN`.
    """

    def mean(scores) -> Optional[float]:
        usable = [s.score for s in scores if s.total > 0]
        return sum(usable) / len(usable) if usable else None

    new_mean = mean(new_scores)
    if new_mean is None:
        return False
    old_mean = mean(old_scores)
    return old_mean is None or new_mean >= old_mean + margin
