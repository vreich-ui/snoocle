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
- **AUDIO fault** -> store it, mark the version timing-unreliable, do not
  retry. The document is as good as this recording allows.
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

Collapse runs are deliberately NOT an escalation path. They are already
handled deterministically, before grading, by
``timing.collapse_guard``: a run gets spread over the span to the next distinct
time using the beat grid where one exists, and a run with no span to spread
into is left exactly as found and recorded as such. A run that reaches the
grader is therefore one nothing could honestly fix — "could not time this
region" beats fabricated spacing, and asking a model to invent the spacing
instead is the worst of both.
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

    @property
    def acts(self) -> bool:
        return (
            self.retry
            or self.search
            or self.mark_timing_unreliable
            or self.reanalyze_full_accuracy
        )

    def describe(self) -> str:
        actions = [
            name
            for name, on in (
                ("retry", self.retry),
                ("targeted-search", self.search),
                ("mark-timing-unreliable", self.mark_timing_unreliable),
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
            "reanalyzeFullAccuracy": self.reanalyze_full_accuracy,
            "reason": self.reason,
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

    if attribution.fault is Fault.AUDIO:
        return Escalation(
            retry=False, search=False, mark_timing_unreliable=True,
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
        lines.append(
            f"- **{metric.name}** = {metric.value:.2f} (needs "
            f"{'<=' if metric.name == 'interpolationShare' else '>='} {metric.threshold}): "
            f"{metric.detail}"
        )
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
