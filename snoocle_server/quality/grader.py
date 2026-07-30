"""The deterministic grade: how good is this reconciled document, really?

Every signal here already existed somewhere in the pipeline and was thrown
away. Two production runs make the cost concrete:

- ``wil-per--rolling-stones-paint-it-black-live-1966``: timing-snap recorded
  "matched 9/86 chord placement(s)" (0.10) on one attempt and 32/66 (0.48) on
  the next, both stored without comment. The track is 220.6s; the last line is
  timed at 86.5s; lines 17-20 share one timestamp; every section has a null
  start/endTime.
- ``bob-marley--three-little-birds``: ten reconciles in sixteen minutes, match
  ratios 0.41 0.51 0.42 0.53 0.68 0.55 0.48 0.47 0.63 0.49 — no convergence,
  nothing noticing.

So: one pure function, no model, no network, that reads a Song against the MIR
analysis it was built from and returns a per-metric grade plus a verdict.

Design rules this module holds to:

- **Reuse, never reimplement.** The chord match ratio is ``timing.snap``'s own
  matcher (via ``timing.root_match``), timing coverage is
  ``timing.collapse_guard.timing_coverage``, collapse runs are that module's
  ``find_collapsed_runs``, interpolation share counts placements sitting at
  ``timing.snap.INTERPOLATED_CONFIDENCE``. A grade that measured something
  subtly different from what the pipeline actually did would be worse than no
  grade at all.
- **"Not measurable" is not "bad".** A metric with no inputs (no MIR, no
  duration, no key, no sections) reports ``None`` and is excluded from the
  overall — it never drags a document down for evidence it was never given.
- **Every run gets graded and recorded.** :func:`grade_provenance_entry`
  produces the provenance entry for the grade regardless of the verdict: the
  history of a song's grades is the thing that would have made the Marley
  non-convergence visible on the second run instead of the tenth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

from ..mir.base import MirAnalysis
from ..schema.song import ProvenanceEntry, Song
from ..timing.collapse_guard import COLLAPSE_RUN_THRESHOLD, find_collapsed_runs, timing_coverage
from ..timing.confidence import best_matching_line
from ..timing.root_match import match_chord_roots, sounding_segments
from ..timing.snap import INTERPOLATED_CONFIDENCE
from .theory import TheoryReport, theory_validity

#: Section kinds where a line legitimately has no words.
_WORDLESS_SECTION_KINDS = frozenset(
    {"intro", "instrumental", "solo", "interlude", "breakdown", "outro"}
)

#: A verdict needs something to stand on. Below this many measurable metrics
#: the grade is "unknown" — a document nobody could measure must not be
#: escalated as if it had been found wanting.
MIN_MEASURED_METRICS = 2

#: How many offenders any one metric reports. The point is to hand a retry
#: specific indexes, not to ship the whole document back.
MAX_OFFENDERS = 12


@dataclass(frozen=True)
class GradeThresholds:
    """Where "good enough" sits for each metric. Every value is configurable
    (see :meth:`from_settings`); the defaults are read off the production
    evidence in the module docstring, not guessed."""

    chord_match_ratio: float = 0.5  # Paint It Black scored 0.10 and 0.48
    timing_coverage: float = 0.6  # ... at 39% coverage
    interpolation_share: float = 0.5  # MAXIMUM: over half interpolated is not timing
    collapse_runs: int = 0  # MAXIMUM: any run surviving the guard is a defect
    section_coverage: float = 0.75
    theory_validity: float = 0.85
    lyric_completeness: float = 0.95
    overall: float = 0.6  # below this the verdict is "fail"

    @classmethod
    def from_settings(cls, settings=None) -> "GradeThresholds":
        """Thresholds from runtime config (``SNOOCLE_QUALITY_*``).

        A missing or unreadable setting keeps this class's default — a grader
        must never fail a run over its own configuration.
        """
        if settings is None:
            from ..config import settings as default_settings

            settings = default_settings
        values = {}
        for f in cls.__dataclass_fields__:
            attr = f"quality_{f}"
            value = getattr(settings, attr, None)
            if value is not None:
                values[f] = value
        try:
            return cls(**values)
        except Exception:  # noqa: BLE001 — bad config degrades to defaults
            return cls()


@dataclass(frozen=True)
class Metric:
    """One measurement, its threshold, and who to blame for it.

    ``value`` is the raw measurement in its own units (a ratio, or a count);
    ``score`` is that value normalized so 1.0 is perfect, which is what the
    overall is a weighted mean of. Both are reported: "3 collapsed runs" is
    what a human wants to read, and the normalized form is what a number can
    be computed from.
    """

    name: str
    value: Optional[float]
    score: Optional[float]
    ok: Optional[bool]
    threshold: float
    weight: float
    detail: str = ""
    offenders: list[dict] = field(default_factory=list)

    @property
    def measured(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict:
        out: dict = {
            "value": round(self.value, 4) if self.value is not None else None,
            "threshold": self.threshold,
            "ok": self.ok,
        }
        if self.score is not None and self.score != self.value:
            out["score"] = round(self.score, 4)
        if self.detail:
            out["detail"] = self.detail
        if self.offenders:
            out["offenders"] = self.offenders[:MAX_OFFENDERS]
        return out


@dataclass(frozen=True)
class Grade:
    """The full grade: every metric, the weighted overall, and a verdict.

    Verdicts: ``fail`` when the weighted overall is below its threshold OR when
    half of the measurable metrics fail (perfect chords and words cannot make
    an untimed document playable); ``warn`` when something failed but the
    overall holds; ``pass`` when nothing failed; ``unknown`` when fewer than
    :data:`MIN_MEASURED_METRICS` metrics could be measured at all. Only
    ``fail`` can escalate — see :mod:`quality.escalation`.
    """

    verdict: str  # "pass" | "warn" | "fail" | "unknown"
    overall: Optional[float]
    metrics: tuple[Metric, ...]
    thresholds: GradeThresholds

    def metric(self, name: str) -> Optional[Metric]:
        return next((m for m in self.metrics if m.name == name), None)

    def value(self, name: str) -> Optional[float]:
        m = self.metric(name)
        return m.value if m is not None else None

    @property
    def measured(self) -> tuple[Metric, ...]:
        return tuple(m for m in self.metrics if m.measured)

    @property
    def failing(self) -> tuple[Metric, ...]:
        return tuple(m for m in self.metrics if m.ok is False)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "overall": round(self.overall, 4) if self.overall is not None else None,
            "metrics": {m.name: m.to_dict() for m in self.metrics},
            "failing": [m.name for m in self.failing],
        }

    def describe(self) -> str:
        parts = []
        for m in self.metrics:
            if not m.measured:
                continue
            flag = "" if m.ok is not False else " FAIL"
            parts.append(f"{m.name}={m.value:.2f}{flag}")
        head = (
            f"{self.verdict} (overall {self.overall:.2f})"
            if self.overall is not None
            else f"{self.verdict} (not enough measurable signal)"
        )
        return head + (": " + ", ".join(parts) if parts else "")


# --- individual metrics ------------------------------------------------------


def _flat_chords(song: Song) -> list[tuple[int, int, str]]:
    """(lineIndex, charIndex, symbol) in reading order — the same order
    ``timing.snap`` flattens placements in, so indexes line up with it."""
    return [
        (line.lineIndex, p.charIndex, p.chord)
        for line in song.lines
        for p in line.chordPlacements
    ]


def _chord_match_metric(
    song: Song, mir: Optional[MirAnalysis], t: GradeThresholds
) -> Metric:
    chords = _flat_chords(song)
    if mir is None or not chords:
        return Metric(
            name="chordMatchRatio", value=None, score=None, ok=None,
            threshold=t.chord_match_ratio, weight=0.25,
            detail=("no MIR analysis to match against" if mir is None
                    else "the document has no chords"),
        )
    segments = sounding_segments(mir)
    matches = match_chord_roots([c for _li, _ci, c in chords], segments)
    ratio = len(matches) / len(chords)
    offenders = [
        {"lineIndex": li, "charIndex": ci, "chord": chord}
        for idx, (li, ci, chord) in enumerate(chords)
        if idx not in matches
    ]
    return Metric(
        name="chordMatchRatio", value=ratio, score=ratio, ok=ratio >= t.chord_match_ratio,
        threshold=t.chord_match_ratio, weight=0.25,
        detail=(
            f"{len(matches)}/{len(chords)} placement(s) match the audio's chord "
            f"timeline ({len(segments)} sounding segment(s))"
        ),
        offenders=offenders,
    )


def _timing_coverage_metric(
    song: Song, duration: Optional[float], t: GradeThresholds
) -> Metric:
    coverage = timing_coverage(list(song.lines), duration)
    if coverage is None:
        return Metric(
            name="timingCoverage", value=None, score=None, ok=None,
            threshold=t.timing_coverage, weight=0.2,
            detail="track duration unknown — nothing to measure coverage against",
        )
    timed = [line for line in song.lines if line.timeSeconds is not None]
    detail = (
        f"line timings span {coverage:.1%} of the {duration:.1f}s track "
        f"({len(timed)}/{len(song.lines)} line(s) timed"
        + (
            f", last at {max(line.timeSeconds for line in timed):.1f}s)"
            if timed
            else ", none timed)"
        )
    )
    return Metric(
        name="timingCoverage", value=coverage, score=coverage,
        ok=coverage >= t.timing_coverage, threshold=t.timing_coverage, weight=0.2,
        detail=detail,
    )


def _interpolation_metric(song: Song, t: GradeThresholds) -> Metric:
    timed = [
        (line.lineIndex, p)
        for line in song.lines
        for p in line.chordPlacements
        if p.timeSeconds is not None
    ]
    if not timed:
        return Metric(
            name="interpolationShare", value=None, score=None, ok=None,
            threshold=t.interpolation_share, weight=0.1,
            detail="no placement carries a time — nothing was interpolated or measured",
        )
    interpolated = [
        (li, p)
        for li, p in timed
        if p.confidence is not None and p.confidence <= INTERPOLATED_CONFIDENCE + 1e-9
    ]
    share = len(interpolated) / len(timed)
    return Metric(
        name="interpolationShare", value=share, score=1.0 - share,
        ok=share <= t.interpolation_share, threshold=t.interpolation_share, weight=0.1,
        detail=(
            f"{len(interpolated)}/{len(timed)} timed placement(s) sit at the "
            f"interpolated confidence tier ({INTERPOLATED_CONFIDENCE}) — derived "
            f"from neighbours, not heard in the audio"
        ),
        offenders=[
            {"lineIndex": li, "charIndex": p.charIndex, "chord": p.chord}
            for li, p in interpolated
        ],
    )


def _collapse_metric(song: Song, t: GradeThresholds) -> Metric:
    line_times = [line.timeSeconds for line in song.lines]
    entries = sum(1 for x in line_times if x is not None)
    runs: list[dict] = []
    collapsed_entries = 0

    for start, end, value in find_collapsed_runs(line_times):
        collapsed_entries += end - start - 1
        runs.append(
            {
                "kind": "lines",
                "fromLineIndex": song.lines[start].lineIndex,
                "toLineIndex": song.lines[end - 1].lineIndex,
                "timeSeconds": round(value, 3),
                "count": end - start,
            }
        )
    for line in song.lines:
        p_times = [p.timeSeconds for p in line.chordPlacements]
        entries += sum(1 for x in p_times if x is not None)
        for start, end, value in find_collapsed_runs(p_times):
            collapsed_entries += end - start - 1
            runs.append(
                {
                    "kind": "placements",
                    "lineIndex": line.lineIndex,
                    "fromCharIndex": line.chordPlacements[start].charIndex,
                    "toCharIndex": line.chordPlacements[end - 1].charIndex,
                    "timeSeconds": round(value, 3),
                    "count": end - start,
                }
            )

    if not entries:
        return Metric(
            name="collapseRuns", value=None, score=None, ok=None,
            threshold=float(t.collapse_runs), weight=0.1,
            detail="nothing is timed — there is no collapse to detect",
        )
    return Metric(
        name="collapseRuns", value=float(len(runs)),
        score=max(0.0, 1.0 - collapsed_entries / entries),
        ok=len(runs) <= t.collapse_runs, threshold=float(t.collapse_runs), weight=0.1,
        detail=(
            f"{len(runs)} run(s) of >={COLLAPSE_RUN_THRESHOLD} consecutive entries "
            f"sharing one timestamp survive the collapse guard "
            f"({collapsed_entries}/{entries} timed entr(y/ies) piled onto an anchor)"
        ),
        offenders=runs,
    )


def _section_coverage_metric(
    song: Song, duration: Optional[float], t: GradeThresholds
) -> Metric:
    weight = 0.1
    if not song.sections:
        return Metric(
            name="sectionCoverage", value=None, score=None, ok=None,
            threshold=t.section_coverage, weight=weight,
            detail="the document has no sections",
        )
    if duration is None or duration <= 0:
        return Metric(
            name="sectionCoverage", value=None, score=None, ok=None,
            threshold=t.section_coverage, weight=weight,
            detail="track duration unknown — section spans have nothing to cover",
        )

    spans = sorted(
        (max(0.0, s.startTime), min(duration, s.endTime))
        for s in song.sections
        if s.startTime is not None and s.endTime is not None
    )
    untimed = [s for s in song.sections if s.startTime is None or s.endTime is None]
    offenders: list[dict] = [
        {"sectionIndex": s.sectionIndex, "name": s.name, "reason": "no startTime/endTime"}
        for s in untimed
    ]
    if not spans:
        return Metric(
            name="sectionCoverage", value=0.0, score=0.0, ok=False,
            threshold=t.section_coverage, weight=weight,
            detail=(
                f"none of the {len(song.sections)} section(s) carries a "
                f"startTime/endTime, so they span 0.0s of the {duration:.1f}s track"
            ),
            offenders=offenders,
        )

    covered = 0.0
    overlaps = 0.0
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in spans:
        end = max(start, end)
        if start > cursor:
            gaps.append((cursor, start))
        else:
            overlaps += min(cursor - start, end - start)
        covered += max(0.0, end - max(start, cursor))
        cursor = max(cursor, end)
    if cursor < duration:
        gaps.append((cursor, duration))

    for a, b in gaps:
        if b - a > 0.5:  # sub-second seams are rounding, not gaps
            offenders.append({"gap": [round(a, 2), round(b, 2)], "reason": "no section covers this span"})
    if overlaps > 0.5:
        offenders.append({"overlapSeconds": round(overlaps, 2), "reason": "sections overlap in time"})

    coverage = max(0.0, min(1.0, covered / duration))
    return Metric(
        name="sectionCoverage", value=coverage, score=coverage,
        ok=(coverage >= t.section_coverage and overlaps <= 0.5 and not untimed),
        threshold=t.section_coverage, weight=weight,
        detail=(
            f"sections span {coverage:.1%} of the {duration:.1f}s track; "
            f"{len(untimed)}/{len(song.sections)} untimed, "
            f"{sum(1 for a, b in gaps if b - a > 0.5)} gap(s), "
            f"{overlaps:.1f}s of overlap"
        ),
        offenders=offenders,
    )


def _theory_metric(
    song: Song, mir: Optional[MirAnalysis], t: GradeThresholds
) -> tuple[Metric, TheoryReport]:
    key_name = song.metadata.key or (mir.key if mir is not None else None)
    report = theory_validity(_flat_chords(song), key_name)
    detail = (
        f"{report.explained}/{report.total} chord(s) explainable in "
        f"{report.key} (diatonic, borrowed major, or secondary dominant)"
        if report.share is not None
        else (report.unavailable_reason or "not measurable")
    )
    return (
        Metric(
            name="theoryValidity", value=report.share, score=report.share,
            ok=(report.share >= t.theory_validity) if report.share is not None else None,
            threshold=t.theory_validity, weight=0.15, detail=detail,
            offenders=list(report.unexplained),
        ),
        report,
    )


def _wordless_line_indexes(song: Song) -> set[int]:
    """Lines a section explicitly says have no words (intro/solo/outro/...)."""
    out: set[int] = set()
    for s in song.sections:
        if s.kind in _WORDLESS_SECTION_KINDS:
            out.update(range(s.startLineIndex, s.endLineIndex + 1))
    return out


def _lyric_completeness_metric(
    song: Song, candidates: Sequence, t: GradeThresholds
) -> Metric:
    weight = 0.1
    if not song.lines:
        return Metric(
            name="lyricCompleteness", value=None, score=None, ok=None,
            threshold=t.lyric_completeness, weight=weight,
            detail="the document has no lines",
        )
    wordless_ok = _wordless_line_indexes(song)
    offenders: list[dict] = []
    for line in song.lines:
        if not line.lyrics.strip():
            if line.lineIndex in wordless_ok:
                continue  # an instrumental line, declared as such by its section
            offenders.append(
                {
                    "lineIndex": line.lineIndex,
                    "reason": "empty lyrics with no instrumental section to explain them",
                }
            )
            continue
        # Words with no visible provenance: this run HAD sources and none of
        # them contains anything close to this line. The lyric-reference
        # protocol makes this impossible on providers that speak it (see
        # reconcile/lyric_refs.py) — which is exactly why it is worth
        # measuring on the ones that don't.
        if candidates and not any(
            best_matching_line(list(c.lines), line.lyrics) is not None for c in candidates
        ):
            offenders.append(
                {
                    "lineIndex": line.lineIndex,
                    "reason": f"no candidate source contains these words ({len(candidates)} searched)",
                }
            )
    # Overrides are the lyric-reference protocol's escape hatch: the words were
    # WRITTEN by the model instead of referenced into a source. Legitimate, and
    # still a gap in provenance worth counting against completeness.
    overrides = sum(1 for p in song.provenance if p.action == "lyric-override")
    accounted = len(song.lines) - len(offenders) - min(overrides, len(song.lines))
    share = max(0.0, accounted / len(song.lines))
    return Metric(
        name="lyricCompleteness", value=share, score=share,
        ok=share >= t.lyric_completeness, threshold=t.lyric_completeness, weight=weight,
        detail=(
            f"{accounted}/{len(song.lines)} line(s) have words accounted for "
            f"({len(offenders)} unexplained empty, {overrides} model-written override(s))"
        ),
        offenders=offenders,
    )


# --- the grade ---------------------------------------------------------------


def grade_song(
    song: Song,
    mir: Optional[MirAnalysis] = None,
    candidates: Sequence = (),
    *,
    duration: Optional[float] = None,
    thresholds: Optional[GradeThresholds] = None,
) -> Grade:
    """Grade `song` against the audio analysis it was built from.

    Pure and deterministic: the same inputs always produce the same grade, no
    model call and no network. `candidates` are read by the lyric-completeness
    metric (words no source contains have no visible provenance); scoring the
    candidates THEMSELVES against the audio is a different question, answered
    by ``reconcile.match.score_candidate`` and used by
    :mod:`quality.attribution` to decide whose fault a bad grade is.
    """
    t = thresholds or GradeThresholds.from_settings()
    resolved_duration = (
        duration
        if duration is not None
        else (
            (mir.duration_seconds if mir is not None and mir.duration_seconds else None)
            or song.audio.durationSeconds
        )
    )

    theory, _report = _theory_metric(song, mir, t)
    metrics = (
        _chord_match_metric(song, mir, t),
        _timing_coverage_metric(song, resolved_duration, t),
        _interpolation_metric(song, t),
        _collapse_metric(song, t),
        _section_coverage_metric(song, resolved_duration, t),
        theory,
        _lyric_completeness_metric(song, list(candidates), t),
    )

    scored = [m for m in metrics if m.score is not None]
    total_weight = sum(m.weight for m in scored)
    overall = (
        sum(m.score * m.weight for m in scored) / total_weight if total_weight else None
    )

    failing = [m for m in metrics if m.ok is False]
    measured = [m for m in metrics if m.measured]
    # Half of what could be measured failing IS a failing document, however
    # well the other half scored: the weighted overall alone lets a document
    # with perfect chords, key and lyrics but no usable timing at all
    # (coverage 0.00, everything interpolated, a five-line collapse) land above
    # the line, and that document is not one anybody can play along to.
    half_of_measured = max(2, -(-len(measured) // 2))
    if len(scored) < MIN_MEASURED_METRICS or overall is None:
        verdict = "unknown"
    elif overall < t.overall or len(failing) >= half_of_measured:
        verdict = "fail"
    elif failing:
        verdict = "warn"
    else:
        verdict = "pass"

    return Grade(verdict=verdict, overall=overall, metrics=metrics, thresholds=t)


def timing_unreliable_provenance_entry(
    attribution, *, actor: str = "snoocle-server/quality"
) -> ProvenanceEntry:
    """Mark a stored version's timing as not to be trusted (AUDIO fault).

    The document is still worth storing — its chords and words can be right
    while its times cannot be — and a reader has to be able to tell. This is a
    provenance entry rather than a Song field on purpose: it is a statement
    about how this version came to be, which is exactly what provenance is,
    and it needs no schema change to be readable by anything already reading
    a song's history.
    """
    return ProvenanceEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        actor=actor,
        action="timing-unreliable",
        confidence=None,
        notes=(
            "this version's timing is NOT reliable and was not retried: "
            + attribution.reason
        ),
    )


def grade_provenance_entry(
    grade: Grade, *, attribution=None, actor: str = "snoocle-server/quality"
) -> ProvenanceEntry:
    """The provenance entry for a grade — written on EVERY run.

    A grade is valuable as history even when no action follows it: ten
    consecutive runs at match ratio ~0.5 are only visible as
    non-convergence if each of them wrote down what it scored.
    """
    notes = grade.describe()
    if attribution is not None:
        notes += f"; fault: {attribution.describe()}"
    return ProvenanceEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        actor=actor,
        action="quality-grade",
        confidence=round(grade.overall, 3) if grade.overall is not None else None,
        notes=notes,
    )
