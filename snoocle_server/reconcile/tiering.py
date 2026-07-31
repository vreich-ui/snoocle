"""Deterministic effort-to-model routing for the in-process Anthropic agent."""

from __future__ import annotations

from typing import Sequence

from ..discovery.models import CandidateSource
from ..mir.base import MirAnalysis
from .match import score_candidates


def normalize_effort_level(value: str | None) -> str:
    """Normalize old internal names and public aliases to low|standard|high."""
    lowered = (value or "standard").lower()
    if lowered in {"low", "fast"}:
        return "low"
    if lowered in {"high", "thorough"}:
        return "high"
    return "standard"


def anthropic_effort(level: str) -> str:
    """Anthropic's API calls the public standard level ``medium``."""
    return {"low": "low", "standard": "medium", "high": "high"}[
        normalize_effort_level(level)
    ]


def unresolved_sheet_conflict(
    candidates: Sequence[CandidateSource],
    mir: MirAnalysis | None,
    *,
    thresholds=None,
) -> str | None:
    """Explain a real source conflict only when MIR cannot select a winner.

    Sheet disagreement uses the same transposition-invariant agreement ratio
    as quality attribution.  MIR breaks the tie only when one sheet both clears
    the established MIR-agreement floor and leads the runner-up by the existing
    model-margin threshold.  No model judgment enters this decision.
    """
    # Lazy by design: quality.attribution imports reconcile.match, while the
    # provider registry imports this module. Importing it at module load would
    # close that package cycle during API startup.
    from ..quality.attribution import AttributionThresholds, candidate_agreement

    t = thresholds or AttributionThresholds.from_settings()
    agreement = candidate_agreement(candidates)
    if agreement is None or agreement >= t.source_agreement:
        return None

    scores = [score for score in score_candidates(candidates, mir) if score.total > 0]
    ranked = sorted(scores, key=lambda score: score.score, reverse=True)
    if mir is not None and ranked:
        best = ranked[0]
        runner_up = ranked[1].score if len(ranked) > 1 else 0.0
        if best.score >= t.mir_agreement and best.score - runner_up >= t.model_margin:
            return None

    score_text = ", ".join(
        f"{score.source_id or 'candidate'}={score.matched}/{score.total} "
        f"matches, {len(score.conflicts)} conflicts ({score.score:.0%})"
        for score in ranked
    ) or "no sheet could be scored against MIR"
    mir_text = "MIR unavailable" if mir is None else f"MIR tie unresolved: {score_text}"
    return (
        f"candidate sheets disagree (agreement {agreement:.0%} < "
        f"{t.source_agreement:.0%}) and {mir_text}"
    )
