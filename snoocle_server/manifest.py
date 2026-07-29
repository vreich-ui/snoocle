"""Evidence manifest — what the reconciler is being handed, and where it came from.

The pipeline gathers evidence and passes it along with no statement about its
state. An agent handed a MIR block and three sources cannot tell "this evidence
is good and current" from "gathering partly failed and I should go look
harder" — and the difference changes what it should do.

This repo already learned that lesson narrowly: ``_NOTES_ONLY_BANNER`` in
``reconcile/prompt.py`` exists because a model shown two empty evidence
sections reconstructs the song from memory rather than reading the absence as
deliberate. The manifest generalizes the fix — state the evidence's provenance
and quality explicitly instead of leaving the model to infer it from what's
present.

**The manifest is DESCRIPTIVE.** It is never a source of truth, is never parsed
back, and nothing downstream branches on it. Delete it and every output stays
correct — just arrived at less efficiently. That constraint is what lets it be
generous: it can say "this MIR came from cache and the audio was actually
listened to three weeks ago" without any risk that a bug in the manifest
corrupts a song.

On ordering: LRC alignment runs AFTER reconciliation (pipeline step 5c), so the
copy injected into the prompt honestly reports it as ``"pending"``. The
pipeline then fills the real outcome in for the copy returned to the caller and
recorded on the run trace, which is where "what did this run reuse vs
recompute?" actually gets asked.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

log = logging.getLogger(__name__)

#: Provenance actors the server writes itself. An entry by anyone else is a
#: human's edit — the only signal available for `priorSong.humanEdited`, since
#: a caller-supplied prior song carries no separate "edited" flag.
_MACHINE_ACTOR_PREFIXES = ("snoocle-server/", "reconcile:")


def _age_days(iso_timestamp: str | None) -> Optional[float]:
    if not iso_timestamp:
        return None
    try:
        ts = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - ts
    return round(delta.total_seconds() / 86400.0, 1)


def _coverage(mir) -> str:
    """How much of the track the analysis actually covered, in words."""
    windows = getattr(mir, "analyzed_windows", None) or []
    if len(windows) == 1 and windows[0].start <= 0.01 and (
        mir.duration_seconds and windows[0].end >= mir.duration_seconds - 0.01
    ):
        return "full-track"
    if not windows:
        return "unknown"
    spans = ", ".join(f"{w.start:.0f}-{w.end:.0f}s" for w in windows)
    return f"sampled: {spans}"


def _mir_block(mir, cache_info) -> dict:
    if mir is None:
        return {"status": "absent"}
    block: dict[str, Any] = {
        "status": f"cache-{cache_info.status}" if cache_info is not None else "fresh",
        "engines": dict(mir.engines),
        "bpm": mir.bpm,
        "key": mir.key,
        "coverage": _coverage(mir),
    }
    if cache_info is not None:
        block["analyzedAt"] = cache_info.analyzed_at
        age = _age_days(cache_info.analyzed_at)
        if age is not None:
            block["ageDays"] = age
    return block


def _sources_block(candidates, cache_info) -> dict:
    block: dict[str, Any] = {
        "status": f"cache-{cache_info.status}" if cache_info is not None else "absent",
        "count": len(candidates or []),
    }
    if cache_info is not None:
        block["gatheredAt"] = cache_info.gathered_at
        age = _age_days(cache_info.gathered_at)
        if age is not None:
            block["ageDays"] = age
    # NOTE: no `scores` field. Per-candidate scoring (reconcile.match.
    # score_candidate) does not exist on this branch, and inventing a scoring
    # function here would make the manifest a source of truth about quality
    # rather than a description of provenance. When that module lands, add the
    # field here — do not synthesize a stand-in.
    return block


def _prior_song_block(prior_song: dict | None) -> dict:
    if not prior_song:
        return {"exists": False}
    block: dict[str, Any] = {"exists": True}

    try:
        from .schema import Song
        from .store.base import version_sha

        block["version"] = version_sha(Song.model_validate(prior_song))
    except Exception:  # noqa: BLE001 — a manifest must never fail a run
        pass

    provenance = prior_song.get("provenance") or []
    # `matchRatio` is NOT a new metric invented here: it is the prior song's own
    # recorded timing-snap ratio — the fraction of its chord placements that
    # were matched to the MIR timeline when it was built. It says "how
    # audio-grounded is the document you are being asked to correct", which is
    # exactly the context that makes a prior song trustworthy or not. A prior
    # built by a run that reused an earlier audio analysis records the same
    # ratio under `timing-carry-forward`, so both count — otherwise a song
    # re-analyzed with listen=off would report no timing grounding at all.
    snaps = [
        p for p in provenance
        if p.get("action") in ("timing-snap", "timing-carry-forward")
    ]
    if snaps and snaps[-1].get("confidence") is not None:
        try:
            block["matchRatio"] = round(float(snaps[-1]["confidence"]), 3)
        except (TypeError, ValueError):
            pass

    actors = [str(p.get("actor") or "") for p in provenance]
    block["humanEdited"] = any(
        actor and not actor.startswith(_MACHINE_ACTOR_PREFIXES) for actor in actors
    )
    return block


def build_evidence_manifest(
    *,
    mir=None,
    mir_cache=None,
    candidates=None,
    discovery_cache=None,
    prior_song: dict | None = None,
    scope=None,
    guidance: str | None = None,
    guidance_origin: str | None = None,
    lrc: dict | None = None,
) -> dict:
    """Describe the state of every input this run is handing the reconciler.

    Every argument is optional and every failure mode degrades to a less
    informative manifest rather than an exception — see the module docstring
    for why that is the right trade for something purely descriptive.
    """
    try:
        request: dict[str, Any] = {
            "scope": (
                {"listen": scope.listen, "reconcile": scope.reconcile}
                if scope is not None
                else None
            ),
            "notes": guidance or None,
        }
        if guidance and guidance_origin:
            request["notesOrigin"] = guidance_origin
        return {
            "mir": _mir_block(mir, mir_cache),
            "sources": _sources_block(candidates, discovery_cache),
            "priorSong": _prior_song_block(prior_song),
            # Filled in after step 5c; "pending" is the honest value at the
            # moment the reconciler sees this.
            "lrcAlign": lrc if lrc is not None else {"status": "pending"},
            "request": request,
        }
    except Exception as e:  # noqa: BLE001 — descriptive only; never fail a run
        log.warning("evidence manifest assembly failed: %s", e)
        return {"error": "manifest unavailable"}


def lrc_block(status: str, lines_matched: int = 0, lines_total: int = 0) -> dict:
    """The ``lrcAlign`` block, built after alignment has actually run."""
    block: dict[str, Any] = {"status": status}
    if status in ("hit", "miss"):
        block["linesMatched"] = lines_matched
        block["linesTotal"] = lines_total
    return block
