"""Complete model-free song processing orchestration.

This module composes the existing acquisition/cache, MIR, discovery,
selection, baseline, alignment, and quality services.  It contains no model
provider or reconciliation imports by design.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .audio.acquire import acquire as acquire_audio
from .deterministic import (
    CandidateSelectionStrategy,
    DeterministicAlignmentResult,
    StageObservation,
    align_song_deterministically_service,
    build_song_from_candidate,
    observe,
    select_candidate_deterministically,
)
from .discovery import CandidateSource, discover_sources
from .discovery.cache import discover_cached
from .identity import IdentityUnresolvedError, resolve_identity_from_evidence
from .mir import MirAnalysis, analyze_audio
from .mir.cache import analyze_cached
from .schema.song import slugify_song_id
from .timing.lrc import LrcLine


@dataclass
class DeterministicProcessResult:
    status: str
    reason: str | None
    song_id: str
    observations: list[StageObservation]
    cache: dict[str, str]
    alignment: DeterministicAlignmentResult | None = None
    selection: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        stages = [item.to_dict() for item in self.observations]
        total_ms = sum(item.elapsed_ms for item in self.observations)
        result: dict[str, Any] = {
            "status": self.status,
            "reason": self.reason,
            "songId": self.song_id,
            "cache": dict(self.cache),
            "stages": stages,
            "totals": {
                "elapsedMs": total_ms,
                "modelCalls": 0,
                "modelCostUSD": 0,
            },
        }
        if self.selection is not None:
            result["selection"] = self.selection
            result["conflicts"] = list(self.selection.get("conflicts") or [])
        if self.alignment is not None:
            aligned = self.alignment.to_dict()
            result.update(
                {
                    "song": aligned["song"],
                    "alignmentReport": aligned["alignmentReport"],
                    "reviewQueue": aligned["reviewQueue"],
                    "quality": aligned["quality"],
                    "conflictPacket": aligned["conflictPacket"],
                }
            )
        return result


def _with_cache_status(observation: StageObservation, status: str) -> StageObservation:
    normalized = status if status in {"hit", "miss"} else (
        "miss" if status in {"refresh", "disabled"} else "not_applicable"
    )
    return dataclasses.replace(observation, cache_status=normalized)


def _completed_status(alignment: DeterministicAlignmentResult) -> tuple[str, str | None]:
    verdict = alignment.quality.grade.verdict
    if verdict == "fail":
        return "needs_review", "quality_gate_failed"
    if verdict == "unknown":
        return "needs_review", "quality_evidence_insufficient"
    if alignment.review_queue:
        return "needs_review", "low_confidence_placements"
    return "completed", None


def process_song_deterministically_service(
    *,
    title: str,
    artist: str,
    audio_path: str | None = None,
    recording_id: str | None = None,
    mir: MirAnalysis | None = None,
    candidates: Sequence[CandidateSource] | None = None,
    lrc_lines: Sequence[LrcLine] | None = None,
    use_lrc: bool = False,
    selection_strategy: CandidateSelectionStrategy = "strict",
    max_candidates: int = 8,
    mir_accuracy: str = "standard",
    refresh_mir_cache: bool = False,
    refresh_discovery_cache: bool = False,
) -> DeterministicProcessResult:
    """Process one song through the complete deterministic service chain."""
    observations: list[StageObservation] = []
    cache = {"audio": "not_applicable", "mir": "not_applicable", "discovery": "not_applicable"}

    try:
        identity, observation = observe(
            "identity",
            lambda: resolve_identity_from_evidence(artist=artist, title=title),
            input_summary={
                "titlePresent": bool(title.strip()),
                "artistPresent": bool(artist.strip()),
            },
            summarize=lambda value: {
                "method": value.method,
                "confidence": value.confidence,
            },
        )
    except IdentityUnresolvedError as error:
        observation = StageObservation(
            name="identity",
            elapsed_ms=0,
            input_summary={
                "titlePresent": bool(title.strip()),
                "artistPresent": bool(artist.strip()),
            },
            output_summary={"status": "needs_review", "missing": error.missing},
        )
        selection = {
            "status": "needs_review",
            "reason": "identity_unresolved",
            "selectedSourceId": None,
            "ranked": [],
            "conflicts": [
                {
                    "type": "identity",
                    "missing": error.missing,
                    "evidenceTried": error.evidence_tried,
                }
            ],
        }
        return DeterministicProcessResult(
            status="needs_review",
            reason="identity_unresolved",
            song_id="",
            observations=[observation],
            cache=cache,
            selection=selection,
        )
    observations.append(observation)
    song_id = slugify_song_id(identity.artist, identity.title)

    resolved_path: str | None = None
    youtube_video_id: str | None = recording_id
    if audio_path is not None:
        path = Path(audio_path)
        if not path.is_file():
            raise ValueError("audio_path does not name an existing file")
        resolved_path = str(path)
        observations.append(
            StageObservation(
                name="acquire_audio",
                elapsed_ms=0,
                input_summary={"callerAudioPath": True},
                output_summary={"source": "caller_path", "recordingId": recording_id},
            )
        )
    elif mir is not None:
        observations.append(
            StageObservation(
                name="acquire_audio",
                elapsed_ms=0,
                input_summary={"required": False},
                output_summary={"source": "caller_mir"},
            )
        )
    else:
        acquired, observation = observe(
            "acquire_audio",
            lambda: acquire_audio(
                title=identity.title,
                artist=identity.artist,
                video_url_or_id=recording_id,
            ),
            input_summary={"recordingIdPresent": recording_id is not None},
            summarize=lambda value: {
                "recordingId": value.video_id,
                "durationSeconds": value.duration_seconds,
            },
        )
        cache["audio"] = "hit" if acquired.from_cache else "miss"
        observation = _with_cache_status(observation, cache["audio"])
        observations.append(observation)
        resolved_path = acquired.path
        youtube_video_id = acquired.video_id

    if mir is not None:
        resolved_mir = mir
        observations.append(
            StageObservation(
                name="mir",
                elapsed_ms=0,
                input_summary={"callerMir": True},
                output_summary={
                    "durationSeconds": mir.duration_seconds,
                    "beats": len(mir.beats),
                    "chords": len(mir.chords),
                },
            )
        )
    else:
        if resolved_path is None:
            raise ValueError("audio or MIR evidence is required")
        (resolved_mir, mir_info), observation = observe(
            "mir",
            lambda: analyze_cached(
                resolved_path,
                accuracy=mir_accuracy,
                compute=lambda: analyze_audio(resolved_path, accuracy=mir_accuracy),
                refresh=refresh_mir_cache,
            ),
            input_summary={"accuracy": mir_accuracy, "refresh": refresh_mir_cache},
            summarize=lambda value: {
                "durationSeconds": value[0].duration_seconds,
                "beats": len(value[0].beats),
                "chords": len(value[0].chords),
            },
        )
        cache["mir"] = mir_info.status
        observations.append(_with_cache_status(observation, mir_info.status))

    if candidates is not None:
        resolved_candidates = list(candidates)
        observations.append(
            StageObservation(
                name="discovery",
                elapsed_ms=0,
                input_summary={"callerCandidates": True},
                output_summary={"candidates": len(resolved_candidates)},
            )
        )
    else:
        (resolved_candidates, discovery_info), observation = observe(
            "discovery",
            lambda: discover_cached(
                identity.title,
                identity.artist,
                max_candidates=max_candidates,
                discover=lambda: discover_sources(
                    identity.title,
                    identity.artist,
                    max_candidates=max_candidates,
                ),
                refresh=refresh_discovery_cache,
            ),
            input_summary={"maxCandidates": max_candidates, "refresh": refresh_discovery_cache},
            summarize=lambda value: {"candidates": len(value[0])},
        )
        cache["discovery"] = discovery_info.status
        observations.append(_with_cache_status(observation, discovery_info.status))

    selection, observation = observe(
        "candidate_selection",
        lambda: select_candidate_deterministically(
            resolved_candidates,
            resolved_mir,
            strategy=selection_strategy,
        ),
        input_summary={
            "candidates": len(resolved_candidates),
            "strategy": selection_strategy,
        },
        summarize=lambda value: {
            "status": value.status,
            "selectedSourceId": value.selected.source_id if value.selected else None,
            "conflicts": len(value.conflicts),
        },
    )
    observations.append(observation)
    selection_payload = selection.to_dict()
    if selection.status != "selected" or selection.selected is None:
        return DeterministicProcessResult(
            status="needs_review",
            reason=selection.reason or "candidate_selection_unresolved",
            song_id=song_id,
            observations=observations,
            cache=cache,
            selection=selection_payload,
        )

    baseline, observation = observe(
        "baseline",
        lambda: build_song_from_candidate(
            selection.selected.candidate,
            song_id=song_id,
            title=identity.title,
            artist=identity.artist,
            youtube_video_id=youtube_video_id,
        ),
        input_summary={"sourceId": selection.selected.source_id},
        summarize=lambda value: {
            "lines": len(value.lines),
            "placements": sum(len(line.chordPlacements) for line in value.lines),
        },
    )
    observations.append(observation)

    alignment = align_song_deterministically_service(
        baseline,
        resolved_mir,
        candidates=resolved_candidates,
        lrc_lines=lrc_lines,
        use_lrc=use_lrc,
    )
    observations.extend(alignment.observations)
    alignment.observations = observations
    status, reason = _completed_status(alignment)
    return DeterministicProcessResult(
        status=status,
        reason=reason,
        song_id=song_id,
        observations=observations,
        cache=cache,
        alignment=alignment,
        selection=selection_payload,
    )
