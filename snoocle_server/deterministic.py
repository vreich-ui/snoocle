"""Deterministic-first song construction and alignment services.

This module is the model-free orchestration boundary.  It deliberately wraps
the algorithms that already live in discovery, MIR, timing, quality, and
reconciliation modules instead of reimplementing them.  MCP, HTTP, tests, and
the full pipeline can therefore call the same deterministic units directly.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Sequence, TypeVar

from . import __version__
from .discovery import CandidateSource
from .mir import MirAnalysis
from .quality import QualityDecision, evaluate as evaluate_quality
from .quality.grader import grade_provenance_entry
from .reconcile.match import score_candidates
from .reconcile.patch_ops import apply_patch, describe_op_set
from .schema.song import (
    AudioInfo,
    DisplayPreferences,
    ProvenanceEntry,
    Section,
    Song,
    SongMetadata,
)
from .timing.collapse_guard import find_collapsed_runs, guard_against_collapsed_timing
from .timing.confidence import build_review_queue, score_song
from .timing.lrc import LrcLine, apply_lrc, fetch_lrc, match_lrc_to_lines
from .timing.realign import retime_sections
from .timing.root_match import match_chord_roots, sounding_segments
from .timing.snap import INTERPOLATED_CONFIDENCE, snap_chords


AgentPolicy = Literal["never", "unresolved_only", "always"]
CandidateSelectionStrategy = Literal["best", "strict"]

MAX_CANDIDATES = 20
MAX_LINES = 2_000
MAX_LRC_LINES = 5_000
MAX_JSON_BYTES = 5_000_000
SELECTION_CONFLICT_MARGIN = 0.03

_STABLE_TIMESTAMP = "1970-01-01T00:00:00+00:00"

_KIND_WORDS = {
    "intro": "intro",
    "verse": "verse",
    "pre-chorus": "prechorus",
    "prechorus": "prechorus",
    "chorus": "chorus",
    "post-chorus": "postchorus",
    "postchorus": "postchorus",
    "bridge": "bridge",
    "solo": "solo",
    "instrumental": "instrumental",
    "interlude": "interlude",
    "breakdown": "breakdown",
    "outro": "outro",
    "ending": "outro",
    "coda": "outro",
    "hook": "chorus",
    "refrain": "chorus",
}


class DeterministicPipelineError(ValueError):
    """A deterministic input cannot safely produce the requested result."""


@dataclass(frozen=True)
class StageObservation:
    name: str
    elapsed_ms: int
    cache_status: str = "not_applicable"
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "elapsedMs": self.elapsed_ms,
            "cacheStatus": self.cache_status,
            "modelCalls": 0,
            "modelCostUSD": 0,
            "inputSummary": self.input_summary,
            "outputSummary": self.output_summary,
            "warnings": list(self.warnings),
        }


T = TypeVar("T")


def observe(
    name: str,
    fn: Callable[[], T],
    *,
    cache_status: str = "not_applicable",
    input_summary: Optional[dict[str, Any]] = None,
    summarize: Optional[Callable[[T], dict[str, Any]]] = None,
) -> tuple[T, StageObservation]:
    start = time.perf_counter()
    value = fn()
    elapsed_ms = max(0, round((time.perf_counter() - start) * 1000))
    return value, StageObservation(
        name=name,
        elapsed_ms=elapsed_ms,
        cache_status=_cache_status(cache_status),
        input_summary=input_summary or {},
        output_summary=summarize(value) if summarize is not None else {},
    )


def _cache_status(value: str) -> str:
    if value == "hit":
        return "hit"
    if value in {"miss", "refresh", "disabled"}:
        return "miss"
    return "not_applicable"


def _kind_for(name: str) -> str:
    low = name.casefold()
    for word, kind in _KIND_WORDS.items():
        if word in low:
            return kind
    return "other"


def _stable_timestamp(candidate: CandidateSource) -> str:
    # A source timestamp is part of the input and keeps repeated construction
    # byte-identical.  Legacy/manual candidates without one use a fixed epoch
    # rather than making deterministic output depend on wall-clock time.
    return candidate.retrievedAt or _STABLE_TIMESTAMP


def _bounded_candidates(candidates: Sequence[CandidateSource]) -> list[CandidateSource]:
    resolved = list(candidates)
    if len(resolved) > MAX_CANDIDATES:
        raise DeterministicPipelineError(
            f"candidate count {len(resolved)} exceeds the limit of {MAX_CANDIDATES}"
        )
    for candidate in resolved:
        if len(candidate.lines) > MAX_LINES:
            raise DeterministicPipelineError(
                f"candidate {candidate.sourceId!r} has {len(candidate.lines)} lines; "
                f"limit is {MAX_LINES}"
            )
    return resolved


def _candidate_signature(candidate: CandidateSource) -> str:
    content = [
        {
            "lyrics": line.lyrics,
            "placements": [(p.charIndex, p.chord) for p in line.chordPlacements],
        }
        for line in candidate.lines
    ]
    canonical = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RankedCandidate:
    source_id: str
    deterministic_score: float
    source_confidence: float
    parse_confidence: float
    mir_score: float | None
    transposition: int
    matched: int
    total: int
    coverage: str | None
    signature: str
    candidate: CandidateSource = field(compare=False, repr=False)

    def to_dict(self, *, include_candidate: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sourceId": self.source_id,
            "score": round(self.deterministic_score, 6),
            "sourceConfidence": round(self.source_confidence, 6),
            "parseConfidence": round(self.parse_confidence, 6),
            "mirScore": round(self.mir_score, 6) if self.mir_score is not None else None,
            "transposition": self.transposition,
            "matched": self.matched,
            "total": self.total,
            "coverage": self.coverage,
        }
        if include_candidate:
            result["candidate"] = self.candidate.model_dump(mode="json")
        return result


def rank_candidates_deterministically(
    candidates: Sequence[CandidateSource], mir: Optional[MirAnalysis] = None
) -> list[RankedCandidate]:
    """Rank parsed sources with stable, explicit evidence weights.

    MIR agreement dominates when it is measurable.  Retrieval/parse confidence
    and full-song coverage break ties.  Input order never decides a tie:
    ``sourceId`` is the final stable key.
    """
    resolved = _bounded_candidates(candidates)
    mir_scores = score_candidates(resolved, mir)
    ranked: list[RankedCandidate] = []
    for candidate, measured in zip(resolved, mir_scores):
        parse_confidence = (
            candidate.parseConfidence
            if candidate.parseConfidence is not None
            else candidate.confidence
        )
        coverage = 1.0 if candidate.coverage == "full-song" else 0.0
        if mir is not None and measured.total > 0 and sounding_segments(mir):
            score = (
                0.60 * measured.score
                + 0.20 * parse_confidence
                + 0.15 * candidate.confidence
                + 0.05 * coverage
            )
            mir_score: float | None = measured.score
        else:
            score = 0.50 * parse_confidence + 0.40 * candidate.confidence + 0.10 * coverage
            mir_score = None
        ranked.append(
            RankedCandidate(
                source_id=candidate.sourceId,
                deterministic_score=score,
                source_confidence=candidate.confidence,
                parse_confidence=parse_confidence,
                mir_score=mir_score,
                transposition=measured.transposition,
                matched=measured.matched,
                total=measured.total,
                coverage=candidate.coverage,
                signature=_candidate_signature(candidate),
                candidate=candidate,
            )
        )
    return sorted(ranked, key=lambda item: (-item.deterministic_score, item.source_id))


@dataclass(frozen=True)
class CandidateSelection:
    status: str
    selected: Optional[RankedCandidate]
    ranked: tuple[RankedCandidate, ...]
    reason: Optional[str] = None
    conflicts: tuple[dict[str, Any], ...] = ()

    def to_dict(self, *, include_candidates: bool = False) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "selectedSourceId": self.selected.source_id if self.selected else None,
            "ranked": [
                item.to_dict(include_candidate=include_candidates) for item in self.ranked
            ],
            "conflicts": list(self.conflicts),
        }


def select_candidate_deterministically(
    candidates: Sequence[CandidateSource],
    mir: Optional[MirAnalysis] = None,
    *,
    strategy: CandidateSelectionStrategy = "strict",
    conflict_margin: float = SELECTION_CONFLICT_MARGIN,
) -> CandidateSelection:
    if strategy not in {"best", "strict"}:
        raise DeterministicPipelineError(f"unknown candidate selection strategy: {strategy}")
    ranked = tuple(rank_candidates_deterministically(candidates, mir))
    if not ranked:
        return CandidateSelection(
            status="needs_review",
            selected=None,
            ranked=(),
            reason="no_candidate_sources",
        )
    top = ranked[0]
    if not top.candidate.lines or not any(line.chordPlacements for line in top.candidate.lines):
        return CandidateSelection(
            status="needs_review",
            selected=None,
            ranked=ranked,
            reason="candidate_cannot_form_baseline",
        )
    if strategy == "strict" and len(ranked) > 1:
        second = ranked[1]
        close = top.deterministic_score - second.deterministic_score <= conflict_margin
        different = top.signature != second.signature
        if close and different:
            return CandidateSelection(
                status="needs_review",
                selected=None,
                ranked=ranked,
                reason="candidate_sources_conflict",
                conflicts=(
                    {
                        "type": "candidate_selection",
                        "sourceIds": [top.source_id, second.source_id],
                        "scores": [
                            round(top.deterministic_score, 6),
                            round(second.deterministic_score, 6),
                        ],
                        "scoreDelta": round(
                            top.deterministic_score - second.deterministic_score, 6
                        ),
                    },
                ),
            )
    return CandidateSelection(status="selected", selected=top, ranked=ranked)


def build_song_from_candidate(
    candidate: CandidateSource,
    *,
    song_id: str,
    title: str,
    artist: str,
    youtube_video_id: str | None = None,
) -> Song:
    """Convert one parsed candidate into a schema-valid, untimed Song.

    Lyrics, chord symbols, placements, and character indexes are copied from
    the parsed source exactly.  No timing is inferred and no model is called.
    """
    _bounded_candidates([candidate])
    if not candidate.lines:
        raise DeterministicPipelineError(
            f"candidate {candidate.sourceId!r} has no parsed lines"
        )
    if not any(line.chordPlacements for line in candidate.lines):
        raise DeterministicPipelineError(
            f"candidate {candidate.sourceId!r} has no chord placements"
        )

    lines = []
    for index, source_line in enumerate(candidate.lines):
        # Remove any accidental source timing while preserving lyric strings,
        # placement order, char indexes, chord identities, and display hints.
        placements = [
            placement.model_copy(
                update={"timeSeconds": None, "confidence": None, "beat": None}
            )
            for placement in source_line.chordPlacements
        ]
        lines.append(
            source_line.model_copy(
                update={
                    "lineIndex": index,
                    "chordPlacements": placements,
                    "timeSeconds": None,
                    "confidence": None,
                }
            )
        )

    starts: list[tuple[str, int]] = []
    seen_starts: set[int] = set()
    for hint in sorted(candidate.sectionStarts, key=lambda value: value.startLineIndex):
        if hint.startLineIndex >= len(lines) or hint.startLineIndex in seen_starts:
            continue
        starts.append((hint.name, hint.startLineIndex))
        seen_starts.add(hint.startLineIndex)
    if not starts:
        starts = [((candidate.sectionsHint or ["Song"])[0], 0)]
    elif starts[0][1] > 0:
        starts.insert(0, ("Intro", 0))

    sections: list[Section] = []
    for position, (name, start) in enumerate(starts):
        end = starts[position + 1][1] - 1 if position + 1 < len(starts) else len(lines) - 1
        if end < start:
            continue
        sections.append(
            Section(
                sectionIndex=len(sections),
                name=name,
                kind=_kind_for(name),
                startLineIndex=start,
                endLineIndex=end,
            )
        )

    source_refs = [value for value in (candidate.url, candidate.sourceId) if value]
    song = Song(
        id=song_id,
        metadata=SongMetadata(
            title=title,
            artist=artist,
            key=candidate.declaredKey,
        ),
        displayPreferences=DisplayPreferences(capo=0, tuning="standard"),
        audio=AudioInfo(youtubeVideoId=youtube_video_id),
        sections=sections,
        lines=lines,
        provenance=[
            ProvenanceEntry(
                timestamp=_stable_timestamp(candidate),
                actor=f"snoocle-server/{__version__}",
                action="deterministic-baseline",
                sources=source_refs,
                confidence=candidate.confidence,
                notes=(
                    f"constructed deterministically from {candidate.sourceId}; "
                    "source text and sounding-pitch chord placements preserved"
                ),
            )
        ],
    )
    return Song.model_validate(song.model_dump(mode="json"))


def _new_provenance_stable(song: Song, original_count: int) -> Song:
    entries = list(song.provenance)
    for index in range(original_count, len(entries)):
        entries[index] = entries[index].model_copy(update={"timestamp": _STABLE_TIMESTAMP})
    return Song.model_validate(song.model_copy(update={"provenance": entries}).model_dump())


def _flat_chords(song: Song) -> list[str]:
    return [placement.chord for line in song.lines for placement in line.chordPlacements]


def _matched_counts(song: Song, mir: Optional[MirAnalysis]) -> tuple[int, int]:
    symbols = _flat_chords(song)
    if mir is None:
        return 0, len(symbols)
    return len(match_chord_roots(symbols, sounding_segments(mir))), len(symbols)


def _collapse_count(song: Song) -> int:
    total = len(find_collapsed_runs([line.timeSeconds for line in song.lines]))
    for line in song.lines:
        total += len(
            find_collapsed_runs([placement.timeSeconds for placement in line.chordPlacements])
        )
    return total


def _alignment_summary(song: Song) -> dict[str, Any]:
    placements = [p for line in song.lines for p in line.chordPlacements]
    return {
        "lines": len(song.lines),
        "placements": len(placements),
        "timedLines": sum(1 for line in song.lines if line.timeSeconds is not None),
        "timedPlacements": sum(1 for p in placements if p.timeSeconds is not None),
        "timedSections": sum(
            1 for section in song.sections
            if section.startTime is not None and section.endTime is not None
        ),
    }


def build_conflict_packet(
    song: Song,
    mir: Optional[MirAnalysis],
    candidates: Sequence[CandidateSource],
    review_queue: Sequence[dict[str, Any]],
    *,
    document_version: Optional[str] = None,
) -> dict[str, Any]:
    """Create the compact, lyric-free packet an agent is allowed to see."""
    conflicts: list[dict[str, Any]] = []
    for ordinal, item in enumerate(review_queue):
        line_index = int(item["lineIndex"])
        char_index = int(item["charIndex"])
        evidence: list[str] = []
        for candidate in candidates:
            if line_index >= len(candidate.lines):
                continue
            for placement in candidate.lines[line_index].chordPlacements:
                if placement.charIndex == char_index and placement.chord not in evidence:
                    evidence.append(placement.chord)
        mir_evidence: list[dict[str, Any]] = []
        placement_time = None
        if line_index < len(song.lines):
            placement_time = next(
                (
                    p.timeSeconds
                    for p in song.lines[line_index].chordPlacements
                    if p.charIndex == char_index
                ),
                None,
            )
        if mir is not None and placement_time is not None:
            for segment in mir.chords:
                if segment.start <= placement_time <= segment.end:
                    mir_evidence.append(
                        {
                            "start": round(segment.start, 3),
                            "end": round(segment.end, 3),
                            "chord": segment.chord,
                        }
                    )
                    break
        conflicts.append(
            {
                "type": "chord_identity",
                "lineId": f"L{line_index}",
                "placementId": f"P{ordinal}",
                "lineIndex": line_index,
                "charIndex": char_index,
                "existingChord": item["chord"],
                "candidateEvidence": evidence,
                "mirEvidence": mir_evidence,
                "confidence": item.get("confidence"),
                "reasons": list(item.get("reasons") or []),
            }
        )
    return {
        "songId": song.id,
        "recordingId": song.audio.analyzedVideoId or song.audio.youtubeVideoId,
        "documentVersion": document_version,
        "musicalContext": {
            "key": song.metadata.key or (mir.key if mir else None),
            "bpm": song.metadata.bpm or (mir.bpm if mir else None),
            "timeSignature": song.metadata.timeSignature or (mir.time_signature if mir else None),
        },
        "conflicts": conflicts,
        "allowedOperations": describe_op_set().split(", "),
    }


@dataclass
class DeterministicAlignmentResult:
    song: Song
    report: dict[str, Any]
    observations: list[StageObservation]
    quality: QualityDecision
    review_queue: list[dict[str, Any]]
    conflict_packet: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        total_ms = sum(item.elapsed_ms for item in self.observations)
        return {
            "song": self.song.model_dump(mode="json"),
            "alignmentReport": self.report,
            "reviewQueue": self.review_queue,
            "quality": self.quality.to_dict(),
            "conflictPacket": self.conflict_packet,
            "stages": [item.to_dict() for item in self.observations],
            "totals": {
                "elapsedMs": total_ms,
                "modelCalls": 0,
                "modelCostUSD": 0,
            },
        }


def align_song_deterministically_service(
    song: Song,
    mir: Optional[MirAnalysis],
    *,
    candidates: Sequence[CandidateSource] = (),
    lrc_lines: Optional[Sequence[LrcLine]] = None,
    use_lrc: bool = False,
    document_version: Optional[str] = None,
) -> DeterministicAlignmentResult:
    """Run the complete model-free timing and quality sequence."""
    resolved_candidates = _bounded_candidates(candidates)
    observations: list[StageObservation] = []
    original_provenance = len(song.provenance)
    collapsed_before = _collapse_count(song)

    current, observation = observe(
        "snap_chords",
        lambda: snap_chords(song, mir),
        input_summary={
            "lines": len(song.lines),
            "placements": len(_flat_chords(song)),
            "mirPresent": mir is not None,
        },
        summarize=_alignment_summary,
    )
    observations.append(observation)

    lrc_match_count = 0
    if use_lrc:
        resolved_lrc = list(lrc_lines) if lrc_lines is not None else (
            fetch_lrc(
                current.metadata.title,
                current.metadata.artist,
                mir.duration_seconds if mir else current.audio.durationSeconds,
            )
            or []
        )
        if len(resolved_lrc) > MAX_LRC_LINES:
            raise DeterministicPipelineError(
                f"LRC line count {len(resolved_lrc)} exceeds the limit of {MAX_LRC_LINES}"
            )

        def apply_lrc_stage() -> Song:
            nonlocal lrc_match_count
            matches = match_lrc_to_lines(resolved_lrc, current)
            lrc_match_count = len(matches)
            return apply_lrc(current, mir, matches)

        current, observation = observe(
            "lrc_alignment",
            apply_lrc_stage,
            input_summary={"enabled": True, "lrcLines": len(resolved_lrc)},
            summarize=_alignment_summary,
        )
    else:
        observation = StageObservation(
            name="lrc_alignment",
            elapsed_ms=0,
            input_summary={"enabled": False},
            output_summary={"matchedLines": 0},
        )
    observations.append(observation)

    duration = (mir.duration_seconds if mir is not None else None) or current.audio.durationSeconds
    current, observation = observe(
        "section_timing",
        lambda: retime_sections(current, duration)[0],
        input_summary={"sections": len(current.sections)},
        summarize=_alignment_summary,
    )
    observations.append(observation)

    current, observation = observe(
        "collapse_guard",
        lambda: guard_against_collapsed_timing(current, duration)[0],
        input_summary={"collapsedRunsBefore": collapsed_before},
        summarize=_alignment_summary,
    )
    observations.append(observation)

    placement_scores = []

    def confidence_stage() -> Song:
        nonlocal placement_scores
        scored, placement_scores = score_song(current, resolved_candidates, mir)
        return scored

    current, observation = observe(
        "confidence_scoring",
        confidence_stage,
        input_summary={"candidates": len(resolved_candidates)},
        summarize=_alignment_summary,
    )
    observations.append(observation)
    review_queue = build_review_queue(placement_scores)

    quality, observation = observe(
        "quality_grading",
        lambda: evaluate_quality(
            current,
            mir,
            resolved_candidates,
            can_search=False,
            can_retry=False,
            sources_expected=bool(resolved_candidates),
        ),
        input_summary={"sourcesExpected": bool(resolved_candidates)},
        summarize=lambda decision: {
            "verdict": decision.grade.verdict,
            "overall": decision.grade.overall,
            "fault": decision.attribution.fault.value,
        },
    )
    observations.append(observation)

    current = current.model_copy(
        update={
            "provenance": list(current.provenance)
            + [grade_provenance_entry(quality.grade, attribution=quality.attribution)]
        }
    )
    current = _new_provenance_stable(current, original_provenance)
    matched, total = _matched_counts(current, mir)
    placements = [p for line in current.lines for p in line.chordPlacements]
    line_coverage = (
        sum(1 for line in current.lines if line.timeSeconds is not None) / len(current.lines)
        if current.lines
        else 0.0
    )
    section_coverage = (
        sum(
            1 for section in current.sections
            if section.startTime is not None and section.endTime is not None
        )
        / len(current.sections)
        if current.sections
        else 0.0
    )
    interpolation_share = (
        sum(1 for p in placements if p.confidence == INTERPOLATED_CONFIDENCE) / len(placements)
        if placements
        else 0.0
    )
    collapsed_after = _collapse_count(current)
    report = {
        "matchedChordCount": matched,
        "unmatchedChordCount": max(0, total - matched),
        "lineTimingCoverage": round(line_coverage, 6),
        "sectionCoverage": round(section_coverage, 6),
        "interpolationShare": round(interpolation_share, 6),
        "collapsedTimingInterventions": max(0, collapsed_before - collapsed_after),
        "collapsedRunsRemaining": collapsed_after,
        "lrcMatchedLines": lrc_match_count,
        "qualityVerdict": quality.grade.verdict,
        "faultAttribution": quality.attribution.fault.value,
        "modelCalls": 0,
        "modelCostUSD": 0,
    }
    packet = build_conflict_packet(
        current,
        mir,
        resolved_candidates,
        review_queue,
        document_version=document_version,
    )
    return DeterministicAlignmentResult(
        song=current,
        report=report,
        observations=observations,
        quality=quality,
        review_queue=review_queue,
        conflict_packet=packet,
    )


def apply_deterministic_patch(song: Song, ops: Sequence[dict[str, Any]]) -> dict[str, Any]:
    patched, applied = apply_patch(song, list(ops))
    return {
        "song": patched,
        "applied": [
            {
                "index": item.index,
                "op": item.op_type,
                "description": item.description,
                "reason": item.reason,
            }
            for item in applied
        ],
    }


def validate_json_size(value: str, *, label: str = "JSON") -> None:
    size = len(value.encode("utf-8"))
    if size > MAX_JSON_BYTES:
        raise DeterministicPipelineError(
            f"{label} payload is {size} bytes; limit is {MAX_JSON_BYTES}"
        )
