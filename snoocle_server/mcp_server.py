"""MCP tool surface for the Snoocle server.

Design notes (patterns reused per the brief):
- Tools mirror the pipeline steps 1:1 (discover_song / acquire_audio /
  analyze_audio / reconcile_song / get_song_version ...), NOT one monolithic
  tool — same shape as Dr-Lurie-Blog/CMS-Agent's step-scoped tools
  (trigger_netlify_build, save_json_blob_publish_by_time).
- Audio tools prefer an opaque temporary `audio_ref`, while retaining a
  server-side path or base64 content for backwards-compatible trusted callers.
  Returned binary outputs are opaque artifacts (and optionally base64), never
  server paths.
- Local-first routing (pdf-tool): the deterministic audio tools never touch
  an LLM; reconcile_song is the only AI-invoking tool.
- save_song exposes expected_version optimistic locking —
  saveRecordIfVersionUnchanged, as in CMS-Agent.

Run: `snoocle-mcp` — stdio transport by default (for a local MCP client /
agent runtime to spawn as a subprocess). Set SNOOCLE_MCP_TRANSPORT=
streamable-http to instead serve MCP over HTTP on $PORT/SNOOCLE_MCP_PORT
(e.g. as a second Cloud Run service, gated by Cloud Run IAM auth — see
docs/DEPLOY_CLOUD_RUN.md). SSE is also available for older clients.
"""

from __future__ import annotations

import base64
import asyncio
import dataclasses
import json
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Literal, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from . import __version__
from .audio import utils as audio_utils
from .audio.artifacts import (
    AudioArtifactError,
    AudioArtifactNotFound,
    AudioArtifactValidationError,
    get_audio_artifact_store,
)
from .audio.acquire import acquire as _acquire
from .config import settings
from .deterministic import (
    MAX_CANDIDATES,
    MAX_JSON_BYTES,
    MAX_LINES,
    DeterministicPipelineError,
    build_song_from_candidate as _build_song_from_candidate,
    rank_candidates_deterministically as _rank_candidates_deterministically,
    select_candidate_deterministically as _select_candidate_deterministically,
    align_song_deterministically_service as _align_song_deterministically,
)
from .deterministic_process import (
    process_song_deterministically_service as _process_song_deterministically,
)
from .discovery import CandidateSource, discover_sources
from .discovery.service import candidate_from_text as _candidate_from_text
from .identity import (
    IdentityUnresolvedError,
    require_resolved_song_id,
    resolve_identity_from_evidence,
    song_id_has_unknown_segment,
)
from .manifest import build_evidence_manifest as _build_evidence_manifest
from .manifest import lrc_block as _lrc_block
from .mir import MirAnalysis, analyze_audio as _analyze_audio
from .mir.base import Beat
from .mir.beats import extend_beat_grid as _extend_beat_grid
from .mir.pipeline import analyze_window as _analyze_window
from .mir.cache import analyze_cached as _analyze_cached
from .pipeline import PipelineStepError, get_store, run_pipeline_async
from .reconcile import provider_capabilities
from .reconcile.admission import reconcile_admitted
from .reconcile.match import score_candidate as _score_candidate
from .reconcile.patch_ops import apply_patch as _apply_patch
from .reconcile.patch_ops import parse_ops_response as _parse_ops_response
from .quality.gate import evaluate as _evaluate_quality
from .quality.theory import theory_validity as _theory_validity
from .schema import Song, song_json_schema
from .scope import AnalysisScope
from .store import IdentityCollisionError, backend_label as _store_backend_label
from .store.base import StoreError, VersionConflictError
from .store.identity_rename import rename_song_identity
from .store.runs import get_run_store
from .timing.carry_forward import carry_forward_timing as _carry_forward_timing
from .timing.collapse_guard import guard_against_collapsed_timing as _guard_collapsed
from .timing.confidence import build_review_queue as _build_review_queue
from .timing.confidence import score_song as _score_song_confidence
from .timing.lrc import LrcLine, apply_lrc as _apply_lrc
from .timing.lrc import fetch_lrc_match as _fetch_lrc_match
from .timing.lrc import match_lrc_to_lines as _match_lrc_to_lines
from .timing.offset import estimate_offset as _estimate_offset
from .timing.realign import retime_sections as _retime_sections
from .timing.snap import snap_chords as _snap_chords
from .test_output import TestOutputRejectedError, require_test_output_opt_in
from .tool_contract import (
    TOOL_CONTRACT_VERSION,
    apply_tool_contract,
    registered_tool_contracts,
)

MAX_BEATS = 10_000
MAX_LRC_LINES = 5_000
MAX_MATCHES = MAX_LINES

def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _identity_error(error: IdentityUnresolvedError) -> dict:
    return {"detail": str(error), **error.to_dict()}


def _identity_collision_error(error: IdentityCollisionError) -> dict:
    return {"detail": str(error), "errorCode": error.code}


class _MCPDeterministicInputError(ValueError):
    def __init__(self, code: str, message: str, **details: object) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


def _bounded_payload(value: str, *, label: str) -> None:
    size = len(value.encode("utf-8"))
    if size > MAX_JSON_BYTES:
        raise _MCPDeterministicInputError(
            "payload_too_large",
            f"{label} payload is {size} bytes; limit is {MAX_JSON_BYTES}",
            actualBytes=size,
            maxBytes=MAX_JSON_BYTES,
        )


def _bounded_text_lines(value: str, *, label: str) -> int:
    _bounded_payload(value, label=label)
    line_count = len(value.splitlines())
    if line_count > MAX_LINES:
        raise _MCPDeterministicInputError(
            "too_many_lines",
            f"{label} has {line_count} lines; limit is {MAX_LINES}",
            actualLines=line_count,
            maxLines=MAX_LINES,
        )
    return line_count


def _required_text(value: str, *, label: str, max_chars: int = 500) -> str:
    if not value.strip():
        raise _MCPDeterministicInputError(
            "missing_required_input", f"{label} must not be empty", field=label
        )
    if len(value) > max_chars:
        raise _MCPDeterministicInputError(
            "input_too_long",
            f"{label} has {len(value)} characters; limit is {max_chars}",
            field=label,
            actualCharacters=len(value),
            maxCharacters=max_chars,
        )
    return value


def _json_payload(value: str, *, label: str, expected: type) -> object:
    _bounded_payload(value, label=label)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise _MCPDeterministicInputError(
            "invalid_json",
            f"{label} is not valid JSON",
            line=error.lineno,
            column=error.colno,
        ) from error
    if not isinstance(parsed, expected):
        expected_name = "object" if expected is dict else "array"
        raise _MCPDeterministicInputError(
            "invalid_json_shape",
            f"{label} must be a JSON {expected_name}",
            expected=expected_name,
        )
    return parsed


def _candidate_payload(value: str) -> CandidateSource:
    parsed = _json_payload(value, label="candidate_json", expected=dict)
    candidate = CandidateSource.model_validate(parsed)
    if len(candidate.lines) > MAX_LINES:
        raise _MCPDeterministicInputError(
            "too_many_lines",
            f"candidate has {len(candidate.lines)} lines; limit is {MAX_LINES}",
            actualLines=len(candidate.lines),
            maxLines=MAX_LINES,
        )
    return candidate


def _candidate_list_payload(value: str) -> list[CandidateSource]:
    parsed = _json_payload(value, label="candidates_json", expected=list)
    if len(parsed) > MAX_CANDIDATES:
        raise _MCPDeterministicInputError(
            "too_many_candidates",
            f"candidate count {len(parsed)} exceeds the limit of {MAX_CANDIDATES}",
            actualCandidates=len(parsed),
            maxCandidates=MAX_CANDIDATES,
        )
    candidates = [CandidateSource.model_validate(item) for item in parsed]
    for candidate in candidates:
        if len(candidate.lines) > MAX_LINES:
            raise _MCPDeterministicInputError(
                "too_many_lines",
                f"candidate {candidate.sourceId!r} has {len(candidate.lines)} lines; "
                f"limit is {MAX_LINES}",
                sourceId=candidate.sourceId,
                actualLines=len(candidate.lines),
                maxLines=MAX_LINES,
            )
    return candidates


def _mir_payload(value: str | None) -> MirAnalysis | None:
    if value is None:
        return None
    parsed = _json_payload(value, label="mir_json", expected=dict)
    mir = MirAnalysis.model_validate(parsed)
    if len(mir.beats) > MAX_BEATS:
        raise _MCPDeterministicInputError(
            "too_many_beats",
            f"MIR has {len(mir.beats)} beats; limit is {MAX_BEATS}",
            actualBeats=len(mir.beats),
            maxBeats=MAX_BEATS,
        )
    return mir


def _song_payload(value: str, *, label: str = "song_json") -> Song:
    parsed = _json_payload(value, label=label, expected=dict)
    lines = parsed.get("lines", [])
    if isinstance(lines, list) and len(lines) > MAX_LINES:
        raise _MCPDeterministicInputError(
            "too_many_lines",
            f"{label} has {len(lines)} lines; limit is {MAX_LINES}",
            actualLines=len(lines),
            maxLines=MAX_LINES,
        )
    return Song.model_validate(parsed)


def _lrc_payload(value: str) -> list[LrcLine]:
    parsed = _json_payload(value, label="lrc_json", expected=list)
    if len(parsed) > MAX_LRC_LINES:
        raise _MCPDeterministicInputError(
            "too_many_lrc_lines",
            f"LRC line count {len(parsed)} exceeds the limit of {MAX_LRC_LINES}",
            actualLines=len(parsed),
            maxLines=MAX_LRC_LINES,
        )
    try:
        return [LrcLine(time=float(item["time"]), text=str(item["text"])) for item in parsed]
    except (KeyError, TypeError, ValueError) as error:
        raise _MCPDeterministicInputError(
            "invalid_lrc_lines", "each LRC line must contain numeric time and text"
        ) from error


def _existing_path(value: str, *, label: str) -> Path:
    _required_text(value, label=label, max_chars=4_096)
    path = Path(value)
    if not path.is_file():
        raise _MCPDeterministicInputError(
            "file_not_found", f"{label} does not name an existing file", field=label
        )
    return path


@contextmanager
def _resolved_audio_input(
    audio_path: str | None,
    audio_ref: str | None,
    *,
    label: str = "audio",
):
    """Resolve an opaque artifact or the deprecated trusted local-path input."""
    if audio_path is not None and audio_ref is not None:
        raise _MCPDeterministicInputError(
            "invalid_audio_source",
            f"provide at most one of {label}_path or {label}_ref",
        )
    if audio_ref is not None:
        try:
            with get_audio_artifact_store().materialize(audio_ref) as path:
                yield path
        except AudioArtifactNotFound as error:
            raise _MCPDeterministicInputError(
                "audio_artifact_not_found",
                f"{label}_ref does not name an available audio artifact",
                field=f"{label}_ref",
            ) from error
        except AudioArtifactError as error:
            raise _MCPDeterministicInputError(
                "audio_artifact_unavailable",
                f"{label}_ref could not be resolved because artifact storage is unavailable",
                field=f"{label}_ref",
            ) from error
        return
    if audio_path is not None:
        yield _existing_path(audio_path, label=f"{label}_path")
        return
    yield None


def _validation_details(error: ValidationError) -> list[dict]:
    return json.loads(error.json(include_input=False, include_url=False))


def _deterministic_mcp_response(
    operation: str,
    input_summary: dict,
    call: Callable[[], dict],
    summarize: Callable[[dict], dict],
    *,
    network_access: str = "none",
    cache_access: str = "none",
    persistence: str = "none",
) -> dict:
    start = time.perf_counter()
    access = {
        "network": network_access,
        "cache": cache_access,
        "persistence": persistence,
    }
    try:
        result = call()
    except _MCPDeterministicInputError as error:
        elapsed_ms = max(0, round((time.perf_counter() - start) * 1000))
        return {
            "ok": False,
            "error": {"code": error.code, "message": str(error), "details": error.details},
            "elapsedMs": elapsed_ms,
            "cacheStatus": "not_applicable",
            "modelCalls": 0,
            "modelCostUSD": 0,
            "inputSummary": input_summary,
            "outputSummary": {"status": "error", "operation": operation},
            "warnings": [],
            "access": access,
        }
    except ValidationError as error:
        elapsed_ms = max(0, round((time.perf_counter() - start) * 1000))
        return {
            "ok": False,
            "error": {
                "code": "schema_validation_failed",
                "message": "input does not conform to the required schema",
                "details": {"issues": _validation_details(error)},
            },
            "elapsedMs": elapsed_ms,
            "cacheStatus": "not_applicable",
            "modelCalls": 0,
            "modelCostUSD": 0,
            "inputSummary": input_summary,
            "outputSummary": {"status": "error", "operation": operation},
            "warnings": [],
            "access": access,
        }
    except DeterministicPipelineError as error:
        elapsed_ms = max(0, round((time.perf_counter() - start) * 1000))
        return {
            "ok": False,
            "error": {"code": "deterministic_input_error", "message": str(error)},
            "elapsedMs": elapsed_ms,
            "cacheStatus": "not_applicable",
            "modelCalls": 0,
            "modelCostUSD": 0,
            "inputSummary": input_summary,
            "outputSummary": {"status": "error", "operation": operation},
            "warnings": [],
            "access": access,
        }
    except Exception:  # noqa: BLE001 - MCP callers receive a stable error, never a traceback
        elapsed_ms = max(0, round((time.perf_counter() - start) * 1000))
        return {
            "ok": False,
            "error": {
                "code": "deterministic_operation_failed",
                "message": f"{operation} could not be completed",
            },
            "elapsedMs": elapsed_ms,
            "cacheStatus": "not_applicable",
            "modelCalls": 0,
            "modelCostUSD": 0,
            "inputSummary": input_summary,
            "outputSummary": {"status": "error", "operation": operation},
            "warnings": [],
            "access": access,
        }
    elapsed_ms = max(0, round((time.perf_counter() - start) * 1000))
    return {
        "ok": True,
        "result": result,
        "elapsedMs": elapsed_ms,
        "cacheStatus": "not_applicable",
        "modelCalls": 0,
        "modelCostUSD": 0,
        "inputSummary": input_summary,
        "outputSummary": summarize(result),
        "warnings": [],
        "access": access,
    }


mcp = FastMCP(
    "snoocle",
    instructions=(
        "Snoocle audio-to-song-data foundry (personal-use). Pipeline tools: "
        "discover_song -> acquire_audio -> analyze_audio -> reconcile_song, or "
        "analyze_and_store_song for the full flow with Firestore-backed, "
        "content-versioned persistence. "
        "Model-free source tools parse, score, rank, select, build an untimed "
        "baseline, and validate Song JSON without network or persistence. "
        "Deterministic audio utilities (convert/trim/normalize/probe) never invoke AI."
    ),
)


def _materialize_input(
    input_path: Optional[str], input_base64: Optional[str], input_format: str = "bin"
) -> Path:
    """Server-side path wins; base64 is the fallback for clients that can't
    reference server files."""
    if input_path:
        p = Path(input_path)
        if not p.exists():
            raise ValueError(f"no such file: {input_path}")
        return p
    if input_base64:
        p = Path(tempfile.mkdtemp(prefix="snoocle-mcp-")) / f"in.{input_format.lstrip('.')}"
        p.write_bytes(base64.b64decode(input_base64))
        return p
    raise ValueError("provide input_path or input_base64")


@contextmanager
def _materialized_tool_input(
    input_path: str | None,
    input_ref: str | None,
    input_base64: str | None,
    input_format: str,
):
    sources = sum(value is not None for value in (input_path, input_ref, input_base64))
    if sources != 1:
        raise ValueError("provide exactly one of input_ref, input_path, or input_base64")
    if input_ref is not None or input_path is not None:
        with _resolved_audio_input(input_path, input_ref, label="input") as resolved:
            if resolved is None:  # pragma: no cover - guarded above
                raise ValueError("audio input is required")
            yield resolved
        return
    yield _materialize_input(None, input_base64, input_format)


def _audio_result(dst: Path, return_base64: bool) -> dict:
    try:
        artifact = get_audio_artifact_store().create(
            dst, filename=dst.name, source="mcp"
        )
    except AudioArtifactError as error:
        raise RuntimeError("audio artifact storage could not retain the output") from error
    out: dict = {
        "artifact": artifact.to_public(),
        "probe": dataclasses.asdict(audio_utils.probe(dst)),
    }
    if return_base64:
        out["base64"] = base64.b64encode(dst.read_bytes()).decode()
    return out


# --- deterministic source, candidate, and baseline tools --------------------


@mcp.tool()
def parse_candidate_text(
    text: str,
    source_id: str,
    url: Optional[str] = None,
    title: Optional[str] = None,
    retrieved_at: Optional[str] = None,
) -> dict:
    """Parse caller-supplied chord-sheet text into one CandidateSource.

    Deterministic and local: no network, cache, model, or persistence access.
    The raw text is bounded to MAX_JSON_BYTES and MAX_LINES before parsing.
    """

    def call() -> dict:
        _bounded_text_lines(text, label="text")
        _required_text(source_id, label="source_id", max_chars=200)
        if url is not None and len(url) > 2_048:
            raise _MCPDeterministicInputError(
                "input_too_long", "url exceeds the 2048 character limit", field="url"
            )
        if title is not None and len(title) > 500:
            raise _MCPDeterministicInputError(
                "input_too_long", "title exceeds the 500 character limit", field="title"
            )
        if retrieved_at is not None and len(retrieved_at) > 100:
            raise _MCPDeterministicInputError(
                "input_too_long",
                "retrieved_at exceeds the 100 character limit",
                field="retrieved_at",
            )
        candidate = _candidate_from_text(
            text,
            source_id,
            url=url,
            title=title,
            retrieved_at=retrieved_at or "1970-01-01T00:00:00+00:00",
        )
        if candidate is None:
            raise _MCPDeterministicInputError(
                "candidate_not_plausible",
                "text does not contain enough chord-sheet evidence to form a candidate",
            )
        return {"candidate": candidate.model_dump(mode="json")}

    return _deterministic_mcp_response(
        "parse_candidate_text",
        {
            "bytes": len(text.encode("utf-8")),
            "lines": len(text.splitlines()),
            "sourceId": source_id,
        },
        call,
        lambda result: {
            "sourceId": result["candidate"]["sourceId"],
            "lines": len(result["candidate"]["lines"]),
            "placements": sum(
                len(line["chordPlacements"]) for line in result["candidate"]["lines"]
            ),
        },
    )


@mcp.tool()
def score_candidate_against_mir(candidate_json: str, mir_json: str) -> dict:
    """Score one CandidateSource against MIR across all 12 transpositions.

    Deterministic and local: no network, cache, model, or persistence access.
    """

    def call() -> dict:
        candidate = _candidate_payload(candidate_json)
        mir = _mir_payload(mir_json)
        score = _score_candidate(candidate, mir)
        return {
            "sourceId": score.source_id,
            "score": round(score.score, 6),
            "transposition": score.transposition,
            "matched": score.matched,
            "total": score.total,
            "conflicts": score.conflicts,
        }

    return _deterministic_mcp_response(
        "score_candidate_against_mir",
        {
            "candidateBytes": len(candidate_json.encode("utf-8")),
            "mirBytes": len(mir_json.encode("utf-8")),
        },
        call,
        lambda result: {
            "sourceId": result["sourceId"],
            "matched": result["matched"],
            "total": result["total"],
            "transposition": result["transposition"],
        },
    )


@mcp.tool()
def rank_candidates_deterministically(
    candidates_json: str, mir_json: Optional[str] = None
) -> dict:
    """Rank bounded CandidateSources using the deterministic core service.

    MIR is optional. The tool has no network, cache, model, or persistence access.
    """

    def call() -> dict:
        candidates = _candidate_list_payload(candidates_json)
        ranked = _rank_candidates_deterministically(candidates, _mir_payload(mir_json))
        return {"ranked": [item.to_dict() for item in ranked]}

    return _deterministic_mcp_response(
        "rank_candidates_deterministically",
        {
            "candidatesBytes": len(candidates_json.encode("utf-8")),
            "mirBytes": len(mir_json.encode("utf-8")) if mir_json is not None else 0,
        },
        call,
        lambda result: {
            "candidates": len(result["ranked"]),
            "topSourceId": result["ranked"][0]["sourceId"] if result["ranked"] else None,
        },
    )


@mcp.tool()
def select_candidate_deterministically(
    candidates_json: str,
    strategy: Literal["best", "strict"],
    mir_json: Optional[str] = None,
) -> dict:
    """Select a candidate with the explicit ``best`` or ``strict`` strategy.

    Deterministic and local: no network, cache, model, or persistence access.
    """

    def call() -> dict:
        candidates = _candidate_list_payload(candidates_json)
        selection = _select_candidate_deterministically(
            candidates, _mir_payload(mir_json), strategy=strategy
        )
        return selection.to_dict()

    return _deterministic_mcp_response(
        "select_candidate_deterministically",
        {
            "candidatesBytes": len(candidates_json.encode("utf-8")),
            "mirBytes": len(mir_json.encode("utf-8")) if mir_json is not None else 0,
            "strategy": strategy,
        },
        call,
        lambda result: {
            "status": result["status"],
            "selectedSourceId": result["selectedSourceId"],
            "candidates": len(result["ranked"]),
        },
    )


@mcp.tool()
def build_song_baseline(
    candidate_json: str,
    song_id: str,
    title: str,
    artist: str,
    youtube_video_id: Optional[str] = None,
) -> dict:
    """Build a schema-valid, untimed Song baseline from one CandidateSource.

    Lyrics and chord placements are copied exactly. Capo is zero. The tool has
    no network, cache, model, or persistence access.
    """

    def call() -> dict:
        candidate = _candidate_payload(candidate_json)
        _required_text(song_id, label="song_id", max_chars=200)
        _required_text(title, label="title")
        _required_text(artist, label="artist")
        song = _build_song_from_candidate(
            candidate,
            song_id=song_id,
            title=title,
            artist=artist,
            youtube_video_id=youtube_video_id,
        )
        return {"song": song.model_dump(mode="json")}

    return _deterministic_mcp_response(
        "build_song_baseline",
        {
            "candidateBytes": len(candidate_json.encode("utf-8")),
            "songId": song_id,
            "youtubeVideoIdPresent": youtube_video_id is not None,
        },
        call,
        lambda result: {
            "songId": result["song"]["id"],
            "lines": len(result["song"]["lines"]),
            "sections": len(result["song"]["sections"]),
            "timedLines": sum(
                line["timeSeconds"] is not None for line in result["song"]["lines"]
            ),
        },
    )


@mcp.tool()
def validate_song_json(song_json: str) -> dict:
    """Validate caller-supplied Song JSON against the existing Song schema.

    Returns the normalized schema-valid document. No network, cache, model, or
    persistence access occurs.
    """

    def call() -> dict:
        parsed = _json_payload(song_json, label="song_json", expected=dict)
        lines = parsed.get("lines", [])
        if isinstance(lines, list) and len(lines) > MAX_LINES:
            raise _MCPDeterministicInputError(
                "too_many_lines",
                f"song has {len(lines)} lines; limit is {MAX_LINES}",
                actualLines=len(lines),
                maxLines=MAX_LINES,
            )
        song = Song.model_validate(parsed)
        return {"valid": True, "song": song.model_dump(mode="json")}

    return _deterministic_mcp_response(
        "validate_song_json",
        {"songBytes": len(song_json.encode("utf-8"))},
        call,
        lambda result: {
            "valid": result["valid"],
            "songId": result["song"]["id"],
            "lines": len(result["song"]["lines"]),
        },
    )


# --- deterministic MIR, timing, quality, and evidence leaf tools -----------


@mcp.tool()
def analyze_full_track_mir(
    audio_path: Optional[str] = None,
    accuracy: Literal["standard", "thorough"] = "standard",
    audio_ref: Optional[str] = None,
) -> dict:
    """Analyze one caller-provided server audio file across the full track.

    Resolves an opaque artifact or reads the named legacy local file. No
    acquisition, cache, model, or persistence access occurs; an artifact may
    read the configured shared backend.
    """

    def call() -> dict:
        with _resolved_audio_input(audio_path, audio_ref) as path:
            if path is None:
                raise _MCPDeterministicInputError(
                    "missing_audio_source", "provide audio_ref or audio_path"
                )
            mir = _analyze_audio(path, accuracy=accuracy)
        return {"analysis": mir.model_dump(mode="json")}

    return _deterministic_mcp_response(
        "analyze_full_track_mir",
        {"audioRef": audio_ref, "legacyAudioPath": audio_path is not None, "accuracy": accuracy},
        call,
        lambda result: {
            "durationSeconds": result["analysis"]["duration_seconds"],
            "beats": len(result["analysis"]["beats"]),
            "chords": len(result["analysis"]["chords"]),
        },
        network_access="artifact_backend" if audio_ref is not None else "none",
    )


@mcp.tool()
def analyze_mir_window(
    audio_path: Optional[str] = None,
    start_seconds: float = 0.0,
    end_seconds: float = 0.0,
    audio_ref: Optional[str] = None,
) -> dict:
    """Analyze a bounded window of one caller-provided server audio file."""

    def call() -> dict:
        with _resolved_audio_input(audio_path, audio_ref) as path:
            if path is None:
                raise _MCPDeterministicInputError(
                    "missing_audio_source", "provide audio_ref or audio_path"
                )
            if start_seconds < 0 or end_seconds <= start_seconds:
                raise _MCPDeterministicInputError(
                    "invalid_time_window",
                    "start_seconds must be non-negative and end_seconds must be greater",
                )
            mir = _analyze_window(path, start_seconds, end_seconds)
        return {"analysis": mir.model_dump(mode="json")}

    return _deterministic_mcp_response(
        "analyze_mir_window",
        {
            "audioRef": audio_ref,
            "legacyAudioPath": audio_path is not None,
            "startSeconds": start_seconds,
            "endSeconds": end_seconds,
        },
        call,
        lambda result: {
            "beats": len(result["analysis"]["beats"]),
            "chords": len(result["analysis"]["chords"]),
            "windows": len(result["analysis"]["analyzed_windows"]),
        },
        network_access="artifact_backend" if audio_ref is not None else "none",
    )


@mcp.tool()
def extend_mir_beat_grid(
    beats_json: str,
    duration_seconds: float,
    time_signature: Optional[str] = None,
    gap_bars: float = 2.0,
    min_confirmed_beats: int = 16,
) -> dict:
    """Extend a bounded beat grid using the core tempo-continuation service."""

    def call() -> dict:
        parsed = _json_payload(beats_json, label="beats_json", expected=list)
        if len(parsed) > MAX_BEATS:
            raise _MCPDeterministicInputError(
                "too_many_beats", f"beat count exceeds {MAX_BEATS}", maxBeats=MAX_BEATS
            )
        if duration_seconds < 0 or gap_bars < 0 or min_confirmed_beats < 1:
            raise _MCPDeterministicInputError(
                "invalid_beat_grid_options",
                "duration and gap_bars must be non-negative; min_confirmed_beats must be positive",
            )
        beats = [Beat.model_validate(item) for item in parsed]
        extended = _extend_beat_grid(
            beats,
            duration_seconds,
            time_signature,
            gap_bars=gap_bars,
            min_confirmed_beats=min_confirmed_beats,
        )
        if len(extended) > MAX_BEATS:
            raise _MCPDeterministicInputError(
                "too_many_beats", f"extended beat count exceeds {MAX_BEATS}", maxBeats=MAX_BEATS
            )
        return {"beats": [beat.model_dump(mode="json") for beat in extended]}

    return _deterministic_mcp_response(
        "extend_mir_beat_grid",
        {"beatsBytes": len(beats_json.encode()), "durationSeconds": duration_seconds},
        call,
        lambda result: {
            "beats": len(result["beats"]),
            "inferred": sum(not beat["detected"] for beat in result["beats"]),
        },
    )


@mcp.tool()
def snap_song_to_mir(song_json: str, mir_json: Optional[str] = None) -> dict:
    """Snap a Song's lines and chord placements to optional MIR evidence."""

    def call() -> dict:
        song = _snap_chords(_song_payload(song_json), _mir_payload(mir_json))
        return {"song": song.model_dump(mode="json")}

    return _deterministic_mcp_response(
        "snap_song_to_mir",
        {"songBytes": len(song_json.encode()), "mirPresent": mir_json is not None},
        call,
        lambda result: {
            "songId": result["song"]["id"],
            "timedLines": sum(line["timeSeconds"] is not None for line in result["song"]["lines"]),
        },
    )


@mcp.tool()
def carry_forward_song_timing(
    song_json: str,
    prior_song_json: str,
    audio_fallback_json: Optional[str] = None,
    prior_version: Optional[str] = None,
) -> dict:
    """Copy only confidently matched timing from a prior Song version."""

    def call() -> dict:
        updated, stats = _carry_forward_timing(
            _song_payload(song_json),
            _song_payload(prior_song_json, label="prior_song_json"),
            audio_fallback=(
                _song_payload(audio_fallback_json, label="audio_fallback_json")
                if audio_fallback_json is not None
                else None
            ),
            prior_version=prior_version,
        )
        return {"song": updated.model_dump(mode="json"), "stats": dataclasses.asdict(stats)}

    return _deterministic_mcp_response(
        "carry_forward_song_timing",
        {"songBytes": len(song_json.encode()), "priorSongBytes": len(prior_song_json.encode())},
        call,
        lambda result: {"songId": result["song"]["id"], **result["stats"]},
    )


@mcp.tool()
def lookup_lrc(title: str, artist: str, duration_seconds: Optional[float] = None) -> dict:
    """Look up synced lyrics through LRCLIB; this is the sole networked leaf."""

    def call() -> dict:
        _required_text(title, label="title")
        _required_text(artist, label="artist")
        if duration_seconds is not None and duration_seconds <= 0:
            raise _MCPDeterministicInputError(
                "invalid_duration", "duration_seconds must be positive when supplied"
            )
        match = _fetch_lrc_match(title, artist, duration_seconds)
        if match is None:
            return {"found": False, "match": None}
        if len(match.lines) > MAX_LRC_LINES:
            raise _MCPDeterministicInputError(
                "too_many_lrc_lines", f"LRC result exceeds {MAX_LRC_LINES} lines"
            )
        return {
            "found": True,
            "match": {
                "trackName": match.track_name,
                "artistName": match.artist_name,
                "lines": [dataclasses.asdict(line) for line in match.lines],
            },
        }

    return _deterministic_mcp_response(
        "lookup_lrc",
        {"title": title, "artist": artist, "durationSeconds": duration_seconds},
        call,
        lambda result: {
            "found": result["found"],
            "lines": len(result["match"]["lines"]) if result["match"] else 0,
        },
        network_access="lrclib",
    )


@mcp.tool()
def match_lrc_to_song(lrc_json: str, song_json: str) -> dict:
    """Monotonically match caller-provided LRC lines to Song lines."""

    def call() -> dict:
        song = _song_payload(song_json)
        matches = _match_lrc_to_lines(_lrc_payload(lrc_json), song)
        return {
            "matches": [
                {"lineIndex": idx, "timeSeconds": value[0], "similarity": value[1]}
                for idx, value in sorted(matches.items())
            ]
        }

    return _deterministic_mcp_response(
        "match_lrc_to_song",
        {"lrcBytes": len(lrc_json.encode()), "songBytes": len(song_json.encode())},
        call,
        lambda result: {"matchedLines": len(result["matches"])},
    )


def _matches_payload(value: str) -> dict[int, tuple[float, float]]:
    parsed = _json_payload(value, label="matches_json", expected=list)
    if len(parsed) > MAX_MATCHES:
        raise _MCPDeterministicInputError(
            "too_many_matches", f"match count exceeds {MAX_MATCHES}", maxMatches=MAX_MATCHES
        )
    result: dict[int, tuple[float, float]] = {}
    try:
        for item in parsed:
            line_index = item["lineIndex"]
            if not isinstance(line_index, int) or isinstance(line_index, bool) or line_index < 0:
                raise ValueError
            time_seconds = float(item["timeSeconds"])
            similarity = float(item["similarity"])
            if time_seconds < 0 or not 0 <= similarity <= 1 or line_index in result:
                raise ValueError
            result[line_index] = (time_seconds, similarity)
    except (KeyError, TypeError, ValueError) as error:
        raise _MCPDeterministicInputError(
            "invalid_lrc_matches",
            "matches require unique non-negative lineIndex/timeSeconds and similarity in [0, 1]",
        ) from error
    return result


@mcp.tool()
def apply_lrc_to_song(
    song_json: str, matches_json: str, mir_json: Optional[str] = None
) -> dict:
    """Apply matched LRC anchors and interpolate/re-snap remaining timing."""

    def call() -> dict:
        song = _apply_lrc(
            _song_payload(song_json), _mir_payload(mir_json), _matches_payload(matches_json)
        )
        return {"song": song.model_dump(mode="json")}

    return _deterministic_mcp_response(
        "apply_lrc_to_song",
        {"songBytes": len(song_json.encode()), "matchesBytes": len(matches_json.encode())},
        call,
        lambda result: {
            "songId": result["song"]["id"],
            "timedLines": sum(line["timeSeconds"] is not None for line in result["song"]["lines"]),
        },
    )


@mcp.tool()
def retime_song_sections(song_json: str, duration_seconds: Optional[float] = None) -> dict:
    """Derive section boundaries from a Song's line timing."""

    def call() -> dict:
        if duration_seconds is not None and duration_seconds <= 0:
            raise _MCPDeterministicInputError("invalid_duration", "duration_seconds must be positive")
        song, changed = _retime_sections(_song_payload(song_json), duration_seconds)
        return {"song": song.model_dump(mode="json"), "sectionsChanged": changed}

    return _deterministic_mcp_response(
        "retime_song_sections",
        {"songBytes": len(song_json.encode()), "durationSeconds": duration_seconds},
        call,
        lambda result: {"songId": result["song"]["id"], "sectionsChanged": result["sectionsChanged"]},
    )


@mcp.tool()
def guard_song_timing_collapse(
    song_json: str, duration_seconds: Optional[float] = None
) -> dict:
    """Spread repairable collapsed timing runs and report the intervention."""

    def call() -> dict:
        if duration_seconds is not None and duration_seconds <= 0:
            raise _MCPDeterministicInputError("invalid_duration", "duration_seconds must be positive")
        song, entry = _guard_collapsed(_song_payload(song_json), duration_seconds)
        return {"song": song.model_dump(mode="json"), "provenance": entry.model_dump(mode="json")}

    return _deterministic_mcp_response(
        "guard_song_timing_collapse",
        {"songBytes": len(song_json.encode()), "durationSeconds": duration_seconds},
        call,
        lambda result: {
            "songId": result["song"]["id"],
            "confidence": result["provenance"]["confidence"],
        },
    )


@mcp.tool()
def score_song_confidence(
    song_json: str,
    candidates_json: str = "[]",
    mir_json: Optional[str] = None,
    review_threshold: float = 0.6,
) -> dict:
    """Score every placement and build an explicit low-confidence review queue."""

    def call() -> dict:
        if not 0 <= review_threshold <= 1:
            raise _MCPDeterministicInputError(
                "invalid_threshold", "review_threshold must be in [0, 1]"
            )
        song, scores = _score_song_confidence(
            _song_payload(song_json), _candidate_list_payload(candidates_json), _mir_payload(mir_json)
        )
        return {
            "song": song.model_dump(mode="json"),
            "scores": [dataclasses.asdict(score) for score in scores],
            "reviewQueue": _build_review_queue(scores, threshold=review_threshold),
        }

    return _deterministic_mcp_response(
        "score_song_confidence",
        {"songBytes": len(song_json.encode()), "candidatesBytes": len(candidates_json.encode())},
        call,
        lambda result: {
            "placements": len(result["scores"]), "reviewItems": len(result["reviewQueue"])
        },
    )


@mcp.tool()
def evaluate_song_quality(
    song_json: str,
    candidates_json: str = "[]",
    mir_json: Optional[str] = None,
    can_search: bool = True,
    can_retry: bool = True,
    retries_spent: int = 0,
    searches_spent: int = 0,
    sources_expected: bool = True,
) -> dict:
    """Grade, attribute faults, and decide escalation in the canonical order."""

    def call() -> dict:
        if retries_spent < 0 or searches_spent < 0:
            raise _MCPDeterministicInputError(
                "invalid_budget", "spent retry and search counts must be non-negative"
            )
        decision = _evaluate_quality(
            _song_payload(song_json),
            _mir_payload(mir_json),
            _candidate_list_payload(candidates_json),
            can_search=can_search,
            can_retry=can_retry,
            retries_spent=retries_spent,
            searches_spent=searches_spent,
            sources_expected=sources_expected,
        )
        return {"quality": decision.to_dict()}

    return _deterministic_mcp_response(
        "evaluate_song_quality",
        {"songBytes": len(song_json.encode()), "retriesSpent": retries_spent, "searchesSpent": searches_spent},
        call,
        lambda result: {
            "verdict": result["quality"]["grade"]["verdict"],
            "fault": result["quality"]["attribution"]["fault"],
            "retry": result["quality"]["escalation"]["retry"],
            "search": result["quality"]["escalation"]["search"],
        },
    )


@mcp.tool()
def validate_song_theory(song_json: str, key_name: Optional[str] = None) -> dict:
    """Check whether stored chords are explainable in an explicit or stored key."""

    def call() -> dict:
        song = _song_payload(song_json)
        chords = [
            (line.lineIndex, placement.charIndex, placement.chord)
            for line in song.lines
            for placement in line.chordPlacements
        ]
        report = _theory_validity(chords, key_name or song.metadata.key)
        return {"theory": report.to_dict()}

    return _deterministic_mcp_response(
        "validate_song_theory",
        {"songBytes": len(song_json.encode()), "keyOverride": key_name},
        call,
        lambda result: {
            "share": result["theory"]["share"], "total": result["theory"]["total"]
        },
    )


@mcp.tool()
def calculate_recording_offset(
    reference_audio_path: str = "",
    other_audio_path: str = "",
    max_offset_seconds: float = 30.0,
    reference_audio_ref: Optional[str] = None,
    other_audio_ref: Optional[str] = None,
) -> dict:
    """Estimate a constant offset between two caller-provided recordings."""

    def call() -> dict:
        with _resolved_audio_input(
            reference_audio_path or None, reference_audio_ref, label="reference_audio"
        ) as reference:
            with _resolved_audio_input(
                other_audio_path or None, other_audio_ref, label="other_audio"
            ) as other:
                if reference is None or other is None:
                    raise _MCPDeterministicInputError(
                        "missing_audio_source", "provide both reference and other audio"
                    )
                if not 0 < max_offset_seconds <= 300:
                    raise _MCPDeterministicInputError(
                        "invalid_offset_bound", "max_offset_seconds must be in (0, 300]"
                    )
                estimate = _estimate_offset(
                    reference, other, max_offset_seconds=max_offset_seconds
                )
        return {"offsetSeconds": estimate.offset_seconds, "confidence": estimate.confidence}

    return _deterministic_mcp_response(
        "calculate_recording_offset",
        {
            "referenceAudioRef": reference_audio_ref,
            "otherAudioRef": other_audio_ref,
            "legacyReferencePath": bool(reference_audio_path),
            "legacyOtherPath": bool(other_audio_path),
        },
        call,
        lambda result: result,
        network_access=(
            "artifact_backend"
            if reference_audio_ref is not None or other_audio_ref is not None
            else "none"
        ),
    )


@mcp.tool()
def apply_deterministic_song_patch(song_json: str, patch_json: str) -> dict:
    """Validate and apply the closed, at-most-20-operation Song patch protocol."""

    def call() -> dict:
        document = _json_payload(patch_json, label="patch_json", expected=dict)
        try:
            ops = _parse_ops_response(document)
            song, applied = _apply_patch(_song_payload(song_json), ops)
        except ValueError as error:
            raise _MCPDeterministicInputError("invalid_patch", str(error)) from error
        return {
            "song": song.model_dump(mode="json"),
            "applied": [dataclasses.asdict(item) for item in applied],
        }

    return _deterministic_mcp_response(
        "apply_deterministic_song_patch",
        {"songBytes": len(song_json.encode()), "patchBytes": len(patch_json.encode())},
        call,
        lambda result: {"songId": result["song"]["id"], "operations": len(result["applied"])},
    )


@mcp.tool()
def build_song_evidence_manifest(
    candidates_json: str = "[]",
    mir_json: Optional[str] = None,
    prior_song_json: Optional[str] = None,
    listen: bool = True,
    reconcile: bool = True,
    guidance: Optional[str] = None,
    guidance_origin: Optional[str] = None,
    recording_variant: Optional[str] = None,
    lrc_status: str = "pending",
    lrc_lines_matched: int = 0,
    lrc_lines_total: int = 0,
) -> dict:
    """Build the canonical evidence-state manifest without fetching or storing."""

    def call() -> dict:
        if lrc_status not in {"pending", "hit", "miss", "disabled", "unavailable"}:
            raise _MCPDeterministicInputError("invalid_lrc_status", "unsupported lrc_status")
        if lrc_lines_matched < 0 or lrc_lines_total < 0 or lrc_lines_matched > lrc_lines_total:
            raise _MCPDeterministicInputError(
                "invalid_lrc_counts", "LRC counts must be non-negative and matched <= total"
            )
        if guidance is not None:
            _bounded_payload(guidance, label="guidance")
        candidates = _candidate_list_payload(candidates_json)
        prior = (
            _song_payload(prior_song_json, label="prior_song_json").model_dump(mode="json")
            if prior_song_json is not None
            else None
        )
        manifest = _build_evidence_manifest(
            mir=_mir_payload(mir_json),
            candidates=candidates,
            prior_song=prior,
            scope=AnalysisScope(listen=listen, reconcile=reconcile),
            guidance=guidance,
            guidance_origin=guidance_origin,
            recording_variant=recording_variant,
            lrc=_lrc_block(lrc_status, lrc_lines_matched, lrc_lines_total),
        )
        return {"manifest": manifest}

    return _deterministic_mcp_response(
        "build_song_evidence_manifest",
        {"candidatesBytes": len(candidates_json.encode()), "mirPresent": mir_json is not None},
        call,
        lambda result: {
            "mirStatus": result["manifest"].get("mir", {}).get("status"),
            "sourceCount": result["manifest"].get("sources", {}).get("count", 0),
            "lrcStatus": result["manifest"].get("lrcAlign", {}).get("status"),
        },
    )


# --- deterministic orchestration tools -------------------------------------


def _optional_candidates_payload(value: str | None) -> list[CandidateSource] | None:
    return _candidate_list_payload(value) if value is not None else None


def _optional_lrc_payload(value: str | None) -> list[LrcLine] | None:
    return _lrc_payload(value) if value is not None else None


def _persistence_expected_version(persist: bool, expected_version: str | None) -> str | None:
    if persist and expected_version is None:
        raise _MCPDeterministicInputError(
            "missing_expected_version",
            "expected_version is required when persist=true; use an empty string for create-only",
        )
    if not persist and expected_version is not None:
        raise _MCPDeterministicInputError(
            "unexpected_expected_version",
            "expected_version is only valid when persist=true",
        )
    return expected_version or None


def _save_deterministic_song(song: Song, expected_version: str | None, message: str) -> dict:
    try:
        saved = get_store().save(
            song,
            message,
            expected_version=expected_version,
            enforce_expected=True,
        )
    except VersionConflictError as error:
        raise _MCPDeterministicInputError(
            "version_conflict",
            str(error),
            expectedVersion=expected_version,
            currentVersion=get_store().current_version(song.id),
        ) from error
    except StoreError as error:
        raise _MCPDeterministicInputError("store_rejected", str(error)) from error
    return dataclasses.asdict(saved)


def _save_deterministic_trace(
    *,
    run_type: str,
    song_id: str,
    status: str,
    reason: str | None,
    stages: list[dict],
    totals: dict,
    cache: dict,
    report: dict | None = None,
    quality: dict | None = None,
    persistence: dict | None = None,
) -> str:
    run_id = str(uuid.uuid4())
    timestamp = _now_iso()
    trace = {
        "runId": run_id,
        "runType": run_type,
        "songId": song_id,
        "status": status,
        "reason": reason,
        "startedAt": timestamp,
        "finishedAt": timestamp,
        "steps": stages,
        "totals": totals,
        "cache": cache,
        "alignmentReport": report,
        "quality": quality,
        "persistence": persistence or {"requested": False, "stored": False},
        "modelCalls": 0,
        "modelCostUSD": 0,
        "usageReliable": True,
    }
    get_run_store().save_run(trace)
    return run_id


def _overall_cache_status(cache: dict[str, str]) -> str:
    statuses = set(cache.values())
    if statuses & {"miss", "refresh", "disabled"}:
        return "miss"
    if "hit" in statuses:
        return "hit"
    return "not_applicable"


def _alignment_status(result) -> tuple[str, str | None]:
    verdict = result.quality.grade.verdict
    if verdict == "fail":
        return "needs_review", "quality_gate_failed"
    if verdict == "unknown":
        return "needs_review", "quality_evidence_insufficient"
    if result.review_queue:
        return "needs_review", "low_confidence_placements"
    return "completed", None


def _resolve_alignment_song(
    song_json: str | None, song_id: str | None, song_version: str | None
) -> tuple[Song, str | None, str]:
    if (song_json is None) == (song_id is None):
        raise _MCPDeterministicInputError(
            "invalid_song_source", "provide exactly one of song_json or song_id"
        )
    if song_json is not None:
        if song_version is not None:
            raise _MCPDeterministicInputError(
                "unexpected_song_version", "song_version requires song_id"
            )
        song = _song_payload(song_json)
        return song, None, "caller"
    try:
        song = get_store().get(song_id, song_version)
    except StoreError as error:
        raise _MCPDeterministicInputError("song_not_found", str(error)) from error
    loaded_version = song_version or get_store().current_version(song.id)
    return song, loaded_version, "store"


def _resolve_alignment_mir(
    *,
    mir_json: str | None,
    cached_mir_json: str | None,
    audio_path: str | None,
    audio_ref: str | None,
    recording_id: str | None,
    mir_accuracy: str,
    refresh_mir_cache: bool,
) -> tuple[MirAnalysis | None, dict[str, str], str | None]:
    sources = sum(
        value is not None
        for value in (mir_json, cached_mir_json, audio_path, audio_ref, recording_id)
    )
    if sources > 1:
        raise _MCPDeterministicInputError(
            "invalid_mir_source",
            "provide at most one of mir_json, cached_mir_json, audio_path, audio_ref, or recording_id",
        )
    cache = {"audio": "not_applicable", "mir": "not_applicable"}
    if mir_json is not None:
        return _mir_payload(mir_json), cache, None
    if cached_mir_json is not None:
        cache["mir"] = "hit"
        return _mir_payload(cached_mir_json), cache, None
    path: str | None = None
    resolved_recording_id = recording_id
    if audio_path is not None or audio_ref is not None:
        with _resolved_audio_input(audio_path, audio_ref) as resolved:
            if resolved is None:  # pragma: no cover - guarded by branch
                raise _MCPDeterministicInputError(
                    "missing_audio_source", "provide audio_ref or audio_path"
                )
            path = str(resolved)
            mir, info = _analyze_cached(
                path,
                accuracy=mir_accuracy,
                compute=lambda: _analyze_audio(path, accuracy=mir_accuracy),
                refresh=refresh_mir_cache,
            )
        cache["mir"] = info.status
        return mir, cache, resolved_recording_id
    elif recording_id is not None:
        acquired = _acquire(video_url_or_id=recording_id)
        path = acquired.path
        resolved_recording_id = acquired.video_id
        cache["audio"] = "hit" if acquired.from_cache else "miss"
    if path is None:
        return None, cache, resolved_recording_id
    mir, info = _analyze_cached(
        path,
        accuracy=mir_accuracy,
        compute=lambda: _analyze_audio(path, accuracy=mir_accuracy),
        refresh=refresh_mir_cache,
    )
    cache["mir"] = info.status
    return mir, cache, resolved_recording_id


def _align_song_worker(
    *,
    song_json: str | None,
    song_id: str | None,
    song_version: str | None,
    mir_json: str | None,
    cached_mir_json: str | None,
    audio_path: str | None,
    audio_ref: str | None,
    recording_id: str | None,
    candidates_json: str,
    lrc_json: str | None,
    use_lrc: bool,
    mir_accuracy: str,
    refresh_mir_cache: bool,
    persist: bool,
    expected_version: str | None,
) -> dict:
    def call() -> dict:
        expected = _persistence_expected_version(persist, expected_version)
        song, loaded_version, song_source = _resolve_alignment_song(
            song_json, song_id, song_version
        )
        mir, cache, resolved_recording_id = _resolve_alignment_mir(
            mir_json=mir_json,
            cached_mir_json=cached_mir_json,
            audio_path=audio_path,
            audio_ref=audio_ref,
            recording_id=recording_id,
            mir_accuracy=mir_accuracy,
            refresh_mir_cache=refresh_mir_cache,
        )
        candidates = _candidate_list_payload(candidates_json)
        lrc_lines = _optional_lrc_payload(lrc_json)
        result = _align_song_deterministically(
            song,
            mir,
            candidates=candidates,
            lrc_lines=lrc_lines,
            use_lrc=use_lrc,
            document_version=loaded_version,
        )
        payload = result.to_dict()
        status, reason = _alignment_status(result)
        persisted = {"requested": persist, "stored": False}
        if persist:
            try:
                saved = _save_deterministic_song(
                    result.song, expected, "Deterministic alignment"
                )
            except _MCPDeterministicInputError as error:
                failed_persistence = {
                    "requested": True,
                    "stored": False,
                    "errorCode": error.code,
                }
                _save_deterministic_trace(
                    run_type="deterministic-align",
                    song_id=result.song.id,
                    status="failed",
                    reason=error.code,
                    stages=payload["stages"],
                    totals=payload["totals"],
                    cache=cache,
                    report=payload["alignmentReport"],
                    quality=payload["quality"],
                    persistence=failed_persistence,
                )
                raise
            persisted = {"requested": True, "stored": True, **saved}
        payload.update(
            {
                "status": status,
                "reason": reason,
                "cache": cache,
                "songSource": song_source,
                "recordingId": resolved_recording_id,
                "persistence": persisted,
            }
        )
        run_id = _save_deterministic_trace(
            run_type="deterministic-align",
            song_id=result.song.id,
            status=status,
            reason=reason,
            stages=payload["stages"],
            totals=payload["totals"],
            cache=cache,
            report=payload["alignmentReport"],
            quality=payload["quality"],
            persistence=persisted,
        )
        payload["runId"] = run_id
        return payload

    network = []
    if recording_id is not None:
        network.append("audio_acquisition")
    if use_lrc and lrc_json is None:
        network.append("lrclib")
    response = _deterministic_mcp_response(
        "align_song_deterministically",
        {
            "songSource": "json" if song_json is not None else "store",
            "mirSource": (
                "json" if mir_json is not None else
                "cache" if cached_mir_json is not None else
                "artifact" if audio_ref is not None else
                "audio" if audio_path is not None else
                "recording" if recording_id is not None else "none"
            ),
            "persist": persist,
        },
        call,
        lambda result: {
            "status": result["status"],
            "songId": result["song"]["id"],
            "qualityVerdict": result["quality"]["grade"]["verdict"],
            "reviewItems": len(result["reviewQueue"]),
            "stored": result["persistence"]["stored"],
        },
        network_access=",".join(network) or "none",
        cache_access=(
            "mir,audio"
            if audio_path is not None or audio_ref is not None or recording_id is not None
            else "none"
        ),
        persistence="run_trace_and_optional_song",
    )
    if response["ok"]:
        response["cacheStatus"] = _overall_cache_status(response["result"]["cache"])
    return response


@mcp.tool()
async def align_song_deterministically(
    song_json: str = "",
    song_id: Optional[str] = None,
    song_version: Optional[str] = None,
    mir_json: str = "",
    cached_mir_json: str = "",
    audio_path: Optional[str] = None,
    audio_ref: Optional[str] = None,
    recording_id: Optional[str] = None,
    candidates_json: str = "[]",
    lrc_json: str = "",
    use_lrc: bool = False,
    mir_accuracy: Literal["standard", "thorough"] = "standard",
    refresh_mir_cache: bool = False,
    persist: bool = False,
    expected_version: Optional[str] = None,
) -> dict:
    """Run snap→LRC→sections→collapse→confidence→quality without a model.

    Song persistence is opt-in and requires explicit optimistic locking. A
    bounded run trace is always persisted. Blocking acquisition/MIR work runs
    in a worker thread rather than on the MCP async event loop.
    """
    return await asyncio.to_thread(
        _align_song_worker,
        song_json=song_json or None,
        song_id=song_id,
        song_version=song_version,
        mir_json=mir_json or None,
        cached_mir_json=cached_mir_json or None,
        audio_path=audio_path,
        audio_ref=audio_ref,
        recording_id=recording_id,
        candidates_json=candidates_json,
        lrc_json=lrc_json or None,
        use_lrc=use_lrc,
        mir_accuracy=mir_accuracy,
        refresh_mir_cache=refresh_mir_cache,
        persist=persist,
        expected_version=expected_version,
    )


def _process_song_worker(
    *,
    title: str,
    artist: str,
    audio_path: str | None,
    audio_ref: str | None,
    recording_id: str | None,
    mir_json: str | None,
    candidates_json: str | None,
    lrc_json: str | None,
    use_lrc: bool,
    selection_strategy: str,
    max_candidates: int,
    mir_accuracy: str,
    refresh_mir_cache: bool,
    refresh_discovery_cache: bool,
    persist: bool,
    expected_version: str | None,
) -> dict:
    def call() -> dict:
        expected = _persistence_expected_version(persist, expected_version)
        if len(title) > 500 or len(artist) > 500:
            raise _MCPDeterministicInputError(
                "input_too_long", "title and artist are limited to 500 characters"
            )
        if not 1 <= max_candidates <= MAX_CANDIDATES:
            raise _MCPDeterministicInputError(
                "invalid_candidate_limit",
                f"max_candidates must be between 1 and {MAX_CANDIDATES}",
            )
        if selection_strategy not in {"best", "strict"}:
            raise _MCPDeterministicInputError(
                "invalid_selection_strategy", "selection_strategy must be best or strict"
            )
        candidates = _optional_candidates_payload(candidates_json)
        lrc_lines = _optional_lrc_payload(lrc_json)
        mir = _mir_payload(mir_json)
        with _resolved_audio_input(audio_path, audio_ref) as resolved_audio:
            result = _process_song_deterministically(
                title=title,
                artist=artist,
                audio_path=str(resolved_audio) if resolved_audio is not None else None,
                recording_id=recording_id,
                mir=mir,
                candidates=candidates,
                lrc_lines=lrc_lines,
                use_lrc=use_lrc,
                selection_strategy=selection_strategy,
                max_candidates=max_candidates,
                mir_accuracy=mir_accuracy,
                refresh_mir_cache=refresh_mir_cache,
                refresh_discovery_cache=refresh_discovery_cache,
            )
        payload = result.to_dict()
        persisted = {"requested": persist, "stored": False}
        if persist and result.alignment is not None:
            try:
                saved = _save_deterministic_song(
                    result.alignment.song, expected, "Deterministic processing"
                )
            except _MCPDeterministicInputError as error:
                failed_persistence = {
                    "requested": True,
                    "stored": False,
                    "errorCode": error.code,
                }
                _save_deterministic_trace(
                    run_type="deterministic-process",
                    song_id=result.song_id,
                    status="failed",
                    reason=error.code,
                    stages=payload["stages"],
                    totals=payload["totals"],
                    cache=payload["cache"],
                    report=payload.get("alignmentReport"),
                    quality=payload.get("quality"),
                    persistence=failed_persistence,
                )
                raise
            persisted = {"requested": True, "stored": True, **saved}
        elif persist:
            persisted["reason"] = "song_not_produced"
        payload["persistence"] = persisted
        run_id = _save_deterministic_trace(
            run_type="deterministic-process",
            song_id=result.song_id,
            status=result.status,
            reason=result.reason,
            stages=payload["stages"],
            totals=payload["totals"],
            cache=payload["cache"],
            report=payload.get("alignmentReport"),
            quality=payload.get("quality"),
            persistence=persisted,
        )
        payload["runId"] = run_id
        return payload

    network = []
    if audio_path is None and audio_ref is None and mir_json is None:
        network.append("audio_acquisition")
    if candidates_json is None:
        network.append("discovery")
    if use_lrc and lrc_json is None:
        network.append("lrclib")
    response = _deterministic_mcp_response(
        "process_song_deterministically",
        {
            "title": title,
            "artist": artist,
            "callerAudio": audio_path is not None or audio_ref is not None,
            "audioRef": audio_ref,
            "callerMir": mir_json is not None,
            "callerCandidates": candidates_json is not None,
            "persist": persist,
        },
        call,
        lambda result: {
            "status": result["status"],
            "songId": result["songId"],
            "reason": result["reason"],
            "stored": result["persistence"]["stored"],
        },
        network_access=",".join(network) or "none",
        cache_access="audio,mir,discovery",
        persistence="run_trace_and_optional_song",
    )
    if response["ok"]:
        response["cacheStatus"] = _overall_cache_status(response["result"]["cache"])
    return response


@mcp.tool()
async def process_song_deterministically(
    title: str,
    artist: str,
    audio_path: Optional[str] = None,
    audio_ref: Optional[str] = None,
    recording_id: Optional[str] = None,
    mir_json: str = "",
    candidates_json: str = "",
    lrc_json: str = "",
    use_lrc: bool = False,
    selection_strategy: Literal["best", "strict"] = "strict",
    max_candidates: int = 8,
    mir_accuracy: Literal["standard", "thorough"] = "standard",
    refresh_mir_cache: bool = False,
    refresh_discovery_cache: bool = False,
    persist: bool = False,
    expected_version: Optional[str] = None,
) -> dict:
    """Build and align a Song through the complete deterministic pipeline.

    The tool never calls reconciliation or a model provider. Song persistence
    is opt-in with an explicit expected version; run traces are always saved.
    All blocking acquisition, discovery, and MIR work runs in a worker thread.
    """
    return await asyncio.to_thread(
        _process_song_worker,
        title=title,
        artist=artist,
        audio_path=audio_path,
        audio_ref=audio_ref,
        recording_id=recording_id,
        mir_json=mir_json or None,
        candidates_json=candidates_json or None,
        lrc_json=lrc_json or None,
        use_lrc=use_lrc,
        selection_strategy=selection_strategy,
        max_candidates=max_candidates,
        mir_accuracy=mir_accuracy,
        refresh_mir_cache=refresh_mir_cache,
        refresh_discovery_cache=refresh_discovery_cache,
        persist=persist,
        expected_version=expected_version,
    )


# --- pipeline steps ----------------------------------------------------------


def _store_acquired_audio(acquired, *, source: Literal["youtube", "mcp"] = "youtube"):
    try:
        return get_audio_artifact_store().create(
            acquired.path,
            filename=Path(acquired.path).name,
            source=source,
            youtube_video_id=acquired.video_id,
        )
    except AudioArtifactError as error:
        raise RuntimeError("audio artifact storage could not retain the recording") from error


@mcp.tool()
def discover_song(title: str, artist: str, max_candidates: int = 8) -> dict:
    """Find candidate chord/lyric text sources for a song via general web
    search (step 2-3). Returns parsed, sounding-pitch-normalized candidates,
    each with confidence/provenance — kept separate for reconciliation."""
    cands = discover_sources(title, artist, max_candidates=max_candidates)
    return {"count": len(cands), "candidates": [c.model_dump() for c in cands]}


@mcp.tool()
def acquire_audio(
    title: Optional[str] = None,
    artist: Optional[str] = None,
    youtube_url_or_id: Optional[str] = None,
) -> dict:
    """Acquire the song's recording from YouTube server-side (personal-use
    tool). Give a video URL/id, or title+artist to search. Cached by video id."""
    acquired = _acquire(title=title, artist=artist, video_url_or_id=youtube_url_or_id)
    artifact = _store_acquired_audio(acquired)
    return {
        "artifact": artifact.to_public(),
        "youtubeVideoId": acquired.video_id,
        "videoTitle": acquired.video_title,
        "fromCache": acquired.from_cache,
    }


@mcp.tool()
def analyze_audio(
    audio_path: Optional[str] = None,
    audio_ref: Optional[str] = None,
    input_base64: Optional[str] = None,
    input_format: str = "bin",
    title: Optional[str] = None,
    artist: Optional[str] = None,
    youtube_url_or_id: Optional[str] = None,
    accuracy: str = "standard",
) -> dict:
    """MIR analysis of a recording (step 4): beats/downbeats, chord timeline,
    structural sections, bpm, key — audio-grounded, independent of any text
    source. Provide ONE of: audio_path (a server-side file); input_base64 (the
    bytes of an uploaded audio OR video file — set input_format to its
    extension, e.g. "mp4"/"mov"/"mp3"; for video the audio track is extracted
    automatically); or acquisition params (title/artist/youtube_url_or_id) to
    fetch from YouTube first. Chords are the sounding harmony, never a
    fretboard shape."""
    video_id = None
    result_audio_ref = audio_ref
    supplied = sum(value is not None for value in (audio_path, audio_ref, input_base64))
    if supplied > 1:
        raise ValueError("provide at most one of audio_ref, audio_path, or input_base64")
    if supplied == 0:
        acquired = _acquire(title=title, artist=artist, video_url_or_id=youtube_url_or_id)
        audio_path = acquired.path
        video_id = acquired.video_id
        result_audio_ref = _store_acquired_audio(acquired).audio_ref
        analysis = _analyze_audio(audio_path, accuracy=accuracy)
    elif audio_ref is not None:
        with _resolved_audio_input(None, audio_ref) as resolved:
            if resolved is None:  # pragma: no cover - guarded above
                raise ValueError("audio_ref is required")
            analysis = _analyze_audio(resolved, accuracy=accuracy)
    elif input_base64 is not None:
        if len(input_base64) > ((settings.audio_artifact_max_bytes + 2) // 3) * 4 + 8:
            raise ValueError("uploaded audio exceeds the configured artifact size limit")
        materialized = _materialize_input(None, input_base64, input_format)
        analysis = _analyze_audio(materialized, accuracy=accuracy)
        try:
            artifact = get_audio_artifact_store().create(
                materialized,
                filename=f"input.{input_format.lstrip('.')}",
                source="mcp",
            )
        except AudioArtifactValidationError:
            # Backwards compatibility: this input has historically accepted
            # ffmpeg-readable video containers too. Analyze those ephemerally,
            # but do not mislabel them as reusable audio artifacts.
            result_audio_ref = None
        except AudioArtifactError as error:
            raise RuntimeError("audio artifact storage could not retain the upload") from error
        else:
            result_audio_ref = artifact.audio_ref
    else:
        with _resolved_audio_input(audio_path, None) as resolved:
            if resolved is None:  # pragma: no cover - guarded above
                raise ValueError("audio_path is required")
            analysis = _analyze_audio(resolved, accuracy=accuracy)
    return {
        "audioRef": result_audio_ref,
        "youtubeVideoId": video_id,
        "analysis": analysis.model_dump(),
    }


@mcp.tool()
def reconcile_song(
    title: str,
    artist: str,
    candidates_json: Optional[str] = None,
    mir_json: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    audio_path: Optional[str] = None,
    audio_ref: Optional[str] = None,
    attach_audio: Optional[bool] = None,
    youtube_video_id: Optional[str] = None,
    media_url: Optional[str] = None,
    force: bool = False,
    force_reason: Optional[str] = None,
    batch_id: Optional[str] = None,
    effort_level: Optional[str] = None,
    prior_song: Optional[dict] = None,
) -> dict:
    """Reconcile candidate text sources + MIR analysis into a schema-compliant
    Song JSON via the configured reconciler (step 5). candidates_json/mir_json
    accept the outputs of discover_song / analyze_audio; when candidates_json
    is omitted, discovery runs first. Does NOT persist — use save_song or
    analyze_and_store_song for that. provider: anthropic | openai | gemini |
    agent | mock. The "agent" provider delegates to an external agent
    workspace's MCP server (SNOOCLE_AGENT_MCP_URL), sending title/artist,
    media_url (YouTube watch URL or other media URL; derived from
    youtube_video_id when omitted), and the timestamped MIR chord timeline."""
    if candidates_json:
        candidates = [CandidateSource.model_validate(c) for c in json.loads(candidates_json)]
    else:
        # With no already-gathered page to recover from, reject an unresolved
        # identity before a search can spend time or create a later run.
        try:
            identity = resolve_identity_from_evidence(artist=artist, title=title)
        except IdentityUnresolvedError as error:
            return _identity_error(error)
        title, artist = identity.title, identity.artist
        candidates = discover_sources(title, artist)
    mir = None
    if mir_json:
        payload = json.loads(mir_json)
        mir = MirAnalysis.model_validate(payload.get("analysis", payload))
    # No `timing_authority`: this tool returns the document to the agent and
    # persists nothing (save_song is a separate call that runs no timing pass
    # either), so no deterministic pass will re-time it and the model's timing
    # must survive intact. See reconcile/engine.py's TimingAuthority.
    try:
        with _resolved_audio_input(audio_path, audio_ref) as resolved_audio:
            admitted = reconcile_admitted(
                title,
                artist,
                candidates,
                mir,
                provider=provider,
                model=model,
                audio_path=str(resolved_audio) if resolved_audio is not None else None,
                attach_audio=attach_audio,
                youtube_video_id=youtube_video_id,
                media_url=media_url,
                force=force,
                force_reason=force_reason,
                batch_id=batch_id,
                effort_level=effort_level,
                prior_song=prior_song,
            )
    except IdentityUnresolvedError as error:
        return _identity_error(error)
    except Exception as error:
        from .usage import BudgetExceededError
        if isinstance(error, BudgetExceededError):
            return error.to_dict()
        raise
    result = admitted.result
    return {
        "song": result.song.model_dump(),
        "provider": result.provider,
        "model": result.model,
        "attempts": result.attempts,
        "audioAttached": result.audio_attached,
        "runId": admitted.recorder.trace.run_id,
        "evidenceManifest": admitted.evidence_manifest,
    }


@mcp.tool()
async def analyze_and_store_song(
    title: Optional[str] = None,
    artist: Optional[str] = None,
    youtube_url_or_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    skip_audio: bool = False,
    expected_version: Optional[str] = None,
    accuracy: Optional[str] = None,
    guidance: Optional[str] = None,
    scope: Optional[dict] = None,
    allow_timing_loss: bool = False,
    force: bool = False,
    force_reason: Optional[str] = None,
    batch_id: Optional[str] = None,
    effort_level: Optional[str] = None,
    agent_policy: Optional[Literal["never", "unresolved_only", "always"]] = None,
    allow_test_output: bool = False,
) -> dict:
    """Full pipeline: (resolve) -> discover -> acquire -> MIR -> reconcile ->
    commit a new version to the Firestore-backed store (never overwrites;
    content-hash versions). Give title+artist, a youtube_url_or_id, or both.
    Video metadata is only ever a FALLBACK for whichever of title/artist you
    did not supply — it never overrides one you did give, even when a
    youtube_url_or_id is also present. The song id is permanent (content-hash
    versioned store), so a caller-supplied identity always wins.
    Each external step runs under its own timeout so the call can't hang; a
    fatal step failure raises with the step name. Returns the song, the per-step
    report, and the stored version sha.

    guidance (optional): free-text instructions for THIS run — the
    human-in-the-loop correction path ("change the C to a B in line 12", "the
    bridge is Bm, not D"), handed to the reconciler as high-priority evidence so
    a re-analysis honors the fix instead of rediscovering from scratch. Stored as
    the song's notes before any expensive step runs, so a run that dies at
    acquire or reconcile does not eat it, and applied ONCE: it replays into a
    retry that passes none, then stops replaying as soon as it has landed in a
    stored version. It never disturbs a standing preference already on file
    (set_song_notes) — if one is set, BOTH are in force for this run, combined.
    What the run is SHOWN can be less: a notes-only run (inferred from a
    targeted correction, or asked for explicitly) is handed the correction
    ALONE, because its contract is "change nothing else" and a standing
    rendering preference is no part of that; a run that is not notes-only sees
    both. `steps.notes` says which happened. Max 8000 chars, same ceiling as
    set_song_notes — over it is a rejection here, not a truncation later.
    For a standing instruction that should shape EVERY later analyze
    ("capo-free voicings please") use set_song_notes instead; get_song_notes
    shows what a run passing no guidance would replay.

    scope (optional, same shape as the HTTP API's `scope`):
    {"listen": bool, "reconcile": bool} — which evidence-gathering stages this
    run may do. listen=false reuses the existing audio analysis instead of
    re-downloading and re-analyzing; reconcile=false skips web source
    discovery. Both false means "apply the notes to the prior song and gather
    nothing". OMIT it for the full pipeline (the default) — when guidance
    names a specific chord, line, or section and scope is omitted, notes-only
    scope is INFERRED automatically (an explicit scope always overrides
    inference). A targeted, non-lyric correction in notes-only scope is
    applied as a PATCH against the prior document rather than a regenerated
    Song — see docs/ARCHITECTURE.md, "A targeted correction is a PATCH, not a
    rewrite" — which is what makes a one-chord fix immune to both the content
    filter and timing loss.

    listen=false carries the prior version's timing (beat grid, bpm, chord and
    line times, section times) forward onto the new document, and fails if
    there is no prior version to carry it from. allow_timing_loss=true opts
    out of that and of the guard that refuses to store a version which drops
    audio-derived data the prior one had — only pass it when losing that data
    is the actual intent."""
    from .store.song_notes import length_error as notes_length_error

    # Same per-slot ceiling and same message as set_song_notes: `guidance` is a
    # CORRECTION write, and both slots of one contract must refuse an over-long
    # body the same way. Before the pipeline, so nothing is paid for a run that
    # is going to be refused.
    guidance_problem = notes_length_error(guidance)
    if guidance_problem:
        raise ValueError(guidance_problem)
    try:
        report = await run_pipeline_async(
            title,
            artist,
            youtube_url_or_id=youtube_url_or_id,
            provider=provider,
            model=model,
            skip_audio=skip_audio,
            expected_version=expected_version,
            accuracy=accuracy,
            guidance=guidance,
            # None in -> None out: an omitted scope must stay omitted all the way
            # down, or every MCP caller silently starts getting a "scope" step.
            scope=AnalysisScope.parse(scope),
            allow_timing_loss=allow_timing_loss,
            force=force,
            force_reason=force_reason,
            batch_id=batch_id,
            effort_level=effort_level,
            agent_policy=agent_policy,
            allow_test_output=allow_test_output,
        )
    except PipelineStepError as error:
        if error.error_code == "identity_unresolved" and isinstance(error.__cause__, IdentityUnresolvedError):
            return _identity_error(error.__cause__)
        if error.error_code == "identity_collision" and isinstance(error.__cause__, IdentityCollisionError):
            return _identity_collision_error(error.__cause__)
        if error.error_code == "budget_exceeded":
            from .usage import BudgetExceededError
            cause = error.__cause__
            return (
                cause.to_dict()
                if isinstance(cause, BudgetExceededError)
                else {"code": "budget_exceeded", "error": str(error)}
            )
        raise
    response = {
        "songId": report.song_id,
        "status": report.status,
        "reason": report.reason,
        "agentPolicy": report.agent_policy,
        "steps": report.steps,
        "storedVersion": report.stored_version,
        "runId": report.run_id,
    }
    if report.deterministic_result is not None:
        response["deterministicResult"] = report.deterministic_result
    if report.agent_patch is not None:
        response["agentPatch"] = report.agent_patch
    if report.reconcile is not None:
        response["song"] = report.reconcile.song.model_dump()
    return response


@mcp.tool()
async def realign_song_to_recording(
    song_id: str,
    video_id: str,
    source_version: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    analysis_depth: Optional[str] = None,
    expected_version: Optional[str] = None,
    allow_same_recording: bool = False,
    allow_timing_loss: bool = False,
) -> dict:
    """Mode B — "analyze <video_id> as the timing reference for <song_id>".

    Re-times an EXISTING song document to a DIFFERENT recording of the same
    song: full MIR analysis of that video, then the document's own lyrics and
    chord sequence re-snapped onto the new timeline. Lyrics and chord identities
    carry over untouched; the transposition between the document's key and the
    recording's is derived from the chord sequence and applied.

    Stored as a new version of the SAME song id — this is the same song, timed
    against different audio, and it records which video supplied the timing.

    Mostly deterministic and normally free of model tokens: a model is consulted
    ONLY when a structural comparison finds a difference the stored document
    cannot explain (an extra chorus, a truncated outro). Use it when a version's
    timing came out unreliable on a poor recording (see get_song's provenance for
    a "timing-unreliable" entry, and `suggest_better_recordings` for candidates).

    Refuses (and says so) when the given video turns out to be the SAME recording
    the song is already timed against: that case wants one cross-correlation via
    the video-offset endpoint, not a full analysis. allow_same_recording=true
    overrides."""
    from .realign import realign_song_async

    report = await realign_song_async(
        song_id,
        video_id,
        provider=provider,
        model=model,
        analysis_depth=analysis_depth,
        expected_version=expected_version,
        source_version=source_version,
        allow_same_recording=allow_same_recording,
        allow_timing_loss=allow_timing_loss,
    )
    return report.to_dict()


@mcp.tool()
def suggest_better_recordings(song_id: str, limit: int = 5) -> dict:
    """Ranked alternative recordings of a song. REPORTS ONLY — never analyzes.

    For a song whose timing came out unreliable because its recording is poor
    (a lo-fi live video, a broadcast rip): a better recording is the fix, and
    this finds candidates — official/studio audio preferred, covers and lessons
    ranked down, the recording the song is already timed against excluded.

    Nothing is downloaded or analyzed. Each suggestion carries the exact action
    that would spend, which is `realign_song_to_recording(song_id, video_id)`:
    that is a full analysis of a second track, so it stays an explicit choice."""
    from .recordings import suggest_recordings

    song = get_store().get(song_id)
    return suggest_recordings(
        song, reason="requested for this song", limit=max(1, min(limit, 10))
    ).to_dict()


# --- versioned store ---------------------------------------------------------


@mcp.tool()
def list_songs() -> dict:
    """List song ids present in the store."""
    return {"songs": get_store().list_songs()}


@mcp.tool()
def get_song(song_id: str, version: Optional[str] = None) -> dict:
    """Fetch a song's JSON from the store — latest, or a specific stored
    version sha."""
    return get_store().get(song_id, version=version).model_dump()


@mcp.tool()
def list_song_versions(song_id: str) -> dict:
    """List the stored versions (sha, timestamp, message) of a song,
    newest first."""
    return {
        "songId": song_id,
        "versions": [dataclasses.asdict(v) for v in get_store().versions(song_id)],
    }


@mcp.tool()
def diff_song_versions(song_id: str, version_a: str, version_b: str) -> str:
    """Unified diff (pretty-printed JSON) of a song between two stored versions."""
    return get_store().diff(song_id, version_a, version_b)


@mcp.tool()
def save_song(
    song_json: str,
    message: str = "Manual save",
    expected_version: Optional[str] = None,
    allow_test_output: bool = False,
) -> dict:
    """Validate and commit a Song JSON as a new version. expected_version
    enables optimistic locking (save-if-version-unchanged): the save is
    rejected if the stored version moved since you read it. Provenance is
    append-only — the new document must extend the stored history."""
    song = Song.model_validate_json(song_json)
    try:
        require_resolved_song_id(song.id)
        require_test_output_opt_in(song, allow_test_output)
    except IdentityUnresolvedError as error:
        return _identity_error(error)
    except TestOutputRejectedError as error:
        return {"error": {"code": error.error_code, "message": str(error)}}
    try:
        saved = get_store().save(song, message, expected_version=expected_version)
    except IdentityCollisionError as error:
        return _identity_collision_error(error)
    return dataclasses.asdict(saved)


@mcp.tool()
def diagnose_mock_songs() -> dict:
    """Read-only inventory of stored testOnly/mock/placeholder Song documents."""
    from .test_output import mock_output_reasons

    findings = []
    store = get_store()
    for song_id in store.list_songs():
        try:
            song = store.get(song_id)
        except Exception as error:  # noqa: BLE001
            findings.append({"songId": song_id, "reasons": ["unreadable"], "error": str(error)[:300]})
            continue
        reasons = mock_output_reasons(song)
        if reasons:
            findings.append(
                {
                    "songId": song.id,
                    "title": song.metadata.title,
                    "artist": song.metadata.artist,
                    "version": store.current_version(song.id),
                    "reasons": reasons,
                }
            )
    return {"count": len(findings), "songs": findings, "readOnly": True}


@mcp.tool()
def list_capabilities() -> dict:
    """Describe every registered MCP tool and its operational behavior.

    Existing capability keys remain stable. ``toolContract`` is the exhaustive
    GUI-facing contract also published on each tool's namespaced MCP ``_meta``.
    """
    entries = []
    for name, tool, contract in registered_tool_contracts(mcp):
        if "song_version_optional" in contract.persistence:
            persistence = (
                "run trace; Song write only when persist=true with expected-version locking"
            )
        elif any(kind.endswith("_write") or kind == "filesystem" for kind in contract.persistence):
            persistence = "writes"
        elif contract.persistence:
            persistence = "reads"
        else:
            persistence = "read-only or none"
        parameters = getattr(tool, "parameters", None) or getattr(tool, "input_schema", None)
        output_schema = getattr(tool, "output_schema", None)
        contract_wire = contract.to_wire()
        entries.append(
            {
                "name": name,
                "group": contract.category,
                "execution": contract.execution,
                "networkAccess": "possible" if contract.network_access else "none",
                "persistence": persistence,
                "inputType": parameters or "typed MCP arguments",
                "outputType": output_schema or "structured JSON",
                "cacheBehavior": {
                    "none": "none",
                    "read": "may read deterministic cache",
                    "read_write": "may read/write deterministic cache",
                }[contract.cache_behavior],
                "costClass": contract.cost_class,
                "title": contract.title,
                "browserSafety": contract.browser_safety,
                "inputArtifactKinds": list(contract.input_artifact_kinds),
                "outputArtifactKinds": list(contract.output_artifact_kinds),
                "accessMode": contract.access_mode,
                "readOnly": contract.read_only,
                "destructive": contract.destructive,
                "idempotent": contract.idempotent,
                "networkAccessKinds": list(contract.network_access),
                "modelUse": contract.model_use,
                "persistenceKinds": list(contract.persistence),
                "expectedDuration": contract.expected_duration,
                "specializedRenderer": contract.specialized_renderer,
                "toolContract": contract_wire,
            }
        )
    grouped: dict[str, list[dict]] = {}
    for entry in entries:
        grouped.setdefault(entry["group"], []).append(entry)
    return {
        "contractVersion": TOOL_CONTRACT_VERSION,
        "toolCount": len(entries),
        "coveredToolCount": len(entries),
        "groups": grouped,
        "tools": entries,
    }


@mcp.tool()
def set_song_identity(song_id: str, artist: str, title: str) -> dict:
    """Manually name a legacy needs-identity song.

    Moves the full version history and every run trace to the new permanent id
    atomically. The target must not already exist; this tool never guesses or
    merges two histories.
    """
    try:
        return rename_song_identity(
            get_store(), get_run_store(), song_id, artist=artist, title=title
        ).to_dict()
    except IdentityUnresolvedError as error:
        return _identity_error(error)


@mcp.tool()
def list_songs_needing_identity() -> dict:
    """List legacy songs with an unresolved id for manual naming."""
    songs = [
        {
            "songId": summary.id,
            "artist": summary.artist,
            "title": summary.title,
            "needsIdentity": True,
        }
        for summary in get_store().list_song_summaries()
        if song_id_has_unknown_segment(summary.id)
    ]
    return {"songs": songs}


# --- per-song reconciliation notes -------------------------------------------
# Free-text "how to build this song" instructions, stored beside the song (never
# in it — the Song document is the app's content contract) and handed to a later
# analyze as reconciler guidance. Mirrors the REST trio (GET/PUT/DELETE
# /v1/songs/{songId}/notes) so the note an analyze would replay is visible,
# settable and clearable from here too: without these, an MCP caller inherits
# whatever an HTTP client happened to leave behind and cannot even see it.
#
# A song's notes have TWO INDEPENDENT lifetimes (store/song_notes.py): a
# durable `preference` (this surface) and a single-shot `correction` (an
# analyze request's `guidance`). Both may be set at once, and an analyze that
# replays or receives guidance while both exist acts on their COMBINATION
# (`song_notes.combine_guidance`). The tools below report them as two
# separate objects rather than one `notes` string, because a flat field could
# not say which lifetime it was reading, or that both were in force —
# exactly the ambiguity that made one slot unable to hold two lifetimes. The
# flat `notes`/`updatedAt` keys are still present, as a deprecated
# compatibility view of `preference` mirroring the REST response.


def _component(doc: dict | None) -> dict | None:
    """One lifetime's sub-record for the wire, or None when it isn't set."""
    if not doc:
        return None
    return {"notes": doc.get("notes", "") or "", "updatedAt": doc.get("updated_at")}


def _notes_doc(song_id: str, record: dict | None) -> dict:
    preference = _component((record or {}).get("preference"))
    correction_raw = (record or {}).get("correction")
    correction = _component(correction_raw)
    if correction is not None:
        correction["appliedToVersion"] = (correction_raw or {}).get("applied_to_version")
    return {
        "songId": song_id,
        # DEPRECATED flat compatibility view of `preference`, mirroring the REST
        # response key-for-key (api.py's `_notes_response`) — one store, two
        # surfaces, and a shape that differs between them is a shape neither can
        # be described by. Read `preference`/`correction` instead.
        "notes": (preference or {}).get("notes", ""),
        "updatedAt": (preference or {}).get("updatedAt"),
        "preference": preference,
        "correction": correction,
    }


@mcp.tool()
def get_song_notes(song_id: str) -> dict:
    """Read a song's stored reconciliation notes — the guidance an analyze that
    passes none would replay. Never an error: "no notes" is a normal answer
    (preference: null, correction: null).

    `preference` is a standing instruction and replays on every later analyze
    that passes none. `correction` came from one
    analyze_and_store_song(guidance=...) call and replays only until its
    `appliedToVersion` names the stored version it landed in — after that it
    is already in the document and is NOT applied again. Both may be set at
    once, in which case both are IN FORCE and a run acts on their combination
    (a labeled "Standing preference: ... / Requested correction: ..." string),
    never one silently standing in for the other — except on a notes-only run,
    which is shown the correction alone (see analyze_and_store_song)."""
    from .store.song_notes import get_song_notes_store

    return _notes_doc(song_id, get_song_notes_store().get_record(song_id))


@mcp.tool()
def set_song_notes(song_id: str, notes: str) -> dict:
    """Store a song's reconciliation notes as a durable PREFERENCE — a standing
    instruction ("capo-free voicings please", "the bridge is Bm, not D") replayed
    as guidance by every later analyze of this song id that passes none, until
    it is changed or cleared. Replaces whatever PREFERENCE was there
    (empty/whitespace clears it, same as clear_song_notes) but never touches a
    pending CORRECTION left behind by an analyze — the two lifetimes are
    stored independently, so writing one never disturbs the other; when both
    are set, a later analyze's guidance is the two combined. Max 8000 chars.

    For a one-off fix to the document you already have ("change the C to a B in
    line 12"), pass it as analyze_and_store_song(guidance=...) instead: that
    applies to that run, survives a run that dies before storing, and then stops
    replaying — a correction that replays forever silently re-applies itself to
    later runs nobody attached it to."""
    from .store.song_notes import get_song_notes_store, length_error

    problem = length_error(notes)
    if problem:
        raise ValueError(problem)
    store = get_song_notes_store()
    store.set_preference(song_id, notes)  # empty/whitespace deletes
    return _notes_doc(song_id, store.get_record(song_id))


@mcp.tool()
def clear_song_notes(song_id: str) -> dict:
    """Delete a song's stored reconciliation notes — BOTH the standing
    preference and any pending correction, so the next analyze that sends no
    guidance replays nothing. There is no partial clear here: a stray pending
    correction is rare (it only outlives one run when that run died before
    storing, and the next successful analyze consumes it on its own), so if
    you only want to keep a standing preference intact, simply don't call
    this. `deleted` is false when there was nothing stored in either slot."""
    from .store.song_notes import get_song_notes_store

    return {"songId": song_id, "deleted": get_song_notes_store().delete(song_id)}


# --- deterministic audio utilities (no AI) -----------------------------------


@mcp.tool()
def convert_audio(
    output_format: str,
    input_path: Optional[str] = None,
    input_ref: Optional[str] = None,
    input_base64: Optional[str] = None,
    input_format: str = "bin",
    return_base64: bool = False,
) -> dict:
    """Convert an audio file between formats (mp3/wav/m4a/flac/ogg/opus) with
    ffmpeg — deterministic, no AI. Prefer input_ref; input_path (trusted local
    callers) and input_base64 (+input_format) remain compatible."""
    with _materialized_tool_input(input_path, input_ref, input_base64, input_format) as src:
        dst = src.parent / f"{src.stem}.converted.{output_format.lstrip('.')}"
        audio_utils.convert(src, dst)
        return _audio_result(dst, return_base64)


@mcp.tool()
def trim_audio(
    start_seconds: float,
    end_seconds: float,
    input_path: Optional[str] = None,
    input_ref: Optional[str] = None,
    input_base64: Optional[str] = None,
    input_format: str = "bin",
    output_format: Optional[str] = None,
    return_base64: bool = False,
) -> dict:
    """Crop an audio file to [start_seconds, end_seconds) with ffmpeg —
    deterministic, exact cut points, no AI."""
    with _materialized_tool_input(input_path, input_ref, input_base64, input_format) as src:
        fmt = (output_format or src.suffix.lstrip(".") or "wav").lstrip(".")
        dst = src.parent / f"{src.stem}.trimmed.{fmt}"
        audio_utils.trim(src, dst, start_seconds, end_seconds)
        return _audio_result(dst, return_base64)


@mcp.tool()
def normalize_audio(
    input_path: Optional[str] = None,
    input_ref: Optional[str] = None,
    input_base64: Optional[str] = None,
    input_format: str = "bin",
    target_lufs: float = -16.0,
    output_format: Optional[str] = None,
    return_base64: bool = False,
) -> dict:
    """EBU R128 loudness-normalize an audio file with ffmpeg — no AI."""
    with _materialized_tool_input(input_path, input_ref, input_base64, input_format) as src:
        fmt = (output_format or src.suffix.lstrip(".") or "wav").lstrip(".")
        dst = src.parent / f"{src.stem}.normalized.{fmt}"
        audio_utils.normalize(src, dst, target_lufs=target_lufs)
        return _audio_result(dst, return_base64)


@mcp.tool()
def probe_audio(
    input_path: Optional[str] = None,
    input_ref: Optional[str] = None,
    input_base64: Optional[str] = None,
    input_format: str = "bin",
) -> dict:
    """Inspect an audio file (duration, codec, sample rate, channels) with
    ffprobe — no AI."""
    with _materialized_tool_input(input_path, input_ref, input_base64, input_format) as src:
        return dataclasses.asdict(audio_utils.probe(src))


# --- meta --------------------------------------------------------------------


@mcp.tool()
def server_status() -> dict:
    """Service version, configured LLM providers (and audio-input capability),
    active MIR engines, and store location."""
    import shutil

    def has(mod: str) -> bool:
        try:
            __import__(mod)
            return True
        except Exception:  # noqa: BLE001
            return False

    from .mir.chordrec import chord_engine_id, chord_model_status

    return {
        "version": __version__,
        "ffmpeg": shutil.which(settings.ffmpeg_bin) is not None,
        "mirEngines": {
            "beats": "madmom" if has("madmom") else "librosa-fallback",
            "chords": chord_engine_id(),
            "structure": "songformer" if settings.songformer_dir else "librosa-agglomerative-fallback",
        },
        "chordModel": chord_model_status(),
        "llmProviders": provider_capabilities(),
        "store": _store_backend_label(),
    }


@mcp.tool()
def get_song_schema() -> dict:
    """The Song JSON schema every produced document conforms to (iOS SongStore
    compatible), including the sounding-harmony chord rule."""
    return song_json_schema()


# --- agent programming & observability ---------------------------------------
# Program the in-process reconciliation agent and inspect its work. Config
# writes are gated by the SAME SNOOCLE_API_TOKEN as everything else (see
# authz.require_admin_token_configured) — a write on an unauthenticated service
# is refused. The Song OUTPUT CONTRACT (strict schema, sounding-pitch chords,
# capo=0) is NOT programmable: it is always enforced by the reconcile engine.


@mcp.tool()
def get_agent_config() -> dict:
    """Read the runtime-editable agent config (instructions, tool budgets,
    effort, model) plus the built-in defaults it overrides. Empty config means
    the agent runs with its built-in behavior."""
    from .reconcile.agent_config import AgentConfig, config_version
    from .store.agent_config import get_agent_config_store

    doc = get_agent_config_store().get()
    cfg = AgentConfig.model_validate(doc) if doc else AgentConfig()
    return {"config": cfg.model_dump(), "configVersion": config_version(cfg),
            "isDefault": cfg.is_default()}


@mcp.tool()
def set_agent_config(config_json: str) -> dict:
    """Program the agent: set instructions_extra / theory_rules /
    retrieval_recipe / instructions_override (all optional), max_turns, effort,
    max_fetch (deterministic sheet-search calls), max_windows, disabled_tools,
    source_site_preferences, and model. The output
    contract and schema enforcement are NOT editable. Requires SNOOCLE_API_TOKEN
    to be configured on the server."""
    import json as _json

    from .authz import require_admin_token_configured
    from .reconcile.agent_config import AgentConfig, config_version
    from .store.agent_config import get_agent_config_store

    require_admin_token_configured()
    cfg = AgentConfig.model_validate(_json.loads(config_json))
    doc = cfg.model_dump()
    doc["updated_at"] = _now_iso()
    doc["source"] = "mcp"
    get_agent_config_store().set(doc)
    return {"status": "stored", "configVersion": config_version(cfg), "updatedAt": doc["updated_at"]}


@mcp.tool()
def reset_agent_config() -> dict:
    """Clear the agent config back to built-in defaults. Requires
    SNOOCLE_API_TOKEN."""
    from .authz import require_admin_token_configured
    from .store.agent_config import get_agent_config_store

    require_admin_token_configured()
    get_agent_config_store().clear()
    return {"status": "reset"}


@mcp.tool()
def list_song_runs(song_id: str, limit: int = 20) -> dict:
    """Recent reconciliation runs for a song, newest first (summaries only —
    fetch a run's steps/MIR with get_run)."""
    from .store.runs import get_run_store

    return {"songId": song_id, "runs": get_run_store().list_runs(song_id, limit=limit)}


@mcp.tool()
def get_run(run_id: str) -> dict:
    """One reconciliation run's full step trace + MIR — the agent's reasoning,
    tool calls, and repair rounds."""
    from .store.runs import fetch_run

    run = fetch_run(run_id)
    if run is None:
        return {"error": f"no such run: {run_id}"}
    return run


@mcp.tool()
def get_usage_summary(window: str = "7d") -> dict:
    """Compact per-day, per-song, and per-model token/cost rollups."""
    from .reconcile.engine import _load_agent_config
    from .store.runs import get_run_store
    from .usage_summary import build_usage_summary

    try:
        return build_usage_summary(
            get_run_store(), window=window, cfg=_load_agent_config()
        )
    except ValueError as e:
        return {"code": "invalid_window", "error": str(e)}


@mcp.tool()
def get_scorecard() -> dict:
    """Score every gold-marked song's current version against its gold — the
    agent's report card (content metrics + aggregate)."""
    from .eval.scorecard import build_scorecard

    return build_scorecard(get_store())


@mcp.tool()
def set_gold_version(song_id: str, version: str) -> dict:
    """Mark one of a song's stored versions as the ground-truth 'gold' the agent
    is scored against."""
    from .store.evals import get_eval_store

    try:
        get_store().get(song_id, version=version)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    get_eval_store().set_gold(song_id, version)
    return {"songId": song_id, "goldVersion": version}


@mcp.tool()
def score_song_version(song_id: str, candidate_version: str | None = None) -> dict:
    """Score a candidate version (default: current) against the song's gold."""
    from .eval import score_song
    from .store.evals import get_eval_store

    gold_version = get_eval_store().get_gold(song_id)
    if not gold_version:
        return {"error": f"no gold version set for {song_id}"}
    try:
        gold = get_store().get(song_id, version=gold_version)
        cand = get_store().get(song_id, version=candidate_version)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return {"songId": song_id, "goldVersion": gold_version,
            "candidateVersion": candidate_version or get_store().current_version(song_id),
            "metrics": score_song(cand, gold)}


# Attach the contract only after the final decorator has populated FastMCP's
# registry. Import fails closed if either side grows without the other.
apply_tool_contract(mcp)


_LOCALHOST_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_LOCALHOST_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def resolve_http_transport(env: dict):
    """Pure resolver for the HTTP transport's bind host + security settings.

    Extracted from main() so the security posture is unit-testable without
    spawning a server. Returns (host: str, port: int, security_settings).

    Bind loopback-only unless a remote-serving mode is explicitly configured.
    The DNS-rebinding Host check alone can't protect a 0.0.0.0 bind — the Host
    header is client-controlled, so a LAN client can send `Host: localhost:
    <port>` to satisfy the localhost allowlist. Binding 127.0.0.1 for a local
    smoke test keeps the port off the LAN entirely; remote serving (Cloud Run
    sets SNOOCLE_MCP_TRUST_PROXY=true and needs 0.0.0.0 for routed traffic)
    opts into the wider bind.

    A non-loopback SNOOCLE_MCP_HOST is a remote-serving intent, so it REQUIRES
    a security mode too (ALLOWED_HOSTS or TRUST_PROXY). Without one it would
    widen the bind while leaving the localhost-only fallback policy in place —
    rejecting real remote clients AND letting any reachable client spoof
    `Host: localhost:<port>` — so it's rejected with a clear error rather than
    silently creating that insecure state.

    Security settings are constructed explicitly in every branch rather than
    mutating FastMCP's default: on mcp 1.10.x that default is None (mutating
    would AttributeError) and its middleware treats None as protection-OFF
    "for backwards compatibility", so relying on it would silently leave a
    local run open. Explicit construction is safe-by-default on every version.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    _LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}

    allowed = [h.strip() for h in env.get("SNOOCLE_MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    trust_proxy = _truthy(env.get("SNOOCLE_MCP_TRUST_PROXY"))
    remote_mode = bool(allowed) or trust_proxy

    explicit_host = env.get("SNOOCLE_MCP_HOST")
    if explicit_host and explicit_host not in _LOOPBACK and not remote_mode:
        raise ValueError(
            f"SNOOCLE_MCP_HOST={explicit_host!r} exposes the MCP server beyond "
            "loopback but no host-security mode is set. Also set "
            "SNOOCLE_MCP_ALLOWED_HOSTS=<host[,host...]> (keeps the DNS-rebinding "
            "check on) or SNOOCLE_MCP_TRUST_PROXY=true (only behind an "
            "authenticating proxy such as Cloud Run IAM)."
        )

    host = explicit_host or ("0.0.0.0" if remote_mode else "127.0.0.1")
    port = int(env.get("PORT", env.get("SNOOCLE_MCP_PORT", "8080")))

    if allowed:
        # Protection ON, scoped STRICTLY to the operator's hosts. This branch
        # binds 0.0.0.0 (remote-reachable), so localhost must NOT be appended:
        # allowing `localhost:*` here would let a LAN client spoof
        # `Host: localhost:<port>` to bypass the allowlist the operator set to
        # narrow access. If a local Host value is genuinely needed, the
        # operator adds it to SNOOCLE_MCP_ALLOWED_HOSTS explicitly. (Cloud
        # Run's startup probe is a TCP socket check — it sends no HTTP Host
        # header, so nothing here depends on the localhost entries.)
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(allowed),
            allowed_origins=[
                *(f"https://{h}" for h in allowed),
                *(f"http://{h}" for h in allowed),
            ],
        )
    elif trust_proxy:
        # Explicit opt-out: only safe behind an authenticating proxy (Cloud Run
        # IAM). The deployed *.run.app hostname is assigned at deploy time so it
        # can't be hardcoded into an allowlist; this is the escape hatch.
        security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    else:
        # Neither configured: loopback bind (above) + protection ON,
        # localhost-only. Defense in depth for a local smoke test.
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=_LOCALHOST_HOSTS,
            allowed_origins=_LOCALHOST_ORIGINS,
        )
    return host, port, security


def main() -> None:
    """Entrypoint for the `snoocle-mcp` console script.

    Defaults to stdio — the standard way an MCP client (Claude Desktop, an
    agent runtime) spawns this as a local subprocess. Set
    SNOOCLE_MCP_TRANSPORT=streamable-http to instead serve as a long-running
    HTTP process (e.g. deployed to Cloud Run as its own service, behind
    Cloud Run IAM auth rather than any app-level auth).

    In HTTP mode the port binds to loopback (127.0.0.1) with the SDK's Host
    (DNS-rebinding) check ON, so a local run is not exposed on the LAN.
    Opt into remote serving with one of:
      * SNOOCLE_MCP_ALLOWED_HOSTS  comma-separated Host values to allow
        (e.g. "snoocle-mcp-xxxx.run.app"); binds 0.0.0.0, check stays on,
        scoped to those hosts. Preferred once the deployed hostname is known.
      * SNOOCLE_MCP_TRUST_PROXY=true  binds 0.0.0.0 and disables the Host
        check — ONLY correct behind an authenticating proxy (Cloud Run IAM).
    """
    import os

    transport = os.environ.get("SNOOCLE_MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run()
        return
    if transport not in ("streamable-http", "sse"):
        raise ValueError(
            f"unsupported SNOOCLE_MCP_TRANSPORT {transport!r} "
            "(expected stdio | streamable-http | sse)"
        )
    host, port, security = resolve_http_transport(dict(os.environ))
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.settings.transport_security = security

    # Stateless HTTP by default (no persistent server->client SSE stream).
    # Required for a Cloud Run --concurrency=1 deployment: otherwise the MCP
    # client's long-lived GET SSE stream (opened after initialize) occupies the
    # single request slot and every subsequent tool-call POST queues behind it
    # until it times out — a deadlock. This tool server issues no
    # server-initiated notifications, so statelessness costs nothing here.
    # Opt out (restore the stateful session + SSE stream) with
    # SNOOCLE_MCP_STATELESS=false — then the service needs concurrency >= 2.
    if transport == "streamable-http" and _truthy(os.environ.get("SNOOCLE_MCP_STATELESS", "true")):
        mcp.settings.stateless_http = True
        mcp.settings.json_response = True
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
