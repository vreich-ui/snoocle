"""Exhaustive MCP tool classification and GUI presentation metadata.

The protocol has standard risk hints, but it deliberately has no vocabulary for
Snoocle artifacts, browser exposure, expected duration, or result renderers.
This module keeps that small extension in one namespaced ``_meta`` value and
refuses to start when the registered surface and the contract drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from mcp.types import ToolAnnotations

TOOL_CONTRACT_VERSION = 1
TOOL_META_KEY = "snoocle/toolContract"

BrowserSafety = Literal["safe", "confirmation_required", "server_filesystem_restricted"]
ExpectedDuration = Literal["instant", "seconds", "minutes"]
ModelUse = Literal["none", "conditional", "required"]

_BROWSER_SAFETY = {"safe", "confirmation_required", "server_filesystem_restricted"}
_DURATIONS = {"instant", "seconds", "minutes"}
_MODEL_USE = {"none", "conditional", "required"}


class ToolContractError(RuntimeError):
    """The exposed MCP registry and Snoocle's presentation contract disagree."""


@dataclass(frozen=True)
class ToolContract:
    title: str
    category: str
    browser_safety: BrowserSafety
    input_artifact_kinds: tuple[str, ...]
    output_artifact_kinds: tuple[str, ...]
    read_only: bool
    destructive: bool
    idempotent: bool
    network_access: tuple[str, ...]
    model_use: ModelUse
    persistence: tuple[str, ...]
    cache_behavior: Literal["none", "read", "read_write"]
    expected_duration: ExpectedDuration
    specialized_renderer: str

    def __post_init__(self) -> None:
        if not self.title or not self.category:
            raise ValueError("tool title and category are required")
        if self.browser_safety not in _BROWSER_SAFETY:
            raise ValueError(f"unsupported browser safety: {self.browser_safety}")
        if self.expected_duration not in _DURATIONS:
            raise ValueError(f"unsupported expected duration: {self.expected_duration}")
        if self.model_use not in _MODEL_USE:
            raise ValueError(f"unsupported model use: {self.model_use}")
        if not self.input_artifact_kinds or not self.output_artifact_kinds:
            raise ValueError("input and output artifact kinds are required")
        if self.read_only and self.destructive:
            raise ValueError("a read-only tool cannot be destructive")

    @property
    def access_mode(self) -> str:
        return "read" if self.read_only else "write"

    @property
    def open_world(self) -> bool:
        return any(kind.startswith("external:") for kind in self.network_access)

    @property
    def execution(self) -> str:
        if self.model_use == "required":
            return "model-backed"
        if self.model_use == "conditional":
            return (
                "deterministic-first; model-backed only by explicit policy "
                "or actionable MODEL conflict"
            )
        return "deterministic"

    @property
    def cost_class(self) -> str:
        return {"none": "none", "conditional": "conditional-model", "required": "model"}[
            self.model_use
        ]

    def annotations(self) -> ToolAnnotations:
        return ToolAnnotations(
            title=self.title,
            readOnlyHint=self.read_only,
            destructiveHint=self.destructive,
            idempotentHint=self.idempotent,
            openWorldHint=self.open_world,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "schemaVersion": TOOL_CONTRACT_VERSION,
            "category": self.category,
            "browserSafety": self.browser_safety,
            "inputArtifactKinds": list(self.input_artifact_kinds),
            "outputArtifactKinds": list(self.output_artifact_kinds),
            "access": {
                "mode": self.access_mode,
                "readOnly": self.read_only,
                "destructive": self.destructive,
                "idempotent": self.idempotent,
            },
            "networkAccess": list(self.network_access),
            "modelUse": self.model_use,
            "persistence": list(self.persistence),
            "cacheBehavior": self.cache_behavior,
            "expectedDuration": self.expected_duration,
            "specializedRenderer": self.specialized_renderer,
        }


def _title(name: str) -> str:
    words = name.replace("_", " ").title()
    for source, display in (("Mir", "MIR"), ("Lrc", "LRC"), ("Json", "JSON")):
        words = words.replace(source, display)
    return words


def _contract(
    name: str,
    category: str,
    browser_safety: BrowserSafety,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    *,
    read_only: bool = True,
    destructive: bool = False,
    idempotent: bool = True,
    network: tuple[str, ...] = (),
    model_use: ModelUse = "none",
    persistence: tuple[str, ...] = (),
    cache: Literal["none", "read", "read_write"] = "none",
    duration: ExpectedDuration = "instant",
    renderer: str = "json",
) -> ToolContract:
    return ToolContract(
        title=_title(name),
        category=category,
        browser_safety=browser_safety,
        input_artifact_kinds=inputs,
        output_artifact_kinds=outputs,
        read_only=read_only,
        destructive=destructive,
        idempotent=idempotent,
        network_access=network,
        model_use=model_use,
        persistence=persistence,
        cache_behavior=cache,
        expected_duration=duration,
        specialized_renderer=renderer,
    )


# This is intentionally exhaustive instead of having a default. A new
# ``@mcp.tool`` must make an explicit product/safety decision here before the
# server can import or CI can pass.
TOOL_CONTRACTS: dict[str, ToolContract] = {
    "parse_candidate_text": _contract(
        "parse_candidate_text", "parsing", "safe", ("chord_sheet_text",),
        ("candidate_source",), renderer="candidate",
    ),
    "score_candidate_against_mir": _contract(
        "score_candidate_against_mir", "baseline", "safe",
        ("candidate_source", "mir_analysis"), ("candidate_score",),
        renderer="candidate_ranking",
    ),
    "rank_candidates_deterministically": _contract(
        "rank_candidates_deterministically", "baseline", "safe",
        ("candidate_collection", "mir_analysis"), ("candidate_ranking",),
        renderer="candidate_ranking",
    ),
    "select_candidate_deterministically": _contract(
        "select_candidate_deterministically", "baseline", "safe",
        ("candidate_collection", "mir_analysis"), ("candidate_selection",),
        renderer="candidate_ranking",
    ),
    "build_song_baseline": _contract(
        "build_song_baseline", "baseline", "safe",
        ("candidate_source", "song_identity"), ("song",), renderer="song",
    ),
    "validate_song_json": _contract(
        "validate_song_json", "parsing", "safe", ("song",), ("song_validation", "song"),
        renderer="song",
    ),
    "analyze_full_track_mir": _contract(
        "analyze_full_track_mir", "MIR", "server_filesystem_restricted",
        ("audio_artifact", "audio_file"),
        ("mir_analysis",), network=("artifact_backend",),
        duration="minutes", renderer="mir",
    ),
    "analyze_mir_window": _contract(
        "analyze_mir_window", "MIR", "server_filesystem_restricted",
        ("audio_artifact", "audio_file"),
        ("mir_analysis",), network=("artifact_backend",),
        duration="seconds", renderer="mir",
    ),
    "extend_mir_beat_grid": _contract(
        "extend_mir_beat_grid", "MIR", "safe", ("beat_grid",), ("beat_grid",),
        renderer="mir",
    ),
    "snap_song_to_mir": _contract(
        "snap_song_to_mir", "alignment", "safe", ("song", "mir_analysis"), ("song",),
        renderer="song",
    ),
    "carry_forward_song_timing": _contract(
        "carry_forward_song_timing", "alignment", "safe", ("song", "song_version"),
        ("song", "timing_report"), renderer="song",
    ),
    "lookup_lrc": _contract(
        "lookup_lrc", "source retrieval", "safe", ("song_identity",), ("lrc",),
        network=("external:lrclib",), duration="seconds", renderer="lrc",
    ),
    "match_lrc_to_song": _contract(
        "match_lrc_to_song", "parsing", "safe", ("lrc", "song"), ("lrc_matches",),
        renderer="lrc",
    ),
    "apply_lrc_to_song": _contract(
        "apply_lrc_to_song", "alignment", "safe", ("song", "lrc_matches", "mir_analysis"),
        ("song",), renderer="song",
    ),
    "retime_song_sections": _contract(
        "retime_song_sections", "alignment", "safe", ("song",), ("song", "timing_report"),
        renderer="song",
    ),
    "guard_song_timing_collapse": _contract(
        "guard_song_timing_collapse", "alignment", "safe", ("song",),
        ("song", "timing_report"), renderer="song",
    ),
    "score_song_confidence": _contract(
        "score_song_confidence", "alignment", "safe", ("song", "candidate_source", "mir_analysis"),
        ("song", "confidence_report", "review_queue"), renderer="quality_report",
    ),
    "evaluate_song_quality": _contract(
        "evaluate_song_quality", "quality", "safe", ("song", "candidate_source", "mir_analysis"),
        ("quality_report",), renderer="quality_report",
    ),
    "validate_song_theory": _contract(
        "validate_song_theory", "quality", "safe", ("song",), ("theory_report",),
        duration="seconds", renderer="quality_report",
    ),
    "calculate_recording_offset": _contract(
        "calculate_recording_offset", "audio", "server_filesystem_restricted",
        ("audio_artifact", "audio_file"), ("recording_offset",),
        network=("artifact_backend",), duration="seconds",
        renderer="recording_offset",
    ),
    "apply_deterministic_song_patch": _contract(
        "apply_deterministic_song_patch", "alignment", "safe", ("song", "song_patch"),
        ("song", "applied_patch"), renderer="song",
    ),
    "build_song_evidence_manifest": _contract(
        "build_song_evidence_manifest", "quality", "safe",
        ("candidate_collection", "mir_analysis", "song"), ("evidence_manifest",),
        renderer="evidence_manifest",
    ),
    "align_song_deterministically": _contract(
        "align_song_deterministically", "alignment", "server_filesystem_restricted",
        ("song", "audio_artifact", "audio_file", "recording", "mir_analysis", "candidate_collection", "lrc"),
        ("song", "run_trace", "quality_report"), read_only=False, idempotent=False,
        network=("external:youtube", "external:lrclib", "store_backend", "artifact_backend"),
        persistence=("run_trace_write", "song_version_optional", "cache"),
        cache="read_write", duration="minutes", renderer="song",
    ),
    "process_song_deterministically": _contract(
        "process_song_deterministically", "alignment", "server_filesystem_restricted",
        ("song_identity", "audio_artifact", "audio_file", "recording", "mir_analysis", "candidate_collection", "lrc"),
        ("song", "run_trace", "quality_report"), read_only=False, idempotent=False,
        network=("external:web", "external:youtube", "external:lrclib", "store_backend", "artifact_backend"),
        persistence=("run_trace_write", "song_version_optional", "cache"),
        cache="read_write", duration="minutes", renderer="song",
    ),
    "discover_song": _contract(
        "discover_song", "source retrieval", "safe", ("song_identity",),
        ("candidate_collection",), network=("external:web",), persistence=("cache",),
        cache="read_write", duration="seconds", renderer="candidate_list",
    ),
    "acquire_audio": _contract(
        "acquire_audio", "audio", "confirmation_required", ("song_identity", "recording"),
        ("audio_artifact", "recording"), read_only=False, idempotent=False,
        network=("external:youtube", "artifact_backend"),
        persistence=("cache", "temporary_artifact"),
        cache="read_write", duration="minutes", renderer="audio",
    ),
    "analyze_audio": _contract(
        "analyze_audio", "MIR", "server_filesystem_restricted",
        ("audio_artifact", "audio_file", "song_identity", "recording"),
        ("mir_analysis", "audio_artifact"),
        read_only=False, idempotent=False,
        network=("external:youtube", "artifact_backend"),
        persistence=("cache", "temporary_artifact", "ephemeral_filesystem"), cache="read_write",
        duration="minutes", renderer="mir",
    ),
    "reconcile_song": _contract(
        "reconcile_song", "agent reconciliation", "server_filesystem_restricted",
        ("song_identity", "candidate_collection", "mir_analysis", "audio_artifact", "audio_file", "song"),
        ("song", "run_trace", "evidence_manifest"), read_only=False, idempotent=False,
        network=("external:web", "external:model_provider", "store_backend", "artifact_backend"),
        model_use="required",
        persistence=("run_trace_write", "run_admission_write", "song_store_read"),
        duration="minutes", renderer="song",
    ),
    "analyze_and_store_song": _contract(
        "analyze_and_store_song", "agent reconciliation", "confirmation_required",
        ("song_identity", "recording", "analysis_guidance", "song"),
        ("song", "song_version", "run_trace", "quality_report"), read_only=False,
        destructive=True, idempotent=False,
        network=("external:web", "external:youtube", "external:model_provider", "store_backend"),
        model_use="conditional",
        persistence=("song_version_write", "run_trace_write", "notes_write", "cache"),
        cache="read_write", duration="minutes", renderer="song",
    ),
    "realign_song_to_recording": _contract(
        "realign_song_to_recording", "alignment", "confirmation_required", ("song", "recording"),
        ("song", "song_version", "run_trace", "quality_report"), read_only=False,
        idempotent=False,
        network=("external:youtube", "external:model_provider", "store_backend"),
        model_use="conditional", persistence=("song_version_write", "run_trace_write", "cache"),
        cache="read_write", duration="minutes", renderer="song",
    ),
    "suggest_better_recordings": _contract(
        "suggest_better_recordings", "audio", "safe", ("song",), ("recording_collection",),
        network=("external:youtube", "store_backend"), persistence=("song_store_read",),
        duration="seconds", renderer="recording_list",
    ),
    "list_songs": _contract(
        "list_songs", "storage", "safe", ("song_store",), ("song_collection",),
        network=("store_backend",), persistence=("song_store_read",), renderer="song_list",
    ),
    "get_song": _contract(
        "get_song", "storage", "safe", ("song_id", "song_version"), ("song",),
        network=("store_backend",), persistence=("song_store_read",), renderer="song",
    ),
    "list_song_versions": _contract(
        "list_song_versions", "storage", "safe", ("song_id",), ("song_version_collection",),
        network=("store_backend",), persistence=("song_store_read",), renderer="version_list",
    ),
    "diff_song_versions": _contract(
        "diff_song_versions", "storage", "safe", ("song_version", "song_version"),
        ("song_diff",), network=("store_backend",), persistence=("song_store_read",),
        renderer="diff",
    ),
    "save_song": _contract(
        "save_song", "storage", "confirmation_required", ("song",), ("song_version",),
        read_only=False, network=("store_backend",), persistence=("song_version_write",),
        renderer="song_version",
    ),
    "diagnose_mock_songs": _contract(
        "diagnose_mock_songs", "diagnostics", "safe", ("song_store",),
        ("diagnostic_report",), network=("store_backend",), persistence=("song_store_read",),
        renderer="diagnostic_report",
    ),
    "list_capabilities": _contract(
        "list_capabilities", "diagnostics", "safe", ("tool_registry",),
        ("tool_catalog",), renderer="tool_catalog",
    ),
    "set_song_identity": _contract(
        "set_song_identity", "identity", "confirmation_required", ("song_id", "song_identity"),
        ("identity_migration",), read_only=False, destructive=True,
        network=("store_backend",), persistence=("song_store_write", "run_trace_write"),
        renderer="identity",
    ),
    "list_songs_needing_identity": _contract(
        "list_songs_needing_identity", "identity", "safe", ("song_store",),
        ("song_collection",), network=("store_backend",), persistence=("song_store_read",),
        renderer="song_list",
    ),
    "get_song_notes": _contract(
        "get_song_notes", "storage", "safe", ("song_id",), ("song_notes",),
        network=("store_backend",), persistence=("notes_read",), renderer="song_notes",
    ),
    "set_song_notes": _contract(
        "set_song_notes", "storage", "confirmation_required", ("song_id", "song_notes"),
        ("song_notes",), read_only=False, destructive=True, network=("store_backend",),
        persistence=("notes_write",), renderer="song_notes",
    ),
    "clear_song_notes": _contract(
        "clear_song_notes", "storage", "confirmation_required", ("song_id",),
        ("deletion_result",), read_only=False, destructive=True, network=("store_backend",),
        persistence=("notes_write",), renderer="song_notes",
    ),
    "convert_audio": _contract(
        "convert_audio", "audio", "server_filesystem_restricted", ("audio_artifact", "audio_file"),
        ("audio_artifact",), read_only=False, destructive=True,
        network=("artifact_backend",), persistence=("temporary_artifact",),
        duration="seconds", renderer="audio",
    ),
    "trim_audio": _contract(
        "trim_audio", "audio", "server_filesystem_restricted", ("audio_artifact", "audio_file"),
        ("audio_artifact",), read_only=False, destructive=True,
        network=("artifact_backend",), persistence=("temporary_artifact",),
        duration="seconds", renderer="audio",
    ),
    "normalize_audio": _contract(
        "normalize_audio", "audio", "server_filesystem_restricted", ("audio_artifact", "audio_file"),
        ("audio_artifact",), read_only=False, destructive=True,
        network=("artifact_backend",), persistence=("temporary_artifact",),
        duration="seconds", renderer="audio",
    ),
    "probe_audio": _contract(
        "probe_audio", "audio", "server_filesystem_restricted",
        ("audio_artifact", "audio_file"), ("audio_probe",),
        network=("artifact_backend",), persistence=("ephemeral_filesystem",),
        renderer="audio_probe",
    ),
    "server_status": _contract(
        "server_status", "diagnostics", "safe", ("service",), ("service_status",),
        renderer="status",
    ),
    "get_song_schema": _contract(
        "get_song_schema", "parsing", "safe", ("schema_request",), ("json_schema",),
        renderer="json_schema",
    ),
    "get_agent_config": _contract(
        "get_agent_config", "agent reconciliation", "safe", ("agent_config_store",),
        ("agent_config",), network=("store_backend",), persistence=("agent_config_read",),
        renderer="agent_config",
    ),
    "set_agent_config": _contract(
        "set_agent_config", "agent reconciliation", "confirmation_required", ("agent_config",),
        ("agent_config_version",), read_only=False, destructive=True,
        network=("store_backend",), persistence=("agent_config_write",),
        renderer="agent_config",
    ),
    "reset_agent_config": _contract(
        "reset_agent_config", "agent reconciliation", "confirmation_required", ("agent_config_store",),
        ("reset_result",), read_only=False, destructive=True, network=("store_backend",),
        persistence=("agent_config_write",), renderer="agent_config",
    ),
    "list_song_runs": _contract(
        "list_song_runs", "diagnostics", "safe", ("song_id",), ("run_trace_collection",),
        network=("store_backend",), persistence=("run_trace_read",), renderer="run_list",
    ),
    "get_run": _contract(
        "get_run", "diagnostics", "safe", ("run_id",), ("run_trace",),
        network=("store_backend",), persistence=("run_trace_read",), renderer="run_trace",
    ),
    "get_usage_summary": _contract(
        "get_usage_summary", "diagnostics", "safe", ("usage_window",), ("usage_summary",),
        network=("store_backend",), persistence=("run_trace_read", "agent_config_read"),
        renderer="usage_summary",
    ),
    "get_scorecard": _contract(
        "get_scorecard", "diagnostics", "safe", ("song_store", "evaluation_store"),
        ("scorecard",), network=("store_backend",),
        persistence=("song_store_read", "evaluation_read"), duration="seconds",
        renderer="scorecard",
    ),
    "set_gold_version": _contract(
        "set_gold_version", "storage", "confirmation_required", ("song_version",),
        ("gold_version",), read_only=False, destructive=True, network=("store_backend",),
        persistence=("song_store_read", "evaluation_write"), renderer="scorecard",
    ),
    "score_song_version": _contract(
        "score_song_version", "diagnostics", "safe", ("song_version", "gold_version"),
        ("score_report",), network=("store_backend",),
        persistence=("song_store_read", "evaluation_read"), duration="seconds",
        renderer="scorecard",
    ),
}


def _registered_tools(mcp: Any) -> dict[str, Any]:
    manager = getattr(mcp, "_tool_manager", None)
    registered = getattr(manager, "_tools", None)
    if not isinstance(registered, dict):
        raise ToolContractError("FastMCP registered-tool inventory is unavailable")
    return registered


def validate_registered_tool_contract(mcp: Any) -> None:
    registered_names = set(_registered_tools(mcp))
    contract_names = set(TOOL_CONTRACTS)
    unclassified = sorted(registered_names - contract_names)
    stale = sorted(contract_names - registered_names)
    if unclassified or stale:
        details = []
        if unclassified:
            details.append(f"unclassified registered tools: {', '.join(unclassified)}")
        if stale:
            details.append(f"contract entries without registered tools: {', '.join(stale)}")
        raise ToolContractError("; ".join(details))


def apply_tool_contract(mcp: Any) -> None:
    """Validate the complete surface, then attach standard and Snoocle metadata."""
    validate_registered_tool_contract(mcp)
    for name, tool in _registered_tools(mcp).items():
        contract = TOOL_CONTRACTS[name]
        tool.title = contract.title
        tool.annotations = contract.annotations()
        # FastMCP 1.10 (the declared dependency floor) supports titles and all
        # standard annotations, but its internal Tool model predates MCP tool
        # ``_meta``. Newer 1.x releases expose ``meta``. Keep the standard hints
        # on every supported release and publish the extension on tools/list
        # only where the SDK can represent it; list_capabilities always carries
        # the same complete contract.
        if "meta" in getattr(type(tool), "model_fields", {}):
            tool.meta = {**(tool.meta or {}), TOOL_META_KEY: contract.to_wire()}


def registered_tool_contracts(mcp: Any) -> list[tuple[str, Any, ToolContract]]:
    """Return registered tools plus classifications in stable display order."""
    validate_registered_tool_contract(mcp)
    return [
        (name, tool, TOOL_CONTRACTS[name])
        for name, tool in sorted(_registered_tools(mcp).items())
    ]
