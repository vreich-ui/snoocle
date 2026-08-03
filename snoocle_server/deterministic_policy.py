"""Deterministic-first production policy and the one bounded agent patch.

This module is intentionally smaller than the legacy reconciliation engine.
It accepts only the deterministic core's compact conflict packet, applies the
closed patch vocabulary locally, and reruns the deterministic post-passes.
It never asks a model to regenerate a Song.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Literal

from .deterministic import (
    DeterministicAlignmentResult,
    StageObservation,
    align_song_deterministically_service,
)
from .deterministic_process import DeterministicProcessResult
from .quality.attribution import Fault
from .reconcile.engine import extract_json
from .reconcile.patch_ops import apply_patch, parse_ops_response
from .reconcile.providers import ProviderError, get_provider, provider_preflight
from .usage import cost_usd, persisted_usage


AgentPolicy = Literal["never", "unresolved_only", "always"]
DEFAULT_AGENT_POLICY: AgentPolicy = "unresolved_only"
MAX_CONFLICT_PACKET_BYTES = 64_000
MAX_CONFLICTS = 50


class AgentPolicyError(ValueError):
    """A policy value or bounded-patch contract was rejected."""


@dataclass(frozen=True)
class AgentPatchOutcome:
    alignment: DeterministicAlignmentResult
    provider: str
    model: str
    usage: dict[str, int]
    cost_usd: float
    applied_operations: tuple[dict[str, Any], ...]
    observation: StageObservation

    def to_dict(self) -> dict[str, Any]:
        return {
            "invoked": True,
            "provider": self.provider,
            "model": self.model,
            "modelCalls": 1,
            "modelCostUSD": self.cost_usd,
            "usage": persisted_usage(self.usage),
            "appliedOperations": list(self.applied_operations),
        }


def resolve_agent_policy(value: str | None) -> AgentPolicy:
    resolved = (value or DEFAULT_AGENT_POLICY).strip().lower()
    if resolved not in {"never", "unresolved_only", "always"}:
        raise AgentPolicyError(
            "agent_policy must be one of: never, unresolved_only, always"
        )
    return resolved  # type: ignore[return-value]


def validate_compact_conflict_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Reject anything beyond the deterministic core's lyric-free contract."""
    encoded = json.dumps(packet, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if len(encoded.encode("utf-8")) > MAX_CONFLICT_PACKET_BYTES:
        raise AgentPolicyError(
            f"conflict packet exceeds {MAX_CONFLICT_PACKET_BYTES} bytes"
        )
    allowed_top = {
        "songId",
        "recordingId",
        "documentVersion",
        "musicalContext",
        "conflicts",
        "allowedOperations",
    }
    unexpected = sorted(set(packet) - allowed_top)
    if unexpected:
        raise AgentPolicyError(
            "conflict packet contains forbidden top-level fields: "
            + ", ".join(unexpected)
        )
    conflicts = packet.get("conflicts")
    if not isinstance(conflicts, list):
        raise AgentPolicyError("conflict packet conflicts must be an array")
    if len(conflicts) > MAX_CONFLICTS:
        raise AgentPolicyError(
            f"conflict packet has {len(conflicts)} conflicts; limit is {MAX_CONFLICTS}"
        )
    forbidden_key_fragments = (
        "lyrics",
        "songjson",
        "beatgrid",
        "provenance",
        "schema",
        "sourceurl",
    )

    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                normalized = str(key).replace("_", "").lower()
                if any(fragment in normalized for fragment in forbidden_key_fragments):
                    raise AgentPolicyError(
                        f"conflict packet contains forbidden field {key!r}"
                    )
                inspect(nested)
        elif isinstance(value, list):
            for nested in value:
                inspect(nested)
        elif isinstance(value, str) and len(value) > 2_000:
            raise AgentPolicyError("conflict packet contains an oversized string")

    inspect(packet)
    return packet


def model_conflict_is_actionable(result: DeterministicProcessResult) -> bool:
    alignment = result.alignment
    return bool(
        alignment is not None
        and alignment.quality.attribution.fault is Fault.MODEL
        and alignment.quality.attribution.actionable
        and alignment.conflict_packet.get("conflicts")
    )


def run_bounded_agent_patch(
    result: DeterministicProcessResult,
    *,
    provider_name: str | None,
    model: str | None = None,
) -> AgentPatchOutcome:
    """Make exactly one compact model call, apply ops locally, then re-grade."""
    if not model_conflict_is_actionable(result) or result.alignment is None:
        raise AgentPolicyError(
            "bounded agent patch requires an actionable MODEL conflict"
        )
    problem = provider_preflight(provider_name)
    if problem:
        raise ProviderError(problem)
    provider = get_provider(provider_name)
    if provider.name == "mock":
        raise AgentPolicyError("provider=mock cannot perform a production agent patch")
    if provider.name == "anthropic-agent":
        raise AgentPolicyError(
            "anthropic-agent does not implement the compact conflict-patch contract"
        )

    packet = validate_compact_conflict_packet(result.alignment.conflict_packet)
    system = (
        "Return exactly one JSON object with an ops array. Use only an operation "
        "listed in conflictPacket.allowedOperations. Resolve only the listed "
        "conflicts. Do not emit lyrics, a Song document, beat grids, provenance, "
        "schemas, source URLs, markdown, or commentary."
    )
    turn = json.dumps({"conflictPacket": packet}, separators=(",", ":"))
    if getattr(provider, "wants_context", False):
        provider.context = {"compact_conflict_packet": packet}

    started = time.perf_counter()
    response = provider.complete(
        system,
        [{"role": "user", "text": turn}],
        model=model,
        max_tokens=2_048,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    try:
        document = json.loads(extract_json(response.text))
        ops = parse_ops_response(document)
        patched, applied = apply_patch(result.alignment.song, ops)
    except Exception as error:  # validation errors must remain structured
        raise AgentPolicyError(f"bounded agent patch was rejected: {error}") from error

    rerun = align_song_deterministically_service(
        patched,
        result.mir,
        candidates=result.candidates,
        use_lrc=False,
    )
    usage = dict(response.usage or {})
    patch_cost = cost_usd(usage, response.model)
    observation = StageObservation(
        name="bounded_agent_patch",
        elapsed_ms=elapsed_ms,
        input_summary={
            "conflicts": len(packet["conflicts"]),
            "packetBytes": len(turn.encode("utf-8")),
        },
        output_summary={
            "operationsApplied": len(applied),
            "qualityVerdict": rerun.quality.grade.verdict,
            "fault": rerun.quality.attribution.fault.value,
            "modelCalls": 1,
            "modelCostUSD": patch_cost,
        },
        model_calls=1,
        model_cost_usd=patch_cost,
    )
    rerun.observations = [*result.observations, observation, *rerun.observations]
    return AgentPatchOutcome(
        alignment=rerun,
        provider=response.provider,
        model=response.model,
        usage=usage,
        cost_usd=patch_cost,
        applied_operations=tuple(
            {
                "index": item.index,
                "op": item.op_type,
                "description": item.description,
                "reason": item.reason,
            }
            for item in applied
        ),
        observation=observation,
    )
