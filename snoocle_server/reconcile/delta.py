"""Compact, deterministic reconciliation deltas for an existing Song.

The normal reconciliation path used to make the model repeat the entire prior
Song even when only a handful of chord placements changed.  This module is the
small model-facing write contract for that case.  It is deliberately narrower
than JSON Patch: only fields reconciliation owns may change, and every update
is applied to a validated prior Song before the ordinary timing and schema
post-passes run.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ..schema import Song
from .lyric_refs import (
    MAX_OVERRIDE_SHARE,
    LyricOverride,
    _check_placements,
    _resolve_line_text,
    UnresolvableLyricRefError,
)


class ReconcileDeltaError(ValueError):
    """A model-produced delta is malformed or cannot apply to its prior Song."""


@dataclass(frozen=True)
class AppliedDelta:
    song: Song
    lyric_refs_resolved: int = 0
    lyric_overrides: tuple[LyricOverride, ...] = ()
    patch_bytes: int = 0
    full_bytes: int = 0

    @property
    def ratio(self) -> float:
        return self.patch_bytes / self.full_bytes if self.full_bytes else 0.0


def reconcile_delta_json_schema() -> dict:
    """The compact domain-delta schema handed to model-backed providers."""
    placement = {
        "type": "object",
        "additionalProperties": False,
        "required": ["charIndex", "chord"],
        "properties": {
            "charIndex": {"type": "integer", "minimum": 0},
            "chord": {"type": "string"},
            "voicingHint": {"type": ["string", "null"]},
        },
    }
    lyric_ref = {
        "type": "object",
        "additionalProperties": False,
        "required": ["sourceId", "line"],
        "properties": {
            "sourceId": {"type": "string"},
            "line": {"type": "integer", "minimum": 0},
        },
    }
    line_change = {
        "type": "object",
        "additionalProperties": False,
        "required": ["lineIndex"],
        "properties": {
            "lineIndex": {"type": "integer", "minimum": 0},
            "chordPlacements": {"type": "array", "items": placement},
            "lyricRef": lyric_ref,
            "lyricOverride": {"type": "string"},
            "lyricOverrideReason": {"type": "string"},
            "lyrics": {"type": "string", "const": ""},
        },
    }
    section = {
        "type": "object",
        "additionalProperties": False,
        "required": ["sectionIndex", "name", "kind", "startLineIndex", "endLineIndex"],
        "properties": {
            "sectionIndex": {"type": "integer", "minimum": 0},
            "name": {"type": "string"},
            "kind": {
                "type": "string",
                "enum": [
                    "intro", "verse", "prechorus", "chorus", "bridge",
                    "solo", "interlude", "breakdown", "outro", "other",
                ],
            },
            "startLineIndex": {"type": "integer", "minimum": 0},
            "endLineIndex": {"type": "integer", "minimum": 0},
            "startTime": {"type": ["number", "null"], "minimum": 0},
            "endTime": {"type": ["number", "null"], "minimum": 0},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["lineChanges"],
        "properties": {
            "lineChanges": {"type": "array", "items": line_change},
            "sections": {"type": "array", "items": section},
            "metadata": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "album": {"type": ["string", "null"]},
                    "year": {"type": ["integer", "null"]},
                    "key": {"type": ["string", "null"]},
                    "timeSignature": {"type": ["string", "null"]},
                },
            },
            "displayPreferences": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "capo": {"type": "integer", "const": 0},
                    "tuning": {"type": "string"},
                },
            },
        },
    }


def strip_postpass_schema(base_schema: dict, *, mir_present: bool) -> dict:
    """Remove fields a deterministic MIR post-pass will overwrite.

    The real Song schema remains unchanged.  This is only the schema the model
    writes on a first reconcile, and only when MIR exists to refill the omitted
    values.
    """
    schema = copy.deepcopy(base_schema)
    if not mir_present:
        return schema

    def remove(definition: str, *names: str) -> None:
        node = (schema.get("$defs") or {}).get(definition) or {}
        props = node.get("properties") or {}
        for name in names:
            props.pop(name, None)
        if "required" in node:
            node["required"] = [name for name in node["required"] if name not in names]

    remove("ChordPlacement", "timeSeconds", "confidence", "beat")
    remove("Line", "timeSeconds", "confidence")
    remove("AudioInfo", "syncMap", "beats")
    remove("SongMetadata", "bpm")
    return schema


def _canonical_size(value: Any) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def apply_reconcile_delta(
    prior: Song,
    raw_delta: Any,
    ref_index: dict[str, list[str]],
) -> AppliedDelta:
    """Apply one compact model delta and return a freshly validated Song."""
    if not isinstance(raw_delta, dict):
        raise ReconcileDeltaError("reconciliation patch must be a JSON object")
    allowed_top = {"lineChanges", "sections", "metadata", "displayPreferences"}
    unknown = set(raw_delta) - allowed_top
    if unknown:
        raise ReconcileDeltaError(f"unknown reconciliation patch field(s): {sorted(unknown)}")
    changes = raw_delta.get("lineChanges")
    if not isinstance(changes, list):
        raise ReconcileDeltaError("lineChanges must be an array (use [] when no lines change)")

    document = prior.model_dump(mode="json")
    positions = {line["lineIndex"]: pos for pos, line in enumerate(document["lines"])}
    seen: set[int] = set()
    overrides: list[LyricOverride] = []
    refs_resolved = 0
    allowed_line = {
        "lineIndex", "chordPlacements", "lyricRef", "lyricOverride",
        "lyricOverrideReason", "lyrics",
    }
    lyric_keys = {"lyricRef", "lyricOverride", "lyricOverrideReason", "lyrics"}
    forbidden_placement = {"timeSeconds", "confidence", "beat"}

    for change in changes:
        if not isinstance(change, dict):
            raise ReconcileDeltaError("each lineChanges entry must be an object")
        extra = set(change) - allowed_line
        if extra:
            raise ReconcileDeltaError(f"unknown line change field(s): {sorted(extra)}")
        line_index = change.get("lineIndex")
        if not isinstance(line_index, int) or isinstance(line_index, bool):
            raise ReconcileDeltaError("lineChanges.lineIndex must be an integer")
        if line_index in seen:
            raise ReconcileDeltaError(f"line {line_index} is changed more than once")
        seen.add(line_index)
        if line_index not in positions:
            raise ReconcileDeltaError(f"line {line_index} does not exist in the prior Song")

        current = dict(document["lines"][positions[line_index]])
        if "chordPlacements" in change:
            placements = change["chordPlacements"]
            if not isinstance(placements, list):
                raise ReconcileDeltaError(f"line {line_index}: chordPlacements must be an array")
            for placement in placements:
                if not isinstance(placement, dict):
                    raise ReconcileDeltaError(f"line {line_index}: every placement must be an object")
                forbidden = forbidden_placement & set(placement)
                if forbidden:
                    raise ReconcileDeltaError(
                        f"line {line_index}: timing fields are post-pass-owned and forbidden: "
                        f"{sorted(forbidden)}"
                    )
            current["chordPlacements"] = placements

        present_lyric_keys = lyric_keys & set(change)
        if present_lyric_keys:
            protocol_line = {k: change[k] for k in present_lyric_keys}
            protocol_line["lineIndex"] = line_index
            protocol_line["chordPlacements"] = current.get("chordPlacements") or []
            try:
                text, override = _resolve_line_text(protocol_line, line_index, ref_index)
                _check_placements(protocol_line, line_index, text, "reconciliation delta")
            except UnresolvableLyricRefError:
                raise
            except Exception as error:
                raise ReconcileDeltaError(str(error)) from error
            current["lyrics"] = text
            if override is not None:
                overrides.append(override)
            elif "lyricRef" in change:
                refs_resolved += 1
        document["lines"][positions[line_index]] = current

    if len(overrides) > max(1, math.ceil(MAX_OVERRIDE_SHARE * len(document["lines"]))):
        raise ReconcileDeltaError(
            f"{len(overrides)} lyric overrides exceed the reconciliation patch ceiling"
        )

    if "sections" in raw_delta:
        if not isinstance(raw_delta["sections"], list):
            raise ReconcileDeltaError("sections must be an array")
        for section in raw_delta["sections"]:
            if not isinstance(section, dict):
                raise ReconcileDeltaError("each section must be an object")
        document["sections"] = raw_delta["sections"]

    for block in ("metadata", "displayPreferences"):
        if block in raw_delta:
            update = raw_delta[block]
            if not isinstance(update, dict):
                raise ReconcileDeltaError(f"{block} must be an object")
            allowed = (
                {"album", "year", "key", "timeSignature"}
                if block == "metadata" else {"capo", "tuning"}
            )
            extra = set(update) - allowed
            if extra:
                raise ReconcileDeltaError(f"unknown {block} field(s): {sorted(extra)}")
            if block == "displayPreferences" and update.get("capo", 0) != 0:
                raise ReconcileDeltaError("displayPreferences.capo must be 0")
            document[block] = {**document.get(block, {}), **update}

    try:
        song = Song.model_validate(document)
    except ValidationError as error:
        raise ReconcileDeltaError(str(error)) from error
    return AppliedDelta(
        song=song,
        lyric_refs_resolved=refs_resolved,
        lyric_overrides=tuple(overrides),
        patch_bytes=_canonical_size(raw_delta),
        full_bytes=_canonical_size(song.model_dump(mode="json")),
    )
