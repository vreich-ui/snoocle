"""Plan the one-time repair for the 2026-07-31 Splean identity incident.

The bad ``unknown--unknown`` document contains versions from more than one
recording, so it cannot use the normal whole-history ``set_song_identity``
operation. This planner classifies each legacy version/run independently from
the evidence already stored with it. It is dry-run only: its output is the
reviewable input for the subsequent operator-approved re-home operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .base import SongRepository
from .runs import RunRepository


TARGETS = {
    "romans": "splean--romans",
    "vyhoda-net": "splean--vyhoda-net",
    "bog-ustal-nas-lyubit": "splean--bog-ustal-nas-lyubit",
}
_LEGACY_IDS = ("unknown--unknown", "splean--unknown")


@dataclass
class PlannedRehome:
    source_song_id: str
    target_song_id: str
    versions: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


@dataclass
class SpleanMigrationPlan:
    moves: list[PlannedRehome] = field(default_factory=list)
    unresolved_versions: list[str] = field(default_factory=list)
    unresolved_runs: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = ["Splean identity migration plan (DRY RUN — no writes):"]
        for move in self.moves:
            lines.append(f"- {move.source_song_id} -> {move.target_song_id}")
            lines.append(f"  versions: {', '.join(move.versions) or '(none)'}")
            lines.append(f"  runs: {', '.join(move.run_ids) or '(none)'}")
            lines.append(f"  evidence: {', '.join(move.evidence) or '(none)'}")
        if self.unresolved_versions:
            lines.append(f"- unresolved versions (not moved): {', '.join(self.unresolved_versions)}")
        if self.unresolved_runs:
            lines.append(f"- unresolved runs (not moved): {', '.join(self.unresolved_runs)}")
        if len(lines) == 1:
            lines.append("- no legacy Splean documents found")
        return "\n".join(lines)


def _strings(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def _target_from_values(values: Iterable[str]) -> tuple[str | None, list[str]]:
    evidence: list[str] = []
    found: set[str] = set()
    for raw in values:
        value = raw.casefold()
        for marker, target in TARGETS.items():
            if marker in value:
                found.add(target)
                evidence.append(raw)
    return (next(iter(found)) if len(found) == 1 else None), list(dict.fromkeys(evidence))


def _song_values(song) -> Iterable[str]:
    yield song.metadata.artist
    yield song.metadata.title
    if song.audio.contentHash:
        yield song.audio.contentHash
    for entry in song.provenance:
        yield from entry.sources
        if entry.notes:
            yield entry.notes


def plan_splean_identity_migration(
    song_store: SongRepository, run_store: RunRepository
) -> SpleanMigrationPlan:
    """Group the three affected songs by stored URL/title/audio evidence.

    A run lacking its own source URL can inherit the target of the unique audio
    content hash it records. Ambiguous or missing evidence is reported and
    deliberately left unmoved.
    """
    plan = SpleanMigrationPlan()
    moves: dict[tuple[str, str], PlannedRehome] = {}
    audio_target: dict[str, str] = {}

    def move_for(source_id: str, target: str) -> PlannedRehome:
        return moves.setdefault((source_id, target), PlannedRehome(source_id, target))

    pending_runs: list[tuple[str, dict]] = []
    for source_id in _LEGACY_IDS:
        for version in song_store.versions(source_id):
            try:
                song = song_store.get(source_id, version.version)
            except Exception:
                plan.unresolved_versions.append(f"{source_id}@{version.version}")
                continue
            target, evidence = _target_from_values(_song_values(song))
            # The incident's third document has one known intended destination;
            # preserve that explicit operator mapping even if its early run did
            # not retain a page URL in provenance.
            if target is None and source_id == "splean--unknown":
                target, evidence = "splean--bog-ustal-nas-lyubit", ["legacy source id: splean--unknown"]
            label = f"{source_id}@{version.version}"
            if target is None:
                plan.unresolved_versions.append(label)
                continue
            move = move_for(source_id, target)
            move.versions.append(version.version)
            move.evidence.extend(evidence)
            if song.audio.contentHash:
                prior = audio_target.get(song.audio.contentHash)
                if prior is None or prior == target:
                    audio_target[song.audio.contentHash] = target

        for summary in run_store.list_runs(source_id, limit=None):
            run = run_store.get_run(summary["runId"]) or summary
            pending_runs.append((source_id, run))

    for source_id, run in pending_runs:
        target, evidence = _target_from_values(_strings(run))
        if target is None:
            hashes = [value for value in _strings(run) if len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)]
            mapped = {audio_target[value.casefold()] for value in hashes if value.casefold() in audio_target}
            target = next(iter(mapped)) if len(mapped) == 1 else None
            if target:
                evidence = hashes
        if target is None and source_id == "splean--unknown":
            target, evidence = "splean--bog-ustal-nas-lyubit", ["legacy source id: splean--unknown"]
        run_id = str(run.get("runId") or "(missing-run-id)")
        if target is None:
            plan.unresolved_runs.append(f"{source_id}:{run_id}")
            continue
        move = move_for(source_id, target)
        move.run_ids.append(run_id)
        move.evidence.extend(evidence)

    for move in moves.values():
        move.versions.sort()
        move.run_ids.sort()
        move.evidence = list(dict.fromkeys(move.evidence))
    plan.moves = sorted(moves.values(), key=lambda move: (move.source_song_id, move.target_song_id))
    plan.unresolved_versions.sort()
    plan.unresolved_runs.sort()
    return plan
