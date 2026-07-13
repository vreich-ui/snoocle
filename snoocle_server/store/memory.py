"""In-process, in-memory :class:`SongRepository`.

Hermetic and dependency-free: the whole analyze -> persist -> fetch -> versions
path runs offline with no GCP project, no network, and no key files. Used by
the test suite, by CI, and as the local-dev fallback when Firestore isn't
configured. Semantics (version shas, optimistic locking, append-only
provenance, diffs) mirror the Firestore backend exactly.

State lives for the process lifetime only — it does NOT survive a restart, so
it is not the production store. A single module-level instance is shared via
``store.get_repository()`` so all callers in one process see the same data.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from ..schema import Song
from .base import (
    SaveResult,
    SongRepository,
    SongVersion,
    StoreError,
    VersionConflictError,
    check_provenance_append_only,
    next_timestamp,
    unified_song_diff,
    version_sha,
)


@dataclass
class _Version:
    song: dict  # full Song JSON (mode="json")
    message: str
    timestamp: str
    parent: str | None


@dataclass
class _Record:
    title: str
    artist: str
    latest_version: str
    updated_at: str
    versions: dict[str, _Version] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)  # append order == oldest->newest


class InMemorySongRepository(SongRepository):
    def __init__(self) -> None:
        self._songs: dict[str, _Record] = {}
        self._lock = threading.RLock()

    def list_songs(self) -> list[str]:
        with self._lock:
            return sorted(self._songs)

    def current_version(self, song_id: str) -> str | None:
        with self._lock:
            rec = self._songs.get(song_id)
            return rec.latest_version if rec else None

    def get(self, song_id: str, version: str | None = None) -> Song:
        with self._lock:
            rec = self._songs.get(song_id)
            if rec is None:
                raise StoreError(f"song {song_id!r} not found")
            if version is None:
                return Song.model_validate(rec.versions[rec.latest_version].song)
            snap = rec.versions.get(version)
            if snap is None:
                raise StoreError(f"song {song_id!r} at version {version!r} not found")
            return Song.model_validate(snap.song)

    def versions(self, song_id: str) -> list[SongVersion]:
        with self._lock:
            rec = self._songs.get(song_id)
            if rec is None:
                return []
            # newest first
            return [
                SongVersion(version=sha, timestamp=rec.versions[sha].timestamp,
                            message=rec.versions[sha].message)
                for sha in reversed(rec.order)
            ]

    def diff(self, song_id: str, version_a: str, version_b: str) -> str:
        with self._lock:
            rec = self._songs.get(song_id)
            if rec is None:
                raise StoreError(f"song {song_id!r} not found")
            try:
                a = rec.versions[version_a].song
                b = rec.versions[version_b].song
            except KeyError as e:
                raise StoreError(f"song {song_id!r}: unknown version {e.args[0]!r}") from e
        return unified_song_diff(a, b, f"{song_id}@{version_a}", f"{song_id}@{version_b}")

    def save(
        self,
        song: Song,
        message: str,
        expected_version: str | None = None,
        enforce_expected: bool = False,
    ) -> SaveResult:
        with self._lock:
            rec = self._songs.get(song.id)
            current = rec.latest_version if rec else None
            if (enforce_expected or expected_version is not None) and expected_version != current:
                raise VersionConflictError(
                    f"song {song.id!r}: expected version {expected_version!r} "
                    f"but store has {current!r}"
                )
            if rec is not None:
                check_provenance_append_only(rec.versions[current].song, song)

            sha = version_sha(song)
            if rec is not None and sha == current:
                # identical content -> idempotent no-op; keep the immutable snapshot
                snap = rec.versions[sha]
                return SaveResult(song.id, sha, snap.timestamp, snap.message)

            prev_ts = rec.versions[current].timestamp if rec is not None else None
            ts = next_timestamp(prev_ts)
            snapshot = _Version(
                song=song.model_dump(mode="json"), message=message, timestamp=ts, parent=current
            )
            if rec is None:
                rec = _Record(
                    title=song.metadata.title,
                    artist=song.metadata.artist,
                    latest_version=sha,
                    updated_at=ts,
                )
                self._songs[song.id] = rec
            rec.versions[sha] = snapshot
            rec.order.append(sha)
            rec.latest_version = sha
            rec.updated_at = ts
            rec.title = song.metadata.title
            rec.artist = song.metadata.artist
            return SaveResult(song.id, sha, ts, message)
