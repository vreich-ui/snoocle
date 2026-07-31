"""Atomic repair of a legacy song identity.

New work never creates an unresolved id, but historic documents still need an
operator-controlled path out of ``unknown--unknown`` / ``artist--unknown``.
This module moves *every* immutable version and every recorded run to the new
id in one backend transaction. It intentionally does not merge into a target
that already exists: deciding how two independently-versioned histories should
interleave is a separate human decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..identity import require_resolved_song_id
from ..schema import Song
from ..schema.song import slugify_song_id
from .base import SongRepository, StoreError, song_has_timing, summarize_song, version_sha
from .runs import RunRepository


class IdentityRenameError(StoreError):
    pass


@dataclass(frozen=True)
class IdentityRenameResult:
    old_song_id: str
    new_song_id: str
    version_map: dict[str, str]
    run_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "oldSongId": self.old_song_id,
            "songId": self.new_song_id,
            "versionMap": self.version_map,
            "migratedRunIds": self.run_ids,
        }


def _renamed_song(song: Song, song_id: str, artist: str, title: str) -> Song:
    metadata = song.metadata.model_copy(update={"artist": artist, "title": title})
    return Song.model_validate(song.model_copy(update={"id": song_id, "metadata": metadata}).model_dump())


def rename_song_identity(
    song_store: SongRepository,
    run_store: RunRepository,
    old_song_id: str,
    *,
    artist: str,
    title: str,
) -> IdentityRenameResult:
    """Rename one complete song history and its runs atomically.

    Only legacy source ids are expected at this boundary, so the *new* id is
    validated but the old id is intentionally not: it may contain ``unknown``
    and must remain readable long enough to be repaired.
    """
    new_song_id = slugify_song_id(artist, title)
    require_resolved_song_id(new_song_id)
    if old_song_id == new_song_id:
        raise IdentityRenameError("old and new song ids are the same")

    from .firestore_store import FirestoreSongRepository
    from .memory import InMemorySongRepository
    from .runs import FirestoreRunRepository, InMemoryRunRepository

    if isinstance(song_store, InMemorySongRepository) and isinstance(run_store, InMemoryRunRepository):
        return _rename_memory(song_store, run_store, old_song_id, new_song_id, artist, title)
    if isinstance(song_store, FirestoreSongRepository) and isinstance(run_store, FirestoreRunRepository):
        return _rename_firestore(song_store, run_store, old_song_id, new_song_id, artist, title)
    raise IdentityRenameError(
        "song and run stores must use the same supported backend for atomic identity migration"
    )


def _rename_memory(song_store, run_store, old_id: str, new_id: str, artist: str, title: str) -> IdentityRenameResult:
    # Both repositories use a private lock because a cross-store transaction is
    # not exposed in their intentionally small public contracts. Acquiring in
    # a stable order makes the composite operation atomic for all process users.
    with song_store._lock:
        with run_store._lock:
            old = song_store._songs.get(old_id)
            if old is None:
                raise IdentityRenameError(f"song {old_id!r} not found")
            if new_id in song_store._songs:
                raise IdentityRenameError(f"target song {new_id!r} already exists; refusing to merge histories")

            transformed: dict[str, Song] = {}
            version_map: dict[str, str] = {}
            for old_version in old.order:
                source = Song.model_validate(old.versions[old_version].song)
                moved = _renamed_song(source, new_id, artist, title)
                new_version = version_sha(moved)
                if new_version in version_map.values():
                    raise IdentityRenameError("rename would collapse two distinct legacy versions")
                transformed[old_version] = moved
                version_map[old_version] = new_version

            from .memory import _Record, _Version

            moved_versions: dict[str, _Version] = {}
            moved_order: list[str] = []
            for old_version in old.order:
                original = old.versions[old_version]
                new_version = version_map[old_version]
                parent = version_map.get(original.parent) if original.parent else None
                moved_versions[new_version] = _Version(
                    song=transformed[old_version].model_dump(mode="json"),
                    message=original.message,
                    timestamp=original.timestamp,
                    parent=parent,
                )
                moved_order.append(new_version)

            latest = version_map[old.latest_version]
            latest_song = transformed[old.latest_version]
            song_store._songs[new_id] = _Record(
                title=title,
                artist=artist,
                latest_version=latest,
                updated_at=old.updated_at,
                versions=moved_versions,
                order=moved_order,
                youtube_video_id=latest_song.audio.youtubeVideoId,
                has_timing=song_has_timing(latest_song),
            )
            del song_store._songs[old_id]

            run_ids: list[str] = []
            for run_id, run in run_store._runs.items():
                if run.get("songId") == old_id:
                    run["songId"] = new_id
                    run["migratedFromSongId"] = old_id
                    run_ids.append(run_id)
            return IdentityRenameResult(old_id, new_id, version_map, sorted(run_ids))


def _rename_firestore(song_store, run_store, old_id: str, new_id: str, artist: str, title: str) -> IdentityRenameResult:
    """Firestore equivalent of :func:`_rename_memory` using one transaction."""
    firestore = song_store._firestore
    old_ref, new_ref = song_store._song_ref(old_id), song_store._song_ref(new_id)
    old_doc = old_ref.get()
    if not old_doc.exists:
        raise IdentityRenameError(f"song {old_id!r} not found")
    if new_ref.get().exists:
        raise IdentityRenameError(f"target song {new_id!r} already exists; refusing to merge histories")

    version_docs = list(old_ref.collection("versions").stream())
    if not version_docs:
        raise IdentityRenameError(f"song {old_id!r} has no versions to migrate")
    transformed: dict[str, Song] = {}
    version_map: dict[str, str] = {}
    for doc in version_docs:
        payload = doc.to_dict() or {}
        source = Song.model_validate(payload["song"])
        moved = _renamed_song(source, new_id, artist, title)
        transformed[doc.id] = moved
        version_map[doc.id] = version_sha(moved)
    if len(set(version_map.values())) != len(version_map):
        raise IdentityRenameError("rename would collapse two distinct legacy versions")

    run_docs = list(run_store._col.where("songId", "==", old_id).stream())
    # Each version is deleted and recreated, plus old/new root documents and
    # every run update. Firestore caps one transaction at 500 writes.
    writes = 2 * len(version_docs) + 2 + len(run_docs)
    if writes > 500:
        raise IdentityRenameError(f"identity migration needs {writes} writes; split it manually below Firestore's 500-write limit")

    old_data = old_doc.to_dict() or {}
    latest_old = old_data.get("latestVersion")
    if latest_old not in version_map:
        raise IdentityRenameError("song latestVersion is missing from its versions collection")
    latest_song = transformed[latest_old]
    latest_new = version_map[latest_old]

    @firestore.transactional
    def _txn(transaction):
        current_old = old_ref.get(transaction=transaction)
        if not current_old.exists:
            raise IdentityRenameError(f"song {old_id!r} changed during migration")
        if (current_old.to_dict() or {}).get("latestVersion") != latest_old:
            raise IdentityRenameError(f"song {old_id!r} changed during migration")
        if new_ref.get(transaction=transaction).exists:
            raise IdentityRenameError(f"target song {new_id!r} was created during migration")
        for doc in version_docs:
            original = doc.to_dict() or {}
            parent = original.get("parent")
            transaction.set(
                new_ref.collection("versions").document(version_map[doc.id]),
                {
                    "song": transformed[doc.id].model_dump(mode="json"),
                    "message": original.get("message", ""),
                    "timestamp": original.get("timestamp", ""),
                    "parent": version_map.get(parent) if parent else None,
                },
            )
            transaction.delete(old_ref.collection("versions").document(doc.id))
        transaction.set(
            new_ref,
            {
                "song": latest_song.model_dump(mode="json"),
                **summarize_song(latest_song, latest_new, old_data.get("updatedAt", "")),
            },
        )
        transaction.delete(old_ref)
        for doc in run_docs:
            transaction.update(
                run_store._col.document(doc.id),
                {"songId": new_id, "migratedFromSongId": old_id},
            )

    _txn(song_store._client.transaction())
    return IdentityRenameResult(old_id, new_id, version_map, sorted(doc.id for doc in run_docs))
