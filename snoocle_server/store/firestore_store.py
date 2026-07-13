"""Firestore (Native mode) :class:`SongRepository` — the durable store.

Data model (per the build brief):

- ``songs/{songId}`` — the latest Song plus denormalized ``{title, artist,
  latestVersion, updatedAt}`` so GET /v1/songs and title/artist queries are
  cheap (no need to open the versions subcollection).
- ``songs/{songId}/versions/{versionSha}`` — an immutable snapshot
  ``{song, message, timestamp, parent}``. ``versionSha`` is the content hash
  (see :func:`version_sha`); ``parent`` is the sha this was written on top of,
  or ``None`` for the first version.

Writes go through a Firestore transaction so the read-check-write of the
optimistic lock is atomic even under concurrent writers. Uses Application
Default Credentials (no key files); the project comes from
``GOOGLE_CLOUD_PROJECT``.
"""

from __future__ import annotations

import logging

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

log = logging.getLogger(__name__)


class FirestoreSongRepository(SongRepository):
    def __init__(
        self,
        project: str | None = None,
        database: str = "(default)",
        collection: str = "songs",
    ) -> None:
        # Lazy import so the store package (and the in-memory backend) works in
        # environments without google-cloud-firestore installed.
        from google.cloud import firestore

        self._firestore = firestore
        kwargs: dict = {}
        if project:
            kwargs["project"] = project
        if database and database != "(default)":
            kwargs["database"] = database
        self._client = firestore.Client(**kwargs)
        self._collection_name = collection
        log.info(
            "Firestore store ready: project=%s database=%s collection=%s",
            self._client.project, database, collection,
        )

    # --- internals -------------------------------------------------------

    @property
    def _collection(self):
        return self._client.collection(self._collection_name)

    def _song_ref(self, song_id: str):
        return self._collection.document(song_id)

    def _version_ref(self, song_id: str, sha: str):
        return self._song_ref(song_id).collection("versions").document(sha)

    # --- reads -----------------------------------------------------------

    def list_songs(self) -> list[str]:
        # list_documents() returns references without reading the (large) song
        # blobs — cheap id enumeration.
        return sorted(ref.id for ref in self._collection.list_documents())

    def current_version(self, song_id: str) -> str | None:
        data = self._song_ref(song_id).get().to_dict()
        return data.get("latestVersion") if data else None

    def get(self, song_id: str, version: str | None = None) -> Song:
        if version is None:
            data = self._song_ref(song_id).get().to_dict()
            if not data:
                raise StoreError(f"song {song_id!r} not found")
            return Song.model_validate(data["song"])
        data = self._version_ref(song_id, version).get().to_dict()
        if not data:
            raise StoreError(f"song {song_id!r} at version {version!r} not found")
        return Song.model_validate(data["song"])

    def versions(self, song_id: str) -> list[SongVersion]:
        if not self._song_ref(song_id).get().exists:
            return []
        query = (
            self._song_ref(song_id)
            .collection("versions")
            .order_by("timestamp", direction=self._firestore.Query.DESCENDING)
        )
        out: list[SongVersion] = []
        for doc in query.stream():
            d = doc.to_dict() or {}
            out.append(SongVersion(doc.id, d.get("timestamp", ""), d.get("message", "")))
        return out

    def diff(self, song_id: str, version_a: str, version_b: str) -> str:
        a = self._version_ref(song_id, version_a).get().to_dict()
        b = self._version_ref(song_id, version_b).get().to_dict()
        if not a:
            raise StoreError(f"song {song_id!r}: unknown version {version_a!r}")
        if not b:
            raise StoreError(f"song {song_id!r}: unknown version {version_b!r}")
        return unified_song_diff(
            a["song"], b["song"], f"{song_id}@{version_a}", f"{song_id}@{version_b}"
        )

    # --- write (transactional CAS) --------------------------------------

    def save(
        self,
        song: Song,
        message: str,
        expected_version: str | None = None,
        enforce_expected: bool = False,
    ) -> SaveResult:
        song_ref = self._song_ref(song.id)
        firestore = self._firestore

        @firestore.transactional
        def _txn(transaction) -> SaveResult:
            data = song_ref.get(transaction=transaction).to_dict()
            current = data.get("latestVersion") if data else None
            if (enforce_expected or expected_version is not None) and expected_version != current:
                raise VersionConflictError(
                    f"song {song.id!r}: expected version {expected_version!r} "
                    f"but store has {current!r}"
                )
            if data:
                check_provenance_append_only(data.get("song") or {}, song)

            sha = version_sha(song)
            if data and sha == current:
                # identical content -> idempotent no-op; snapshot is immutable.
                return SaveResult(song.id, sha, data.get("updatedAt", ""), message)

            prev_ts = data.get("updatedAt") if data else None
            ts = next_timestamp(prev_ts)
            song_json = song.model_dump(mode="json")
            transaction.set(
                self._version_ref(song.id, sha),
                {"song": song_json, "message": message, "timestamp": ts, "parent": current},
            )
            transaction.set(
                song_ref,
                {
                    "song": song_json,
                    "title": song.metadata.title,
                    "artist": song.metadata.artist,
                    "latestVersion": sha,
                    "updatedAt": ts,
                },
            )
            return SaveResult(song.id, sha, ts, message)

        return _txn(self._client.transaction())
