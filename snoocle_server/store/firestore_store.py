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

import functools
import logging

from ..schema import Song
from .base import (
    SaveResult,
    SongRepository,
    SongVersion,
    StoreError,
    StoreUnavailableError,
    VersionConflictError,
    YouTubeCookieRecord,
    check_provenance_append_only,
    count_cookie_lines,
    next_timestamp,
    now_iso,
    unified_song_diff,
    version_sha,
)

log = logging.getLogger(__name__)


def _is_infra_error(exc: BaseException) -> bool:
    """True for a Firestore/gRPC/auth backend failure (database missing,
    permission denied, unavailable, unauthenticated, connection error) — as
    opposed to a bug in our own code."""
    mod = type(exc).__module__ or ""
    return mod.startswith(("google.api_core", "google.auth", "grpc"))


def _translate_infra_errors(fn):
    """Wrap a repository method so backend failures surface as
    StoreUnavailableError (-> HTTP 503) instead of a bare 500, while our own
    StoreError/VersionConflictError and real bugs pass through unchanged."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except (StoreError, StoreUnavailableError):
            raise
        except Exception as e:  # noqa: BLE001
            if _is_infra_error(e):
                raise StoreUnavailableError(f"Firestore backend error: {e}") from e
            raise

    return wrapper


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

    def _config_ref(self, name: str):
        # a separate collection so config never shows up in list_songs()
        return self._client.collection("snoocle_config").document(name)

    # --- YouTube cookies -------------------------------------------------

    @_translate_infra_errors
    def set_youtube_cookies(self, cookies_txt: str, source: str) -> YouTubeCookieRecord:
        ts, lc = now_iso(), count_cookie_lines(cookies_txt)
        self._config_ref("youtube").set(
            {"cookies": cookies_txt, "updatedAt": ts, "source": source, "lineCount": lc}
        )
        return YouTubeCookieRecord(ts, source, lc)

    @_translate_infra_errors
    def get_youtube_cookies_txt(self) -> str | None:
        d = self._config_ref("youtube").get().to_dict()
        return d.get("cookies") if d else None

    @_translate_infra_errors
    def youtube_cookies_status(self) -> YouTubeCookieRecord | None:
        d = self._config_ref("youtube").get().to_dict()
        if not d:
            return None
        return YouTubeCookieRecord(d.get("updatedAt", ""), d.get("source", ""), d.get("lineCount", 0))

    @_translate_infra_errors
    def clear_youtube_cookies(self) -> None:
        self._config_ref("youtube").delete()

    # --- reads -----------------------------------------------------------

    @_translate_infra_errors
    def list_songs(self) -> list[str]:
        # list_documents() returns references without reading the (large) song
        # blobs — cheap id enumeration.
        return sorted(ref.id for ref in self._collection.list_documents())

    @_translate_infra_errors
    def current_version(self, song_id: str) -> str | None:
        data = self._song_ref(song_id).get().to_dict()
        return data.get("latestVersion") if data else None

    @_translate_infra_errors
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

    @_translate_infra_errors
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

    @_translate_infra_errors
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

    @_translate_infra_errors
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
