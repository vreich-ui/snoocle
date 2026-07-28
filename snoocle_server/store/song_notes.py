"""Per-song reconciliation notes — free-text instructions about how to BUILD a
song, replayed as reconciler guidance on every analyze.

Kept out of the Song document on purpose: the Song schema is the iOS-facing
wire contract for song *content*, and workflow state like "the bridge is Bm,
not D" has no business accreting onto it. Same backend resolution as every
other store (Firestore ``song_notes`` collection, one document per songId, or
in-memory).
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

from . import _resolve_backend

log = logging.getLogger(__name__)

# The wire contract's ceiling (contract §1). Enforced by the API, which turns a
# longer body into a 400 naming this limit rather than storing a runaway paste.
MAX_NOTES_CHARS = 8000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SongNotesStore:
    def get(self, song_id: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def get_record(self, song_id: str) -> dict | None:  # pragma: no cover - interface
        raise NotImplementedError

    def set(self, song_id: str, notes: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def delete(self, song_id: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class InMemorySongNotesStore(SongNotesStore):
    def __init__(self) -> None:
        self._docs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, song_id: str) -> str:
        with self._lock:
            doc = self._docs.get(song_id)
            return (doc or {}).get("notes", "") or ""

    def get_record(self, song_id: str) -> dict | None:
        with self._lock:
            doc = self._docs.get(song_id)
            return dict(doc) if doc is not None else None

    def set(self, song_id: str, notes: str) -> str:
        text = (notes or "").strip()
        if not text:
            # "no notes" and "empty notes" are the same state to every reader,
            # so storing a blank record would only leave a tombstone to explain.
            self.delete(song_id)
            return ""
        with self._lock:
            self._docs[song_id] = {"notes": text, "updated_at": _now()}
        return text

    def delete(self, song_id: str) -> bool:
        with self._lock:
            return self._docs.pop(song_id, None) is not None


class FirestoreSongNotesStore(SongNotesStore):
    _COLLECTION = "song_notes"

    def __init__(self, project: str | None = None, database: str = "(default)") -> None:
        from google.cloud import firestore

        kwargs: dict = {}
        if project:
            kwargs["project"] = project
        if database and database != "(default)":
            kwargs["database"] = database
        self._client = firestore.Client(**kwargs)

    def _ref(self, song_id: str):
        return self._client.collection(self._COLLECTION).document(song_id)

    def get(self, song_id: str) -> str:
        doc = self.get_record(song_id)
        return (doc or {}).get("notes", "") or ""

    def get_record(self, song_id: str) -> dict | None:
        snap = self._ref(song_id).get()
        return snap.to_dict() if snap.exists else None

    def set(self, song_id: str, notes: str) -> str:
        text = (notes or "").strip()
        if not text:
            self.delete(song_id)
            return ""
        self._ref(song_id).set({"notes": text, "updated_at": _now()})
        return text

    def delete(self, song_id: str) -> bool:
        ref = self._ref(song_id)
        existed = ref.get().exists
        ref.delete()
        return existed


_store: SongNotesStore | None = None
_lock = threading.Lock()


def build_song_notes_store() -> SongNotesStore:
    backend, project = _resolve_backend()
    if backend == "firestore":
        from ..config import settings

        return FirestoreSongNotesStore(project=project, database=settings.firestore_database)
    return InMemorySongNotesStore()


def get_song_notes_store() -> SongNotesStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = build_song_notes_store()
                log.info("song notes store backend: %s", type(_store).__name__)
    return _store


def reset_song_notes_store() -> None:
    global _store
    with _lock:
        _store = None
