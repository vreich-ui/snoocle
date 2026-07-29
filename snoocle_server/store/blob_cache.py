"""Keyed JSON-blob cache — the storage layer under the MIR and discovery caches.

Both caches want the same thing: "I computed this once under a key that
encodes every input; give it back instead of recomputing." They differ in
exactly one axis, so that axis is the parameter:

- **MIR has no TTL.** Its key is a content hash of the audio bytes plus the
  engine ids plus the accuracy profile — everything that can change the answer
  is already in the key, so an entry can never go stale. A TTL there would only
  throw away work that is still correct.
- **Discovery has a ~30 day TTL.** Its key is (title, artist, backends), and
  the web behind that key genuinely changes: new sheets get published, old
  URLs rot. The key cannot encode "what the internet says today", so time is
  the only available invalidator.

Same backend resolution as every other store (a Firestore collection, or
in-memory for tests and local dev).

Entries are stored as ``{"value": <blob>, "storedAt": <iso>, "key": <key>}``.
``storedAt`` is what lets a cache hit report WHEN the underlying work actually
happened, which is what keeps provenance truthful.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone

from . import _resolve_backend

log = logging.getLogger(__name__)

# Firestore's hard per-document ceiling is 1 MiB. Stay well under it: a blob
# that doesn't fit is not an error (the caller just recomputes next time), but
# silently attempting the write would raise from deep inside a best-effort path.
MAX_BLOB_BYTES = 900_000


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat(timespec="seconds")


def _is_expired(stored_at: str, ttl_seconds: float | None) -> bool:
    """True when an entry is past its TTL. A malformed/absent timestamp is
    treated as expired: a cache is an optimization, so the safe direction on
    corrupt metadata is to recompute rather than serve something unreadable."""
    if ttl_seconds is None:
        return False
    if not stored_at:
        return True
    try:
        ts = datetime.fromisoformat(stored_at)
    except ValueError:
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return _now() - ts > timedelta(seconds=ttl_seconds)


class BlobCacheStore:
    """A keyed JSON blob cache. ``ttl_seconds=None`` means entries never expire."""

    def __init__(self, ttl_seconds: float | None = None) -> None:
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> tuple[dict, str] | None:
        """``(value, stored_at)`` for a live entry, or None on miss/expiry."""
        raise NotImplementedError  # pragma: no cover - interface

    def put(self, key: str, value: dict) -> str | None:
        """Store a blob; returns its ``storedAt``, or None when it was too
        large to store (the caller simply recomputes next time)."""
        raise NotImplementedError  # pragma: no cover - interface

    def clear(self) -> None:
        """Drop everything — for tests and for an operator-forced rebuild."""
        raise NotImplementedError  # pragma: no cover - interface

    def _too_large(self, key: str, value: dict) -> bool:
        size = len(json.dumps(value).encode("utf-8"))
        if size <= MAX_BLOB_BYTES:
            return False
        log.warning(
            "blob cache: skipping oversized entry key=%s size=%d > %d",
            key, size, MAX_BLOB_BYTES,
        )
        return True


class InMemoryBlobCache(BlobCacheStore):
    def __init__(self, ttl_seconds: float | None = None) -> None:
        super().__init__(ttl_seconds)
        self._docs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[dict, str] | None:
        with self._lock:
            doc = self._docs.get(key)
        if doc is None:
            return None
        stored_at = doc.get("storedAt", "")
        if _is_expired(stored_at, self.ttl_seconds):
            with self._lock:
                self._docs.pop(key, None)
            return None
        return doc["value"], stored_at

    def put(self, key: str, value: dict) -> str | None:
        if self._too_large(key, value):
            return None
        stored_at = _now_iso()
        with self._lock:
            self._docs[key] = {"value": value, "storedAt": stored_at, "key": key}
        return stored_at

    def clear(self) -> None:
        with self._lock:
            self._docs.clear()


class FirestoreBlobCache(BlobCacheStore):
    def __init__(
        self,
        collection: str,
        ttl_seconds: float | None = None,
        project: str | None = None,
        database: str = "(default)",
    ) -> None:
        super().__init__(ttl_seconds)
        from google.cloud import firestore

        kwargs: dict = {}
        if project:
            kwargs["project"] = project
        if database and database != "(default)":
            kwargs["database"] = database
        self._client = firestore.Client(**kwargs)
        self._collection = collection

    def _ref(self, key: str):
        return self._client.collection(self._collection).document(key)

    def get(self, key: str) -> tuple[dict, str] | None:
        # A cache read must never be able to fail the request it is speeding
        # up: any backend problem degrades to a miss.
        try:
            snap = self._ref(key).get()
            if not snap.exists:
                return None
            doc = snap.to_dict() or {}
        except Exception as e:  # noqa: BLE001
            log.warning("blob cache read failed (treating as miss) key=%s: %s", key, e)
            return None
        stored_at = doc.get("storedAt", "")
        if "value" not in doc or _is_expired(stored_at, self.ttl_seconds):
            return None
        return doc["value"], stored_at

    def put(self, key: str, value: dict) -> str | None:
        if self._too_large(key, value):
            return None
        stored_at = _now_iso()
        try:
            self._ref(key).set({"value": value, "storedAt": stored_at, "key": key})
        except Exception as e:  # noqa: BLE001 — a failed write must not fail the run
            log.warning("blob cache write failed key=%s: %s", key, e)
            return None
        return stored_at

    def clear(self) -> None:
        for doc in self._client.collection(self._collection).list_documents():
            doc.delete()


def build_blob_cache(collection: str, ttl_seconds: float | None) -> BlobCacheStore:
    backend, project = _resolve_backend()
    if backend == "firestore":
        from ..config import settings

        return FirestoreBlobCache(
            collection=collection,
            ttl_seconds=ttl_seconds,
            project=project,
            database=settings.firestore_database,
        )
    return InMemoryBlobCache(ttl_seconds=ttl_seconds)
