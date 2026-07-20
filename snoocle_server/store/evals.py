"""Gold-version pointers for the eval harness.

"Gold" for a song is simply one of its stored versions, marked as ground truth.
This keeps a small map songId -> {goldVersion, updatedAt} in the same backend as
the song store (Firestore ``snoocle_evals`` collection, or in-memory). Scoring
then loads that version via the song repository and diffs a candidate against it.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

from . import _resolve_backend

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EvalStore:
    def set_gold(self, song_id: str, version: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_gold(self, song_id: str) -> str | None:  # pragma: no cover
        raise NotImplementedError

    def list_gold(self) -> list[dict]:  # pragma: no cover
        raise NotImplementedError


class InMemoryEvalStore(EvalStore):
    def __init__(self) -> None:
        self._gold: dict[str, dict] = {}
        self._lock = threading.Lock()

    def set_gold(self, song_id: str, version: str) -> None:
        with self._lock:
            self._gold[song_id] = {"songId": song_id, "goldVersion": version, "updatedAt": _now()}

    def get_gold(self, song_id: str) -> str | None:
        with self._lock:
            rec = self._gold.get(song_id)
            return rec["goldVersion"] if rec else None

    def list_gold(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._gold.values()]


class FirestoreEvalStore(EvalStore):
    _COLLECTION = "snoocle_evals"

    def __init__(self, project: str | None = None, database: str = "(default)") -> None:
        from google.cloud import firestore

        kwargs: dict = {}
        if project:
            kwargs["project"] = project
        if database and database != "(default)":
            kwargs["database"] = database
        self._client = firestore.Client(**kwargs)

    @property
    def _col(self):
        return self._client.collection(self._COLLECTION)

    def set_gold(self, song_id: str, version: str) -> None:
        self._col.document(song_id).set(
            {"songId": song_id, "goldVersion": version, "updatedAt": _now()}
        )

    def get_gold(self, song_id: str) -> str | None:
        snap = self._col.document(song_id).get()
        return snap.to_dict().get("goldVersion") if snap.exists else None

    def list_gold(self) -> list[dict]:
        return [d.to_dict() for d in self._col.stream()]


_store: EvalStore | None = None
_lock = threading.Lock()


def build_eval_store() -> EvalStore:
    backend, project = _resolve_backend()
    if backend == "firestore":
        from ..config import settings

        return FirestoreEvalStore(project=project, database=settings.firestore_database)
    return InMemoryEvalStore()


def get_eval_store() -> EvalStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = build_eval_store()
                log.info("eval store backend: %s", type(_store).__name__)
    return _store


def reset_eval_store() -> None:
    global _store
    with _lock:
        _store = None
