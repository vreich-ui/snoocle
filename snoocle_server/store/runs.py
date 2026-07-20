"""Run-trace persistence — durable storage for the agent's step-by-step logic.

A :class:`RunRepository` keeps the JSON trace of each reconciliation run so the
GUI can replay it after the fact (the live in-process registry only covers a
run still in progress on this instance). Two backends, selected by the SAME
configuration as the song store: Firestore (durable, a ``song_runs``
collection) and in-memory (hermetic, for tests/local dev).

Traces are opaque dicts (see :mod:`reconcile.trace`); this layer only stores,
fetches, and lists them newest-first per song.
"""

from __future__ import annotations

import logging
import threading

from . import _resolve_backend

log = logging.getLogger(__name__)


class RunRepository:
    """Abstract run store: save one trace, fetch by id, list by song."""

    def save_run(self, run: dict) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def get_run(self, run_id: str) -> dict | None:  # pragma: no cover - interface
        raise NotImplementedError

    def list_runs(self, song_id: str, limit: int = 20) -> list[dict]:  # pragma: no cover
        raise NotImplementedError


class InMemoryRunRepository(RunRepository):
    def __init__(self) -> None:
        self._runs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def save_run(self, run: dict) -> None:
        with self._lock:
            self._runs[run["runId"]] = dict(run)

    def get_run(self, run_id: str) -> dict | None:
        with self._lock:
            run = self._runs.get(run_id)
            return dict(run) if run is not None else None

    def list_runs(self, song_id: str, limit: int = 20) -> list[dict]:
        with self._lock:
            runs = [r for r in self._runs.values() if r.get("songId") == song_id]
        runs.sort(key=lambda r: r.get("startedAt") or "", reverse=True)
        return [_summary(r) for r in runs[:limit]]


class FirestoreRunRepository(RunRepository):
    _COLLECTION = "song_runs"

    def __init__(self, project: str | None = None, database: str = "(default)") -> None:
        from google.cloud import firestore

        kwargs: dict = {}
        if project:
            kwargs["project"] = project
        if database and database != "(default)":
            kwargs["database"] = database
        self._client = firestore.Client(**kwargs)
        log.info("Firestore run store ready: collection=%s", self._COLLECTION)

    @property
    def _col(self):
        return self._client.collection(self._COLLECTION)

    def save_run(self, run: dict) -> None:
        self._col.document(run["runId"]).set(run)

    def get_run(self, run_id: str) -> dict | None:
        snap = self._col.document(run_id).get()
        return snap.to_dict() if snap.exists else None

    def list_runs(self, song_id: str, limit: int = 20) -> list[dict]:
        # Filter by song, sort in Python to avoid requiring a composite index.
        docs = self._col.where("songId", "==", song_id).stream()
        runs = [d.to_dict() for d in docs]
        runs.sort(key=lambda r: r.get("startedAt") or "", reverse=True)
        return [_summary(r) for r in runs[:limit]]


_LARGE_FIELDS = {"steps", "mir", "mirWindows"}


def _summary(run: dict) -> dict:
    """A run without its large payloads (steps, MIR data) — for list views."""
    return {k: v for k, v in run.items() if k not in _LARGE_FIELDS}


_repo: RunRepository | None = None
_lock = threading.Lock()


def build_run_repository() -> RunRepository:
    backend, project = _resolve_backend()
    if backend == "firestore":
        from ..config import settings

        return FirestoreRunRepository(project=project, database=settings.firestore_database)
    return InMemoryRunRepository()


def get_run_store() -> RunRepository:
    global _repo
    if _repo is None:
        with _lock:
            if _repo is None:
                _repo = build_run_repository()
                log.info("run store backend: %s", type(_repo).__name__)
    return _repo


def reset_run_store() -> None:
    global _repo
    with _lock:
        _repo = None


def fetch_run(run_id: str) -> dict | None:
    """A run's full trace: the live in-process record first (a run still in
    progress on this instance), then the durable store. Shared by the REST
    route and the MCP tool so the two surfaces can't drift."""
    from ..reconcile.trace import get_live_run

    live = get_live_run(run_id)
    if live is not None:
        return live.to_dict()
    return get_run_store().get_run(run_id)
