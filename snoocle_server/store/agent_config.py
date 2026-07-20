"""Persistence for the runtime-editable agent config.

Stored in the same `snoocle_config` collection as YouTube cookies (document
`"agent"`, sibling of `"youtube"`), selected by the same backend resolution as
every other store. The value is an opaque dict (an `AgentConfig` dump); this
layer only stores, fetches, and clears it.
"""

from __future__ import annotations

import logging
import threading

from . import _resolve_backend

log = logging.getLogger(__name__)


class AgentConfigStore:
    def get(self) -> dict | None:  # pragma: no cover - interface
        raise NotImplementedError

    def set(self, doc: dict) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def clear(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryAgentConfigStore(AgentConfigStore):
    def __init__(self) -> None:
        self._doc: dict | None = None
        self._lock = threading.Lock()

    def get(self) -> dict | None:
        with self._lock:
            return dict(self._doc) if self._doc is not None else None

    def set(self, doc: dict) -> None:
        with self._lock:
            self._doc = dict(doc)

    def clear(self) -> None:
        with self._lock:
            self._doc = None


class FirestoreAgentConfigStore(AgentConfigStore):
    _COLLECTION = "snoocle_config"
    _DOC = "agent"

    def __init__(self, project: str | None = None, database: str = "(default)") -> None:
        from google.cloud import firestore

        kwargs: dict = {}
        if project:
            kwargs["project"] = project
        if database and database != "(default)":
            kwargs["database"] = database
        self._client = firestore.Client(**kwargs)

    @property
    def _ref(self):
        return self._client.collection(self._COLLECTION).document(self._DOC)

    def get(self) -> dict | None:
        snap = self._ref.get()
        return snap.to_dict() if snap.exists else None

    def set(self, doc: dict) -> None:
        self._ref.set(doc)

    def clear(self) -> None:
        self._ref.delete()


_store: AgentConfigStore | None = None
_lock = threading.Lock()


def build_agent_config_store() -> AgentConfigStore:
    backend, project = _resolve_backend()
    if backend == "firestore":
        from ..config import settings

        return FirestoreAgentConfigStore(project=project, database=settings.firestore_database)
    return InMemoryAgentConfigStore()


def get_agent_config_store() -> AgentConfigStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = build_agent_config_store()
                log.info("agent config store backend: %s", type(_store).__name__)
    return _store


def reset_agent_config_store() -> None:
    global _store
    with _lock:
        _store = None
