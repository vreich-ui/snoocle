"""Song persistence — a backend-agnostic repository with a Firestore (durable)
and an in-memory (hermetic) implementation, selected by configuration.
"""

from __future__ import annotations

import logging
import os
import threading

from ..config import settings
from .base import (  # noqa: F401
    SaveResult,
    SongRepository,
    SongVersion,
    StoreError,
    StoreUnavailableError,
    VersionConflictError,
    version_sha,
)

log = logging.getLogger(__name__)

_repo: SongRepository | None = None
_repo_lock = threading.Lock()


def _resolve_backend() -> tuple[str, str | None]:
    """(backend, project) after resolving the "auto" default."""
    backend = (settings.store_backend or "auto").lower()
    project = settings.google_cloud_project or os.environ.get("GOOGLE_CLOUD_PROJECT", "") or None
    emulator = os.environ.get("FIRESTORE_EMULATOR_HOST")
    if backend == "auto":
        backend = "firestore" if (project or emulator) else "memory"
    if backend == "firestore" and not project:
        # Emulator runs need *a* project id even though it's not authenticated.
        project = "snoocle-local"
    return backend, project


def backend_label() -> str:
    """Short backend id for /healthz (no client construction)."""
    return _resolve_backend()[0]


def build_repository() -> SongRepository:
    backend, project = _resolve_backend()
    if backend == "firestore":
        from .firestore_store import FirestoreSongRepository

        return FirestoreSongRepository(
            project=project,
            database=settings.firestore_database,
            collection=settings.firestore_collection,
        )
    if backend == "memory":
        from .memory import InMemorySongRepository

        return InMemorySongRepository()
    raise StoreError(
        f"unknown SNOOCLE_STORE_BACKEND {backend!r} (expected auto|firestore|memory)"
    )


def get_repository() -> SongRepository:
    """The process-wide song repository (built once, then reused)."""
    global _repo
    if _repo is None:
        with _repo_lock:
            if _repo is None:
                _repo = build_repository()
                log.info("song repository backend: %s", type(_repo).__name__)
    return _repo


def reset_repository() -> None:
    """Drop the cached repository (tests; reconfiguration)."""
    global _repo
    with _repo_lock:
        _repo = None
