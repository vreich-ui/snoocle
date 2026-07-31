"""Atomic admission control for paid reconciliation runs.

The idempotency key is derived from the song identity, agent configuration,
and the content identifiers in the evidence manifest.  Firestore keeps one
transactionally-updated lock document per key so separate Cloud Run instances
cannot both admit the same work.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from . import _resolve_backend

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=value.tzinfo or timezone.utc).astimezone(timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)
    return None


def evidence_hash(manifest: dict | None) -> str:
    """Hash only evidence content identifiers, never descriptive scores/status.

    Cache hit/miss labels and ageDays can change without changing evidence.
    Conversely, a new analysis time, engine set, source id set, LRC fingerprint,
    scope, or guidance describes materially different reconciliation input.
    """
    manifest = manifest or {}
    mir = manifest.get("mir") or {}
    sources = manifest.get("sources") or {}
    lrc = manifest.get("lrcAlign") or {}
    request = manifest.get("request") or {}
    source_ids = sources.get("ids") or sources.get("sourceIds") or []
    projection = {
        "mir": {
            "analyzedAt": mir.get("analyzedAt"),
            "engines": mir.get("engines") or {},
        },
        "sources": {
            "ids": sorted(str(value) for value in source_ids),
            "gatheredAt": sources.get("gatheredAt"),
        },
        "lrcAlign": {"fingerprint": lrc.get("fingerprint")},
        "scope": request.get("scope", manifest.get("scope")),
        "guidance": request.get("guidance", request.get("notes", manifest.get("guidance"))),
    }
    canonical = json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def idempotency_key(song_id: str, config_version: str, manifest: dict | None) -> tuple[str, str]:
    digest = evidence_hash(manifest)
    raw = f"{song_id}{config_version}{digest}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), digest


@dataclass(frozen=True)
class Admission:
    key: str
    evidence_hash: str
    run_id: str
    forced: bool = False
    force_reason: str | None = None


class DuplicateRunError(RuntimeError):
    """A stable machine-readable refusal shared by REST and MCP surfaces."""

    code = "duplicate_run"

    def __init__(self, blocking_run_id: str, summary: dict | None = None):
        self.blocking_run_id = blocking_run_id
        self.summary = deepcopy(summary) if summary else None
        payload: dict = {"code": self.code, "blockingRunId": blocking_run_id}
        if self.summary is not None:
            payload["existingRun"] = self.summary
        super().__init__(json.dumps(payload, sort_keys=True, separators=(",", ":")))


class RunAdmissionRepository:
    def admit(
        self,
        *,
        key: str,
        evidence_hash: str,
        run_id: str,
        lease_seconds: float,
        completed_ttl_seconds: float,
        force: bool = False,
        force_reason: str | None = None,
        now: datetime | None = None,
    ) -> Admission:
        raise NotImplementedError

    def complete(
        self, admission: Admission, *, retry: bool, summary: dict, now: datetime | None = None
    ) -> None:
        raise NotImplementedError

    def abandon(self, admission: Admission) -> None:
        raise NotImplementedError


def _admission_doc(
    *, admission: Admission, now: datetime, lease_seconds: float
) -> dict:
    return {
        "idempotencyKey": admission.key,
        "evidenceHash": admission.evidence_hash,
        "runId": admission.run_id,
        "status": "in_flight",
        "startedAt": now,
        "leaseExpiresAt": now + timedelta(seconds=lease_seconds),
        "forced": admission.forced,
        "forceReason": admission.force_reason,
    }


def _blocking(doc: dict | None, *, now: datetime, completed_ttl_seconds: float) -> bool:
    if not doc:
        return False
    if doc.get("status") == "in_flight":
        expires = _as_utc(doc.get("leaseExpiresAt"))
        return expires is not None and expires > now
    if doc.get("status") == "completed" and doc.get("retry") is False:
        completed = _as_utc(doc.get("completedAt"))
        return completed is not None and completed + timedelta(seconds=completed_ttl_seconds) > now
    return False


class InMemoryRunAdmissionRepository(RunAdmissionRepository):
    def __init__(self) -> None:
        self._locks: dict[str, dict] = {}
        self._lock = threading.Lock()

    def admit(
        self,
        *,
        key: str,
        evidence_hash: str,
        run_id: str,
        lease_seconds: float,
        completed_ttl_seconds: float,
        force: bool = False,
        force_reason: str | None = None,
        now: datetime | None = None,
    ) -> Admission:
        now = (now or _utcnow()).astimezone(timezone.utc)
        if force and not (force_reason or "").strip():
            raise ValueError("forceReason is required when force=true")
        admission = Admission(key, evidence_hash, run_id, force, (force_reason or "").strip() or None)
        with self._lock:
            current = self._locks.get(key)
            if not force and _blocking(current, now=now, completed_ttl_seconds=completed_ttl_seconds):
                raise DuplicateRunError(current["runId"], current.get("summary"))
            self._locks[key] = _admission_doc(
                admission=admission, now=now, lease_seconds=lease_seconds
            )
        return admission

    def complete(
        self, admission: Admission, *, retry: bool, summary: dict, now: datetime | None = None
    ) -> None:
        now = (now or _utcnow()).astimezone(timezone.utc)
        with self._lock:
            current = self._locks.get(admission.key)
            if not current or current.get("runId") != admission.run_id:
                return
            current.update(
                status="completed", completedAt=now, retry=bool(retry), summary=deepcopy(summary)
            )

    def abandon(self, admission: Admission) -> None:
        with self._lock:
            current = self._locks.get(admission.key)
            if current and current.get("runId") == admission.run_id:
                del self._locks[admission.key]


class FirestoreRunAdmissionRepository(RunAdmissionRepository):
    _COLLECTION = "run-locks"

    def __init__(self, project: str | None = None, database: str = "(default)") -> None:
        from google.cloud import firestore

        kwargs: dict = {}
        if project:
            kwargs["project"] = project
        if database and database != "(default)":
            kwargs["database"] = database
        self._client = firestore.Client(**kwargs)
        self._firestore = firestore
        log.info("Firestore run admission ready: collection=%s", self._COLLECTION)

    def admit(
        self,
        *,
        key: str,
        evidence_hash: str,
        run_id: str,
        lease_seconds: float,
        completed_ttl_seconds: float,
        force: bool = False,
        force_reason: str | None = None,
        now: datetime | None = None,
    ) -> Admission:
        now = (now or _utcnow()).astimezone(timezone.utc)
        if force and not (force_reason or "").strip():
            raise ValueError("forceReason is required when force=true")
        admission = Admission(key, evidence_hash, run_id, force, (force_reason or "").strip() or None)
        ref = self._client.collection(self._COLLECTION).document(key)
        transaction = self._client.transaction()

        @self._firestore.transactional
        def transact(txn):
            snap = ref.get(transaction=txn)
            current = snap.to_dict() if snap.exists else None
            if not force and _blocking(
                current, now=now, completed_ttl_seconds=completed_ttl_seconds
            ):
                raise DuplicateRunError(current["runId"], current.get("summary"))
            txn.set(ref, _admission_doc(admission=admission, now=now, lease_seconds=lease_seconds))

        transact(transaction)
        return admission

    def complete(
        self, admission: Admission, *, retry: bool, summary: dict, now: datetime | None = None
    ) -> None:
        now = (now or _utcnow()).astimezone(timezone.utc)
        ref = self._client.collection(self._COLLECTION).document(admission.key)
        transaction = self._client.transaction()

        @self._firestore.transactional
        def transact(txn):
            snap = ref.get(transaction=txn)
            current = snap.to_dict() if snap.exists else None
            if not current or current.get("runId") != admission.run_id:
                return
            txn.update(
                ref,
                {
                    "status": "completed",
                    "completedAt": now,
                    "retry": bool(retry),
                    "summary": deepcopy(summary),
                },
            )

        transact(transaction)

    def abandon(self, admission: Admission) -> None:
        ref = self._client.collection(self._COLLECTION).document(admission.key)
        transaction = self._client.transaction()

        @self._firestore.transactional
        def transact(txn):
            snap = ref.get(transaction=txn)
            current = snap.to_dict() if snap.exists else None
            if current and current.get("runId") == admission.run_id:
                txn.delete(ref)

        transact(transaction)


_repo: RunAdmissionRepository | None = None
_repo_lock = threading.Lock()


def build_run_admission_repository() -> RunAdmissionRepository:
    backend, project = _resolve_backend()
    if backend == "firestore":
        from ..config import settings

        return FirestoreRunAdmissionRepository(
            project=project, database=settings.firestore_database
        )
    return InMemoryRunAdmissionRepository()


def get_run_admission_store() -> RunAdmissionRepository:
    global _repo
    if _repo is None:
        with _repo_lock:
            if _repo is None:
                _repo = build_run_admission_repository()
    return _repo


def reset_run_admission_store() -> None:
    global _repo
    with _repo_lock:
        _repo = None
