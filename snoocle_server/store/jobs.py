"""Durable analysis-job queue — the broker half of the Mac-worker split.

Phase D shipped an in-process asyncio worker. That worked, but it forced the
Cloud Run service into instance-based billing with a warm instance
(`--no-cpu-throttling --min-instances=1`, ~$53/month) purely so background work
could keep a CPU, and it put a 6-8 GB ML image on a machine that is slower at
the work than the laptop already on the desk.

So the server stops being a worker and becomes a **broker**: jobs live here,
and whoever is willing does them.

    server            worker (Wolf's Mac, launchd)
    ------            ----------------------------
    queued  <-------- POST /v1/queue/claim      (lease it)
    leased  <-------- POST /v1/queue/{id}/heartbeat
    done    <-------- POST /v1/queue/{id}/complete
    error   <-------- POST /v1/queue/{id}/fail

The worker only ever makes OUTBOUND https requests, so nothing has to be
opened, tunnelled or port-forwarded on the Mac.

**Leases, not assignments.** A claim is a time-boxed lease. A worker that dies,
sleeps or loses its network stops heartbeating, the lease expires, and the job
returns to `queued` for anyone to pick up. That is the only mechanism that
makes "the laptop lid closed mid-song" a non-event, and it is enforced on read
(:func:`JobRepository.list_jobs` reclaims lazily) so it needs no cron, no
scheduler, and no always-on CPU anywhere.

Both backends implement identical semantics; Firestore uses a transaction for
the claim so two workers can never lease the same job.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from . import _resolve_backend

QUEUED = "queued"
LEASED = "leased"
DONE = "done"
ERROR = "error"
CANCELLED = "cancelled"

TERMINAL = frozenset({DONE, ERROR, CANCELLED})

# How long a claim is good for without a heartbeat. Comfortably longer than the
# worker's heartbeat interval (30s) but well short of a human's patience — a
# job stuck behind a sleeping laptop should be re-runnable within minutes.
DEFAULT_LEASE_SECONDS = 300

# A job that has failed this many times stops being retried automatically.
# Retrying a genuinely broken song forever would starve the queue and hide the
# failure; the admin can still retry it by hand.
MAX_ATTEMPTS = 3


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class Job:
    """One song waiting to be analyzed."""

    id: str
    title: Optional[str] = None
    artist: Optional[str] = None
    youtube_url_or_id: Optional[str] = None
    provider: Optional[str] = None
    analysis_depth: Optional[str] = None
    # Extra work the worker should do beyond the reconcile pipeline — e.g.
    # ["stems"]. Named capabilities rather than booleans so a worker can
    # advertise what it can actually do and the broker can match.
    wants: list[str] = field(default_factory=list)

    status: str = QUEUED
    queued_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    # lease
    worker: Optional[str] = None
    lease_expires_at: Optional[str] = None
    heartbeat_at: Optional[str] = None

    # result
    song_id: Optional[str] = None
    run_id: Optional[str] = None
    stored_version: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0

    @property
    def label(self) -> str:
        if self.title and self.artist:
            return f"{self.artist} — {self.title}"
        return self.title or self.youtube_url_or_id or self.id

    def lease_expired(self, at: Optional[datetime] = None) -> bool:
        if self.status != LEASED:
            return False
        expiry = _parse(self.lease_expires_at)
        return expiry is None or expiry <= (at or _now())

    def to_json(self) -> dict:
        data = asdict(self)
        return {
            "id": data["id"],
            "label": self.label,
            "title": data["title"],
            "artist": data["artist"],
            "youtubeUrlOrId": data["youtube_url_or_id"],
            "provider": data["provider"],
            "analysisDepth": data["analysis_depth"],
            "wants": data["wants"],
            "status": data["status"],
            "queuedAt": data["queued_at"],
            "startedAt": data["started_at"],
            "finishedAt": data["finished_at"],
            "worker": data["worker"],
            "leaseExpiresAt": data["lease_expires_at"],
            "heartbeatAt": data["heartbeat_at"],
            "songId": data["song_id"],
            "runId": data["run_id"],
            "storedVersion": data["stored_version"],
            "error": data["error"],
            "attempts": data["attempts"],
        }

    def to_worker_json(self) -> dict:
        """What a claiming worker needs — the instruction, not the bookkeeping."""
        return {
            "id": self.id,
            "title": self.title,
            "artist": self.artist,
            "youtubeUrlOrId": self.youtube_url_or_id,
            "provider": self.provider,
            "analysisDepth": self.analysis_depth,
            "wants": self.wants,
            "leaseExpiresAt": self.lease_expires_at,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class JobRepository:
    """Broker contract. Implementations must make `claim` atomic."""

    # --- writes from the admin ------------------------------------------

    def submit(self, specs: list[dict], provider: str | None = None,
               analysis_depth: str | None = None,
               wants: list[str] | None = None) -> list[Job]:
        raise NotImplementedError

    def retry(self, job_id: str) -> Job:
        raise NotImplementedError

    def cancel(self, job_id: str) -> Job:
        raise NotImplementedError

    def clear_finished(self) -> int:
        raise NotImplementedError

    # --- writes from a worker -------------------------------------------

    def claim(self, worker: str, capabilities: list[str] | None = None,
              lease_seconds: int = DEFAULT_LEASE_SECONDS) -> Job | None:
        raise NotImplementedError

    def heartbeat(self, job_id: str, worker: str,
                  lease_seconds: int = DEFAULT_LEASE_SECONDS) -> Job:
        raise NotImplementedError

    def complete(self, job_id: str, worker: str, *, song_id: str,
                 run_id: str | None = None, stored_version: str | None = None) -> Job:
        raise NotImplementedError

    def fail(self, job_id: str, worker: str, error: str, *,
             retry: bool = True) -> Job:
        raise NotImplementedError

    # --- reads -----------------------------------------------------------

    def list_jobs(self, limit: int = 200) -> list[Job]:
        raise NotImplementedError

    def get(self, job_id: str) -> Job | None:
        raise NotImplementedError

    def stats(self) -> dict:
        jobs = self.list_jobs(limit=1000)
        counts: dict[str, int] = {}
        for job in jobs:
            counts[job.status] = counts.get(job.status, 0) + 1

        # "Is a worker there?" is answered by evidence, not by a flag: the most
        # recent heartbeat across live leases. A dashboard that claimed a
        # worker was connected because a boolean said so would be exactly the
        # lie this design exists to avoid.
        latest: Optional[datetime] = None
        current: Optional[str] = None
        for job in jobs:
            beat = _parse(job.heartbeat_at)
            if beat and (latest is None or beat > latest):
                latest = beat
                current = job.worker
        return {
            "counts": counts,
            "lastHeartbeatAt": _iso(latest),
            "lastWorker": current,
            "workerSeenRecently": bool(
                latest and (_now() - latest) < timedelta(seconds=DEFAULT_LEASE_SECONDS)
            ),
            "leaseSeconds": DEFAULT_LEASE_SECONDS,
            "maxAttempts": MAX_ATTEMPTS,
        }


class WrongWorkerError(RuntimeError):
    """A worker tried to act on a job it does not hold the lease for."""


class UnknownJobError(KeyError):
    pass


class _Shared:
    """Backend-agnostic state transitions, so the two implementations cannot
    drift on the rules (only on how they persist them)."""

    @staticmethod
    def apply_claim(job: Job, worker: str, lease_seconds: int) -> Job:
        now = _now()
        job.status = LEASED
        job.worker = worker
        job.started_at = job.started_at or _iso(now)
        job.heartbeat_at = _iso(now)
        job.lease_expires_at = _iso(now + timedelta(seconds=lease_seconds))
        job.attempts += 1
        job.error = None
        return job

    @staticmethod
    def apply_reclaim(job: Job) -> Job:
        """An expired lease returns the job to the pool — unless it has burned
        through its attempts, in which case it is parked as an error so the
        queue moves on and the failure stays visible."""
        if job.attempts >= MAX_ATTEMPTS:
            job.status = ERROR
            job.error = (
                f"lease expired {job.attempts} times (worker stopped responding); "
                "not retried automatically"
            )
            job.finished_at = _iso(_now())
        else:
            job.status = QUEUED
            job.error = "lease expired — worker stopped responding; requeued"
        job.worker = None
        job.lease_expires_at = None
        return job

    @staticmethod
    def check_holder(job: Job, worker: str) -> None:
        if job.status != LEASED or job.worker != worker:
            raise WrongWorkerError(
                f"job {job.id} is {job.status}"
                + (f" held by {job.worker!r}" if job.worker else "")
                + f", not leased to {worker!r}"
            )

    @staticmethod
    def matches(job: Job, capabilities: list[str]) -> bool:
        """A worker only gets a job whose `wants` it can satisfy."""
        return all(w in capabilities for w in (job.wants or []))


# ---------------------------------------------------------------------------
# In-memory backend (tests, local dev, and any run without a GCP project)
# ---------------------------------------------------------------------------


class InMemoryJobRepository(JobRepository):
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._seq = 0
        self._lock = threading.RLock()

    def _next_id(self) -> str:
        self._seq += 1
        return f"q{self._seq:04d}"

    def submit(self, specs, provider=None, analysis_depth=None, wants=None):
        created: list[Job] = []
        with self._lock:
            for spec in specs:
                job = Job(
                    id=self._next_id(),
                    title=spec.get("title"),
                    artist=spec.get("artist"),
                    youtube_url_or_id=spec.get("youtubeUrlOrId"),
                    provider=spec.get("provider") or provider,
                    analysis_depth=spec.get("analysisDepth") or analysis_depth,
                    wants=list(spec.get("wants") or wants or []),
                    queued_at=_iso(_now()) or "",
                )
                self._jobs[job.id] = job
                self._order.append(job.id)
                created.append(job)
        return created

    def _reclaim_expired(self) -> None:
        for job in self._jobs.values():
            if job.lease_expired():
                _Shared.apply_reclaim(job)

    def claim(self, worker, capabilities=None, lease_seconds=DEFAULT_LEASE_SECONDS):
        caps = list(capabilities or [])
        with self._lock:
            self._reclaim_expired()
            for job_id in self._order:
                job = self._jobs[job_id]
                if job.status == QUEUED and _Shared.matches(job, caps):
                    return _Shared.apply_claim(job, worker, lease_seconds)
        return None

    def heartbeat(self, job_id, worker, lease_seconds=DEFAULT_LEASE_SECONDS):
        with self._lock:
            job = self._require(job_id)
            _Shared.check_holder(job, worker)
            now = _now()
            job.heartbeat_at = _iso(now)
            job.lease_expires_at = _iso(now + timedelta(seconds=lease_seconds))
            return job

    def complete(self, job_id, worker, *, song_id, run_id=None, stored_version=None):
        with self._lock:
            job = self._require(job_id)
            _Shared.check_holder(job, worker)
            job.status = DONE
            job.song_id = song_id
            job.run_id = run_id
            job.stored_version = stored_version
            job.error = None
            job.finished_at = _iso(_now())
            job.lease_expires_at = None
            return job

    def fail(self, job_id, worker, error, *, retry=True):
        with self._lock:
            job = self._require(job_id)
            _Shared.check_holder(job, worker)
            job.error = str(error)[:2000]
            job.lease_expires_at = None
            job.worker = None
            if retry and job.attempts < MAX_ATTEMPTS:
                job.status = QUEUED
            else:
                job.status = ERROR
                job.finished_at = _iso(_now())
            return job

    def retry(self, job_id):
        with self._lock:
            job = self._require(job_id)
            if job.status not in TERMINAL:
                raise ValueError(f"job {job_id} is {job.status}, not finished")
            job.status = QUEUED
            job.error = None
            job.worker = None
            job.started_at = None
            job.finished_at = None
            job.lease_expires_at = None
            job.attempts = 0  # a human asked for this; give it a full budget
            job.queued_at = _iso(_now()) or ""
            return job

    def cancel(self, job_id):
        with self._lock:
            job = self._require(job_id)
            if job.status != QUEUED:
                raise ValueError(f"job {job_id} is {job.status}; only queued jobs cancel")
            job.status = CANCELLED
            job.finished_at = _iso(_now())
            return job

    def clear_finished(self):
        with self._lock:
            keep = [i for i in self._order if self._jobs[i].status not in TERMINAL]
            removed = len(self._order) - len(keep)
            for i in set(self._order) - set(keep):
                self._jobs.pop(i, None)
            self._order = keep
            return removed

    def list_jobs(self, limit=200):
        with self._lock:
            self._reclaim_expired()
            return [self._jobs[i] for i in self._order[-limit:]]

    def get(self, job_id):
        with self._lock:
            self._reclaim_expired()
            return self._jobs.get(job_id)

    def _require(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise UnknownJobError(job_id)
        return job


# ---------------------------------------------------------------------------
# Firestore backend (durable — this is what Cloud Run uses)
# ---------------------------------------------------------------------------


class FirestoreJobRepository(JobRepository):
    """Jobs in a `song_jobs` collection.

    The only genuinely delicate operation is `claim`: two workers polling at
    the same moment must not both get the same job. That is done inside a
    Firestore transaction — read the candidate, verify it is still queued,
    write the lease — so the loser sees the conflict and simply asks again.
    """

    _COLLECTION = "song_jobs"

    def __init__(self, project: str | None = None, database: str = "(default)") -> None:
        from google.cloud import firestore

        kwargs: dict = {}
        if project:
            kwargs["project"] = project
        if database and database != "(default)":
            kwargs["database"] = database
        self._client = firestore.Client(**kwargs)
        self._firestore = firestore
        self._seq_lock = threading.Lock()

    @property
    def _col(self):
        return self._client.collection(self._COLLECTION)

    def _doc(self, job_id: str):
        return self._col.document(job_id)

    def _save(self, job: Job) -> Job:
        self._doc(job.id).set(asdict(job))
        return job

    def _load(self, job_id: str) -> Job:
        snap = self._doc(job_id).get()
        if not snap.exists:
            raise UnknownJobError(job_id)
        return Job.from_dict(snap.to_dict() or {})

    # --- admin writes ----------------------------------------------------

    def submit(self, specs, provider=None, analysis_depth=None, wants=None):
        created: list[Job] = []
        # Ids are time-ordered so the natural document order is submission
        # order without needing an index on a separate field.
        stamp = _now().strftime("%Y%m%d%H%M%S%f")
        with self._seq_lock:
            for i, spec in enumerate(specs):
                job = Job(
                    id=f"{stamp}-{i:03d}",
                    title=spec.get("title"),
                    artist=spec.get("artist"),
                    youtube_url_or_id=spec.get("youtubeUrlOrId"),
                    provider=spec.get("provider") or provider,
                    analysis_depth=spec.get("analysisDepth") or analysis_depth,
                    wants=list(spec.get("wants") or wants or []),
                    queued_at=_iso(_now()) or "",
                )
                created.append(self._save(job))
        return created

    def retry(self, job_id):
        job = self._load(job_id)
        if job.status not in TERMINAL:
            raise ValueError(f"job {job_id} is {job.status}, not finished")
        job.status = QUEUED
        job.error = None
        job.worker = None
        job.started_at = None
        job.finished_at = None
        job.lease_expires_at = None
        job.attempts = 0
        job.queued_at = _iso(_now()) or ""
        return self._save(job)

    def cancel(self, job_id):
        job = self._load(job_id)
        if job.status != QUEUED:
            raise ValueError(f"job {job_id} is {job.status}; only queued jobs cancel")
        job.status = CANCELLED
        job.finished_at = _iso(_now())
        return self._save(job)

    def clear_finished(self):
        removed = 0
        for snap in self._col.stream():
            job = Job.from_dict(snap.to_dict() or {})
            if job.status in TERMINAL:
                snap.reference.delete()
                removed += 1
        return removed

    # --- worker writes ---------------------------------------------------

    def claim(self, worker, capabilities=None, lease_seconds=DEFAULT_LEASE_SECONDS):
        caps = list(capabilities or [])
        for job in self._all(reclaim=True):
            if job.status != QUEUED or not _Shared.matches(job, caps):
                continue
            claimed = self._try_claim(job.id, worker, caps, lease_seconds)
            if claimed is not None:
                return claimed
            # Lost the race — keep scanning rather than failing the poll.
        return None

    def _try_claim(self, job_id: str, worker: str, caps: list[str],
                   lease_seconds: int) -> Job | None:
        doc = self._doc(job_id)

        @self._firestore.transactional
        def _txn(transaction):
            snap = doc.get(transaction=transaction)
            if not snap.exists:
                return None
            job = Job.from_dict(snap.to_dict() or {})
            if job.lease_expired():
                _Shared.apply_reclaim(job)
            if job.status != QUEUED or not _Shared.matches(job, caps):
                return None  # somebody else got there first
            _Shared.apply_claim(job, worker, lease_seconds)
            transaction.set(doc, asdict(job))
            return job

        return _txn(self._client.transaction())

    def heartbeat(self, job_id, worker, lease_seconds=DEFAULT_LEASE_SECONDS):
        job = self._load(job_id)
        _Shared.check_holder(job, worker)
        now = _now()
        job.heartbeat_at = _iso(now)
        job.lease_expires_at = _iso(now + timedelta(seconds=lease_seconds))
        return self._save(job)

    def complete(self, job_id, worker, *, song_id, run_id=None, stored_version=None):
        job = self._load(job_id)
        _Shared.check_holder(job, worker)
        job.status = DONE
        job.song_id = song_id
        job.run_id = run_id
        job.stored_version = stored_version
        job.error = None
        job.finished_at = _iso(_now())
        job.lease_expires_at = None
        return self._save(job)

    def fail(self, job_id, worker, error, *, retry=True):
        job = self._load(job_id)
        _Shared.check_holder(job, worker)
        job.error = str(error)[:2000]
        job.lease_expires_at = None
        job.worker = None
        if retry and job.attempts < MAX_ATTEMPTS:
            job.status = QUEUED
        else:
            job.status = ERROR
            job.finished_at = _iso(_now())
        return self._save(job)

    # --- reads -----------------------------------------------------------

    def _all(self, reclaim: bool = False) -> list[Job]:
        jobs = [Job.from_dict(s.to_dict() or {}) for s in self._col.stream()]
        jobs.sort(key=lambda j: j.id)
        if reclaim:
            for job in jobs:
                if job.lease_expired():
                    self._save(_Shared.apply_reclaim(job))
        return jobs

    def list_jobs(self, limit=200):
        return self._all(reclaim=True)[-limit:]

    def get(self, job_id):
        try:
            job = self._load(job_id)
        except UnknownJobError:
            return None
        if job.lease_expired():
            self._save(_Shared.apply_reclaim(job))
        return job


# ---------------------------------------------------------------------------
# Resolution — same backend selection as the song and run stores
# ---------------------------------------------------------------------------

_repo: JobRepository | None = None
_repo_lock = threading.Lock()


def build_job_repository() -> JobRepository:
    backend, project = _resolve_backend()
    if backend == "firestore":
        from ..config import settings

        return FirestoreJobRepository(project=project, database=settings.firestore_database)
    return InMemoryJobRepository()


def get_job_store() -> JobRepository:
    global _repo
    if _repo is None:
        with _repo_lock:
            if _repo is None:
                _repo = build_job_repository()
    return _repo


def reset_job_store() -> None:
    global _repo
    with _repo_lock:
        _repo = None
