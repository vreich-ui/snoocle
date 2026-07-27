"""Batch analysis queue (master plan D2).

Wolf's actual workflow is "here are twenty songs, go", not "sit and watch one
song analyze". This is that: a submitted list becomes queued job records, one
in-process asyncio worker drains them through the SAME pipeline the single-song
endpoint uses, and the admin polls for statuses.

Deliberately small and deliberately in-process:

- **One worker, strictly sequential.** The pipeline is CPU-bound (MIR) and hits
  rate-limited APIs (the reconciler, YouTube). Running two at once makes both
  slower and doubles the odds of a 429. Sequential is the honest choice.
- **A failed job never stalls the queue.** Every job is wrapped; a failure marks
  that job `error` with the message and the worker moves straight to the next.
- **State is in memory.** These are ephemeral jobs, not durable records — the
  durable artifacts are the songs and run traces the pipeline already writes.
  A restart loses the pending queue, which is why `GET /v1/queue` reports the
  worker's liveness and the deploy docs call for ``--min-instances=1``.

The queue holds no locks while a job runs, so submitting and listing stay
responsive through a multi-minute analysis.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

log = logging.getLogger(__name__)

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
ERROR = "error"
CANCELLED = "cancelled"

TERMINAL = {DONE, ERROR, CANCELLED}

# Refuse absurd submissions outright rather than accepting them and then
# spending an hour discovering the caller pasted their clipboard.
MAX_ITEMS_PER_SUBMIT = 100


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class QueueJob:
    id: str
    title: Optional[str] = None
    artist: Optional[str] = None
    youtube_url_or_id: Optional[str] = None
    provider: Optional[str] = None
    analysis_depth: Optional[str] = None
    status: str = QUEUED
    queued_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    song_id: Optional[str] = None
    run_id: Optional[str] = None
    stored_version: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0

    @property
    def label(self) -> str:
        """How the job reads in a list before it has resolved to a song."""
        if self.title and self.artist:
            return f"{self.artist} — {self.title}"
        return self.title or self.youtube_url_or_id or self.id

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "title": self.title,
            "artist": self.artist,
            "youtubeUrlOrId": self.youtube_url_or_id,
            "provider": self.provider,
            "analysisDepth": self.analysis_depth,
            "status": self.status,
            "queuedAt": self.queued_at,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "songId": self.song_id,
            "runId": self.run_id,
            "storedVersion": self.stored_version,
            "error": self.error,
            "attempts": self.attempts,
        }


def parse_batch_line(line: str) -> Optional[dict]:
    """One line of the admin's "add many" textarea -> a job spec.

    Accepted, in this order:
      - a YouTube URL or bare 11-char id
      - ``Artist - Title`` (also ``Artist – Title`` / ``Artist — Title``)
      - anything else is treated as a title with no artist, which the pipeline
        can still work with when it finds the song by search.

    Returns None for a blank or comment line.
    """
    line = (line or "").strip()
    if not line or line.startswith("#"):
        return None

    lowered = line.lower()
    if lowered.startswith(("http://", "https://")) or "youtu" in lowered:
        return {"youtubeUrlOrId": line}
    # A bare video id: exactly 11 chars of the YouTube alphabet.
    if len(line) == 11 and all(c.isalnum() or c in "-_" for c in line):
        return {"youtubeUrlOrId": line}

    for dash in (" — ", " – ", " - "):
        if dash in line:
            artist, title = line.split(dash, 1)
            artist, title = artist.strip(), title.strip()
            if artist and title:
                return {"title": title, "artist": artist}
    return {"title": line}


class BatchQueue:
    """FIFO job queue with a single background worker."""

    def __init__(self) -> None:
        self._jobs: dict[str, QueueJob] = {}
        self._order: list[str] = []
        self._pending: asyncio.Queue[str] | None = None
        self._worker: asyncio.Task | None = None
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._current: Optional[str] = None

    # --- lifecycle ------------------------------------------------------

    def start(self) -> None:
        """Called from the app lifespan. Idempotent."""
        if self._worker is not None and not self._worker.done():
            return
        self._pending = asyncio.Queue()
        # Re-arm anything left queued from a previous worker generation.
        with self._lock:
            for job_id in self._order:
                if self._jobs[job_id].status == QUEUED:
                    self._pending.put_nowait(job_id)
        self._worker = asyncio.create_task(self._run(), name="snoocle-batch-worker")

    async def stop(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        try:
            await self._worker
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        self._worker = None

    @property
    def worker_alive(self) -> bool:
        return self._worker is not None and not self._worker.done()

    # --- submission -----------------------------------------------------

    def submit(self, specs: list[dict], provider: str | None = None,
               analysis_depth: str | None = None) -> list[QueueJob]:
        if len(specs) > MAX_ITEMS_PER_SUBMIT:
            raise ValueError(
                f"too many items: {len(specs)} (max {MAX_ITEMS_PER_SUBMIT} per submit)"
            )
        created: list[QueueJob] = []
        with self._lock:
            for spec in specs:
                job = QueueJob(
                    id=f"q{next(self._ids):04d}",
                    title=spec.get("title"),
                    artist=spec.get("artist"),
                    youtube_url_or_id=spec.get("youtubeUrlOrId"),
                    provider=spec.get("provider") or provider,
                    analysis_depth=spec.get("analysisDepth") or analysis_depth,
                )
                self._jobs[job.id] = job
                self._order.append(job.id)
                created.append(job)
        for job in created:
            self._enqueue(job.id)
        return created

    def _enqueue(self, job_id: str) -> None:
        if self._pending is None:
            # No running loop yet (submitted before lifespan start, e.g. in a
            # sync test): the job stays QUEUED and start() will pick it up.
            return
        self._pending.put_nowait(job_id)

    def retry(self, job_id: str) -> QueueJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status not in TERMINAL:
                raise ValueError(f"job {job_id} is {job.status}, not finished")
            job.status = QUEUED
            job.error = None
            job.started_at = None
            job.finished_at = None
            job.queued_at = _now()
        self._enqueue(job_id)
        return job

    def cancel(self, job_id: str) -> QueueJob:
        """Cancel a job that hasn't started. A running job is left alone —
        killing a half-written analysis mid-store would be worse."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status != QUEUED:
                raise ValueError(f"job {job_id} is {job.status}; only queued jobs cancel")
            job.status = CANCELLED
            job.finished_at = _now()
            return job

    # --- reads ----------------------------------------------------------

    def list_jobs(self, limit: int = 200) -> list[QueueJob]:
        with self._lock:
            ids = self._order[-limit:]
            return [self._jobs[i] for i in ids]

    def get(self, job_id: str) -> QueueJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def stats(self) -> dict:
        with self._lock:
            counts: dict[str, int] = {}
            for job in self._jobs.values():
                counts[job.status] = counts.get(job.status, 0) + 1
        return {
            "workerAlive": self.worker_alive,
            "current": self._current,
            "counts": counts,
        }

    def clear_finished(self) -> int:
        with self._lock:
            keep = [i for i in self._order if self._jobs[i].status not in TERMINAL]
            removed = len(self._order) - len(keep)
            for i in self._order:
                if i not in keep:
                    self._jobs.pop(i, None)
            self._order = keep
        return removed

    # --- the worker -----------------------------------------------------

    async def _run(self) -> None:
        assert self._pending is not None
        while True:
            job_id = await self._pending.get()
            job = self.get(job_id)
            if job is None or job.status != QUEUED:
                continue  # cancelled, or re-queued twice
            await self._process(job)

    async def _process(self, job: QueueJob) -> None:
        from .pipeline import run_pipeline_async

        with self._lock:
            job.status = RUNNING
            job.started_at = _now()
            job.attempts += 1
            self._current = job.id
        try:
            report = await run_pipeline_async(
                job.title,
                job.artist,
                youtube_url_or_id=job.youtube_url_or_id,
                provider=job.provider,
                analysis_depth=job.analysis_depth,
            )
            with self._lock:
                job.song_id = report.song_id
                job.run_id = report.run_id
                job.stored_version = report.stored_version
                job.status = DONE
        except asyncio.CancelledError:
            with self._lock:
                job.status = QUEUED  # shutting down: leave it for next start()
                self._current = None
            raise
        except Exception as e:  # noqa: BLE001
            # One bad song must not stall nineteen good ones.
            log.warning("batch job %s failed: %s", job.id, e)
            with self._lock:
                job.status = ERROR
                job.error = str(e)[:2000]
        finally:
            with self._lock:
                job.finished_at = _now()
                if self._current == job.id:
                    self._current = None


_queue: BatchQueue | None = None
_queue_lock = threading.Lock()


def get_queue() -> BatchQueue:
    global _queue
    if _queue is None:
        with _queue_lock:
            if _queue is None:
                _queue = BatchQueue()
    return _queue


def reset_queue() -> None:
    global _queue
    with _queue_lock:
        _queue = None


def to_json(jobs: list[QueueJob]) -> list[dict]:
    return [j.to_json() for j in jobs]


__all__: list[str] = [
    "BatchQueue",
    "QueueJob",
    "get_queue",
    "reset_queue",
    "parse_batch_line",
    "to_json",
    "QUEUED",
    "RUNNING",
    "DONE",
    "ERROR",
    "CANCELLED",
    "MAX_ITEMS_PER_SUBMIT",
]
