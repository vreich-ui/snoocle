"""The worker loop, tested against the real broker over a real HTTP client.

No mocked transport: the worker talks to the actual FastAPI app through
`TestClient`, so what's exercised is the true protocol — status codes,
payload shapes and all. Only the pipeline itself is stubbed, since running
madmom on a fixture is a different test's job.

The cases that matter are the ones a laptop actually produces: a queue that is
empty, a song that is broken, and a lid that closes mid-analysis.
"""

from __future__ import annotations

from datetime import timedelta

import httpx
import pytest
from fastapi.testclient import TestClient

import snoocle_server.api as api_mod
from snoocle_server.api import app
from snoocle_server.store.jobs import (
    DONE,
    ERROR,
    QUEUED,
    InMemoryJobRepository,
    _now,
)
from snoocle_server.worker import (
    Worker,
    WorkerConfig,
    _is_worth_retrying,
    default_worker_name,
    detect_capabilities,
)


@pytest.fixture()
def store():
    return InMemoryJobRepository()


@pytest.fixture()
def server(monkeypatch, store):
    monkeypatch.setattr(api_mod, "get_job_store", lambda: store)
    return TestClient(app)


def make_worker(server, store, run_job, **overrides) -> Worker:
    """A Worker wired to the in-process app instead of the network."""
    config = WorkerConfig(
        base_url="http://testserver",
        name=overrides.pop("name", "test-mac"),
        capabilities=overrides.pop("capabilities", ["analyze"]),
        once=overrides.pop("once", True),
        idle_poll_seconds=0,
        **overrides,
    )
    worker = Worker(config, run_job=run_job)
    # Swap the real network client for one bound to the in-process ASGI app.
    # Everything above the transport — paths, payloads, status codes — is the
    # genuine protocol.
    worker.client.close()
    worker.client = httpx.Client(
        transport=server._transport,
        base_url="http://testserver",
        headers={"User-Agent": "snoocle-worker/test"},
    )
    return worker


# --- capability detection ---------------------------------------------------


def test_capabilities_always_include_analyze_and_are_import_proven():
    caps = detect_capabilities()
    assert "analyze" in caps
    # Over-claiming means claiming jobs the machine then fails, so each extra
    # capability must correspond to a module that genuinely imports.
    for cap, module in (("stems", "demucs"), ("align", "whisperx"), ("melody", "torchcrepe")):
        if cap in caps:
            __import__(module)


def test_worker_name_is_human_recognisable():
    assert default_worker_name()


def test_a_worker_without_a_server_url_refuses_to_start():
    with pytest.raises(ValueError, match="SNOOCLE_SERVER_URL"):
        Worker(WorkerConfig(base_url=""))


# --- retry classification ---------------------------------------------------


@pytest.mark.parametrize("message,expected", [
    ("connection reset by peer", True),
    ("rate limited, try again", True),
    ("timed out", True),
    ("Video unavailable", False),
    ("this video has been removed", False),
    ("no candidate sources found", False),
    ("private video", False),
])
def test_retry_classification(message, expected):
    assert _is_worth_retrying(RuntimeError(message)) is expected


# --- the happy path ---------------------------------------------------------


def test_a_claimed_job_runs_and_is_reported_complete(server, store):
    seen = []

    def run_job(job):
        seen.append(job)
        return {"songId": "a--t", "runId": "r1", "storedVersion": "v1"}

    store.submit([{"title": "T", "artist": "A"}])
    worker = make_worker(server, store, run_job)
    assert worker.run_forever() == 1

    assert seen[0]["title"] == "T"
    job = store.list_jobs()[0]
    assert job.status == DONE
    assert (job.song_id, job.run_id, job.stored_version) == ("a--t", "r1", "v1")


def test_an_empty_queue_is_a_quiet_no_op(server, store):
    def run_job(job):  # pragma: no cover - must never be called
        raise AssertionError("nothing was queued")

    worker = make_worker(server, store, run_job)
    assert worker.run_forever() == 0


def test_the_worker_only_claims_work_it_advertised(server, store):
    store.submit([{"title": "needs stems", "wants": ["stems"]}])

    def run_job(job):  # pragma: no cover
        raise AssertionError("should not have been claimed")

    plain = make_worker(server, store, run_job, capabilities=["analyze"])
    assert plain.run_forever() == 0
    assert store.list_jobs()[0].status == QUEUED

    beefy = make_worker(server, store, lambda job: {"songId": "a--b"},
                        capabilities=["analyze", "stems"])
    assert beefy.run_forever() == 1
    assert store.list_jobs()[0].status == DONE


# --- failure handling -------------------------------------------------------


def test_a_transient_failure_is_reported_and_requeued(server, store):
    store.submit([{"title": "T"}])

    def run_job(job):
        raise RuntimeError("connection reset by peer")

    worker = make_worker(server, store, run_job)
    worker.run_forever()

    job = store.list_jobs()[0]
    assert job.status == QUEUED, "a network blip should be tried again"
    assert "connection reset" in job.error


def test_a_permanent_failure_is_not_retried(server, store):
    store.submit([{"title": "T"}])

    def run_job(job):
        raise RuntimeError("Video unavailable")

    worker = make_worker(server, store, run_job)
    worker.run_forever()

    job = store.list_jobs()[0]
    assert job.status == ERROR
    assert job.attempts == 1, "a dead video should not burn the whole retry budget"


def test_one_broken_song_does_not_stop_the_worker(server, store):
    store.submit([{"title": "bad"}, {"title": "good"}])
    ran = []

    def run_job(job):
        ran.append(job["title"])
        if job["title"] == "bad":
            raise RuntimeError("Video unavailable")
        return {"songId": "a--good"}

    worker = make_worker(server, store, run_job, once=False)
    # Drain exactly the two jobs, then stop the loop.
    original = worker.claim

    def claim_then_stop():
        job = original()
        if job is None:
            worker.stop()
        return job

    worker.claim = claim_then_stop
    worker.run_forever()

    assert ran == ["bad", "good"]
    statuses = {j.title: j.status for j in store.list_jobs()}
    assert statuses == {"bad": ERROR, "good": DONE}


# --- the laptop-lid case ----------------------------------------------------


def test_work_finished_after_the_lease_expired_is_dropped_not_forced(server, store):
    """The lid closed, the lease expired, another worker took the job. When
    this worker wakes and finishes, it must NOT overwrite the new holder's
    result — it drops its own."""
    store.submit([{"title": "T"}])

    def run_job(job):
        # While "running", the lease expires and somebody else claims it.
        held = store.get(job["id"])
        held.lease_expires_at = (_now() - timedelta(seconds=1)).isoformat()
        store.list_jobs()                 # reclaim
        store.claim("other-mac")          # taken over
        return {"songId": "a--stale"}

    worker = make_worker(server, store, run_job)
    worker.run_forever()

    job = store.list_jobs()[0]
    assert job.worker == "other-mac"
    assert job.status != DONE, "the stale worker must not have completed it"
    assert job.song_id is None


def test_a_completed_song_survives_a_lost_lease_report(server, store):
    """If the lease is gone at report time the song is still saved by the
    pipeline — losing the bookkeeping must not look like losing the work."""
    store.submit([{"title": "T"}])
    worker = make_worker(server, store, lambda job: {"songId": "a--t"})

    job = worker.claim()
    store.get(job["id"]).lease_expires_at = (_now() - timedelta(seconds=1)).isoformat()
    store.list_jobs()

    # Reporting against a lost lease returns 409 and must not raise.
    worker._report_success(job["id"], {"songId": "a--t"})


def test_the_worker_survives_a_server_that_is_down():
    """A closed laptop and a dead server look the same from here: keep trying,
    never crash."""
    config = WorkerConfig(base_url="http://127.0.0.1:1", once=True,
                          name="offline-mac", capabilities=["analyze"])
    worker = Worker(config, run_job=lambda job: {"songId": "x"})
    worker.client = httpx.Client(base_url="http://127.0.0.1:1", timeout=0.05)
    assert worker.run_forever() == 0  # returns, does not raise
    worker.close()


def test_stop_is_honoured_between_jobs(server, store):
    store.submit([{"title": "one"}, {"title": "two"}])
    ran = []

    def run_job(job):
        ran.append(job["title"])
        worker.stop()
        return {"songId": "a--" + job["title"]}

    worker = make_worker(server, store, run_job, once=False)
    worker.run_forever()

    assert ran == ["one"], "stop() must finish the job in flight, then exit"
    assert store.list_jobs()[1].status == QUEUED
