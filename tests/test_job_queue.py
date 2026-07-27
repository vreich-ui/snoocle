"""The broker: durable jobs, leases, and the claim/heartbeat/report protocol.

The interesting behaviour is all failure behaviour. A worker is a laptop — it
sleeps, it loses wifi, it gets closed mid-song. What these tests pin is that
none of those become a stuck queue: a lease that stops being renewed expires,
the job goes back in the pool, and the worker that wakes up holding a stale
lease is told to drop its work rather than race the machine that took over.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

import snoocle_server.api as api_mod
from snoocle_server.api import app
from snoocle_server.batch import MAX_ITEMS_PER_SUBMIT, parse_batch_line
from snoocle_server.store import jobs as jobs_mod
from snoocle_server.store.jobs import (
    CANCELLED,
    DONE,
    ERROR,
    LEASED,
    MAX_ATTEMPTS,
    QUEUED,
    InMemoryJobRepository,
    Job,
    UnknownJobError,
    WrongWorkerError,
    _now,
)


@pytest.fixture()
def store():
    return InMemoryJobRepository()


@pytest.fixture()
def client(monkeypatch, store):
    monkeypatch.setattr(api_mod, "get_job_store", lambda: store)
    return TestClient(app)


def _expire(job: Job, seconds: int = 1) -> None:
    """Backdate a lease so it is already expired, without sleeping."""
    job.lease_expires_at = (_now() - timedelta(seconds=seconds)).isoformat()


# --- parsing the textarea ---------------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        ("", None),
        ("   ", None),
        ("# a comment", None),
        ("https://youtu.be/dQw4w9WgXcQ", {"youtubeUrlOrId": "https://youtu.be/dQw4w9WgXcQ"}),
        ("dQw4w9WgXcQ", {"youtubeUrlOrId": "dQw4w9WgXcQ"}),
        ("The Beatles - Let It Be", {"title": "Let It Be", "artist": "The Beatles"}),
        ("Radiohead — Creep", {"title": "Creep", "artist": "Radiohead"}),
        ("Just A Title", {"title": "Just A Title"}),
    ],
)
def test_parse_batch_line(line, expected):
    assert parse_batch_line(line) == expected


def test_hyphenated_title_splits_on_the_first_separator_only():
    assert parse_batch_line("Crosby - Stills - Nash") == {
        "artist": "Crosby", "title": "Stills - Nash",
    }


# --- claiming ---------------------------------------------------------------


def test_claim_hands_out_jobs_in_submission_order(store):
    store.submit([{"title": "one"}, {"title": "two"}])
    assert store.claim("mac").title == "one"
    assert store.claim("mac").title == "two"
    assert store.claim("mac") is None


def test_a_claimed_job_is_not_handed_to_a_second_worker(store):
    store.submit([{"title": "only"}])
    assert store.claim("mac-a") is not None
    assert store.claim("mac-b") is None


def test_claim_records_the_lease_and_counts_the_attempt(store):
    store.submit([{"title": "t"}])
    job = store.claim("mac", lease_seconds=120)
    assert job.status == LEASED
    assert job.worker == "mac"
    assert job.attempts == 1
    assert job.lease_expires_at and job.heartbeat_at
    assert not job.lease_expired()


def test_a_worker_only_gets_jobs_it_can_do(store):
    store.submit([{"title": "needs stems", "wants": ["stems"]}])
    assert store.claim("plain-worker", capabilities=["analyze"]) is None
    got = store.claim("beefy", capabilities=["analyze", "stems"])
    assert got is not None and got.title == "needs stems"


def test_capable_worker_skips_past_an_unsatisfiable_job_to_one_it_can_do(store):
    store.submit([{"title": "stems only", "wants": ["stems"]}, {"title": "plain"}])
    got = store.claim("plain-worker", capabilities=["analyze"])
    assert got is not None and got.title == "plain"


# --- leases expiring (the laptop-lid case) ----------------------------------


def test_an_expired_lease_returns_the_job_to_the_queue(store):
    store.submit([{"title": "t"}])
    job = store.claim("sleeping-mac")
    _expire(job)

    requeued = store.claim("awake-mac")
    assert requeued is not None
    assert requeued.id == job.id
    assert requeued.worker == "awake-mac"
    assert requeued.attempts == 2


def test_listing_reclaims_expired_leases_without_any_scheduler(store):
    store.submit([{"title": "t"}])
    job = store.claim("mac")
    _expire(job)

    listed = store.list_jobs()[0]
    assert listed.status == QUEUED
    assert listed.worker is None
    assert "lease expired" in (listed.error or "")


def test_a_job_that_keeps_losing_its_lease_is_parked_not_retried_forever(store):
    store.submit([{"title": "t"}])
    for _ in range(MAX_ATTEMPTS):
        job = store.claim("flaky-mac")
        assert job is not None
        _expire(job)

    parked = store.list_jobs()[0]
    assert parked.status == ERROR
    assert parked.attempts == MAX_ATTEMPTS
    assert store.claim("mac") is None, "a parked job must not be handed out again"


def test_heartbeating_keeps_a_lease_alive(store):
    store.submit([{"title": "t"}])
    job = store.claim("mac", lease_seconds=60)
    _expire(job)
    store.heartbeat(job.id, "mac", lease_seconds=60)

    assert store.list_jobs()[0].status == LEASED
    assert store.claim("other") is None


# --- acting on a job you no longer hold -------------------------------------


def test_a_stale_worker_cannot_heartbeat_complete_or_fail(store):
    store.submit([{"title": "t"}])
    job = store.claim("old-mac")
    _expire(job)
    store.list_jobs()          # reclaim
    store.claim("new-mac")     # somebody else took it

    for call in (
        lambda: store.heartbeat(job.id, "old-mac"),
        lambda: store.complete(job.id, "old-mac", song_id="a--b"),
        lambda: store.fail(job.id, "old-mac", "boom"),
    ):
        with pytest.raises(WrongWorkerError):
            call()


def test_unknown_jobs_raise_rather_than_silently_succeeding(store):
    with pytest.raises(UnknownJobError):
        store.heartbeat("nope", "mac")


# --- completing and failing --------------------------------------------------


def test_complete_records_the_result(store):
    store.submit([{"title": "t"}])
    job = store.claim("mac")
    done = store.complete(job.id, "mac", song_id="a--b", run_id="r1", stored_version="v1")
    assert (done.status, done.song_id, done.run_id) == (DONE, "a--b", "r1")
    assert done.finished_at and done.lease_expires_at is None


def test_a_retryable_failure_goes_back_in_the_queue(store):
    store.submit([{"title": "t"}])
    job = store.claim("mac")
    failed = store.fail(job.id, "mac", "network blip")
    assert failed.status == QUEUED
    assert failed.error == "network blip"
    assert store.claim("mac") is not None


def test_a_permanent_failure_stops_immediately(store):
    store.submit([{"title": "t"}])
    job = store.claim("mac")
    failed = store.fail(job.id, "mac", "Video unavailable", retry=False)
    assert failed.status == ERROR
    assert store.claim("mac") is None


def test_failures_stop_retrying_at_the_attempt_cap(store):
    store.submit([{"title": "t"}])
    for _ in range(MAX_ATTEMPTS):
        job = store.claim("mac")
        assert job is not None
        store.fail(job.id, "mac", "keeps breaking")
    assert store.list_jobs()[0].status == ERROR
    assert store.claim("mac") is None


def test_one_bad_job_does_not_block_the_ones_behind_it(store):
    store.submit([{"title": "bad"}, {"title": "good"}])
    bad = store.claim("mac")
    store.fail(bad.id, "mac", "Video unavailable", retry=False)
    assert store.claim("mac").title == "good"


# --- admin actions ----------------------------------------------------------


def test_retry_gives_a_parked_job_a_fresh_budget(store):
    store.submit([{"title": "t"}])
    job = store.claim("mac")
    store.fail(job.id, "mac", "permanent", retry=False)

    revived = store.retry(job.id)
    assert revived.status == QUEUED
    assert revived.attempts == 0, "a human asking again means a full budget"
    assert revived.error is None
    assert store.claim("mac") is not None


def test_retry_refuses_a_job_still_in_flight(store):
    store.submit([{"title": "t"}])
    job = store.claim("mac")
    with pytest.raises(ValueError):
        store.retry(job.id)


def test_cancel_applies_only_to_queued_jobs(store):
    store.submit([{"title": "t"}, {"title": "u"}])
    jobs = store.list_jobs()
    assert store.cancel(jobs[0].id).status == CANCELLED
    leased = store.claim("mac")
    with pytest.raises(ValueError):
        store.cancel(leased.id)


def test_clear_finished_keeps_live_work(store):
    store.submit([{"title": "done"}, {"title": "waiting"}])
    job = store.claim("mac")
    store.complete(job.id, "mac", song_id="a--b")
    assert store.clear_finished() == 1
    assert [j.title for j in store.list_jobs()] == ["waiting"]


# --- worker liveness --------------------------------------------------------


def test_stats_report_a_worker_only_on_evidence_of_a_heartbeat(store):
    assert store.stats()["workerSeenRecently"] is False

    store.submit([{"title": "t"}])
    job = store.claim("wolfs-mac")
    stats = store.stats()
    assert stats["workerSeenRecently"] is True
    assert stats["lastWorker"] == "wolfs-mac"
    assert stats["counts"][LEASED] == 1


def test_a_long_silent_worker_stops_counting_as_present(store):
    store.submit([{"title": "t"}])
    job = store.claim("wolfs-mac")
    job.heartbeat_at = (_now() - timedelta(seconds=jobs_mod.DEFAULT_LEASE_SECONDS + 60)).isoformat()
    _expire(job)
    assert store.stats()["workerSeenRecently"] is False


# --- the HTTP surface -------------------------------------------------------


def test_submit_accepts_a_pasted_list(client):
    r = client.post("/v1/queue", json={
        "text": "The Beatles - Let It Be\n\n# skip\nhttps://youtu.be/dQw4w9WgXcQ\n",
        "analysisDepth": "fast",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["queued"] == 2
    assert body["jobs"][0]["title"] == "Let It Be"
    assert body["jobs"][0]["analysisDepth"] == "fast"


def test_submit_rejects_an_empty_or_oversized_list(client):
    assert client.post("/v1/queue", json={"text": "\n# nothing\n"}).status_code == 400
    big = {"items": [{"title": f"t{i}"} for i in range(MAX_ITEMS_PER_SUBMIT + 1)]}
    assert client.post("/v1/queue", json=big).status_code == 400


def test_an_empty_queue_answers_a_claim_with_204_not_an_error(client):
    """A worker polls every 15s forever; 'nothing to do' must be the quiet,
    ordinary path."""
    r = client.post("/v1/queue/claim", json={"worker": "mac"})
    assert r.status_code == 204


def test_the_full_protocol_round_trips_over_http(client):
    client.post("/v1/queue", json={"items": [{"title": "T", "artist": "A"}]})

    claim = client.post("/v1/queue/claim", json={
        "worker": "mac", "capabilities": ["analyze"],
    })
    assert claim.status_code == 200
    job = claim.json()
    assert job["title"] == "T"
    assert "leaseExpiresAt" in job

    beat = client.post(f"/v1/queue/{job['id']}/heartbeat", json={"worker": "mac"})
    assert beat.status_code == 200 and beat.json()["status"] == "leased"

    done = client.post(f"/v1/queue/{job['id']}/complete", json={
        "worker": "mac", "songId": "a--t", "runId": "r1", "storedVersion": "v1",
    })
    assert done.status_code == 200
    assert done.json()["status"] == "done"
    assert done.json()["songId"] == "a--t"


def test_a_lost_lease_answers_409_so_the_worker_stops(client, store):
    client.post("/v1/queue", json={"items": [{"title": "T"}]})
    job = client.post("/v1/queue/claim", json={"worker": "old"}).json()
    _expire(store.get(job["id"]))
    store.list_jobs()
    client.post("/v1/queue/claim", json={"worker": "new"})

    for path, body in (
        ("heartbeat", {"worker": "old"}),
        ("complete", {"worker": "old", "songId": "a--b"}),
        ("fail", {"worker": "old", "error": "x"}),
    ):
        r = client.post(f"/v1/queue/{job['id']}/{path}", json=body)
        assert r.status_code == 409, path


def test_worker_endpoints_404_on_an_unknown_job(client):
    for path, body in (
        ("heartbeat", {"worker": "mac"}),
        ("complete", {"worker": "mac", "songId": "a--b"}),
        ("fail", {"worker": "mac", "error": "x"}),
        ("retry", None),
        ("cancel", None),
    ):
        r = client.post(f"/v1/queue/nope/{path}", json=body)
        assert r.status_code == 404, path


def test_status_endpoint_exposes_worker_liveness(client):
    client.post("/v1/queue", json={"items": [{"title": "T"}]})
    body = client.get("/v1/queue").json()
    assert body["workerSeenRecently"] is False
    assert body["leaseSeconds"] > 0
    assert body["maxPerSubmit"] == MAX_ITEMS_PER_SUBMIT

    client.post("/v1/queue/claim", json={"worker": "wolfs-mac"})
    body = client.get("/v1/queue").json()
    assert body["workerSeenRecently"] is True
    assert body["lastWorker"] == "wolfs-mac"


def test_the_server_starts_no_background_worker():
    """The whole point of the split: nothing in the app's lifespan needs a CPU
    outside a request, so Cloud Run can stay at --min-instances=0."""
    import inspect

    source = inspect.getsource(api_mod._lifespan)
    assert "create_task" not in source
    assert "queue.start" not in source
