"""Batch analysis queue (master plan D2).

The behaviours that matter operationally: jobs run in submission order, ONE at
a time, and a failure marks that job and moves on rather than stalling the
queue behind it.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

import snoocle_server.api as api_mod
from snoocle_server.api import app
from snoocle_server.batch import (
    DONE,
    ERROR,
    MAX_ITEMS_PER_SUBMIT,
    QUEUED,
    BatchQueue,
    parse_batch_line,
    reset_queue,
)
from snoocle_server.store.memory import InMemorySongRepository


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def fresh_queue():
    reset_queue()
    yield
    reset_queue()


# --- parsing the "add many" textarea ---------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        ("", None),
        ("   ", None),
        ("# a comment", None),
        ("https://youtu.be/dQw4w9WgXcQ", {"youtubeUrlOrId": "https://youtu.be/dQw4w9WgXcQ"}),
        ("https://www.youtube.com/watch?v=abc", {"youtubeUrlOrId": "https://www.youtube.com/watch?v=abc"}),
        ("dQw4w9WgXcQ", {"youtubeUrlOrId": "dQw4w9WgXcQ"}),
        ("The Beatles - Let It Be", {"title": "Let It Be", "artist": "The Beatles"}),
        ("Radiohead — Creep", {"title": "Creep", "artist": "Radiohead"}),
        ("Radiohead – Creep", {"title": "Creep", "artist": "Radiohead"}),
        ("Just A Title", {"title": "Just A Title"}),
    ],
)
def test_parse_batch_line(line, expected):
    assert parse_batch_line(line) == expected


def test_hyphenated_title_still_splits_on_the_first_separator():
    assert parse_batch_line("Crosby - Stills - Nash") == {
        "artist": "Crosby",
        "title": "Stills - Nash",
    }


# --- the worker -------------------------------------------------------------


async def _drain(queue: BatchQueue, expected: int, timeout: float = 5.0) -> None:
    """Wait until `expected` jobs have reached a terminal state."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        done = [j for j in queue.list_jobs() if j.status in (DONE, ERROR)]
        if len(done) >= expected:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"queue did not drain: {[(j.id, j.status) for j in queue.list_jobs()]}"
    )


@pytest.mark.anyio
async def test_jobs_process_sequentially_in_submission_order(monkeypatch, anyio_backend):
    order: list[str] = []
    concurrent = 0
    max_concurrent = 0

    async def fake_pipeline(title, artist, **kwargs):
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.01)
        order.append(title)
        concurrent -= 1

        class Report:
            song_id = f"artist--{title}"
            run_id = "run-" + title
            stored_version = "abc123"

        return Report()

    monkeypatch.setattr("snoocle_server.pipeline.run_pipeline_async", fake_pipeline)

    queue = BatchQueue()
    queue.start()
    queue.submit([{"title": "one", "artist": "A"},
                  {"title": "two", "artist": "A"},
                  {"title": "three", "artist": "A"}])
    await _drain(queue, 3)
    await queue.stop()

    assert order == ["one", "two", "three"]
    assert max_concurrent == 1, "the worker must be strictly sequential"
    assert [j.status for j in queue.list_jobs()] == [DONE, DONE, DONE]
    assert queue.list_jobs()[0].song_id == "artist--one"


@pytest.mark.anyio
async def test_a_failure_does_not_stall_the_queue(monkeypatch, anyio_backend):
    async def fake_pipeline(title, artist, **kwargs):
        if title == "bad":
            raise RuntimeError("reconcile blew up")

        class Report:
            song_id = "a--" + title
            run_id = "r"
            stored_version = "v"

        return Report()

    monkeypatch.setattr("snoocle_server.pipeline.run_pipeline_async", fake_pipeline)

    queue = BatchQueue()
    queue.start()
    queue.submit([{"title": "good1", "artist": "A"},
                  {"title": "bad", "artist": "A"},
                  {"title": "good2", "artist": "A"}])
    await _drain(queue, 3)
    await queue.stop()

    jobs = queue.list_jobs()
    assert [j.status for j in jobs] == [DONE, ERROR, DONE]
    assert "reconcile blew up" in jobs[1].error
    # The job AFTER the failure still ran — that is the whole point.
    assert jobs[2].song_id == "a--good2"


@pytest.mark.anyio
async def test_retry_reruns_a_failed_job(monkeypatch, anyio_backend):
    attempts = {"n": 0}

    async def flaky(title, artist, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient")

        class Report:
            song_id = "a--b"
            run_id = "r"
            stored_version = "v"

        return Report()

    monkeypatch.setattr("snoocle_server.pipeline.run_pipeline_async", flaky)

    queue = BatchQueue()
    queue.start()
    queue.submit([{"title": "t", "artist": "A"}])
    await _drain(queue, 1)
    assert queue.list_jobs()[0].status == ERROR

    queue.retry(queue.list_jobs()[0].id)
    await _drain(queue, 1)
    await queue.stop()

    job = queue.list_jobs()[0]
    assert job.status == DONE
    assert job.attempts == 2
    assert job.error is None


def test_retry_refuses_a_job_that_has_not_finished():
    queue = BatchQueue()
    job = queue.submit([{"title": "t", "artist": "A"}])[0]
    with pytest.raises(ValueError):
        queue.retry(job.id)


def test_cancel_only_applies_to_queued_jobs():
    queue = BatchQueue()
    job = queue.submit([{"title": "t", "artist": "A"}])[0]
    assert queue.cancel(job.id).status == "cancelled"
    with pytest.raises(ValueError):
        queue.cancel(job.id)  # already terminal


def test_submitting_more_than_the_cap_is_rejected():
    queue = BatchQueue()
    with pytest.raises(ValueError, match="too many items"):
        queue.submit([{"title": f"t{i}"} for i in range(MAX_ITEMS_PER_SUBMIT + 1)])


def test_jobs_submitted_before_the_worker_starts_are_picked_up_on_start():
    """The lifespan starts the worker; anything queued by an earlier
    generation (or before startup) must be re-armed, not orphaned."""
    queue = BatchQueue()
    queue.submit([{"title": "t", "artist": "A"}])
    assert queue.list_jobs()[0].status == QUEUED

    async def check():
        seen = []

        async def fake(title, artist, **kwargs):
            seen.append(title)

            class R:
                song_id = "a--b"
                run_id = "r"
                stored_version = "v"

            return R()

        import snoocle_server.pipeline as pipe

        original = pipe.run_pipeline_async
        pipe.run_pipeline_async = fake
        try:
            queue.start()
            await _drain(queue, 1)
        finally:
            pipe.run_pipeline_async = original
            await queue.stop()
        return seen

    assert asyncio.run(check()) == ["t"]


# --- the endpoints ----------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    repo = InMemorySongRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: repo)
    monkeypatch.setattr("snoocle_server.pipeline.get_store", lambda: repo)
    return TestClient(app)


def test_queue_endpoint_accepts_a_textarea_paste(client):
    r = client.post("/v1/queue", json={
        "text": "The Beatles - Let It Be\n\n# skip me\nhttps://youtu.be/dQw4w9WgXcQ\n",
        "analysisDepth": "fast",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["queued"] == 2
    assert body["jobs"][0]["title"] == "Let It Be"
    assert body["jobs"][0]["analysisDepth"] == "fast"
    assert body["jobs"][1]["youtubeUrlOrId"] == "https://youtu.be/dQw4w9WgXcQ"


def test_queue_endpoint_accepts_structured_items(client):
    r = client.post("/v1/queue", json={"items": [{"title": "T", "artist": "A"}]})
    assert r.status_code == 200
    assert r.json()["jobs"][0]["label"] == "A — T"


def test_empty_submission_is_a_400(client):
    assert client.post("/v1/queue", json={"text": "\n\n# nothing\n"}).status_code == 400
    assert client.post("/v1/queue", json={}).status_code == 400


def test_queue_status_reports_worker_liveness_and_counts(client):
    client.post("/v1/queue", json={"items": [{"title": "T", "artist": "A"}]})
    body = client.get("/v1/queue").json()
    assert "workerAlive" in body
    assert body["maxPerSubmit"] == MAX_ITEMS_PER_SUBMIT
    assert sum(body["counts"].values()) == len(body["jobs"])


def test_retry_and_cancel_return_404_for_unknown_jobs(client):
    assert client.post("/v1/queue/nope/retry").status_code == 404
    assert client.post("/v1/queue/nope/cancel").status_code == 404


def test_clearing_finished_jobs_leaves_pending_ones(client):
    client.post("/v1/queue", json={"items": [{"title": "T", "artist": "A"}]})
    before = len(client.get("/v1/queue").json()["jobs"])
    removed = client.request("DELETE", "/v1/queue").json()["removed"]
    after = len(client.get("/v1/queue").json()["jobs"])
    assert after == before - removed
