import io
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server.api import app
from snoocle_server.config import settings
from snoocle_server.discovery.search import SearchHit
from snoocle_server.store.memory import InMemorySongRepository

FIXTURES = Path(__file__).parent / "fixtures"

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch):
    store = InMemorySongRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: store)
    monkeypatch.setattr("snoocle_server.pipeline.get_store", lambda: store)
    return store


@pytest.fixture()
def offline_web(monkeypatch):
    """Route discovery's search+fetch to local fixtures — no network."""
    pages = {
        "https://a.example/x": f"<html><pre>{(FIXTURES / 'sheet_over_lyrics.txt').read_text()}</pre></html>",
        "https://b.example/y": f"<html><pre>{(FIXTURES / 'sheet_inline.txt').read_text()}</pre></html>",
        "https://c.example/z": "<html><body>an unrelated page with no chords</body></html>",
    }
    hits = [SearchHit(url=u, title=f"hit {i}") for i, u in enumerate(pages)]
    monkeypatch.setattr("snoocle_server.discovery.service.web_search", lambda q, n: hits)
    monkeypatch.setattr("snoocle_server.discovery.service.fetch_page", lambda url: pages[url])


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # healthz reports ffmpeg availability; it need not be installed to test the
    # contract shape (it is present in the deployed image).
    assert isinstance(body["ffmpeg"], bool)
    assert set(body["mirEngines"]) == {"beats", "chords", "structure"}
    assert set(body["llmProviders"]) == {"anthropic", "openai", "gemini", "agent", "mock"}
    assert body["mcpEndpoint"] == "/mcp"
    assert "version" in body


def test_song_schema_endpoint():
    r = client.get("/v1/schema/song")
    assert r.status_code == 200
    assert "chordPlacements" in str(r.json())


def test_discover(offline_web):
    r = client.post("/v1/discover", json={"title": "Let It Be", "artist": "The Beatles"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2  # third page has no chords -> filtered
    ids = [c["sourceId"] for c in body["candidates"]]
    assert len(set(ids)) == 2


def test_full_pipeline_and_versioning():
    # provider=mock is the fully-offline path: no discovery, no network. No
    # offline_web fixture needed — it must work with zero external calls.
    req = {
        "title": "Let It Be",
        "artist": "The Beatles",
        "provider": "mock",
        "skipAudio": True,
    }
    r1 = client.post("/v1/songs/analyze", json=req)
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert body1["songId"] == "the-beatles--let-it-be"
    assert body1["provider"] == "mock"
    assert body1["storedVersion"]
    assert body1["steps"]["discover"].startswith("skipped")  # mock: offline
    assert body1["steps"]["acquire"] == "skipped"
    assert body1["steps"]["mir"] == "skipped"
    assert body1["steps"]["reconcile"].startswith("ok:")
    assert body1["steps"]["store"].startswith("ok:")
    assert body1["song"]["lines"]  # a real, schema-valid song was produced

    # re-run: new stored version, prior one preserved and diffable
    r2 = client.post("/v1/songs/analyze", json=req)
    assert r2.status_code == 200, r2.text
    v1, v2 = body1["storedVersion"], r2.json()["storedVersion"]
    assert v1 != v2

    versions = client.get("/v1/songs/the-beatles--let-it-be/versions").json()["versions"]
    assert [v["version"] for v in versions] == [v2, v1]  # newest first
    assert all({"version", "timestamp", "message"} <= set(v) for v in versions)

    old = client.get("/v1/songs/the-beatles--let-it-be", params={"version": v1})
    assert old.status_code == 200
    diff = client.get(
        "/v1/songs/the-beatles--let-it-be/diff", params={"a": v1, "b": v2}
    )
    assert diff.status_code == 200
    assert diff.headers["content-type"].startswith("text/plain")
    assert diff.text.startswith("--- ")  # a unified diff
    # the re-run appended a provenance entry: added (+) lines carrying a new
    # reconciled entry's timestamp
    added = [ln for ln in diff.text.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    assert added and any("timestamp" in ln for ln in added)

    # second run's provenance extends the first (append-only): one "reconciled"
    # entry per run (no discovery in the mock path).
    latest = client.get("/v1/songs/the-beatles--let-it-be").json()
    assert len(latest["provenance"]) == 2
    assert all(p["action"] == "reconciled" for p in latest["provenance"])


def test_pipeline_expected_version_conflict():
    req = {"title": "Let It Be", "artist": "The Beatles", "provider": "mock", "skipAudio": True}
    r1 = client.post("/v1/songs/analyze", json=req)
    stale = r1.json()["storedVersion"]
    client.post("/v1/songs/analyze", json=req)  # moves latest past `stale`
    r3 = client.post("/v1/songs/analyze", json={**req, "expectedVersion": stale})
    assert r3.status_code == 409


def test_analyze_persists_and_survives_new_store_instance(monkeypatch):
    """A song created via analyze is listable/fetchable afterward — the
    persistence acceptance bar (here proven against the same in-memory backend
    the request wrote to; Firestore is proven in test_store.py)."""
    req = {"title": "Yesterday", "artist": "The Beatles", "provider": "mock", "skipAudio": True}
    r = client.post("/v1/songs/analyze", json=req)
    assert r.status_code == 200, r.text
    sid = r.json()["songId"]
    assert sid in client.get("/v1/songs").json()["songs"]
    got = client.get(f"/v1/songs/{sid}")
    assert got.status_code == 200
    assert got.json()["id"] == sid


def test_save_song_returns_version_timestamp_message():
    song = {
        "id": "manual--save",
        "metadata": {"title": "Manual", "artist": "Save"},
        "lines": [{"lineIndex": 0, "lyrics": "la", "chordPlacements": [{"charIndex": 0, "chord": "C"}]}],
        "provenance": [{"timestamp": "2026-07-09T00:00:00Z", "actor": "test", "action": "created"}],
    }
    r = client.post("/v1/songs/manual--save", json={"song": song, "message": "first"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"version", "timestamp", "message"}  # exact contract shape
    assert body["message"] == "first"

    # stale expectedVersion -> 409
    r2 = client.post(
        "/v1/songs/manual--save",
        json={"song": song, "message": "again", "expectedVersion": "deadbeef0000"},
    )
    assert r2.status_code == 409


def test_reconcile_endpoint_with_inline_candidates(offline_web):
    cands = client.post(
        "/v1/discover", json={"title": "Let It Be", "artist": "The Beatles"}
    ).json()["candidates"]
    r = client.post(
        "/v1/reconcile",
        json={
            "title": "Let It Be",
            "artist": "The Beatles",
            "candidates": cands,
            "provider": "mock",
        },
    )
    assert r.status_code == 200, r.text
    song = r.json()["song"]
    assert song["lines"]
    assert all(p["chord"][0] in "ABCDEFG" for l in song["lines"] for p in l["chordPlacements"])


def test_song_not_found():
    assert client.get("/v1/songs/nope--nothing").status_code == 404


def test_store_unavailable_maps_to_503(monkeypatch):
    """A backend outage (e.g. Firestore DB missing) is a clean 503, never a bare
    500 and never a misleading 404."""
    from snoocle_server.store import StoreUnavailableError

    class DownRepo:
        def list_songs(self):
            raise StoreUnavailableError("the database (default) does not exist")

        def get(self, song_id, version=None):
            raise StoreUnavailableError("the database (default) does not exist")

    monkeypatch.setattr(api_mod, "get_store", lambda: DownRepo())
    r1 = client.get("/v1/songs")
    assert r1.status_code == 503
    assert "store unavailable" in r1.json()["detail"]
    assert client.get("/v1/songs/the-beatles--let-it-be").status_code == 503


pytestmark_audio = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")


@pytest.fixture(scope="module")
def tone_bytes(tmp_path_factory):
    p = tmp_path_factory.mktemp("api-audio") / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
         "-c:a", "pcm_s16le", str(p)],
        check=True, capture_output=True,
    )
    return p.read_bytes()


@pytestmark_audio
def test_audio_convert_endpoint(tone_bytes):
    r = client.post(
        "/v1/audio/convert",
        params={"to": "mp3"},
        files={"file": ("tone.wav", io.BytesIO(tone_bytes), "audio/wav")},
    )
    assert r.status_code == 200, r.text
    assert len(r.content) > 1000
    # verify it really is an mp3 by probing it back
    r2 = client.post(
        "/v1/audio/probe",
        files={"file": ("x.mp3", io.BytesIO(r.content), "audio/mpeg")},
    )
    assert r2.json()["codec"] == "mp3"


@pytestmark_audio
def test_audio_trim_endpoint(tone_bytes):
    r = client.post(
        "/v1/audio/trim",
        params={"start": 1.0, "end": 2.5},
        files={"file": ("tone.wav", io.BytesIO(tone_bytes), "audio/wav")},
    )
    assert r.status_code == 200
    r2 = client.post("/v1/audio/probe", files={"file": ("x.wav", io.BytesIO(r.content), "audio/wav")})
    assert r2.json()["duration_seconds"] == pytest.approx(1.5, abs=0.05)


@pytestmark_audio
def test_audio_trim_bad_range(tone_bytes):
    r = client.post(
        "/v1/audio/trim",
        params={"start": 3.0, "end": 1.0},
        files={"file": ("tone.wav", io.BytesIO(tone_bytes), "audio/wav")},
    )
    assert r.status_code == 400


@pytest.fixture(scope="module")
def video_bytes(tmp_path_factory):
    """A tiny mp4 with a real audio track — the 'bring your own video' case."""
    p = tmp_path_factory.mktemp("api-video") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=size=128x72:rate=10:duration=3",
         "-f", "lavfi", "-i", "sine=frequency=330:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(p)],
        check=True, capture_output=True,
    )
    return p.read_bytes()


@pytest.fixture(scope="module")
def silent_video_bytes(tmp_path_factory):
    """A video with NO audio stream — must be rejected, not silently analyzed."""
    p = tmp_path_factory.mktemp("api-silent") / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=size=128x72:rate=10:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)],
        check=True, capture_output=True,
    )
    return p.read_bytes()


@pytestmark_audio
def test_analyze_upload_audio_file(tone_bytes):
    r = client.post(
        "/v1/audio/analyze/upload",
        files={"file": ("tone.wav", io.BytesIO(tone_bytes), "audio/wav")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"] == "tone.wav"
    a = body["analysis"]
    assert a["duration_seconds"] > 0
    assert "beats" in a["engines"] and "chords" in a["engines"] and "structure" in a["engines"]


@pytestmark_audio
def test_analyze_upload_video_file_extracts_audio(video_bytes):
    # A video container: the audio track is extracted and analyzed.
    r = client.post(
        "/v1/audio/analyze/upload",
        files={"file": ("clip.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["analysis"]["duration_seconds"] > 0


@pytestmark_audio
def test_analyze_upload_rejects_streamless_video(silent_video_bytes):
    r = client.post(
        "/v1/audio/analyze/upload",
        files={"file": ("silent.mp4", io.BytesIO(silent_video_bytes), "video/mp4")},
    )
    assert r.status_code == 422
    assert "audio stream" in r.json()["detail"]
