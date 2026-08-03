from __future__ import annotations

import io
import math
import struct
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server.api import app
from snoocle_server.audio import artifacts
from snoocle_server.audio.artifacts import (
    AudioArtifactNotFound,
    GCSAudioArtifactStore,
    LocalAudioArtifactStore,
)
from snoocle_server.config import settings
from snoocle_server.mir.base import MirAnalysis


TOKEN = "artifact-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
client = TestClient(app)


def wav_bytes(duration: float = 1.0, sample_rate: int = 8_000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        frames = [
            struct.pack("<h", int(8_000 * math.sin(2 * math.pi * 440 * index / sample_rate)))
            for index in range(int(duration * sample_rate))
        ]
        stream.writeframes(b"".join(frames))
    return output.getvalue()


@pytest.fixture(autouse=True)
def isolated_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "audio_artifact_backend", "local")
    monkeypatch.setattr(settings, "audio_artifact_dir", tmp_path / "artifacts")
    monkeypatch.setattr(settings, "audio_artifact_ttl_seconds", 3_600)
    monkeypatch.setattr(settings, "audio_artifact_max_bytes", 2_000_000)
    monkeypatch.setattr(settings, "audio_artifact_max_duration_seconds", 60.0)
    monkeypatch.setattr(settings, "audio_artifact_quota_bytes", 4_000_000)
    monkeypatch.setattr(settings, "audio_artifact_quota_count", 10)
    monkeypatch.setattr(settings, "api_token", "")
    artifacts.reset_audio_artifact_store()
    yield
    artifacts.reset_audio_artifact_store()


def upload(*, headers: dict | None = None, name: str = "tone.wav", mime: str = "audio/wav"):
    return client.post(
        "/v1/audio/artifacts",
        headers=headers,
        files={"file": (name, io.BytesIO(wav_bytes()), mime)},
    )


def test_upload_is_authenticated_and_never_discloses_a_server_path(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "api_token", TOKEN)
    assert upload().status_code == 401

    response = upload(headers=AUTH)
    assert response.status_code == 201, response.text
    artifact = response.json()["artifact"]
    assert artifacts._REF_RE.fullmatch(artifact["audioRef"])
    assert artifact["playbackUrl"].endswith(f"/{artifact['audioRef']}/content")
    rendered = str(response.json())
    assert str(tmp_path) not in rendered
    assert "audioPath" not in rendered and "path" not in artifact

    assert client.get(artifact["playbackUrl"]).status_code == 401
    assert client.get(artifact["playbackUrl"], headers=AUTH).status_code == 200


def test_real_type_size_and_duration_limits_are_enforced(monkeypatch):
    wrong_type = upload(name="pretend.mp3", mime="audio/mpeg")
    assert wrong_type.status_code == 422
    assert "contents" in wrong_type.json()["detail"]

    monkeypatch.setattr(settings, "audio_artifact_max_bytes", 100)
    too_large = upload()
    assert too_large.status_code == 413

    monkeypatch.setattr(settings, "audio_artifact_max_bytes", 2_000_000)
    monkeypatch.setattr(settings, "audio_artifact_max_duration_seconds", 0.1)
    too_long = upload()
    assert too_long.status_code == 422
    assert "duration" in too_long.json()["detail"]


def test_range_playback_delete_and_untrusted_refs_are_bounded():
    created = upload().json()["artifact"]
    url = created["playbackUrl"]

    complete = client.get(url)
    assert complete.status_code == 200
    assert complete.headers["accept-ranges"] == "bytes"
    assert complete.headers["cache-control"] == "private, no-store"

    partial = client.get(url, headers={"Range": "bytes=4-19"})
    assert partial.status_code == 206
    assert partial.content == complete.content[4:20]
    assert partial.headers["content-range"] == f"bytes 4-19/{len(complete.content)}"

    suffix = client.get(url, headers={"Range": "bytes=-8"})
    assert suffix.status_code == 206
    assert suffix.content == complete.content[-8:]

    invalid = client.get(url, headers={"Range": "bytes=999999-"})
    assert invalid.status_code == 416
    assert invalid.headers["content-range"] == f"bytes */{len(complete.content)}"

    assert client.get("/v1/audio/artifacts/not-a-ref/content").status_code == 404
    assert client.get("/v1/audio/artifacts/%2e%2e%2fetc%2fpasswd/content").status_code == 404

    deleted = client.delete(f"/v1/audio/artifacts/{created['audioRef']}")
    assert deleted.status_code == 200
    assert client.get(url).status_code == 404


def test_ttl_cleanup_removes_expired_objects(monkeypatch):
    clock = datetime(2026, 8, 3, tzinfo=timezone.utc)
    monkeypatch.setattr(artifacts, "_utcnow", lambda: clock)
    created = upload().json()["artifact"]
    assert client.get(created["playbackUrl"]).status_code == 200

    clock += timedelta(hours=2)
    cleaned = client.post("/v1/audio/artifacts/cleanup")
    assert cleaned.json() == {"removed": 1}
    assert client.get(created["playbackUrl"]).status_code == 404


def test_rest_analysis_resolves_ref_without_returning_a_path(monkeypatch):
    created = upload().json()["artifact"]
    seen = []
    monkeypatch.setattr(
        api_mod,
        "analyze_audio",
        lambda path, accuracy="standard": seen.append(Path(path))
        or MirAnalysis(duration_seconds=1.0),
    )
    response = client.post("/v1/audio/analyze", json={"audioRef": created["audioRef"]})
    assert response.status_code == 200, response.text
    assert response.json()["audioRef"] == created["audioRef"]
    assert "audioPath" not in response.json()
    assert seen[0].suffix == ".wav"
    assert client.post(
        "/v1/audio/analyze",
        json={"audioRef": created["audioRef"], "audioPath": "/etc/passwd"},
    ).status_code == 422


def test_concurrent_requests_cannot_race_past_object_quota(monkeypatch):
    monkeypatch.setattr(settings, "audio_artifact_quota_count", 1)

    def request() -> int:
        return upload().status_code

    with ThreadPoolExecutor(max_workers=4) as pool:
        statuses = list(pool.map(lambda _: request(), range(4)))
    assert statuses.count(201) == 1
    assert statuses.count(429) == 3


def test_total_byte_quota_is_checked_before_storage(monkeypatch):
    size = len(wav_bytes())
    monkeypatch.setattr(settings, "audio_artifact_quota_bytes", size + 10)
    assert upload().status_code == 201
    refused = upload()
    assert refused.status_code == 429
    assert "total bytes" in refused.json()["detail"]


def test_youtube_acquisition_returns_a_reference_not_its_cache_path(monkeypatch, tmp_path):
    source = tmp_path / "download.wav"
    source.write_bytes(wav_bytes())
    monkeypatch.setattr(
        api_mod,
        "acquire",
        lambda **kwargs: SimpleNamespace(
            path=str(source),
            video_id="dQw4w9WgXcQ",
            video_title="Fixture",
            duration_seconds=1.0,
            from_cache=False,
        ),
    )
    response = client.post(
        "/v1/audio/artifacts/acquire", json={"youtubeUrlOrId": "dQw4w9WgXcQ"}
    )
    assert response.status_code == 201, response.text
    assert response.json()["artifact"]["youtubeVideoId"] == "dQw4w9WgXcQ"
    assert str(source) not in response.text


class FakeNotFound(Exception):
    code = 404


class FakeBlob:
    def __init__(self, values: dict[str, tuple[bytes, str]], name: str):
        self.values = values
        self.name = name

    def upload_from_filename(self, filename, content_type, if_generation_match):
        if if_generation_match == 0 and self.name in self.values:
            raise RuntimeError("precondition")
        self.values[self.name] = (Path(filename).read_bytes(), content_type)

    def upload_from_string(self, value, content_type, if_generation_match):
        if if_generation_match == 0 and self.name in self.values:
            raise RuntimeError("precondition")
        self.values[self.name] = (value.encode(), content_type)

    def download_as_bytes(self, start=None, end=None):
        if self.name not in self.values:
            raise FakeNotFound()
        value = self.values[self.name][0]
        return value[slice(start, None if end is None else end + 1)]

    def download_to_filename(self, filename):
        Path(filename).write_bytes(self.download_as_bytes())

    def delete(self):
        if self.name not in self.values:
            raise FakeNotFound()
        del self.values[self.name]


class FakeBucket:
    def __init__(self, values):
        self.values = values

    def blob(self, name):
        return FakeBlob(self.values, name)

    def list_blobs(self, prefix):
        return [FakeBlob(self.values, name) for name in sorted(self.values) if name.startswith(prefix)]


class FakeStorageClient:
    def __init__(self):
        self.values: dict[str, tuple[bytes, str]] = {}

    def bucket(self, name):
        return FakeBucket(self.values)


def test_gcs_backend_is_shared_across_instances_without_public_urls(tmp_path):
    source = tmp_path / "shared.wav"
    source.write_bytes(wav_bytes())
    cloud = FakeStorageClient()
    first = GCSAudioArtifactStore("private-bucket", "temporary/audio", client=cloud)
    second = GCSAudioArtifactStore("private-bucket", "temporary/audio", client=cloud)

    created = first.create(source, filename="shared.wav")
    assert second.get(created.audio_ref) == created
    assert b"".join(second.iter_content(created.audio_ref, 2, 12)) == source.read_bytes()[2:13]
    with second.materialize(created.audio_ref) as local:
        assert local.read_bytes() == source.read_bytes()
    assert "private-bucket" not in str(created.to_public())
    assert second.delete(created.audio_ref) is True
    with pytest.raises(AudioArtifactNotFound):
        first.get(created.audio_ref)
