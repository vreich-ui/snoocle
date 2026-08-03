from __future__ import annotations

import io
import math
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from snoocle_server import mcp_server
from snoocle_server.audio import artifacts
from snoocle_server.config import settings
from snoocle_server.mir.base import MirAnalysis


def _wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8_000)
        stream.writeframes(
            b"".join(
                struct.pack("<h", int(8_000 * math.sin(2 * math.pi * 440 * i / 8_000)))
                for i in range(8_000)
            )
        )
    return output.getvalue()


@pytest.fixture()
def audio_ref(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "audio_artifact_backend", "local")
    monkeypatch.setattr(settings, "audio_artifact_dir", tmp_path / "artifacts")
    monkeypatch.setattr(settings, "audio_artifact_max_bytes", 2_000_000)
    monkeypatch.setattr(settings, "audio_artifact_max_duration_seconds", 60.0)
    monkeypatch.setattr(settings, "audio_artifact_quota_bytes", 4_000_000)
    monkeypatch.setattr(settings, "audio_artifact_quota_count", 10)
    artifacts.reset_audio_artifact_store()
    source = tmp_path / "input.wav"
    source.write_bytes(_wav())
    created = artifacts.get_audio_artifact_store().create(source, filename=source.name)
    yield created.audio_ref, source
    artifacts.reset_audio_artifact_store()


def test_mir_wrappers_resolve_refs_and_keep_legacy_paths(monkeypatch, audio_ref):
    ref, source = audio_ref
    seen: list[Path] = []
    mir = MirAnalysis(duration_seconds=1.0)
    monkeypatch.setattr(
        mcp_server,
        "_analyze_audio",
        lambda path, accuracy="standard": seen.append(Path(path)) or mir,
    )
    monkeypatch.setattr(
        mcp_server,
        "_analyze_window",
        lambda path, start, end: seen.append(Path(path)) or mir,
    )

    by_ref = mcp_server.analyze_full_track_mir(audio_ref=ref)
    legacy = mcp_server.analyze_full_track_mir(str(source))
    window = mcp_server.analyze_mir_window(None, 0, 1, audio_ref=ref)

    assert by_ref["ok"] is True and legacy["ok"] is True and window["ok"] is True
    assert seen[0].suffix == ".wav" and seen[2].suffix == ".wav"
    assert seen[1] == source
    assert str(source) not in str(by_ref["inputSummary"])


def test_mir_wrapper_ref_errors_are_stable_and_do_not_probe_paths(audio_ref):
    ref, source = audio_ref
    both = mcp_server.analyze_full_track_mir(str(source), audio_ref=ref)
    missing = mcp_server.analyze_full_track_mir(audio_ref="aud_" + "x" * 32)
    malformed = mcp_server.analyze_full_track_mir(audio_ref="../../etc/passwd")

    assert both["error"]["code"] == "invalid_audio_source"
    assert missing["error"]["code"] == "audio_artifact_not_found"
    assert malformed["error"]["code"] == "audio_artifact_not_found"
    assert "/etc/passwd" not in missing["error"]["message"]


def test_pipeline_and_audio_utility_wrappers_return_opaque_artifacts(monkeypatch, audio_ref):
    ref, _ = audio_ref
    mir = MirAnalysis(duration_seconds=1.0)
    monkeypatch.setattr(mcp_server, "_analyze_audio", lambda *args, **kwargs: mir)

    analyzed = mcp_server.analyze_audio(audio_ref=ref)
    probed = mcp_server.probe_audio(input_ref=ref)

    assert analyzed["audioRef"] == ref
    assert "audioPath" not in analyzed and "path" not in analyzed
    assert probed["duration_seconds"] == pytest.approx(1.0)


def test_youtube_mcp_acquisition_does_not_return_cache_path(monkeypatch, audio_ref):
    _, source = audio_ref
    monkeypatch.setattr(
        mcp_server,
        "_acquire",
        lambda **kwargs: SimpleNamespace(
            path=str(source),
            video_id="dQw4w9WgXcQ",
            video_title="Fixture",
            duration_seconds=1.0,
            from_cache=True,
        ),
    )
    result = mcp_server.acquire_audio(youtube_url_or_id="dQw4w9WgXcQ")
    assert result["artifact"]["audioRef"].startswith("aud_")
    assert str(source) not in str(result)
