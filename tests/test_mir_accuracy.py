"""Fast-accuracy MIR sampling: window placement, music-onset detection, and
the API accuracy parameter. Pure-logic and librosa-only tests — no ffmpeg."""

from __future__ import annotations

import pytest

from snoocle_server.mir.pipeline import fast_windows


def test_fast_windows_spread_across_musical_span():
    windows = fast_windows(300.0, music_start=20.0, window_seconds=40.0, count=3)
    assert len(windows) == 3
    # inside the musical span, in order, non-overlapping, correct length
    prev_end = 20.0
    for start, end in windows:
        assert start >= prev_end
        assert end - start == pytest.approx(40.0, abs=0.1)
        prev_end = end
    assert windows[-1][1] <= 300.0
    # anchored after the intro, not at 0
    assert windows[0][0] > 20.0


def test_fast_windows_short_song_collapses_to_single_window():
    # 90s of music can't fit 3x40s windows -> one window over the musical span
    assert fast_windows(100.0, music_start=10.0, window_seconds=40.0, count=3) == [(10.0, 100.0)]


def test_fast_windows_degenerate_inputs():
    assert fast_windows(30.0, music_start=45.0, window_seconds=40.0, count=3) == [(0.0, 30.0)]
    assert fast_windows(0.0, music_start=0.0, window_seconds=40.0, count=3) == [(0.0, 0.0)]


def test_detect_music_start_skips_leading_silence(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")

    from snoocle_server.mir.pipeline import detect_music_start

    sr = 22050
    silence = np.random.randn(8 * sr).astype("float32") * 0.001  # 8s near-silence
    t = np.arange(20 * sr) / sr
    music = (0.4 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 277.2 * t)).astype("float32")
    path = tmp_path / "late_start.wav"
    sf.write(path, np.concatenate([silence, music]), sr, subtype="PCM_16")

    start = detect_music_start(path)
    assert 6.0 <= start <= 10.0


def test_detect_music_start_immediate_music(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")

    from snoocle_server.mir.pipeline import detect_music_start

    sr = 22050
    t = np.arange(15 * sr) / sr
    music = (0.4 * np.sin(2 * np.pi * 220.0 * t)).astype("float32")
    path = tmp_path / "immediate.wav"
    sf.write(path, music, sr, subtype="PCM_16")

    assert detect_music_start(path) <= 1.0


def test_analyze_windows_offsets_back_to_track_time(monkeypatch, tmp_path):
    """Per-window results must be shifted back into the ORIGINAL track's time
    coordinates so the chord timeline still aligns with the video."""
    from pathlib import Path

    from snoocle_server.mir import pipeline as mp
    from snoocle_server.mir.base import ChordSegment

    monkeypatch.setattr(mp, "trim", lambda src, dst, s, e: Path(dst).write_bytes(b"x"))
    monkeypatch.setattr(mp, "to_analysis_wav", lambda src, dst, **k: Path(dst).write_bytes(b"x"))
    monkeypatch.setattr(
        mp, "track_beats",
        lambda wav: ([(0.5, 1), (1.0, 2)], 120.0, "4/4", "librosa-fallback"),
    )
    monkeypatch.setattr(
        mp, "recognize_chords",
        lambda wav, beats: ([ChordSegment(start=0.0, end=2.0, chord="C")], "chroma-template-fallback"),
    )

    analysis = mp._analyze_windows(Path("x.wav"), 300.0, [(40.0, 80.0), (150.0, 190.0)], str(tmp_path))

    assert [c.start for c in analysis.chords] == [40.0, 150.0]
    assert [round(b.time, 1) for b in analysis.beats] == [40.5, 41.0, 150.5, 151.0]
    assert analysis.bpm == 120.0
    assert analysis.duration_seconds == 300.0
    assert analysis.sections == []
    assert analysis.engines["sampling"] == "fast: 40-80s, 150-190s"


def test_analyze_api_accepts_accuracy():
    from fastapi.testclient import TestClient

    from snoocle_server import api as api_mod
    from snoocle_server.store.memory import InMemorySongRepository

    api_mod.get_store = lambda: InMemorySongRepository()
    client = TestClient(api_mod.app)

    r = client.post(
        "/v1/songs/analyze",
        json={"title": "X", "artist": "Y", "provider": "mock", "skipAudio": True,
              "accuracy": "fast"},
    )
    assert r.status_code == 200, r.text

    r = client.post(
        "/v1/songs/analyze",
        json={"title": "X", "artist": "Y", "provider": "mock", "skipAudio": True,
              "accuracy": "warp-speed"},
    )
    assert r.status_code == 422
