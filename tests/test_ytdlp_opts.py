"""yt-dlp YouTube-auth accommodations (cookies + player clients).

YouTube blocks datacenter IPs with a bot check; these knobs let a Cloud Run
deploy authenticate. Off by default (no config -> yt-dlp opts unchanged).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from snoocle_server.audio import acquire
from snoocle_server.audio.acquire import _ytdlp_opts
from snoocle_server.config import settings


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(settings, "ytdlp_cookies", "")
    monkeypatch.setattr(settings, "ytdlp_cookies_file", "")
    monkeypatch.setattr(settings, "ytdlp_player_clients", "")
    monkeypatch.setattr(settings, "ytdlp_proxy", "")
    monkeypatch.setattr(settings, "ytdlp_cache_dir", "")
    # isolate from runtime-uploaded cookies (tested separately)
    monkeypatch.setattr(acquire, "_stored_cookies_txt", lambda: None)
    monkeypatch.setattr(acquire, "_materialized", {})


def test_passthrough_when_unconfigured():
    base = {"quiet": True, "noplaylist": True}
    assert _ytdlp_opts(base) == base
    assert "cookiefile" not in base  # input not mutated


def test_cookies_file_path(monkeypatch):
    monkeypatch.setattr(settings, "ytdlp_cookies_file", "/mnt/secrets/cookies.txt")
    assert _ytdlp_opts({"quiet": True})["cookiefile"] == "/mnt/secrets/cookies.txt"


def test_inline_cookies_materialized_to_a_file(monkeypatch):
    content = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tabc123\n"
    monkeypatch.setattr(settings, "ytdlp_cookies", content)
    opts = _ytdlp_opts({"quiet": True})
    written = Path(opts["cookiefile"])
    assert written.exists()
    assert written.read_text() == content


def test_cookies_file_wins_over_inline(monkeypatch):
    monkeypatch.setattr(settings, "ytdlp_cookies_file", "/mnt/cookies.txt")
    monkeypatch.setattr(settings, "ytdlp_cookies", "inline-should-be-ignored")
    assert _ytdlp_opts({})["cookiefile"] == "/mnt/cookies.txt"


def test_player_clients(monkeypatch):
    monkeypatch.setattr(settings, "ytdlp_player_clients", "default, android , ios,tv")
    opts = _ytdlp_opts({"quiet": True})
    assert opts["extractor_args"]["youtube"]["player_client"] == ["default", "android", "ios", "tv"]
    assert opts["quiet"] is True  # base preserved


def test_proxy(monkeypatch):
    monkeypatch.setattr(settings, "ytdlp_proxy", "socks5://localhost:1055")
    assert _ytdlp_opts({"quiet": True})["proxy"] == "socks5://localhost:1055"
    monkeypatch.setattr(settings, "ytdlp_proxy", "")
    assert "proxy" not in _ytdlp_opts({"quiet": True})


def test_cache_dir(monkeypatch):
    monkeypatch.setattr(settings, "ytdlp_cache_dir", "/data/ytdlp-cache")
    assert _ytdlp_opts({"quiet": True})["cachedir"] == "/data/ytdlp-cache"
    monkeypatch.setattr(settings, "ytdlp_cache_dir", "")
    assert "cachedir" not in _ytdlp_opts({"quiet": True})


def test_download_opts_prefer_small_audio_and_parallel_fragments(tmp_path, monkeypatch):
    """The download call must never pull video when audio-only exists, and
    must download HLS/DASH fragments in parallel (throttling mitigation)."""
    import yt_dlp

    from snoocle_server.audio.acquire import AcquisitionError, download_audio

    captured: dict = {}

    class FakeYDL:
        def __init__(self, opts):
            captured.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, *a, **k):
            raise RuntimeError("stop before any network")

    monkeypatch.setattr(settings, "audio_cache_dir", tmp_path)
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    with pytest.raises(AcquisitionError):
        download_audio("AAAAAAAAAAA")

    assert captured["format"].startswith("bestaudio[abr<=160]/bestaudio")
    assert captured["concurrent_fragment_downloads"] >= 2
