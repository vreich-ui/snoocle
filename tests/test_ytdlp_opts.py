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
    monkeypatch.setattr(acquire, "_cookie_tmpfile", None)


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
