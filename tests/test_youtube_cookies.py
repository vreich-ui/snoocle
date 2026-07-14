"""In-app YouTube cookie upload: the iOS app harvests a signed-in session and
POSTs it so server-side yt-dlp can get past YouTube's datacenter bot-check.

These endpoints hold the user's Google session, so they require the app-level
token to be configured (SNOOCLE_API_TOKEN); otherwise they refuse (409)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server.api import app
from snoocle_server.audio import acquire
from snoocle_server.config import settings
from snoocle_server.store.memory import InMemorySongRepository

client = TestClient(app)
TOKEN = "cfg-token"
H = {"Authorization": f"Bearer {TOKEN}"}
COOKIES = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tabc123\n"


@pytest.fixture()
def store_and_auth(monkeypatch):
    store = InMemorySongRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: store)
    monkeypatch.setattr(settings, "api_token", TOKEN)
    return store


def test_cookie_endpoints_refuse_when_token_not_configured(monkeypatch):
    monkeypatch.setattr(api_mod, "get_store", lambda: InMemorySongRepository())
    monkeypatch.setattr(settings, "api_token", "")  # unauthenticated service
    assert client.get("/v1/config/youtube-cookies").status_code == 409
    r = client.post("/v1/config/youtube-cookies", json={"cookiesTxt": COOKIES})
    assert r.status_code == 409
    assert "SNOOCLE_API_TOKEN" in r.json()["detail"]


def test_store_status_clear_roundtrip(store_and_auth):
    store = store_and_auth
    assert client.get("/v1/config/youtube-cookies", headers=H).json() == {"configured": False}

    r = client.post("/v1/config/youtube-cookies", headers=H, json={"cookiesTxt": COOKIES, "source": "app"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {"status": "stored", "updatedAt": body["updatedAt"], "source": "app", "lineCount": 1}

    status = client.get("/v1/config/youtube-cookies", headers=H).json()
    assert status["configured"] is True and status["lineCount"] == 1 and status["source"] == "app"
    # the actual cookies reached the store for server-side yt-dlp use
    assert store.get_youtube_cookies_txt() == COOKIES

    assert client.delete("/v1/config/youtube-cookies", headers=H).json() == {"status": "cleared"}
    assert client.get("/v1/config/youtube-cookies", headers=H).json() == {"configured": False}


def test_structured_cookies_converted_to_netscape(store_and_auth):
    store = store_and_auth
    r = client.post(
        "/v1/config/youtube-cookies",
        headers=H,
        json={"cookies": [
            {"name": "SID", "value": "abc", "domain": ".youtube.com", "path": "/",
             "expires": 1893456000, "secure": True},
            {"name": "HSID", "value": "def", "secure": False},
        ]},
    )
    assert r.status_code == 200, r.text
    txt = store.get_youtube_cookies_txt()
    assert txt.startswith("# Netscape HTTP Cookie File")
    assert "\t".join([".youtube.com", "TRUE", "/", "TRUE", "1893456000", "SID", "abc"]) in txt
    assert "\t".join([".youtube.com", "TRUE", "/", "FALSE", "0", "HSID", "def"]) in txt


def test_empty_cookies_rejected(store_and_auth):
    assert client.post("/v1/config/youtube-cookies", headers=H, json={}).status_code == 422
    r = client.post("/v1/config/youtube-cookies", headers=H, json={"cookiesTxt": "# only a comment\n"})
    assert r.status_code == 422


def test_uploaded_cookies_feed_yt_dlp(monkeypatch):
    """Stored cookies take precedence over env config in _resolve_cookiefile,
    and land in the yt-dlp opts."""
    store = InMemorySongRepository()
    store.set_youtube_cookies(COOKIES, source="app")
    monkeypatch.setattr("snoocle_server.store.get_repository", lambda: store)
    monkeypatch.setattr(acquire, "_materialized", {})
    monkeypatch.setattr(settings, "ytdlp_cookies", "env-should-be-ignored")

    path = acquire._resolve_cookiefile()
    assert Path(path).read_text() == COOKIES
    assert acquire._ytdlp_opts({"quiet": True})["cookiefile"] == path
