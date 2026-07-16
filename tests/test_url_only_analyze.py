"""URL-only analyze: give a media URL and let the pipeline derive title/artist
from the video's own metadata (no title+artist required)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server import pipeline as pipeline_mod
from snoocle_server.api import app
from snoocle_server.audio.acquire import ResolvedMeta, derive_title_artist
from snoocle_server.store.memory import InMemorySongRepository

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch):
    store = InMemorySongRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: store)
    monkeypatch.setattr("snoocle_server.pipeline.get_store", lambda: store)
    return store


@pytest.mark.parametrize(
    "info, expected",
    [
        # explicit music metadata wins over the noisy video title
        ({"track": "Let It Be", "artist": "The Beatles", "title": "Let It Be (Remastered 2009)"},
         ("Let It Be", "The Beatles")),
        # "Artist - Title (Official ...)" with the decoration stripped
        ({"title": "The Beatles - Let It Be (Official Music Video)", "uploader": "TheBeatlesVEVO"},
         ("Let It Be", "The Beatles")),
        # en-dash separator
        ({"title": "Queen – Bohemian Rhapsody (Official Video)", "uploader": "Queen Official"},
         ("Bohemian Rhapsody", "Queen")),
        # bracketed decoration
        ({"title": "Radiohead - Creep [Official Video]", "uploader": "Radiohead"},
         ("Creep", "Radiohead")),
        # YouTube Music "- Topic" channel, no separator in the title
        ({"title": "Yesterday", "uploader": "The Beatles - Topic"},
         ("Yesterday", "The Beatles")),
        # plain fallback: title as-is, uploader as artist
        ({"title": "Some Jam", "uploader": "GarageBandTV"},
         ("Some Jam", "GarageBandTV")),
        # quoted song name, no dash separator (live/one-off uploads); uploader
        # is a show channel, not the artist
        ({"title": "Blues Traveler \"Hook\" at Howard Stern's 1996 Birthday Show",
          "uploader": "The Howard Stern Show"},
         ("Hook", "Blues Traveler")),
        # same pattern with curly quotes
        ({"title": "Blues Traveler “Hook” at Howard Stern's 1996 Birthday Show",
          "uploader": "The Howard Stern Show"},
         ("Hook", "Blues Traveler")),
        # dash separator with a quoted right side: wrapping quotes stripped
        ({"title": 'Taylor Swift - "Blank Space" (Official Video)', "uploader": "TaylorSwiftVEVO"},
         ("Blank Space", "Taylor Swift")),
    ],
)
def test_derive_title_artist(info, expected):
    assert derive_title_artist(info) == expected


def test_analyze_from_url_only_derives_identity(monkeypatch):
    """No title/artist — just a URL. The pipeline resolves them and persists a
    song under the derived id."""
    meta = ResolvedMeta(
        video_id="dQw4w9WgXcQ",
        video_title="Rick Astley - Never Gonna Give You Up (Official Video)",
        title="Never Gonna Give You Up",
        artist="Rick Astley",
    )
    # deterministic + offline: mock provider skips discovery, and we stub the
    # metadata fetch so no network is touched.
    monkeypatch.setattr(pipeline_mod, "extract_metadata", lambda url: meta)

    r = client.post(
        "/v1/songs/analyze",
        json={"youtubeUrlOrId": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
              "provider": "mock", "skipAudio": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["songId"] == "rick-astley--never-gonna-give-you-up"
    assert body["steps"]["resolve"].startswith("ok:")
    assert body["song"]["metadata"]["title"] == "Never Gonna Give You Up"
    assert body["song"]["metadata"]["artist"] == "Rick Astley"
    assert body["storedVersion"]
    # persisted under the derived id
    assert "rick-astley--never-gonna-give-you-up" in client.get("/v1/songs").json()["songs"]


def test_analyze_requires_identity_or_url():
    # neither title+artist nor a URL -> 422 (bad input)
    assert client.post("/v1/songs/analyze", json={"provider": "mock", "skipAudio": True}).status_code == 422
    # title without artist is also insufficient
    assert client.post("/v1/songs/analyze", json={"title": "Solo", "provider": "mock"}).status_code == 422


def test_title_artist_still_work_without_url():
    # the original path is unchanged
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Let It Be", "artist": "The Beatles", "provider": "mock", "skipAudio": True},
    )
    assert r.status_code == 200, r.text
    assert r.json()["songId"] == "the-beatles--let-it-be"
    assert "resolve" not in r.json()["steps"]  # no derivation needed
