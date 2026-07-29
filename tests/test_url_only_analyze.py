"""URL-only analyze: give a media URL and let the pipeline derive title/artist
from the video's own metadata (no title+artist required).

Also covers the precedence rule for the general case: video-derived metadata
is a FALLBACK for whatever the caller did not supply, never an override of
what they did. A song id is permanent (content-hash versioned store), so
letting a supplied title/artist get silently replaced by a guess would mint
a wrong id with no way to correct it in place."""

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


# ===========================================================================
# Precedence: explicit title/artist beat anything derived from video metadata
# ===========================================================================
#
# A song id is permanent (content-hash versioned store): if a caller-supplied
# title/artist were ever silently discarded in favor of a video-derived guess,
# the resulting id would be wrong forever, with no in-place fix. These pin
# that a full supply skips derivation entirely, and a partial supply derives
# ONLY the missing half.

# What the (fictional) uploader's own metadata says — a channel name and a
# raw, undecorated video title, standing in for the live-store incident's
# "Wil Per" / "Rolling Stones -  Paint It Black LIVE (1966)".
_WRONG_META = ResolvedMeta(
    video_id="flSmiIne-4k",
    video_title="Rolling Stones -  Paint It Black LIVE (1966)",
    title="Paint It Black LIVE (1966)",
    artist="Wil Per",
    uploader="Wil Per",
)


def test_explicit_title_and_artist_survive_a_youtube_url(monkeypatch):
    """Both supplied: identity resolution must not run AT ALL — not just
    "run and get overridden after the fact". Reproduces the live-store
    incident (artist="The Rolling Stones", title="Paint It Black",
    youtubeUrlOrId="flSmiIne-4k") and asserts the id/metadata reflect what
    was given, never the uploader's channel or the raw video title."""
    def boom(url):
        raise AssertionError("extract_metadata ran despite explicit title+artist")
    monkeypatch.setattr(pipeline_mod, "extract_metadata", boom)

    r = client.post(
        "/v1/songs/analyze",
        json={
            "title": "Paint It Black", "artist": "The Rolling Stones",
            "youtubeUrlOrId": "flSmiIne-4k",
            "provider": "mock", "skipAudio": True,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["songId"] == "the-rolling-stones--paint-it-black"
    assert "resolve" not in body["steps"], "no derivation should have run at all"
    assert body["song"]["metadata"]["title"] == "Paint It Black"
    assert body["song"]["metadata"]["artist"] == "The Rolling Stones"


def test_explicit_artist_survives_alongside_a_derived_title(monkeypatch):
    """Partial supply: artist given, title omitted. Only the title is
    derived (from wrong-looking metadata here, on purpose) — the given
    artist must never be replaced by the uploader's channel name."""
    monkeypatch.setattr(pipeline_mod, "extract_metadata", lambda url: _WRONG_META)

    r = client.post(
        "/v1/songs/analyze",
        json={
            "artist": "The Rolling Stones", "youtubeUrlOrId": "flSmiIne-4k",
            "provider": "mock", "skipAudio": True,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["song"]["metadata"]["artist"] == "The Rolling Stones"
    assert body["song"]["metadata"]["artist"] != "Wil Per"
    assert body["songId"].startswith("the-rolling-stones--")


def test_explicit_title_survives_alongside_a_derived_artist(monkeypatch):
    """Partial supply, the other way: title given, artist omitted. Only the
    artist is derived — the given title must never be replaced by the raw,
    undecorated video title."""
    monkeypatch.setattr(pipeline_mod, "extract_metadata", lambda url: _WRONG_META)

    r = client.post(
        "/v1/songs/analyze",
        json={
            "title": "Paint It Black", "youtubeUrlOrId": "flSmiIne-4k",
            "provider": "mock", "skipAudio": True,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["song"]["metadata"]["title"] == "Paint It Black"
    assert body["song"]["metadata"]["title"] != "Paint It Black LIVE (1966)"


def test_the_mcp_tool_forwards_explicit_title_and_artist_unchanged(monkeypatch):
    """Same precedence rule, verified at the MCP tool boundary named in the
    bug report — analyze_and_store_song must not itself transform or drop
    title/artist before (or after) calling the pipeline."""
    import asyncio

    from snoocle_server import mcp_server as mcp_mod

    seen: dict = {}

    async def fake_pipeline(title, artist, **kwargs):
        seen["title"], seen["artist"] = title, artist
        seen.update(kwargs)
        report = pipeline_mod.PipelineReport(song_id="the-rolling-stones--paint-it-black")
        report.reconcile = type(
            "R", (), {"song": type("S", (), {"model_dump": lambda self: {}})()}
        )()
        return report

    monkeypatch.setattr(mcp_mod, "run_pipeline_async", fake_pipeline)
    asyncio.run(mcp_mod.analyze_and_store_song(
        title="Paint It Black", artist="The Rolling Stones",
        youtube_url_or_id="flSmiIne-4k",
    ))

    assert seen["title"] == "Paint It Black"
    assert seen["artist"] == "The Rolling Stones"
    assert seen["youtube_url_or_id"] == "flSmiIne-4k"


def test_title_artist_still_work_without_url():
    # the original path is unchanged
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Let It Be", "artist": "The Beatles", "provider": "mock", "skipAudio": True},
    )
    assert r.status_code == 200, r.text
    assert r.json()["songId"] == "the-beatles--let-it-be"
    assert "resolve" not in r.json()["steps"]  # no derivation needed
