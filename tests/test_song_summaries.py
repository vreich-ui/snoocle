"""List-view song summaries (master plan C1) and the shared UI assets (C0).

`GET /v1/songs` grew an ADDITIVE `items` array so the play-along player's
library grid can render titles, artwork and a timing badge in one request.
The original `songs` array (plain ids) is untouched — the iOS app, the MCP
tools and every existing script read that, and none of them are being asked
to change.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import snoocle_server.api as api_mod
from snoocle_server.api import app
from snoocle_server.schema import Song
from snoocle_server.store import song_has_timing
from snoocle_server.store.memory import InMemorySongRepository

client = TestClient(app)


@pytest.fixture(autouse=True)
def store(monkeypatch):
    """Same isolation the rest of the suite uses: a fresh in-memory repository
    per test, injected into both the API and the pipeline."""
    repo = InMemorySongRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: repo)
    monkeypatch.setattr("snoocle_server.pipeline.get_store", lambda: repo)
    return repo


def _song(song_id: str, *, title: str, artist: str, video: str | None = None,
          line_time: float | None = None) -> Song:
    line: dict = {
        "lineIndex": 0,
        "lyrics": "When I find myself",
        "chordPlacements": [{"charIndex": 0, "chord": "C"}],
    }
    if line_time is not None:
        line["timeSeconds"] = line_time
    payload: dict = {
        "id": song_id,
        "metadata": {"title": title, "artist": artist},
        "lines": [line],
    }
    if video:
        payload["audio"] = {"youtubeVideoId": video}
    return Song.model_validate(payload)


# --- song_has_timing --------------------------------------------------------


def test_has_timing_false_without_any_times():
    assert song_has_timing(_song("a--b", title="A", artist="B")) is False


def test_has_timing_true_from_line_time():
    assert song_has_timing(_song("a--b", title="A", artist="B", line_time=1.5)) is True


def test_has_timing_true_from_chord_time():
    song = Song.model_validate(
        {
            "id": "a--b",
            "metadata": {"title": "A", "artist": "B"},
            "lines": [
                {
                    "lineIndex": 0,
                    "lyrics": "x",
                    "chordPlacements": [{"charIndex": 0, "chord": "C", "timeSeconds": 2.0}],
                }
            ],
        }
    )
    assert song_has_timing(song) is True


def test_has_timing_true_from_legacy_syncmap_only():
    """A v1 song analyzed before Phase A carries its line times ONLY in
    syncMap — it must still show a timing badge, not read as untimed."""
    song = Song.model_validate(
        {
            "id": "a--b",
            "metadata": {"title": "A", "artist": "B"},
            "audio": {
                "youtubeVideoId": "dQw4w9WgXcQ",
                "syncMap": [{"lineIndex": 0, "time": 3.25}],
            },
            "lines": [{"lineIndex": 0, "lyrics": "x", "chordPlacements": []}],
        }
    )
    assert song_has_timing(song) is True


# --- the store contract -----------------------------------------------------


def test_memory_store_summaries_are_denormalized_and_sorted(store):
    store.save(_song("zz--last", title="Last", artist="Z", video="dQw4w9WgXcQ"), "first")
    store.save(_song("aa--first", title="First", artist="A", line_time=0.5), "first")

    rows = store.list_song_summaries()
    assert [r.id for r in rows] == ["aa--first", "zz--last"]

    first, last = rows
    assert (first.title, first.artist) == ("First", "A")
    assert first.has_timing is True
    assert first.youtube_video_id is None
    assert last.youtube_video_id == "dQw4w9WgXcQ"
    assert last.has_timing is False
    assert last.latest_version and last.updated_at


def test_summaries_track_the_latest_save(store):
    store.save(_song("a--b", title="Old", artist="B"), "v1")
    saved = store.save(_song("a--b", title="New", artist="B", line_time=9.0), "v2")

    row = store.list_song_summaries()[0]
    assert row.title == "New"
    assert row.has_timing is True
    assert row.latest_version == saved.version


def test_base_fallback_summary_matches_the_backend_override(store):
    """The SongRepository base implementation (used by any future backend that
    doesn't project) must agree with the memory backend's fast path on
    everything but the write timestamp, which it cannot know."""
    from snoocle_server.store.base import SongRepository

    store.save(_song("a--b", title="T", artist="A", video="dQw4w9WgXcQ", line_time=1.0), "v1")

    fast = store.list_song_summaries()[0]
    slow = SongRepository.list_song_summaries(store)[0]
    assert (slow.id, slow.title, slow.artist) == (fast.id, fast.title, fast.artist)
    assert slow.youtube_video_id == fast.youtube_video_id
    assert slow.has_timing == fast.has_timing
    assert slow.latest_version == fast.latest_version


# --- the endpoint -----------------------------------------------------------


def test_songs_endpoint_keeps_the_id_array_and_adds_items(store):
    store.save(_song("a--b", title="T", artist="A", video="dQw4w9WgXcQ"), "v1")
    body = client.get("/v1/songs").json()

    assert body["songs"] == ["a--b"]  # unchanged contract
    assert body["items"] == [
        {
            "id": "a--b",
            "title": "T",
            "artist": "A",
            "latestVersion": body["items"][0]["latestVersion"],
            "updatedAt": body["items"][0]["updatedAt"],
            "youtubeVideoId": "dQw4w9WgXcQ",
            "hasTiming": False,
        }
    ]


def test_songs_endpoint_is_empty_but_well_formed_with_no_songs():
    body = client.get("/v1/songs").json()
    assert body == {"songs": [], "items": []}


# --- C0: the shared assets are served --------------------------------------


def test_shared_ui_assets_served_and_revalidated():
    for path in ("/ui/tokens.css", "/ui/common.js"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.headers.get("cache-control") == "no-cache", path


def test_admin_shell_loads_tokens_and_common_before_app():
    html = client.get("/ui/").text
    assert "tokens.css" in html
    assert html.index("common.js") < html.index("app.js")


def test_admin_stylesheet_has_no_hardcoded_hex_colours():
    """§3.5 is binding: style.css consumes tokens, it does not define colours.
    (tokens.css is where hex literals belong.)"""
    import re

    css = client.get("/ui/style.css").text
    # Strip comments before scanning so an example in prose can't trip this.
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    assert re.search(r"#[0-9a-fA-F]{3,8}\b", css) is None
