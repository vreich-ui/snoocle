"""The static single-page GUI served by the FastAPI app.

Server-side only (no browser automation): the shell and its assets are served,
`/` redirects into `/ui/`, the auth exemption is exactly the shell (every API
call still needs the token), and the inline bracket line format round-trips and
produces schema-valid Song lines.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from snoocle_server.api import app
from snoocle_server.config import settings
from snoocle_server.schema import Song

client = TestClient(app)
TOKEN = "s3cr3t-personal-token"


# --- serving ---------------------------------------------------------------


def test_root_redirects_to_ui():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (301, 302, 307, 308)
    assert r.headers["location"] == "/ui/"


def test_ui_index_is_served_html():
    r = client.get("/ui/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "app.js" in r.text


def test_ui_static_assets_served():
    assert client.get("/ui/app.js").status_code == 200
    assert client.get("/ui/style.css").status_code == 200


# --- auth: shell exempt, API gated -----------------------------------------


@pytest.fixture()
def token_enabled(monkeypatch):
    monkeypatch.setattr(settings, "api_token", TOKEN)


def test_ui_shell_is_exempt_but_api_is_gated(token_enabled):
    # the static shell loads without a token...
    assert client.get("/ui/").status_code == 200
    assert client.get("/", follow_redirects=False).status_code in (301, 302, 307, 308)
    # ...but every API call still requires it
    assert client.get("/v1/songs").status_code == 401
    r = client.get("/v1/songs", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200


# --- bracket line format ----------------------------------------------------
# A 10-line reference implementation of the SAME rules app.js uses, pinned here
# so the documented format stays schema-valid and round-trips.

_BRACKET_RE = re.compile(r"\[([^\]]*)\]")


def bracket_text_to_line(raw: str, index: int = 0) -> dict:
    lyrics = ""
    placements = []
    last = 0
    for m in _BRACKET_RE.finditer(raw):
        lyrics += raw[last:m.start()]
        placements.append({"charIndex": len(lyrics), "chord": m.group(1)})
        last = m.end()
    lyrics += raw[last:]
    return {"lineIndex": index, "lyrics": lyrics, "chordPlacements": placements}


def line_to_bracket_text(line: dict) -> str:
    lyrics = line["lyrics"]
    out = ""
    cursor = 0
    for p in sorted(line["chordPlacements"], key=lambda p: p["charIndex"]):
        idx = max(0, min(p["charIndex"], len(lyrics)))
        out += lyrics[cursor:idx]
        out += "[" + p["chord"] + "]"
        cursor = max(cursor, idx)
    out += lyrics[cursor:]
    return out


def test_bracket_format_matches_documented_example():
    parsed = bracket_text_to_line("[C]When I [G]find")
    assert parsed == {
        "lineIndex": 0,
        "lyrics": "When I find",
        "chordPlacements": [
            {"charIndex": 0, "chord": "C"},
            {"charIndex": 7, "chord": "G"},
        ],
    }


def test_bracket_format_round_trips():
    original = {
        "lineIndex": 0,
        "lyrics": "When I find myself in times of trouble",
        "chordPlacements": [
            {"charIndex": 0, "chord": "C"},
            {"charIndex": 21, "chord": "G"},
            {"charIndex": 30, "chord": "Am"},
        ],
    }
    assert bracket_text_to_line(line_to_bracket_text(original)) == original


def test_parsed_bracket_line_validates_as_song():
    parsed = bracket_text_to_line("[C]When I [G]find")
    song = Song.model_validate(
        {
            "id": "the-beatles--let-it-be",
            "metadata": {"title": "Let It Be", "artist": "The Beatles"},
            "lines": [parsed],
        }
    )
    assert song.lines[0].lyrics == "When I find"
    assert [p.chord for p in song.lines[0].chordPlacements] == ["C", "G"]
    assert [p.charIndex for p in song.lines[0].chordPlacements] == [0, 7]
