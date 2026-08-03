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


def test_studio_shell_and_compiled_assets_are_served():
    r = client.get("/studio/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "/studio/assets/" in r.text
    asset = next(part.split('"')[0] for part in r.text.split('src="') if "/studio/assets/" in part)
    assert client.get(asset).status_code == 200


def test_studio_direct_route_refresh_uses_the_spa_shell():
    r = client.get("/studio/runs")
    assert r.status_code == 200
    assert "Snoocle Studio" in r.text
    assert client.get("/studio/assets/not-a-real-file.js").status_code == 404


def test_studio_shell_is_exempt_but_api_stays_gated(token_enabled):
    assert client.get("/studio/configuration").status_code == 200
    assert client.get("/v1/songs").status_code == 401
    # Prefix matching must not make similarly named future routes public.
    assert client.get("/studio-private").status_code == 401


def test_workbench_exposes_ranked_source_preferences():
    script = client.get("/ui/app.js").text
    assert "Ranked source sites" in script
    assert "source_site_preferences" in script
    assert all(locale in script for locale in ("global", "russian", "hebrew"))


def test_ui_assets_revalidate_so_deploys_are_not_stale():
    # no-cache => the browser revalidates each load (cheap 304 when unchanged),
    # so a deploy shipping new app.js is never masked by a cached old one.
    for path in ("/ui/", "/ui/app.js", "/ui/style.css"):
        assert client.get(path).headers.get("cache-control") == "no-cache", path


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
