"""The play-along player at /ui/play/ (master plan Phase C).

Server-side only — no browser automation. What's checked here is what the
server is actually responsible for: every asset the player loads is served,
the vendored libraries are present and are the real thing, the PWA manifest is
installable, and the shell stays exempt from the bearer token while the API
behind it does not.

The player's *logic* is covered by the JS unit tests under `tests_js/`, which
this module also runs so `python -m pytest` stays the single gate.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from snoocle_server.api import app
from snoocle_server.config import settings

client = TestClient(app)
REPO = Path(__file__).resolve().parent.parent
UI = REPO / "snoocle_server" / "ui"


# --- the shell and its assets ----------------------------------------------

PLAYER_ASSETS = [
    "/ui/play/",
    "/ui/play/player.css",
    "/ui/play/player.js",
    "/ui/play/controls.js",
    "/ui/play/diagrams.js",
    "/ui/play/theory.js",
    "/ui/play/timedscroll.js",
    "/ui/play/chordbox.js",
    "/ui/play/sw.js",
    "/ui/play/manifest.webmanifest",
    "/ui/play/icons/icon.svg",
    "/ui/play/icons/icon-192.png",
    "/ui/play/icons/icon-512.png",
    "/ui/play/icons/icon-maskable-512.png",
    "/ui/tokens.css",
    "/ui/common.js",
    "/ui/vendor/chords-db/guitar.json",
    "/ui/vendor/chords-db/ukulele.json",
]


@pytest.mark.parametrize("path", PLAYER_ASSETS)
def test_player_asset_is_served(path):
    r = client.get(path)
    assert r.status_code == 200, path


def test_player_shell_is_html_and_loads_its_scripts():
    html = client.get("/ui/play/").text
    for src in ("../common.js", "timedscroll.js", "theory.js", "chordbox.js",
                "diagrams.js", "controls.js", "player.js", "../tokens.css",
                "player.css"):
        assert src in html, src
    # common.js must load before player.js — the player calls api()/el() at
    # module scope order.
    assert html.index("../common.js") < html.index("player.js")


def test_player_shell_is_token_exempt_like_the_admin(monkeypatch):
    monkeypatch.setattr(settings, "api_token", "s3cr3t")
    assert client.get("/ui/play/").status_code == 200
    assert client.get("/ui/play/player.js").status_code == 200
    # ...but the data it fetches is still gated.
    assert client.get("/v1/songs").status_code == 401


# --- PWA (C5) ---------------------------------------------------------------


def test_manifest_is_valid_and_installable():
    r = client.get("/ui/play/manifest.webmanifest")
    # A manifest served as text/plain is rejected by some browsers.
    assert r.headers["content-type"].startswith("application/manifest+json")
    m = json.loads(r.text)
    assert m["start_url"] == "./"
    assert m["display"] == "standalone"
    # Installability minimum: a name, and a >=192px and a >=512px icon.
    assert m["name"]
    sizes = {icon["sizes"] for icon in m["icons"]}
    assert "192x192" in sizes and "512x512" in sizes
    assert any(icon.get("purpose") == "maskable" for icon in m["icons"])


def test_every_manifest_icon_actually_exists():
    m = json.loads(client.get("/ui/play/manifest.webmanifest").text)
    for icon in m["icons"]:
        r = client.get("/ui/play/" + icon["src"])
        assert r.status_code == 200, icon["src"]


def test_service_worker_precaches_only_assets_that_exist():
    """A service worker that precaches a 404 installs a broken offline mode."""
    sw = (UI / "play" / "sw.js").read_text()
    listed = [
        line.strip().strip('",')
        for line in sw.split("var SHELL_ASSETS = [")[1].split("];")[0].splitlines()
        if line.strip().startswith('"')
    ]
    assert listed, "SHELL_ASSETS did not parse"
    for rel in listed:
        if rel in ("./", "./index.html"):
            continue
        path = (UI / "play" / rel).resolve()
        assert path.exists(), f"service worker precaches a missing file: {rel}"


def test_service_worker_never_caches_the_video_origin():
    sw = (UI / "play" / "sw.js").read_text()
    assert "url.origin !== self.location.origin" in sw
    assert "youtube" not in sw.lower() or "not cached" in sw.lower()


# --- vendored libraries -----------------------------------------------------


def test_vendored_chords_db_has_the_suffixes_the_fallback_ladder_needs():
    db = json.loads((UI / "vendor" / "chords-db" / "guitar.json").read_text())
    suffixes = {e["suffix"] for entries in db["chords"].values() for e in entries}
    for needed in ("major", "minor", "7", "m7", "maj7", "dim", "sus4", "sus2", "9"):
        assert needed in suffixes, needed
    # sharp keys are spelled "Csharp" in the chord index — diagrams.js relies
    # on that exact token.
    assert "Csharp" in db["chords"]
    assert db["main"]["strings"] == 6


def test_vendored_libraries_carry_their_licences():
    licence = UI / "vendor" / "chords-db" / "LICENSE"
    assert licence.exists()
    assert "MIT" in licence.read_text()
    assert (UI / "vendor" / "README.md").exists()


def test_the_diagram_renderer_is_ours_and_has_no_dependency():
    """vexchords' minified bundle is broken in current Chromium and its
    unminified one is 152 KB gzipped, so the chord box is drawn in-repo.
    Guard against it creeping back in as a script tag."""
    assert not (UI / "vendor" / "vexchords").exists()
    assert "vexchords" not in client.get("/ui/play/").text
    box = (UI / "play" / "chordbox.js").read_text()
    assert "createElementNS" in box  # plain SVG, no library


def test_no_cdn_references_in_player_assets():
    """§0.7: vendored, not fetched. The ONLY permitted runtime external is
    YouTube's IFrame Player API, which C2 names explicitly."""
    allowed = ("https://www.youtube.com/iframe_api", "https://img.youtube.com/vi/")
    for path in sorted((UI / "play").glob("*.js")):
        text = path.read_text()
        for line in text.splitlines():
            if "https://" not in line:
                continue
            if any(a in line for a in allowed):
                continue
            # URLs inside comments (docs/links) are fine.
            stripped = line.strip()
            if stripped.startswith(("*", "//", "/*")):
                continue
            raise AssertionError(f"{path.name}: external URL in code: {line.strip()}")


# --- the JS unit tests ------------------------------------------------------


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_unit_tests_pass():
    """Runs `node --test tests_js/*.test.mjs` so the pure scroll/theory/diagram
    logic is gated by the same command as everything else."""
    result = subprocess.run(
        ["node", "--test", "tests_js/*.test.mjs"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    assert "# fail 0" in result.stdout or "ℹ fail 0" in result.stdout
