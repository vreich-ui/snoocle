"""Packaging: every browser asset must survive the wheel build.

This exists because of a defect that all 389 other tests were blind to. The
Docker image runs the INSTALLED package, not the source tree, so a static file
that isn't declared as package data simply isn't there in production — while
every server-side test, which imports from the source tree, keeps passing.

The original `package-data = ["ui/*"]` did exactly that: `*` does not cross a
directory separator, so it shipped the admin and silently dropped the whole
player (`ui/play/`, its icons, and `ui/vendor/`). A Cloud Run deploy would have
served `/ui/` fine and 404'd `/ui/play/`.

Rather than assert a hardcoded file list (which rots), this walks the real ui/
tree and checks each file against the declared glob patterns — so a new asset
directory added later is covered the day it appears.
"""

from __future__ import annotations

import fnmatch
import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "snoocle_server"
UI = PKG / "ui"
STUDIO = PKG / "studio"


def _package_data_patterns() -> list[str]:
    with (REPO / "pyproject.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    return cfg["tool"]["setuptools"]["package-data"]["snoocle_server"]


def _matches(rel: str, patterns: list[str]) -> bool:
    """setuptools glob semantics: `*` stops at a separator, `**` does not."""
    for pattern in patterns:
        if "**" in pattern:
            # `ui/**/*` covers ui/ at any depth.
            prefix = pattern.split("**")[0].rstrip("/")
            if rel.startswith(prefix + "/") or fnmatch.fnmatch(rel, pattern):
                return True
        elif fnmatch.fnmatch(rel, pattern) and rel.count("/") == pattern.count("/"):
            return True
    return False


def _ui_files() -> list[str]:
    return [
        str(p.relative_to(PKG)).replace("\\", "/")
        for p in sorted(UI.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    ]


def test_every_ui_file_is_declared_as_package_data():
    patterns = _package_data_patterns()
    missing = [rel for rel in _ui_files() if not _matches(rel, patterns)]
    assert not missing, (
        "these browser assets would NOT ship in the wheel (and would 404 in "
        f"the Docker image): {missing}\npackage-data patterns: {patterns}"
    )


def test_the_glob_actually_reaches_the_player_and_its_vendor_files():
    """A regression guard aimed squarely at the `*` vs `**` mistake."""
    patterns = _package_data_patterns()
    for rel in (
        "ui/play/index.html",
        "ui/play/player.js",
        "ui/play/timedscroll.js",
        "ui/play/chordbox.js",
        "ui/play/sw.js",
        "ui/play/manifest.webmanifest",
        "ui/play/icons/icon-192.png",
        "ui/vendor/chords-db/guitar.json",
    ):
        assert (PKG / rel).exists(), f"fixture drift: {rel} is gone"
        assert _matches(rel, patterns), f"{rel} would be dropped from the wheel"


@pytest.mark.parametrize("rel", ["ui/index.html", "ui/app.js", "ui/admin-d.js", "ui/tokens.css"])
def test_admin_assets_still_covered(rel):
    assert _matches(rel, _package_data_patterns())


def test_built_studio_assets_are_declared_as_package_data():
    """The installed wheel, not the source checkout, serves /studio/."""
    assert (STUDIO / "index.html").exists()
    missing = [
        str(p.relative_to(PKG)).replace("\\", "/")
        for p in sorted(STUDIO.rglob("*"))
        if p.is_file() and not _matches(str(p.relative_to(PKG)).replace("\\", "/"), _package_data_patterns())
    ]
    assert not missing


# --- dependency ceilings ------------------------------------------------------
#
# The other half of "the image runs the INSTALLED package": the image also
# resolves dependencies fresh at build time, while a developer's venv keeps
# whatever it resolved months ago. So an upstream major release can break every
# deploy without a single line of our code changing, and the whole test suite
# stays green on the stale venv that can no longer reproduce it.
#
# That is not hypothetical — it is what happened: `mcp` was declared `>=1.10.0`
# with no ceiling, a rebuild resolved to 2.0.0, and 2.0.0 removed
# `mcp.server.fastmcp`, which `mcp_server.py` imports at module scope. Every
# Cloud Run instance exited(1) at import before it could bind $PORT.


def _dependency_specs() -> dict[str, str]:
    with (REPO / "pyproject.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    specs = {}
    for raw in cfg["project"]["dependencies"]:
        name = re.split(r"[<>=!\[ ]", raw, maxsplit=1)[0].strip()
        specs[name] = raw
    return specs


def test_mcp_is_capped_below_the_release_that_dropped_fastmcp():
    """A floor alone is not a version constraint — it is an open invitation."""
    spec = _dependency_specs().get("mcp")
    assert spec is not None, "mcp is no longer a declared dependency"
    assert "<" in spec, (
        "the upper bound on `mcp` is load-bearing: 2.0.0 removed "
        "mcp.server.fastmcp and every container died at import. Removing this "
        "ceiling requires porting mcp_server.py to the 2.x server API first."
    )


def test_the_module_the_ceiling_protects_actually_imports():
    """Pins the reason the ceiling exists, so it fails loudly rather than
    turning back into a deploy-time surprise."""
    from mcp.server.fastmcp import FastMCP  # noqa: F401
