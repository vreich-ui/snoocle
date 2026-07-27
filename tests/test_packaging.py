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
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "snoocle_server"
UI = PKG / "ui"


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
