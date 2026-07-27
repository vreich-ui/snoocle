#!/usr/bin/env python3
"""Regenerate the player's PWA raster icons from icon.svg's geometry.

Run after editing snoocle_server/ui/play/icons/icon.svg:

    python scripts/make_player_icons.py

Deliberately drawn with Pillow rather than an SVG rasterizer so the repo needs
no cairo/librsvg toolchain (master plan §0.7's no-build-step spirit applies to
assets too). The shapes below are the same rectangles icon.svg declares; keep
the two in step by hand — the icon is six rectangles and a dot.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "snoocle_server" / "ui" / "play" / "icons"

BG = (0x0E, 0x11, 0x16, 255)
CHORD = (0xFF, 0xB4, 0x54, 255)
DIM = (0x8B, 0x96, 0xA5, 255)
ACCENT = (0x5B, 0x8C, 0xFF, 255)
CANVAS = 512


def draw(size: int, maskable: bool) -> Image.Image:
    img = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if maskable:
        # A maskable icon must survive a circular crop: full bleed, art inset.
        d.rectangle([0, 0, CANVAS, CANVAS], fill=BG)
        scale = 0.78
    else:
        d.rounded_rectangle([0, 0, CANVAS - 1, CANVAS - 1], radius=112, fill=BG)
        scale = 1.0

    c = CANVAS / 2

    def rect(x, y, w, h, r, fill):
        x, y = c + (x - c) * scale, c + (y - c) * scale
        d.rounded_rectangle([x, y, x + w * scale, y + h * scale],
                            radius=r * scale, fill=fill)

    for x, w in ((96, 86), (214, 60), (306, 110)):
        rect(x, 150, w, 26, 13, CHORD)
    rect(96, 214, 320, 18, 9, DIM)
    rect(96, 256, 248, 18, 9, DIM)
    rect(96, 316, 320, 18, 9, (*ACCENT[:3], 115))
    rect(96, 316, 150, 18, 9, ACCENT)

    cx, cy, r = 246, 325, 26
    cx, cy, r = c + (cx - c) * scale, c + (cy - c) * scale, r * scale
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ACCENT)

    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    for size, name, maskable in (
        (192, "icon-192.png", False),
        (512, "icon-512.png", False),
        (180, "icon-180.png", False),
        (512, "icon-maskable-512.png", True),
    ):
        draw(size, maskable).save(OUT / name)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
