"""Fetch a candidate page and extract chord-sheet-ish text, site-agnostically.

Preference order: <pre> blocks (the near-universal chord-sheet container),
then a whole-page tag-strip fallback. No site-specific selectors.
"""

from __future__ import annotations

import html as html_mod
import re

import httpx

from ..config import settings

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"

_PRE_RE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.DOTALL | re.IGNORECASE)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_NOISE_BLOCK_RE = re.compile(
    r"<(nav|aside|header|footer)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_CONTENT_RE = re.compile(
    r"<(main|article)[^>]*>(.*?)</\1>|"
    r"<div[^>]+class=[\"'][^\"']*(?:kix-page|docs-material|chord|song-content)[^\"']*[\"'][^>]*>(.*?)</div>",
    re.DOTALL | re.IGNORECASE,
)
_BREAK_RE = re.compile(r"<br\s*/?>|</(?:div|p|li|tr|h[1-6]|section)\s*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def fetch_page(url: str) -> str:
    r = httpx.get(
        url,
        headers={"User-Agent": _UA, "Accept-Language": "en"},
        timeout=settings.fetch_timeout_seconds,
        follow_redirects=True,
    )
    r.raise_for_status()
    return r.text


def _strip_tags(fragment: str) -> str:
    fragment = _BREAK_RE.sub("\n", fragment)
    fragment = _TAG_RE.sub("", fragment)
    return html_mod.unescape(fragment)


def extract_sheet_text(page_html: str) -> str:
    """Best-effort chord-sheet text from arbitrary HTML."""
    cleaned = _NOISE_BLOCK_RE.sub("", _SCRIPT_STYLE_RE.sub("", page_html))
    pres = [_strip_tags(m.group(1)) for m in _PRE_RE.finditer(cleaned)]
    if pres:
        return "\n\n".join(pres)
    fragments = [next(value for value in m.groups()[1:] if value is not None) for m in _CONTENT_RE.finditer(cleaned)]
    texts = [_strip_tags(fragment) for fragment in fragments]
    texts.append(_strip_tags(cleaned))
    # Pick the semantic container with the strongest actual chord-sheet parse.
    # This rejects ad-heavy sidebars even when they contain much more text than
    # the song. The full page remains the final fallback.
    from .chordsheet import parse_chord_sheet
    text = max(
        texts,
        key=lambda value: (
            parse_chord_sheet(value).placement_count,
            parse_chord_sheet(value).lyric_line_count,
        ),
    )
    # collapse the tag-soup blank-line noise but keep line structure
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[str] = []
    blank = 0
    for ln in lines:
        if not ln.strip():
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(ln)
    return "\n".join(out)
