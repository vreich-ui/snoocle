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
    fragment = fragment.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    fragment = _TAG_RE.sub("", fragment)
    return html_mod.unescape(fragment)


def extract_sheet_text(page_html: str) -> str:
    """Best-effort chord-sheet text from arbitrary HTML."""
    cleaned = _SCRIPT_STYLE_RE.sub("", page_html)
    pres = [_strip_tags(m.group(1)) for m in _PRE_RE.finditer(cleaned)]
    if pres:
        return "\n\n".join(pres)
    text = _strip_tags(cleaned)
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
