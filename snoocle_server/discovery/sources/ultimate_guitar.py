"""Ultimate Guitar discovery source (master plan B5) — personal use only.

UG has no documented, stable public API. Every third-party integration
(scrapers like Pilfer's/joncardasis') works against the same undocumented
surface: UG's own web pages embed the JSON their React frontend hydrates
from in a `<div class="js-store" data-content="...">` attribute — on the
search-results page AND every tab page. That's what this module reads.
`_JSON_PATHS` below is the ONE place documenting exactly which key paths we
read, with fallbacks, precisely because HTML/JSON-shape drift on an
undocumented endpoint is the expected failure mode here, not the exception
(see the module-level resilience rule).

Resilience rule (matches every other discovery source's contract): ANY
failure — network, HTTP status, missing/renamed JSON keys, empty content —
returns `[]`/`None` and logs at INFO. This source NEVER raises into the
pipeline. It is also OFF by default (`SNOOCLE_SOURCE_UG=0`) since it depends
on an undocumented endpoint that can break, change shape, or draw UG's ToS
attention — one flag disables it instantly.

Chords found here are UG's community-contributed FRETBOARD-SHAPE symbols,
not sounding harmony — exactly like every other text source. The declared
capo (from the tab's own metadata, falling back to whatever the generic
parser found written into the sheet text itself) is transposed away before
these ever become a CandidateSource, satisfying the sounding-harmony rule
the same way `discovery.service.candidate_from_text` does for web results.
"""

from __future__ import annotations

import html
import json
import logging
import math
import re
from typing import Any, Optional

import httpx

from ...chords import ChordParseError, shape_to_sounding
from ...config import settings
from ...schema.song import Line
from ..chordsheet import parse_chord_sheet
from ..models import CandidateSource, SectionStart

log = logging.getLogger(__name__)

# --- constants block: every URL/selector UG-specific lives here, so a real
# layout change only ever needs an edit in this one place. ------------------
_SEARCH_URL = "https://www.ultimate-guitar.com/search.php"
_TAB_TYPES = ("Official", "Chords")  # exclude Tab/Bass/Ukulele/Pro
_JS_STORE_RE = re.compile(r'class="js-store"[^>]*data-content="(?P<blob>[^"]*)"')
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
_TIMEOUT_SECONDS = 20.0

# UG's own inline-chord markup ([ch]C[/ch]) and tab-block wrapper
# ([tab]...[/tab], used for fret-diagram runs, not lyric/chord lines).
_CH_TAG_RE = re.compile(r"\[ch\](.*?)\[/ch\]", re.IGNORECASE | re.DOTALL)
_TAB_BLOCK_RE = re.compile(r"\[tab\].*?\[/tab\]", re.IGNORECASE | re.DOTALL)

# The js-store JSON shape is undocumented and has been reported to vary by
# UG frontend version; each of these tries several candidate key paths in
# order and uses the first one present, rather than betting the whole
# source on a single exact shape.
_SEARCH_RESULT_LIST_PATHS = (
    ("results",),
    ("other_search_results",),
    ("tabs",),
)
_TAB_CONTENT_PATHS = (("tab_view", "wiki_tab", "content"),)
_TAB_CAPO_PATHS = (
    ("tab", "capo"),
    ("tab_view", "meta", "capo"),
    ("meta", "capo"),
)
_TAB_TUNING_PATHS = (
    ("tab_view", "meta", "tuning", "name"),
    ("meta", "tuning", "name"),
    ("tab", "tuning"),
)
_TAB_RATING_PATHS = (("tab", "rating"), ("tab_view", "tab", "rating"))
_TAB_VOTES_PATHS = (("tab", "votes"), ("tab_view", "tab", "votes"))
_TAB_SONG_NAME_PATHS = (("tab", "song_name"),)
_TAB_ARTIST_NAME_PATHS = (("tab", "artist_name"),)


def _get_path(data: dict, path: tuple[str, ...]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _get_first(data: dict, paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        v = _get_path(data, path)
        if v is not None:
            return v
    return None


def _extract_js_store(page_html: str) -> Optional[dict]:
    m = _JS_STORE_RE.search(page_html)
    if not m:
        return None
    try:
        return json.loads(html.unescape(m.group("blob")))
    except (json.JSONDecodeError, ValueError) as e:
        log.info("ultimate-guitar: js-store blob did not parse as JSON: %s", e)
        return None


def _ug_headers() -> dict:
    return {"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml"}


def _page_data(page_html: str) -> Optional[dict]:
    store = _extract_js_store(page_html)
    if not store:
        return None
    return _get_path(store, ("store", "page", "data"))


def search_ultimate_guitar(title: str, artist: str, max_results: int = 5) -> list[dict]:
    """Raw UG search hits: [{id, song_name, artist_name, tab_url, rating,
    votes, type}, ...] — best-effort, `[]` on ANY failure."""
    try:
        r = httpx.get(
            _SEARCH_URL,
            params={"search_type": "title", "value": f"{artist} {title}".strip()},
            headers=_ug_headers(),
            timeout=_TIMEOUT_SECONDS,
        )
        r.raise_for_status()
        data = _page_data(r.text)
        if not data:
            return []
        results = _get_first(data, _SEARCH_RESULT_LIST_PATHS) or []
        hits = [
            hit
            for hit in results
            if isinstance(hit, dict) and hit.get("type") in _TAB_TYPES and hit.get("tab_url")
        ]
        hits.sort(
            key=lambda hit: (
                0 if str(hit.get("type", "")).casefold() == "official" else 1,
                -float(hit.get("rating") or 0),
                -int(hit.get("votes") or 0),
            )
        )
        return hits[:max_results]
    except Exception as e:  # noqa: BLE001 — best-effort, never raises
        log.info("ultimate-guitar search failed (continuing without it): %s", e)
        return []


def fetch_tab(tab_url: str) -> Optional[dict]:
    """{content, capo, tuning, rating, votes, song_name, artist_name} for one
    tab page, or `None` on ANY failure (network, non-2xx, no js-store, no
    chord content)."""
    try:
        r = httpx.get(tab_url, headers=_ug_headers(), timeout=_TIMEOUT_SECONDS)
        r.raise_for_status()
        data = _page_data(r.text)
        if not data:
            return None
        content = _get_first(data, _TAB_CONTENT_PATHS)
        if not content:
            return None
        return {
            "content": content,
            "capo": _get_first(data, _TAB_CAPO_PATHS) or 0,
            "tuning": _get_first(data, _TAB_TUNING_PATHS),
            "rating": _get_first(data, _TAB_RATING_PATHS),
            "votes": _get_first(data, _TAB_VOTES_PATHS),
            "song_name": _get_first(data, _TAB_SONG_NAME_PATHS),
            "artist_name": _get_first(data, _TAB_ARTIST_NAME_PATHS),
        }
    except Exception as e:  # noqa: BLE001 — best-effort, never raises
        log.info("ultimate-guitar tab fetch failed for %s (continuing without it): %s", tab_url, e)
        return None


def _ug_markup_to_bracket(content: str) -> str:
    """UG's `[ch]C[/ch]` -> our `[C]` inline-bracket convention (identical
    semantics, different tag) so the EXISTING generic chord-sheet parser can
    read it unmodified; `[tab]...[/tab]` wrappers are stripped (their
    contents are ASCII fret diagrams, not lyric/chord lines)."""
    text = _TAB_BLOCK_RE.sub("", content)
    text = _CH_TAG_RE.sub(lambda m: f"[{m.group(1)}]", text)
    return html.unescape(text)


def _confidence_prior(rating: Optional[float], votes: Optional[int]) -> float:
    """min(0.95, 0.5 + 0.1*log10(votes+1) + 0.05*rating) — per the master
    plan: UG's community rating/vote count is real prior evidence (not just
    a tiebreaker), so a well-voted tab starts reconciliation with a
    meaningfully higher confidence than an arbitrary, unvetted web hit."""
    votes = votes or 0
    rating = rating or 0.0
    score = 0.5 + 0.1 * math.log10(votes + 1) + 0.05 * rating
    return round(min(0.95, max(0.0, score)), 3)


def _retranspose(lines: list[Line], capo: int) -> list[Line]:
    """Re-apply the sounding-harmony transform with UG's METADATA capo, for
    the case where the sheet text itself carried no "Capo X" line (so the
    generic parser's own declared_capo came back 0 and left chords
    untransposed). Composable with the parser's own capo-0 pass: transposing
    an already-capo-0-transposed (i.e. unchanged) chord by `capo` semitones
    is identical to transposing the raw symbol by `capo` directly."""
    out = []
    for line in lines:
        new_placements = []
        for p in line.chordPlacements:
            try:
                sounding = shape_to_sounding(p.chord, capo)
            except (ChordParseError, ValueError):
                sounding = p.chord  # leave as-is; validator will reject if truly bad
            new_placements.append(p.model_copy(update={"chord": sounding}))
        out.append(line.model_copy(update={"chordPlacements": new_placements}))
    return out


def discover_ultimate_guitar(title: str, artist: str, max_candidates: int = 3) -> list[CandidateSource]:
    """Ultimate Guitar candidates — `[]` unless `SNOOCLE_SOURCE_UG=1`.
    Every stage is independently best-effort: one tab failing (dead link,
    no chords, unparseable) is skipped, never fatal to the others or to the
    caller's overall discovery run.
    """
    if not settings.source_ug_enabled:
        return []
    hits = search_ultimate_guitar(title, artist, max_results=max_candidates * 3)
    hits.sort(
        key=lambda hit: (
            0 if str(hit.get("type", "")).casefold() == "official" else 1,
            -float(hit.get("rating") or 0),
            -int(hit.get("votes") or 0),
        )
    )
    candidates: list[CandidateSource] = []
    for hit in hits:
        if len(candidates) >= max_candidates:
            break
        tab_url = hit.get("tab_url")
        tab = fetch_tab(tab_url) if tab_url else None
        if tab is None:
            continue
        candidate = candidate_from_ug_tab(
            hit, tab, title=title, artist=artist,
            source_id=f"ultimate-guitar-{hit.get('id') or len(candidates) + 1}",
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def candidate_from_ug_tab(
    hit: dict,
    tab: dict,
    *,
    title: str,
    artist: str,
    source_id: str,
) -> CandidateSource | None:
    """Parse one already-fetched UG tab; shared by discovery and prefetch."""
    sheet = parse_chord_sheet(_ug_markup_to_bracket(tab["content"]))
    if not sheet.is_plausible:
        return None

    declared_capo = int(tab.get("capo") or 0) or sheet.declared_capo
    lines = sheet.lines
    if declared_capo and not sheet.declared_capo:
        lines = _retranspose(lines, declared_capo)

    rating = tab.get("rating") if tab.get("rating") is not None else hit.get("rating")
    votes = tab.get("votes") if tab.get("votes") is not None else hit.get("votes")
    confidence = _confidence_prior(rating, votes)
    notes_parts = [
        f"ultimate-guitar type={hit.get('type')} rating={rating} votes={votes}"
    ]
    if declared_capo:
        notes_parts.append(
            f"source declared capo {declared_capo}; chords transposed to sounding "
            f"pitch at ingestion (+{declared_capo} semitones)"
        )
    tuning = tab.get("tuning")
    if tuning and str(tuning).strip().lower() not in ("", "standard", "e a d g b e"):
        notes_parts.append(f"declared tuning: {tuning}")
    full_song = (
        len(lines) >= 8
        and sheet.placement_count >= 8
        and (len(sheet.sections_hint) >= 2 or sheet.lyric_line_count >= 12)
    )
    return CandidateSource(
        sourceId=source_id,
        url=hit.get("tab_url"),
        title=f"{tab.get('artist_name') or artist} - {tab.get('song_name') or title}",
        declaredCapo=declared_capo,
        declaredKey=sheet.declared_key,
        confidence=confidence,
        parseConfidence=confidence,
        coverage="full-song" if full_song else "partial",
        sectionsHint=sheet.sections_hint,
        sectionStarts=[
            SectionStart(name=n, startLineIndex=i)
            for n, i in sheet.section_starts
            if i < len(lines)
        ],
        lines=lines,
        notes="; ".join(notes_parts),
    )
