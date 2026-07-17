"""Text-source discovery orchestration.

Gather GENEROUSLY: reconciliation quality (and cost — fewer repair cycles)
improves with more independent candidate sources, so we keep every plausible
parse up to the configured cap, not just the best one or two. Candidates
remain separate, each with its own confidence/provenance, until the
reconciliation step.

`search_fn` / `fetch_fn` are injectable so the pipeline is testable offline
and alternative discovery mechanisms (e.g. an MCP-side search) can be
plugged in.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Callable

from .chordsheet import parse_chord_sheet
from .fetch import extract_sheet_text, fetch_page
from .models import CandidateSource, SectionStart
from .search import SearchError, SearchHit, web_search
from ..audio.acquire import parse_dash_title, parse_quoted_track
from ..config import settings

log = logging.getLogger(__name__)

SearchFn = Callable[[str, int], list[SearchHit]]
FetchFn = Callable[[str], str]

_QUOTE_RE = re.compile(r'["“”„‟]+')


def _phrase(term: str) -> str:
    """Exact-phrase query term; quotes embedded in the term itself (common in
    video-derived titles like 'Blues Traveler "Hook" at ...') would otherwise
    terminate the phrase early and garble the whole query."""
    return '"' + " ".join(_QUOTE_RE.sub(" ", term).split()) + '"'


def _confidence(sheet) -> float:
    """Heuristic pre-reconciliation confidence for a parsed sheet."""
    score = 0.2
    score += min(sheet.placement_count / 60.0, 0.4)  # chord density
    score += min(sheet.lyric_line_count / 40.0, 0.2)  # lyric coverage
    if sheet.sections_hint:
        score += 0.1
    if sheet.declared_key:
        score += 0.05
    return round(min(score, 0.95), 3)


def candidate_from_text(text: str, source_id: str, url: str | None = None, title: str | None = None) -> CandidateSource | None:
    sheet = parse_chord_sheet(text)
    if not sheet.is_plausible:
        return None
    notes = None
    if sheet.declared_capo:
        notes = (
            f"source declared capo {sheet.declared_capo}; chords transposed to "
            f"sounding pitch at ingestion (+{sheet.declared_capo} semitones)"
        )
    return CandidateSource(
        sourceId=source_id,
        url=url,
        title=title,
        retrievedAt=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        declaredCapo=sheet.declared_capo,
        declaredKey=sheet.declared_key,
        confidence=_confidence(sheet),
        sectionsHint=sheet.sections_hint,
        sectionStarts=[
            SectionStart(name=n, startLineIndex=i)
            for n, i in sheet.section_starts
            if i < len(sheet.lines)
        ],
        lines=sheet.lines,
        notes=notes,
    )


def discover_sources(
    title: str,
    artist: str,
    max_candidates: int | None = None,
    search_fn: SearchFn | None = None,
    fetch_fn: FetchFn | None = None,
) -> list[CandidateSource]:
    """General web search -> fetch -> parse -> ranked candidate sources."""
    max_candidates = max_candidates or settings.search_max_candidates
    search_fn = search_fn or (lambda q, n: web_search(q, n))
    fetch_fn = fetch_fn or fetch_page

    # Video-derived identities ('Amy Winehouse - Back To Black' with the channel
    # name as the artist, or 'Artist "Track" at some show') rarely match any
    # chord sheet literally. When the literal identity finds nothing — INCLUDING
    # when its over-specific query makes every search backend return zero hits —
    # retry with the cleaner identity embedded in the title itself.
    primary_error: SearchError | None = None
    try:
        candidates = _search_and_parse(title, artist, max_candidates, search_fn, fetch_fn)
    except SearchError as e:
        primary_error = e
        candidates = []

    if not candidates:
        extracted = _embedded_identity(title, artist)
        if extracted:
            ex_artist, ex_track = extracted
            log.info(
                "discovery: 0 candidates for %s — %s; retrying as %s — %s",
                artist, title, ex_artist, ex_track,
            )
            try:
                candidates = _search_and_parse(
                    ex_track, ex_artist, max_candidates, search_fn, fetch_fn
                )
            except SearchError:
                pass  # fall through to the primary outcome below
        if not candidates and primary_error is not None:
            raise primary_error

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


def _embedded_identity(title: str, artist: str) -> tuple[str, str] | None:
    """(artist, track) recovered from the title itself: a quoted song name
    ('Artist "Track" at ...') or an 'Artist - Track' separator. None when the
    title carries no cleaner identity than the literal request."""
    extracted = parse_quoted_track(title) or parse_dash_title(title)
    if extracted and (extracted[0], extracted[1]) != (artist, title):
        return extracted
    return None


def _search_and_parse(
    title: str,
    artist: str,
    max_candidates: int,
    search_fn: SearchFn,
    fetch_fn: FetchFn,
) -> list[CandidateSource]:
    query = f"{_phrase(title)} {_phrase(artist)} chords"
    # ask for more hits than we need: many pages won't parse into a sheet
    hits = search_fn(query, max_candidates * 3)
    log.info("discovery: %d search hits for %s — %s", len(hits), artist, title)

    candidates: list[CandidateSource] = []
    for n, hit in enumerate(hits, start=1):
        if len(candidates) >= max_candidates:
            break
        try:
            page = fetch_fn(hit.url)
        except Exception as e:  # noqa: BLE001 — a dead page never kills discovery
            log.info("discovery: fetch failed for %s: %s", hit.url, e)
            continue
        text = extract_sheet_text(page)
        cand = candidate_from_text(text, source_id=f"web-{n}", url=hit.url, title=hit.title)
        if cand is not None:
            candidates.append(cand)
    return candidates
