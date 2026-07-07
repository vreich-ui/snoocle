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
from datetime import datetime, timezone
from typing import Callable

from .chordsheet import parse_chord_sheet
from .fetch import extract_sheet_text, fetch_page
from .models import CandidateSource, SectionStart
from .search import SearchHit, web_search
from ..config import settings

log = logging.getLogger(__name__)

SearchFn = Callable[[str, int], list[SearchHit]]
FetchFn = Callable[[str], str]


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

    query = f'"{title}" "{artist}" chords'
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

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates
