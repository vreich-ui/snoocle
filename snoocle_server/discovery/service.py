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
_ORIGINAL_FETCH_PAGE = fetch_page


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


def candidate_from_text(
    text: str,
    source_id: str,
    url: str | None = None,
    title: str | None = None,
    *,
    retrieved_at: str | None = None,
) -> CandidateSource | None:
    sheet = parse_chord_sheet(text)
    if not sheet.is_plausible:
        return None
    notes = None
    if sheet.declared_capo:
        notes = (
            f"source declared capo {sheet.declared_capo}; chords transposed to "
            f"sounding pitch at ingestion (+{sheet.declared_capo} semitones)"
        )
    confidence = _confidence(sheet)
    full_song = (
        len(sheet.lines) >= 8
        and sheet.placement_count >= 8
        and (len(sheet.sections_hint) >= 2 or sheet.lyric_line_count >= 12)
    )
    return CandidateSource(
        sourceId=source_id,
        url=url,
        title=title,
        retrievedAt=(
            retrieved_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        ),
        declaredCapo=sheet.declared_capo,
        declaredKey=sheet.declared_key,
        confidence=confidence,
        parseConfidence=confidence,
        coverage="full-song" if full_song else "partial",
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
    *,
    with_report: bool = False,
    song_id: str | None = None,
    recording_variant: str | None = None,
    site_preferences: dict[str, list[str]] | None = None,
) -> list[CandidateSource]:
    """Compatibility wrapper returning deterministic prefetch candidates."""
    result = discover_sources_with_report(
        title, artist, max_candidates=max_candidates,
        search_fn=search_fn, fetch_fn=fetch_fn,
        song_id=song_id, recording_variant=recording_variant,
        site_preferences=site_preferences,
    )
    return result if with_report else result.candidates


def discover_sources_with_report(
    title: str,
    artist: str,
    max_candidates: int | None = None,
    search_fn: SearchFn | None = None,
    fetch_fn: FetchFn | None = None,
    *,
    song_id: str | None = None,
    recording_variant: str | None = None,
    site_preferences: dict[str, list[str]] | None = None,
):
    """Ranked multi-format gather plus its trace-ready decision report."""
    from ..schema.song import slugify_song_id
    from .prefetch import PrefetchResult, gather_chord_sheets, infer_recording_variant

    max_candidates = max_candidates or settings.source_prefetch_max
    search_fn = search_fn or (lambda q, n: web_search(q, n))
    if fetch_fn is None:
        # Production needs the multi-format resource fetcher (content type +
        # bytes for PDF/DOC/image). Keep the long-standing ``service.fetch_page``
        # monkeypatch seam for offline callers and tests that replaced it.
        if fetch_page is _ORIGINAL_FETCH_PAGE:
            from .resource import fetch_resource

            fetch_fn = fetch_resource
        else:
            fetch_fn = fetch_page
    song_id = song_id or slugify_song_id(artist, title)
    recording_variant = recording_variant or infer_recording_variant(title)

    # Video-derived identities ('Amy Winehouse - Back To Black' with the channel
    # name as the artist, or 'Artist "Track" at some show') rarely match any
    # chord sheet literally. When the literal identity finds nothing — INCLUDING
    # when its over-specific query makes every search backend return zero hits —
    # retry with the cleaner identity embedded in the title itself.
    primary_error: SearchError | None = None
    try:
        gathered = gather_chord_sheets(
            song_id, artist, title, recording_variant,
            max_sheets=max_candidates, search_fn=search_fn, fetch_fn=fetch_fn,
            site_preferences=site_preferences,
        )
    except SearchError as e:
        primary_error = e
        gathered = PrefetchResult(candidates=[], report={"error": str(e)})

    if not gathered.candidates:
        extracted = _embedded_identity(title, artist)
        if extracted:
            ex_artist, ex_track = extracted
            log.info(
                "discovery: 0 candidates for %s — %s; retrying as %s — %s",
                artist, title, ex_artist, ex_track,
            )
            try:
                fallback = gather_chord_sheets(
                    slugify_song_id(ex_artist, ex_track), ex_artist, ex_track,
                    infer_recording_variant(ex_track), max_sheets=max_candidates,
                    search_fn=search_fn, fetch_fn=fetch_fn,
                    site_preferences=site_preferences,
                )
                gathered = PrefetchResult(
                    candidates=fallback.candidates,
                    report={
                        **fallback.report,
                        "fallbackFrom": {"artist": artist, "title": title},
                        "attempts": [gathered.report, fallback.report],
                    },
                )
            except SearchError:
                pass  # fall through to the primary outcome below

    if not gathered.candidates and primary_error is not None:
        raise primary_error
    return gathered


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
