"""Deterministic, ranked chord-sheet gathering before reconciliation."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.parse
from dataclasses import dataclass
from typing import Callable

from ..config import settings
from .models import CandidateSource
from .resource import FetchedResource, extract_resource_text, fetch_resource, normalize_resource
from .search import SearchError, SearchHit, web_search
from .service import candidate_from_text
from .source_cache import get_source_cache, lookup_url, store_parsed
from .sources.ultimate_guitar import (
    candidate_from_ug_tab,
    fetch_tab,
    search_ultimate_guitar,
)

log = logging.getLogger(__name__)

DEFAULT_SITE_PREFERENCES = {
    "global": ["ultimate-guitar.com", "chordify.net", "songsterr.com"],
    "russian": ["amdm.ru", "akkords.pro", "mychords.net", "5lad.net"],
    "hebrew": [],
}

_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_HEBREW_RE = re.compile(r"[\u0590-\u05ff]")
_LIVE_RE = re.compile(r"\b(live|concert|unplugged|acoustic)\b", re.IGNORECASE)
_COVER_RE = re.compile(r"\bcover\b", re.IGNORECASE)
_QUOTE_RE = re.compile(r'["“”„‟]+')
_UG_TAB_ID_RE = re.compile(r"-(\d+)(?:[/?#]|$)")


@dataclass(frozen=True)
class RankedHit:
    hit: SearchHit
    site: str
    official: bool
    preference_rank: int
    original_rank: int
    ug_data: dict | None = None

    def to_dict(self, rank: int) -> dict:
        return {
            "rank": rank,
            "url": self.hit.url,
            "title": self.hit.title,
            "site": self.site,
            "official": self.official,
            "preferenceRank": self.preference_rank,
        }


@dataclass(frozen=True)
class PrefetchResult:
    candidates: list[CandidateSource]
    report: dict


def infer_locale(title: str, artist: str) -> str:
    value = f"{artist} {title}"
    if _CYRILLIC_RE.search(value):
        return "russian"
    if _HEBREW_RE.search(value):
        return "hebrew"
    return "global"


def infer_recording_variant(title: str, *, is_cover: bool = False) -> str:
    if is_cover or _COVER_RE.search(title or ""):
        return "cover"
    if _LIVE_RE.search(title or ""):
        return "live"
    return "studio"


def effective_site_preferences(overrides: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    result = {name: list(domains) for name, domains in DEFAULT_SITE_PREFERENCES.items()}
    if overrides:
        for locale, domains in overrides.items():
            result[str(locale)] = [str(domain).strip().casefold() for domain in domains if str(domain).strip()]
    return result


def _site(url: str) -> str:
    return (urllib.parse.urlparse(url).hostname or "").casefold().removeprefix("www.")


def _matches_site(site: str, preference: str) -> bool:
    preference = preference.casefold().removeprefix("www.")
    return site == preference or site.endswith("." + preference)


def _is_official(hit: SearchHit, ug_data: dict | None = None) -> bool:
    if ug_data and str(ug_data.get("type", "")).casefold() == "official":
        return True
    haystack = f"{hit.title} {hit.snippet} {hit.url}".casefold()
    return "official" in haystack


def rank_search_hits(
    hits: list[tuple[SearchHit, dict | None]] | list[SearchHit],
    *,
    locale: str,
    site_preferences: dict[str, list[str]] | None = None,
) -> list[RankedHit]:
    preferences = effective_site_preferences(site_preferences)
    ordered_sites = list(preferences.get("global", []))
    if locale != "global":
        ordered_sites.extend(preferences.get(locale, []))

    ranked: list[RankedHit] = []
    for original_rank, item in enumerate(hits):
        hit, ug_data = item if isinstance(item, tuple) else (item, None)
        site = _site(hit.url)
        official = _is_official(hit, ug_data)
        # UG official is an explicit tier ahead of every other source. A
        # non-official UG result takes UG's normal configured position.
        if _matches_site(site, "ultimate-guitar.com") and official:
            preference_rank = -1
        else:
            preference_rank = next(
                (i for i, domain in enumerate(ordered_sites) if _matches_site(site, domain)),
                len(ordered_sites) + 100,
            )
        ranked.append(RankedHit(
            hit=hit,
            site=site,
            official=official,
            preference_rank=preference_rank,
            original_rank=original_rank,
            ug_data=ug_data,
        ))
    ranked.sort(key=lambda item: (item.preference_rank, item.original_rank))
    return ranked


def _query(title: str, artist: str, variant: str) -> str:
    def phrase(value: str) -> str:
        return '"' + " ".join(_QUOTE_RE.sub(" ", value).split()) + '"'

    suffix = f" {variant}" if variant and variant != "studio" else ""
    return f"{phrase(title)} {phrase(artist)} chords{suffix}"


def _stable_source_id(url: str) -> str:
    return "sheet-" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]


def _ug_source_id(url: str) -> str:
    match = _UG_TAB_ID_RE.search(url)
    return f"ultimate-guitar-{match.group(1)}" if match else _stable_source_id(url)


def _candidate_with_retrieval(
    candidate: CandidateSource,
    *,
    content_hash: str,
    content_type: str,
    cache_status: str,
    gathered_at: str | None = None,
) -> CandidateSource:
    return candidate.model_copy(update={
        "contentHash": content_hash,
        "contentType": content_type,
        "cacheStatus": cache_status,
        "retrievedAt": gathered_at or candidate.retrievedAt,
        "parseConfidence": (
            candidate.parseConfidence
            if candidate.parseConfidence is not None
            else candidate.confidence
        ),
    })


def gather_chord_sheets(
    song_id: str,
    artist: str,
    title: str,
    recording_variant: str = "studio",
    *,
    query: str | None = None,
    max_sheets: int | None = None,
    search_fn: Callable[[str, int], list[SearchHit]] | None = None,
    fetch_fn: Callable[[str], object] | None = None,
    ug_search_fn: Callable[[str, str, int], list[dict]] | None = None,
    ug_fetch_fn: Callable[[str], dict | None] | None = None,
    cache_store=None,
    site_preferences: dict[str, list[str]] | None = None,
) -> PrefetchResult:
    """Search, rank, fetch, parse, capo-normalize, and cache top-N sheets."""
    max_sheets = max_sheets if max_sheets is not None else settings.source_prefetch_max
    search_fn = search_fn or (lambda value, limit: web_search(value, limit))
    fetch_fn = fetch_fn or fetch_resource
    ug_search_fn = ug_search_fn or (
        lambda song, performer, limit: search_ultimate_guitar(song, performer, limit)
    )
    ug_fetch_fn = ug_fetch_fn or fetch_tab
    cache_store = cache_store or get_source_cache()
    locale = infer_locale(title, artist)
    fixed_query = query or _query(title, artist, recording_variant)

    search_error: SearchError | None = None
    try:
        web_hits = search_fn(fixed_query, max(max_sheets * 4, 10))
    except SearchError as error:
        search_error = error
        web_hits = []
    combined: list[tuple[SearchHit, dict | None]] = [(hit, None) for hit in web_hits]
    if settings.source_ug_enabled:
        for hit in ug_search_fn(title, artist, max(max_sheets * 3, 6)):
            url = str(hit.get("tab_url") or "")
            if url:
                combined.append((SearchHit(
                    url=url,
                    title=f"{hit.get('artist_name') or artist} - {hit.get('song_name') or title} "
                          f"{hit.get('type') or ''}".strip(),
                ), hit))

    # Same URL can arrive from general web search and UG's dedicated index.
    # Keep the dedicated metadata so official/type/rating information survives.
    deduped: dict[str, tuple[SearchHit, dict | None]] = {}
    for item in combined:
        prior = deduped.get(item[0].url)
        if prior is None or (prior[1] is None and item[1] is not None):
            deduped[item[0].url] = item
    ranked = rank_search_hits(
        list(deduped.values()), locale=locale, site_preferences=site_preferences
    )
    chosen = ranked[:max_sheets]
    outcomes: list[dict] = []
    candidates: list[CandidateSource] = []

    for rank, item in enumerate(chosen, start=1):
        url = item.hit.url
        cached = lookup_url(url, cache_store)
        if cached is not None:
            candidate, meta = cached
            candidate = _candidate_with_retrieval(
                candidate,
                content_hash=str(meta.get("contentHash") or candidate.contentHash or ""),
                content_type=str(meta.get("contentType") or candidate.contentType or ""),
                cache_status="hit",
                gathered_at=meta.get("gatheredAt"),
            )
            candidates.append(candidate)
            outcomes.append({
                "rank": rank, "url": url, "status": "cache-hit",
                "contentHash": candidate.contentHash,
                "contentType": candidate.contentType,
                "parseConfidence": candidate.parseConfidence,
            })
            continue

        try:
            if item.ug_data is not None or _matches_site(item.site, "ultimate-guitar.com"):
                tab = ug_fetch_fn(url)
                if tab is None:
                    raise RuntimeError("Ultimate Guitar page did not contain a chord sheet")
                canonical = json.dumps(tab, sort_keys=True, ensure_ascii=False).encode("utf-8")
                content_hash = hashlib.sha256(canonical).hexdigest()
                content_type = "application/x-ultimate-guitar"
                ug_data = item.ug_data or {
                    "tab_url": url,
                    "song_name": title,
                    "artist_name": artist,
                    "type": "Official" if item.official else "Chords",
                }
                candidate = candidate_from_ug_tab(
                    ug_data, tab, title=title, artist=artist,
                    source_id=_ug_source_id(url),
                )
            else:
                resource: FetchedResource = normalize_resource(fetch_fn(url), url=url)
                content_hash = resource.content_hash
                content_type = resource.content_type
                text = extract_resource_text(resource)
                candidate = candidate_from_text(
                    text, source_id=_stable_source_id(url), url=url, title=item.hit.title
                )
            if candidate is None:
                raise ValueError("content is not a plausible chord sheet")
            candidate = _candidate_with_retrieval(
                candidate,
                content_hash=content_hash,
                content_type=content_type,
                cache_status="miss",
            )
            gathered_at = store_parsed(
                url, content_hash, content_type, candidate, cache_store
            )
            candidate = candidate.model_copy(update={"retrievedAt": gathered_at})
            candidates.append(candidate)
            outcomes.append({
                "rank": rank, "url": url, "status": "parsed",
                "contentHash": content_hash, "contentType": content_type,
                "parseConfidence": candidate.parseConfidence,
                "coverage": candidate.coverage,
            })
        except Exception as error:  # noqa: BLE001 — one source never kills the gather
            outcomes.append({
                "rank": rank, "url": url, "status": "failed", "error": str(error)[:500]
            })
            log.info("prefetch failed rank=%d url=%s: %s", rank, url, error)

    report = {
        "songId": song_id,
        "artist": artist,
        "title": title,
        "recordingVariant": recording_variant,
        "locale": locale,
        "query": fixed_query,
        "rankedResults": [item.to_dict(rank) for rank, item in enumerate(ranked, start=1)],
        "chosenUrls": [item.hit.url for item in chosen],
        "outcomes": outcomes,
        "parsedSheets": len(candidates),
        **({"searchError": str(search_error)} if search_error is not None else {}),
    }
    if not candidates and search_error is not None:
        raise search_error
    return PrefetchResult(candidates=candidates, report=report)
