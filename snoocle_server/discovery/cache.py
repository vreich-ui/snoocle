"""Discovery result cache — the same web search, run once a month.

Unlike the MIR cache, this one NEEDS a TTL. The MIR key is a content hash of
the audio: everything that can change the answer is in the key, so an entry
can never go stale. Discovery's key is (title, artist, backends) and the web
behind it genuinely changes — new sheets get published, URLs rot, a site
reorganizes. There is no hash of "what the internet says today", so time is the
only invalidator available.

Sources change slowly, though, so the window is generous (~30 days by default,
``SNOOCLE_DISCOVERY_CACHE_TTL_DAYS``). Within it, re-analyzing a song reuses
the sheets rather than re-running a search that is rate-limited from a
datacenter IP anyway.

Keyed on the identity actually searched for, recording variant, backend set,
and ordered site preferences, because each can change what ranks into the top
N. ``max_candidates`` is in the key too: asking for more must not be answered
from a narrower run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from ..config import settings
from .models import CandidateSource

log = logging.getLogger(__name__)


def _backend_set() -> list[str]:
    """Every configured source channel, in stable order — the generic search
    backends plus the site-specific Ultimate Guitar channel when enabled."""
    backends = sorted(b.strip() for b in settings.search_backends.split(",") if b.strip())
    if settings.source_ug_enabled:
        backends.append("ultimate-guitar")
    return backends


def discovery_cache_key(
    title: str,
    artist: str,
    max_candidates: int,
    *,
    recording_variant: str | None = None,
    site_preferences: dict[str, list[str]] | None = None,
) -> str:
    components = {
        "v": 2,  # key-scheme version: bump to invalidate everything at once
        # Case/whitespace-insensitive: "let it be" and "Let It Be" search the
        # same web, and a stray space shouldn't cost a search.
        "title": " ".join(title.lower().split()),
        "artist": " ".join(artist.lower().split()),
        "backends": _backend_set(),
        "maxCandidates": int(max_candidates),
        "recordingVariant": recording_variant or "studio",
        # Ranking configuration changes WHICH top-N pages get parsed, so it is
        # part of the query result just as much as the backend roster is.
        "sitePreferences": site_preferences or {},
    }
    canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DiscoveryCacheInfo:
    """Cache provenance for one discovery run — outside the candidate list, for
    the same reason MirCacheInfo sits outside MirAnalysis: a hit must produce
    byte-identical candidates to a miss."""

    status: str  # "hit" | "miss" | "refresh" | "disabled"
    gathered_at: str  # ISO-8601 of the ORIGINAL search (== now for a miss)
    key: str = ""
    report: dict | None = None

    @property
    def from_cache(self) -> bool:
        return self.status == "hit"


_store = None
_lock = threading.Lock()


def get_discovery_cache():
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                from ..store.blob_cache import build_blob_cache

                _store = build_blob_cache(
                    settings.discovery_cache_collection,
                    ttl_seconds=settings.discovery_cache_ttl_days * 86400.0,
                )
                log.info("discovery cache backend: %s", type(_store).__name__)
    return _store


def reset_discovery_cache() -> None:
    global _store
    with _lock:
        _store = None


def discover_cached(
    title: str,
    artist: str,
    *,
    max_candidates: int,
    discover: Callable[[], list[CandidateSource]],
    refresh: bool = False,
    recording_variant: str | None = None,
    site_preferences: dict[str, list[str]] | None = None,
) -> tuple[list[CandidateSource], DiscoveryCacheInfo]:
    """Run ``discover`` unless an identical search is already cached.

    ``discover`` is injected for the same reason the MIR cache injects
    ``compute``: the caller keeps owning how the work is done, so existing
    seams keep working untouched.

    Never raises on cache trouble — a cache problem degrades to a live search.
    An EMPTY result is deliberately not cached: discovery returning nothing is
    usually a throttled backend rather than a fact about the song, and freezing
    that for a month would turn a transient failure into a lasting one.
    """
    def run_discover() -> tuple[list[CandidateSource], dict | None]:
        result = discover()
        if hasattr(result, "candidates") and hasattr(result, "report"):
            return list(result.candidates), dict(result.report)
        return result, None

    if not settings.discovery_cache_enabled:
        candidates, report = run_discover()
        return candidates, DiscoveryCacheInfo(
            status="disabled", gathered_at=_now_iso(), report=report
        )

    store = get_discovery_cache()
    key = discovery_cache_key(
        title,
        artist,
        max_candidates,
        recording_variant=recording_variant,
        site_preferences=site_preferences,
    )

    if not refresh:
        hit = store.get(key)
        if hit is not None:
            value, stored_at = hit
            try:
                candidates = [CandidateSource.model_validate(c) for c in value["candidates"]]
            except Exception as e:  # noqa: BLE001 — a bad entry is just a miss
                log.warning("discovery cache: unreadable entry key=%s (%s)", key, e)
            else:
                log.info(
                    "discovery cache hit key=%s candidates=%d gatheredAt=%s",
                    key, len(candidates), stored_at,
                )
                return candidates, DiscoveryCacheInfo(
                    status="hit", gathered_at=stored_at, key=key,
                    report=value.get("report"),
                )

    candidates, report = run_discover()
    if not candidates:
        return candidates, DiscoveryCacheInfo(
            status="refresh" if refresh else "miss", gathered_at=_now_iso(), key=key,
            report=report,
        )
    stored_at = store.put(
        key, {
            "candidates": [c.model_dump(mode="json") for c in candidates],
            "report": report,
        }
    ) or _now_iso()
    return candidates, DiscoveryCacheInfo(
        status="refresh" if refresh else "miss", gathered_at=stored_at, key=key,
        report=report,
    )


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
