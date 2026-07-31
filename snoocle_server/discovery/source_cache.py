"""Parsed chord sheets cached by URL and immutable content hash."""

from __future__ import annotations

import hashlib
import threading

from ..config import settings
from .models import CandidateSource

_store = None
_lock = threading.Lock()


def get_source_cache():
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                from ..store.blob_cache import build_blob_cache

                _store = build_blob_cache(
                    settings.sources_cache_collection,
                    ttl_seconds=settings.sources_cache_ttl_days * 86400.0,
                )
    return _store


def reset_source_cache() -> None:
    global _store
    with _lock:
        _store = None


def _url_key(url: str) -> str:
    return "url-" + hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def _content_key(url: str, content_hash: str) -> str:
    raw = f"{url.strip()}\n{content_hash}".encode("utf-8")
    return "content-" + hashlib.sha256(raw).hexdigest()


def lookup_url(url: str, store=None) -> tuple[CandidateSource, dict] | None:
    store = store or get_source_cache()
    pointer = store.get(_url_key(url))
    if pointer is None:
        return None
    value, gathered_at = pointer
    try:
        candidate = CandidateSource.model_validate(value["candidate"])
    except Exception:
        return None
    return candidate, {
        "status": "hit",
        "gatheredAt": value.get("gatheredAt") or gathered_at,
        "contentHash": value.get("contentHash"),
        "contentType": value.get("contentType"),
    }


def lookup_content(url: str, content_hash: str, store=None) -> tuple[CandidateSource, dict] | None:
    store = store or get_source_cache()
    hit = store.get(_content_key(url, content_hash))
    if hit is None:
        return None
    value, gathered_at = hit
    try:
        candidate = CandidateSource.model_validate(value["candidate"])
    except Exception:
        return None
    return candidate, {
        "status": "content-hit",
        "gatheredAt": value.get("gatheredAt") or gathered_at,
        "contentHash": content_hash,
        "contentType": value.get("contentType"),
    }


def store_parsed(url: str, content_hash: str, content_type: str,
                 candidate: CandidateSource, store=None) -> str:
    store = store or get_source_cache()
    payload = {
        "candidate": candidate.model_dump(mode="json"),
        "contentHash": content_hash,
        "contentType": content_type,
    }
    gathered_at = store.put(_content_key(url, content_hash), payload) or candidate.retrievedAt or ""
    payload["gatheredAt"] = gathered_at
    store.put(_url_key(url), payload)
    return gathered_at
