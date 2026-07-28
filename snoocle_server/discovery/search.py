"""Pluggable web-search backends for text-source discovery.

GUARDRAIL: this is general web search — the service never hardcodes a
scraper against a single named chord/lyric site. Queries are generic
("<title>" "<artist>" chords) and whatever the web returns is parsed by the
generic chord-sheet parser.

Backends (ordered preference via SNOOCLE_SEARCH_BACKENDS):
- brave     : Brave Search API (needs SNOOCLE_BRAVE_API_KEY)
- serpapi   : SerpAPI (needs SNOOCLE_SERPAPI_API_KEY)
- duckduckgo: DuckDuckGo HTML endpoint (no key; least robust)

A backend that errors or is unconfigured falls through to the next one.

Operational note (2026-07): duckduckgo throttles datacenter IPs hard —
measured at roughly 2 successes in 8 attempts from Cloud Run's address range.
It does not report the refusal as an error status; it answers HTTP 202 carrying
its homepage shell, so the parser finds no results and the failure surfaces as
"0 results", indistinguishable from a query that genuinely matched nothing.
Hence two things here: _ddg_once inspects the body and raises a named error,
and _search_duckduckgo retries with backoff (the refusal is a throttle, not a
verdict — the same query succeeds moments later).

Even with retries this is not a backend to depend on in production. Set a
Brave or SerpAPI key for that. The Ultimate Guitar source (SNOOCLE_SOURCE_UG)
defaults on precisely because it is the only keyless channel that answers
datacenter IPs reliably.
"""

from __future__ import annotations

import html as html_mod
import logging
import re
import time
import urllib.parse
from dataclasses import dataclass

import httpx

from ..config import settings

log = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"


@dataclass
class SearchHit:
    url: str
    title: str
    snippet: str = ""


class SearchError(RuntimeError):
    pass


def _search_brave(query: str, max_results: int) -> list[SearchHit]:
    if not settings.brave_api_key:
        raise SearchError("brave: no API key configured")
    r = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": max_results},
        headers={"X-Subscription-Token": settings.brave_api_key, "Accept": "application/json"},
        timeout=settings.fetch_timeout_seconds,
    )
    r.raise_for_status()
    results = (r.json().get("web") or {}).get("results") or []
    return [
        SearchHit(url=x["url"], title=x.get("title", ""), snippet=x.get("description", ""))
        for x in results[:max_results]
    ]


def _search_serpapi(query: str, max_results: int) -> list[SearchHit]:
    if not settings.serpapi_api_key:
        raise SearchError("serpapi: no API key configured")
    r = httpx.get(
        "https://serpapi.com/search.json",
        params={"q": query, "num": max_results, "api_key": settings.serpapi_api_key},
        timeout=settings.fetch_timeout_seconds,
    )
    r.raise_for_status()
    results = r.json().get("organic_results") or []
    return [
        SearchHit(url=x["link"], title=x.get("title", ""), snippet=x.get("snippet", ""))
        for x in results[:max_results]
    ]


_DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL,
)


_DDG_ATTEMPTS = 4
_DDG_BACKOFF_SECONDS = (1.5, 3.0, 6.0)


def _ddg_once(query: str, max_results: int) -> list[SearchHit]:
    r = httpx.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": _UA},
        timeout=settings.fetch_timeout_seconds,
        follow_redirects=True,
    )
    r.raise_for_status()
    # A refusal arrives as 202 (sometimes 200) carrying DDG's homepage shell,
    # NOT as an error status — so raise_for_status() sails through, the parser
    # finds nothing, and the caller reports "0 results", which is
    # indistinguishable from a query that genuinely matched nothing. That cost
    # real debugging time: the "0 results" sent the investigation after the
    # parser when the endpoint had simply declined to answer. The absence of
    # result anchors is the tell; name it for what it is.
    if r.status_code == 202 or 'class="result__a"' not in r.text:
        raise SearchError(
            f"duckduckgo: the HTML endpoint refused this request (HTTP "
            f"{r.status_code}, no results in the body). It heavily rate-limits "
            "datacenter IPs, so this backend is unreliable from Cloud Run — "
            "configure SNOOCLE_BRAVE_API_KEY or SNOOCLE_SERPAPI_API_KEY for "
            "dependable general search; the Ultimate Guitar source "
            "(SNOOCLE_SOURCE_UG, on by default) covers the common case."
        )
    hits: list[SearchHit] = []
    for m in _DDG_RESULT_RE.finditer(r.text):
        href = html_mod.unescape(m.group("href"))
        # DDG wraps results in a redirect: //duckduckgo.com/l/?uddg=<url>&...
        if "uddg=" in href:
            q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            href = q.get("uddg", [href])[0]
        title = re.sub(r"<[^>]+>", "", m.group("title")).strip()
        hits.append(SearchHit(url=href, title=html_mod.unescape(title)))
        if len(hits) >= max_results:
            break
    return hits


def _search_duckduckgo(query: str, max_results: int) -> list[SearchHit]:
    """DDG with backoff, because the refusal is a throttle and not a verdict.

    Measured from a datacenter IP: 2 of 8 single attempts came back with
    results, the rest were the 202 shell or a connection reset. Those are not
    the same failure as "this query has no answer" — the identical query
    succeeds moments later. One shot per query therefore threw away most of the
    backend's actual capacity, and every discard landed on the caller as
    "0 results".

    Four attempts at ~1.5/3/6s took that 2-in-8 per-attempt rate to 4-in-6
    end-to-end when measured the same way, at a worst case of ~10s of sleeping
    on the failure path — paid only when the alternative was returning nothing.
    Connection resets are retried on the same footing; they are the same
    throttle wearing a different hat.
    """
    last: Exception | None = None
    for attempt in range(_DDG_ATTEMPTS):
        try:
            return _ddg_once(query, max_results)
        except (SearchError, httpx.HTTPError) as e:
            last = e
            if attempt < len(_DDG_BACKOFF_SECONDS):
                log.info(
                    "duckduckgo attempt %d/%d refused (%s); retrying in %.1fs",
                    attempt + 1, _DDG_ATTEMPTS, e, _DDG_BACKOFF_SECONDS[attempt],
                )
                time.sleep(_DDG_BACKOFF_SECONDS[attempt])
    assert last is not None
    raise SearchError(f"{last} (after {_DDG_ATTEMPTS} attempts)") from last


def _search_static(query: str, max_results: int) -> list[SearchHit]:
    """Fixture backend for offline development/e2e testing: hits come from the
    SNOOCLE_STATIC_SEARCH_HITS env var (JSON list of {url, title})."""
    import json
    import os

    raw = os.environ.get("SNOOCLE_STATIC_SEARCH_HITS", "")
    if not raw:
        raise SearchError("static: SNOOCLE_STATIC_SEARCH_HITS not set")
    hits = [SearchHit(url=h["url"], title=h.get("title", "")) for h in json.loads(raw)]
    return hits[:max_results]


_BACKENDS = {
    "brave": _search_brave,
    "serpapi": _search_serpapi,
    "duckduckgo": _search_duckduckgo,
    "static": _search_static,
}


def web_search(query: str, max_results: int = 10, backends: str | None = None) -> list[SearchHit]:
    """Run `query` against the first working configured backend."""
    order = [b.strip() for b in (backends or settings.search_backends).split(",") if b.strip()]
    errors: list[str] = []
    for name in order:
        fn = _BACKENDS.get(name)
        if fn is None:
            errors.append(f"{name}: unknown backend")
            continue
        try:
            hits = fn(query, max_results)
            if hits:
                return hits
            errors.append(f"{name}: 0 results")
        except Exception as e:  # noqa: BLE001 — fall through to next backend
            errors.append(f"{name}: {e}")
            log.warning("search backend %s failed: %s", name, e)
    raise SearchError("all search backends failed: " + "; ".join(errors))
