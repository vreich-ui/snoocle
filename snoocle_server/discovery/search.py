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
"""

from __future__ import annotations

import html as html_mod
import logging
import re
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


def _search_duckduckgo(query: str, max_results: int) -> list[SearchHit]:
    r = httpx.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": _UA},
        timeout=settings.fetch_timeout_seconds,
        follow_redirects=True,
    )
    r.raise_for_status()
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


_BACKENDS = {
    "brave": _search_brave,
    "serpapi": _search_serpapi,
    "duckduckgo": _search_duckduckgo,
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
