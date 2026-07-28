"""Search backends: the failure has to be legible, and keys have to be used.

Every test here exists because of one incident. Discovery on Cloud Run reported
`all search backends failed: duckduckgo: 0 results` for every query. "0 results"
reads like a parser problem or an unlucky query, so that is where the
investigation went. It was neither: DuckDuckGo's HTML endpoint throttles
datacenter IPs and signals the refusal with HTTP 202 plus its homepage shell —
a success status carrying no results. The parser was fine.

Three things follow, and each is pinned below: the refusal must be reported as
a refusal; it must be retried, because it is a throttle and not a verdict (the
same query succeeds moments later — measured 2 successes in 8 single attempts);
and the retry must stay bounded.

The second half of the incident: SNOOCLE_BRAVE_API_KEY existed as a setting the
whole time, and setting it would have done nothing, because the backend order
defaulted to duckduckgo alone.
"""

from __future__ import annotations

import httpx
import pytest

from snoocle_server.config import settings
from snoocle_server.discovery import search as search_mod
from snoocle_server.discovery.search import SearchError, web_search

# What a live block actually looks like: 202, no error, no result anchors.
_BLOCKED_BODY = (
    '<!DOCTYPE html><html><head><title>DuckDuckGo</title>'
    '<link rel="stylesheet" href="/s.css"></head>'
    '<body class="body--html"><div id="links" class="results"></div></body></html>'
)

_REAL_BODY = (
    '<div class="result results_links"><h2 class="result__title">'
    '<a rel="nofollow" class="result__a" href="https://tabs.example/let-it-be">'
    "Let It Be <b>Chords</b></a></h2></div>"
)


def _response(status: int, body: str) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        text=body,
        request=httpx.Request("POST", "https://html.duckduckgo.com/html/"),
    )


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    """The retry backoff is real seconds in production and must be zero here."""
    monkeypatch.setattr(search_mod.time, "sleep", lambda _s: None)


def test_a_202_shell_is_reported_as_a_block_not_as_zero_results(monkeypatch):
    monkeypatch.setattr(
        search_mod.httpx, "post", lambda *a, **k: _response(202, _BLOCKED_BODY)
    )
    with pytest.raises(SearchError) as exc:
        search_mod._ddg_once("let it be beatles chords", 10)
    msg = str(exc.value)
    # The point of the test is the diagnosis, not the exception type.
    assert "datacenter" in msg, f"the message must name the actual cause: {msg}"
    assert "0 results" not in msg


def test_a_200_with_no_result_anchors_is_also_a_block(monkeypatch):
    """DDG has served the same empty shell under a 200. Status alone is not the
    tell — the absence of result anchors is."""
    monkeypatch.setattr(
        search_mod.httpx, "post", lambda *a, **k: _response(200, _BLOCKED_BODY)
    )
    with pytest.raises(SearchError):
        search_mod._ddg_once("let it be beatles chords", 10)


def test_a_real_result_page_still_parses(monkeypatch):
    """The block detector must not eat working responses."""
    monkeypatch.setattr(
        search_mod.httpx, "post", lambda *a, **k: _response(200, _REAL_BODY)
    )
    hits = search_mod._ddg_once("let it be beatles chords", 10)
    assert [h.url for h in hits] == ["https://tabs.example/let-it-be"]


def test_a_throttled_attempt_is_retried_rather_than_reported_as_failure(monkeypatch):
    """The whole point of the retry: a refusal is a throttle, not a verdict.

    Two refusals then a real page must produce hits, not an error — that is the
    exact sequence measured against the live endpoint.
    """
    responses = [
        _response(202, _BLOCKED_BODY),
        _response(200, _BLOCKED_BODY),
        _response(200, _REAL_BODY),
    ]
    calls = []

    def fake_post(*a, **k):  # noqa: ANN001
        calls.append(1)
        return responses[len(calls) - 1]

    monkeypatch.setattr(search_mod.httpx, "post", fake_post)
    hits = search_mod._search_duckduckgo("let it be beatles chords", 10)
    assert [h.url for h in hits] == ["https://tabs.example/let-it-be"]
    assert len(calls) == 3


def test_a_connection_reset_counts_as_a_throttle_too(monkeypatch):
    """Resets and 202-shells are the same throttle wearing different hats, so
    an httpx transport error must not skip the remaining attempts."""
    calls = []

    def fake_post(*a, **k):  # noqa: ANN001
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("[Errno 104] Connection reset by peer")
        return _response(200, _REAL_BODY)

    monkeypatch.setattr(search_mod.httpx, "post", fake_post)
    assert len(search_mod._search_duckduckgo("q", 10)) == 1
    assert len(calls) == 2


def test_retries_are_bounded_and_the_final_error_says_how_many(monkeypatch):
    """Backoff must not become an unbounded hang inside a request."""
    calls = []

    def fake_post(*a, **k):  # noqa: ANN001
        calls.append(1)
        return _response(202, _BLOCKED_BODY)

    monkeypatch.setattr(search_mod.httpx, "post", fake_post)
    with pytest.raises(SearchError) as exc:
        search_mod._search_duckduckgo("q", 10)
    assert len(calls) == search_mod._DDG_ATTEMPTS
    assert "attempts" in str(exc.value)
    assert len(search_mod._DDG_BACKOFF_SECONDS) == search_mod._DDG_ATTEMPTS - 1, (
        "one sleep per gap between attempts — a mismatch either hangs after the "
        "last attempt or silently drops a retry"
    )


def test_configuring_a_key_is_enough_to_make_that_backend_run():
    """A key that the default backend order never consults is not a feature."""
    default = type(settings).model_fields["search_backends"].default
    order = [b.strip() for b in default.split(",")]
    assert order[:2] == ["brave", "serpapi"], (
        "keyed backends must lead the default order, otherwise setting "
        f"SNOOCLE_BRAVE_API_KEY silently does nothing (order={order})"
    )
    assert "duckduckgo" in order, "the keyless fallback should stay available for local dev"


def test_unconfigured_keyed_backends_fall_through_cheaply(monkeypatch):
    """The cost of having brave+serpapi in the default order when no key is set
    must be zero network calls, not two failed round trips."""
    monkeypatch.setattr(settings, "brave_api_key", "")
    monkeypatch.setattr(settings, "serpapi_api_key", "")

    def no_network(*a, **k):  # noqa: ANN001
        raise AssertionError("an unconfigured backend must not touch the network")

    monkeypatch.setattr(search_mod.httpx, "get", no_network)
    monkeypatch.setattr(
        search_mod.httpx, "post", lambda *a, **k: _response(202, _BLOCKED_BODY)
    )

    with pytest.raises(SearchError) as exc:
        web_search("anything", backends="brave,serpapi,duckduckgo")
    msg = str(exc.value)
    assert "brave: no API key" in msg and "serpapi: no API key" in msg
    assert "datacenter" in msg, "the aggregate error must carry the real diagnosis"
