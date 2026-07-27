"""Master plan B5: Ultimate Guitar discovery source.

Runs entirely against saved fixture JSON (no live network) — httpx.get is
monkeypatched to serve canned js-store-shaped HTML for the search page and
one tab page. See discovery/sources/ultimate_guitar.py's module docstring
for why the js-store JSON shape is defended with fallback key paths rather
than a single hard-coded contract.
"""

from __future__ import annotations

import html
import json

import pytest

from snoocle_server.config import settings
from snoocle_server.discovery.sources import ultimate_guitar as ug_mod
from snoocle_server.discovery.sources.ultimate_guitar import (
    _confidence_prior,
    _ug_markup_to_bracket,
    discover_ultimate_guitar,
)


def _js_store_html(data: dict) -> str:
    blob = html.escape(json.dumps(data), quote=True)
    return f'<html><body><div class="js-store" data-content="{blob}"></div></body></html>'


_SEARCH_JSON = {
    "store": {
        "page": {
            "data": {
                "results": [
                    {
                        "id": 2296,
                        "song_name": "Let It Be",
                        "artist_name": "The Beatles",
                        "tab_url": "https://tabs.ultimate-guitar.com/tab/the-beatles/let-it-be-chords-2296",
                        "rating": 4.8,
                        "votes": 29000,
                        "type": "Chords",
                    },
                    {
                        "id": 9999,
                        "song_name": "Let It Be (Live)",
                        "artist_name": "The Beatles",
                        "tab_url": "https://tabs.ultimate-guitar.com/tab/the-beatles/let-it-be-tab-9999",
                        "rating": 3.0,
                        "votes": 5,
                        "type": "Tab",  # not a Chords result -- must be filtered
                    },
                ]
            }
        }
    }
}


def _tab_json(capo: int = 0) -> dict:
    return {
        "store": {
            "page": {
                "data": {
                    "tab": {"song_name": "Let It Be", "artist_name": "The Beatles", "rating": 4.8, "votes": 29000},
                    "tab_view": {
                        "wiki_tab": {
                            "content": (
                                "[Verse 1]\n"
                                "[ch]C[/ch]When I find my[ch]G[/ch]self in times of [ch]Am[/ch]trouble\n"
                                "[ch]F[/ch]Mother Mary comes to me\n"
                                "[ch]C[/ch]Speaking words of [ch]G[/ch]wisdom\n"
                                "[ch]Am[/ch]Let it [ch]F[/ch]be\n"
                                "[tab]e|---0---3---|[/tab]"
                            )
                        },
                        "meta": {"tuning": {"name": "Standard"}, "capo": capo},
                    },
                }
            }
        }
    }


class _FakeResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def _ug_enabled(monkeypatch):
    monkeypatch.setattr(settings, "source_ug_enabled", True)


def test_disabled_by_default_makes_no_calls(monkeypatch):
    monkeypatch.setattr(settings, "source_ug_enabled", False)

    def boom(*a, **k):  # noqa: ANN001
        raise AssertionError("must not call the network when disabled")

    monkeypatch.setattr(ug_mod.httpx, "get", boom)
    assert discover_ultimate_guitar("Let It Be", "The Beatles") == []


def test_discover_end_to_end_against_fixtures(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        if url == ug_mod._SEARCH_URL:
            return _FakeResponse(200, _js_store_html(_SEARCH_JSON))
        assert url == "https://tabs.ultimate-guitar.com/tab/the-beatles/let-it-be-chords-2296"
        return _FakeResponse(200, _js_store_html(_tab_json()))

    monkeypatch.setattr(ug_mod.httpx, "get", fake_get)
    candidates = discover_ultimate_guitar("Let It Be", "The Beatles")

    assert len(candidates) == 1  # the Tab-type hit was filtered out
    c = candidates[0]
    assert c.sourceId == "ultimate-guitar-2296"
    assert c.title == "The Beatles - Let It Be"
    assert c.declaredCapo == 0
    assert c.sectionsHint == ["Verse 1"]
    assert [p.chord for p in c.lines[0].chordPlacements] == ["C", "G", "Am"]
    assert [p.chord for p in c.lines[1].chordPlacements] == ["F"]
    assert c.confidence == _confidence_prior(4.8, 29000)
    assert "rating=4.8" in c.notes
    assert "votes=29000" in c.notes


def test_metadata_capo_retransposes_when_sheet_text_has_no_capo_line(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        if url == ug_mod._SEARCH_URL:
            return _FakeResponse(200, _js_store_html(_SEARCH_JSON))
        return _FakeResponse(200, _js_store_html(_tab_json(capo=5)))

    monkeypatch.setattr(ug_mod.httpx, "get", fake_get)
    candidates = discover_ultimate_guitar("Let It Be", "The Beatles")

    assert len(candidates) == 1
    c = candidates[0]
    assert c.declaredCapo == 5
    # Am shape + capo 5 -> Dm sounding (5 semitones up: A -> D)
    assert [p.chord for p in c.lines[0].chordPlacements] == ["F", "C", "Dm"]
    assert [p.chord for p in c.lines[1].chordPlacements] == ["A#"]
    assert "source declared capo 5" in c.notes


def test_search_network_error_returns_empty(monkeypatch):
    def boom(*a, **k):  # noqa: ANN001
        raise RuntimeError("connection refused")

    monkeypatch.setattr(ug_mod.httpx, "get", boom)
    assert discover_ultimate_guitar("Let It Be", "The Beatles") == []


def test_tab_fetch_failure_is_skipped_not_fatal(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        if url == ug_mod._SEARCH_URL:
            return _FakeResponse(200, _js_store_html(_SEARCH_JSON))
        return _FakeResponse(404, "not found")

    monkeypatch.setattr(ug_mod.httpx, "get", fake_get)
    assert discover_ultimate_guitar("Let It Be", "The Beatles") == []


def test_missing_js_store_returns_empty(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        return _FakeResponse(200, "<html><body>no js-store here</body></html>")

    monkeypatch.setattr(ug_mod.httpx, "get", fake_get)
    assert discover_ultimate_guitar("Let It Be", "The Beatles") == []


def test_ug_markup_to_bracket_converts_ch_tags_and_strips_tab_blocks():
    text = "[ch]Am[/ch]Hello[tab]e|--0--|[/tab] world[ch]G[/ch]"
    assert _ug_markup_to_bracket(text) == "[Am]Hello world[G]"


def test_confidence_prior_rewards_rating_and_votes():
    low = _confidence_prior(rating=None, votes=None)
    mid = _confidence_prior(rating=3.0, votes=10)
    high = _confidence_prior(rating=5.0, votes=50000)
    assert low < mid < high
    assert high <= 0.95


def test_confidence_prior_capped_at_0_95():
    assert _confidence_prior(rating=5.0, votes=10_000_000) == 0.95


def test_discover_sources_merges_in_ug_candidates_when_enabled(monkeypatch):
    """discover_sources (the generic web-search orchestrator) must fold UG
    candidates in additively -- not as a fallback only tried when the
    generic web search comes up empty."""
    from snoocle_server.discovery.service import discover_sources

    def fake_get(url, params=None, headers=None, timeout=None):
        if url == ug_mod._SEARCH_URL:
            return _FakeResponse(200, _js_store_html(_SEARCH_JSON))
        return _FakeResponse(200, _js_store_html(_tab_json()))

    monkeypatch.setattr(ug_mod.httpx, "get", fake_get)
    candidates = discover_sources(
        "Let It Be",
        "The Beatles",
        search_fn=lambda q, n: [],  # generic web search finds nothing
        fetch_fn=lambda url: "",
    )
    assert any(c.sourceId == "ultimate-guitar-2296" for c in candidates)


def test_discover_sources_ignores_ug_when_disabled(monkeypatch):
    from snoocle_server.discovery.service import discover_sources

    monkeypatch.setattr(settings, "source_ug_enabled", False)

    def boom(*a, **k):  # noqa: ANN001
        raise AssertionError("must not call the network when UG is disabled")

    monkeypatch.setattr(ug_mod.httpx, "get", boom)
    candidates = discover_sources(
        "Let It Be", "The Beatles", search_fn=lambda q, n: [], fetch_fn=lambda url: ""
    )
    assert candidates == []
