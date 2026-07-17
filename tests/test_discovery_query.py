"""Discovery builds a well-formed exact-phrase query even when the title or
artist themselves contain quotes (typical for video-derived identities), and
falls back to the quoted-track identity when the literal one finds nothing."""

from __future__ import annotations

from pathlib import Path

from snoocle_server.discovery.search import SearchHit
from snoocle_server.discovery.service import discover_sources

FIXTURES = Path(__file__).parent / "fixtures"


def test_query_sanitizes_embedded_quotes():
    queries: list[str] = []

    def search_fn(query: str, n: int) -> list:
        queries.append(query)
        return []

    discover_sources(
        'Blues Traveler "Hook" at Howard Stern\'s 1996 Birthday Show',
        "The Howard Stern Show",
        search_fn=search_fn,
    )
    query = queries[0]
    # exactly two quoted phrases (title, artist) — nothing nested or dangling
    assert query.count('"') == 4
    assert query == (
        '"Blues Traveler Hook at Howard Stern\'s 1996 Birthday Show" '
        '"The Howard Stern Show" chords'
    )


def test_query_unchanged_for_plain_identity():
    captured: dict = {}

    def search_fn(query: str, n: int) -> list:
        captured["query"] = query
        return []

    discover_sources("Hook", "Blues Traveler", search_fn=search_fn)
    assert captured["query"] == '"Hook" "Blues Traveler" chords'


def test_falls_back_to_quoted_track_identity_when_literal_finds_nothing():
    """A video-derived identity ('Artist "Track" at some show' / uploader) has
    no literal chord-sheet match; the retry extracts the real song identity and
    its results flow through to candidates."""
    queries: list[str] = []
    sheet = (FIXTURES / "sheet_over_lyrics.txt").read_text()

    def search_fn(query: str, n: int) -> list[SearchHit]:
        queries.append(query)
        if len(queries) == 1:
            return []  # nothing for the literal video title
        return [SearchHit(url="https://example.com/hook-chords", title="Hook chords")]

    cands = discover_sources(
        'Blues Traveler "Hook" at Howard Stern\'s 1996 Birthday Show',
        "The Howard Stern Show",
        search_fn=search_fn,
        fetch_fn=lambda url: sheet,
    )
    assert queries == [
        '"Blues Traveler Hook at Howard Stern\'s 1996 Birthday Show" '
        '"The Howard Stern Show" chords',
        '"Hook" "Blues Traveler" chords',
    ]
    assert len(cands) == 1
    assert cands[0].url == "https://example.com/hook-chords"


def test_no_fallback_for_plain_identity_with_no_results():
    """No quoted song name in the title -> a single search, no second guess."""
    queries: list[str] = []

    def search_fn(query: str, n: int) -> list:
        queries.append(query)
        return []

    assert discover_sources("Hook", "Blues Traveler", search_fn=search_fn) == []
    assert len(queries) == 1
