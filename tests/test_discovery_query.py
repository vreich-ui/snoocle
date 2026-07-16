"""Discovery builds a well-formed exact-phrase query even when the title or
artist themselves contain quotes (typical for video-derived identities)."""

from __future__ import annotations

from snoocle_server.discovery.service import discover_sources


def test_query_sanitizes_embedded_quotes():
    captured: dict = {}

    def search_fn(query: str, n: int) -> list:
        captured["query"] = query
        return []

    discover_sources(
        'Blues Traveler "Hook" at Howard Stern\'s 1996 Birthday Show',
        "The Howard Stern Show",
        search_fn=search_fn,
    )
    query = captured["query"]
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
