"""Discovery result cache — the one that DOES expire.

The contrast with the MIR cache is the point of these tests. MIR keys on a
content hash of the audio, so nothing outside the key can change the answer and
entries never need to expire. Discovery keys on (title, artist, backends), and
the web behind that key genuinely changes — so time is the only invalidator
available, and it has to actually work.
"""

from __future__ import annotations

import pytest

from snoocle_server.config import settings
from snoocle_server.discovery.cache import (
    DiscoveryCacheInfo,
    discover_cached,
    discovery_cache_key,
    get_discovery_cache,
    reset_discovery_cache,
)
from snoocle_server.discovery.models import CandidateSource


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(settings, "store_backend", "memory", raising=False)
    monkeypatch.setattr(settings, "discovery_cache_enabled", True, raising=False)
    monkeypatch.setattr(settings, "discovery_cache_ttl_days", 30.0, raising=False)
    reset_discovery_cache()
    yield
    reset_discovery_cache()


def make_candidates(n: int = 3) -> list[CandidateSource]:
    return [
        CandidateSource(sourceId=f"web-{i}", url=f"https://example.test/{i}", confidence=0.9 - i * 0.1)
        for i in range(n)
    ]


class _Counter:
    def __init__(self, candidates=None):
        self.calls = 0
        self.candidates = make_candidates() if candidates is None else candidates

    def __call__(self):
        self.calls += 1
        return self.candidates


def test_hit_returns_identical_candidates_without_searching():
    discover = _Counter()
    fresh, fresh_info = discover_cached("Let It Be", "The Beatles", max_candidates=8, discover=discover)
    cached, cached_info = discover_cached("Let It Be", "The Beatles", max_candidates=8, discover=discover)

    assert discover.calls == 1
    assert fresh_info.status == "miss" and cached_info.status == "hit"
    assert [c.model_dump(mode="json") for c in cached] == [
        c.model_dump(mode="json") for c in fresh
    ]


def test_the_discovery_cache_expires(monkeypatch):
    """The property the MIR cache deliberately does NOT have."""
    discover = _Counter()
    discover_cached("Let It Be", "The Beatles", max_candidates=8, discover=discover)
    assert discover.calls == 1

    # Age every entry past the TTL by rewriting its stored timestamp.
    store = get_discovery_cache()
    for doc in store._docs.values():
        doc["storedAt"] = "2020-01-01T00:00:00+00:00"

    _, info = discover_cached("Let It Be", "The Beatles", max_candidates=8, discover=discover)
    assert discover.calls == 2, "an entry past its TTL must be re-searched"
    assert info.status == "miss"


def test_the_mir_cache_does_not_expire(tmp_path, monkeypatch):
    """The contrast, asserted directly: an ancient MIR entry still hits,
    because its key already encodes everything that could change the answer."""
    from snoocle_server.mir.base import MirAnalysis
    from snoocle_server.mir.cache import analyze_cached, get_mir_cache, reset_mir_cache

    monkeypatch.setattr(settings, "mir_cache_enabled", True, raising=False)
    monkeypatch.setattr(
        "snoocle_server.mir.cache.predicted_engines",
        lambda a: {"beats": "madmom", "chords": "chord-cnn-lstm", "structure": "songformer"},
    )
    reset_mir_cache()
    try:
        audio = tmp_path / "song.wav"
        audio.write_bytes(b"audio")
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return MirAnalysis(
                engines={"beats": "madmom", "chords": "chord-cnn-lstm", "structure": "songformer"}
            )

        analyze_cached(audio, accuracy="standard", compute=compute)
        for doc in get_mir_cache()._docs.values():
            doc["storedAt"] = "2020-01-01T00:00:00+00:00"

        _, info = analyze_cached(audio, accuracy="standard", compute=compute)
        assert calls["n"] == 1, "a MIR entry must never expire on age alone"
        assert info.status == "hit"
    finally:
        reset_mir_cache()


def test_different_identity_misses():
    discover = _Counter()
    discover_cached("Let It Be", "The Beatles", max_candidates=8, discover=discover)
    discover_cached("Hey Jude", "The Beatles", max_candidates=8, discover=discover)
    assert discover.calls == 2


def test_identity_is_case_and_whitespace_insensitive():
    """Same search, spelled differently, shouldn't cost a second search."""
    discover = _Counter()
    discover_cached("Let It Be", "The Beatles", max_candidates=8, discover=discover)
    _, info = discover_cached("  let it  be ", "the beatles", max_candidates=8, discover=discover)
    assert discover.calls == 1
    assert info.status == "hit"


def test_backend_set_is_in_the_key(monkeypatch):
    """Changing which backends are enabled changes what comes back."""
    monkeypatch.setattr(settings, "search_backends", "brave,duckduckgo", raising=False)
    discover = _Counter()
    discover_cached("X", "Y", max_candidates=8, discover=discover)

    monkeypatch.setattr(settings, "search_backends", "brave", raising=False)
    discover_cached("X", "Y", max_candidates=8, discover=discover)
    assert discover.calls == 2


def test_ultimate_guitar_toggle_is_in_the_key(monkeypatch):
    monkeypatch.setattr(settings, "source_ug_enabled", True, raising=False)
    discover = _Counter()
    discover_cached("X", "Y", max_candidates=8, discover=discover)

    monkeypatch.setattr(settings, "source_ug_enabled", False, raising=False)
    discover_cached("X", "Y", max_candidates=8, discover=discover)
    assert discover.calls == 2


def test_site_ranking_and_recording_variant_are_in_the_key():
    discover = _Counter()
    discover_cached(
        "X", "Y", max_candidates=3, discover=discover,
        recording_variant="studio",
        site_preferences={"global": ["ultimate-guitar.com"]},
    )
    discover_cached(
        "X", "Y", max_candidates=3, discover=discover,
        recording_variant="live",
        site_preferences={"global": ["ultimate-guitar.com"]},
    )
    discover_cached(
        "X", "Y", max_candidates=3, discover=discover,
        recording_variant="live",
        site_preferences={"global": ["chordify.net"]},
    )
    assert discover.calls == 3


def test_asking_for_more_candidates_misses():
    """A narrower run's results must not answer a wider request."""
    discover = _Counter()
    discover_cached("X", "Y", max_candidates=3, discover=discover)
    discover_cached("X", "Y", max_candidates=8, discover=discover)
    assert discover.calls == 2


def test_empty_results_are_not_cached():
    """Discovery returning nothing is usually a throttled backend, not a fact
    about the song — freezing it for a month would turn a transient failure
    into a lasting one."""
    empty = _Counter(candidates=[])
    discover_cached("X", "Y", max_candidates=8, discover=empty)
    discover_cached("X", "Y", max_candidates=8, discover=empty)
    assert empty.calls == 2


def test_refresh_forces_a_new_search():
    discover = _Counter()
    discover_cached("X", "Y", max_candidates=8, discover=discover)
    _, info = discover_cached("X", "Y", max_candidates=8, discover=discover, refresh=True)
    assert discover.calls == 2
    assert info.status == "refresh"


def test_config_flag_disables_the_cache(monkeypatch):
    monkeypatch.setattr(settings, "discovery_cache_enabled", False, raising=False)
    discover = _Counter()
    discover_cached("X", "Y", max_candidates=8, discover=discover)
    _, info = discover_cached("X", "Y", max_candidates=8, discover=discover)
    assert discover.calls == 2
    assert info.status == "disabled"


def test_cache_key_is_stable():
    a = discovery_cache_key("Let It Be", "The Beatles", 8)
    b = discovery_cache_key("Let It Be", "The Beatles", 8)
    assert a == b


def test_cache_info_from_cache_property():
    assert DiscoveryCacheInfo(status="hit", gathered_at="x").from_cache is True
    assert DiscoveryCacheInfo(status="miss", gathered_at="x").from_cache is False
