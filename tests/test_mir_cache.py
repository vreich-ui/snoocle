"""MIR result cache: same audio analyzed once, with nothing else changed.

The cache is a pure optimization, so the tests are mostly about what must NOT
shift: a hit deserializes to an object byte-identical to what a miss computes,
every key component invalidates independently, and provenance still tells the
truth about when the audio was actually listened to.

The motivating evidence: one song accumulated 10 stored versions inside 16
minutes, each re-running the full stack over byte-identical audio and each
returning bpm=152.0, key=A major.
"""

from __future__ import annotations

import pytest

from snoocle_server.config import settings
from snoocle_server.mir.base import AnalyzedWindow, Beat, ChordSegment, MirAnalysis, StructureSegment
from snoocle_server.mir.cache import (
    WINDOW_ACCURACY,
    MirCacheInfo,
    analyze_cached,
    audio_sha256,
    engine_fingerprint,
    mir_cache_key,
    reset_mir_cache,
)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    """A real in-memory cache store per test, and the caches on by default."""
    monkeypatch.setattr(settings, "store_backend", "memory", raising=False)
    monkeypatch.setattr(settings, "mir_cache_enabled", True, raising=False)
    reset_mir_cache()
    yield
    reset_mir_cache()


@pytest.fixture()
def audio(tmp_path):
    p = tmp_path / "song.wav"
    p.write_bytes(b"RIFF-pretend-audio-bytes" * 100)
    return p


def make_analysis(bpm: float = 152.0, key: str = "A major") -> MirAnalysis:
    """A fixture shaped like a real full-track result — every field populated,
    so a round-trip that drops or coerces one is caught."""
    return MirAnalysis(
        engines={"beats": "madmom", "chords": "chord-cnn-lstm", "structure": "songformer"},
        duration_seconds=181.4,
        bpm=bpm,
        time_signature="4/4",
        key=key,
        beats=[Beat(time=0.5, position=1), Beat(time=0.895, position=2)],
        chords=[
            ChordSegment(start=0.0, end=2.5, chord="A"),
            ChordSegment(start=2.5, end=5.0, chord="D"),
            ChordSegment(start=5.0, end=7.5, chord="N"),
        ],
        sections=[StructureSegment(start=0.0, end=30.0, label="verse")],
        analyzed_windows=[AnalyzedWindow(start=0.0, end=181.4)],
    )


class _Counter:
    """A compute function that records how many times it actually ran."""

    def __init__(self, analysis: MirAnalysis | None = None):
        self.calls = 0
        self.analysis = analysis or make_analysis()

    def __call__(self) -> MirAnalysis:
        self.calls += 1
        return self.analysis


def _engines(chords="chord-cnn-lstm", beats="madmom", structure="songformer"):
    return {"beats": beats, "chords": chords, "structure": structure}


# --- the core property: caching is invisible ---------------------------------


def test_hit_is_byte_identical_to_a_miss(audio, monkeypatch):
    """The whole design rests on this: a cached analysis deserializes to
    exactly what recomputing would have produced, field for field."""
    monkeypatch.setattr("snoocle_server.mir.cache.predicted_engines", lambda a: _engines())
    compute = _Counter()

    fresh, fresh_info = analyze_cached(audio, accuracy="standard", compute=compute)
    cached, cached_info = analyze_cached(audio, accuracy="standard", compute=compute)

    assert compute.calls == 1, "the second call must not have recomputed"
    assert fresh_info.status == "miss"
    assert cached_info.status == "hit"
    assert cached.model_dump(mode="json") == fresh.model_dump(mode="json")
    assert cached == fresh
    # Spot-check the exact values from the motivating evidence.
    assert cached.bpm == 152.0 and cached.key == "A major"


def test_second_run_reuses_instead_of_recomputing(audio, monkeypatch):
    """Ten identical executions of a pure function become one."""
    monkeypatch.setattr("snoocle_server.mir.cache.predicted_engines", lambda a: _engines())
    compute = _Counter()
    for _ in range(10):
        analyze_cached(audio, accuracy="standard", compute=compute)
    assert compute.calls == 1


# --- each key component invalidates independently ----------------------------


def test_different_audio_bytes_miss(tmp_path, monkeypatch):
    """Keyed on CONTENT, not on a video id: a re-upload under the same id has
    different bytes and must miss."""
    monkeypatch.setattr("snoocle_server.mir.cache.predicted_engines", lambda a: _engines())
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    a.write_bytes(b"original upload")
    b.write_bytes(b"re-upload, remastered")
    compute = _Counter()

    analyze_cached(a, accuracy="standard", compute=compute)
    analyze_cached(b, accuracy="standard", compute=compute)
    assert compute.calls == 2


def test_identical_audio_under_two_names_hits(tmp_path, monkeypatch):
    """The flip side: two different ids resolving to identical audio SHOULD
    share a cache entry."""
    monkeypatch.setattr("snoocle_server.mir.cache.predicted_engines", lambda a: _engines())
    a = tmp_path / "video-aaa.wav"
    b = tmp_path / "video-bbb.wav"
    payload = b"byte for byte the same recording"
    a.write_bytes(payload)
    b.write_bytes(payload)
    compute = _Counter()

    analyze_cached(a, accuracy="standard", compute=compute)
    _, info = analyze_cached(b, accuracy="standard", compute=compute)
    assert compute.calls == 1
    assert info.status == "hit"


def test_different_chord_engine_misses(audio, monkeypatch):
    """Losing the heavy model to the chroma fallback must invalidate, not
    silently serve the other engine's answer."""
    engines = {"value": _engines(chords="chord-cnn-lstm")}
    monkeypatch.setattr(
        "snoocle_server.mir.cache.predicted_engines", lambda a: engines["value"]
    )
    compute = _Counter()
    analyze_cached(audio, accuracy="standard", compute=compute)

    engines["value"] = _engines(chords="chroma-template-fallback")
    compute.analysis = make_analysis()
    compute.analysis = compute.analysis.model_copy(
        update={"engines": _engines(chords="chroma-template-fallback")}
    )
    _, info = analyze_cached(audio, accuracy="standard", compute=compute)
    assert compute.calls == 2
    assert info.status == "miss"


def test_different_structure_engine_misses(audio, monkeypatch):
    """Swapping SongFormer in or out invalidates too."""
    engines = {"value": _engines(structure="librosa-agglomerative-fallback")}
    monkeypatch.setattr(
        "snoocle_server.mir.cache.predicted_engines", lambda a: engines["value"]
    )
    compute = _Counter()
    compute.analysis = make_analysis().model_copy(update={"engines": engines["value"]})
    analyze_cached(audio, accuracy="standard", compute=compute)

    engines["value"] = _engines(structure="songformer")
    compute.analysis = make_analysis().model_copy(update={"engines": engines["value"]})
    analyze_cached(audio, accuracy="standard", compute=compute)
    assert compute.calls == 2


def test_different_accuracy_misses(audio, monkeypatch):
    monkeypatch.setattr("snoocle_server.mir.cache.predicted_engines", lambda a: _engines())
    compute = _Counter()
    analyze_cached(audio, accuracy="standard", compute=compute)
    analyze_cached(audio, accuracy="thorough", compute=compute)
    assert compute.calls == 2


def test_different_window_misses_and_same_window_hits(audio, monkeypatch):
    """A probe of 72.86-105.71s caches independently of the full-track run and
    of every other probe."""
    monkeypatch.setattr("snoocle_server.mir.cache.predicted_engines", lambda a: _engines())
    compute = _Counter()

    analyze_cached(audio, accuracy=WINDOW_ACCURACY, window=(72.86, 105.71), compute=compute)
    analyze_cached(audio, accuracy=WINDOW_ACCURACY, window=(120.0, 150.0), compute=compute)
    assert compute.calls == 2, "a different window must miss"

    _, info = analyze_cached(
        audio, accuracy=WINDOW_ACCURACY, window=(72.86, 105.71), compute=compute
    )
    assert compute.calls == 2, "the same window must hit"
    assert info.status == "hit"


def test_a_window_probe_does_not_collide_with_the_full_track_run(audio, monkeypatch):
    monkeypatch.setattr("snoocle_server.mir.cache.predicted_engines", lambda a: _engines())
    compute = _Counter()
    analyze_cached(audio, accuracy="standard", compute=compute)
    analyze_cached(audio, accuracy=WINDOW_ACCURACY, window=(0.0, 30.0), compute=compute)
    assert compute.calls == 2


def test_fast_window_settings_are_in_the_key(audio, monkeypatch):
    """The fast windows are a pure function of the audio and these settings,
    so changing them must invalidate."""
    monkeypatch.setattr("snoocle_server.mir.cache.predicted_engines", lambda a: _engines())
    monkeypatch.setattr(settings, "mir_fast_window_count", 3, raising=False)
    compute = _Counter()
    analyze_cached(audio, accuracy="fast", compute=compute)

    monkeypatch.setattr(settings, "mir_fast_window_count", 5, raising=False)
    analyze_cached(audio, accuracy="fast", compute=compute)
    assert compute.calls == 2


# --- the bypass --------------------------------------------------------------


def test_refresh_forces_a_miss_and_overwrites(audio, monkeypatch):
    """The escape hatch for an engine upgraded in place without its id
    changing — the one change the key cannot see."""
    monkeypatch.setattr("snoocle_server.mir.cache.predicted_engines", lambda a: _engines())
    compute = _Counter()
    analyze_cached(audio, accuracy="standard", compute=compute)
    assert compute.calls == 1

    compute.analysis = make_analysis(bpm=76.0, key="D minor")  # "upgraded" engine
    refreshed, info = analyze_cached(audio, accuracy="standard", compute=compute, refresh=True)
    assert compute.calls == 2
    assert info.status == "refresh"
    assert refreshed.bpm == 76.0

    # And the overwrite stuck: the next ordinary call serves the new answer.
    after, after_info = analyze_cached(audio, accuracy="standard", compute=compute)
    assert compute.calls == 2
    assert after_info.status == "hit"
    assert after.bpm == 76.0


def test_config_flag_disables_the_cache(audio, monkeypatch):
    monkeypatch.setattr("snoocle_server.mir.cache.predicted_engines", lambda a: _engines())
    monkeypatch.setattr(settings, "mir_cache_enabled", False, raising=False)
    compute = _Counter()
    analyze_cached(audio, accuracy="standard", compute=compute)
    _, info = analyze_cached(audio, accuracy="standard", compute=compute)
    assert compute.calls == 2
    assert info.status == "disabled"


# --- runtime-fallback safety -------------------------------------------------


def test_a_runtime_fallback_is_not_served_as_the_heavy_engine(audio, monkeypatch):
    """The read/write key asymmetry. When the heavy model is present but
    crashes, the result is written under the FALLBACK's key — so the next run,
    still predicting the heavy engine, retries it instead of being handed the
    chroma answer forever."""
    monkeypatch.setattr(
        "snoocle_server.mir.cache.predicted_engines",
        lambda a: _engines(chords="chord-cnn-lstm"),
    )
    compute = _Counter()
    # The stack fell back at runtime: predicted cnn-lstm, actually got chroma.
    compute.analysis = make_analysis().model_copy(
        update={"engines": _engines(chords="chroma-template-fallback")}
    )

    analyze_cached(audio, accuracy="standard", compute=compute)
    _, info = analyze_cached(audio, accuracy="standard", compute=compute)
    assert compute.calls == 2, "a degraded result must not satisfy a heavy-engine lookup"
    assert info.status == "miss"


def test_removing_the_heavy_model_hits_the_earlier_fallback_entry(audio, monkeypatch):
    """The other direction of the same asymmetry: once the prediction itself
    becomes the fallback, those fallback-keyed entries start hitting."""
    fallback = _engines(chords="chroma-template-fallback")
    monkeypatch.setattr(
        "snoocle_server.mir.cache.predicted_engines",
        lambda a: _engines(chords="chord-cnn-lstm"),
    )
    compute = _Counter()
    compute.analysis = make_analysis().model_copy(update={"engines": fallback})
    analyze_cached(audio, accuracy="standard", compute=compute)

    # Operator removes the model dir; the prediction now matches reality.
    monkeypatch.setattr("snoocle_server.mir.cache.predicted_engines", lambda a: fallback)
    _, info = analyze_cached(audio, accuracy="standard", compute=compute)
    assert compute.calls == 1
    assert info.status == "hit"


# --- key derivation ----------------------------------------------------------


def test_engine_fingerprint_excludes_the_sampling_slot():
    """`sampling` renders the chosen windows, which aren't known until the
    audio is read — including it would make fast-accuracy lookups unpredictable
    and therefore never hit."""
    with_sampling = {**_engines(), "sampling": "fast: 10-40s, 90-120s"}
    assert engine_fingerprint(with_sampling) == engine_fingerprint(_engines())


def test_cache_key_is_stable_and_order_independent():
    a = mir_cache_key(audio_hash="abc", engines=_engines(), accuracy="standard")
    b = mir_cache_key(
        audio_hash="abc",
        engines={"structure": "songformer", "chords": "chord-cnn-lstm", "beats": "madmom"},
        accuracy="standard",
    )
    assert a == b


def test_window_bounds_are_rounded_so_float_noise_does_not_miss():
    a = mir_cache_key(
        audio_hash="abc", engines=_engines(), accuracy=WINDOW_ACCURACY, window=(72.86, 105.71)
    )
    b = mir_cache_key(
        audio_hash="abc",
        engines=_engines(),
        accuracy=WINDOW_ACCURACY,
        window=(72.8600000000001, 105.7099999999999),
    )
    assert a == b


def test_audio_sha256_is_content_addressed(tmp_path):
    a, b, c = (tmp_path / n for n in ("a", "b", "c"))
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    c.write_bytes(b"different")
    assert audio_sha256(a) == audio_sha256(b)
    assert audio_sha256(a) != audio_sha256(c)


def test_cache_info_from_cache_property():
    assert MirCacheInfo(status="hit", analyzed_at="x").from_cache is True
    for status in ("miss", "refresh", "disabled"):
        assert MirCacheInfo(status=status, analyzed_at="x").from_cache is False


def test_unreadable_entry_degrades_to_a_miss(audio, monkeypatch):
    """A cache is an optimization: a corrupt entry must recompute, not raise."""
    monkeypatch.setattr("snoocle_server.mir.cache.predicted_engines", lambda a: _engines())
    from snoocle_server.mir.cache import get_mir_cache

    compute = _Counter()
    analyze_cached(audio, accuracy="standard", compute=compute)

    store = get_mir_cache()
    for key in list(store._docs):  # in-memory backend
        store._docs[key]["value"] = {"totally": "not a MirAnalysis"}

    analysis, info = analyze_cached(audio, accuracy="standard", compute=compute)
    assert compute.calls == 2
    assert info.status == "miss"
    assert analysis.bpm == 152.0


def test_unhashable_audio_degrades_to_uncached(tmp_path, monkeypatch):
    monkeypatch.setattr("snoocle_server.mir.cache.predicted_engines", lambda a: _engines())
    compute = _Counter()
    analysis, info = analyze_cached(
        tmp_path / "does-not-exist.wav", accuracy="standard", compute=compute
    )
    assert compute.calls == 1
    assert info.status == "disabled"
    assert analysis.bpm == 152.0
