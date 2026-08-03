"""Cache + manifest wired into the real pipeline and prompt.

The unit suites cover the cache and the manifest in isolation. What's left is
the part that could still be semantically visible: that provenance keeps
telling the truth about when the audio was heard, that the manifest actually
reaches the model, and that the caches are orthogonal to `scope` rather than a
second way of expressing it.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from snoocle_server import pipeline as pipeline_mod
from snoocle_server.api import app
from snoocle_server.config import settings
from snoocle_server.discovery.cache import reset_discovery_cache
from snoocle_server.mir.base import AnalyzedWindow, ChordSegment, MirAnalysis
from snoocle_server.mir.cache import MirCacheInfo, reset_mir_cache

client = TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    monkeypatch.setattr(settings, "store_backend", "memory", raising=False)
    reset_mir_cache()
    reset_discovery_cache()
    yield
    reset_mir_cache()
    reset_discovery_cache()


def make_mir() -> MirAnalysis:
    return MirAnalysis(
        engines={"beats": "madmom", "chords": "chord-cnn-lstm", "structure": "songformer"},
        duration_seconds=181.4,
        bpm=152.0,
        key="A major",
        chords=[ChordSegment(start=0.0, end=2.5, chord="A")],
        analyzed_windows=[AnalyzedWindow(start=0.0, end=181.4)],
    )


# --- provenance stays truthful ------------------------------------------------


def _mir_provenance_notes(song) -> str:
    entries = [p for p in song.provenance if p.action == "mir-analysis"]
    assert entries, "expected an mir-analysis provenance entry"
    return entries[-1].notes or ""


def test_fresh_analysis_provenance_does_not_claim_a_cache_hit():
    from snoocle_server.reconcile import reconcile

    result = reconcile(
        "Three Little Birds", "Bob Marley", [], make_mir(),
        provider_name="mock", song_id="bob-marley--three-little-birds",
        mir_cache=MirCacheInfo(status="miss", analyzed_at="2026-07-28T10:00:00+00:00"),
    )
    notes = _mir_provenance_notes(result.song)
    assert "bpm=152.0" in notes and "key=A major" in notes
    assert "cache" not in notes


def test_cached_analysis_provenance_records_the_reuse_and_the_original_time():
    """Reading a song must still tell you when its audio was ACTUALLY listened
    to — collapsing that into the version's own timestamp would make the
    history claim a re-listen that never happened."""
    from snoocle_server.reconcile import reconcile

    result = reconcile(
        "Three Little Birds", "Bob Marley", [], make_mir(),
        provider_name="mock", song_id="bob-marley--three-little-birds",
        mir_cache=MirCacheInfo(status="hit", analyzed_at="2026-07-07T09:15:00+00:00"),
    )
    notes = _mir_provenance_notes(result.song)
    assert "reused from cache" in notes
    assert "analyzedAt=2026-07-07T09:15:00+00:00" in notes


def test_provenance_is_unchanged_when_no_cache_info_is_supplied():
    """Every pre-cache caller passes nothing — their provenance must read
    exactly as it always did."""
    from snoocle_server.reconcile import reconcile

    result = reconcile(
        "Three Little Birds", "Bob Marley", [], make_mir(),
        provider_name="mock", song_id="bob-marley--three-little-birds",
    )
    assert _mir_provenance_notes(result.song) == (
        "audio-grounded analysis; bpm=152.0, key=A major"
    )


# --- the manifest reaches the model -------------------------------------------


def test_manifest_is_injected_into_the_rendered_prompt():
    from snoocle_server.reconcile.prompt import build_user_prompt
    from snoocle_server.schema import song_json_schema

    manifest = {
        "mir": {"status": "cache-hit", "bpm": 152.0, "key": "A major"},
        "sources": {"status": "cache-hit", "count": 3},
    }
    prompt = build_user_prompt(
        "Three Little Birds", "Bob Marley", [], None, song_json_schema(),
        "bob-marley--three-little-birds", None, evidence_manifest=manifest,
    )
    assert "evidenceManifest" in prompt
    assert "cache-hit" in prompt
    # And it says plainly that reused evidence is still good evidence — the
    # failure mode the manifest exists to prevent is a model reading "cached"
    # as "stale" and re-deriving from memory.
    assert "as valid as freshly gathered" in prompt


def test_absent_manifest_leaves_the_prompt_byte_identical():
    """Every pre-manifest caller (and test_provider_parity) must be unaffected."""
    from snoocle_server.reconcile.prompt import build_user_prompt
    from snoocle_server.schema import song_json_schema

    args = ("Let It Be", "The Beatles", [], None, song_json_schema(), "the-beatles--let-it-be", None)
    assert build_user_prompt(*args) == build_user_prompt(*args, evidence_manifest=None)
    assert "evidenceManifest" not in build_user_prompt(*args)


def test_manifest_reaches_the_in_process_agents_first_user_message():
    """The agentic providers send structured JSON, not the rendered prompt, so
    the manifest has to be in THAT payload too — otherwise agentic runs (the
    default provider family) never see it. Exercises the real builder."""
    from snoocle_server.discovery.models import CandidateSource
    from snoocle_server.reconcile.anthropic_agent import AnthropicAgentProvider

    provider = AnthropicAgentProvider()
    provider.context = {
        "song_id": "x--y", "title": "Y", "artist": "X",
        "mir": None, "candidates": [CandidateSource(sourceId="web-0")],
        "song_schema": {}, "depth": None,
        "evidence_manifest": {"mir": {"status": "cache-hit", "bpm": 152.0}},
    }
    # The builder returns a message envelope with the payload serialized into
    # its content — assert on what actually goes over the wire.
    message = provider._build_first_user_message()
    payload = json.loads(message["content"])
    assert payload["evidenceManifest"] == {"mir": {"status": "cache-hit", "bpm": 152.0}}


def test_agents_first_user_message_omits_the_manifest_when_absent():
    from snoocle_server.reconcile.anthropic_agent import AnthropicAgentProvider

    provider = AnthropicAgentProvider()
    provider.context = {
        "song_id": "x--y", "title": "Y", "artist": "X",
        "mir": None, "candidates": [], "song_schema": {}, "depth": None,
    }
    payload = json.loads(provider._build_first_user_message()["content"])
    assert "evidenceManifest" not in payload


# --- end-to-end through the API ------------------------------------------------


def _analyze(**extra) -> dict:
    body = {
        "title": "Cache",
        "artist": "Tester",
        "provider": "mock",
        "skipAudio": True,
        "allowTestOutput": True,
    }
    body.update(extra)
    r = client.post("/v1/songs/analyze", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_analyze_response_carries_the_manifest():
    body = _analyze()
    assert body["song"]["testOnly"] is True
    manifest = body["evidenceManifest"]
    assert set(manifest) >= {"mir", "sources", "priorSong", "lrcAlign", "request"}
    # mock skips LRC entirely, and the manifest says so rather than staying
    # stuck on the "pending" it was built with.
    assert manifest["lrcAlign"]["status"] == "skipped"


def test_run_trace_records_the_manifest():
    body = _analyze()
    run = client.get(f"/v1/runs/{body['runId']}")
    assert run.status_code == 200, run.text
    assert run.json()["evidenceManifest"]["mir"]["status"] == "absent"


def test_refresh_cache_defaults_off_and_is_accepted():
    """The per-request bypass exists and does not disturb an ordinary run."""
    assert _analyze()["songId"] == "tester--cache"
    assert _analyze(
        refreshCache=True,
        force=True,
        forceReason="test intentionally verifies an explicit refresh request",
    )["songId"] == "tester--cache"


# --- orthogonality with scope ---------------------------------------------------


def test_the_cache_does_not_redefine_scope_listen_off(monkeypatch):
    """`scope.listen=false` still means "don't acquire, don't analyze" — NOT
    "serve the analysis from cache". The two concepts must not collapse: one is
    about whether to listen at all, the other about how cheaply."""
    def boom(*a, **k):  # noqa: ANN001
        raise AssertionError("audio was acquired/analyzed despite listen=false")

    monkeypatch.setattr(pipeline_mod, "acquire", boom)
    monkeypatch.setattr(pipeline_mod, "analyze_audio", boom)
    monkeypatch.setattr(pipeline_mod, "analyze_cached", boom)
    monkeypatch.setattr(pipeline_mod, "discover_sources", lambda *a, **k: [])

    body = _analyze(skipAudio=False, scope={"listen": False, "reconcile": True})
    assert body["steps"]["mir"].startswith("skipped (scope:")
    assert body["evidenceManifest"]["mir"]["status"] == "absent"


def test_manifest_reports_the_scope_it_was_given():
    body = _analyze(scope={"listen": True, "reconcile": False})
    assert body["evidenceManifest"]["request"]["scope"] == {"listen": True, "reconcile": False}


def test_absent_scope_still_reports_null_in_the_manifest():
    assert _analyze()["evidenceManifest"]["request"]["scope"] is None


# --- the step report ------------------------------------------------------------


def test_step_text_is_unchanged_on_a_cold_cache(monkeypatch):
    """A cold-cache run must report exactly the text it always did — the cache
    annotation only appears when something was actually reused."""
    monkeypatch.setattr(pipeline_mod, "discover_sources", lambda *a, **k: [])
    body = _analyze()
    assert "(cache" not in body["steps"].get("discover", "")


def test_cache_suffix_only_annotates_reuse():
    from snoocle_server.pipeline import _cache_suffix

    assert _cache_suffix("miss") == ""
    assert _cache_suffix("disabled") == ""
    assert _cache_suffix("hit") == " (cache hit)"
    assert _cache_suffix("refresh") == " (cache refresh)"
