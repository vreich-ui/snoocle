"""Evidence manifest: telling the reconciler what it's holding.

The manifest exists because an agent handed evidence with no context cannot
distinguish "this is good and current" from "gathering failed, go look harder".
These tests cover the four states that distinction actually turns on — cold
cache, warm cache, a prior song to correct, and a notes-only scope — plus the
constraint that makes the whole thing safe: it is descriptive, so nothing
downstream may depend on it and nothing in it may break a run.
"""

from __future__ import annotations

import pytest

from snoocle_server.discovery.cache import DiscoveryCacheInfo
from snoocle_server.discovery.models import CandidateSource
from snoocle_server.manifest import build_evidence_manifest, lrc_block
from snoocle_server.mir.base import AnalyzedWindow, MirAnalysis
from snoocle_server.mir.cache import MirCacheInfo
from snoocle_server.scope import AnalysisScope


def make_mir(**overrides) -> MirAnalysis:
    base = dict(
        engines={"beats": "madmom", "chords": "chord-cnn-lstm", "structure": "songformer"},
        duration_seconds=181.4,
        bpm=152.0,
        key="A major",
        analyzed_windows=[AnalyzedWindow(start=0.0, end=181.4)],
    )
    base.update(overrides)
    return MirAnalysis(**base)


def make_candidates(n=3):
    return [CandidateSource(sourceId=f"web-{i}", confidence=0.9) for i in range(n)]


def make_prior_song(provenance=None) -> dict:
    return {
        "id": "bob-marley--three-little-birds",
        "metadata": {"title": "Three Little Birds", "artist": "Bob Marley"},
        "lines": [
            {"lineIndex": 0, "lyrics": "don't worry", "chordPlacements": [{"charIndex": 0, "chord": "A"}]}
        ],
        "provenance": provenance if provenance is not None else [],
    }


# --- the four states --------------------------------------------------------


def test_cold_cache_manifest():
    """Nothing reused: every block says so, so the agent knows the evidence is
    freshly gathered rather than stale."""
    m = build_evidence_manifest(
        mir=make_mir(),
        mir_cache=MirCacheInfo(status="miss", analyzed_at="2026-07-28T10:00:00+00:00"),
        candidates=make_candidates(3),
        discovery_cache=DiscoveryCacheInfo(status="miss", gathered_at="2026-07-28T10:00:00+00:00"),
    )
    assert m["mir"]["status"] == "cache-miss"
    assert m["mir"]["bpm"] == 152.0
    assert m["mir"]["key"] == "A major"
    assert m["mir"]["coverage"] == "full-track"
    assert m["mir"]["engines"]["chords"] == "chord-cnn-lstm"
    assert m["sources"]["status"] == "cache-miss"
    assert m["sources"]["count"] == 3
    assert m["priorSong"] == {"exists": False}
    assert m["lrcAlign"] == {"status": "pending"}


def test_warm_cache_manifest_reports_when_the_audio_was_really_heard():
    """A cache hit must carry the ORIGINAL analysis time — that is the whole
    point of surfacing it, and what keeps 'reused' from reading as 'stale'."""
    m = build_evidence_manifest(
        mir=make_mir(),
        mir_cache=MirCacheInfo(status="hit", analyzed_at="2026-07-07T10:00:00+00:00"),
        candidates=make_candidates(3),
        discovery_cache=DiscoveryCacheInfo(status="hit", gathered_at="2026-07-07T10:00:00+00:00"),
    )
    assert m["mir"]["status"] == "cache-hit"
    assert m["mir"]["analyzedAt"] == "2026-07-07T10:00:00+00:00"
    assert m["mir"]["ageDays"] > 0
    assert m["sources"]["status"] == "cache-hit"
    assert m["sources"]["ageDays"] > 0


def test_prior_song_exists_manifest():
    prior = make_prior_song(
        provenance=[
            {"timestamp": "2026-07-01T00:00:00Z", "actor": "snoocle-server/0.1.0",
             "action": "timing-snap", "confidence": 0.91,
             "notes": "matched 21/23 chord placement(s) to the MIR timeline"},
        ]
    )
    m = build_evidence_manifest(prior_song=prior)
    assert m["priorSong"]["exists"] is True
    assert m["priorSong"]["matchRatio"] == 0.91
    assert m["priorSong"]["humanEdited"] is False
    assert m["priorSong"]["version"], "a content-hash version should be derivable"


def test_prior_song_human_edited_is_detected_from_provenance():
    """The only available signal: an entry by an actor the server never writes."""
    machine_only = make_prior_song(
        provenance=[
            {"timestamp": "2026-07-01T00:00:00Z", "actor": "snoocle-server/0.1.0", "action": "mir-analysis"},
            {"timestamp": "2026-07-01T00:00:00Z", "actor": "reconcile:anthropic/claude-opus-4-8", "action": "reconciled"},
        ]
    )
    assert build_evidence_manifest(prior_song=machine_only)["priorSong"]["humanEdited"] is False

    human_touched = make_prior_song(
        provenance=[
            {"timestamp": "2026-07-01T00:00:00Z", "actor": "snoocle-server/0.1.0", "action": "mir-analysis"},
            {"timestamp": "2026-07-02T00:00:00Z", "actor": "wolf@example.com", "action": "manual-edit"},
        ]
    )
    assert build_evidence_manifest(prior_song=human_touched)["priorSong"]["humanEdited"] is True


def test_prior_song_without_a_timing_snap_entry_omits_match_ratio():
    """matchRatio is READ from the prior song's own recorded snap ratio, never
    synthesized — no entry, no field."""
    m = build_evidence_manifest(prior_song=make_prior_song(provenance=[]))
    assert m["priorSong"]["exists"] is True
    assert "matchRatio" not in m["priorSong"]


def test_notes_only_scope_manifest():
    """The state _NOTES_ONLY_BANNER was written for: nothing gathered, on
    purpose. The manifest has to make the deliberateness legible."""
    m = build_evidence_manifest(
        mir=None,
        mir_cache=None,
        candidates=[],
        discovery_cache=None,
        prior_song=make_prior_song(),
        scope=AnalysisScope(listen=False, reconcile=False),
        guidance="the bridge is Bm, not D",
        guidance_origin="stored notes",
    )
    assert m["mir"]["status"] == "absent"
    assert m["sources"] == {"status": "absent", "count": 0}
    assert m["priorSong"]["exists"] is True
    assert m["request"]["scope"] == {"listen": False, "reconcile": False}
    assert m["request"]["notes"] == "the bridge is Bm, not D"
    assert m["request"]["notesOrigin"] == "stored notes"


def test_absent_scope_is_reported_as_null_not_as_a_full_scope():
    """"No opinion" and "explicitly full" are different requests, and the
    manifest must not blur them — the pipeline treats them differently."""
    assert build_evidence_manifest()["request"]["scope"] is None
    assert build_evidence_manifest(scope=AnalysisScope())["request"]["scope"] == {
        "listen": True, "reconcile": True
    }


# --- coverage ---------------------------------------------------------------


def test_coverage_reports_sampled_windows():
    mir = make_mir(
        analyzed_windows=[AnalyzedWindow(start=14.0, end=44.0), AnalyzedWindow(start=90.0, end=120.0)],
        engines={"beats": "madmom", "chords": "chord-cnn-lstm", "structure": "skipped (fast accuracy)"},
    )
    m = build_evidence_manifest(mir=mir, mir_cache=MirCacheInfo(status="miss", analyzed_at="x"))
    assert m["mir"]["coverage"] == "sampled: 14-44s, 90-120s"


def test_coverage_unknown_when_no_windows_recorded():
    m = build_evidence_manifest(
        mir=make_mir(analyzed_windows=[]), mir_cache=MirCacheInfo(status="miss", analyzed_at="x")
    )
    assert m["mir"]["coverage"] == "unknown"


# --- the descriptive-only constraint ----------------------------------------


def test_manifest_never_raises_on_malformed_input():
    """It is purely descriptive, so a bad input must degrade the manifest, not
    fail the run that would otherwise have succeeded."""
    m = build_evidence_manifest(prior_song={"not": "a song", "provenance": "not a list"})
    assert isinstance(m, dict)


def test_manifest_survives_an_unparseable_prior_song():
    m = build_evidence_manifest(prior_song={"id": "!!! invalid slug !!!"})
    assert m["priorSong"]["exists"] is True
    assert "version" not in m["priorSong"], "an unvalidatable song has no derivable version"


def test_no_scores_field_until_a_real_scorer_exists():
    """reconcile.match.score_candidate does not exist on this branch. Inventing
    a stand-in would make the manifest a source of truth about quality rather
    than a description of provenance — so the field is simply absent."""
    import importlib.util

    assert importlib.util.find_spec("snoocle_server.reconcile.match") is None, (
        "reconcile.match now exists — add the `scores` field to _sources_block "
        "using its score_candidate, rather than leaving this test asserting absence"
    )
    m = build_evidence_manifest(
        candidates=make_candidates(3),
        discovery_cache=DiscoveryCacheInfo(status="miss", gathered_at="x"),
    )
    assert "scores" not in m["sources"]


# --- lrcAlign ---------------------------------------------------------------


def test_lrc_block_shapes():
    assert lrc_block("hit", 21, 32) == {"status": "hit", "linesMatched": 21, "linesTotal": 32}
    assert lrc_block("miss", 0, 32) == {"status": "miss", "linesMatched": 0, "linesTotal": 32}
    assert lrc_block("skipped") == {"status": "skipped"}
    assert lrc_block("failed") == {"status": "failed"}


def test_lrc_starts_pending_because_alignment_runs_after_reconciliation():
    """Ordering honesty: LRC is pipeline step 5c, reconciliation is step 5, so
    'pending' is the true state at the moment the reconciler reads this."""
    assert build_evidence_manifest()["lrcAlign"] == {"status": "pending"}
