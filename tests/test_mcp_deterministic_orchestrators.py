from __future__ import annotations

import json

import pytest

from snoocle_server import mcp_server
from snoocle_server.deterministic import build_song_from_candidate
from snoocle_server.discovery.models import CandidateSource, SectionStart
from snoocle_server.mir.base import ChordSegment, MirAnalysis
from snoocle_server.schema import ChordPlacement, Line
from snoocle_server.store.memory import InMemorySongRepository
from snoocle_server.store.runs import InMemoryRunRepository


def _candidate() -> CandidateSource:
    chords = ["C", "G", "Am", "F"]
    return CandidateSource(
        sourceId="sheet-a",
        retrievedAt="2026-08-02T00:00:00Z",
        confidence=0.95,
        parseConfidence=0.95,
        coverage="full-song",
        sectionStarts=[SectionStart(name="Verse", startLineIndex=0)],
        lines=[
            Line(
                lineIndex=index,
                lyrics=f"line {index}",
                chordPlacements=[ChordPlacement(charIndex=0, chord=chord)],
            )
            for index, chord in enumerate(chords)
        ],
    )


def _mir() -> MirAnalysis:
    return MirAnalysis(
        engines={"beats": "test", "chords": "test", "structure": "test"},
        duration_seconds=16,
        key="C major",
        chords=[
            ChordSegment(start=index * 4, end=(index + 1) * 4, chord=chord)
            for index, chord in enumerate(["C", "G", "Am", "F"])
        ],
    )


def _json(value) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value)


@pytest.fixture()
def stores(monkeypatch):
    songs = InMemorySongRepository()
    runs = InMemoryRunRepository()
    monkeypatch.setattr(mcp_server, "get_store", lambda: songs)
    monkeypatch.setattr(mcp_server, "get_run_store", lambda: runs)
    return songs, runs


def _assert_zero_model(response: dict) -> None:
    assert response["modelCalls"] == 0
    assert response["modelCostUSD"] == 0
    assert response["result"]["totals"]["modelCalls"] == 0
    assert response["result"]["totals"]["modelCostUSD"] == 0
    assert all(stage["modelCalls"] == 0 for stage in response["result"]["stages"])


@pytest.mark.anyio
async def test_aligner_is_repeatable_and_runs_stages_in_contract_order(stores):
    candidate = _candidate()
    song = build_song_from_candidate(
        candidate, song_id="artist--title", title="Title", artist="Artist"
    )
    args = {
        "song_json": _json(song),
        "mir_json": _json(_mir()),
        "candidates_json": _json([candidate.model_dump(mode="json")]),
    }

    first = await mcp_server.align_song_deterministically(**args)
    second = await mcp_server.align_song_deterministically(**args)

    assert first["ok"] is True
    assert first["result"]["song"] == second["result"]["song"]
    assert [stage["name"] for stage in first["result"]["stages"]] == [
        "snap_chords",
        "lrc_alignment",
        "section_timing",
        "collapse_guard",
        "confidence_scoring",
        "quality_grading",
    ]
    report = first["result"]["alignmentReport"]
    assert report["matchedChordCount"] == 4
    assert report["unmatchedChordCount"] == 0
    assert "lineTimingCoverage" in report
    assert "sectionCoverage" in report
    assert "interpolationShare" in report
    assert "collapsedTimingInterventions" in report
    _assert_zero_model(first)
    assert stores[1].get_run(first["result"]["runId"])["runType"] == "deterministic-align"


@pytest.mark.anyio
async def test_complete_processor_stage_order_and_direct_callability(stores, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("model or reconciliation boundary called")

    monkeypatch.setattr(mcp_server, "reconcile_admitted", forbidden)
    monkeypatch.setattr(mcp_server, "run_pipeline_async", forbidden)
    candidate = _candidate()
    response = await mcp_server.process_song_deterministically(
        "Title",
        "Artist",
        mir_json=_json(_mir()),
        candidates_json=_json([candidate.model_dump(mode="json")]),
    )

    assert response["ok"] is True
    assert [stage["name"] for stage in response["result"]["stages"]] == [
        "identity",
        "acquire_audio",
        "mir",
        "discovery",
        "candidate_selection",
        "baseline",
        "snap_chords",
        "lrc_alignment",
        "section_timing",
        "collapse_guard",
        "confidence_scoring",
        "quality_grading",
    ]
    assert response["result"]["song"]["id"] == "artist--title"
    assert response["result"]["selection"]["selectedSourceId"] == "sheet-a"
    _assert_zero_model(response)


@pytest.mark.anyio
async def test_processor_stops_early_with_stable_structured_reason(stores):
    response = await mcp_server.process_song_deterministically(
        "Title", "Artist", mir_json=_json(_mir()), candidates_json="[]"
    )

    assert response["ok"] is True
    assert response["result"]["status"] == "needs_review"
    assert response["result"]["reason"] == "no_candidate_sources"
    assert response["result"]["conflicts"] == []
    assert "song" not in response["result"]
    assert [stage["name"] for stage in response["result"]["stages"]] == [
        "identity", "acquire_audio", "mir", "discovery", "candidate_selection"
    ]
    assert stores[1].get_run(response["result"]["runId"])["reason"] == "no_candidate_sources"

    identity = await mcp_server.process_song_deterministically(
        "", "", mir_json=_json(_mir()), candidates_json="[]"
    )
    assert identity["ok"] is True
    assert identity["result"]["status"] == "needs_review"
    assert identity["result"]["reason"] == "identity_unresolved"
    assert identity["result"]["conflicts"][0]["type"] == "identity"


@pytest.mark.anyio
async def test_song_persistence_is_opt_in_and_optimistically_locked(stores):
    songs, runs = stores
    candidate = _candidate()
    song = build_song_from_candidate(
        candidate, song_id="artist--title", title="Title", artist="Artist"
    )
    initial = songs.save(song, "initial")
    args = {
        "song_json": _json(song),
        "mir_json": _json(_mir()),
        "candidates_json": _json([candidate.model_dump(mode="json")]),
        "persist": True,
    }

    missing = await mcp_server.align_song_deterministically(**args)
    assert missing["ok"] is False
    assert missing["error"]["code"] == "missing_expected_version"
    assert songs.current_version(song.id) == initial.version

    loaded = await mcp_server.align_song_deterministically(
        song_id=song.id,
        song_version=initial.version,
        cached_mir_json=_json(_mir()),
        candidates_json=args["candidates_json"],
    )
    assert loaded["ok"] is True
    assert loaded["result"]["songSource"] == "store"
    assert loaded["result"]["cache"]["mir"] == "hit"
    assert loaded["cacheStatus"] == "hit"

    stored = await mcp_server.align_song_deterministically(
        **args, expected_version=initial.version
    )
    assert stored["ok"] is True
    assert stored["result"]["persistence"]["stored"] is True
    assert songs.current_version(song.id) != initial.version

    stale = await mcp_server.align_song_deterministically(
        **args, expected_version=initial.version
    )
    assert stale["ok"] is False
    assert stale["error"]["code"] == "version_conflict"
    assert any(
        run["status"] == "failed" and run["reason"] == "version_conflict"
        for run in runs.list_all_runs()
    )


@pytest.mark.anyio
async def test_blocking_orchestrator_work_is_dispatched_to_a_thread(stores, monkeypatch):
    calls = []

    async def fake_to_thread(fn, /, *args, **kwargs):
        calls.append(fn.__name__)
        return fn(*args, **kwargs)

    monkeypatch.setattr(mcp_server.asyncio, "to_thread", fake_to_thread)
    response = await mcp_server.process_song_deterministically(
        "Title", "Artist", mir_json=_json(_mir()), candidates_json="[]"
    )

    assert response["ok"] is True
    assert calls == ["_process_song_worker"]
