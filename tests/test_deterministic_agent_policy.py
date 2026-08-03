from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest

from snoocle_server import deterministic_policy, pipeline as pipeline_mod
from snoocle_server.deterministic import (
    align_song_deterministically_service,
    build_song_from_candidate,
)
from snoocle_server.deterministic_policy import (
    AgentPolicyError,
    model_conflict_is_actionable,
    run_bounded_agent_patch,
    validate_compact_conflict_packet,
)
from snoocle_server.deterministic_process import DeterministicProcessResult
from snoocle_server.discovery.models import CandidateSource, SectionStart
from snoocle_server.mir.base import ChordSegment, MirAnalysis
from snoocle_server.quality.attribution import Attribution, Fault
from snoocle_server.reconcile.providers import LLMResponse
from snoocle_server.schema import ChordPlacement, Line
from snoocle_server.store.memory import InMemorySongRepository


def _candidate() -> CandidateSource:
    return CandidateSource(
        sourceId="sheet-a",
        retrievedAt="2026-08-03T00:00:00Z",
        confidence=0.95,
        parseConfidence=0.95,
        coverage="full-song",
        sectionStarts=[SectionStart(name="Verse", startLineIndex=0)],
        lines=[
            Line(
                lineIndex=index,
                lyrics=f"private lyric {index}",
                chordPlacements=[ChordPlacement(charIndex=0, chord="C")],
            )
            for index in range(4)
        ],
    )


def _mir(chord: str = "C") -> MirAnalysis:
    return MirAnalysis(
        engines={"beats": "test", "chords": "test", "structure": "test"},
        duration_seconds=16,
        key="C major",
        chords=[
            ChordSegment(start=index * 4, end=(index + 1) * 4, chord=chord)
            for index in range(4)
        ],
    )


def _process_with_fault(fault: Fault) -> DeterministicProcessResult:
    candidate = _candidate()
    mir = _mir("G")
    song = build_song_from_candidate(
        candidate, song_id="artist--title", title="Title", artist="Artist"
    )
    alignment = align_song_deterministically_service(song, mir, candidates=[candidate])
    alignment.quality = dataclasses.replace(
        alignment.quality,
        attribution=Attribution(
            fault=fault,
            actionable=fault is Fault.MODEL,
            reason="test attribution",
        ),
    )
    alignment.conflict_packet["conflicts"] = [
        {
            "type": "chord_identity",
            "lineId": "L0",
            "placementId": "P0",
            "lineIndex": 0,
            "charIndex": 0,
            "existingChord": "C",
            "candidateEvidence": ["C"],
            "mirEvidence": [{"start": 0, "end": 4, "chord": "G"}],
            "confidence": 0.2,
            "reasons": ["mir disagreement"],
        }
    ]
    return DeterministicProcessResult(
        status="needs_review",
        reason="quality_gate_failed",
        song_id=song.id,
        observations=list(alignment.observations),
        cache={"audio": "not_applicable", "mir": "not_applicable", "discovery": "not_applicable"},
        alignment=alignment,
        mir=mir,
        candidates=(candidate,),
    )


def test_only_actionable_model_faults_are_patch_eligible():
    assert model_conflict_is_actionable(_process_with_fault(Fault.MODEL)) is True
    for fault in (Fault.SOURCE, Fault.AUDIO, Fault.UNKNOWN, Fault.NONE):
        assert model_conflict_is_actionable(_process_with_fault(fault)) is False


@pytest.mark.parametrize("field", ["lyrics", "songJson", "beats", "provenance", "songSchema", "sourceUrl"])
def test_compact_packet_rejects_forbidden_content(field):
    packet = _process_with_fault(Fault.MODEL).alignment.conflict_packet
    packet[field] = "forbidden"
    with pytest.raises(AgentPolicyError):
        validate_compact_conflict_packet(packet)


def test_bounded_patch_receives_only_conflicts_and_applies_locally(monkeypatch):
    process = _process_with_fault(Fault.MODEL)
    captured = {}

    class FakeProvider:
        name = "openai"
        wants_context = False

        def complete(self, system, turns, model=None, max_tokens=None, audio=None):
            captured.update(system=system, turns=turns, max_tokens=max_tokens)
            return LLMResponse(
                text=json.dumps(
                    {
                        "ops": [
                            {
                                "op": "replace_chord",
                                "lineIndex": 0,
                                "charIndex": 0,
                                "from": "C",
                                "to": "G",
                                "reason": "MIR evidence",
                            }
                        ]
                    }
                ),
                provider="openai",
                model="test-model",
                usage={"input_tokens": 10, "output_tokens": 5},
            )

    monkeypatch.setattr(deterministic_policy, "provider_preflight", lambda name: None)
    monkeypatch.setattr(deterministic_policy, "get_provider", lambda name: FakeProvider())
    outcome = run_bounded_agent_patch(process, provider_name="openai")

    serialized_input = json.dumps(captured["turns"]).casefold()
    assert "private lyric" not in serialized_input
    assert "provenance" not in serialized_input
    assert '"song"' not in serialized_input
    assert captured["max_tokens"] == 2_048
    assert outcome.alignment.song.lines[0].lyrics == "private lyric 0"
    assert outcome.alignment.song.lines[0].chordPlacements[0].chord == "G"
    assert len(outcome.applied_operations) == 1
    assert outcome.observation.output_summary["modelCalls"] == 1


def test_invalid_policy_is_structured():
    with pytest.raises(AgentPolicyError, match="never, unresolved_only, always"):
        deterministic_policy.resolve_agent_policy("sometimes")


@pytest.mark.parametrize(
    "fault",
    [Fault.SOURCE, Fault.AUDIO, Fault.UNKNOWN, Fault.NONE],
)
def test_pipeline_never_invokes_agent_for_non_model_faults(monkeypatch, fault):
    process = _process_with_fault(fault)
    invoked = []
    monkeypatch.setattr(
        pipeline_mod,
        "process_song_deterministically_service",
        lambda **kwargs: process,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "run_bounded_agent_patch",
        lambda *args, **kwargs: invoked.append((args, kwargs)),
    )

    report = asyncio.run(
        pipeline_mod.run_pipeline_async(
            "Title",
            "Artist",
            agent_policy="unresolved_only",
            store=InMemorySongRepository(),
        )
    )

    assert invoked == []
    assert report.status == "needs_review"
    assert report.steps["store"] == "skipped (deterministic result needs review)"
    assert report.deterministic_result["totals"]["modelCalls"] == 0


def test_pipeline_never_policy_suppresses_actionable_model_patch(monkeypatch):
    process = _process_with_fault(Fault.MODEL)
    invoked = []
    monkeypatch.setattr(
        pipeline_mod,
        "process_song_deterministically_service",
        lambda **kwargs: process,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "run_bounded_agent_patch",
        lambda *args, **kwargs: invoked.append((args, kwargs)),
    )

    report = asyncio.run(
        pipeline_mod.run_pipeline_async(
            "Title",
            "Artist",
            agent_policy="never",
            store=InMemorySongRepository(),
        )
    )

    assert invoked == []
    assert report.status == "needs_review"
    assert report.steps["agent"] == "skipped (agent_policy=never)"
    assert report.deterministic_result["totals"]["modelCalls"] == 0


def test_valid_deterministic_result_stores_without_agent(monkeypatch):
    process = _process_with_fault(Fault.NONE)
    process.status = "completed"
    process.reason = None
    invoked = []
    repository = InMemorySongRepository()
    monkeypatch.setattr(
        pipeline_mod,
        "process_song_deterministically_service",
        lambda **kwargs: process,
    )
    monkeypatch.setattr(
        pipeline_mod,
        "run_bounded_agent_patch",
        lambda *args, **kwargs: invoked.append((args, kwargs)),
    )

    report = asyncio.run(
        pipeline_mod.run_pipeline_async(
            "Title",
            "Artist",
            agent_policy="unresolved_only",
            store=repository,
        )
    )

    assert invoked == []
    assert report.status == "completed"
    assert report.stored_version is not None
    assert repository.get("artist--title").id == "artist--title"
    assert report.deterministic_result["totals"]["modelCalls"] == 0
