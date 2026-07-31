"""Compact reconcile output and deterministic effort-tier model routing."""

from __future__ import annotations

import asyncio
import json
import types

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server import mcp_server as mcp_mod
from snoocle_server.config import settings
from snoocle_server.discovery.models import CandidateSource
from snoocle_server.mir.base import ChordSegment, MirAnalysis
from snoocle_server.reconcile import reconcile
from snoocle_server.reconcile.anthropic_agent import AnthropicAgentProvider
from snoocle_server.reconcile.delta import (
    ReconcileDeltaError,
    apply_reconcile_delta,
    strip_postpass_schema,
)
from snoocle_server.reconcile.depth import resolve_depth
from snoocle_server.reconcile.engine import ReconcileError
from snoocle_server.reconcile.providers import LLMProvider, LLMResponse
from snoocle_server.reconcile.trace import start_run
from snoocle_server.schema import Song, song_json_schema


def _song() -> Song:
    return Song.model_validate({
        "id": "example--compact",
        "metadata": {
            "title": "Compact", "artist": "Example", "key": "C major", "bpm": 120,
        },
        "displayPreferences": {"capo": 0, "tuning": "standard"},
        "audio": {
            "durationSeconds": 180,
            "beats": [{"time": 1, "measure": 1, "beatInMeasure": 1}],
            "syncMap": [{"lineIndex": 0, "time": 12}],
        },
        "sections": [{
            "sectionIndex": 0, "name": "Verse", "kind": "verse",
            "startLineIndex": 0, "endLineIndex": 1, "startTime": 12, "endTime": 28,
        }],
        "lines": [
            {
                "lineIndex": 0, "lyrics": "first line", "timeSeconds": 12,
                "confidence": 0.9,
                "chordPlacements": [{
                    "charIndex": 0, "chord": "C", "timeSeconds": 12,
                    "confidence": 0.8, "beat": {"measure": 1, "beat": 1},
                }],
            },
            {
                "lineIndex": 1, "lyrics": "second line", "timeSeconds": 20,
                "chordPlacements": [{"charIndex": 0, "chord": "F"}],
            },
        ],
        "provenance": [],
    })


def test_golden_delta_matches_full_rewrite_and_logs_size():
    prior = _song()
    delta = {
        "lineChanges": [{
            "lineIndex": 1,
            "chordPlacements": [
                {"charIndex": 0, "chord": "G"},
                {"charIndex": 7, "chord": "C"},
            ],
        }],
        "sections": [{
            "sectionIndex": 0, "name": "Verse 1", "kind": "verse",
            "startLineIndex": 0, "endLineIndex": 1,
        }],
    }
    applied = apply_reconcile_delta(prior, delta, {"prior-song": ["first line", "second line"]})

    full_rewrite = prior.model_dump(mode="json")
    full_rewrite["lines"][1]["chordPlacements"] = delta["lineChanges"][0]["chordPlacements"]
    full_rewrite["sections"] = delta["sections"]
    expected = Song.model_validate(full_rewrite)
    assert applied.song == expected
    assert applied.patch_bytes < applied.full_bytes
    print(
        f"compact patch bytes={applied.patch_bytes}; full Song bytes={applied.full_bytes}; "
        f"ratio={applied.ratio:.3f}"
    )


def test_delta_rejects_model_owned_timing_fields():
    with pytest.raises(ReconcileDeltaError, match="post-pass-owned"):
        apply_reconcile_delta(
            _song(),
            {"lineChanges": [{
                "lineIndex": 0,
                "chordPlacements": [{
                    "charIndex": 0, "chord": "G", "timeSeconds": 12,
                }],
            }]},
            {},
        )


def test_first_reconcile_schema_excludes_mir_postpass_fields():
    schema = strip_postpass_schema(song_json_schema(), mir_present=True)
    definitions = schema["$defs"]
    assert not {"timeSeconds", "confidence", "beat"} & set(
        definitions["ChordPlacement"]["properties"]
    )
    assert not {"timeSeconds", "confidence"} & set(definitions["Line"]["properties"])
    assert {"startTime", "endTime"} <= set(definitions["Section"]["properties"])
    assert not {"syncMap", "beats"} & set(definitions["AudioInfo"]["properties"])
    assert "bpm" not in definitions["SongMetadata"]["properties"]


class _DeltaProvider(LLMProvider):
    name = "delta-test"
    default_model = "mid-test"
    emits_lyric_refs = True

    def __init__(self, payloads: list[dict]):
        self.payloads = payloads
        self.calls = 0

    def complete(self, system, turns, model=None, max_tokens=None, audio=None):
        payload = self.payloads[self.calls]
        self.calls += 1
        return LLMResponse(
            text=json.dumps(payload), provider=self.name, model=self.default_model,
            usage={"input_tokens": 10, "output_tokens": 5},
        )


def test_prior_version_uses_patch_and_invalid_patch_gets_one_repair(monkeypatch):
    provider = _DeltaProvider([
        {"lineChanges": [{"lineIndex": 99, "chordPlacements": []}]},
        {"lineChanges": [{"lineIndex": 1, "chordPlacements": [{"charIndex": 0, "chord": "G"}]}]},
    ])
    monkeypatch.setattr("snoocle_server.reconcile.engine.get_provider", lambda _=None: provider)
    recorder = start_run("example--compact", provider.name, "standard")
    result = reconcile(
        "Compact", "Example", [], None, provider_name=provider.name,
        prior_song=_song().model_dump(mode="json"), trace=recorder,
    )
    assert provider.calls == 2
    assert result.output_format == "patch"
    assert result.song.lines[1].chordPlacements[0].chord == "G"
    assert result.patch_size_vs_full["ratio"] < 1
    assert recorder.trace.output_format == "patch"
    assert recorder.trace.patch_size_vs_full["patchBytes"] > 0


def test_invalid_patch_fails_after_exactly_one_repair(monkeypatch):
    provider = _DeltaProvider([
        {"lineChanges": [{"lineIndex": 99}]},
        {"lineChanges": [{"lineIndex": 98}]},
        {"lineChanges": []},  # must never be consumed
    ])
    monkeypatch.setattr("snoocle_server.reconcile.engine.get_provider", lambda _=None: provider)
    with pytest.raises(ReconcileError, match="after 2 attempts"):
        reconcile(
            "Compact", "Example", [], None, provider_name=provider.name,
            prior_song=_song().model_dump(mode="json"),
        )
    assert provider.calls == 2


def _candidate(source_id: str, chords: list[str]) -> CandidateSource:
    return CandidateSource.model_validate({
        "sourceId": source_id,
        "lines": [{
            "lineIndex": 0,
            "lyrics": "",
            "chordPlacements": [
                {"charIndex": index, "chord": chord} for index, chord in enumerate(chords)
            ],
        }],
    })


def _capture_agent_turn(
    monkeypatch, *, depth: str, candidates: list[CandidateSource],
    mir: MirAnalysis | None = None, tool_then_final: bool = False,
):
    calls: list[dict] = []

    class _Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            if tool_then_final and len(calls) == 1:
                return types.SimpleNamespace(
                    stop_reason="tool_use",
                    content=[types.SimpleNamespace(
                        type="tool_use", id="window-1", name="analyze_audio_window",
                        input={"start_seconds": 0, "end_seconds": 10},
                    )],
                    usage=types.SimpleNamespace(input_tokens=100, output_tokens=10),
                    container=None,
                )
            return types.SimpleNamespace(
                stop_reason="end_turn",
                content=[types.SimpleNamespace(type="text", text="{}")],
                usage=types.SimpleNamespace(input_tokens=100, output_tokens=10),
                container=None,
            )

    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    provider = AnthropicAgentProvider()
    monkeypatch.setattr(
        provider, "_create_client", lambda: types.SimpleNamespace(messages=_Messages())
    )
    recorder = start_run("example--compact", provider.name, depth)
    provider.trace = recorder
    provider.context = {
        "song_id": "example--compact",
        "title": "Compact",
        "artist": "Example",
        "candidates": candidates,
        "mir": mir or MirAnalysis(engines={"chords": "fixture"}, duration_seconds=30),
        "song_schema": {"type": "object"},
        "ref_index": {},
        "depth": resolve_depth(depth),
        "output_format": "full",
    }
    provider.complete("ignored", [{"role": "user", "text": "ignored"}])
    return calls, recorder


def test_standard_uses_mid_model_without_a_real_conflict(monkeypatch):
    calls, recorder = _capture_agent_turn(
        monkeypatch, depth="standard", candidates=[_candidate("one", ["C", "G", "Am", "F"])]
    )
    assert calls[0]["model"] == settings.anthropic_agent_standard_model
    assert recorder.trace.opus_escalation == {"fired": False, "reason": None}


def test_standard_escalates_one_turn_only_for_unresolved_sheet_conflict(monkeypatch):
    calls, recorder = _capture_agent_turn(
        monkeypatch,
        depth="standard",
        candidates=[
            _candidate("one", ["C", "G", "Am", "F"]),
            _candidate("two", ["C", "Db", "D", "Eb"]),
        ],
        tool_then_final=True,
    )
    assert [call["model"] for call in calls] == [
        settings.anthropic_agent_model,
        settings.anthropic_agent_standard_model,
    ]
    assert recorder.trace.opus_escalation["fired"] is True
    assert "candidate sheets disagree" in recorder.trace.opus_escalation["reason"]
    assert recorder.trace.model_per_turn == [
        settings.anthropic_agent_model,
        settings.anthropic_agent_standard_model,
    ]


def test_standard_does_not_escalate_when_mir_breaks_the_sheet_tie(monkeypatch):
    mir = MirAnalysis(
        engines={"chords": "fixture"},
        duration_seconds=20,
        chords=[
            ChordSegment(start=index, end=index + 1, chord=chord)
            for index, chord in enumerate(["C", "G", "Am", "F"])
        ],
    )
    calls, recorder = _capture_agent_turn(
        monkeypatch,
        depth="standard",
        candidates=[
            _candidate("one", ["C", "G", "Am", "F"]),
            _candidate("two", ["C", "Db", "D", "Eb"]),
        ],
        mir=mir,
    )
    assert calls[0]["model"] == settings.anthropic_agent_standard_model
    assert recorder.trace.opus_escalation == {"fired": False, "reason": None}


def test_high_effort_uses_opus_throughout(monkeypatch):
    calls, recorder = _capture_agent_turn(
        monkeypatch, depth="thorough",
        candidates=[_candidate("one", ["C", "G", "Am", "F"])],
        tool_then_final=True,
    )
    assert [call["model"] for call in calls] == [
        settings.anthropic_agent_model, settings.anthropic_agent_model,
    ]
    assert recorder.trace.effort_level == "high"


def _pipeline_report():
    result = types.SimpleNamespace(
        song=_song(), provider="mock", model="mock-reconciler-v1", attempts=1,
        audio_attached=False, usage={}, output_format="full", patch_size_vs_full=None,
    )
    return types.SimpleNamespace(
        song_id="example--compact", steps={}, stored_version="v1", run_id="run-1",
        evidence_manifest={}, recording_suggestions=None, reconcile=result,
    )


def test_rest_effort_level_is_threaded_to_pipeline(monkeypatch):
    captured = {}

    async def fake_pipeline(*args, **kwargs):
        captured.update(kwargs)
        return _pipeline_report()

    monkeypatch.setattr(api_mod, "run_pipeline_async", fake_pipeline)
    response = TestClient(api_mod.app).post(
        "/v1/songs/analyze",
        json={
            "title": "Compact", "artist": "Example", "provider": "mock",
            "skipAudio": True, "effortLevel": "high",
        },
    )
    assert response.status_code == 200
    assert captured["effort_level"] == "high"


def test_mcp_effort_level_is_threaded_to_pipeline(monkeypatch):
    captured = {}

    async def fake_pipeline(*args, **kwargs):
        captured.update(kwargs)
        return _pipeline_report()

    monkeypatch.setattr(mcp_mod, "run_pipeline_async", fake_pipeline)
    result = asyncio.run(
        mcp_mod.analyze_and_store_song(
            title="Compact", artist="Example", provider="mock",
            skip_audio=True, effort_level="low",
        )
    )
    assert result["songId"] == "example--compact"
    assert captured["effort_level"] == "low"


def test_mcp_invalid_effort_level_is_rejected():
    with pytest.raises(ValueError, match="effortLevel"):
        asyncio.run(
            mcp_mod.analyze_and_store_song(
                title="Compact", artist="Example", provider="mock",
                skip_audio=True, effort_level="ultra",
            )
        )
