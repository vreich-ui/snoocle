"""Run-trace capture, persistence, depth presets, and the run API.

The reconciler used to discard its process; these tests pin that it now records
a step trace (inputs + final for every provider, plus model/tool steps for the
in-process agent), persists it to the run store, and exposes it over the REST
API so the GUI can replay a run.
"""

from __future__ import annotations

import json
import types

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server import pipeline as pipeline_mod
from snoocle_server.api import app
from snoocle_server.config import settings
from snoocle_server.mir.base import MirAnalysis
from snoocle_server.reconcile import reconcile
from snoocle_server.reconcile import anthropic_agent as agent_mod
from snoocle_server.reconcile.anthropic_agent import AnthropicAgentProvider
from snoocle_server.reconcile.depth import resolve_depth
from snoocle_server.reconcile.trace import start_run
from snoocle_server.store.memory import InMemorySongRepository
from snoocle_server.store.runs import InMemoryRunRepository, reset_run_store
from tests.fake_agent_mcp import _SONG

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_stores(monkeypatch):
    store = InMemorySongRepository()
    runs = InMemoryRunRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: store)
    monkeypatch.setattr("snoocle_server.pipeline.get_store", lambda: store)
    monkeypatch.setattr("snoocle_server.pipeline.get_run_store", lambda: runs)
    monkeypatch.setattr("snoocle_server.store.runs.get_run_store", lambda: runs)
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    reset_run_store()
    return store, runs


# --- depth presets ----------------------------------------------------------


def test_depth_presets_expand_to_distinct_knobs():
    fast, standard, thorough = (resolve_depth(x) for x in ("fast", "standard", "thorough"))
    assert (fast.accuracy, fast.effort, fast.time_align) == ("fast", "low", False)
    assert (standard.accuracy, standard.effort) == ("standard", "medium")
    assert (thorough.accuracy, thorough.effort, thorough.time_align) == ("thorough", "high", True)
    # budgets grow with depth
    assert fast.max_web_search < thorough.max_web_search
    # unknown -> standard
    assert resolve_depth("bogus").name == "standard"
    assert resolve_depth(None).name == "standard"


# --- engine records a trace for any provider --------------------------------


def test_reconcile_records_inputs_and_final_trace_for_mock():
    recorder = start_run("anon--x", "mock", "standard")
    result = reconcile(
        "X", "Anon", candidates=[], mir=None, provider_name="mock", trace=recorder,
    )
    assert result.trace is not None
    kinds = [s.kind for s in result.trace.steps]
    assert kinds[0] == "inputs"
    assert kinds[-1] == "final"
    # the final step summarizes the produced song
    assert "lines" in result.trace.steps[-1].summary


# --- the in-process agent records model + tool steps ------------------------


def _text(t):
    return types.SimpleNamespace(type="text", text=t)


def _tool_use(i, name, inp):
    return types.SimpleNamespace(type="tool_use", id=i, name=name, input=inp)


def _response(stop, content):
    return types.SimpleNamespace(
        stop_reason=stop, content=content,
        usage=types.SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0),
    )


def test_agent_trace_captures_turns_and_tool_calls(monkeypatch):
    queue = [
        _response("tool_use", [_tool_use("t1", "analyze_audio_window",
                                         {"start_seconds": 5, "end_seconds": 15})]),
        _response("end_turn", [_text(json.dumps(_SONG))]),
    ]

    class _Fake:
        def __init__(self):
            self.messages = self

        def create(self, **kwargs):
            return queue.pop(0)

    monkeypatch.setattr(AnthropicAgentProvider, "_create_client", lambda self: _Fake())

    recorder = start_run("the-beatles--let-it-be", "anthropic-agent", "standard")
    result = reconcile(
        "Let It Be", "The Beatles", candidates=[],
        mir=MirAnalysis(engines={"chords": "test"}, duration_seconds=200.0, key="C major"),
        provider_name="anthropic-agent", youtube_video_id="QDYfEBY9NM4", trace=recorder,
    )
    kinds = [s.kind for s in result.trace.steps]
    assert "model" in kinds   # at least one model turn recorded
    assert "tool" in kinds    # the analyze_audio_window call recorded
    tool_step = next(s for s in result.trace.steps if s.kind == "tool")
    assert tool_step.detail["tool"] == "analyze_audio_window"


# --- MIR persistence: full timeline on the run, un-truncated ----------------


def test_attach_mir_survives_truncation_but_step_details_do_not():
    """A 200-segment chord timeline must persist whole on run.mir, while a
    same-size list inside a step detail still hits the 50-item cap."""
    from snoocle_server.mir.base import AnalyzedWindow, ChordSegment, MirAnalysis

    chords = [ChordSegment(start=float(i), end=float(i + 1), chord="C") for i in range(200)]
    mir = MirAnalysis(
        engines={"chords": "chord-cnn-lstm"}, duration_seconds=200.0, key="C major",
        chords=chords, analyzed_windows=[AnalyzedWindow(start=0.0, end=200.0)],
    )
    recorder = start_run("s--x", "anthropic-agent", "standard")
    recorder.attach_mir(mir.to_run_payload())
    recorder.step("inputs", "x", "y", detail={"chords": [{"chord": "C"} for _ in range(200)]})

    d = recorder.trace.to_dict()
    assert len(d["mir"]["chordTimeline"]) == 200          # full, un-truncated
    assert d["mir"]["analyzedWindows"] == [{"start": 0.0, "end": 200.0}]
    assert len(d["steps"][0]["detail"]["chords"]) == 50   # step detail still capped


def test_agent_run_records_mir_window_with_clamped_span(monkeypatch):
    """An analyze_audio_window probe lands on run.mirWindows. With no real audio
    the tool errors, so patch it to return a window+chords like a real run."""
    def _fake_window(audio_path, start_seconds, end_seconds):
        return {"window": {"start": 5.0, "end": 15.0},
                "chords": [{"start": 5.0, "end": 15.0, "chord": "G"}],
                "beats": 20, "bpm": 100.0}

    monkeypatch.setattr(agent_mod, "analyze_audio_window", _fake_window)
    queue = [
        _response("tool_use", [_tool_use("t1", "analyze_audio_window",
                                         {"start_seconds": 5, "end_seconds": 15})]),
        _response("end_turn", [_text(json.dumps(_SONG))]),
    ]

    class _Fake:
        def __init__(self): self.messages = self
        def create(self, **kwargs): return queue.pop(0)

    monkeypatch.setattr(AnthropicAgentProvider, "_create_client", lambda self: _Fake())
    recorder = start_run("the-beatles--let-it-be", "anthropic-agent", "standard")
    reconcile(
        "Let It Be", "The Beatles", candidates=[],
        mir=MirAnalysis(engines={"chords": "t"}, duration_seconds=200.0, key="C major"),
        provider_name="anthropic-agent", youtube_video_id="QDYfEBY9NM4",
        audio_path="/tmp/whatever.wav", trace=recorder,
    )
    windows = recorder.trace.mir_windows
    assert len(windows) == 1
    assert windows[0]["window"] == {"start": 5.0, "end": 15.0}
    assert windows[0]["chords"][0]["chord"] == "G"


def test_to_run_payload_caps_and_flags_truncation():
    from snoocle_server.mir.base import ChordSegment, MirAnalysis

    mir = MirAnalysis(
        engines={"chords": "c"}, duration_seconds=300.0,
        chords=[ChordSegment(start=float(i), end=float(i) + 1, chord="C") for i in range(5000)],
    )
    payload = mir.to_run_payload(max_chords=3000)
    assert len(payload["chordTimeline"]) <= 3000
    assert payload["truncated"] is True
    # a normal-size song is untouched
    small = MirAnalysis(engines={}, duration_seconds=200.0,
                        chords=[ChordSegment(start=0.0, end=1.0, chord="C")])
    assert small.to_run_payload()["truncated"] is False


def test_run_summary_strips_mir_payloads():
    from snoocle_server.store.runs import _summary

    slim = _summary({"runId": "r", "songId": "s", "steps": [1], "mir": {"big": 1},
                     "mirWindows": [1, 2], "status": "ok"})
    assert "steps" not in slim and "mir" not in slim and "mirWindows" not in slim
    assert slim["status"] == "ok"


# --- run store round-trips --------------------------------------------------


def test_run_store_saves_and_lists_newest_first():
    store = InMemoryRunRepository()
    store.save_run({"runId": "a", "songId": "s1", "startedAt": "2026-01-01T00:00:00Z", "steps": [1]})
    store.save_run({"runId": "b", "songId": "s1", "startedAt": "2026-01-02T00:00:00Z", "steps": [1, 2]})
    store.save_run({"runId": "c", "songId": "s2", "startedAt": "2026-01-03T00:00:00Z", "steps": []})

    got = store.get_run("b")
    assert got["runId"] == "b"
    listed = store.list_runs("s1")
    assert [r["runId"] for r in listed] == ["b", "a"]  # newest first
    assert "steps" not in listed[0]  # summaries omit the step list


# --- end-to-end via the API -------------------------------------------------


def test_analyze_returns_run_id_and_trace_is_fetchable():
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Offline", "artist": "Tester", "provider": "mock",
              "skipAudio": True, "analysisDepth": "fast"},
    )
    assert r.status_code == 200, r.text
    run_id = r.json()["runId"]
    assert run_id

    run = client.get(f"/v1/runs/{run_id}")
    assert run.status_code == 200
    body = run.json()
    assert body["depth"] == "fast"
    assert body["status"] == "ok"
    assert body["steps"]  # steps were recorded

    sid = r.json()["songId"]
    runs = client.get(f"/v1/songs/{sid}/runs")
    assert runs.status_code == 200
    assert any(x["runId"] == run_id for x in runs.json()["runs"])


def test_unknown_run_is_404():
    assert client.get("/v1/runs/does-not-exist").status_code == 404


def test_guidance_and_prior_song_are_accepted():
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Fixme", "artist": "Tester", "provider": "mock", "skipAudio": True,
              "guidance": "the chorus is G not C", "priorSong": {"any": "shape"}},
    )
    assert r.status_code == 200, r.text
