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
