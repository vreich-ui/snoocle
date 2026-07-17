"""The in-process "anthropic-agent" reconciliation provider.

The agent loop runs INSIDE the server (Anthropic SDK + server-side search +
local chord-sheet/MIR tools). These tests drive it with a fully faked Anthropic
client — no network, no real `anthropic` objects — by scripting responses as
`types.SimpleNamespace` blocks and monkeypatching
`AnthropicAgentProvider._create_client`. They verify:
- the happy path flows a tool call and returns a validated Song,
- repair rounds continue the SAME conversation (full tool history preserved),
- a local tool that errors is fed back as an is_error tool_result,
- an unconfigured key and an over-long loop fail with clear ProviderErrors,
- provider_capabilities() tracks the key.
"""

from __future__ import annotations

import json
import pathlib
import types

import pytest

from snoocle_server.config import settings
from snoocle_server.mir.base import Beat, ChordSegment, MirAnalysis, StructureSegment
from snoocle_server.reconcile import reconcile
from snoocle_server.reconcile import anthropic_agent as agent_mod
from snoocle_server.reconcile.anthropic_agent import AnthropicAgentProvider
from snoocle_server.reconcile.providers import ProviderError, provider_capabilities

# The canonical valid Song document (same fixture the MCP-agent tests use).
from tests.fake_agent_mcp import _SONG

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"


# --- fake Anthropic client -------------------------------------------------


def _text(text: str):
    return types.SimpleNamespace(type="text", text=text)


def _tool_use(tool_id: str, name: str, tool_input: dict):
    return types.SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def _response(stop_reason: str, content: list, in_tok: int = 100, out_tok: int = 40):
    return types.SimpleNamespace(
        stop_reason=stop_reason,
        content=content,
        usage=types.SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
    )


class _FakeClient:
    """Returns scripted responses from a SHARED queue (so repair rounds — which
    create a fresh client — keep consuming the same script in order). Snapshots
    the `messages` passed to each create() so tests can inspect them."""

    def __init__(self, queue: list, captured: dict):
        self._queue = queue
        self._captured = captured
        self.messages = self  # so client.messages.create(...) resolves here

    def create(self, **kwargs):
        self._captured.setdefault("calls", []).append(list(kwargs.get("messages") or []))
        self._captured["last_kwargs"] = kwargs
        if not self._queue:
            raise AssertionError("fake Anthropic client ran out of scripted responses")
        return self._queue.pop(0)


def _install(monkeypatch, queue: list) -> dict:
    """Monkeypatch _create_client to hand out fake clients backed by `queue`.

    Returns a `captured` dict that ends up holding the provider instance and the
    per-create() message snapshots.
    """
    captured: dict = {}

    def _create(self):
        captured["provider"] = self
        return _FakeClient(queue, captured)

    monkeypatch.setattr(AnthropicAgentProvider, "_create_client", _create)
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    return captured


def _mir() -> MirAnalysis:
    return MirAnalysis(
        engines={"beats": "madmom", "chords": "chord-cnn-lstm", "structure": "songformer"},
        duration_seconds=243.0,
        bpm=73.5,
        time_signature="4/4",
        key="C major",
        beats=[Beat(time=0.8, position=1), Beat(time=1.6, position=2)],
        chords=[
            ChordSegment(start=13.1, end=15.2, chord="C"),
            ChordSegment(start=15.2, end=17.3, chord="G"),
        ],
        sections=[StructureSegment(start=13.0, end=25.0, label="verse")],
    )


def _tool_result_user_messages(msgs: list) -> list:
    """User messages whose content is a list of tool_result blocks."""
    out = []
    for m in msgs:
        content = m.get("content")
        if m.get("role") == "user" and isinstance(content, list):
            if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                out.append(m)
    return out


# --- scenario 1: happy path -------------------------------------------------


def test_happy_path_tool_call_then_valid_song(monkeypatch):
    # fetch_chord_sheet reaches no network: fetch_page returns a real fixture
    # sheet, then the real extract/parse turn it into a candidate.
    sheet = (_FIXTURES / "sheet_over_lyrics.txt").read_text()
    fetched = {}

    def _fake_fetch_page(url: str) -> str:
        fetched["url"] = url
        return sheet

    monkeypatch.setattr(agent_mod, "fetch_page", _fake_fetch_page)

    queue = [
        _response("tool_use", [_tool_use("t1", "fetch_chord_sheet", {"url": "https://ex/let-it-be"})]),
        _response("end_turn", [_text(json.dumps(_SONG))]),
    ]
    captured = _install(monkeypatch, queue)

    result = reconcile(
        "Let It Be",
        "The Beatles",
        candidates=[],
        mir=_mir(),
        provider_name="anthropic-agent",
        youtube_video_id="QDYfEBY9NM4",
    )

    assert result.provider == "anthropic-agent"
    assert result.attempts == 1
    assert result.song.id == "the-beatles--let-it-be"
    assert result.song.metadata.title == "Let It Be"
    # accumulated token usage surfaced from response.usage
    assert result.usage.get("input_tokens", 0) > 0

    # the fetch tool was actually invoked with the model's URL
    assert fetched["url"] == "https://ex/let-it-be"
    # and its tool_result went back in a user message
    assert _tool_result_user_messages(captured["provider"]._messages)

    # request shape: consolidated effort + per-turn prefix caching — the two
    # wall-clock levers for the loop (see config.anthropic_agent_effort)
    kwargs = captured["last_kwargs"]
    assert kwargs["output_config"] == {"effort": settings.anthropic_agent_effort}
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    assert kwargs["thinking"] == {"type": "adaptive"}
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in kwargs


# --- scenario 2: repair round continues the same conversation ---------------


def test_repair_round_continues_same_conversation(monkeypatch):
    # Uses analyze_audio_window (no audio -> harmless error result, no network)
    # so the focus is purely the repair-round conversation continuation.
    queue = [
        _response("tool_use", [_tool_use("t1", "analyze_audio_window", {"start_seconds": 5, "end_seconds": 15})]),
        _response("end_turn", [_text(json.dumps({"bad": True}))]),  # attempt 1: invalid
        _response("end_turn", [_text(json.dumps(_SONG))]),          # attempt 2 (repair): valid
    ]
    captured = _install(monkeypatch, queue)

    result = reconcile(
        "Let It Be",
        "The Beatles",
        candidates=[],
        mir=_mir(),
        provider_name="anthropic-agent",
        youtube_video_id="QDYfEBY9NM4",
    )

    assert result.attempts == 2
    msgs = captured["provider"]._messages
    # same conversation carried across rounds: two assistant answers + the
    # round-1 tool history are all still present (not a reset per attempt)
    assistant_turns = [m for m in msgs if m.get("role") == "assistant"]
    assert len(assistant_turns) >= 2
    assert _tool_result_user_messages(msgs)
    # the final create() saw the full prior history (grew beyond a single turn)
    assert len(captured["calls"][-1]) >= 4


# --- scenario 3: local tool error is fed back, loop still completes ----------


def test_analyze_audio_window_without_audio_is_error_but_completes(monkeypatch):
    queue = [
        _response("tool_use", [_tool_use("t1", "analyze_audio_window", {"start_seconds": 10, "end_seconds": 20})]),
        _response("end_turn", [_text(json.dumps(_SONG))]),
    ]
    captured = _install(monkeypatch, queue)

    result = reconcile(
        "Let It Be",
        "The Beatles",
        candidates=[],
        mir=_mir(),
        provider_name="anthropic-agent",
        youtube_video_id="QDYfEBY9NM4",
        # no audio_path -> analyze_audio_window returns an error object
    )

    assert result.song.id == "the-beatles--let-it-be"
    tr_msgs = _tool_result_user_messages(captured["provider"]._messages)
    assert tr_msgs
    block = tr_msgs[0]["content"][0]
    assert block["is_error"] is True
    assert "no audio available" in block["content"]


# --- scenario 4: unconfigured key -------------------------------------------


def test_unconfigured_key_is_clear_error(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    with pytest.raises(ProviderError, match="SNOOCLE_ANTHROPIC_API_KEY"):
        reconcile(
            "Let It Be", "The Beatles", candidates=[], mir=_mir(),
            provider_name="anthropic-agent",
        )


# --- scenario 5: max turns exceeded -----------------------------------------


class _AlwaysToolUse:
    def __init__(self):
        self.messages = self

    def create(self, **kwargs):
        # never emits a final answer — always asks for another tool call
        return _response(
            "tool_use",
            [_tool_use("t", "analyze_audio_window", {"start_seconds": 0, "end_seconds": 5})],
        )


def test_max_turns_exceeded_raises(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(settings, "anthropic_agent_max_turns", 3)
    monkeypatch.setattr(AnthropicAgentProvider, "_create_client", lambda self: _AlwaysToolUse())

    with pytest.raises(ProviderError, match="exceeded max turns"):
        reconcile(
            "Let It Be", "The Beatles", candidates=[], mir=_mir(),
            provider_name="anthropic-agent",
        )


# --- scenario 6: capabilities track the key ---------------------------------


def test_anthropic_agent_in_provider_capabilities(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    caps = provider_capabilities()
    assert "anthropic-agent" in caps
    assert caps["anthropic-agent"]["configured"] is False
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    assert provider_capabilities()["anthropic-agent"]["configured"] is True
