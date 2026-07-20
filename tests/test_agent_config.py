"""Runtime agent programming: config model, prompt assembly, precedence, and
the REST + MCP surfaces that edit it.

The load-bearing invariant: an operator can reshape the agent's instructions and
tooling, but the OUTPUT CONTRACT is always appended and never editable.
"""

from __future__ import annotations

import json
import types

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server.api import app
from snoocle_server.config import settings
from snoocle_server.mir.base import MirAnalysis
from snoocle_server.reconcile import reconcile
from snoocle_server.reconcile.agent_config import AgentConfig, config_version
from snoocle_server.reconcile.anthropic_agent import (
    _OUTPUT_CONTRACT,
    _PROMPT_THEORY,
    AnthropicAgentProvider,
    _build_tools,
    build_system_blocks,
)
from snoocle_server.reconcile.trace import start_run
from snoocle_server.store.agent_config import InMemoryAgentConfigStore, reset_agent_config_store
from snoocle_server.store.memory import InMemorySongRepository
from tests.fake_agent_mcp import _SONG


# --- config model -----------------------------------------------------------


def test_config_validation_rejects_bad_effort_and_tools():
    with pytest.raises(ValueError):
        AgentConfig(effort="ludicrous")
    with pytest.raises(ValueError):
        AgentConfig(disabled_tools=["not_a_tool"])
    # valid ones pass
    AgentConfig(effort="high", disabled_tools=["web_fetch"], max_turns=8)


def test_is_default_and_version_fingerprint():
    assert AgentConfig().is_default()
    a = AgentConfig(instructions_extra="prefer barre chords")
    assert not a.is_default()
    # version is stable and ignores provenance fields
    assert config_version(a) == config_version(
        AgentConfig(instructions_extra="prefer barre chords", updated_at="x", source="y")
    )
    assert config_version(a) != config_version(AgentConfig())


# --- prompt assembly --------------------------------------------------------


def _text(cfg):
    return build_system_blocks(cfg)[0]["text"]


def test_output_contract_always_present_even_with_override():
    over = _text(AgentConfig(instructions_override="Only speak in limericks."))
    assert "Only speak in limericks." in over
    assert _OUTPUT_CONTRACT in over  # contract survives an override


def test_extra_appended_and_sections_swapped():
    extra = _text(AgentConfig(instructions_extra="ZZ-EXTRA-ZZ"))
    assert "ZZ-EXTRA-ZZ" in extra
    assert _PROMPT_THEORY in extra  # default theory still there

    swapped = _text(AgentConfig(theory_rules="MY-THEORY-RULES"))
    assert "MY-THEORY-RULES" in swapped
    assert _PROMPT_THEORY not in swapped  # replaced

    assert _OUTPUT_CONTRACT in _text(None)  # default path


def test_build_tools_drops_disabled():
    names = {t["name"] for t in _build_tools(2, 3, frozenset({"web_fetch", "analyze_audio_window"}))}
    assert "web_fetch" not in names and "analyze_audio_window" not in names
    assert "web_search" in names and "fetch_chord_sheet" in names


# --- precedence (config beats depth beats settings) -------------------------


def _mir():
    return MirAnalysis(engines={"chords": "t"}, duration_seconds=200.0, key="C major")


def test_config_overrides_reach_the_request(monkeypatch):
    """A stored config sets effort, model, and disables a tool — all must land
    in the messages.create call and the system prompt."""
    captured = {}

    class _Fake:
        def __init__(self): self.messages = self
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return types.SimpleNamespace(
                stop_reason="end_turn",
                content=[types.SimpleNamespace(type="text", text=json.dumps(_SONG))],
                usage=types.SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0),
            )

    monkeypatch.setattr(AnthropicAgentProvider, "_create_client", lambda self: _Fake())
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")

    store = InMemoryAgentConfigStore()
    store.set(AgentConfig(effort="high", model="claude-opus-4-8",
                          instructions_extra="XX-RULE-XX",
                          disabled_tools=["web_fetch"]).model_dump())
    monkeypatch.setattr("snoocle_server.store.agent_config.get_agent_config_store", lambda: store)

    recorder = start_run("the-beatles--let-it-be", "anthropic-agent", "standard")
    reconcile("Let It Be", "The Beatles", candidates=[], mir=_mir(),
              provider_name="anthropic-agent", youtube_video_id="QDYfEBY9NM4", trace=recorder)

    kw = captured["kwargs"]
    assert kw["output_config"] == {"effort": "high"}      # cfg effort
    assert kw["model"] == "claude-opus-4-8"               # cfg model
    assert "XX-RULE-XX" in kw["system"][0]["text"]        # cfg extra instructions
    assert "web_fetch" not in {t["name"] for t in kw["tools"]}  # cfg disabled tool
    # and the run is stamped with the config fingerprint
    assert recorder.trace.config_version is not None


# --- REST endpoints ---------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(api_mod, "get_store", lambda: InMemorySongRepository())
    store = InMemoryAgentConfigStore()
    monkeypatch.setattr("snoocle_server.store.agent_config.get_agent_config_store", lambda: store)
    reset_agent_config_store()
    return TestClient(app)


def test_rest_requires_token(client, monkeypatch):
    monkeypatch.setattr(settings, "api_token", "")
    assert client.get("/v1/config/agent").status_code == 409


def test_rest_roundtrip_and_reset(client, monkeypatch):
    monkeypatch.setattr(settings, "api_token", "tok")
    h = {"Authorization": "Bearer tok"}
    assert client.get("/v1/config/agent", headers=h).json()["isDefault"] is True

    put = client.put("/v1/config/agent", headers=h,
                     json={"instructions_extra": "prefer open chords", "effort": "high"})
    assert put.status_code == 200 and put.json()["configVersion"]
    got = client.get("/v1/config/agent", headers=h).json()
    assert got["config"]["instructions_extra"] == "prefer open chords"
    assert got["isDefault"] is False
    # defaults block is present so the GUI can show placeholders + the locked contract
    assert "lockedOutputContract" in got["defaults"]

    assert client.put("/v1/config/agent", headers=h, json={"effort": "nope"}).status_code == 422
    assert client.delete("/v1/config/agent", headers=h).status_code == 200
    assert client.get("/v1/config/agent", headers=h).json()["isDefault"] is True


# --- MCP tool functions (callable without a live server / ffmpeg) -----------


def test_mcp_agent_config_tools_roundtrip(monkeypatch):
    from snoocle_server import mcp_server

    store = InMemoryAgentConfigStore()
    monkeypatch.setattr("snoocle_server.store.agent_config.get_agent_config_store", lambda: store)
    monkeypatch.setattr(settings, "api_token", "tok")

    assert mcp_server.get_agent_config()["isDefault"] is True
    out = mcp_server.set_agent_config(json.dumps({"effort": "low", "max_turns": 5}))
    assert out["status"] == "stored"
    assert mcp_server.get_agent_config()["config"]["effort"] == "low"
    mcp_server.reset_agent_config()
    assert mcp_server.get_agent_config()["isDefault"] is True


def test_mcp_set_agent_config_refuses_without_token(monkeypatch):
    from snoocle_server import mcp_server
    from snoocle_server.authz import AdminAuthNotConfigured

    monkeypatch.setattr(settings, "api_token", "")
    with pytest.raises(AdminAuthNotConfigured):
        mcp_server.set_agent_config("{}")
