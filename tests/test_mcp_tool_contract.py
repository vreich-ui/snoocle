"""The MCP surface is a fail-closed, GUI-consumable product contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from snoocle_server import mcp_server
from snoocle_server.tool_contract import (
    TOOL_CONTRACTS,
    TOOL_CONTRACT_VERSION,
    TOOL_META_KEY,
    ToolContractError,
    apply_tool_contract,
    validate_registered_tool_contract,
)


def _registered() -> dict:
    return mcp_server.mcp._tool_manager._tools


def test_every_registered_tool_has_exactly_one_contract_entry():
    registered = _registered()
    assert set(TOOL_CONTRACTS) == set(registered)

    for name, tool in registered.items():
        contract = TOOL_CONTRACTS[name]
        assert tool.title == contract.title
        assert tool.annotations == contract.annotations()
        if "meta" in type(tool).model_fields:
            assert tool.meta == {TOOL_META_KEY: contract.to_wire()}
        else:
            assert not hasattr(tool, "meta")


def test_contract_rejects_a_new_unclassified_registered_tool():
    fake_mcp = SimpleNamespace(
        _tool_manager=SimpleNamespace(_tools={**_registered(), "new_unreviewed_tool": object()})
    )

    with pytest.raises(ToolContractError, match="unclassified registered tools: new_unreviewed_tool"):
        validate_registered_tool_contract(fake_mcp)


def test_contract_rejects_a_stale_entry(monkeypatch):
    monkeypatch.setitem(TOOL_CONTRACTS, "removed_tool", TOOL_CONTRACTS["server_status"])

    with pytest.raises(ToolContractError, match="contract entries without registered tools: removed_tool"):
        validate_registered_tool_contract(mcp_server.mcp)


def test_contract_supports_fastmcp_versions_without_tool_meta():
    class LegacyTool:
        __slots__ = ("title", "annotations")
        model_fields = {"title": object(), "annotations": object()}

        def __init__(self):
            self.title = None
            self.annotations = None

    tools = {name: LegacyTool() for name in TOOL_CONTRACTS}
    legacy_mcp = SimpleNamespace(_tool_manager=SimpleNamespace(_tools=tools))

    apply_tool_contract(legacy_mcp)

    assert tools["get_song"].title == "Get Song"
    assert tools["get_song"].annotations == TOOL_CONTRACTS["get_song"].annotations()
    assert not hasattr(tools["get_song"], "meta")


@pytest.mark.anyio
async def test_tools_list_publishes_standard_annotations_and_namespaced_metadata():
    listed = {tool.name: tool for tool in await mcp_server.mcp.list_tools()}
    assert set(listed) == set(TOOL_CONTRACTS)

    for name, contract in TOOL_CONTRACTS.items():
        protocol_tool = listed[name]
        assert protocol_tool.title == contract.title
        assert protocol_tool.annotations == contract.annotations()
        if "meta" in type(protocol_tool).model_fields:
            assert protocol_tool.meta == {TOOL_META_KEY: contract.to_wire()}
        else:
            assert not hasattr(protocol_tool, "meta")


def test_capability_catalog_preserves_legacy_fields_and_adds_the_full_contract():
    result = mcp_server.list_capabilities()
    entries = {entry["name"]: entry for entry in result["tools"]}

    assert result["contractVersion"] == TOOL_CONTRACT_VERSION
    assert result["toolCount"] == result["coveredToolCount"] == len(_registered())
    assert set(entries) == set(_registered())
    for name, tool in _registered().items():
        entry = entries[name]
        contract = TOOL_CONTRACTS[name]
        for legacy_key in (
            "group",
            "execution",
            "networkAccess",
            "persistence",
            "inputType",
            "outputType",
            "cacheBehavior",
            "costClass",
        ):
            assert entry[legacy_key]
        assert entry["inputType"] == tool.parameters
        assert entry["toolContract"] == contract.to_wire()
        assert entry["browserSafety"] == contract.browser_safety
        assert entry["inputArtifactKinds"] == list(contract.input_artifact_kinds)
        assert entry["outputArtifactKinds"] == list(contract.output_artifact_kinds)
        assert entry["accessMode"] == contract.access_mode
        assert entry["readOnly"] is contract.read_only
        assert entry["destructive"] is contract.destructive
        assert entry["networkAccessKinds"] == list(contract.network_access)
        assert entry["modelUse"] == contract.model_use
        assert entry["persistenceKinds"] == list(contract.persistence)
        assert entry["expectedDuration"] == contract.expected_duration
        assert entry["specializedRenderer"] == contract.specialized_renderer


def test_representative_gui_safety_and_presentation_decisions_are_pinned():
    get_song = TOOL_CONTRACTS["get_song"]
    assert get_song.read_only is True
    assert get_song.destructive is False
    assert get_song.specialized_renderer == "song"
    assert get_song.output_artifact_kinds == ("song",)

    clear_notes = TOOL_CONTRACTS["clear_song_notes"]
    assert clear_notes.browser_safety == "confirmation_required"
    assert clear_notes.access_mode == "write"
    assert clear_notes.destructive is True

    reconcile = TOOL_CONTRACTS["reconcile_song"]
    assert reconcile.model_use == "required"
    assert reconcile.open_world is True
    assert "run_trace_write" in reconcile.persistence

    local_mir = TOOL_CONTRACTS["analyze_full_track_mir"]
    assert local_mir.browser_safety == "server_filesystem_restricted"
    assert local_mir.expected_duration == "minutes"
    assert local_mir.network_access == ()

    align = TOOL_CONTRACTS["align_song_deterministically"]
    assert align.model_use == "none"
    assert "song_version_optional" in align.persistence
    assert align.specialized_renderer == "song"
