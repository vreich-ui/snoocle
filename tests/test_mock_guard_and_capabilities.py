from __future__ import annotations

import json

import pytest

from snoocle_server import mcp_server
from snoocle_server.reconcile.mock_reconciler import reconcile_deterministically
from snoocle_server.schema import Song
from snoocle_server.store.memory import InMemorySongRepository
from snoocle_server.test_output import TestOutputRejectedError, require_test_output_opt_in


def _mock_song() -> Song:
    return reconcile_deterministically(
        title="Test Song",
        artist="Test Artist",
        song_id="test-artist--test-song",
        youtube_video_id=None,
        candidates=[],
        mir=None,
    )


def test_mock_output_is_tainted_and_requires_store_opt_in():
    song = _mock_song()
    assert song.testOnly is True
    with pytest.raises(TestOutputRejectedError):
        require_test_output_opt_in(song, False)
    require_test_output_opt_in(song, True)


def test_mcp_save_and_read_only_diagnostic(monkeypatch):
    store = InMemorySongRepository()
    monkeypatch.setattr(mcp_server, "get_store", lambda: store)
    song = _mock_song()

    rejected = mcp_server.save_song(song.model_dump_json())
    assert rejected["error"]["code"] == "test_output_not_allowed"
    saved = mcp_server.save_song(song.model_dump_json(), allow_test_output=True)
    assert saved["song_id"] == song.id

    diagnostic = mcp_server.diagnose_mock_songs()
    assert diagnostic["readOnly"] is True
    assert diagnostic["songs"][0]["songId"] == song.id
    assert "testOnly" in diagnostic["songs"][0]["reasons"]


def test_list_capabilities_covers_every_registered_tool():
    result = mcp_server.list_capabilities()
    registered = set(mcp_server.mcp._tool_manager._tools)
    described = {entry["name"] for entry in result["tools"]}
    assert described == registered
    assert result["toolCount"] == result["coveredToolCount"] == len(registered)
    for entry in result["tools"]:
        assert entry["group"]
        assert entry["execution"]
        assert entry["networkAccess"]
        assert entry["persistence"]
        assert entry["inputType"]
        assert entry["outputType"]
        assert entry["cacheBehavior"]
        assert entry["costClass"]
