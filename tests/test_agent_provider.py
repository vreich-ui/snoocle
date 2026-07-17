"""The "agent" reconciliation provider: Snoocle as an MCP CLIENT.

Reconciliation is delegated to an external agent workspace's MCP server.
These tests spawn a fake agent server (tests/fake_agent_mcp.py) over
streamable HTTP and verify the full integration contract:
- Snoocle sends title, artist, mediaUrl, and the TIMESTAMPED chord changes
- the returned Song JSON flows through the engine's schema validation and
  server-side finalization exactly like a direct-LLM response
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

from snoocle_server.config import settings
from snoocle_server.mir.base import Beat, ChordSegment, MirAnalysis, StructureSegment
from snoocle_server.reconcile import reconcile
from snoocle_server.reconcile.providers import ProviderError, provider_capabilities


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _fake_agent(capture_path):
    port = _free_port()
    env = {
        **os.environ,
        "FAKE_AGENT_PORT": str(port),
        "FAKE_AGENT_CAPTURE": str(capture_path),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.fake_agent_mcp"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        for _ in range(50):
            if proc.poll() is not None:
                raise RuntimeError(f"fake agent exited early: {proc.stdout.read()}")
            try:
                httpx.post(url, json={}, timeout=1.0)
                break
            except httpx.ConnectError:
                time.sleep(0.2)
        else:
            raise RuntimeError("fake agent never started listening")
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


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
            ChordSegment(start=17.3, end=19.4, chord="Am7"),
        ],
        sections=[StructureSegment(start=13.0, end=25.0, label="verse")],
    )


def test_agent_provider_calls_remote_mcp_and_returns_validated_song(tmp_path, monkeypatch):
    capture = tmp_path / "captured.json"
    with _fake_agent(capture) as url:
        monkeypatch.setattr(settings, "agent_mcp_url", url)
        result = reconcile(
            "Let It Be",
            "The Beatles",
            candidates=[],
            mir=_mir(),
            provider_name="agent",
            youtube_video_id="QDYfEBY9NM4",
        )

    assert result.provider == "agent"
    assert result.model == "mcp:reconcile_song"
    assert result.attempts == 1
    # engine finalization applied on top of the agent's document
    assert result.song.id == "the-beatles--let-it-be"
    assert result.song.provenance and result.song.provenance[-1].action == "reconciled"
    assert "agent" in result.song.provenance[-1].actor

    # the integration contract: what the agent workspace actually received
    sent = json.loads(capture.read_text())["request"]
    assert sent["title"] == "Let It Be"
    assert sent["artist"] == "The Beatles"
    assert sent["mediaUrl"] == "https://www.youtube.com/watch?v=QDYfEBY9NM4"
    assert sent["youtubeVideoId"] == "QDYfEBY9NM4"
    assert sent["chords"] == [
        {"start": 13.1, "end": 15.2, "chord": "C"},
        {"start": 15.2, "end": 17.3, "chord": "G"},
        {"start": 17.3, "end": 19.4, "chord": "Am7"},
    ]
    assert sent["mir"]["bpm"] == 73.5
    assert "songSchema" in sent


def test_agent_provider_explicit_media_url_wins(tmp_path, monkeypatch):
    capture = tmp_path / "captured.json"
    with _fake_agent(capture) as url:
        monkeypatch.setattr(settings, "agent_mcp_url", url)
        reconcile(
            "Let It Be",
            "The Beatles",
            candidates=[],
            mir=_mir(),
            provider_name="agent",
            youtube_video_id="QDYfEBY9NM4",
            media_url="https://media.example.com/recordings/let-it-be.mp4",
        )
    sent = json.loads(capture.read_text())["request"]
    assert sent["mediaUrl"] == "https://media.example.com/recordings/let-it-be.mp4"


def test_agent_provider_unconfigured_is_clear_error(monkeypatch):
    monkeypatch.setattr(settings, "agent_mcp_url", "")
    with pytest.raises(ProviderError, match="SNOOCLE_AGENT_MCP_URL"):
        reconcile(
            "Let It Be", "The Beatles", candidates=[], mir=_mir(), provider_name="agent"
        )


def test_agent_appears_in_provider_capabilities(monkeypatch):
    monkeypatch.setattr(settings, "agent_mcp_url", "")
    caps = provider_capabilities()
    assert caps["agent"]["configured"] is False
    monkeypatch.setattr(settings, "agent_mcp_url", "http://127.0.0.1:9/mcp")
    assert provider_capabilities()["agent"]["configured"] is True


NODES = "snoocle_source_search,snoocle_source_compare,snoocle_reconciler"


def _chain_calls(capture) -> list[dict]:
    return [json.loads(line) for line in capture.read_text().splitlines() if line.strip()]


def test_agent_node_chain_mode_runs_all_nodes_and_returns_song(tmp_path, monkeypatch):
    """SNOOCLE_AGENT_MCP_NODES drives the CMS-Agent node graph: each node runs
    via node_execute in order, outputs flow forward as dependencyOutputs, and
    the final node's output is validated as the Song."""
    capture = tmp_path / "captured.jsonl"
    with _fake_agent(capture) as url:
        monkeypatch.setattr(settings, "agent_mcp_url", url)
        monkeypatch.setattr(settings, "agent_mcp_nodes", NODES)
        result = reconcile(
            "Let It Be",
            "The Beatles",
            candidates=[],
            mir=_mir(),
            provider_name="agent",
            youtube_video_id="QDYfEBY9NM4",
        )

    assert result.provider == "agent"
    assert result.model == "mcp:" + NODES.replace(",", "+")
    assert result.song.id == "the-beatles--let-it-be"

    calls = _chain_calls(capture)
    assert [c["nodeId"] for c in calls] == NODES.split(",")
    # every node receives the Snoocle request; downstream nodes receive
    # upstream outputs as dependencyOutputs
    assert all(c["input"]["request"]["title"] == "Let It Be" for c in calls)
    assert calls[0]["dependencyOutputs"] is None
    assert set(calls[1]["dependencyOutputs"]) == {"snoocle_source_search"}
    assert set(calls[2]["dependencyOutputs"]) == {
        "snoocle_source_search", "snoocle_source_compare",
    }


def test_agent_node_chain_refuses_non_openai_execution(tmp_path, monkeypatch):
    """If the workspace ignores executionMode="openai" and runs its mock
    runner, the provider must refuse the stub output loudly — not let it fail
    obscurely (or worse, pass) downstream."""
    capture = tmp_path / "captured.jsonl"
    monkeypatch.setenv("FAKE_AGENT_FORCE_MODE", "mock")
    with _fake_agent(capture) as url:
        monkeypatch.setattr(settings, "agent_mcp_url", url)
        monkeypatch.setattr(settings, "agent_mcp_nodes", NODES)
        with pytest.raises(ProviderError, match="'mock' mode instead of 'openai'"):
            reconcile(
                "Let It Be", "The Beatles", candidates=[], mir=_mir(),
                provider_name="agent", youtube_video_id="QDYfEBY9NM4",
            )


def test_agent_node_chain_repair_round_reruns_only_final_node(tmp_path, monkeypatch):
    """When the final node's Song fails validation, the repair round re-runs
    ONLY the reconciler node (with previousOutput/validationErrors) — upstream
    evidence gathering is not repeated."""
    capture = tmp_path / "captured.jsonl"
    with _fake_agent(capture) as url:
        monkeypatch.setattr(settings, "agent_mcp_url", url)
        monkeypatch.setattr(settings, "agent_mcp_nodes", NODES)

        from snoocle_server.reconcile.providers import AgentMcpProvider

        provider = AgentMcpProvider()
        provider.context = {
            "title": "Let It Be", "artist": "The Beatles", "song_id": "the-beatles--let-it-be",
            "youtube_video_id": None, "media_url": None, "candidates": [], "mir": None,
            "song_schema": {},
        }
        turns = [{"role": "user", "text": "reconcile"}]
        provider.complete("system", turns)
        turns += [
            {"role": "assistant", "text": "{\"bad\": true}"},
            {"role": "user", "text": "validation errors: metadata is required"},
        ]
        provider.complete("system", turns)

    calls = _chain_calls(capture)
    # 3 first-round calls + exactly 1 repair call (the reconciler only)
    assert [c["nodeId"] for c in calls] == NODES.split(",") + ["snoocle_reconciler"]
    repair = calls[-1]
    assert repair["input"]["previousOutput"] == "{\"bad\": true}"
    assert "metadata is required" in repair["input"]["validationErrors"]
    assert set(repair["dependencyOutputs"]) == {
        "snoocle_source_search", "snoocle_source_compare",
    }
