"""A stand-in for the external agent workspace's MCP server.

Exposes the one tool Snoocle's "agent" reconciliation provider calls. It
records the request it receives (so tests can assert on the integration
contract: title/artist/mediaUrl/timestamped chords) and returns a valid Song
JSON — the shape a real Agent SDK workspace with specialty agents would
produce. Run standalone: FAKE_AGENT_PORT=<port> FAKE_AGENT_CAPTURE=<path>
python -m tests.fake_agent_mcp
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fake-agent-workspace")

_SONG = {
    "id": "the-beatles--let-it-be",
    "metadata": {"title": "Let It Be", "artist": "The Beatles"},
    "audio": {
        "youtubeVideoId": "QDYfEBY9NM4",
        "syncMap": [{"lineIndex": 0, "time": 13.2}, {"lineIndex": 1, "time": 18.9}],
    },
    "sections": [
        {
            "sectionIndex": 0,
            "name": "Verse 1",
            "kind": "verse",
            "startLineIndex": 0,
            "endLineIndex": 1,
            "startTime": 13.0,
            "endTime": 25.0,
        }
    ],
    "lines": [
        {
            "lineIndex": 0,
            "lyrics": "When I find myself in times of trouble",
            "chordPlacements": [
                {"charIndex": 7, "chord": "C"},
                {"charIndex": 23, "chord": "G"},
                {"charIndex": 32, "chord": "Am"},
            ],
        },
        {
            "lineIndex": 1,
            "lyrics": "Mother Mary comes to me",
            "chordPlacements": [{"charIndex": 0, "chord": "F"}],
        },
    ],
    "provenance": [],
}


@mcp.tool()
def reconcile_song(
    request: dict,
    previousOutput: str | None = None,
    validationErrors: str | None = None,
) -> str:
    """Reconcile a song from MIR + text-source evidence into Song JSON."""
    capture = os.environ.get("FAKE_AGENT_CAPTURE")
    if capture:
        Path(capture).write_text(
            json.dumps(
                {
                    "request": request,
                    "previousOutput": previousOutput,
                    "validationErrors": validationErrors,
                }
            )
        )
    return json.dumps(_SONG)


@mcp.tool()
def node_execute(
    nodeId: str,
    executionMode: str = "openai",
    input: dict | None = None,
    dependencyOutputs: dict | None = None,
) -> dict:
    """CMS-Agent-style generic node runner (the SNOOCLE_AGENT_MCP_NODES mode).

    Appends one JSON line per call to FAKE_AGENT_CAPTURE so tests can assert
    the chain order, forwarded dependencyOutputs, and repair-round behavior.
    The node id containing 'reconciler' returns the Song; other nodes return
    stage artifacts.
    """
    capture = os.environ.get("FAKE_AGENT_CAPTURE")
    if capture:
        with Path(capture).open("a") as f:
            f.write(json.dumps({
                "tool": "node_execute",
                "nodeId": nodeId,
                "input": input,
                "dependencyOutputs": dependencyOutputs,
            }) + "\n")
    if "reconciler" in nodeId:
        output: dict = _SONG
    elif "compare" in nodeId:
        output = {"artifact": "snoocle_reconciliation_plan.v1", "summary": "ok",
                  "bestSourceId": "web-1", "rankedSources": [], "keyDecision": {}, "adjustments": []}
    else:
        output = {"artifact": "snoocle_song_sources.v1", "summary": "ok", "sources": []}
    # FAKE_AGENT_FORCE_MODE simulates a workspace that ignores the requested
    # executionMode (e.g. silently running its mock runner).
    reported_mode = os.environ.get("FAKE_AGENT_FORCE_MODE") or executionMode
    return {"ok": True, "data": {"execution": {
        "status": "completed",
        "executionMode": reported_mode,
        "nodes": [{"nodeId": nodeId, "status": "completed", "output": output}],
        "errors": [],
    }}}


def main() -> None:
    mcp.settings.host = "127.0.0.1"
    mcp.settings.port = int(os.environ["FAKE_AGENT_PORT"])
    mcp.settings.stateless_http = True
    mcp.settings.json_response = True
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
