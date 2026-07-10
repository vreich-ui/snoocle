"""Single-service topology: one FastAPI app serves REST + the MCP endpoint.

Proves the combined app is the sole writer to the store — a song written via
the REST API is immediately visible through the MCP tools on the SAME process,
so there is no cross-service write race or read staleness.
"""

import json
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

pytestmark = pytest.mark.anyio


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def combined_server(tmp_path):
    port = _free_port()
    env = {
        **os.environ,
        "SNOOCLE_STORE_DIR": str(tmp_path / "store"),
        "SNOOCLE_MCP_TRUST_PROXY": "true",  # single service behind IAM in deploy
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "snoocle_server.api:app",
         "--host", "127.0.0.1", "--port", str(port)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(80):
            if proc.poll() is not None:
                raise RuntimeError(f"combined app exited early: {proc.stdout.read()}")
            try:
                if httpx.get(f"{base}/healthz", timeout=1.0).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.3)
        else:
            raise RuntimeError("combined app never became healthy")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


async def test_rest_write_visible_via_mcp_on_same_service(combined_server):
    base = combined_server

    # REST surface up, and it advertises the embedded MCP endpoint.
    health = httpx.get(f"{base}/healthz").json()
    assert health["status"] == "ok"
    assert health["mcpEndpoint"] == "/mcp"

    # Write a song through the REST API.
    song = {
        "id": "combined--proof",
        "metadata": {"title": "Proof", "artist": "Combined"},
        "lines": [{"lineIndex": 0, "lyrics": "la", "chordPlacements": [{"charIndex": 0, "chord": "C"}]}],
        "provenance": [{"timestamp": "2026-07-09T00:00:00Z", "actor": "test", "action": "created"}],
    }
    rw = httpx.post(f"{base}/v1/songs/combined--proof",
                    json={"song": song, "message": "via REST API"}, timeout=30)
    assert rw.status_code == 200, rw.text

    # Read it back through the MCP endpoint on the SAME service — sole writer,
    # so no cross-service race or staleness: it's immediately visible.
    async with streamablehttp_client(f"{base}/mcp") as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            assert len(tools.tools) >= 15  # full step-scoped tool surface

            got = await s.call_tool("get_song", {"song_id": "combined--proof"})
            assert json.loads(got.content[0].text)["id"] == "combined--proof"

            listed = await s.call_tool("list_songs", {})
            assert "combined--proof" in json.loads(listed.content[0].text)["songs"]
