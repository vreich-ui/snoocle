"""End-to-end MCP test: real client <-> real server over stdio.

Spawns `snoocle_server.mcp_server` as a subprocess and drives it with the
official MCP client, proving the tool surface is callable from any
MCP-compatible client (acceptance step 6).
"""

import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import time

import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

EXPECTED_TOOLS = {
    "discover_song",
    "acquire_audio",
    "analyze_audio",
    "reconcile_song",
    "analyze_and_store_song",
    "list_songs",
    "get_song",
    "list_song_versions",
    "diff_song_versions",
    "save_song",
    "convert_audio",
    "trim_audio",
    "normalize_audio",
    "probe_audio",
    "server_status",
    "get_song_schema",
}


@pytest.fixture()
def tone_wav_b64(tmp_path):
    p = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-c:a", "pcm_s16le", str(p)],
        check=True, capture_output=True,
    )
    return base64.b64encode(p.read_bytes()).decode()


@pytest.mark.anyio
async def test_mcp_tools_over_stdio(tone_wav_b64, tmp_path):
    import os

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "snoocle_server.mcp_server"],
        env={
            "SNOOCLE_STORE_DIR": str(tmp_path / "store"),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/root"),
        },
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert EXPECTED_TOOLS <= names, f"missing tools: {EXPECTED_TOOLS - names}"

            # distinct tools, not a monolith
            assert len(EXPECTED_TOOLS) >= 15

            status = await session.call_tool("server_status", {})
            payload = json.loads(status.content[0].text)
            assert payload["ffmpeg"] is True
            assert set(payload["llmProviders"]) == {"anthropic", "openai", "gemini", "mock"}

            # deterministic audio utility over MCP with base64 fallback (no AI)
            trimmed = await session.call_tool(
                "trim_audio",
                {
                    "start_seconds": 0.5,
                    "end_seconds": 1.5,
                    "input_base64": tone_wav_b64,
                    "input_format": "wav",
                    "return_base64": True,
                },
            )
            out = json.loads(trimmed.content[0].text)
            assert abs(out["probe"]["duration_seconds"] - 1.0) < 0.05
            assert len(base64.b64decode(out["base64"])) > 1000

            schema = await session.call_tool("get_song_schema", {})
            assert "chordPlacements" in schema.content[0].text


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.anyio
async def test_mcp_tools_over_streamable_http(tmp_path):
    """Same tool surface, served over HTTP instead of stdio — the transport
    Cloud Run (and any remote MCP client) actually needs, gated in
    deployment by Cloud Run IAM rather than app-level auth."""
    port = _free_port()
    env = {
        **os.environ,
        "SNOOCLE_MCP_TRANSPORT": "streamable-http",
        "SNOOCLE_MCP_PORT": str(port),
        "SNOOCLE_STORE_DIR": str(tmp_path / "store"),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "snoocle_server.mcp_server"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        for _ in range(50):
            if proc.poll() is not None:
                raise RuntimeError(f"server exited early: {proc.stdout.read()}")
            try:
                httpx.post(url, json={}, timeout=1.0)
                break
            except httpx.ConnectError:
                time.sleep(0.2)
        else:
            raise RuntimeError("server never started listening")

        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert EXPECTED_TOOLS <= names

                status = await session.call_tool("server_status", {})
                payload = json.loads(status.content[0].text)
                assert payload["ffmpeg"] is True

                schema = await session.call_tool("get_song_schema", {})
                assert "chordPlacements" in schema.content[0].text
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture()
def anyio_backend():
    return "asyncio"
