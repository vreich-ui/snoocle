"""End-to-end MCP test: real client <-> real server over stdio.

Spawns `snoocle_server.mcp_server` as a subprocess and drives it with the
official MCP client, proving the tool surface is callable from any
MCP-compatible client (acceptance step 6).
"""

import base64
import contextlib
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


@contextlib.contextmanager
def _http_mcp_server(tmp_path, **extra_env):
    """Spawn the MCP server in streamable-http mode; yield (base_url, port)."""
    port = _free_port()
    env = {
        **os.environ,
        "SNOOCLE_MCP_TRANSPORT": "streamable-http",
        "SNOOCLE_MCP_PORT": str(port),
        "SNOOCLE_STORE_DIR": str(tmp_path / "store"),
        **extra_env,
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
        yield url, port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.anyio
async def test_mcp_tools_over_streamable_http(tmp_path):
    """Same tool surface, served over HTTP instead of stdio — the transport
    Cloud Run (and any remote MCP client) actually needs, gated in
    deployment by Cloud Run IAM rather than app-level auth. TRUST_PROXY set
    to exercise the explicit host-check opt-out a behind-IAM deploy uses."""
    with _http_mcp_server(tmp_path, SNOOCLE_MCP_TRUST_PROXY="true") as (url, _):
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


def test_http_host_protection_default_and_opt_out(tmp_path):
    """The DNS-rebinding host check must be ON by default (spoofed Host -> 421)
    and only bypassed by the explicit SNOOCLE_MCP_TRUST_PROXY opt-out. This is
    the safe-by-default posture on every supported mcp version — on mcp 1.10.x
    the SDK's own default leaves it OFF, which this explicit config overrides."""
    spoof = {"Host": "attacker.example", "Content-Type": "application/json",
             "Accept": "application/json, text/event-stream"}
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "0"}}}

    # Default: no TRUST_PROXY / ALLOWED_HOSTS -> foreign Host rejected with 421.
    with _http_mcp_server(tmp_path) as (url, _):
        r = httpx.post(url, headers=spoof, json=init, timeout=5.0)
        assert r.status_code == 421, f"expected 421 for spoofed Host, got {r.status_code}"

    # Opt-out: TRUST_PROXY disables the check -> foreign Host is NOT 421.
    with _http_mcp_server(tmp_path, SNOOCLE_MCP_TRUST_PROXY="true") as (url, _):
        r = httpx.post(url, headers=spoof, json=init, timeout=5.0)
        assert r.status_code != 421, f"TRUST_PROXY should bypass host check, got {r.status_code}"


@pytest.mark.parametrize(
    "env,expected_host,expected_protection",
    [
        # default: loopback bind, protection ON (localhost-only)
        ({}, "127.0.0.1", True),
        # trust-proxy (Cloud Run behind IAM): 0.0.0.0 bind, protection OFF
        ({"SNOOCLE_MCP_TRUST_PROXY": "true"}, "0.0.0.0", False),
        # explicit allowed hosts: 0.0.0.0 bind, protection ON
        ({"SNOOCLE_MCP_ALLOWED_HOSTS": "snoocle.run.app"}, "0.0.0.0", True),
        # explicit LOOPBACK host without a security mode is fine (still local)
        ({"SNOOCLE_MCP_HOST": "127.0.0.1"}, "127.0.0.1", True),
        # explicit non-loopback host WITH a security mode is honored
        ({"SNOOCLE_MCP_HOST": "10.0.0.5", "SNOOCLE_MCP_TRUST_PROXY": "true"}, "10.0.0.5", False),
    ],
)
def test_resolve_http_transport_bind_and_protection(env, expected_host, expected_protection):
    from snoocle_server.mcp_server import resolve_http_transport

    host, port, security = resolve_http_transport(env)
    assert host == expected_host
    assert security.enable_dns_rebinding_protection is expected_protection
    if env.get("SNOOCLE_MCP_ALLOWED_HOSTS"):
        # STRICTLY the operator's hosts — no localhost appended, or a LAN client
        # could spoof `Host: localhost:<port>` past this 0.0.0.0-bound allowlist.
        assert security.allowed_hosts == ["snoocle.run.app"]
        assert not any("localhost" in h or "127.0.0.1" in h for h in security.allowed_hosts)


def test_resolve_http_transport_rejects_unguarded_nonloopback_host():
    """A non-loopback SNOOCLE_MCP_HOST without ALLOWED_HOSTS/TRUST_PROXY is a
    misconfiguration (wide bind + localhost-only policy) and must be rejected,
    not silently served."""
    from snoocle_server.mcp_server import resolve_http_transport

    import pytest as _pytest

    with _pytest.raises(ValueError, match="no host-security mode"):
        resolve_http_transport({"SNOOCLE_MCP_HOST": "0.0.0.0"})
    with _pytest.raises(ValueError, match="no host-security mode"):
        resolve_http_transport({"SNOOCLE_MCP_HOST": "10.0.0.5"})


def test_resolve_http_transport_port_precedence():
    from snoocle_server.mcp_server import resolve_http_transport

    # $PORT (Cloud Run) wins over SNOOCLE_MCP_PORT
    _, port, _ = resolve_http_transport({"PORT": "9090", "SNOOCLE_MCP_PORT": "1234"})
    assert port == 9090
    _, port, _ = resolve_http_transport({"SNOOCLE_MCP_PORT": "1234"})
    assert port == 1234
    _, port, _ = resolve_http_transport({})
    assert port == 8080


@pytest.fixture()
def anyio_backend():
    return "asyncio"
