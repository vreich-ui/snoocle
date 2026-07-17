"""Optional app-level static bearer token (SNOOCLE_API_TOKEN).

When set, one token authorizes BOTH the REST API and the embedded /mcp
transport (same `Authorization: Bearer <token>` header). When unset, the app is
a pass-through (Cloud Run IAM remains the gate). /healthz is always exempt.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from snoocle_server.api import app
from snoocle_server.config import settings

client = TestClient(app)
TOKEN = "s3cr3t-personal-token"


@pytest.fixture()
def token_enabled(monkeypatch):
    monkeypatch.setattr(settings, "api_token", TOKEN)


def test_no_token_configured_is_passthrough():
    # default: api_token == "" -> no app-level auth, REST reachable without a header
    assert settings.api_token == ""
    assert client.get("/v1/providers").status_code == 200


def test_rest_requires_token_when_configured(token_enabled):
    assert client.get("/v1/providers").status_code == 401
    assert client.get("/v1/providers", headers={"Authorization": "Bearer wrong"}).status_code == 401
    r = client.get("/v1/providers", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200


def test_healthz_is_exempt(token_enabled):
    # liveness probes must work without the token
    assert client.get("/healthz").status_code == 200


def test_ui_shell_is_exempt_but_api_still_gated(token_enabled):
    # the static GUI shell loads without a token (it holds no secrets)...
    assert client.get("/ui/").status_code == 200
    # ...while every API call still requires the token
    assert client.get("/v1/songs").status_code == 401


def test_mcp_endpoint_shares_the_same_token(token_enabled):
    # the /mcp transport is gated by the SAME middleware (one token for both
    # surfaces): no token -> 401 before the request ever reaches MCP. The
    # authorized /mcp handshake is exercised end-to-end in test_combined_app /
    # test_mcp_server against a real server (TestClient doesn't run the MCP
    # session-manager lifespan).
    unauth = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert unauth.status_code == 401
    assert unauth.headers.get("WWW-Authenticate") == "Bearer"


def test_401_body_and_challenge_header(token_enabled):
    r = client.get("/v1/songs")
    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate") == "Bearer"
    assert "detail" in r.json()
