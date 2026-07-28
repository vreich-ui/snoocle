"""OAuth 2.1 authorization server for the MCP endpoint.

Claude's remote-MCP connector refuses to use a static bearer token: it
discovers an authorization server, registers itself (RFC 7591), and runs an
authorization-code flow with S256 PKCE. Without that, the connector fails at
the registration step — which is the bug this exists to fix.

Two halves are tested here. The pure rules (redirect matching, PKCE, audience)
get direct unit tests because they are where a subtle mistake turns into "any
client can get a token". Then the whole flow runs over HTTP, the way Claude
runs it.
"""

from __future__ import annotations

import base64
import hashlib
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from snoocle_server.api import app
from snoocle_server.config import settings
from snoocle_server.oauth import protocol as P
from snoocle_server.oauth import routes as R
from snoocle_server.oauth import store as S
from snoocle_server.oauth.store import (
    InMemoryOAuthRepository,
    issue_code,
    issue_token,
)

TOKEN = "s3cr3t-personal-token"
CLAUDE_REDIRECT = "https://claude.ai/api/mcp/auth_callback"


@pytest.fixture()
def store(monkeypatch):
    """Install an in-memory repo as THE store, not as a patched name.

    Setting the module singleton means every caller resolves to it — the OAuth
    routes and the /v1 admin endpoints alike. Patching `get_oauth_store` in one
    module only would leave the other talking to the real backend, and a test
    that quietly exercises a different store than the code under test is worse
    than no test.
    """
    repo = InMemoryOAuthRepository()
    monkeypatch.setattr(S, "_repo", repo)
    return repo


@pytest.fixture(scope="module")
def live_app():
    """One context-managed client for the whole module.

    The lifespan must run — a request that authenticates successfully reaches
    the MCP session manager, which raises unless it was started. And it can
    only be started ONCE per process, so this is module-scoped rather than
    per-test.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def client(monkeypatch, store, live_app):
    monkeypatch.setattr(settings, "api_token", TOKEN)
    R._PENDING.clear()
    return live_app


def pkce_pair(verifier: str = "a" * 64) -> tuple[str, str]:
    digest = hashlib.sha256(verifier.encode()).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


# ===========================================================================
# Pure rules — where a subtle bug becomes a security hole
# ===========================================================================

# --- redirect URI matching --------------------------------------------------


def test_exact_redirect_match_only():
    registered = [CLAUDE_REDIRECT]
    assert P.redirect_uri_allowed(CLAUDE_REDIRECT, registered)
    # No prefix, suffix or substring matching — each of these is a classic
    # open-redirect that would hand an attacker the authorization code.
    for evil in (
        "https://claude.ai/api/mcp/auth_callback/../../evil",
        "https://claude.ai/api/mcp/auth_callback.evil.com",
        "https://claude.ai.evil.com/api/mcp/auth_callback",
        "https://claude.ai/api/mcp/auth_callbackX",
        "https://evil.com/api/mcp/auth_callback",
        "",
    ):
        assert not P.redirect_uri_allowed(evil, registered), evil


def test_loopback_redirects_ignore_the_port_but_nothing_else():
    """Claude Code declares http://localhost/callback and binds an ephemeral
    port, so the port must not be compared (RFC 8252 §7.3)."""
    registered = ["http://localhost/callback", "http://127.0.0.1/callback"]
    assert P.redirect_uri_allowed("http://localhost:3118/callback", registered)
    assert P.redirect_uri_allowed("http://127.0.0.1:51234/callback", registered)
    # The path still has to match, and a different host is still a different host.
    assert not P.redirect_uri_allowed("http://localhost:3118/other", registered)
    assert not P.redirect_uri_allowed("http://evil.com:3118/callback", registered)


def test_registration_rejects_plaintext_and_fragment_redirects():
    P.validate_registration_redirects([CLAUDE_REDIRECT])
    P.validate_registration_redirects(["http://127.0.0.1/callback"])  # loopback ok
    for bad in (["http://example.com/cb"], ["https://x.com/cb#frag"], []):
        with pytest.raises(P.OAuthError):
            P.validate_registration_redirects(bad)


# --- PKCE -------------------------------------------------------------------


def test_pkce_s256_round_trip():
    verifier, challenge = pkce_pair()
    assert P.verify_pkce(verifier, challenge, "S256")
    assert not P.verify_pkce("wrong-verifier", challenge, "S256")


def test_plain_pkce_is_refused_not_downgraded():
    """Accepting `plain` would silently remove the protection PKCE exists for."""
    assert not P.verify_pkce("abc", "abc", "plain")
    with pytest.raises(P.OAuthError):
        P.require_pkce("abc", "plain")
    with pytest.raises(P.OAuthError):
        P.require_pkce("", "S256")       # missing challenge
    with pytest.raises(P.OAuthError):
        P.require_pkce("abc", None)      # defaults to plain


# --- audience ---------------------------------------------------------------


def test_canonical_resource_normalises_case_and_trailing_slash():
    assert P.canonical_resource("HTTPS://Host.Example/MCP/") == "https://host.example/MCP"
    assert P.canonical_resource("https://h/mcp#frag") == "https://h/mcp"


def test_audience_accepts_this_server_and_its_origin_only():
    target = "https://snoocle.run.app/mcp"
    assert P.resource_matches("https://snoocle.run.app/mcp", target)
    assert P.resource_matches("https://snoocle.run.app", target)       # bare origin
    assert P.resource_matches("https://snoocle.run.app/mcp/", target)  # trailing slash
    # A token minted for anything else must not be usable here.
    assert not P.resource_matches("https://evil.example/mcp", target)
    assert not P.resource_matches("https://snoocle.run.app/v1/songs", target)
    assert not P.resource_matches("", target)


# --- metadata ---------------------------------------------------------------


def test_authorization_server_metadata_advertises_what_claude_checks():
    md = P.authorization_server_metadata("https://h")
    assert md["code_challenge_methods_supported"] == ["S256"]
    assert "none" in md["token_endpoint_auth_methods_supported"]
    assert md["registration_endpoint"].endswith("/oauth/register")
    assert "offline_access" in md["scopes_supported"]
    assert "authorization_code" in md["grant_types_supported"]
    assert "refresh_token" in md["grant_types_supported"]


def test_www_authenticate_points_at_the_metadata():
    header = P.www_authenticate("https://h/.well-known/oauth-protected-resource",
                                scope="snoocle:mcp")
    assert header.startswith("Bearer ")
    assert 'resource_metadata="https://h/.well-known/oauth-protected-resource"' in header
    assert 'scope="snoocle:mcp"' in header


# ===========================================================================
# The flow, over HTTP
# ===========================================================================


def test_mcp_401_carries_the_discovery_header(client):
    """This is the handshake that was missing. Claude only honours
    WWW-Authenticate on a 401, so both the status and the header matter."""
    r = client.post("/mcp", json={})
    assert r.status_code == 401
    header = r.headers["www-authenticate"]
    assert "resource_metadata=" in header
    assert "/.well-known/oauth-protected-resource" in header


def test_discovery_documents_are_served_without_a_token(client):
    for path in (
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-protected-resource/mcp",
        "/.well-known/oauth-authorization-server",
        "/.well-known/openid-configuration",
    ):
        r = client.get(path)
        assert r.status_code == 200, path


def test_protected_resource_metadata_names_this_server(client):
    body = client.get("/.well-known/oauth-protected-resource").json()
    assert body["resource"].endswith("/mcp")
    assert body["authorization_servers"], "must list at least one issuer"
    assert body["authorization_servers"][0] == body["resource"].removesuffix("/mcp")


def test_probe_path_reports_the_resource_the_user_typed(client):
    """RFC 9728 probing: a connector added as https://host/mcp looks under
    /.well-known/oauth-protected-resource/mcp, and the document must name
    exactly that URL or the connector rejects it."""
    body = client.get("/.well-known/oauth-protected-resource/mcp").json()
    assert body["resource"].endswith("/mcp")


def register(client, redirect_uris=None) -> dict:
    r = client.post("/oauth/register", json={
        "client_name": "Claude",
        "redirect_uris": redirect_uris or [CLAUDE_REDIRECT],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    })
    assert r.status_code == 201, r.text
    return r.json()


def test_dynamic_client_registration(client):
    body = register(client)
    assert body["client_id"]
    assert body["redirect_uris"] == [CLAUDE_REDIRECT]
    assert body["token_endpoint_auth_method"] == "none"
    assert "client_id_issued_at" in body


def test_registration_rejects_an_unsafe_redirect(client):
    r = client.post("/oauth/register", json={"redirect_uris": ["http://evil.com/cb"]})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_redirect_uri"


def test_registration_requires_json_not_form(client):
    """RFC 7591 §3.1 is JSON; the token endpoint is form-encoded. Mixing the
    two up is a documented way to break this flow."""
    r = client.post("/oauth/register", data={"redirect_uris": CLAUDE_REDIRECT})
    assert r.status_code == 400


def authorize(client, client_id, challenge, **overrides) -> "object":
    params = {
        "client_id": client_id,
        "redirect_uri": CLAUDE_REDIRECT,
        "response_type": "code",
        "scope": "snoocle:mcp offline_access",
        "state": "xyz",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    params.update(overrides)
    return client.get("/oauth/authorize", params=params, follow_redirects=False)


def consent(client, page_html: str, token: str = TOKEN):
    import re

    request_id = re.search(r'name="request_id" value="([^"]+)"', page_html).group(1)
    return client.post("/oauth/authorize",
                       data={"request_id": request_id, "token": token},
                       follow_redirects=False)


def full_flow(client) -> dict:
    reg = register(client)
    verifier, challenge = pkce_pair()
    page = authorize(client, reg["client_id"], challenge)
    assert page.status_code == 200
    redirect = consent(client, page.text)
    assert redirect.status_code == 302
    query = parse_qs(urlparse(redirect.headers["location"]).query)
    assert query["state"] == ["xyz"]

    token = client.post("/oauth/token", data={
        "grant_type": "authorization_code",
        "code": query["code"][0],
        "redirect_uri": CLAUDE_REDIRECT,
        "client_id": reg["client_id"],
        "code_verifier": verifier,
    })
    assert token.status_code == 200, token.text
    return token.json()


def test_the_whole_flow_yields_a_working_mcp_token(client):
    tokens = full_flow(client)
    assert tokens["token_type"] == "Bearer"
    assert tokens["expires_in"] > 0
    assert tokens["refresh_token"]

    # The point of all of it: this token opens /mcp.
    r = client.post("/mcp", headers={"Authorization": f"Bearer {tokens['access_token']}"},
                    json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert r.status_code != 401


def test_consent_requires_the_owners_token(client):
    reg = register(client)
    _, challenge = pkce_pair()
    page = authorize(client, reg["client_id"], challenge)
    denied = consent(client, page.text, token="wrong")
    assert denied.status_code == 401
    assert "is not right" in denied.text


def test_consent_page_shows_the_redirect_destination(client):
    """The MCP spec requires the user can see where they are being sent."""
    reg = register(client)
    _, challenge = pkce_pair()
    page = authorize(client, reg["client_id"], challenge)
    assert "claude.ai/api/mcp/auth_callback" in page.text


def test_a_loopback_client_gets_an_explicit_warning(client):
    reg = register(client, redirect_uris=["http://localhost/callback"])
    _, challenge = pkce_pair()
    page = authorize(client, reg["client_id"], challenge,
                     redirect_uri="http://localhost:3118/callback")
    assert page.status_code == 200
    assert "running on this computer" in page.text


# --- authorization endpoint refusals ----------------------------------------


def test_unknown_client_is_rendered_not_redirected(client):
    """Before the redirect URI is verified, an error must NOT bounce — that
    would be the open redirect."""
    _, challenge = pkce_pair()
    r = authorize(client, "snc_nope", challenge)
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_client"


def test_a_mismatched_redirect_is_rendered_not_redirected(client):
    reg = register(client)
    _, challenge = pkce_pair()
    r = authorize(client, reg["client_id"], challenge, redirect_uri="https://evil.com/cb")
    assert r.status_code == 400
    assert "redirect_uri" in r.json()["error_description"]


def test_missing_pkce_bounces_to_the_client_with_an_error(client):
    reg = register(client)
    r = client.get("/oauth/authorize", params={
        "client_id": reg["client_id"], "redirect_uri": CLAUDE_REDIRECT,
        "response_type": "code", "state": "xyz",
    }, follow_redirects=False)
    assert r.status_code == 302
    query = parse_qs(urlparse(r.headers["location"]).query)
    assert query["error"] == ["invalid_request"]
    assert query["state"] == ["xyz"]


def test_a_resource_for_another_server_is_refused(client):
    reg = register(client)
    _, challenge = pkce_pair()
    r = authorize(client, reg["client_id"], challenge,
                  resource="https://someone-else.example/mcp")
    assert r.status_code == 302
    assert parse_qs(urlparse(r.headers["location"]).query)["error"] == ["invalid_target"]


# --- token endpoint ---------------------------------------------------------


def test_a_code_cannot_be_redeemed_twice(client):
    reg = register(client)
    verifier, challenge = pkce_pair()
    page = authorize(client, reg["client_id"], challenge)
    redirect = consent(client, page.text)
    code = parse_qs(urlparse(redirect.headers["location"]).query)["code"][0]

    body = {"grant_type": "authorization_code", "code": code,
            "redirect_uri": CLAUDE_REDIRECT, "client_id": reg["client_id"],
            "code_verifier": verifier}
    assert client.post("/oauth/token", data=body).status_code == 200
    replay = client.post("/oauth/token", data=body)
    assert replay.status_code == 400
    assert replay.json()["error"] == "invalid_grant"


def test_a_stolen_code_is_useless_without_the_verifier(client):
    """The whole point of PKCE: intercepting the code is not enough."""
    reg = register(client)
    _, challenge = pkce_pair()
    page = authorize(client, reg["client_id"], challenge)
    redirect = consent(client, page.text)
    code = parse_qs(urlparse(redirect.headers["location"]).query)["code"][0]

    r = client.post("/oauth/token", data={
        "grant_type": "authorization_code", "code": code,
        "redirect_uri": CLAUDE_REDIRECT, "client_id": reg["client_id"],
        "code_verifier": "attacker-guess",
    })
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_grant"


def test_a_code_cannot_be_redeemed_by_a_different_client(client):
    reg_a = register(client)
    reg_b = register(client)
    verifier, challenge = pkce_pair()
    page = authorize(client, reg_a["client_id"], challenge)
    redirect = consent(client, page.text)
    code = parse_qs(urlparse(redirect.headers["location"]).query)["code"][0]

    r = client.post("/oauth/token", data={
        "grant_type": "authorization_code", "code": code,
        "redirect_uri": CLAUDE_REDIRECT, "client_id": reg_b["client_id"],
        "code_verifier": verifier,
    })
    assert r.status_code == 400


def test_refresh_rotates_and_invalidates_the_old_token(client):
    """OAuth 2.1 requires rotation for public clients."""
    tokens = full_flow(client)
    refreshed = client.post("/oauth/token", data={
        "grant_type": "refresh_token", "refresh_token": tokens["refresh_token"],
    })
    assert refreshed.status_code == 200
    new = refreshed.json()
    assert new["refresh_token"] != tokens["refresh_token"]

    reused = client.post("/oauth/token", data={
        "grant_type": "refresh_token", "refresh_token": tokens["refresh_token"],
    })
    assert reused.status_code == 400
    assert reused.json()["error"] == "invalid_grant"


def test_an_expired_refresh_token_reports_invalid_grant(client):
    """Claude keys its recovery on this exact code — returning
    invalid_request instead turns a refresh into a dead connector."""
    r = client.post("/oauth/token", data={
        "grant_type": "refresh_token", "refresh_token": "snr_nonexistent",
    })
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_grant"


def test_unsupported_grant_types_are_named(client):
    r = client.post("/oauth/token", data={"grant_type": "client_credentials"})
    assert r.status_code == 400
    assert r.json()["error"] == "unsupported_grant_type"


def test_token_responses_are_not_cached(client):
    tokens = full_flow(client)
    assert tokens["access_token"]
    r = client.post("/oauth/token", data={
        "grant_type": "refresh_token", "refresh_token": tokens["refresh_token"],
    })
    assert r.headers["cache-control"] == "no-store"


# --- audience enforcement at the resource -----------------------------------


def test_a_token_for_another_audience_does_not_open_mcp(client, store):
    """Access-token privilege restriction: even a token this server issued is
    refused at /mcp if its audience is something else."""
    foreign = issue_token("snc_x", "snoocle:mcp", "https://elsewhere.example/mcp")
    store.save_token(foreign)
    r = client.post("/mcp", headers={"Authorization": f"Bearer {foreign.access_token}"},
                    json={})
    assert r.status_code == 401


def test_an_oauth_token_does_not_open_the_rest_api(client):
    """OAuth tokens are minted for /mcp. Honouring them on /v1 would be the
    audience confusion the spec forbids."""
    tokens = full_flow(client)
    r = client.get("/v1/songs", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert r.status_code == 401


def test_the_static_token_still_works_everywhere(client):
    """The iOS app and the admin UI were not asked to change."""
    assert client.get("/v1/songs", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert client.post("/mcp", headers={"Authorization": f"Bearer {TOKEN}"},
                       json={}).status_code != 401


def test_an_expired_access_token_is_refused(client, store):
    expired = issue_token("snc_x", "snoocle:mcp", "https://testserver/mcp")
    expired.access_expires_at = "2020-01-01T00:00:00+00:00"
    store.save_token(expired)
    r = client.post("/mcp", headers={"Authorization": f"Bearer {expired.access_token}"},
                    json={})
    assert r.status_code == 401


def test_revocation_makes_a_token_stop_working(client):
    tokens = full_flow(client)
    assert client.post("/oauth/revoke", data={"token": tokens["access_token"]}).status_code == 200
    r = client.post("/mcp", headers={"Authorization": f"Bearer {tokens['access_token']}"},
                    json={})
    assert r.status_code == 401


def test_revoking_an_unknown_token_is_still_200(client):
    # RFC 7009: the caller's goal is satisfied either way.
    assert client.post("/oauth/revoke", data={"token": "nope"}).status_code == 200


# --- unauthenticated deployments --------------------------------------------


def test_without_a_configured_token_the_service_is_unchanged(monkeypatch, store, live_app):
    """No SNOOCLE_API_TOKEN means the service was already open (Cloud Run IAM
    gates it). Adding OAuth must not change that posture."""
    monkeypatch.setattr(settings, "api_token", "")
    assert live_app.post("/mcp", json={}).status_code != 401


# --- storage ----------------------------------------------------------------


def test_codes_are_single_use_at_the_store_level(store):
    code = issue_code("c", CLAUDE_REDIRECT, "chal", "S256", "snoocle:mcp", "https://h/mcp")
    store.save_code(code)
    assert store.consume_code(code.code) is not None
    assert store.consume_code(code.code) is None


def test_an_expired_code_is_burned_not_honoured(store):
    code = issue_code("c", CLAUDE_REDIRECT, "chal", "S256", "snoocle:mcp", "https://h/mcp")
    code.expires_at = "2020-01-01T00:00:00+00:00"
    store.save_code(code)
    assert store.consume_code(code.code) is None


# --- registered clients: visibility and revocation ---------------------------


def test_registered_clients_are_listable(client):
    """DCR accepts anyone, so the list is the only thing standing between a
    stray registration and nobody ever noticing it."""
    reg = register(client)
    body = client.get("/v1/oauth/clients", headers={"Authorization": f"Bearer {TOKEN}"}).json()
    assert reg["client_id"] in [c["clientId"] for c in body["clients"]]


def test_deleting_a_client_kills_its_tokens_too(client, store):
    """Revoking a registration but leaving its access token working for an hour
    (and its refresh token for 90 days) is the opposite of what revoke means."""
    tokens = full_flow(client)
    client_id = store.list_clients()[0].client_id
    auth = {"Authorization": f"Bearer {TOKEN}"}

    assert client.delete(f"/v1/oauth/clients/{client_id}", headers=auth).status_code == 200
    # the registration is gone...
    assert client_id not in [c["clientId"] for c in client.get("/v1/oauth/clients", headers=auth).json()["clients"]]
    # ...and so is the token it was holding
    r = client.post("/mcp", headers={"Authorization": f"Bearer {tokens['access_token']}"}, json={})
    assert r.status_code == 401
    refused = client.post("/oauth/token", data={
        "grant_type": "refresh_token", "refresh_token": tokens["refresh_token"],
    })
    assert refused.status_code == 400


def test_deleting_an_unknown_client_is_404(client):
    r = client.delete("/v1/oauth/clients/snc_nope", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 404


def test_an_oauth_token_cannot_enumerate_or_delete_clients(client):
    """A connector must not be able to see or revoke its siblings."""
    tokens = full_flow(client)
    auth = {"Authorization": f"Bearer {tokens['access_token']}"}
    assert client.get("/v1/oauth/clients", headers=auth).status_code == 401
    assert client.delete("/v1/oauth/clients/snc_x", headers=auth).status_code == 401
