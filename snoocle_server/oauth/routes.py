"""The OAuth 2.1 authorization-server endpoints, mounted on the same app.

Snoocle is its own authorization server. For a single-user personal service
that is the simplest correct arrangement: no second deployment, no third-party
IdP, and the token the user already has (`SNOOCLE_API_TOKEN`) is what proves
they are the resource owner at the consent screen.

Endpoints:
  GET  /.well-known/oauth-protected-resource[/<path>]   RFC 9728
  GET  /.well-known/oauth-authorization-server[/<path>] RFC 8414
  POST /oauth/register                                  RFC 7591 (DCR)
  GET  /oauth/authorize                                 consent screen
  POST /oauth/authorize                                 consent submission
  POST /oauth/token                                     code + refresh grants
  POST /oauth/revoke                                    RFC 7009

Note the duplicated `/<path>` metadata routes: RFC 9728 says a client probing
for the metadata of a resource at `/mcp` looks under
`/.well-known/oauth-protected-resource/mcp` before the bare path. Serving both
means discovery works whether the connector was added as `https://host/mcp` or
`https://host`.
"""

from __future__ import annotations

import html
import secrets
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ..config import settings
from . import protocol as P
from .store import (
    Client,
    get_oauth_store,
    issue_code,
    issue_token,
    new_secret,
)

router = APIRouter()

# The MCP transport lives here; it is the resource these tokens are for.
MCP_PATH = "/mcp"


def public_base_url(request: Request) -> str:
    """The externally visible origin.

    Behind Cloud Run's proxy the app sees http on an internal host, so the
    forwarded headers are authoritative. Getting this wrong is fatal in a
    quiet way: every metadata document would advertise unreachable URLs.
    An explicit SNOOCLE_PUBLIC_URL always wins.
    """
    configured = (getattr(settings, "public_url", "") or "").strip()
    if configured:
        return P.canonical_resource(configured)

    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
    forwarded_host = request.headers.get("x-forwarded-host", "").split(",")[0].strip()
    scheme = forwarded_proto or request.url.scheme
    host = forwarded_host or request.headers.get("host") or request.url.netloc
    return P.canonical_resource(f"{scheme}://{host}")


def resource_uri(request: Request) -> str:
    return P.canonical_resource(public_base_url(request) + MCP_PATH)


def resource_metadata_url(request: Request) -> str:
    return f"{public_base_url(request)}/.well-known/oauth-protected-resource"


def _error(exc: P.OAuthError) -> JSONResponse:
    return JSONResponse(exc.to_json(), status_code=exc.status)


# --- discovery --------------------------------------------------------------


@router.get("/.well-known/oauth-protected-resource")
@router.get("/.well-known/oauth-protected-resource/{path:path}")
def protected_resource(request: Request, path: str = "") -> JSONResponse:
    # The document must name the resource the user actually typed into Claude.
    # When probed at /.well-known/oauth-protected-resource/mcp, that is /mcp.
    base = public_base_url(request)
    resource = P.canonical_resource(f"{base}/{path}") if path else resource_uri(request)
    return JSONResponse(
        P.protected_resource_metadata(resource, base),
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/.well-known/oauth-authorization-server")
@router.get("/.well-known/oauth-authorization-server/{path:path}")
def authorization_server(request: Request, path: str = "") -> JSONResponse:
    return JSONResponse(
        P.authorization_server_metadata(public_base_url(request)),
        headers={"Cache-Control": "public, max-age=3600"},
    )


# OpenID Discovery is checked by some clients before RFC 8414; answering it
# with the same document costs nothing and removes a failure mode.
@router.get("/.well-known/openid-configuration")
@router.get("/.well-known/openid-configuration/{path:path}")
def openid_configuration(request: Request, path: str = "") -> JSONResponse:
    return authorization_server(request, path)


# --- dynamic client registration (RFC 7591) --------------------------------


@router.post("/oauth/register")
async def register(request: Request) -> JSONResponse:
    """DCR. Bodies are JSON per RFC 7591 §3.1 — note this differs from the
    token endpoint, which is form-encoded."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _error(P.OAuthError("invalid_client_metadata", "body must be JSON"))
    if not isinstance(body, dict):
        return _error(P.OAuthError("invalid_client_metadata", "body must be a JSON object"))

    try:
        redirect_uris = P.validate_registration_redirects(
            [str(u) for u in (body.get("redirect_uris") or [])]
        )
    except P.OAuthError as e:
        return _error(e)

    client = Client(
        client_id="snc_" + new_secret(18),
        redirect_uris=redirect_uris,
        client_name=str(body.get("client_name") or "MCP client")[:200],
        grant_types=list(body.get("grant_types") or ["authorization_code", "refresh_token"]),
        response_types=list(body.get("response_types") or ["code"]),
        # Public client: Claude registers via DCR without a secret, and issuing
        # one we never verify would be security theatre.
        token_endpoint_auth_method="none",
        scope=P.negotiate_scope(body.get("scope")),
        created_at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
    )
    get_oauth_store().save_client(client)
    return JSONResponse(client.to_registration_response(), status_code=201)


# --- authorization ----------------------------------------------------------

_CONSENT_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Connect to Snoocle</title>
<link rel="stylesheet" href="/ui/tokens.css">
<style>
  body {{ margin:0; min-height:100vh; display:flex; align-items:center;
         justify-content:center; background:var(--bg0); color:var(--text);
         font-family:var(--font-ui); padding:var(--s4); }}
  .card {{ background:var(--bg1); border:1px solid var(--stroke);
           border-radius:var(--r-sheet); padding:var(--s5); max-width:420px; width:100%; }}
  h1 {{ font-size:var(--fs-title); margin:0 0 var(--s2); }}
  p {{ color:var(--text-dim); font-size:var(--fs-small); line-height:1.5; }}
  .facts {{ background:var(--bg2); border-radius:var(--r-control);
            padding:var(--s3); margin:var(--s4) 0; font-size:var(--fs-small); }}
  .facts div {{ display:flex; gap:var(--s2); margin-bottom:var(--s1); }}
  .facts span:first-child {{ color:var(--text-dim); min-width:92px; }}
  .facts code {{ font-family:var(--font-mono); word-break:break-all; }}
  input {{ width:100%; min-height:var(--touch); padding:var(--s2) var(--s3);
           border:1px solid var(--stroke); border-radius:var(--r-control);
           background:var(--bg0); color:var(--text); font-size:var(--fs-body); }}
  button {{ width:100%; min-height:var(--touch); margin-top:var(--s3);
            background:var(--accent); color:var(--accent-text); border:0;
            border-radius:var(--r-control); font-size:var(--fs-body); cursor:pointer; }}
  .warn {{ color:var(--warn); }}
  .err {{ color:var(--danger); font-size:var(--fs-small); margin-top:var(--s2); }}
  label {{ display:block; font-size:var(--fs-small); color:var(--text-dim);
           margin:var(--s3) 0 var(--s1); }}
</style></head>
<body><form class="card" method="post" action="/oauth/authorize">
  <h1>Connect to Snoocle</h1>
  <p>{client_name} is asking to use your Snoocle songs and tools.</p>
  <div class="facts">
    <div><span>Client</span><code>{client_name}</code></div>
    <div><span>Redirect to</span><code class="{redirect_class}">{redirect_host}</code></div>
    <div><span>Access</span><code>{scope}</code></div>
  </div>
  {loopback_warning}
  <label for="token">Your Snoocle API token</label>
  <input id="token" name="token" type="password" autocomplete="current-password"
         autofocus required placeholder="SNOOCLE_API_TOKEN">
  {error_html}
  <input type="hidden" name="request_id" value="{request_id}">
  <button type="submit">Allow access</button>
  <p style="margin-top:var(--s4)">Only continue if you started this from Claude.
     The token is checked here and never sent to the client.</p>
</form></body></html>
"""

# Pending authorization requests, held only between rendering the consent page
# and the user submitting it. Deliberately in-process and short-lived: nothing
# here is a credential, and a lost one just means the user starts again.
_PENDING: dict[str, dict] = {}
_PENDING_LIMIT = 64


@router.get("/oauth/authorize", response_model=None)
def authorize(
    request: Request,
    client_id: str = "",
    redirect_uri: str = "",
    response_type: str = "code",
    scope: str = "",
    state: str = "",
    code_challenge: str = "",
    code_challenge_method: str = "",
    resource: str = "",
) -> HTMLResponse | JSONResponse | RedirectResponse:
    store = get_oauth_store()
    client = store.get_client(client_id) if client_id else None

    # Errors BEFORE the redirect URI is validated must render, not redirect —
    # bouncing to an unverified URI is exactly the open-redirect this check
    # exists to prevent.
    if client is None:
        return _error(P.OAuthError("invalid_client", "unknown client_id", status=400))
    if not P.redirect_uri_allowed(redirect_uri, client.redirect_uris):
        return _error(P.OAuthError(
            "invalid_request",
            "redirect_uri does not match a registered redirect for this client",
        ))

    # From here on the redirect URI is trusted, so errors go back to the client.
    def bounce(error: str, description: str) -> RedirectResponse:
        params = {"error": error, "error_description": description}
        if state:
            params["state"] = state
        sep = "&" if "?" in redirect_uri else "?"
        return RedirectResponse(f"{redirect_uri}{sep}{urlencode(params)}", status_code=302)

    if response_type != "code":
        return bounce("unsupported_response_type", "only response_type=code is supported")
    try:
        challenge, method = P.require_pkce(code_challenge, code_challenge_method)
    except P.OAuthError as e:
        return bounce(e.error, e.description)

    granted_scope = P.negotiate_scope(scope)
    target_resource = P.canonical_resource(resource) if resource else resource_uri(request)
    if not P.resource_matches(target_resource, resource_uri(request)):
        return bounce("invalid_target", "resource does not identify this MCP server")

    request_id = secrets.token_urlsafe(16)
    if len(_PENDING) > _PENDING_LIMIT:
        _PENDING.clear()  # bounded; a dropped pending request just restarts
    _PENDING[request_id] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": method,
        "scope": granted_scope,
        "resource": target_resource,
    }

    return HTMLResponse(_render_consent(request_id, client.client_name,
                                        redirect_uri, granted_scope))


def _render_consent(request_id: str, client_name: str, redirect_uri: str,
                    scope: str, error: str = "") -> str:
    loopback = P.is_loopback(redirect_uri)
    host = html.escape(redirect_uri)
    # The MCP spec requires showing the redirect host, and warning when the only
    # redirect is loopback — any local process can claim to be the client.
    warning = (
        '<p class="warn">This will send access to a program running on this '
        'computer. Only continue if you just started a connection from a local '
        'tool you trust.</p>' if loopback else ""
    )
    return _CONSENT_PAGE.format(
        client_name=html.escape(client_name or "An MCP client"),
        redirect_host=host,
        redirect_class="warn" if loopback else "",
        loopback_warning=warning,
        scope=html.escape(scope),
        request_id=html.escape(request_id),
        error_html=f'<div class="err">{html.escape(error)}</div>' if error else "",
    )


@router.post("/oauth/authorize", response_model=None)
def authorize_submit(request_id: str = Form(""), token: str = Form("")) -> HTMLResponse | RedirectResponse | JSONResponse:
    pending = _PENDING.get(request_id)
    if pending is None:
        return _error(P.OAuthError(
            "invalid_request", "this authorization request expired — start again from Claude"
        ))

    if not _check_owner(token):
        # Keep the request alive so a typo doesn't mean restarting from Claude,
        # but never reveal whether the token was close.
        return HTMLResponse(
            _render_consent(request_id, "", pending["redirect_uri"], pending["scope"],
                            error="That token is not right."),
            status_code=401,
        )

    _PENDING.pop(request_id, None)
    code = issue_code(
        client_id=pending["client_id"],
        redirect_uri=pending["redirect_uri"],
        code_challenge=pending["code_challenge"],
        code_challenge_method=pending["code_challenge_method"],
        scope=pending["scope"],
        resource=pending["resource"],
    )
    get_oauth_store().save_code(code)

    params = {"code": code.code}
    if pending["state"]:
        params["state"] = pending["state"]
    sep = "&" if "?" in pending["redirect_uri"] else "?"
    return RedirectResponse(f"{pending['redirect_uri']}{sep}{urlencode(params)}", status_code=302)


def _check_owner(token: str) -> bool:
    """The resource owner is whoever holds SNOOCLE_API_TOKEN.

    If no token is configured the service is already wide open, so demanding
    one here would be theatre — but we refuse to *pretend* it was checked.
    """
    import secrets as _secrets

    configured = settings.api_token
    if not configured:
        return True
    return bool(token) and _secrets.compare_digest(token, configured)


# --- token ------------------------------------------------------------------


@router.post("/oauth/token")
async def token_endpoint(
    request: Request,
    grant_type: str = Form(""),
    code: str = Form(""),
    redirect_uri: str = Form(""),
    client_id: str = Form(""),
    code_verifier: str = Form(""),
    refresh_token: str = Form(""),
    resource: str = Form(""),
    scope: str = Form(""),
) -> JSONResponse:
    """RFC 6749 §4.1.3: form-encoded, not JSON."""
    store = get_oauth_store()
    try:
        if grant_type == "authorization_code":
            record = store.consume_code(code)
            if record is None:
                # Covers unknown, expired and replayed alike — distinguishing
                # them would tell an attacker which codes once existed.
                raise P.OAuthError("invalid_grant", "authorization code is invalid or expired")
            if client_id and client_id != record.client_id:
                raise P.OAuthError("invalid_grant", "code was issued to a different client")
            if redirect_uri and redirect_uri != record.redirect_uri:
                raise P.OAuthError("invalid_grant", "redirect_uri does not match the authorization request")
            if not P.verify_pkce(code_verifier, record.code_challenge, record.code_challenge_method):
                raise P.OAuthError("invalid_grant", "PKCE verification failed")
            if resource and not P.resource_matches(resource, record.resource):
                raise P.OAuthError("invalid_target", "resource does not match the authorization request")
            issued = issue_token(record.client_id, record.scope, record.resource)

        elif grant_type == "refresh_token":
            previous = store.rotate_refresh_token(refresh_token)
            if previous is None:
                raise P.OAuthError("invalid_grant", "refresh token is invalid or expired")
            # OAuth 2.1 requires rotation for public clients: the old refresh
            # token is already gone, and the new one ships in this response.
            granted = P.negotiate_scope(scope) if scope else previous.scope
            issued = issue_token(previous.client_id, granted, previous.resource)

        else:
            raise P.OAuthError("unsupported_grant_type", f"unsupported grant_type: {grant_type!r}")
    except P.OAuthError as e:
        return _error(e)

    store.save_token(issued)
    return JSONResponse(
        issued.to_token_response(),
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


@router.post("/oauth/revoke")
async def revoke(token: str = Form("")) -> JSONResponse:
    # RFC 7009: always 200, even for an unknown token — the caller's goal
    # ("this token must not work") is satisfied either way.
    if token:
        get_oauth_store().revoke(token)
    return JSONResponse({}, status_code=200)


# --- resource-server side ---------------------------------------------------


def authenticate_bearer(token: str, request: Request) -> Optional[object]:
    """Validate an access token for the MCP resource.

    Returns the token record, or None. Audience is checked here: a token whose
    `resource` isn't this server is rejected even if it is otherwise valid,
    which is what the MCP spec means by access-token privilege restriction.
    """
    if not token:
        return None
    record = get_oauth_store().get_by_access_token(token)
    if record is None or record.access_expired():
        return None
    if not P.resource_matches(record.resource, resource_uri(request)):
        return None
    return record
