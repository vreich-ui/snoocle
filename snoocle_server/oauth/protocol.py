"""Pure OAuth 2.1 / MCP-authorization rules.

Everything here is a pure function so the fiddly, security-relevant decisions —
redirect-URI matching, PKCE verification, audience comparison — are unit-tested
directly rather than only through HTTP.

Getting any of these subtly wrong is the difference between "it connects" and
"anyone can connect", so each one is written to fail closed.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Iterable, Optional
from urllib.parse import urlparse, urlsplit, urlunsplit

# Claude's hosted surfaces (web, Desktop, mobile, Cowork) all use this one.
CLAUDE_REDIRECT = "https://claude.ai/api/mcp/auth_callback"

# Claude Code is a native client on an RFC 8252 loopback redirect with an
# ephemeral port. It declares these two without a port, so the port component
# must be ignored when matching.
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}

SUPPORTED_SCOPES = ["snoocle:mcp", "offline_access"]
DEFAULT_SCOPE = "snoocle:mcp offline_access"


class OAuthError(Exception):
    """An OAuth error with the RFC-defined code the client expects.

    The code matters: Claude refreshes reactively and keys its behaviour on
    `invalid_grant` specifically, so returning a generic `invalid_request` for
    an expired refresh token turns a recoverable state into a dead connector.
    """

    def __init__(self, error: str, description: str = "", status: int = 400) -> None:
        super().__init__(description or error)
        self.error = error
        self.description = description
        self.status = status

    def to_json(self) -> dict:
        body = {"error": self.error}
        if self.description:
            body["error_description"] = self.description
        return body


# --- redirect URIs ----------------------------------------------------------


def is_loopback(uri: str) -> bool:
    try:
        parsed = urlparse(uri)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def normalize_loopback(uri: str) -> str:
    """Strip the port from a loopback redirect so an ephemeral port matches.

    RFC 8252 §7.3 requires port-agnostic matching for the IP-literal form;
    Claude Code needs the same treatment for `localhost`.
    """
    parts = urlsplit(uri)
    host = (parts.hostname or "").lower()
    netloc = f"[{host}]" if ":" in host else host
    return urlunsplit((parts.scheme.lower(), netloc, parts.path, "", ""))


def redirect_uri_allowed(requested: str, registered: Iterable[str]) -> bool:
    """Exact match, except that loopback redirects ignore the port.

    Deliberately strict everywhere else: no prefix matching, no wildcards, no
    "starts with". Open redirection is the classic way an authorization server
    hands codes to an attacker.
    """
    if not requested:
        return False
    registered = list(registered)

    if is_loopback(requested):
        target = normalize_loopback(requested)
        return any(
            is_loopback(candidate) and normalize_loopback(candidate) == target
            for candidate in registered
        )
    return any(hmac.compare_digest(requested, candidate) for candidate in registered)


def validate_registration_redirects(uris: list[str]) -> list[str]:
    """RFC 7591 registration input. HTTPS or loopback only — an http:// URI to
    a real host would leak the code over the wire."""
    if not uris:
        raise OAuthError("invalid_redirect_uri", "at least one redirect_uri is required")
    for uri in uris:
        parsed = urlparse(uri)
        if parsed.fragment:
            raise OAuthError("invalid_redirect_uri", f"redirect_uri must not contain a fragment: {uri}")
        if parsed.scheme == "https":
            continue
        if parsed.scheme == "http" and is_loopback(uri):
            continue
        raise OAuthError(
            "invalid_redirect_uri",
            f"redirect_uri must be https, or http on loopback: {uri}",
        )
    return uris


# --- PKCE -------------------------------------------------------------------


def verify_pkce(verifier: str, challenge: str, method: str) -> bool:
    """S256 only.

    `plain` is accepted by neither OAuth 2.1 nor the MCP spec, and supporting
    it would silently remove the protection PKCE exists to provide, so an
    unknown method fails rather than falling back.
    """
    if method != "S256" or not verifier or not challenge:
        return False
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return hmac.compare_digest(computed, challenge)


def require_pkce(code_challenge: Optional[str], method: Optional[str]) -> tuple[str, str]:
    if not code_challenge:
        raise OAuthError("invalid_request", "code_challenge is required (PKCE)")
    if (method or "plain") != "S256":
        raise OAuthError(
            "invalid_request",
            "code_challenge_method must be S256; plain PKCE is not accepted",
        )
    return code_challenge, "S256"


# --- resource / audience ----------------------------------------------------


def canonical_resource(uri: str) -> str:
    """RFC 8707 canonical form: lowercase scheme+host, no fragment, no trailing
    slash. Used on both sides of every audience comparison so cosmetic
    differences never read as a mismatch."""
    parts = urlsplit(uri.strip())
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, "", ""))


def resource_matches(requested: str, canonical: str) -> bool:
    """Is a token minted for `requested` valid at `canonical`?

    Accepts the server's own canonical URI and its bare origin, because clients
    legitimately send either (`https://host/mcp` or `https://host`). Anything
    else — including another path on the same host — is rejected: a token is
    for this resource or it is for nothing.
    """
    if not requested:
        return False
    req = canonical_resource(requested)
    target = canonical_resource(canonical)
    if req == target:
        return True
    origin = canonical_resource(
        urlunsplit((urlsplit(target).scheme, urlsplit(target).netloc, "", "", ""))
    )
    return req == origin


# --- scopes -----------------------------------------------------------------


def negotiate_scope(requested: Optional[str]) -> str:
    """Grant the intersection of what was asked for and what exists.

    Never grant a scope that wasn't requested, and never fail the whole flow
    over an unknown one — Claude appends `offline_access` on its own, and a
    strict reject would break the connector for a scope we do support.
    """
    if not requested:
        return DEFAULT_SCOPE
    granted = [s for s in requested.split() if s in SUPPORTED_SCOPES]
    return " ".join(granted) if granted else DEFAULT_SCOPE


# --- metadata documents -----------------------------------------------------


def protected_resource_metadata(resource: str, issuer: str) -> dict:
    """RFC 9728. `resource` MUST equal the MCP URL as the user typed it into
    Claude, path included, or the connector rejects the document."""
    return {
        "resource": canonical_resource(resource),
        "authorization_servers": [canonical_resource(issuer)],
        "scopes_supported": SUPPORTED_SCOPES,
        "bearer_methods_supported": ["header"],
        "resource_documentation": f"{canonical_resource(issuer)}/docs",
    }


def authorization_server_metadata(issuer: str) -> dict:
    """RFC 8414. `code_challenge_methods_supported` is not optional in
    practice: spec-compliant clients check it before starting the flow."""
    base = canonical_resource(issuer)
    return {
        "issuer": base,
        "authorization_endpoint": f"{base}/oauth/authorize",
        "token_endpoint": f"{base}/oauth/token",
        "registration_endpoint": f"{base}/oauth/register",
        "revocation_endpoint": f"{base}/oauth/revoke",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        # "none" = public client. Claude registers as one via DCR, and its CIMD
        # client authenticates as one at the token endpoint.
        "token_endpoint_auth_methods_supported": ["none"],
        "scopes_supported": SUPPORTED_SCOPES,
        "service_documentation": f"{base}/docs",
    }


def www_authenticate(resource_metadata_url: str, scope: str = "",
                     error: str = "", description: str = "") -> str:
    """The header that starts the whole dance. Claude only honours it on a 401,
    and without the `resource_metadata` pointer it has to guess the metadata
    location by probing — which fails on any host that can't serve
    /.well-known/*."""
    parts = [f'Bearer resource_metadata="{resource_metadata_url}"']
    if scope:
        parts.append(f'scope="{scope}"')
    if error:
        parts.append(f'error="{error}"')
    if description:
        parts.append(f'error_description="{description}"')
    return ", ".join(parts)
