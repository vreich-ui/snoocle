"""OAuth 2.1 authorization server for the embedded MCP endpoint.

Claude's remote-MCP connector requires OAuth: it discovers the authorization
server from RFC 9728 protected-resource metadata (pointed at by a
`WWW-Authenticate` header on a 401), registers itself via RFC 7591 dynamic
client registration, and runs an authorization-code flow with S256 PKCE.

A static bearer token — which is all Snoocle had — cannot satisfy that, which
is why the connector failed at the registration step.

Snoocle acts as its own authorization server. For a single-user personal
service that is the simplest correct arrangement, and the resource owner is
proven at the consent screen by the `SNOOCLE_API_TOKEN` the user already has.

- `protocol.py` — pure rules (redirect matching, PKCE, audience), unit-tested
- `store.py`    — durable clients/codes/tokens, so a Cloud Run cold start
                  doesn't silently unauthorize the connector
- `routes.py`   — the endpoints and the consent page
"""

from .protocol import OAuthError  # noqa: F401
from .routes import authenticate_bearer, resource_metadata_url, router  # noqa: F401

__all__ = ["router", "authenticate_bearer", "resource_metadata_url", "OAuthError"]
