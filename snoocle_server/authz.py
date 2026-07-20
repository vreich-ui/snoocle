"""Shared authorization guard for config-mutating endpoints/tools.

Editing server config (YouTube cookies, the agent's instructions) on a service
that has no bearer token set is a prompt-injection / abuse surface. Both the
REST API and the MCP tool server must refuse it identically — this module holds
the check so ``mcp_server.py`` can share it without importing ``api.py`` (api
imports mcp_server, so the dependency can only go one way).
"""

from __future__ import annotations

from .config import settings


class AdminAuthNotConfigured(RuntimeError):
    """Raised when a config-mutating operation is attempted but the service has
    no SNOOCLE_API_TOKEN gating it."""


def require_admin_token_configured() -> None:
    if not settings.api_token:
        raise AdminAuthNotConfigured(
            "refusing to manage server configuration on an unauthenticated service; "
            "set SNOOCLE_API_TOKEN (and redeploy) first so this operation is gated"
        )
