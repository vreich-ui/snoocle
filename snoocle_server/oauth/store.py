"""Persistence for the OAuth authorization server.

Everything here MUST survive a process restart. Cloud Run scales this service
to zero between uses, so an in-memory client registry would mean the connector
silently loses its authorization every time the instance recycles — the user
sees "reconnect" over and over with no explanation. Registered clients, refresh
tokens and access tokens therefore live in the same store as songs.

Records are deliberately opaque random strings rather than JWTs. A personal
service does not need signature verification it would then have to key-manage,
and opaque tokens are revocable by deleting a row, which a JWT is not.

Three collections:
  oauth_clients   — dynamically registered clients (RFC 7591)
  oauth_codes     — authorization codes, single-use, ~60s TTL
  oauth_tokens    — access + refresh tokens, audience-bound
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from ..store import _resolve_backend

# Short: a code is redeemed within seconds of the redirect in every real flow.
CODE_TTL_SECONDS = 60
# Claude refreshes reactively on 401 and proactively 5 minutes before expiry,
# so an hour is comfortable while keeping a leaked token short-lived.
ACCESS_TTL_SECONDS = 3600
REFRESH_TTL_SECONDS = 60 * 60 * 24 * 90


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def new_secret(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


@dataclass
class Client:
    """A dynamically registered OAuth client (RFC 7591)."""

    client_id: str
    redirect_uris: list[str] = field(default_factory=list)
    client_name: str = ""
    grant_types: list[str] = field(default_factory=lambda: ["authorization_code", "refresh_token"])
    response_types: list[str] = field(default_factory=lambda: ["code"])
    token_endpoint_auth_method: str = "none"  # public client (DCR/CIMD)
    scope: str = ""
    created_at: str = ""

    def to_registration_response(self) -> dict:
        return {
            "client_id": self.client_id,
            "client_id_issued_at": int(
                (_parse(self.created_at) or _now()).timestamp()
            ),
            "redirect_uris": self.redirect_uris,
            "client_name": self.client_name,
            "grant_types": self.grant_types,
            "response_types": self.response_types,
            "token_endpoint_auth_method": self.token_endpoint_auth_method,
            "scope": self.scope,
        }


@dataclass
class AuthCode:
    """A one-time authorization code, bound to everything that issued it.

    Binding matters: the code alone must not be enough. It carries the client,
    the exact redirect URI, the PKCE challenge and the requested resource, and
    the token endpoint re-checks all four.
    """

    code: str
    client_id: str
    redirect_uri: str
    code_challenge: str
    code_challenge_method: str
    scope: str
    resource: str
    expires_at: str
    used: bool = False

    def expired(self) -> bool:
        exp = _parse(self.expires_at)
        return exp is None or exp <= _now()


@dataclass
class Token:
    """An issued access token and its refresh token.

    `resource` is the audience. The MCP route rejects a token whose audience
    isn't itself, which is what stops a token minted for some other service
    being replayed here (MCP spec: access-token privilege restriction).
    """

    access_token: str
    refresh_token: str
    client_id: str
    scope: str
    resource: str
    access_expires_at: str
    refresh_expires_at: str
    created_at: str = ""

    def access_expired(self) -> bool:
        exp = _parse(self.access_expires_at)
        return exp is None or exp <= _now()

    def refresh_expired(self) -> bool:
        exp = _parse(self.refresh_expires_at)
        return exp is None or exp <= _now()

    def to_token_response(self) -> dict:
        remaining = (_parse(self.access_expires_at) or _now()) - _now()
        return {
            "access_token": self.access_token,
            "token_type": "Bearer",
            "expires_in": max(0, int(remaining.total_seconds())),
            "refresh_token": self.refresh_token,
            "scope": self.scope,
        }


class OAuthRepository:
    """Storage contract. Both backends share the expiry rules above."""

    def save_client(self, client: Client) -> Client: raise NotImplementedError
    def get_client(self, client_id: str) -> Optional[Client]: raise NotImplementedError

    def save_code(self, code: AuthCode) -> AuthCode: raise NotImplementedError
    def consume_code(self, code: str) -> Optional[AuthCode]:
        """Atomically fetch and burn a code. Returns None if it is unknown,
        already used or expired — a replayed code must never succeed."""
        raise NotImplementedError

    def save_token(self, token: Token) -> Token: raise NotImplementedError
    def get_by_access_token(self, access_token: str) -> Optional[Token]:
        raise NotImplementedError
    def rotate_refresh_token(self, refresh_token: str) -> Optional[Token]:
        """Atomically invalidate a refresh token and return the record it
        belonged to. OAuth 2.1 requires rotation for public clients."""
        raise NotImplementedError
    def revoke(self, token: str) -> bool: raise NotImplementedError


class InMemoryOAuthRepository(OAuthRepository):
    def __init__(self) -> None:
        self._clients: dict[str, Client] = {}
        self._codes: dict[str, AuthCode] = {}
        self._tokens: dict[str, Token] = {}       # by access token
        self._refresh: dict[str, str] = {}        # refresh -> access
        self._lock = threading.RLock()

    def save_client(self, client):
        with self._lock:
            self._clients[client.client_id] = client
            return client

    def get_client(self, client_id):
        with self._lock:
            return self._clients.get(client_id)

    def save_code(self, code):
        with self._lock:
            self._codes[code.code] = code
            return code

    def consume_code(self, code):
        with self._lock:
            record = self._codes.get(code)
            if record is None:
                return None
            # Burn it whatever happens next: a code presented twice is either a
            # bug or an attack, and neither should get a second chance.
            self._codes.pop(code, None)
            if record.used or record.expired():
                return None
            record.used = True
            return record

    def save_token(self, token):
        with self._lock:
            self._tokens[token.access_token] = token
            self._refresh[token.refresh_token] = token.access_token
            return token

    def get_by_access_token(self, access_token):
        with self._lock:
            return self._tokens.get(access_token)

    def rotate_refresh_token(self, refresh_token):
        with self._lock:
            access = self._refresh.pop(refresh_token, None)
            if access is None:
                return None
            token = self._tokens.pop(access, None)
            if token is None or token.refresh_expired():
                return None
            return token

    def revoke(self, token):
        with self._lock:
            access = self._refresh.pop(token, None) or token
            record = self._tokens.pop(access, None)
            if record is not None:
                self._refresh.pop(record.refresh_token, None)
                return True
            return False


class FirestoreOAuthRepository(OAuthRepository):
    """Durable across Cloud Run cold starts — which is the whole point.

    Code consumption and refresh rotation run in transactions so a replayed
    code or a concurrently-refreshed token cannot both succeed.
    """

    def __init__(self, project: str | None = None, database: str = "(default)") -> None:
        from google.cloud import firestore

        kwargs: dict = {}
        if project:
            kwargs["project"] = project
        if database and database != "(default)":
            kwargs["database"] = database
        self._client = firestore.Client(**kwargs)
        self._firestore = firestore

    def _col(self, name: str):
        return self._client.collection(name)

    # --- clients ---------------------------------------------------------

    def save_client(self, client):
        self._col("oauth_clients").document(client.client_id).set(asdict(client))
        return client

    def get_client(self, client_id):
        snap = self._col("oauth_clients").document(client_id).get()
        return Client(**snap.to_dict()) if snap.exists else None

    # --- codes -----------------------------------------------------------

    def save_code(self, code):
        self._col("oauth_codes").document(code.code).set(asdict(code))
        return code

    def consume_code(self, code):
        doc = self._col("oauth_codes").document(code)

        @self._firestore.transactional
        def _txn(transaction):
            snap = doc.get(transaction=transaction)
            if not snap.exists:
                return None
            record = AuthCode(**(snap.to_dict() or {}))
            transaction.delete(doc)  # single use, unconditionally
            if record.used or record.expired():
                return None
            return record

        return _txn(self._client.transaction())

    # --- tokens ----------------------------------------------------------

    def save_token(self, token):
        self._col("oauth_tokens").document(token.access_token).set(asdict(token))
        self._col("oauth_refresh").document(token.refresh_token).set(
            {"access_token": token.access_token}
        )
        return token

    def get_by_access_token(self, access_token):
        snap = self._col("oauth_tokens").document(access_token).get()
        return Token(**snap.to_dict()) if snap.exists else None

    def rotate_refresh_token(self, refresh_token):
        ref_doc = self._col("oauth_refresh").document(refresh_token)

        @self._firestore.transactional
        def _txn(transaction):
            snap = ref_doc.get(transaction=transaction)
            if not snap.exists:
                return None
            access = (snap.to_dict() or {}).get("access_token")
            transaction.delete(ref_doc)
            if not access:
                return None
            tok_doc = self._col("oauth_tokens").document(access)
            tok_snap = tok_doc.get(transaction=transaction)
            if not tok_snap.exists:
                return None
            token = Token(**(tok_snap.to_dict() or {}))
            transaction.delete(tok_doc)
            return None if token.refresh_expired() else token

        return _txn(self._client.transaction())

    def revoke(self, token):
        for collection, field_name in (("oauth_tokens", None), ("oauth_refresh", "access_token")):
            doc = self._col(collection).document(token)
            snap = doc.get()
            if not snap.exists:
                continue
            if field_name:
                access = (snap.to_dict() or {}).get(field_name)
                if access:
                    self._col("oauth_tokens").document(access).delete()
            doc.delete()
            return True
        return False


_repo: OAuthRepository | None = None
_lock = threading.Lock()


def build_oauth_repository() -> OAuthRepository:
    backend, project = _resolve_backend()
    if backend == "firestore":
        from ..config import settings

        return FirestoreOAuthRepository(project=project, database=settings.firestore_database)
    return InMemoryOAuthRepository()


def get_oauth_store() -> OAuthRepository:
    global _repo
    if _repo is None:
        with _lock:
            if _repo is None:
                _repo = build_oauth_repository()
    return _repo


def reset_oauth_store() -> None:
    global _repo
    with _lock:
        _repo = None


def issue_token(client_id: str, scope: str, resource: str) -> Token:
    now = _now()
    return Token(
        access_token="sna_" + new_secret(),
        refresh_token="snr_" + new_secret(),
        client_id=client_id,
        scope=scope,
        resource=resource,
        access_expires_at=_iso(now + timedelta(seconds=ACCESS_TTL_SECONDS)),
        refresh_expires_at=_iso(now + timedelta(seconds=REFRESH_TTL_SECONDS)),
        created_at=_iso(now),
    )


def issue_code(client_id: str, redirect_uri: str, code_challenge: str,
               code_challenge_method: str, scope: str, resource: str) -> AuthCode:
    return AuthCode(
        code="snc_" + new_secret(),
        client_id=client_id,
        redirect_uri=redirect_uri,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        scope=scope,
        resource=resource,
        expires_at=_iso(_now() + timedelta(seconds=CODE_TTL_SECONDS)),
    )
