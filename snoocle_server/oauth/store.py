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
  oauth_tokens    — one doc per ACCESS token (audience-bound, ~1h TTL)
  oauth_refresh   — one doc per REFRESH token (~90d TTL). Independently
                    validated: its validity must never be decided by looking
                    at the access token's own (much shorter) expiry — see
                    `RefreshRecord` below and `rotate_refresh_token`.

`oauth_refresh` docs carry their OWN `refresh_expires_at`, `client_id`,
`scope` and `resource` — not just a pointer to the access-token doc. A
refresh must be answerable from the refresh record alone: if it instead
joined through the access-token doc for those fields (as this store did
before), the refresh's effective lifetime would quietly become "however
long that other doc happens to still exist" rather than its own 90 days,
and the failure mode is silent — the endpoint still returns a well-formed
`invalid_grant`, it is just wrong about WHY the token is invalid.
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


@dataclass
class RefreshRecord:
    """A refresh token's OWN record — everything needed to validate and
    honour it without consulting the access-token doc it currently points
    at. `access_token` is kept only so rotation can clean that doc up; it is
    never read to decide whether THIS token is still good.
    """

    refresh_token: str
    access_token: str
    client_id: str
    scope: str
    resource: str
    refresh_expires_at: str
    created_at: str = ""

    def expired(self) -> bool:
        exp = _parse(self.refresh_expires_at)
        return exp is None or exp <= _now()

    @classmethod
    def from_token(cls, token: Token) -> "RefreshRecord":
        return cls(
            refresh_token=token.refresh_token,
            access_token=token.access_token,
            client_id=token.client_id,
            scope=token.scope,
            resource=token.resource,
            refresh_expires_at=token.refresh_expires_at,
            created_at=token.created_at,
        )


class OAuthRepository:
    """Storage contract. Both backends share the expiry rules above."""

    def save_client(self, client: Client) -> Client: raise NotImplementedError
    def get_client(self, client_id: str) -> Optional[Client]: raise NotImplementedError
    def list_clients(self) -> list[Client]:
        """Every registered client, newest first.

        Registration is open by design — that is what DCR means — so the only
        defence against a stray or hostile registration accumulating unnoticed
        is being able to see the list.
        """
        raise NotImplementedError
    def delete_client(self, client_id: str) -> bool:
        """Forget a client AND every token it holds.

        Deleting the registration alone would leave live access tokens working
        for up to an hour and refresh tokens for 90 days, which is the opposite
        of what someone clicking "revoke" means.
        """
        raise NotImplementedError

    def save_code(self, code: AuthCode) -> AuthCode: raise NotImplementedError
    def consume_code(self, code: str) -> Optional[AuthCode]:
        """Atomically fetch and burn a code. Returns None if it is unknown,
        already used or expired — a replayed code must never succeed."""
        raise NotImplementedError

    def save_token(self, token: Token) -> Token:
        """Persist an access token AND write its refresh token's own
        independent record (RefreshRecord) — never just a pointer."""
        raise NotImplementedError

    def get_by_access_token(self, access_token: str) -> Optional[Token]:
        raise NotImplementedError

    def rotate_refresh_token(
        self, refresh_token: str, client_id: str = ""
    ) -> Optional[RefreshRecord]:
        """Atomically validate and consume a refresh token, returning the
        RefreshRecord it belonged to (the caller mints a fresh access+refresh
        pair from its client_id/scope/resource) — or None.

        Validity is decided ENTIRELY from the refresh record itself:
        `record.expired()` against its own `refresh_expires_at`. The linked
        access token's expiry is never consulted — a lapsed 1-hour access
        token must not make a 90-day refresh token look expired too.

        `client_id`, when given, must match the record's or the token is
        rejected — but NOT consumed: a request from the wrong client must
        not be able to burn a token that still belongs to its rightful
        owner. Only "missing", "expired", or "this client's" lead to a
        rotation (delete-and-reissue); everything else leaves the stored
        record untouched.
        """
        raise NotImplementedError

    def revoke(self, token: str) -> bool: raise NotImplementedError


class InMemoryOAuthRepository(OAuthRepository):
    def __init__(self) -> None:
        self._clients: dict[str, Client] = {}
        self._codes: dict[str, AuthCode] = {}
        self._tokens: dict[str, Token] = {}                 # by access token
        self._refresh: dict[str, RefreshRecord] = {}         # by refresh token
        self._lock = threading.RLock()

    def save_client(self, client):
        with self._lock:
            self._clients[client.client_id] = client
            return client

    def get_client(self, client_id):
        with self._lock:
            return self._clients.get(client_id)

    def list_clients(self):
        with self._lock:
            return sorted(self._clients.values(),
                          key=lambda c: c.created_at or "", reverse=True)

    def delete_client(self, client_id):
        with self._lock:
            if self._clients.pop(client_id, None) is None:
                return False
            for token in [t for t in self._tokens.values() if t.client_id == client_id]:
                self._tokens.pop(token.access_token, None)
                self._refresh.pop(token.refresh_token, None)
            for code, record in list(self._codes.items()):
                if record.client_id == client_id:
                    self._codes.pop(code, None)
            return True

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
            self._refresh[token.refresh_token] = RefreshRecord.from_token(token)
            return token

    def get_by_access_token(self, access_token):
        with self._lock:
            return self._tokens.get(access_token)

    def rotate_refresh_token(self, refresh_token, client_id=""):
        with self._lock:
            record = self._refresh.get(refresh_token)
            if record is None:
                return None
            if record.expired():
                # An expired record found is a record we clean up here — but
                # this is CONSUMPTION, not the "wrong client" non-consuming
                # path below, so it happens unconditionally.
                self._refresh.pop(refresh_token, None)
                self._tokens.pop(record.access_token, None)
                return None
            if client_id and record.client_id != client_id:
                return None  # not consumed — still valid for its own client
            self._refresh.pop(refresh_token, None)
            self._tokens.pop(record.access_token, None)
            return record

    def revoke(self, token):
        with self._lock:
            refresh_record = self._refresh.pop(token, None)
            access = refresh_record.access_token if refresh_record is not None else token
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

    def list_clients(self):
        clients = [Client(**(s.to_dict() or {})) for s in self._col("oauth_clients").stream()]
        clients.sort(key=lambda c: c.created_at or "", reverse=True)
        return clients

    def delete_client(self, client_id):
        doc = self._col("oauth_clients").document(client_id)
        if not doc.get().exists:
            return False
        # Tokens first: if this half fails, the client is still listed and the
        # operator can retry. The other order would leave live tokens behind
        # with nothing in the UI pointing at them.
        for snap in self._col("oauth_tokens").where("client_id", "==", client_id).stream():
            record = snap.to_dict() or {}
            refresh = record.get("refresh_token")
            if refresh:
                self._col("oauth_refresh").document(refresh).delete()
            snap.reference.delete()
        # A refresh record now carries its own client_id, so one whose access
        # token doc is already gone (rotated away, or a still-unmigrated
        # legacy pointer) is still reachable directly — not just via the join
        # above.
        for snap in self._col("oauth_refresh").where("client_id", "==", client_id).stream():
            snap.reference.delete()
        for snap in self._col("oauth_codes").where("client_id", "==", client_id).stream():
            snap.reference.delete()
        doc.delete()
        return True

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
        # The refresh record is independent, not a pointer: it carries its
        # own client_id/scope/resource/refresh_expires_at so a refresh is
        # answerable without ever reading the access-token doc.
        self._col("oauth_refresh").document(token.refresh_token).set(
            asdict(RefreshRecord.from_token(token))
        )
        return token

    def get_by_access_token(self, access_token):
        snap = self._col("oauth_tokens").document(access_token).get()
        return Token(**snap.to_dict()) if snap.exists else None

    def rotate_refresh_token(self, refresh_token, client_id=""):
        ref_doc = self._col("oauth_refresh").document(refresh_token)

        @self._firestore.transactional
        def _txn(transaction):
            # All reads first, all writes last — the two documents this
            # touches are read here in full before anything is deleted, so a
            # failed validation (wrong client, or a legacy doc that can't be
            # migrated) never leaves a half-applied transaction behind.
            snap = ref_doc.get(transaction=transaction)
            if not snap.exists:
                return None
            data = snap.to_dict() or {}
            # Pre-migration shape: `{"access_token": ...}` only, nothing of
            # its own to validate against. Join through the access-token doc
            # ONCE to backfill a real record — rotation always writes the new
            # shape, so a given token only ever takes this branch once.
            migrating = "refresh_expires_at" not in data
            access = data.get("access_token") or ""
            access_doc = self._col("oauth_tokens").document(access) if access else None
            access_snap = access_doc.get(transaction=transaction) if access_doc else None

            if migrating:
                if access_snap is None or not access_snap.exists:
                    transaction.delete(ref_doc)
                    return None
                tok = access_snap.to_dict() or {}
                record = RefreshRecord(
                    refresh_token=refresh_token, access_token=access,
                    client_id=tok.get("client_id", ""), scope=tok.get("scope", ""),
                    resource=tok.get("resource", ""),
                    refresh_expires_at=tok.get("refresh_expires_at", ""),
                )
            else:
                record = RefreshRecord(**data)

            if record.expired():
                transaction.delete(ref_doc)
                if access_snap is not None and access_snap.exists:
                    transaction.delete(access_doc)
                return None
            if client_id and record.client_id != client_id:
                return None  # not consumed — still valid for its own client

            transaction.delete(ref_doc)
            if access_snap is not None and access_snap.exists:
                transaction.delete(access_doc)
            return record

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
