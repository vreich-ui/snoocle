"""`OAuthRepository.rotate_refresh_token` — one contract, both backends.

Same convention as tests/test_store.py: the in-memory backend runs everywhere
(hermetic); the Firestore backend runs only when an emulator is reachable
(``FIRESTORE_EMULATOR_HOST``), and otherwise those params skip. The migration
test is Firestore-only — the in-memory backend never had a pre-migration
document shape to migrate FROM.

Run the Firestore params locally with:

    gcloud emulators firestore start --host-port=127.0.0.1:8080
    export FIRESTORE_EMULATOR_HOST=127.0.0.1:8080 GOOGLE_CLOUD_PROJECT=snoocle-test
    pytest tests/test_oauth_store_refresh.py

Why this file exists alongside tests/test_oauth.py's HTTP-level refresh
tests: those prove the route wires the store correctly; these prove the
STORE's own contract — including the one thing only the Firestore backend
can exercise, a refresh record written before this file's migration path
existed (see store.py's `rotate_refresh_token` docstring).
"""

from __future__ import annotations

import os
import uuid

import pytest

from snoocle_server.oauth.store import (
    InMemoryOAuthRepository,
    issue_token,
)


@pytest.fixture(params=["memory", "firestore"])
def repo(request):
    if request.param == "memory":
        yield InMemoryOAuthRepository()
        return
    if not os.environ.get("FIRESTORE_EMULATOR_HOST"):
        pytest.skip("Firestore emulator not running (set FIRESTORE_EMULATOR_HOST)")
    pytest.importorskip("google.cloud.firestore")
    from snoocle_server.oauth.store import FirestoreOAuthRepository

    r = FirestoreOAuthRepository(
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "snoocle-test"),
    )
    suffix = uuid.uuid4().hex[:8]
    # Isolate this run's documents from any other test/run sharing the
    # emulator, and from each other — collection names, not a query filter.
    r._col = lambda name, _s=suffix: r._client.collection(f"{name}_test_{_s}")
    try:
        yield r
    finally:
        for name in ("oauth_tokens", "oauth_refresh", "oauth_clients", "oauth_codes"):
            for doc in r._col(name).list_documents():
                doc.delete()


def _issue(repo, client_id="client-1", scope="mcp", resource="https://x/mcp"):
    tok = issue_token(client_id, scope, resource)
    repo.save_token(tok)
    return tok


# --- (a) access-expired-but-refresh-valid is the live-incident scenario ----


def test_refresh_succeeds_after_access_expiry_but_before_refresh_expiry(repo):
    tok = _issue(repo)
    stored = repo.get_by_access_token(tok.access_token)
    stored.access_expires_at = "2000-01-01T00:00:00+00:00"
    repo.save_token(stored)  # persist the "already expired" access token

    record = repo.rotate_refresh_token(tok.refresh_token)
    assert record is not None
    assert record.client_id == "client-1"
    assert record.scope == "mcp"
    assert record.resource == "https://x/mcp"


# --- (b) rotation is single-use ---------------------------------------------


def test_the_old_refresh_token_is_rejected_after_rotation(repo):
    tok = _issue(repo)
    assert repo.rotate_refresh_token(tok.refresh_token) is not None
    assert repo.rotate_refresh_token(tok.refresh_token) is None


def test_rotation_also_invalidates_the_superseded_access_token(repo):
    """Rotating a refresh token supersedes the access token it was minted
    alongside — a client that refreshes has moved on from it."""
    tok = _issue(repo)
    repo.rotate_refresh_token(tok.refresh_token)
    assert repo.get_by_access_token(tok.access_token) is None


# --- (c) a refresh token past its own expiry is rejected --------------------


def test_a_refresh_token_past_its_own_expiry_is_rejected(repo):
    tok = _issue(repo)
    tok.refresh_expires_at = "2000-01-01T00:00:00+00:00"
    repo.save_token(tok)

    assert repo.rotate_refresh_token(tok.refresh_token) is None


# --- (d) client_id mismatch ---------------------------------------------------


def test_a_client_id_mismatch_is_rejected(repo):
    tok = _issue(repo, client_id="the-real-client")
    assert repo.rotate_refresh_token(tok.refresh_token, client_id="an-impostor") is None


def test_a_client_id_mismatch_does_not_consume_the_token(repo):
    tok = _issue(repo, client_id="the-real-client")
    assert repo.rotate_refresh_token(tok.refresh_token, client_id="an-impostor") is None
    # still good for its rightful owner
    record = repo.rotate_refresh_token(tok.refresh_token, client_id="the-real-client")
    assert record is not None


def test_an_absent_client_id_skips_the_check(repo):
    """Mirrors the authorization_code branch's existing convention (`if
    client_id and client_id != record.client_id`) — omitting it must not
    newly break a caller that never sent one."""
    tok = _issue(repo, client_id="the-real-client")
    assert repo.rotate_refresh_token(tok.refresh_token, client_id="") is not None


# --- an unknown token --------------------------------------------------------


def test_an_unknown_refresh_token_is_rejected(repo):
    assert repo.rotate_refresh_token("snr_never-issued") is None


# --- Firestore-only: a pre-migration document is honoured once, then upgraded


def test_a_legacy_pointer_only_document_is_migrated_on_first_use(repo):
    if not hasattr(repo, "_firestore"):
        pytest.skip("migration path only applies to the Firestore backend's "
                    "pre-migration document shape")
    tok = _issue(repo, client_id="legacy-client")
    # Overwrite with exactly the OLD shape this fix replaces: a bare pointer,
    # nothing of the refresh token's own to validate against.
    repo._col("oauth_refresh").document(tok.refresh_token).set(
        {"access_token": tok.access_token}
    )

    record = repo.rotate_refresh_token(tok.refresh_token, client_id="legacy-client")
    assert record is not None
    assert record.client_id == "legacy-client"
    assert record.refresh_expires_at == tok.refresh_expires_at

    # the old pointer-only doc must not still be sitting there afterward —
    # rotation always writes the new shape or nothing at all
    assert repo.rotate_refresh_token(tok.refresh_token, client_id="legacy-client") is None


def test_a_legacy_pointer_whose_access_doc_is_gone_is_cleanly_rejected(repo):
    """A pre-migration refresh doc has nothing of its own to fall back on if
    the access-token doc it pointed at is already gone — must fail cleanly,
    never crash and never fall back to trusting the record anyway."""
    if not hasattr(repo, "_firestore"):
        pytest.skip("migration path only applies to the Firestore backend's "
                    "pre-migration document shape")
    orphan_refresh = "snr_orphan"
    repo._col("oauth_refresh").document(orphan_refresh).set(
        {"access_token": "sna_never-existed"}
    )
    assert repo.rotate_refresh_token(orphan_refresh) is None
