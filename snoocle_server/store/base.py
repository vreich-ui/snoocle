"""Song-store repository interface + shared, backend-agnostic helpers.

All persistence lives behind :class:`SongRepository`. Two backends implement
it: Firestore (durable, used on Cloud Run) and an in-memory store (hermetic,
used by tests and local dev). Both share the same version model so their
behavior — version shas, optimistic locking, append-only provenance, diffs —
is identical:

- **version sha** = first 12 hex of the sha256 of the song's canonical
  (sorted-key, compact) JSON. Same song content -> same sha, regardless of
  backend or when it was written.
- **optimistic locking**: ``save(..., expected_version=...)`` rejects the write
  with :class:`VersionConflictError` when the stored ``latestVersion`` differs
  from what the caller last read.
- **append-only provenance**: a save whose provenance does not extend the
  stored song's provenance is rejected with :class:`StoreError`.
- each version snapshot records its ``parent`` (the sha it was written on top
  of, or ``None`` for the first), so history is a walkable chain.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..schema import Song

_EPSILON = timedelta(microseconds=1)


class StoreError(RuntimeError):
    pass


class VersionConflictError(StoreError):
    """The record changed since the caller last read it (optimistic-lock miss)."""


class StoreUnavailableError(RuntimeError):
    """The store backend is unreachable or misconfigured — e.g. the Firestore
    database doesn't exist, or the runtime lacks credentials/permissions.

    Deliberately NOT a subclass of StoreError so callers that map StoreError to
    "song not found" (404) don't mistake "backend down" for a missing song: it
    routes to its own 503 instead.
    """


@dataclass
class SongVersion:
    version: str  # content sha
    timestamp: str  # ISO-8601 UTC
    message: str


@dataclass
class SaveResult:
    song_id: str
    version: str  # content sha of the new version
    timestamp: str  # ISO-8601 UTC of the write
    message: str


@dataclass
class YouTubeCookieRecord:
    """Non-secret status of the stored YouTube cookies (never the cookies)."""

    updated_at: str
    source: str  # e.g. "app" (in-app sign-in) | "api" (manual upload)
    line_count: int  # number of cookie entries — a coarse "is it populated" signal


def count_cookie_lines(cookies_txt: str) -> int:
    return sum(
        1 for ln in cookies_txt.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    )


def canonical_json(song: Song) -> str:
    """Deterministic, sorted-key, compact JSON — the hashing input."""
    return json.dumps(song.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def version_sha(song: Song) -> str:
    """Short content hash: first 12 hex of sha256(canonical JSON)."""
    return hashlib.sha256(canonical_json(song).encode("utf-8")).hexdigest()[:12]


def pretty_json(song_dict: dict) -> str:
    """Human-readable, stable rendering used for unified diffs (indent=2,
    sort_keys) — matches the spec's diff format exactly."""
    return json.dumps(song_dict, indent=2, sort_keys=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def next_timestamp(previous_iso: str | None) -> str:
    """A write timestamp guaranteed to be strictly greater than the previous
    version's, so ``versions`` ordered by timestamp desc is deterministic even
    for rapid back-to-back saves (mock analyses commit in well under a
    microsecond apart)."""
    now = datetime.now(timezone.utc)
    if previous_iso:
        try:
            prev = datetime.fromisoformat(previous_iso)
        except ValueError:
            prev = None
        if prev is not None and now <= prev:
            now = prev + _EPSILON
    return now.isoformat()


def check_provenance_append_only(old_song_dict: dict, new_song: Song) -> None:
    """Reject a save that rewrites history instead of extending it."""
    old_prov = old_song_dict.get("provenance", [])
    new_prov = [p.model_dump(mode="json") for p in new_song.provenance]
    if new_prov[: len(old_prov)] != old_prov:
        raise StoreError(
            f"song {new_song.id!r}: provenance is append-only; the new version "
            "must extend the stored provenance history"
        )


def unified_song_diff(a_dict: dict, b_dict: dict, a_label: str, b_label: str) -> str:
    """Unified diff over pretty-printed (indent=2, sort_keys) JSON of two song
    snapshots — the text/plain body of GET /v1/songs/{id}/diff."""
    import difflib

    a_lines = pretty_json(a_dict).splitlines(keepends=True)
    b_lines = pretty_json(b_dict).splitlines(keepends=True)
    diff = difflib.unified_diff(a_lines, b_lines, fromfile=a_label, tofile=b_label)
    text = "".join(diff)
    if text and not text.endswith("\n"):
        text += "\n"
    return text


class SongRepository(ABC):
    """Persistence contract for songs and their version history.

    Implementations MUST enforce optimistic locking and append-only provenance
    identically (see module docstring); the shared helpers above exist so they
    can.
    """

    @abstractmethod
    def list_songs(self) -> list[str]:
        """All song ids present, sorted."""

    @abstractmethod
    def get(self, song_id: str, version: str | None = None) -> Song:
        """The latest song, or a specific version sha. Raises StoreError if the
        song (or that version) is not found."""

    @abstractmethod
    def versions(self, song_id: str) -> list[SongVersion]:
        """Version history, newest first. Empty list if the song is unknown."""

    @abstractmethod
    def current_version(self, song_id: str) -> str | None:
        """The latest version sha, or None if the song doesn't exist."""

    @abstractmethod
    def diff(self, song_id: str, version_a: str, version_b: str) -> str:
        """Unified diff (text/plain) between two versions of a song."""

    @abstractmethod
    def save(
        self,
        song: Song,
        message: str,
        expected_version: str | None = None,
        enforce_expected: bool = False,
    ) -> SaveResult:
        """Persist a new version.

        expected_version: pass the version you read for optimistic locking; a
        mismatch with the stored latestVersion raises VersionConflictError.
        enforce_expected: when True, a None expected_version is only valid for a
        song that does not exist yet (strict create-or-CAS semantics).
        """

    # --- YouTube acquisition cookies (durable server config) -------------
    # Lets the iOS app (or a manual upload) hand the server a signed-in
    # cookies.txt so yt-dlp can get past YouTube's datacenter bot-check, and
    # refresh it later without a redeploy.

    @abstractmethod
    def set_youtube_cookies(self, cookies_txt: str, source: str) -> YouTubeCookieRecord:
        """Persist a cookies.txt; returns its (non-secret) status."""

    @abstractmethod
    def get_youtube_cookies_txt(self) -> str | None:
        """The stored cookies.txt for yt-dlp, or None. (Secret — server-side use.)"""

    @abstractmethod
    def youtube_cookies_status(self) -> YouTubeCookieRecord | None:
        """Non-secret status of the stored cookies, or None if unset."""

    @abstractmethod
    def clear_youtube_cookies(self) -> None:
        """Remove any stored cookies."""
