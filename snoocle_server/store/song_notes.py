"""Per-song reconciliation notes — free-text instructions about how to BUILD a
song, fed to the reconciler as guidance.

Kept out of the Song document on purpose: the Song schema is the iOS-facing
wire contract for song *content*, and workflow state like "the bridge is Bm,
not D" has no business accreting onto it. Same backend resolution as every
other store (Firestore ``song_notes`` collection, one document per songId, or
in-memory).

A song's notes have TWO INDEPENDENT LIFETIMES, stored in two separate fields
of the one document rather than one field with a discriminator:

- ``preference`` — curated through the notes surface (``PUT /v1/songs/{id}/
  notes``, the ``set_song_notes`` MCP tool). A standing instruction about how
  this song should always be built ("capo-free voicings please"): replays as
  guidance on every later analyze that supplies none, until the caller
  changes or clears it.
- ``correction`` — the ``guidance`` string of ONE analyze request ("change the
  C to a B in line 12"). Persisted before any expensive step so a run that
  dies at acquire or reconcile does not eat the user's typed instruction, and
  replayed only until it has landed in a stored version
  (``applied_to_version``, stamped by :func:`SongNotesStore.mark_applied`) —
  after that it is already IN the document the next run starts from, so
  replaying it would re-apply it to a run nobody attached it to.

A single ``kind``-tagged slot (the design this module replaced) cannot hold
both at once: writing a correction wholesale-replaced whatever preference was
there, and once the correction was consumed the song had no effective note at
all — a standing preference does not tolerate being clobbered by every later
`analyze(guidance=...)` call. Two independently-addressable fields fix that:
setting one never touches the other, and each keeps its own lifetime rule.
See :func:`combine_guidance` for what a run does when BOTH are in force.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

from . import _resolve_backend

log = logging.getLogger(__name__)

# The wire contract's ceiling (contract §1). It is a PER-WRITE limit, and it
# bounds EACH SLOT INDEPENDENTLY:
#
# - a PREFERENCE write (`PUT /v1/songs/{id}/notes`, `set_song_notes`), and
# - a CORRECTION write (an analyze request's `guidance`, whether it arrives
#   through `POST /v1/songs/analyze`, `analyze_and_store_song`, or straight
#   into `pipeline.run_pipeline_async`).
#
# Both turn a longer body into a 400/ValueError naming this limit, at the door,
# rather than storing a runaway paste. So the effective guidance a run consumes
# — :func:`combine_guidance`'s result — is at most TWO slots plus the labels
# (2 * MAX_NOTES_CHARS + len("Standing preference: \n\nRequested correction: ")),
# NOT MAX_NOTES_CHARS. That is deliberately stated rather than enforced as a
# third, tighter bound: enforcing a combined ceiling means cutting one half at
# READ time, which throws away characters of a stored instruction that no later
# run can ever replay (`applied_to_version` marks the WHOLE stored text as
# applied, not the slice a prompt happened to see). Bounding each write instead
# refuses the over-long text where the caller is still standing there to see it,
# and nothing downstream ever has to lose any of what it accepted.
MAX_NOTES_CHARS = 8000


def length_error(notes: str | None) -> str | None:
    """The 400/ValueError detail for an over-cap notes write, or ``None`` when
    it fits.

    One function so the preference surfaces (``api.put_song_notes``,
    ``mcp_server.set_song_notes``) and the correction surfaces
    (``api.post_songs_analyze``, ``mcp_server.analyze_and_store_song``,
    ``pipeline.run_pipeline_async``) reject with the SAME message: two slots of
    the same contract that disagreed about how they say "too long" would be two
    contracts.
    """
    n = len(notes or "")
    if n <= MAX_NOTES_CHARS:
        return None
    return f"notes too long: {n} chars (limit {MAX_NOTES_CHARS})"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_preference(text: str) -> dict:
    return {"notes": text, "updated_at": _now()}


def _new_correction(text: str) -> dict:
    # `applied_to_version` exists only on a correction — a preference has
    # nothing to be consumed by.
    return {"notes": text, "updated_at": _now(), "applied_to_version": None}


def _consumable(correction: dict | None, notes: str) -> bool:
    """Whether :meth:`SongNotesStore.mark_applied` may stamp the stored
    ``correction`` sub-record for the ``notes`` a run actually applied.

    Compares against the correction's OWN text only — never a preference, and
    never a combined string a caller might pass by mistake — which is exactly
    why callers must thread the raw correction text through separately from
    whatever combined guidance a run actually saw (see
    :func:`replay_guidance` and ``pipeline._resolve_guidance``).
    """
    return bool(
        correction
        and not correction.get("applied_to_version")
        and (correction.get("notes") or "") == notes
    )


def combine_guidance(preference: str | None, correction: str | None) -> str | None:
    """The deterministic rule for what a run's guidance actually says when a
    durable preference and a pending correction are BOTH in force.

    Both being in force at once is legitimate and probably common: "capo-free
    voicings please" (how to always build this song) and "change the C in
    line 12 to a B" (this document's one wrong chord) are not in tension —
    the reconciler should hear both.

    This is what is IN FORCE, which is not always what a given run acts on:
    a notes-only run is shown the correction alone (its contract is "change
    nothing else"), and a run's ROUTING is decided from the request's own
    correction text and never from this combination. Both of those belong to
    the caller that knows the scope and the origin — see ``pipeline.py``'s
    ``model_guidance`` and ``request_correction``.

    Order is fixed (preference, then correction) and each half a fixed,
    explicit label:

        Standing preference: <preference text>

        Requested correction: <correction text>

    Rationale for the labels: a bare concatenation hands a downstream reader
    (the LLM prompt, a provenance/log line, a human staring at a stored
    record) one string with no way to tell which half is the standing
    instruction and which is the one-time fix — silently recreating the
    "one slot, two lifetimes" confusion this module exists to end, just one
    level higher up a merged blob could still be mistaken for a preference on
    a later read. The label makes that impossible.

    When only ONE of the two is present, it is returned bare (no label,
    unchanged from every existing single-note behavior) — there is nothing to
    disambiguate yet.

    Nothing is cut here, deliberately. Both halves are bounded by
    :data:`MAX_NOTES_CHARS` where each is WRITTEN (see that constant), so this
    function joins two already-bounded strings and the result is bounded by
    construction. Trimming at read time instead was strictly worse: the cut
    text is still in the store, ``applied_to_version`` marks the WHOLE stored
    correction applied against the version a truncated prompt produced, and the
    remainder is therefore never replayed — a silent, permanent loss of
    characters the caller typed, on the one path this module goes out of its way
    never to lose (``pipeline._resolve_guidance`` persists a correction before
    any expensive step for exactly that reason).
    """
    if preference and correction:
        return f"Standing preference: {preference}\n\nRequested correction: {correction}"
    return preference or correction or None


def preference_text(record: dict | None) -> str | None:
    """The standing preference in force in ``record``, or ``None``.

    Separate from :func:`replay_guidance` because a caller can need to know
    WHICH halves were in force without needing the combination — the pipeline's
    report step says "preference + correction combined" or "preference held
    back", and deriving that by comparing strings is how it came to claim a
    preference existed when none did.
    """
    return ((record or {}).get("preference") or {}).get("notes", "").strip() or None


def replay_guidance(record: dict | None) -> tuple[str | None, str | None]:
    """What a run that supplies NO guidance of its own should treat as this
    song's stored guidance, and separately, the raw pending-correction text
    (if any) that fed into it.

    Returns ``(combined_guidance, pending_correction_text)``. The second value
    is what a caller must pass to :meth:`SongNotesStore.mark_applied` if this
    replay is the one that ends up storing a version — NEVER the first value,
    because once a preference is combined in, the first value no longer
    equals the correction's own stored text and the compare-and-set in
    :func:`_consumable` would never match, silently breaking the single-shot
    guarantee.

    A durable preference always replays. A pending correction replays only
    until :meth:`SongNotesStore.mark_applied` has stamped it with the version
    it landed in — a correction already applied is already IN the document
    the next run starts from, and replaying it again would silently re-apply
    a previous request's edit (and, before the pipeline's own origin check,
    reclassify that later run as notes-only).
    """
    if not record:
        return None, None

    preference = preference_text(record)

    correction_rec = record.get("correction") or {}
    correction = (correction_rec.get("notes") or "").strip() or None
    if correction and correction_rec.get("applied_to_version"):
        correction = None  # already spent — nothing left to replay

    return combine_guidance(preference, correction), correction


class SongNotesStore:
    def get(self, song_id: str) -> str:  # pragma: no cover - interface
        """Convenience: the combined text a no-guidance run would replay right
        now (see :func:`replay_guidance`), or ``""``. Mainly for tests/tools —
        production call sites need :meth:`get_record` (they need each half's
        own state — updated_at, applied_to_version — not just combined text)."""
        raise NotImplementedError

    def get_record(self, song_id: str) -> dict | None:  # pragma: no cover - interface
        """``{"preference": {...} | None, "correction": {...} | None}``, or
        ``None`` when NEITHER is stored for this song id."""
        raise NotImplementedError

    def set_preference(self, song_id: str, notes: str) -> str:  # pragma: no cover - interface
        """Replace the durable preference (empty/whitespace clears it). Never
        touches ``correction`` — this independence is the whole fix."""
        raise NotImplementedError

    def set_correction(self, song_id: str, notes: str) -> str:  # pragma: no cover - interface
        """Replace the pending correction (empty/whitespace clears it). Still
        single-shot: a fresh correction replaces whatever unconsumed one was
        there (only one may ever be outstanding), but never touches
        ``preference``.

        Does NOT itself enforce :data:`MAX_NOTES_CHARS`, and must not: the only
        caller is ``pipeline._resolve_guidance``, whose whole surrounding
        try/except exists to make a notes-store failure non-fatal to a run — a
        raise from here would be swallowed there and the run would carry on
        with the over-cap text anyway. The cap is checked at the pipeline's
        door instead (``pipeline.run_pipeline_async``, plus the REST/MCP
        surfaces), before any expensive step. See :func:`length_error`.
        """
        raise NotImplementedError

    def mark_applied(  # pragma: no cover - interface
        self, song_id: str, notes: str, version: str
    ) -> bool:
        """Stamp the stored PENDING CORRECTION as applied to ``version`` —
        the single-shot consumption.

        Compare-and-set on the correction's own text: a no-op (False) when
        there is no pending correction, when it has already been consumed, or
        when the caller replaced it while the run was in flight — a note
        typed during a run belongs to the NEXT one, not to the run that never
        saw it. Never touches ``preference``.
        """
        raise NotImplementedError

    def delete(self, song_id: str) -> bool:  # pragma: no cover - interface
        """Clear BOTH slots — the whole stored slate for this song id, so a
        later run replays nothing. Returns whether anything was there.

        The ONLY delete. There is deliberately no per-slot delete: both the
        REST and MCP surfaces state that a partial clear is not offered (a
        stray pending correction is rare and the next successful analyze
        consumes it on its own), and either slot is already cleared by writing
        it empty — `set_preference(id, "")` / `set_correction(id, "")`. A
        second, narrower delete with no caller was one more implementation
        each backend had to keep honest for nothing.
        """
        raise NotImplementedError


class InMemorySongNotesStore(SongNotesStore):
    def __init__(self) -> None:
        self._docs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, song_id: str) -> str:
        combined, _ = replay_guidance(self.get_record(song_id))
        return combined or ""

    def get_record(self, song_id: str) -> dict | None:
        with self._lock:
            doc = self._docs.get(song_id)
            if not doc:
                return None
            return {
                "preference": dict(doc["preference"]) if doc.get("preference") else None,
                "correction": dict(doc["correction"]) if doc.get("correction") else None,
            }

    def _prune_locked(self, song_id: str, doc: dict) -> None:
        # "no notes in either slot" and "no document" are the same state to
        # every reader, so keeping an empty doc around would only leave a
        # tombstone to explain.
        if not doc.get("preference") and not doc.get("correction"):
            self._docs.pop(song_id, None)

    def set_preference(self, song_id: str, notes: str) -> str:
        text = (notes or "").strip()
        with self._lock:
            doc = self._docs.setdefault(song_id, {})
            if not text:
                doc.pop("preference", None)
                self._prune_locked(song_id, doc)
                return ""
            doc["preference"] = _new_preference(text)
            return text

    def set_correction(self, song_id: str, notes: str) -> str:
        text = (notes or "").strip()
        with self._lock:
            doc = self._docs.setdefault(song_id, {})
            if not text:
                doc.pop("correction", None)
                self._prune_locked(song_id, doc)
                return ""
            doc["correction"] = _new_correction(text)
            return text

    def mark_applied(self, song_id: str, notes: str, version: str) -> bool:
        text = (notes or "").strip()
        with self._lock:
            doc = self._docs.get(song_id)
            correction = (doc or {}).get("correction")
            if not _consumable(correction, text):
                return False
            correction["applied_to_version"] = version
            return True

    def delete(self, song_id: str) -> bool:
        with self._lock:
            return self._docs.pop(song_id, None) is not None


class FirestoreSongNotesStore(SongNotesStore):
    _COLLECTION = "song_notes"

    def __init__(self, project: str | None = None, database: str = "(default)") -> None:
        from google.cloud import firestore

        kwargs: dict = {}
        if project:
            kwargs["project"] = project
        if database and database != "(default)":
            kwargs["database"] = database
        self._client = firestore.Client(**kwargs)

    def _ref(self, song_id: str):
        return self._client.collection(self._COLLECTION).document(song_id)

    def get(self, song_id: str) -> str:
        combined, _ = replay_guidance(self.get_record(song_id))
        return combined or ""

    def get_record(self, song_id: str) -> dict | None:
        snap = self._ref(song_id).get()
        if not snap.exists:
            return None
        doc = snap.to_dict() or {}
        preference = doc.get("preference") or None
        correction = doc.get("correction") or None
        if preference is None and correction is None:
            return None
        return {"preference": preference, "correction": correction}

    def set_preference(self, song_id: str, notes: str) -> str:
        text = (notes or "").strip()
        ref = self._ref(song_id)
        if not text:
            # A plain null write, not `DELETE_FIELD`: it works identically
            # whether or not the document (or even the `correction` sibling
            # field) already exists, with no nonexistent-document edge case
            # to reason about. `get_record` already treats a null/absent
            # field as "nothing here" for any reader.
            ref.set({"preference": None}, merge=True)
            self._prune_if_empty(song_id)
            return ""
        # merge=True touches ONLY the `preference` field of the document — a
        # pending `correction` sibling is untouched. This sibling-field
        # independence (not a doc replace) is the actual fix: the old
        # single-slot design's `set()` replaced the whole document, so
        # writing one lifetime always clobbered the other.
        ref.set({"preference": _new_preference(text)}, merge=True)
        return text

    def set_correction(self, song_id: str, notes: str) -> str:
        text = (notes or "").strip()
        ref = self._ref(song_id)
        if not text:
            ref.set({"correction": None}, merge=True)
            self._prune_if_empty(song_id)
            return ""
        # Still single-shot: this REPLACES whatever unconsumed correction was
        # there (only one may ever be outstanding), so a previous
        # correction's `applied_to_version` can never linger and consume the
        # one that replaced it — same guarantee the old design had, just
        # scoped to this one field instead of the whole document.
        ref.set({"correction": _new_correction(text)}, merge=True)
        return text

    def _prune_if_empty(self, song_id: str) -> None:
        # Storage hygiene only, never required for correctness: `get_record`
        # already treats an all-null document as "no notes" for any reader.
        # This just keeps a song that once had notes and now has none from
        # leaving an empty shell behind forever.
        if self.get_record(song_id) is None:
            self._ref(song_id).delete()

    def mark_applied(self, song_id: str, notes: str, version: str) -> bool:
        text = (notes or "").strip()
        ref = self._ref(song_id)
        snap = ref.get()
        doc = snap.to_dict() if snap.exists else None
        correction = (doc or {}).get("correction")
        if not _consumable(correction, text):
            return False
        # Dotted field-path update: touches only `correction.applied_to_
        # version`, leaving `correction.notes`/`updated_at` and the sibling
        # `preference` field untouched.
        ref.update({"correction.applied_to_version": version})
        return True

    def delete(self, song_id: str) -> bool:
        # A full physical delete (not a null-out of both fields): DELETE
        # clears the whole stored slate for this song id. `existed` reads the
        # FUNCTIONAL state (`get_record`), not raw document existence, so an
        # empty shell left over from clearing both fields individually never
        # counts as "there was something to delete".
        existed = self.get_record(song_id) is not None
        self._ref(song_id).delete()
        return existed


_store: SongNotesStore | None = None
_lock = threading.Lock()


def build_song_notes_store() -> SongNotesStore:
    backend, project = _resolve_backend()
    if backend == "firestore":
        from ..config import settings

        return FirestoreSongNotesStore(project=project, database=settings.firestore_database)
    return InMemorySongNotesStore()


def get_song_notes_store() -> SongNotesStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = build_song_notes_store()
                log.info("song notes store backend: %s", type(_store).__name__)
    return _store


def reset_song_notes_store() -> None:
    global _store
    with _lock:
        _store = None
