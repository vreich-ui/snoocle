"""Audit stored song ids against what identity.py would resolve today.

Song ids are permanent, content-hash-versioned store keys minted by
:mod:`..identity` at analyze time (:func:`..schema.song.slugify_song_id`).
Before PR #45/#51 tightened identity resolution, a handful of documents were
minted under an id derived from a bad guess — a channel name, cover phrasing
read as the artist, an unstripped upload description. Ids cannot be renamed
without orphaning the version history they carry, so nothing here ever
attempts that: it only detects the mismatch, and — on explicit request —
re-analyzes the recording under whatever id ``identity.py`` resolves TODAY,
then points the old document forward with ``Song.supersededBy`` rather than
deleting or rewriting it.

Three independent operations, deliberately kept separate:

- :func:`find_mismatched_identities` — REPORT ONLY. What is stored, what
  ``identity.py`` would resolve now from the stored ``audio.youtubeVideoId``,
  and whether they agree. Never re-analyzes, never writes; as safe to run
  against production as listing songs. A song with no video id to check, or
  whose id already matches, is skipped — it has nothing to report.
- :func:`supersede_song` — the forward pointer itself. One new (append-only)
  version of the OLD id, never a delete: the old content is untouched, the
  old id stays fully readable at every version it ever had, and gains one
  provenance entry plus ``supersededBy``.
- :func:`reanalyze_and_supersede` — the operator-triggered action. Dry run
  unless ``confirm=True``: even then, it re-derives the identity first and
  refuses outright (no pipeline call, no write) when the old id is
  gold-marked, already superseded, has no video id, or already matches —
  "report only, do not decide" extends to never silently relocating a gold
  pointer. Moving that is left to a human.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from ..identity import IdentityError, SongIdentity, resolve_identity
from ..schema.song import ProvenanceEntry, Song, slugify_song_id
from .base import SaveResult, SongRepository, StoreError

_ACTOR = "snoocle-server/maintenance"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_video_identity(video_id: str) -> SongIdentity:
    """What ``identity.py`` would resolve TODAY for a stored YouTube video id
    — a fresh metadata fetch (title/channel/structured fields may have
    changed since the document was first analyzed), then the same refuses-
    rather-than-guesses resolver the pipeline itself uses. Raises
    :class:`IdentityError` exactly as a fresh analyze run would."""
    from ..audio.acquire import extract_metadata

    meta = extract_metadata(video_id)
    return resolve_identity(
        video_title=meta.video_title,
        channel=meta.uploader,
        track=meta.track,
        track_artist=meta.track_artist,
    )


@dataclass
class IdentityCheck:
    """One song's id vs. what identity.py resolves today, or why it
    couldn't be checked at all."""

    song_id: str
    video_id: Optional[str]
    resolved_id: Optional[str]
    confidence: Optional[float]
    matches: bool
    # "match" | "mismatch" | "no_video_id" | "unresolvable" | "error"
    status: str
    detail: Optional[str] = None

    def describe(self) -> str:
        if self.status == "match":
            return f"{self.song_id}: matches (confidence {self.confidence:.2f})"
        if self.status == "no_video_id":
            return f"{self.song_id}: no audio.youtubeVideoId stored — nothing to check"
        if self.status == "unresolvable":
            return f"{self.song_id}: identity.py still can't resolve this today ({self.detail})"
        if self.status == "error":
            return f"{self.song_id}: could not check ({self.detail})"
        return (
            f"{self.song_id} -> {self.resolved_id} "
            f"(confidence {self.confidence:.2f})"
        )


def check_song_identity(
    song: Song, resolve: Callable[[str], SongIdentity] = resolve_video_identity
) -> IdentityCheck:
    """Compare one song's stored id against what `resolve` (identity.py's
    resolver, by default) would mint today from its stored video id."""
    video_id = song.audio.youtubeVideoId if song.audio else None
    if not video_id:
        return IdentityCheck(
            song_id=song.id, video_id=None, resolved_id=None, confidence=None,
            matches=True, status="no_video_id",
        )
    try:
        identity = resolve(video_id)
    except IdentityError as e:
        return IdentityCheck(
            song_id=song.id, video_id=video_id, resolved_id=None, confidence=None,
            matches=False, status="unresolvable", detail=str(e),
        )
    except Exception as e:  # noqa: BLE001 — a fetch failure is a report line, not a crash
        return IdentityCheck(
            song_id=song.id, video_id=video_id, resolved_id=None, confidence=None,
            matches=False, status="error", detail=str(e),
        )
    resolved_id = slugify_song_id(identity.artist, identity.title)
    matches = resolved_id == song.id
    return IdentityCheck(
        song_id=song.id, video_id=video_id, resolved_id=resolved_id,
        confidence=identity.confidence, matches=matches,
        status="match" if matches else "mismatch",
    )


def find_mismatched_identities(
    store: SongRepository,
    *,
    resolve: Callable[[str], SongIdentity] = resolve_video_identity,
    song_ids: Optional[list[str]] = None,
) -> list[IdentityCheck]:
    """Every stored song whose id does NOT match what identity.py would
    resolve today. REPORT ONLY: never writes, never re-analyzes.

    A song with no ``audio.youtubeVideoId`` to check, or whose id already
    matches, is skipped entirely — it has nothing to report. A song that
    could not be checked (an unreadable document, a metadata fetch failure,
    identity.py still refusing today) IS included, since that is itself
    worth an operator's attention.
    """
    ids = song_ids if song_ids is not None else store.list_songs()
    out: list[IdentityCheck] = []
    for song_id in ids:
        try:
            song = store.get(song_id)
        except StoreError as e:
            out.append(
                IdentityCheck(
                    song_id=song_id, video_id=None, resolved_id=None, confidence=None,
                    matches=False, status="error", detail=str(e),
                )
            )
            continue
        check = check_song_identity(song, resolve)
        if check.status in ("match", "no_video_id"):
            continue
        out.append(check)
    return out


def supersede_song(
    store: SongRepository, old_id: str, new_id: str, *, reason: Optional[str] = None
) -> SaveResult:
    """Point `old_id`'s document forward to `new_id`, WITHOUT deleting or
    rewriting its content: one new (append-only) version of the OLD id,
    carrying `supersededBy` and one provenance entry. Every prior version of
    `old_id` — including the one just before this — remains exactly as it
    was and fully readable; the store is append-only by design and old
    versions are the only record of how the library got here.

    Idempotent: calling this again with the same `new_id` is a no-op (the
    content is unchanged, so the store's own idempotent-save behavior
    applies) rather than growing a duplicate provenance entry.
    """
    old_song = store.get(old_id)
    if old_song.supersededBy == new_id:
        return SaveResult(
            song_id=old_id,
            version=store.current_version(old_id) or "",
            timestamp=_now_iso(),
            message=f"already superseded by {new_id!r}",
        )
    entry = ProvenanceEntry(
        timestamp=_now_iso(),
        actor=_ACTOR,
        action="superseded",
        sources=[new_id],
        notes=reason or (
            f"re-analyzed under identity.py's current resolution; "
            f"superseded by {new_id!r}"
        ),
    )
    updated = old_song.model_copy(
        update={
            "supersededBy": new_id,
            "provenance": list(old_song.provenance) + [entry],
        }
    )
    updated = Song.model_validate(updated.model_dump())
    return store.save(
        updated,
        message=f"Superseded by {new_id}",
        expected_version=store.current_version(old_id),
    )


@dataclass
class ReanalyzeReport:
    """The outcome of one `reanalyze_and_supersede` call — always populated
    enough to explain itself, whether it acted, refused, or only previewed."""

    old_id: str
    dry_run: bool = True
    refused: Optional[str] = None
    resolved_id: Optional[str] = None  # identity.py's answer today (the preview)
    confidence: Optional[float] = None
    new_id: Optional[str] = None  # the id the REAL pipeline run actually landed on
    new_song_version: Optional[str] = None
    superseded_version: Optional[str] = None  # old_id's post-supersede version
    steps: dict = field(default_factory=dict)  # the pipeline run's own step report

    @property
    def acted(self) -> bool:
        return self.superseded_version is not None

    def describe(self) -> str:
        if self.refused:
            return f"{self.old_id}: refused — {self.refused}"
        if self.dry_run:
            return (
                f"{self.old_id}: would re-analyze -> {self.resolved_id} "
                f"(confidence {self.confidence:.2f}); re-run with --confirm to act"
            )
        return (
            f"{self.old_id}: re-analyzed -> {self.new_id} "
            f"(version {self.new_song_version}); {self.old_id} superseded "
            f"(version {self.superseded_version})"
        )


def _default_gold_getter(song_id: str) -> Optional[str]:
    from .evals import get_eval_store

    return get_eval_store().get_gold(song_id)


def reanalyze_and_supersede(
    store: SongRepository,
    old_id: str,
    *,
    confirm: bool = False,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    resolve: Callable[[str], SongIdentity] = resolve_video_identity,
    run_pipeline: Optional[Callable[..., object]] = None,
    gold_getter: Callable[[str], Optional[str]] = _default_gold_getter,
) -> ReanalyzeReport:
    """Re-analyze `old_id` under whatever id identity.py resolves today, then
    supersede the old document. DRY RUN BY DEFAULT: without ``confirm=True``
    this previews the resolved identity and refuses/writes nothing at all —
    not even the preview's own metadata fetch counts as a "write", but no
    pipeline runs and no store call is ever made.

    Refuses (populates `refused`, does nothing else) when:
    - `old_id` isn't a stored song, or has no `audio.youtubeVideoId`,
    - `old_id` is already superseded,
    - identity.py still can't resolve it today either,
    - the resolved id is the same as `old_id` (nothing to do),
    - `old_id` carries a gold-eval pointer — moving that to the new id is a
      decision for a human, so this stops rather than moving it, in BOTH
      dry-run and confirmed mode.

    `resolve`, `run_pipeline`, and `gold_getter` are injectable so tests
    never need network, an LLM, or a real eval store.
    """
    report = ReanalyzeReport(old_id=old_id, dry_run=not confirm)

    try:
        old_song = store.get(old_id)
    except StoreError as e:
        report.refused = f"cannot read {old_id!r}: {e}"
        return report

    if old_song.supersededBy:
        report.refused = (
            f"already superseded by {old_song.supersededBy!r}; nothing to do"
        )
        return report

    video_id = old_song.audio.youtubeVideoId if old_song.audio else None
    if not video_id:
        report.refused = "no audio.youtubeVideoId stored — nothing to re-resolve from"
        return report

    try:
        identity = resolve(video_id)
    except IdentityError as e:
        report.refused = f"identity.py still can't resolve this today either: {e}"
        return report

    report.resolved_id = slugify_song_id(identity.artist, identity.title)
    report.confidence = identity.confidence

    if report.resolved_id == old_id:
        report.refused = "already matches what identity.py resolves today; nothing to do"
        return report

    gold_version = gold_getter(old_id)
    if gold_version:
        report.refused = (
            f"gold-marked (version {gold_version}); re-analyzing under a new id "
            "would need the gold pointer moved to that new id, and this will not "
            "do that automatically. Move (or clear) the gold pointer yourself, "
            "then re-run with --confirm."
        )
        return report

    if not confirm:
        return report

    if run_pipeline is None:
        from ..pipeline import run_pipeline as _run_pipeline

        run_pipeline = _run_pipeline

    pipeline_report = run_pipeline(
        title=None,
        artist=None,
        youtube_url_or_id=video_id,
        provider=provider,
        model=model,
        store=store,
    )
    report.new_id = pipeline_report.song_id
    report.new_song_version = pipeline_report.stored_version
    report.steps = dict(pipeline_report.steps)

    if report.new_id == old_id:
        report.refused = "the analyze run resolved the same id again — nothing to supersede"
        return report

    save = supersede_song(store, old_id, report.new_id)
    report.superseded_version = save.version
    return report
