"""MCP tool surface for the Snoocle server.

Design notes (patterns reused per the brief):
- Tools mirror the pipeline steps 1:1 (discover_song / acquire_audio /
  analyze_audio / reconcile_song / get_song_version ...), NOT one monolithic
  tool — same shape as Dr-Lurie-Blog/CMS-Agent's step-scoped tools
  (trigger_netlify_build, save_json_blob_publish_by_time).
- Audio tools accept either a server-side path OR base64 content
  (`input_base64`) and can return base64 — the CMS-Agent `save_artifact`
  fallback for agent environments that can't move raw binary.
- Local-first routing (pdf-tool): the deterministic audio tools never touch
  an LLM; reconcile_song is the only AI-invoking tool.
- save_song exposes expected_version optimistic locking —
  saveRecordIfVersionUnchanged, as in CMS-Agent.

Run: `snoocle-mcp` — stdio transport by default (for a local MCP client /
agent runtime to spawn as a subprocess). Set SNOOCLE_MCP_TRANSPORT=
streamable-http to instead serve MCP over HTTP on $PORT/SNOOCLE_MCP_PORT
(e.g. as a second Cloud Run service, gated by Cloud Run IAM auth — see
docs/DEPLOY_CLOUD_RUN.md). SSE is also available for older clients.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import __version__
from .audio import utils as audio_utils
from .audio.acquire import acquire as _acquire
from .config import settings
from .discovery import CandidateSource, discover_sources
from .identity import (
    IdentityUnresolvedError,
    require_resolved_song_id,
    resolve_identity_from_evidence,
    song_id_has_unknown_segment,
)
from .mir import MirAnalysis, analyze_audio as _analyze_audio
from .pipeline import PipelineStepError, get_store, run_pipeline_async
from .reconcile import provider_capabilities
from .reconcile.admission import reconcile_admitted
from .schema import Song, song_json_schema
from .scope import AnalysisScope
from .store import IdentityCollisionError, backend_label as _store_backend_label
from .store.identity_rename import rename_song_identity
from .store.runs import get_run_store

def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _identity_error(error: IdentityUnresolvedError) -> dict:
    return {"detail": str(error), **error.to_dict()}


def _identity_collision_error(error: IdentityCollisionError) -> dict:
    return {"detail": str(error), "errorCode": error.code}


mcp = FastMCP(
    "snoocle",
    instructions=(
        "Snoocle audio-to-song-data foundry (personal-use). Pipeline tools: "
        "discover_song -> acquire_audio -> analyze_audio -> reconcile_song, or "
        "analyze_and_store_song for the full flow with Firestore-backed, "
        "content-versioned persistence. "
        "Deterministic audio utilities (convert/trim/normalize/probe) never invoke AI."
    ),
)


def _materialize_input(
    input_path: Optional[str], input_base64: Optional[str], input_format: str = "bin"
) -> Path:
    """Server-side path wins; base64 is the fallback for clients that can't
    reference server files."""
    if input_path:
        p = Path(input_path)
        if not p.exists():
            raise ValueError(f"no such file: {input_path}")
        return p
    if input_base64:
        p = Path(tempfile.mkdtemp(prefix="snoocle-mcp-")) / f"in.{input_format.lstrip('.')}"
        p.write_bytes(base64.b64decode(input_base64))
        return p
    raise ValueError("provide input_path or input_base64")


def _audio_result(dst: Path, return_base64: bool) -> dict:
    out: dict = {"path": str(dst), "probe": dataclasses.asdict(audio_utils.probe(dst))}
    if return_base64:
        out["base64"] = base64.b64encode(dst.read_bytes()).decode()
    return out


# --- pipeline steps ----------------------------------------------------------


@mcp.tool()
def discover_song(title: str, artist: str, max_candidates: int = 8) -> dict:
    """Find candidate chord/lyric text sources for a song via general web
    search (step 2-3). Returns parsed, sounding-pitch-normalized candidates,
    each with confidence/provenance — kept separate for reconciliation."""
    cands = discover_sources(title, artist, max_candidates=max_candidates)
    return {"count": len(cands), "candidates": [c.model_dump() for c in cands]}


@mcp.tool()
def acquire_audio(
    title: Optional[str] = None,
    artist: Optional[str] = None,
    youtube_url_or_id: Optional[str] = None,
) -> dict:
    """Acquire the song's recording from YouTube server-side (personal-use
    tool). Give a video URL/id, or title+artist to search. Cached by video id."""
    return dataclasses.asdict(_acquire(title=title, artist=artist, video_url_or_id=youtube_url_or_id))


@mcp.tool()
def analyze_audio(
    audio_path: Optional[str] = None,
    input_base64: Optional[str] = None,
    input_format: str = "bin",
    title: Optional[str] = None,
    artist: Optional[str] = None,
    youtube_url_or_id: Optional[str] = None,
    accuracy: str = "standard",
) -> dict:
    """MIR analysis of a recording (step 4): beats/downbeats, chord timeline,
    structural sections, bpm, key — audio-grounded, independent of any text
    source. Provide ONE of: audio_path (a server-side file); input_base64 (the
    bytes of an uploaded audio OR video file — set input_format to its
    extension, e.g. "mp4"/"mov"/"mp3"; for video the audio track is extracted
    automatically); or acquisition params (title/artist/youtube_url_or_id) to
    fetch from YouTube first. Chords are the sounding harmony, never a
    fretboard shape."""
    video_id = None
    if audio_path is None and input_base64 is None:
        acquired = _acquire(title=title, artist=artist, video_url_or_id=youtube_url_or_id)
        audio_path = acquired.path
        video_id = acquired.video_id
    else:
        # A client-supplied path (validated) or uploaded bytes (materialized to
        # a temp file). Video containers decode fine — MIR strips video first.
        audio_path = str(_materialize_input(audio_path, input_base64, input_format))
    analysis = _analyze_audio(audio_path, accuracy=accuracy)
    return {"audioPath": audio_path, "youtubeVideoId": video_id, "analysis": analysis.model_dump()}


@mcp.tool()
def reconcile_song(
    title: str,
    artist: str,
    candidates_json: Optional[str] = None,
    mir_json: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    audio_path: Optional[str] = None,
    attach_audio: Optional[bool] = None,
    youtube_video_id: Optional[str] = None,
    media_url: Optional[str] = None,
    force: bool = False,
    force_reason: Optional[str] = None,
    batch_id: Optional[str] = None,
    effort_level: Optional[str] = None,
    prior_song: Optional[dict] = None,
) -> dict:
    """Reconcile candidate text sources + MIR analysis into a schema-compliant
    Song JSON via the configured reconciler (step 5). candidates_json/mir_json
    accept the outputs of discover_song / analyze_audio; when candidates_json
    is omitted, discovery runs first. Does NOT persist — use save_song or
    analyze_and_store_song for that. provider: anthropic | openai | gemini |
    agent | mock. The "agent" provider delegates to an external agent
    workspace's MCP server (SNOOCLE_AGENT_MCP_URL), sending title/artist,
    media_url (YouTube watch URL or other media URL; derived from
    youtube_video_id when omitted), and the timestamped MIR chord timeline."""
    if candidates_json:
        candidates = [CandidateSource.model_validate(c) for c in json.loads(candidates_json)]
    else:
        # With no already-gathered page to recover from, reject an unresolved
        # identity before a search can spend time or create a later run.
        try:
            identity = resolve_identity_from_evidence(artist=artist, title=title)
        except IdentityUnresolvedError as error:
            return _identity_error(error)
        title, artist = identity.title, identity.artist
        candidates = discover_sources(title, artist)
    mir = None
    if mir_json:
        payload = json.loads(mir_json)
        mir = MirAnalysis.model_validate(payload.get("analysis", payload))
    # No `timing_authority`: this tool returns the document to the agent and
    # persists nothing (save_song is a separate call that runs no timing pass
    # either), so no deterministic pass will re-time it and the model's timing
    # must survive intact. See reconcile/engine.py's TimingAuthority.
    try:
        admitted = reconcile_admitted(
            title,
            artist,
            candidates,
            mir,
            provider=provider,
            model=model,
            audio_path=audio_path,
            attach_audio=attach_audio,
            youtube_video_id=youtube_video_id,
            media_url=media_url,
            force=force,
            force_reason=force_reason,
            batch_id=batch_id,
            effort_level=effort_level,
            prior_song=prior_song,
        )
    except IdentityUnresolvedError as error:
        return _identity_error(error)
    except Exception as error:
        from .usage import BudgetExceededError
        if isinstance(error, BudgetExceededError):
            return error.to_dict()
        raise
    result = admitted.result
    return {
        "song": result.song.model_dump(),
        "provider": result.provider,
        "model": result.model,
        "attempts": result.attempts,
        "audioAttached": result.audio_attached,
        "runId": admitted.recorder.trace.run_id,
        "evidenceManifest": admitted.evidence_manifest,
    }


@mcp.tool()
async def analyze_and_store_song(
    title: Optional[str] = None,
    artist: Optional[str] = None,
    youtube_url_or_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    skip_audio: bool = False,
    expected_version: Optional[str] = None,
    accuracy: Optional[str] = None,
    guidance: Optional[str] = None,
    scope: Optional[dict] = None,
    allow_timing_loss: bool = False,
    force: bool = False,
    force_reason: Optional[str] = None,
    batch_id: Optional[str] = None,
    effort_level: Optional[str] = None,
) -> dict:
    """Full pipeline: (resolve) -> discover -> acquire -> MIR -> reconcile ->
    commit a new version to the Firestore-backed store (never overwrites;
    content-hash versions). Give title+artist, a youtube_url_or_id, or both.
    Video metadata is only ever a FALLBACK for whichever of title/artist you
    did not supply — it never overrides one you did give, even when a
    youtube_url_or_id is also present. The song id is permanent (content-hash
    versioned store), so a caller-supplied identity always wins.
    Each external step runs under its own timeout so the call can't hang; a
    fatal step failure raises with the step name. Returns the song, the per-step
    report, and the stored version sha.

    guidance (optional): free-text instructions for THIS run — the
    human-in-the-loop correction path ("change the C to a B in line 12", "the
    bridge is Bm, not D"), handed to the reconciler as high-priority evidence so
    a re-analysis honors the fix instead of rediscovering from scratch. Stored as
    the song's notes before any expensive step runs, so a run that dies at
    acquire or reconcile does not eat it, and applied ONCE: it replays into a
    retry that passes none, then stops replaying as soon as it has landed in a
    stored version. It never disturbs a standing preference already on file
    (set_song_notes) — if one is set, BOTH are in force for this run, combined.
    What the run is SHOWN can be less: a notes-only run (inferred from a
    targeted correction, or asked for explicitly) is handed the correction
    ALONE, because its contract is "change nothing else" and a standing
    rendering preference is no part of that; a run that is not notes-only sees
    both. `steps.notes` says which happened. Max 8000 chars, same ceiling as
    set_song_notes — over it is a rejection here, not a truncation later.
    For a standing instruction that should shape EVERY later analyze
    ("capo-free voicings please") use set_song_notes instead; get_song_notes
    shows what a run passing no guidance would replay.

    scope (optional, same shape as the HTTP API's `scope`):
    {"listen": bool, "reconcile": bool} — which evidence-gathering stages this
    run may do. listen=false reuses the existing audio analysis instead of
    re-downloading and re-analyzing; reconcile=false skips web source
    discovery. Both false means "apply the notes to the prior song and gather
    nothing". OMIT it for the full pipeline (the default) — when guidance
    names a specific chord, line, or section and scope is omitted, notes-only
    scope is INFERRED automatically (an explicit scope always overrides
    inference). A targeted, non-lyric correction in notes-only scope is
    applied as a PATCH against the prior document rather than a regenerated
    Song — see docs/ARCHITECTURE.md, "A targeted correction is a PATCH, not a
    rewrite" — which is what makes a one-chord fix immune to both the content
    filter and timing loss.

    listen=false carries the prior version's timing (beat grid, bpm, chord and
    line times, section times) forward onto the new document, and fails if
    there is no prior version to carry it from. allow_timing_loss=true opts
    out of that and of the guard that refuses to store a version which drops
    audio-derived data the prior one had — only pass it when losing that data
    is the actual intent."""
    from .store.song_notes import length_error as notes_length_error

    # Same per-slot ceiling and same message as set_song_notes: `guidance` is a
    # CORRECTION write, and both slots of one contract must refuse an over-long
    # body the same way. Before the pipeline, so nothing is paid for a run that
    # is going to be refused.
    guidance_problem = notes_length_error(guidance)
    if guidance_problem:
        raise ValueError(guidance_problem)
    try:
        report = await run_pipeline_async(
            title,
            artist,
            youtube_url_or_id=youtube_url_or_id,
            provider=provider,
            model=model,
            skip_audio=skip_audio,
            expected_version=expected_version,
            accuracy=accuracy,
            guidance=guidance,
            # None in -> None out: an omitted scope must stay omitted all the way
            # down, or every MCP caller silently starts getting a "scope" step.
            scope=AnalysisScope.parse(scope),
            allow_timing_loss=allow_timing_loss,
            force=force,
            force_reason=force_reason,
            batch_id=batch_id,
            effort_level=effort_level,
        )
    except PipelineStepError as error:
        if error.error_code == "identity_unresolved" and isinstance(error.__cause__, IdentityUnresolvedError):
            return _identity_error(error.__cause__)
        if error.error_code == "identity_collision" and isinstance(error.__cause__, IdentityCollisionError):
            return _identity_collision_error(error.__cause__)
        if error.error_code == "budget_exceeded":
            from .usage import BudgetExceededError
            cause = error.__cause__
            return (
                cause.to_dict()
                if isinstance(cause, BudgetExceededError)
                else {"code": "budget_exceeded", "error": str(error)}
            )
        raise
    assert report.reconcile is not None
    return {
        "songId": report.song_id,
        "steps": report.steps,
        "storedVersion": report.stored_version,
        "runId": report.run_id,
        "song": report.reconcile.song.model_dump(),
    }


@mcp.tool()
async def realign_song_to_recording(
    song_id: str,
    video_id: str,
    source_version: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    analysis_depth: Optional[str] = None,
    expected_version: Optional[str] = None,
    allow_same_recording: bool = False,
    allow_timing_loss: bool = False,
) -> dict:
    """Mode B — "analyze <video_id> as the timing reference for <song_id>".

    Re-times an EXISTING song document to a DIFFERENT recording of the same
    song: full MIR analysis of that video, then the document's own lyrics and
    chord sequence re-snapped onto the new timeline. Lyrics and chord identities
    carry over untouched; the transposition between the document's key and the
    recording's is derived from the chord sequence and applied.

    Stored as a new version of the SAME song id — this is the same song, timed
    against different audio, and it records which video supplied the timing.

    Mostly deterministic and normally free of model tokens: a model is consulted
    ONLY when a structural comparison finds a difference the stored document
    cannot explain (an extra chorus, a truncated outro). Use it when a version's
    timing came out unreliable on a poor recording (see get_song's provenance for
    a "timing-unreliable" entry, and `suggest_better_recordings` for candidates).

    Refuses (and says so) when the given video turns out to be the SAME recording
    the song is already timed against: that case wants one cross-correlation via
    the video-offset endpoint, not a full analysis. allow_same_recording=true
    overrides."""
    from .realign import realign_song_async

    report = await realign_song_async(
        song_id,
        video_id,
        provider=provider,
        model=model,
        analysis_depth=analysis_depth,
        expected_version=expected_version,
        source_version=source_version,
        allow_same_recording=allow_same_recording,
        allow_timing_loss=allow_timing_loss,
    )
    return report.to_dict()


@mcp.tool()
def suggest_better_recordings(song_id: str, limit: int = 5) -> dict:
    """Ranked alternative recordings of a song. REPORTS ONLY — never analyzes.

    For a song whose timing came out unreliable because its recording is poor
    (a lo-fi live video, a broadcast rip): a better recording is the fix, and
    this finds candidates — official/studio audio preferred, covers and lessons
    ranked down, the recording the song is already timed against excluded.

    Nothing is downloaded or analyzed. Each suggestion carries the exact action
    that would spend, which is `realign_song_to_recording(song_id, video_id)`:
    that is a full analysis of a second track, so it stays an explicit choice."""
    from .recordings import suggest_recordings

    song = get_store().get(song_id)
    return suggest_recordings(
        song, reason="requested for this song", limit=max(1, min(limit, 10))
    ).to_dict()


# --- versioned store ---------------------------------------------------------


@mcp.tool()
def list_songs() -> dict:
    """List song ids present in the store."""
    return {"songs": get_store().list_songs()}


@mcp.tool()
def get_song(song_id: str, version: Optional[str] = None) -> dict:
    """Fetch a song's JSON from the store — latest, or a specific stored
    version sha."""
    return get_store().get(song_id, version=version).model_dump()


@mcp.tool()
def list_song_versions(song_id: str) -> dict:
    """List the stored versions (sha, timestamp, message) of a song,
    newest first."""
    return {
        "songId": song_id,
        "versions": [dataclasses.asdict(v) for v in get_store().versions(song_id)],
    }


@mcp.tool()
def diff_song_versions(song_id: str, version_a: str, version_b: str) -> str:
    """Unified diff (pretty-printed JSON) of a song between two stored versions."""
    return get_store().diff(song_id, version_a, version_b)


@mcp.tool()
def save_song(song_json: str, message: str = "Manual save", expected_version: Optional[str] = None) -> dict:
    """Validate and commit a Song JSON as a new version. expected_version
    enables optimistic locking (save-if-version-unchanged): the save is
    rejected if the stored version moved since you read it. Provenance is
    append-only — the new document must extend the stored history."""
    song = Song.model_validate_json(song_json)
    try:
        require_resolved_song_id(song.id)
    except IdentityUnresolvedError as error:
        return _identity_error(error)
    try:
        saved = get_store().save(song, message, expected_version=expected_version)
    except IdentityCollisionError as error:
        return _identity_collision_error(error)
    return dataclasses.asdict(saved)


@mcp.tool()
def set_song_identity(song_id: str, artist: str, title: str) -> dict:
    """Manually name a legacy needs-identity song.

    Moves the full version history and every run trace to the new permanent id
    atomically. The target must not already exist; this tool never guesses or
    merges two histories.
    """
    try:
        return rename_song_identity(
            get_store(), get_run_store(), song_id, artist=artist, title=title
        ).to_dict()
    except IdentityUnresolvedError as error:
        return _identity_error(error)


@mcp.tool()
def list_songs_needing_identity() -> dict:
    """List legacy songs with an unresolved id for manual naming."""
    songs = [
        {
            "songId": summary.id,
            "artist": summary.artist,
            "title": summary.title,
            "needsIdentity": True,
        }
        for summary in get_store().list_song_summaries()
        if song_id_has_unknown_segment(summary.id)
    ]
    return {"songs": songs}


# --- per-song reconciliation notes -------------------------------------------
# Free-text "how to build this song" instructions, stored beside the song (never
# in it — the Song document is the app's content contract) and handed to a later
# analyze as reconciler guidance. Mirrors the REST trio (GET/PUT/DELETE
# /v1/songs/{songId}/notes) so the note an analyze would replay is visible,
# settable and clearable from here too: without these, an MCP caller inherits
# whatever an HTTP client happened to leave behind and cannot even see it.
#
# A song's notes have TWO INDEPENDENT lifetimes (store/song_notes.py): a
# durable `preference` (this surface) and a single-shot `correction` (an
# analyze request's `guidance`). Both may be set at once, and an analyze that
# replays or receives guidance while both exist acts on their COMBINATION
# (`song_notes.combine_guidance`). The tools below report them as two
# separate objects rather than one `notes` string, because a flat field could
# not say which lifetime it was reading, or that both were in force —
# exactly the ambiguity that made one slot unable to hold two lifetimes. The
# flat `notes`/`updatedAt` keys are still present, as a deprecated
# compatibility view of `preference` mirroring the REST response.


def _component(doc: dict | None) -> dict | None:
    """One lifetime's sub-record for the wire, or None when it isn't set."""
    if not doc:
        return None
    return {"notes": doc.get("notes", "") or "", "updatedAt": doc.get("updated_at")}


def _notes_doc(song_id: str, record: dict | None) -> dict:
    preference = _component((record or {}).get("preference"))
    correction_raw = (record or {}).get("correction")
    correction = _component(correction_raw)
    if correction is not None:
        correction["appliedToVersion"] = (correction_raw or {}).get("applied_to_version")
    return {
        "songId": song_id,
        # DEPRECATED flat compatibility view of `preference`, mirroring the REST
        # response key-for-key (api.py's `_notes_response`) — one store, two
        # surfaces, and a shape that differs between them is a shape neither can
        # be described by. Read `preference`/`correction` instead.
        "notes": (preference or {}).get("notes", ""),
        "updatedAt": (preference or {}).get("updatedAt"),
        "preference": preference,
        "correction": correction,
    }


@mcp.tool()
def get_song_notes(song_id: str) -> dict:
    """Read a song's stored reconciliation notes — the guidance an analyze that
    passes none would replay. Never an error: "no notes" is a normal answer
    (preference: null, correction: null).

    `preference` is a standing instruction and replays on every later analyze
    that passes none. `correction` came from one
    analyze_and_store_song(guidance=...) call and replays only until its
    `appliedToVersion` names the stored version it landed in — after that it
    is already in the document and is NOT applied again. Both may be set at
    once, in which case both are IN FORCE and a run acts on their combination
    (a labeled "Standing preference: ... / Requested correction: ..." string),
    never one silently standing in for the other — except on a notes-only run,
    which is shown the correction alone (see analyze_and_store_song)."""
    from .store.song_notes import get_song_notes_store

    return _notes_doc(song_id, get_song_notes_store().get_record(song_id))


@mcp.tool()
def set_song_notes(song_id: str, notes: str) -> dict:
    """Store a song's reconciliation notes as a durable PREFERENCE — a standing
    instruction ("capo-free voicings please", "the bridge is Bm, not D") replayed
    as guidance by every later analyze of this song id that passes none, until
    it is changed or cleared. Replaces whatever PREFERENCE was there
    (empty/whitespace clears it, same as clear_song_notes) but never touches a
    pending CORRECTION left behind by an analyze — the two lifetimes are
    stored independently, so writing one never disturbs the other; when both
    are set, a later analyze's guidance is the two combined. Max 8000 chars.

    For a one-off fix to the document you already have ("change the C to a B in
    line 12"), pass it as analyze_and_store_song(guidance=...) instead: that
    applies to that run, survives a run that dies before storing, and then stops
    replaying — a correction that replays forever silently re-applies itself to
    later runs nobody attached it to."""
    from .store.song_notes import get_song_notes_store, length_error

    problem = length_error(notes)
    if problem:
        raise ValueError(problem)
    store = get_song_notes_store()
    store.set_preference(song_id, notes)  # empty/whitespace deletes
    return _notes_doc(song_id, store.get_record(song_id))


@mcp.tool()
def clear_song_notes(song_id: str) -> dict:
    """Delete a song's stored reconciliation notes — BOTH the standing
    preference and any pending correction, so the next analyze that sends no
    guidance replays nothing. There is no partial clear here: a stray pending
    correction is rare (it only outlives one run when that run died before
    storing, and the next successful analyze consumes it on its own), so if
    you only want to keep a standing preference intact, simply don't call
    this. `deleted` is false when there was nothing stored in either slot."""
    from .store.song_notes import get_song_notes_store

    return {"songId": song_id, "deleted": get_song_notes_store().delete(song_id)}


# --- deterministic audio utilities (no AI) -----------------------------------


@mcp.tool()
def convert_audio(
    output_format: str,
    input_path: Optional[str] = None,
    input_base64: Optional[str] = None,
    input_format: str = "bin",
    return_base64: bool = False,
) -> dict:
    """Convert an audio file between formats (mp3/wav/m4a/flac/ogg/opus) with
    ffmpeg — deterministic, no AI. Provide input_path (server-side) or
    input_base64 (+input_format) for clients that can't reference files."""
    src = _materialize_input(input_path, input_base64, input_format)
    dst = src.parent / f"{src.stem}.converted.{output_format.lstrip('.')}"
    audio_utils.convert(src, dst)
    return _audio_result(dst, return_base64)


@mcp.tool()
def trim_audio(
    start_seconds: float,
    end_seconds: float,
    input_path: Optional[str] = None,
    input_base64: Optional[str] = None,
    input_format: str = "bin",
    output_format: Optional[str] = None,
    return_base64: bool = False,
) -> dict:
    """Crop an audio file to [start_seconds, end_seconds) with ffmpeg —
    deterministic, exact cut points, no AI."""
    src = _materialize_input(input_path, input_base64, input_format)
    fmt = (output_format or src.suffix.lstrip(".") or "wav").lstrip(".")
    dst = src.parent / f"{src.stem}.trimmed.{fmt}"
    audio_utils.trim(src, dst, start_seconds, end_seconds)
    return _audio_result(dst, return_base64)


@mcp.tool()
def normalize_audio(
    input_path: Optional[str] = None,
    input_base64: Optional[str] = None,
    input_format: str = "bin",
    target_lufs: float = -16.0,
    output_format: Optional[str] = None,
    return_base64: bool = False,
) -> dict:
    """EBU R128 loudness-normalize an audio file with ffmpeg — no AI."""
    src = _materialize_input(input_path, input_base64, input_format)
    fmt = (output_format or src.suffix.lstrip(".") or "wav").lstrip(".")
    dst = src.parent / f"{src.stem}.normalized.{fmt}"
    audio_utils.normalize(src, dst, target_lufs=target_lufs)
    return _audio_result(dst, return_base64)


@mcp.tool()
def probe_audio(
    input_path: Optional[str] = None,
    input_base64: Optional[str] = None,
    input_format: str = "bin",
) -> dict:
    """Inspect an audio file (duration, codec, sample rate, channels) with
    ffprobe — no AI."""
    src = _materialize_input(input_path, input_base64, input_format)
    return dataclasses.asdict(audio_utils.probe(src))


# --- meta --------------------------------------------------------------------


@mcp.tool()
def server_status() -> dict:
    """Service version, configured LLM providers (and audio-input capability),
    active MIR engines, and store location."""
    import shutil

    def has(mod: str) -> bool:
        try:
            __import__(mod)
            return True
        except Exception:  # noqa: BLE001
            return False

    from .mir.chordrec import chord_engine_id, chord_model_status

    return {
        "version": __version__,
        "ffmpeg": shutil.which(settings.ffmpeg_bin) is not None,
        "mirEngines": {
            "beats": "madmom" if has("madmom") else "librosa-fallback",
            "chords": chord_engine_id(),
            "structure": "songformer" if settings.songformer_dir else "librosa-agglomerative-fallback",
        },
        "chordModel": chord_model_status(),
        "llmProviders": provider_capabilities(),
        "store": _store_backend_label(),
    }


@mcp.tool()
def get_song_schema() -> dict:
    """The Song JSON schema every produced document conforms to (iOS SongStore
    compatible), including the sounding-harmony chord rule."""
    return song_json_schema()


# --- agent programming & observability ---------------------------------------
# Program the in-process reconciliation agent and inspect its work. Config
# writes are gated by the SAME SNOOCLE_API_TOKEN as everything else (see
# authz.require_admin_token_configured) — a write on an unauthenticated service
# is refused. The Song OUTPUT CONTRACT (strict schema, sounding-pitch chords,
# capo=0) is NOT programmable: it is always enforced by the reconcile engine.


@mcp.tool()
def get_agent_config() -> dict:
    """Read the runtime-editable agent config (instructions, tool budgets,
    effort, model) plus the built-in defaults it overrides. Empty config means
    the agent runs with its built-in behavior."""
    from .reconcile.agent_config import AgentConfig, config_version
    from .store.agent_config import get_agent_config_store

    doc = get_agent_config_store().get()
    cfg = AgentConfig.model_validate(doc) if doc else AgentConfig()
    return {"config": cfg.model_dump(), "configVersion": config_version(cfg),
            "isDefault": cfg.is_default()}


@mcp.tool()
def set_agent_config(config_json: str) -> dict:
    """Program the agent: set instructions_extra / theory_rules /
    retrieval_recipe / instructions_override (all optional), max_turns, effort,
    max_fetch (deterministic sheet-search calls), max_windows, disabled_tools,
    source_site_preferences, and model. The output
    contract and schema enforcement are NOT editable. Requires SNOOCLE_API_TOKEN
    to be configured on the server."""
    import json as _json

    from .authz import require_admin_token_configured
    from .reconcile.agent_config import AgentConfig, config_version
    from .store.agent_config import get_agent_config_store

    require_admin_token_configured()
    cfg = AgentConfig.model_validate(_json.loads(config_json))
    doc = cfg.model_dump()
    doc["updated_at"] = _now_iso()
    doc["source"] = "mcp"
    get_agent_config_store().set(doc)
    return {"status": "stored", "configVersion": config_version(cfg), "updatedAt": doc["updated_at"]}


@mcp.tool()
def reset_agent_config() -> dict:
    """Clear the agent config back to built-in defaults. Requires
    SNOOCLE_API_TOKEN."""
    from .authz import require_admin_token_configured
    from .store.agent_config import get_agent_config_store

    require_admin_token_configured()
    get_agent_config_store().clear()
    return {"status": "reset"}


@mcp.tool()
def list_song_runs(song_id: str, limit: int = 20) -> dict:
    """Recent reconciliation runs for a song, newest first (summaries only —
    fetch a run's steps/MIR with get_run)."""
    from .store.runs import get_run_store

    return {"songId": song_id, "runs": get_run_store().list_runs(song_id, limit=limit)}


@mcp.tool()
def get_run(run_id: str) -> dict:
    """One reconciliation run's full step trace + MIR — the agent's reasoning,
    tool calls, and repair rounds."""
    from .store.runs import fetch_run

    run = fetch_run(run_id)
    if run is None:
        return {"error": f"no such run: {run_id}"}
    return run


@mcp.tool()
def get_usage_summary(window: str = "7d") -> dict:
    """Compact per-day, per-song, and per-model token/cost rollups."""
    from .reconcile.engine import _load_agent_config
    from .store.runs import get_run_store
    from .usage_summary import build_usage_summary

    try:
        return build_usage_summary(
            get_run_store(), window=window, cfg=_load_agent_config()
        )
    except ValueError as e:
        return {"code": "invalid_window", "error": str(e)}


@mcp.tool()
def get_scorecard() -> dict:
    """Score every gold-marked song's current version against its gold — the
    agent's report card (content metrics + aggregate)."""
    from .eval.scorecard import build_scorecard

    return build_scorecard(get_store())


@mcp.tool()
def set_gold_version(song_id: str, version: str) -> dict:
    """Mark one of a song's stored versions as the ground-truth 'gold' the agent
    is scored against."""
    from .store.evals import get_eval_store

    try:
        get_store().get(song_id, version=version)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    get_eval_store().set_gold(song_id, version)
    return {"songId": song_id, "goldVersion": version}


@mcp.tool()
def score_song_version(song_id: str, candidate_version: str | None = None) -> dict:
    """Score a candidate version (default: current) against the song's gold."""
    from .eval import score_song
    from .store.evals import get_eval_store

    gold_version = get_eval_store().get_gold(song_id)
    if not gold_version:
        return {"error": f"no gold version set for {song_id}"}
    try:
        gold = get_store().get(song_id, version=gold_version)
        cand = get_store().get(song_id, version=candidate_version)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return {"songId": song_id, "goldVersion": gold_version,
            "candidateVersion": candidate_version or get_store().current_version(song_id),
            "metrics": score_song(cand, gold)}


_LOCALHOST_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_LOCALHOST_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def resolve_http_transport(env: dict):
    """Pure resolver for the HTTP transport's bind host + security settings.

    Extracted from main() so the security posture is unit-testable without
    spawning a server. Returns (host: str, port: int, security_settings).

    Bind loopback-only unless a remote-serving mode is explicitly configured.
    The DNS-rebinding Host check alone can't protect a 0.0.0.0 bind — the Host
    header is client-controlled, so a LAN client can send `Host: localhost:
    <port>` to satisfy the localhost allowlist. Binding 127.0.0.1 for a local
    smoke test keeps the port off the LAN entirely; remote serving (Cloud Run
    sets SNOOCLE_MCP_TRUST_PROXY=true and needs 0.0.0.0 for routed traffic)
    opts into the wider bind.

    A non-loopback SNOOCLE_MCP_HOST is a remote-serving intent, so it REQUIRES
    a security mode too (ALLOWED_HOSTS or TRUST_PROXY). Without one it would
    widen the bind while leaving the localhost-only fallback policy in place —
    rejecting real remote clients AND letting any reachable client spoof
    `Host: localhost:<port>` — so it's rejected with a clear error rather than
    silently creating that insecure state.

    Security settings are constructed explicitly in every branch rather than
    mutating FastMCP's default: on mcp 1.10.x that default is None (mutating
    would AttributeError) and its middleware treats None as protection-OFF
    "for backwards compatibility", so relying on it would silently leave a
    local run open. Explicit construction is safe-by-default on every version.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    _LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}

    allowed = [h.strip() for h in env.get("SNOOCLE_MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    trust_proxy = _truthy(env.get("SNOOCLE_MCP_TRUST_PROXY"))
    remote_mode = bool(allowed) or trust_proxy

    explicit_host = env.get("SNOOCLE_MCP_HOST")
    if explicit_host and explicit_host not in _LOOPBACK and not remote_mode:
        raise ValueError(
            f"SNOOCLE_MCP_HOST={explicit_host!r} exposes the MCP server beyond "
            "loopback but no host-security mode is set. Also set "
            "SNOOCLE_MCP_ALLOWED_HOSTS=<host[,host...]> (keeps the DNS-rebinding "
            "check on) or SNOOCLE_MCP_TRUST_PROXY=true (only behind an "
            "authenticating proxy such as Cloud Run IAM)."
        )

    host = explicit_host or ("0.0.0.0" if remote_mode else "127.0.0.1")
    port = int(env.get("PORT", env.get("SNOOCLE_MCP_PORT", "8080")))

    if allowed:
        # Protection ON, scoped STRICTLY to the operator's hosts. This branch
        # binds 0.0.0.0 (remote-reachable), so localhost must NOT be appended:
        # allowing `localhost:*` here would let a LAN client spoof
        # `Host: localhost:<port>` to bypass the allowlist the operator set to
        # narrow access. If a local Host value is genuinely needed, the
        # operator adds it to SNOOCLE_MCP_ALLOWED_HOSTS explicitly. (Cloud
        # Run's startup probe is a TCP socket check — it sends no HTTP Host
        # header, so nothing here depends on the localhost entries.)
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(allowed),
            allowed_origins=[
                *(f"https://{h}" for h in allowed),
                *(f"http://{h}" for h in allowed),
            ],
        )
    elif trust_proxy:
        # Explicit opt-out: only safe behind an authenticating proxy (Cloud Run
        # IAM). The deployed *.run.app hostname is assigned at deploy time so it
        # can't be hardcoded into an allowlist; this is the escape hatch.
        security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    else:
        # Neither configured: loopback bind (above) + protection ON,
        # localhost-only. Defense in depth for a local smoke test.
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=_LOCALHOST_HOSTS,
            allowed_origins=_LOCALHOST_ORIGINS,
        )
    return host, port, security


def main() -> None:
    """Entrypoint for the `snoocle-mcp` console script.

    Defaults to stdio — the standard way an MCP client (Claude Desktop, an
    agent runtime) spawns this as a local subprocess. Set
    SNOOCLE_MCP_TRANSPORT=streamable-http to instead serve as a long-running
    HTTP process (e.g. deployed to Cloud Run as its own service, behind
    Cloud Run IAM auth rather than any app-level auth).

    In HTTP mode the port binds to loopback (127.0.0.1) with the SDK's Host
    (DNS-rebinding) check ON, so a local run is not exposed on the LAN.
    Opt into remote serving with one of:
      * SNOOCLE_MCP_ALLOWED_HOSTS  comma-separated Host values to allow
        (e.g. "snoocle-mcp-xxxx.run.app"); binds 0.0.0.0, check stays on,
        scoped to those hosts. Preferred once the deployed hostname is known.
      * SNOOCLE_MCP_TRUST_PROXY=true  binds 0.0.0.0 and disables the Host
        check — ONLY correct behind an authenticating proxy (Cloud Run IAM).
    """
    import os

    transport = os.environ.get("SNOOCLE_MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run()
        return
    if transport not in ("streamable-http", "sse"):
        raise ValueError(
            f"unsupported SNOOCLE_MCP_TRANSPORT {transport!r} "
            "(expected stdio | streamable-http | sse)"
        )
    host, port, security = resolve_http_transport(dict(os.environ))
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.settings.transport_security = security

    # Stateless HTTP by default (no persistent server->client SSE stream).
    # Required for a Cloud Run --concurrency=1 deployment: otherwise the MCP
    # client's long-lived GET SSE stream (opened after initialize) occupies the
    # single request slot and every subsequent tool-call POST queues behind it
    # until it times out — a deadlock. This tool server issues no
    # server-initiated notifications, so statelessness costs nothing here.
    # Opt out (restore the stateful session + SSE stream) with
    # SNOOCLE_MCP_STATELESS=false — then the service needs concurrency >= 2.
    if transport == "streamable-http" and _truthy(os.environ.get("SNOOCLE_MCP_STATELESS", "true")):
        mcp.settings.stateless_http = True
        mcp.settings.json_response = True
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
