"""HTTP API — clean, typed, stateless; each pipeline step is its own endpoint.

The surface deliberately mirrors the MCP tool surface (mcp_server.py) so the
iOS app, curl, and agent callers all drive the same service layer. State
lives only in the git-backed store and the audio cache.
"""

from __future__ import annotations

import dataclasses
import mimetypes
import os
import secrets
import tempfile
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from . import __version__
from .audio import stems
from .audio import utils as audio_utils
from .batch import MAX_ITEMS_PER_SUBMIT, parse_batch_line
from .audio.acquire import AcquisitionError, YouTubeAuthError, acquire
from .config import settings
from .discovery import CandidateSource, discover_sources
from .discovery.search import SearchError
from .mcp_server import mcp as _mcp
from .mcp_server import resolve_http_transport as _resolve_mcp_security
from .mir import MirAnalysis, analyze_audio
from .mir.chordrec import chord_engine_id, chord_model_status
from .pipeline import PipelineStepError, get_store, run_pipeline_async
from .reconcile import (
    ReconcileResult,
    provider_capabilities,
    provider_preflight,
    reconcile,
)
from .reconcile.engine import ReconcileError
from .reconcile.providers import ProviderError
from .scope import AnalysisScope
from .export import to_chordpro, to_txt
from .schema import ProvenanceEntry, Song, song_json_schema
from .timing.offset import estimate_offset
from .store.jobs import (
    DEFAULT_LEASE_SECONDS,
    UnknownJobError,
    WrongWorkerError,
    get_job_store,
)
from .oauth import authenticate_bearer, resource_metadata_url
from .oauth import router as oauth_router
from .oauth.protocol import www_authenticate as oauth_www_authenticate
from .store import (
    CorruptSongError,
    StoreError,
    StoreUnavailableError,
    VersionConflictError,
    backend_label,
    count_cookie_lines,
)

# --- Single-service topology: embed the MCP endpoint in this FastAPI app -----
# One Cloud Run service / container / process serves BOTH the REST API and the
# MCP streamable-HTTP transport (at /mcp), so it is the SOLE writer to the git
# store. That fully serializes writes (no cross-service race) and removes the
# cross-mount read-staleness that a two-service split had. The MCP session
# manager is created here and its lifespan is run by this app's lifespan below
# (Starlette does not run a mounted sub-app's lifespan on its own).
_mcp.settings.stateless_http = True  # no persistent SSE stream (see mcp_server docs)
_mcp.settings.json_response = True
# The /mcp route's DNS-rebinding host check is driven by the same env vars as
# the standalone server (SNOOCLE_MCP_TRUST_PROXY / SNOOCLE_MCP_ALLOWED_HOSTS);
# only the security settings are used here — host/port binding is uvicorn's job
# for the combined app. Defaults to protection-on/localhost.
try:
    _, _, _mcp.settings.transport_security = _resolve_mcp_security(dict(os.environ))
except ValueError:
    # A non-loopback SNOOCLE_MCP_HOST without a security mode is only a
    # standalone-server misconfig; it doesn't bind the combined app (uvicorn
    # does). Fall back to protection-on/localhost rather than failing import.
    from mcp.server.transport_security import TransportSecuritySettings

    _mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"],
    )
_mcp_asgi_app = _mcp.streamable_http_app()  # creates the session manager


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Run the MCP StreamableHTTP session manager for the app's lifetime.
    #
    # There is deliberately NO background worker here. Analysis jobs are held
    # in the store and executed by an external worker that claims them (see
    # store/jobs.py and docs/WORKER.md), which is what lets this service stay
    # on request-based billing at --min-instances=0: nothing here ever needs a
    # CPU outside a request.
    async with _mcp.session_manager.run():
        yield


app = FastAPI(
    title="Snoocle server",
    version=__version__,
    description="Audio-to-song-data foundry: web-sourced chord/lyric text + MIR analysis, "
    "reconciled by a configurable LLM into Snoocle Song JSON. MCP tools at /mcp. "
    "Personal-use tool.",
    lifespan=_lifespan,
)


# Paths that never require a credential: liveness probes, the static GUI shell
# (it carries no secrets — every /v1 call it makes is still gated), and the
# OAuth endpoints themselves, which are by definition reached without a token.
_ALWAYS_OPEN_PREFIXES = ("/ui", "/oauth/", "/.well-known/")
_ALWAYS_OPEN_EXACT = ("/healthz", "/health", "/")


def _is_open_path(path: str) -> bool:
    return path in _ALWAYS_OPEN_EXACT or path.startswith(_ALWAYS_OPEN_PREFIXES)


class _BearerTokenMiddleware:
    """Authentication for both surfaces, with different credentials.

    The REST API takes the static `SNOOCLE_API_TOKEN` — that is what the iOS
    app and the admin UI have, and nothing about it needed to change.

    The MCP transport at /mcp additionally accepts an **OAuth access token**,
    because Claude's remote-MCP connector will not use a static token: it
    discovers an authorization server, registers itself, and runs an
    authorization-code flow. A 401 from /mcp therefore carries the
    `WWW-Authenticate: Bearer resource_metadata="..."` header that starts that
    discovery — Claude ignores the header on any other status, so the 401 is
    load-bearing, not decorative.

    OAuth tokens are accepted ONLY on /mcp. They are minted with /mcp as their
    audience, and honouring them on the REST API would be exactly the
    audience-confusion the MCP spec forbids.

    Active only when SNOOCLE_API_TOKEN is set; otherwise a pass-through, so the
    default posture (Cloud Run IAM gates access) is unchanged. The token is read
    per request so it can be rotated without re-importing the app.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        token = settings.api_token
        if not token or _is_open_path(path):
            await self.app(scope, receive, send)
            return

        auth = Headers(scope=scope).get("authorization", "")
        presented = auth[7:] if auth.startswith("Bearer ") else ""
        is_mcp = path.rstrip("/") == "/mcp"

        if presented and secrets.compare_digest(presented, token):
            await self.app(scope, receive, send)
            return

        if is_mcp and presented:
            from starlette.requests import Request as _Request

            if authenticate_bearer(presented, _Request(scope)) is not None:
                await self.app(scope, receive, send)
                return

        if is_mcp:
            # The handshake: point the client at the metadata that tells it
            # where to authorize.
            from starlette.requests import Request as _Request

            metadata = resource_metadata_url(_Request(scope))
            response = JSONResponse(
                {"error": "invalid_token", "error_description":
                 "authorization required; see the WWW-Authenticate header"},
                status_code=401,
                headers={"WWW-Authenticate": oauth_www_authenticate(
                    metadata, scope="snoocle:mcp",
                    error="invalid_token" if presented else "",
                )},
            )
            await response(scope, receive, send)
            return

        response = JSONResponse(
            {"detail": "missing or invalid bearer token"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
        await response(scope, receive, send)


# Wrap the entire app (REST routes + the /mcp transport appended below).
app.add_middleware(_BearerTokenMiddleware)

# The OAuth 2.1 authorization server + RFC 9728/8414 discovery documents.
# Registered before the /ui static mount so the /.well-known/* routes win.
app.include_router(oauth_router)


@app.exception_handler(StoreUnavailableError)
async def _store_unavailable_handler(request, exc: StoreUnavailableError) -> JSONResponse:
    # The store backend is down/misconfigured (e.g. the Firestore database
    # doesn't exist). 503, not a bare 500 — and never 404, which would falsely
    # read as "song not found".
    return JSONResponse({"detail": f"store unavailable: {exc}"}, status_code=503)


def _store_error_response(e: StoreError, not_found_status: int = 404) -> HTTPException:
    """Map a StoreError to the right HTTP status.

    A CorruptSongError means the document exists but fails validation against
    the current schema -- distinct from "not found" (the common case these
    call sites otherwise assume), so it gets its own 500 with the validation
    detail rather than being reported as a 404.
    """
    if isinstance(e, CorruptSongError):
        return HTTPException(status_code=500, detail=str(e))
    return HTTPException(status_code=not_found_status, detail=str(e))


def _asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return obj


# --- health / meta ---------------------------------------------------------


# Two paths, same handler. `/healthz` is what the Dockerfile HEALTHCHECK and
# Cloud Run's own probes use, and those reach the container directly on
# 127.0.0.1, so they are unaffected by anything in front of the service.
#
# `/health` exists because EXTERNAL monitors are not so lucky: on the deployed
# service, a request for exactly `/healthz` is answered by a Google-branded 404
# at the edge and never reaches this process (no `x-cloud-trace-context` on the
# response — the tell that Cloud Run's ingress never saw it). `/health`,
# `/healthz/`, `/livez`, `/_ah/health` and every other path all get through, so
# it is a single-path rule somewhere above the service, not an app fault.
#
# Rather than depend on getting that rule changed, uptime checks should point
# at `/health`. Both paths return the identical document.
@app.get("/healthz")
@app.get("/health")
def healthz() -> dict:
    import importlib.metadata
    import shutil

    def has(mod: str) -> bool:
        try:
            __import__(mod)
            return True
        except Exception:  # noqa: BLE001
            return False

    def dist_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except Exception:  # noqa: BLE001
            return None

    return {
        "status": "ok",
        "version": __version__,
        "ffmpeg": shutil.which(settings.ffmpeg_bin) is not None,
        # YouTube acquisition health: since yt-dlp 2025.11.12 full support
        # needs an external JS runtime (deno) + the yt-dlp-ejs challenge-solver
        # scripts; without BOTH, downloads fail with "Requested format is not
        # available" because most formats are withheld.
        "ytdlp": {
            "version": dist_version("yt-dlp"),
            "jsRuntime": shutil.which("deno") is not None,
            "challengeSolver": dist_version("yt-dlp-ejs") is not None,
        },
        "mirEngines": {
            "beats": "madmom" if has("madmom") else "librosa-fallback",
            "chords": chord_engine_id(),
            "structure": "songformer" if settings.songformer_dir else "librosa-agglomerative-fallback",
        },
        # Why the chords engine is (or isn't) the heavy model — a configured
        # dir with a missing runner shows up here instead of lying above.
        "chordModel": chord_model_status(),
        "llmProviders": provider_capabilities(),
        # The provider a bare /v1/songs/analyze (no explicit "provider") will
        # use, and whether it can actually serve a request. `ready=false` means
        # every analyze call is doomed at the reconcile step — the usual cause
        # of a "download + MIR then instant 502" loop (fix the server config,
        # not the client).
        "activeProvider": {
            "name": settings.llm_provider.lower(),
            "ready": provider_preflight(settings.llm_provider) is None,
            "problem": provider_preflight(settings.llm_provider),
        },
        "store": backend_label(),  # "firestore" | "memory"
        "mcpEndpoint": _mcp.settings.streamable_http_path,  # embedded MCP transport
    }


@app.get("/v1/schema/song")
def get_song_schema() -> dict:
    return song_json_schema()


@app.get("/v1/providers")
def get_providers() -> dict:
    return provider_capabilities()


# --- step 2-3: text-source discovery ---------------------------------------


class DiscoverRequest(BaseModel):
    title: str
    artist: str
    maxCandidates: Optional[int] = Field(default=None, ge=1, le=20)


@app.post("/v1/discover")
def post_discover(req: DiscoverRequest) -> dict:
    try:
        cands = discover_sources(req.title, req.artist, max_candidates=req.maxCandidates)
    except SearchError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"count": len(cands), "candidates": [c.model_dump() for c in cands]}


# --- step 4: audio acquisition + MIR ----------------------------------------



# Human-readable, action-oriented reasons for machine-readable error codes.
# Clients key UI actions off errorCode (e.g. "youtube_auth_required" -> the
# in-app Reconnect YouTube flow) and show `reason` as the headline message.
_ERROR_REASONS = {
    "youtube_auth_required": (
        "YouTube connection expired or was blocked. Reconnect YouTube "
        "(sign in again in the app) and retry."
    ),
    "provider_not_configured": (
        "The server's reconciliation provider is misconfigured — retrying "
        "cannot succeed until the server settings are fixed (see detail)."
    ),
    "content_filtered": (
        "The model blocked its output under its content-filtering policy for "
        "this song. Try again (it isn't always deterministic), a different "
        "source/upload, or a lower analysis depth."
    ),
}


def _error_response(status_code: int, detail: str, error_code: str | None) -> JSONResponse:
    body: dict = {"detail": detail}
    if error_code:
        body["errorCode"] = error_code
        reason = _ERROR_REASONS.get(error_code)
        if reason:
            body["reason"] = reason
    return JSONResponse(body, status_code=status_code)


def _acquisition_error_response(e: AcquisitionError) -> JSONResponse:
    code = "youtube_auth_required" if isinstance(e, YouTubeAuthError) else None
    return _error_response(502, str(e), code)


class AcquireRequest(BaseModel):
    title: Optional[str] = None
    artist: Optional[str] = None
    youtubeUrlOrId: Optional[str] = None


@app.post("/v1/audio/acquire")
def post_acquire(req: AcquireRequest):
    try:
        acquired = acquire(title=req.title, artist=req.artist, video_url_or_id=req.youtubeUrlOrId)
    except AcquisitionError as e:
        return _acquisition_error_response(e)
    return _asdict(acquired)


class AnalyzeRequest(BaseModel):
    # one of: a server-side audio path, or acquisition parameters
    audioPath: Optional[str] = None
    title: Optional[str] = None
    artist: Optional[str] = None
    youtubeUrlOrId: Optional[str] = None
    # fast: sample a few windows across the musical span (quick + cheap);
    # standard: honor SNOOCLE_MIR_MAX_ANALYSIS_SECONDS; thorough: full track.
    accuracy: Literal["fast", "standard", "thorough"] = "standard"


@app.post("/v1/audio/analyze")
async def post_analyze(req: AnalyzeRequest) -> dict:
    path = req.audioPath
    video_id = None
    if path is None:
        try:
            acquired = await run_in_threadpool(
                acquire, title=req.title, artist=req.artist, video_url_or_id=req.youtubeUrlOrId
            )
        except AcquisitionError as e:
            return _acquisition_error_response(e)
        path = acquired.path
        video_id = acquired.video_id
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"no such audio file: {path}")
    # MIR is CPU-bound and runs for minutes on a full song; offload it so it
    # doesn't block the event loop shared with the embedded MCP transport
    # (same treatment as /v1/audio/analyze/upload).
    try:
        analysis = await run_in_threadpool(analyze_audio, path, req.accuracy)
    except audio_utils.AudioToolError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return {"audioPath": path, "youtubeVideoId": video_id, "analysis": analysis.model_dump()}


@app.post("/v1/audio/analyze/upload")
async def post_analyze_upload(file: UploadFile = File(...)) -> dict:
    """MIR pitch analysis of an UPLOADED audio OR video file — no YouTube, no
    network, no AI. Any ffmpeg-readable container works: audio
    (mp3/wav/m4a/flac/ogg/opus) or video (mp4/mov/webm/mkv/...); for video the
    audio track is extracted with ffmpeg before analysis. Returns beats/
    downbeats, chord timeline (sounding harmony), structural sections, bpm, and
    key. A file with no decodable audio stream is a 422.

    This is the "bring your own recording" path for a file the caller already
    holds — the counterpart to POST /v1/audio/analyze, which takes a server
    path or acquires from YouTube.
    """
    import shutil

    src = await _save_upload(file)
    try:
        # MIR is CPU-bound and can run for seconds; offload it so it doesn't
        # block the event loop shared with the embedded MCP transport.
        analysis = await run_in_threadpool(analyze_audio, src)
    except audio_utils.AudioToolError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    finally:
        shutil.rmtree(src.parent, ignore_errors=True)
    return {"filename": file.filename, "analysis": analysis.model_dump()}


# --- step 5: reconciliation --------------------------------------------------


class ReconcileRequest(BaseModel):
    title: str
    artist: str
    candidates: list[CandidateSource] = Field(default_factory=list)
    mir: Optional[MirAnalysis] = None
    provider: Optional[str] = None  # anthropic | anthropic-agent | openai | gemini | agent | mock
    model: Optional[str] = None
    audioPath: Optional[str] = None
    attachAudio: Optional[bool] = None
    youtubeVideoId: Optional[str] = None
    # For the "agent" provider: the media the song came from (YouTube watch URL
    # or another media URL). Defaults to the YouTube URL when youtubeVideoId set.
    mediaUrl: Optional[str] = None


def _reconcile_response(result: ReconcileResult) -> dict:
    return {
        "song": result.song.model_dump(),
        "provider": result.provider,
        "model": result.model,
        "attempts": result.attempts,
        "audioAttached": result.audio_attached,
        "usage": result.usage,
    }


@app.post("/v1/reconcile")
def post_reconcile(req: ReconcileRequest) -> dict:
    try:
        result = reconcile(
            req.title,
            req.artist,
            req.candidates,
            req.mir,
            provider_name=req.provider,
            model=req.model,
            audio_path=req.audioPath,
            attach_audio=req.attachAudio,
            youtube_video_id=req.youtubeVideoId,
            media_url=req.mediaUrl,
        )
    except (ReconcileError, ProviderError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return _reconcile_response(result)


# --- full pipeline -----------------------------------------------------------


class AnalysisScopeRequest(BaseModel):
    """``{"listen": bool, "reconcile": bool}`` — the two evidence-gathering
    stages of a re-analysis, each independently switchable.

    Each flag defaults to True so a partial object (``{"listen": false}``) can
    only ever turn OFF what it names; a client can never accidentally disable a
    stage by omitting it. The all-off case is legitimate and means "apply my
    notes to the song I already have".
    """

    listen: bool = True      # acquire the audio + run MIR on it
    reconcile: bool = True   # search the web for chord/lyric sources

    def to_scope(self) -> AnalysisScope:
        return AnalysisScope(listen=self.listen, reconcile=self.reconcile)


class PipelineRequest(BaseModel):
    # title+artist may be omitted when youtubeUrlOrId is given — the pipeline
    # derives them from the media's own metadata.
    title: Optional[str] = None
    artist: Optional[str] = None
    youtubeUrlOrId: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    attachAudio: Optional[bool] = None
    skipAudio: bool = False
    maxCandidates: Optional[int] = Field(default=None, ge=1, le=20)
    expectedVersion: Optional[str] = None  # optimistic lock for re-analyses
    # MIR effort/speed trade-off, surfaced to the app UI as an accuracy picker:
    # fast (sampled windows) | standard (default) | thorough (always full track)
    accuracy: Optional[Literal["fast", "standard", "thorough"]] = None
    # Single analysis-depth preset (fast|standard|thorough) that bundles MIR
    # accuracy + agent effort + tool budget + time alignment. Supersedes
    # `accuracy` when set; the app sends this one field.
    analysisDepth: Optional[Literal["fast", "standard", "thorough"]] = None
    # Human-in-the-loop re-run: free-text correction notes and/or the prior
    # human-edited Song, fed to the reconciler as high-priority evidence so a
    # re-analysis honors the user's fixes instead of rediscovering from scratch.
    guidance: Optional[str] = None
    priorSong: Optional[dict] = None
    # Re-analysis SCOPE: which evidence-gathering stages this run may do.
    # ABSENT means "no opinion" -> the full pipeline, exactly as before this
    # field existed; every pre-scope client and test is unaffected. Present
    # means constrain. Reconciliation itself always runs — the flags decide
    # what evidence it is handed, not whether it happens.
    scope: Optional[AnalysisScopeRequest] = None
    # Force the deterministic caches (MIR, discovery) to recompute and
    # overwrite. The caches key on everything that can change an answer, so
    # this is only needed for the one change a key cannot see: an engine
    # upgraded IN PLACE without its id changing. Orthogonal to `scope` —
    # `scope.listen=false` means "don't listen at all", this means "listen
    # again for real".
    refreshCache: bool = False
    # Opt OUT of the audio-data guard: allow this run to store a version that
    # empties audio.beats / nulls metadata.bpm relative to the prior one, and
    # allow a listen=false run with no prior version to carry timing forward
    # from. Both are silent-data-loss shapes, so they are refused by default
    # and only ever happen when a caller says explicitly that it means it.
    allowTimingLoss: bool = False

    @model_validator(mode="after")
    def _identity_or_url(self) -> "PipelineRequest":
        if not ((self.title and self.artist) or self.youtubeUrlOrId):
            raise ValueError("provide title and artist, or youtubeUrlOrId to derive them from")
        return self


@app.post("/v1/songs/analyze")
async def post_songs_analyze(req: PipelineRequest) -> dict:
    try:
        report = await run_pipeline_async(
            req.title,
            req.artist,
            youtube_url_or_id=req.youtubeUrlOrId,
            provider=req.provider,
            model=req.model,
            attach_audio=req.attachAudio,
            skip_audio=req.skipAudio,
            max_candidates=req.maxCandidates,
            expected_version=req.expectedVersion,
            accuracy=req.accuracy,
            analysis_depth=req.analysisDepth,
            guidance=req.guidance,
            prior_song=req.priorSong,
            scope=req.scope.to_scope() if req.scope is not None else None,
            refresh_cache=req.refreshCache,
            allow_timing_loss=req.allowTimingLoss,
        )
    except VersionConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except PipelineStepError as e:
        # A fatal step (reconcile/store) failed or timed out — name it, include
        # the per-step outcomes (str(e) carries the "[steps: ...]" summary),
        # and when the root cause is classified (e.g. dead YouTube session),
        # add errorCode + reason so the client can offer the fix action.
        return _error_response(502, str(e), e.error_code)
    assert report.reconcile is not None
    return {
        "songId": report.song_id,
        "steps": report.steps,
        "storedVersion": report.stored_version,
        "runId": report.run_id,  # fetch the step trace at /v1/runs/{runId}
        # What this run reused vs recomputed (see manifest.py). Descriptive
        # only — the admin UI renders it; no client should branch on it.
        "evidenceManifest": report.evidence_manifest,
        **_reconcile_response(report.reconcile),
    }


# --- per-song reconciliation notes -------------------------------------------
# Free-text "how to build this song" instructions, stored beside the song (not
# in it — the Song document is the app's content contract) and replayed as
# reconciler guidance by every analyze of the same song id.


class NotesRequest(BaseModel):
    notes: str = ""


def _notes_response(song_id: str, doc: dict | None) -> dict:
    return {
        "songId": song_id,
        "notes": (doc or {}).get("notes", "") or "",
        "updatedAt": (doc or {}).get("updated_at"),
    }


@app.get("/v1/songs/{song_id}/notes")
def get_song_notes(song_id: str) -> dict:
    """Notes for a song, or an empty document when there are none.

    Never 404: the app asks this for every song it opens, so "no notes" is a
    normal answer, not an error the client has to special-case.
    """
    from .store.song_notes import get_song_notes_store

    return _notes_response(song_id, get_song_notes_store().get_record(song_id))


@app.put("/v1/songs/{song_id}/notes")
def put_song_notes(song_id: str, req: NotesRequest) -> dict:
    from .store.song_notes import MAX_NOTES_CHARS, get_song_notes_store

    if len(req.notes or "") > MAX_NOTES_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"notes too long: {len(req.notes)} chars (limit {MAX_NOTES_CHARS})"
            ),
        )
    store = get_song_notes_store()
    store.set(song_id, req.notes)  # empty/whitespace deletes
    return _notes_response(song_id, store.get_record(song_id))


@app.delete("/v1/songs/{song_id}/notes")
def delete_song_notes(song_id: str) -> dict:
    from .store.song_notes import get_song_notes_store

    return {"deleted": get_song_notes_store().delete(song_id)}


# --- agent run traces (watch the reconciler's step-by-step logic) ------------


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """The full step trace of one reconciliation run (live record, then store)."""
    from .store.runs import fetch_run

    run = fetch_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
    return run


@app.get("/v1/songs/{song_id}/runs")
def get_song_runs(song_id: str) -> dict:
    """Recent reconciliation runs for a song, newest first (summaries only)."""
    from .store.runs import get_run_store

    runs = get_run_store().list_runs(song_id, limit=25)
    return {"songId": song_id, "runs": runs}


# --- evaluation: score the agent against human-approved gold versions --------


class GoldRequest(BaseModel):
    version: str


def _run_process_metrics(song_id: str) -> dict:
    """Process metrics (cost/effort/latency) from the song's latest run trace."""
    from .store.runs import get_run_store

    store = get_run_store()
    summaries = store.list_runs(song_id, limit=1)
    if not summaries:
        return {}
    run = store.get_run(summaries[0]["runId"]) or {}
    steps = run.get("steps") or []
    repairs = sum(1 for s in steps if s.get("kind") == "repair")
    tool_calls = sum(1 for s in steps if s.get("kind") == "tool")
    final = next((s for s in steps if s.get("kind") == "final"), {})
    usage = (final.get("detail") or {}).get("usage") or {}
    return {
        "runId": run.get("runId"),
        "depth": run.get("depth"),
        "model": run.get("model"),
        "configVersion": run.get("configVersion"),
        "firstPassValid": repairs == 0,
        "attempts": repairs + 1,
        "toolCalls": tool_calls,
        "inputTokens": usage.get("input_tokens"),
        "outputTokens": usage.get("output_tokens"),
    }


@app.put("/v1/songs/{song_id}/gold")
def put_gold(song_id: str, req: GoldRequest) -> dict:
    """Mark one of a song's versions as the ground-truth 'gold' for eval."""
    from .store.evals import get_eval_store

    # the version must exist for this song
    try:
        get_store().get(song_id, version=req.version)
    except StoreError as e:
        raise _store_error_response(e) from e
    get_eval_store().set_gold(song_id, req.version)
    return {"songId": song_id, "goldVersion": req.version}


@app.get("/v1/songs/{song_id}/gold")
def get_gold(song_id: str) -> dict:
    from .store.evals import get_eval_store

    return {"songId": song_id, "goldVersion": get_eval_store().get_gold(song_id)}


@app.get("/v1/songs/{song_id}/score")
def get_score(song_id: str, candidate: Optional[str] = None) -> dict:
    """Score a candidate version (default: current) against the song's gold."""
    from .eval import score_song
    from .store.evals import get_eval_store

    gold_version = get_eval_store().get_gold(song_id)
    if not gold_version:
        raise HTTPException(status_code=400, detail=f"no gold version set for {song_id}")
    store = get_store()
    try:
        gold = store.get(song_id, version=gold_version)
        cand = store.get(song_id, version=candidate)  # None -> current
    except StoreError as e:
        raise _store_error_response(e) from e
    return {
        "songId": song_id,
        "goldVersion": gold_version,
        "candidateVersion": candidate or store.current_version(song_id),
        "metrics": score_song(cand, gold),
    }


@app.get("/v1/eval/scorecard")
def get_scorecard() -> dict:
    """Score every gold-marked song's current version against its gold, and
    attach the latest run's process metrics. The agent's report card."""
    from .eval.scorecard import build_scorecard

    return build_scorecard(get_store(), process_metrics=_run_process_metrics)


def _aggregate_scores(metrics: list[dict]) -> dict:
    if not metrics:
        return {}
    keys = ["chordSimilarity", "chordRootSimilarity", "lyricSimilarity",
            "sectionSimilarity", "overall"]
    out = {k: round(sum(m[k] for m in metrics) / len(metrics), 4) for k in keys}

    # Optional metrics average only over the songs that HAVE them, and stay
    # None when none do — a scorecard column reading "—" means "not measured";
    # 0.0 would mean "measured, and terrible". Those must not be confusable.
    for key, places in (("timingMAE", 3), ("chordTimeCoverage", 4),
                        ("lineTimeCoverage", 4), ("confidentLineShare", 4)):
        present = [m[key] for m in metrics if m.get(key) is not None]
        out[key] = round(sum(present) / len(present), places) if present else None
    return out


# --- analysis job queue (broker) --------------------------------------------
# The server holds jobs; an external worker claims and runs them. See
# store/jobs.py for why, and docs/WORKER.md for the worker itself.
#
# Single-song analysis (POST /v1/songs/analyze) is UNCHANGED and still runs
# in-process — it completes inside its own request, so it needs no worker and
# no always-on CPU. The queue is for work that outlives a request.


class QueueItem(BaseModel):
    title: Optional[str] = None
    artist: Optional[str] = None
    youtubeUrlOrId: Optional[str] = None
    wants: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _identity_or_url(self) -> "QueueItem":
        if not (self.title or self.youtubeUrlOrId):
            raise ValueError("provide a title (with artist when known), or youtubeUrlOrId")
        return self


class QueueRequest(BaseModel):
    """Either structured `items`, or `text` — one song per line, exactly what
    the admin's 'Add many' textarea contains."""

    items: list[QueueItem] = Field(default_factory=list)
    text: Optional[str] = None
    provider: Optional[str] = None
    analysisDepth: Optional[Literal["fast", "standard", "thorough"]] = None
    # Optional extra work for a capable worker, e.g. ["stems"]. A job is only
    # ever handed to a worker that advertises everything it wants.
    wants: list[str] = Field(default_factory=list)


class ClaimRequest(BaseModel):
    worker: str = Field(min_length=1, max_length=128)
    capabilities: list[str] = Field(default_factory=list)
    leaseSeconds: int = Field(default=DEFAULT_LEASE_SECONDS, ge=30, le=3600)


class HeartbeatRequest(BaseModel):
    worker: str
    leaseSeconds: int = Field(default=DEFAULT_LEASE_SECONDS, ge=30, le=3600)


class CompleteRequest(BaseModel):
    worker: str
    songId: str
    runId: Optional[str] = None
    storedVersion: Optional[str] = None


class FailRequest(BaseModel):
    worker: str
    error: str
    # A worker that knows the failure is permanent (a 404 video, an unparseable
    # song) says so, and the broker doesn't burn two more attempts on it.
    retry: bool = True


@app.post("/v1/queue")
def post_queue(req: QueueRequest) -> dict:
    specs: list[dict] = [i.model_dump(exclude_none=True) for i in req.items]
    if req.text:
        for line in req.text.splitlines():
            parsed = parse_batch_line(line)
            if parsed:
                specs.append(parsed)
    if not specs:
        raise HTTPException(status_code=400, detail="no songs to queue")
    if len(specs) > MAX_ITEMS_PER_SUBMIT:
        raise HTTPException(
            status_code=400,
            detail=f"too many items: {len(specs)} (max {MAX_ITEMS_PER_SUBMIT} per submit)",
        )
    jobs = get_job_store().submit(
        specs, provider=req.provider, analysis_depth=req.analysisDepth,
        wants=req.wants,
    )
    return {"queued": len(jobs), "jobs": [j.to_json() for j in jobs]}


@app.get("/v1/queue")
def get_queue_status(limit: int = 200) -> dict:
    store = get_job_store()
    return {
        "jobs": [j.to_json() for j in store.list_jobs(limit=limit)],
        **store.stats(),
        "maxPerSubmit": MAX_ITEMS_PER_SUBMIT,
    }


@app.post("/v1/queue/claim")
def post_queue_claim(req: ClaimRequest) -> dict:
    """A worker asks for work. 204 means 'nothing for you right now' — an
    empty queue is the normal case, not an error, and a worker polling every
    few seconds should not be reading exception paths."""
    job = get_job_store().claim(
        req.worker, capabilities=req.capabilities, lease_seconds=req.leaseSeconds
    )
    if job is None:
        return JSONResponse(status_code=204, content=None)
    return job.to_worker_json()


@app.post("/v1/queue/{job_id}/heartbeat")
def post_queue_heartbeat(job_id: str, req: HeartbeatRequest) -> dict:
    try:
        job = get_job_store().heartbeat(job_id, req.worker, lease_seconds=req.leaseSeconds)
    except UnknownJobError as e:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}") from e
    except WrongWorkerError as e:
        # The lease was lost (expired and reclaimed, or cancelled). 409 tells
        # the worker to stop and drop its result rather than racing whoever
        # holds it now.
        raise HTTPException(status_code=409, detail=str(e)) from e
    return {"leaseExpiresAt": job.lease_expires_at, "status": job.status}


@app.post("/v1/queue/{job_id}/complete")
def post_queue_complete(job_id: str, req: CompleteRequest) -> dict:
    try:
        job = get_job_store().complete(
            job_id, req.worker, song_id=req.songId,
            run_id=req.runId, stored_version=req.storedVersion,
        )
    except UnknownJobError as e:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}") from e
    except WrongWorkerError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return job.to_json()


@app.post("/v1/queue/{job_id}/fail")
def post_queue_fail(job_id: str, req: FailRequest) -> dict:
    try:
        job = get_job_store().fail(job_id, req.worker, req.error, retry=req.retry)
    except UnknownJobError as e:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}") from e
    except WrongWorkerError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return job.to_json()


@app.post("/v1/queue/{job_id}/retry")
def post_queue_retry(job_id: str) -> dict:
    try:
        return get_job_store().retry(job_id).to_json()
    except UnknownJobError as e:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}") from e
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@app.post("/v1/queue/{job_id}/cancel")
def post_queue_cancel(job_id: str) -> dict:
    try:
        return get_job_store().cancel(job_id).to_json()
    except UnknownJobError as e:
        raise HTTPException(status_code=404, detail=f"no such job: {job_id}") from e
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@app.delete("/v1/queue")
def delete_queue_finished() -> dict:
    """Drop finished jobs. Queued and leased jobs are untouched."""
    return {"removed": get_job_store().clear_finished()}


# --- stems (B4) --------------------------------------------------------------
# Separation runs where the ML extras and the CPU are: a worker. This process
# only ever reads the cache and streams files, which is why none of these
# endpoints import demucs. On Cloud Run they will report "no stems yet" until
# something with the [stems] extra has run the job — see docs/STEMS.md for why
# that boundary is where it is.


@app.post("/v1/songs/{song_id}/stems", status_code=202)
def post_song_stems(song_id: str, model: str = stems.DEFAULT_MODEL,
                    force: bool = False) -> dict:
    """Queue a separation. 202 because this takes minutes, not milliseconds."""
    if not stems.known_model(model):
        raise HTTPException(
            status_code=400,
            detail=f"unknown model {model!r} (known: {sorted(stems.MODELS)})",
        )
    try:
        song = get_store().get(song_id)
    except StoreError as e:
        raise _store_error_response(e) from e

    # Separate the upload the song's times were measured against, not whatever
    # video happens to be linked — stems whose timeline disagrees with the
    # song's are worse than no stems.
    video = song.audio.analyzedVideoId or song.audio.youtubeVideoId
    if not video:
        raise HTTPException(
            status_code=409,
            detail="this song has no analyzed video to separate — run an analysis first",
        )

    existing = stems.available(song_id, model)
    if existing and existing.complete and not force:
        return {"status": "cached", **existing.to_json()}

    jobs = get_job_store().submit([{
        "kind": "stems",
        "targetSongId": song_id,
        "youtubeUrlOrId": video,
        "title": song.metadata.title,
        "artist": song.metadata.artist,
        "wants": ["stems"],
    }])
    return {"status": "queued", "model": model, "jobs": [j.to_json() for j in jobs]}


@app.get("/v1/songs/{song_id}/stems")
def get_song_stems(song_id: str, model: str = stems.DEFAULT_MODEL) -> dict:
    found = stems.available(song_id, model)
    if found is None:
        return {"songId": song_id, "model": model, "complete": False,
                "stems": [], "mixes": []}
    return found.to_json()


@app.get("/v1/songs/{song_id}/stems/{name}")
def get_song_stem_audio(song_id: str, name: str, model: str = stems.DEFAULT_MODEL):
    """Stream one stem or derived mix.

    FileResponse handles Range itself, which is what lets a player seek in a
    40 MB backing track instead of refetching it.
    """
    path = stems.stem_path(song_id, name, model)
    if path is None:
        raise HTTPException(status_code=404, detail=f"no stem {name!r} for {song_id}")
    return FileResponse(str(path), media_type="audio/wav", filename=f"{song_id}-{name}.wav")


@app.delete("/v1/songs/{song_id}/stems")
def delete_song_stems(song_id: str, model: Optional[str] = None) -> dict:
    """Reclaim the disk. Everything here regenerates from the source audio."""
    return {"removed": stems.clear(song_id, model)}


@app.post("/v1/songs/{song_id}/align", status_code=202)
def post_song_align(song_id: str, language: str = "en") -> dict:
    """Queue forced alignment of this song's lyrics against its audio (B2).

    Queued rather than run here for the same reason as stems: whisperx and
    torch live on the worker, not in the Cloud Run image. The job additionally
    `wants` the vocals stem's engine implicitly — alignment works on the full
    mix, just less well — so it is not made to depend on stems existing.
    """
    try:
        song = get_store().get(song_id)
    except StoreError as e:
        raise _store_error_response(e) from e

    if not any((line.lyrics or "").strip() for line in song.lines):
        raise HTTPException(
            status_code=409,
            detail="this song has no lyrics to align",
        )
    video = song.audio.analyzedVideoId or song.audio.youtubeVideoId
    if not video:
        raise HTTPException(
            status_code=409,
            detail="this song has no analyzed video to align against — run an analysis first",
        )

    untimed = sum(1 for line in song.lines
                  if (line.lyrics or "").strip() and line.timeSeconds is None)
    jobs = get_job_store().submit([{
        "kind": "align",
        "targetSongId": song_id,
        "youtubeUrlOrId": video,
        "title": song.metadata.title,
        "artist": song.metadata.artist,
        "language": language,
        "wants": ["align"],
    }])
    return {"status": "queued", "untimedLines": untimed,
            "jobs": [j.to_json() for j in jobs]}


# --- registered OAuth clients ------------------------------------------------
# Dynamic client registration is open — that is what the spec requires, and it
# is what lets Claude connect without anyone pre-provisioning anything. The
# cost is that /oauth/register accepts anyone, so registrations accumulate:
# every reconnect, every experiment, every probe. None of them can DO anything
# without the owner's token at the consent screen, but a list nobody can see is
# a list nobody will notice something odd in.
#
# These two endpoints take the static SNOOCLE_API_TOKEN like the rest of /v1 —
# an OAuth token cannot reach them, which is deliberate: a connector must not
# be able to enumerate or delete its siblings.


@app.get("/v1/oauth/clients")
def get_oauth_clients() -> dict:
    from .oauth.store import get_oauth_store

    return {
        "clients": [
            {
                "clientId": c.client_id,
                "clientName": c.client_name,
                "redirectUris": c.redirect_uris,
                "scope": c.scope,
                "createdAt": c.created_at,
            }
            for c in get_oauth_store().list_clients()
        ]
    }


@app.delete("/v1/oauth/clients/{client_id}")
def delete_oauth_client(client_id: str) -> dict:
    """Revoke a registration and every token it holds."""
    from .oauth.store import get_oauth_store

    if not get_oauth_store().delete_client(client_id):
        raise HTTPException(status_code=404, detail=f"no such client: {client_id}")
    return {"deleted": client_id}


# --- step 7: versioned store -------------------------------------------------


@app.get("/v1/songs")
def get_songs() -> dict:
    """The library listing.

    ``songs`` stays exactly what it always was — a sorted list of ids — because
    the iOS app, the MCP tools and every existing script read it that way.
    ``items`` is the ADDITIVE Phase C addition: the same songs as list-view
    rows (title, artist, latestVersion, updatedAt, youtubeVideoId, hasTiming)
    so the player's library grid can render artwork, labels and a timing badge
    in one request instead of N+1 song fetches. Clients that don't know about
    ``items`` are unaffected; clients that do fall back to ``songs`` when a
    server predates it.
    """
    summaries = get_store().list_song_summaries()
    return {
        "songs": [s.id for s in summaries],
        "items": [s.to_json() for s in summaries],
    }


@app.get("/v1/songs/{song_id}")
def get_song(song_id: str, version: Optional[str] = None) -> dict:
    try:
        song = get_store().get(song_id, version=version)
    except StoreError as e:
        raise _store_error_response(e) from e
    return song.model_dump()


@app.get("/v1/songs/{song_id}/versions")
def get_song_versions(song_id: str) -> dict:
    versions = get_store().versions(song_id)
    if not versions:
        raise HTTPException(status_code=404, detail=f"song {song_id!r} not found")
    return {"songId": song_id, "versions": [dataclasses.asdict(v) for v in versions]}


@app.get("/v1/songs/{song_id}/diff", response_class=PlainTextResponse)
def get_song_diff(song_id: str, a: str, b: str) -> str:
    try:
        return get_store().diff(song_id, a, b)
    except StoreError as e:
        raise _store_error_response(e) from e


# --- export (master plan B6) ------------------------------------------------
# Deterministic serializers (export.py) — no LLM, no re-reconciliation. The
# chordpro/txt formats share the inline-bracket layout the generic
# chord-sheet parser (discovery/chordsheet.py) already understands, so
# exporting and re-pasting a song is a round trip, not a one-way dump.

_EXPORT_MEDIA_TYPES = {
    "chordpro": "text/plain; charset=utf-8",
    "txt": "text/plain; charset=utf-8",
    "json": "application/json",
}
_EXPORT_EXTENSIONS = {"chordpro": "cho", "txt": "txt", "json": "json"}


@app.get("/v1/songs/{song_id}/export")
def get_song_export(song_id: str, format: Literal["chordpro", "txt", "json"] = "chordpro"):
    try:
        song = get_store().get(song_id)
    except StoreError as e:
        raise _store_error_response(e) from e

    if format == "json":
        body: Any = song.model_dump()
    elif format == "txt":
        body = to_txt(song)
    else:
        body = to_chordpro(song)

    ext = _EXPORT_EXTENSIONS[format]
    headers = {"Content-Disposition": f'attachment; filename="{song_id}.{ext}"'}
    if format == "json":
        return JSONResponse(body, headers=headers)
    return PlainTextResponse(body, media_type=_EXPORT_MEDIA_TYPES[format], headers=headers)


class SaveSongRequest(BaseModel):
    song: Song
    message: str = "Manual save"
    expectedVersion: Optional[str] = None


@app.post("/v1/songs/{song_id}")
def post_song(song_id: str, req: SaveSongRequest) -> dict:
    if req.song.id != song_id:
        raise HTTPException(status_code=400, detail="song.id does not match URL")
    try:
        saved = get_store().save(req.song, req.message, expected_version=req.expectedVersion)
    except VersionConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except StoreError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"version": saved.version, "timestamp": saved.timestamp, "message": saved.message}


# --- cross-video offset alignment (Phase B / master plan B3) ---------------
# A song's chord/line/beat times are all measured against ONE analyzed
# upload (audio.analyzedVideoId). Playing a DIFFERENT upload of the same
# song (a re-upload, a live version, a lyric video with different intro
# padding) needs a constant seconds-to-add correction rather than a whole
# re-analysis -- this estimates that correction via audio cross-correlation
# (timing/offset.py) and stores it in audio.videoOffsets[videoId].


class VideoOffsetRequest(BaseModel):
    videoId: str
    expectedVersion: Optional[str] = None
    # Skip estimation and trust the caller's value directly (e.g. a human
    # eyeballed it against the player) -- stored at confidence 1.0, and the
    # offset_min_confidence gate never applies.
    offsetSeconds: Optional[float] = None


@app.post("/v1/songs/{song_id}/video-offset")
async def post_video_offset(song_id: str, req: VideoOffsetRequest) -> dict:
    try:
        song = get_store().get(song_id)
    except StoreError as e:
        raise _store_error_response(e) from e

    if req.offsetSeconds is not None:
        offset_seconds = req.offsetSeconds
        confidence = 1.0
        note = "manual override (caller-supplied offsetSeconds)"
    else:
        ref_video_id = song.audio.analyzedVideoId
        if not ref_video_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"song {song_id!r} has no audio.analyzedVideoId -- it hasn't "
                    "been through a full analyze pass yet, so there is no reference "
                    f"audio to compare {req.videoId!r} against. Analyze it first, or "
                    "pass offsetSeconds directly to set the offset manually."
                ),
            )
        try:
            ref_acquired, other_acquired = await run_in_threadpool(
                lambda: (
                    acquire(video_url_or_id=ref_video_id),
                    acquire(video_url_or_id=req.videoId),
                )
            )
        except AcquisitionError as e:
            return _acquisition_error_response(e)
        estimate = await run_in_threadpool(
            estimate_offset,
            ref_acquired.path,
            other_acquired.path,
            settings.offset_max_search_seconds,
        )
        offset_seconds = estimate.offset_seconds
        confidence = estimate.confidence
        if confidence < settings.offset_min_confidence:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"video-offset estimate too unreliable to store: confidence "
                    f"{confidence:.2f} < required {settings.offset_min_confidence:.2f} "
                    f"(estimated offsetSeconds={offset_seconds:.2f}). The two videos "
                    "may not be the same song/performance, or the audio has too "
                    "little rhythmic content to align reliably; pass offsetSeconds "
                    "directly to override."
                ),
            )
        note = f"cross-correlation estimate vs analyzed video {ref_video_id}"

    new_offsets = dict(song.audio.videoOffsets)
    new_offsets[req.videoId] = offset_seconds
    updated_audio = song.audio.model_copy(update={"videoOffsets": new_offsets})
    entry = ProvenanceEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        actor="snoocle-server/timing",
        action="video-offset",
        sources=[req.videoId],
        confidence=confidence,
        notes=note,
    )
    updated = song.model_copy(
        update={"audio": updated_audio, "provenance": list(song.provenance) + [entry]}
    )
    updated = Song.model_validate(updated.model_dump())

    try:
        saved = get_store().save(
            updated, f"video-offset: {req.videoId}", expected_version=req.expectedVersion
        )
    except VersionConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    return {
        "videoId": req.videoId,
        "offsetSeconds": offset_seconds,
        "confidence": confidence,
        "version": saved.version,
    }


# --- YouTube acquisition cookies (in-app sign-in / manual upload) -------------
# The iOS app can open a YouTube sign-in webview, harvest the session cookies,
# and POST them here so server-side yt-dlp gets past YouTube's datacenter
# bot-check — and refresh them later without a redeploy. These endpoints handle
# the user's Google session, so they REQUIRE the app-level token to be
# configured (SNOOCLE_API_TOKEN); otherwise they refuse (409) rather than expose
# session cookies on an unauthenticated service.


class YouTubeCookie(BaseModel):
    name: str
    value: str
    domain: str = ".youtube.com"
    path: str = "/"
    expires: Optional[int] = None  # unix epoch; None/0 = session cookie
    secure: bool = True
    httpOnly: bool = False  # accepted from HTTPCookie; not used in the Netscape line


class YouTubeCookiesRequest(BaseModel):
    # provide the raw Netscape cookies.txt, OR a structured cookie array the app
    # harvests from its webview's cookie store (converted here).
    cookiesTxt: Optional[str] = None
    cookies: Optional[list[YouTubeCookie]] = None
    source: str = "app"

    @model_validator(mode="after")
    def _need_cookies(self) -> "YouTubeCookiesRequest":
        if not (self.cookiesTxt or self.cookies):
            raise ValueError("provide cookiesTxt (Netscape cookies.txt) or a cookies array")
        return self


def _cookies_to_netscape(cookies: list[YouTubeCookie]) -> str:
    lines = ["# Netscape HTTP Cookie File"]
    for c in cookies:
        domain = c.domain or ".youtube.com"
        include_sub = "TRUE" if domain.startswith(".") else "FALSE"
        secure = "TRUE" if c.secure else "FALSE"
        expiry = int(c.expires) if c.expires else 0
        lines.append("\t".join([domain, include_sub, c.path or "/", secure, str(expiry), c.name, c.value]))
    return "\n".join(lines) + "\n"


def _require_app_auth_configured() -> None:
    if not settings.api_token:
        raise HTTPException(
            status_code=409,
            detail=(
                "refusing to manage YouTube session cookies on an unauthenticated service; "
                "set SNOOCLE_API_TOKEN (and redeploy) first so this endpoint is gated"
            ),
        )


@app.post("/v1/config/youtube-cookies")
def post_youtube_cookies(req: YouTubeCookiesRequest) -> dict:
    _require_app_auth_configured()
    txt = req.cookiesTxt if req.cookiesTxt else _cookies_to_netscape(req.cookies or [])
    if count_cookie_lines(txt) == 0:
        raise HTTPException(status_code=422, detail="no cookie entries found")
    rec = get_store().set_youtube_cookies(txt, source=req.source)
    return {"status": "stored", "updatedAt": rec.updated_at, "source": rec.source,
            "lineCount": rec.line_count}


@app.get("/v1/config/youtube-cookies")
def get_youtube_cookies() -> dict:
    _require_app_auth_configured()
    rec = get_store().youtube_cookies_status()
    if rec is None:
        return {"configured": False}
    return {"configured": True, "updatedAt": rec.updated_at, "source": rec.source,
            "lineCount": rec.line_count}


@app.delete("/v1/config/youtube-cookies")
def delete_youtube_cookies() -> dict:
    _require_app_auth_configured()
    get_store().clear_youtube_cookies()
    return {"status": "cleared"}


# --- agent programming: runtime-editable instructions / tooling --------------


def _dt_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _effective_agent_defaults() -> dict:
    """The built-in defaults the GUI shows as placeholders (what runs with no
    override) — kept in one place so the Workbench never hardcodes them."""
    from .reconcile.agent_config import KNOWN_TOOLS
    from .reconcile.anthropic_agent import _OUTPUT_CONTRACT, _PROMPT_RECIPE, _PROMPT_THEORY

    return {
        "theoryRules": _PROMPT_THEORY,
        "retrievalRecipe": _PROMPT_RECIPE,
        "maxTurns": settings.anthropic_agent_max_turns,
        "effort": settings.anthropic_agent_effort,
        "model": settings.llm_model or settings.anthropic_agent_model,
        "budgets": {"maxWebSearch": 2, "maxFetch": 3, "maxWindows": 2},
        "tools": sorted(KNOWN_TOOLS),
        "lockedOutputContract": _OUTPUT_CONTRACT,
    }


@app.get("/v1/config/agent")
def get_agent_config_endpoint() -> dict:
    from .reconcile.agent_config import AgentConfig, config_version
    from .store.agent_config import get_agent_config_store

    _require_app_auth_configured()
    doc = get_agent_config_store().get()
    cfg = AgentConfig.model_validate(doc) if doc else AgentConfig()
    return {
        "config": cfg.model_dump(),
        "configVersion": config_version(cfg),
        "isDefault": cfg.is_default(),
        "defaults": _effective_agent_defaults(),
    }


@app.put("/v1/config/agent")
def put_agent_config_endpoint(body: dict) -> dict:
    from pydantic import ValidationError

    from .reconcile.agent_config import AgentConfig, config_version
    from .store.agent_config import get_agent_config_store

    _require_app_auth_configured()
    try:
        cfg = AgentConfig.model_validate(body)
    except ValidationError as e:
        # drop ctx (holds a non-JSON-serializable exception) and the url noise
        raise HTTPException(
            status_code=422, detail=e.errors(include_url=False, include_context=False)
        ) from e
    doc = cfg.model_dump()
    doc["updated_at"] = _dt_now()
    doc["source"] = "rest"
    get_agent_config_store().set(doc)
    return {"status": "stored", "configVersion": config_version(cfg), "updatedAt": doc["updated_at"]}


class InstructionsAppendRequest(BaseModel):
    add: str = ""


# One process-wide lock around the append's read-modify-write: two clients
# promoting a note at the same time must not both read the same config and
# write back disjoint versions, silently dropping one instruction.
_INSTRUCTIONS_APPEND_LOCK = threading.Lock()


@app.post("/v1/config/agent/instructions")
def post_agent_instructions_endpoint(req: InstructionsAppendRequest) -> dict:
    """Append one standing instruction to `instructions_extra`, atomically.

    Exists so a client never has to read-modify-write the whole agent config
    (and clobber a concurrent edit) just to add a line.
    """
    from .reconcile.agent_config import AgentConfig, config_version
    from .store.agent_config import get_agent_config_store

    _require_app_auth_configured()
    line = (req.add or "").strip()
    if not line:
        raise HTTPException(status_code=400, detail="`add` must be a non-empty instruction")

    store = get_agent_config_store()
    with _INSTRUCTIONS_APPEND_LOCK:
        doc = store.get()
        cfg = AgentConfig.model_validate(doc) if doc else AgentConfig()
        existing = cfg.instructions_extra or ""
        # Re-promoting the same note is a no-op, not a duplicated instruction.
        if line not in [ln.strip() for ln in existing.splitlines() if ln.strip()]:
            merged = f"{existing.rstrip()}\n{line}" if existing.strip() else line
            new_doc = cfg.model_copy(update={"instructions_extra": merged}).model_dump()
            new_doc["updated_at"] = _dt_now()
            new_doc["source"] = "rest"
            store.set(new_doc)
            cfg = AgentConfig.model_validate(new_doc)
    return {
        "config": cfg.model_dump(),
        "configVersion": config_version(cfg),
        "isDefault": cfg.is_default(),
    }


@app.delete("/v1/config/agent")
def delete_agent_config_endpoint() -> dict:
    from .store.agent_config import get_agent_config_store

    _require_app_auth_configured()
    get_agent_config_store().clear()
    return {"status": "reset"}


# --- deterministic audio utilities (no AI) -----------------------------------


async def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "audio").suffix or ".bin"
    tmp = Path(tempfile.mkdtemp(prefix="snoocle-upload-")) / f"in{suffix}"
    tmp.write_bytes(await upload.read())
    return tmp


def _serve(path: Path) -> FileResponse:
    return FileResponse(path, filename=path.name)


@app.post("/v1/audio/convert")
async def post_convert(to: str, file: UploadFile = File(...)) -> FileResponse:
    src = await _save_upload(file)
    dst = src.with_name(f"converted.{to.lstrip('.')}")
    try:
        audio_utils.convert(src, dst)
    except audio_utils.AudioToolError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _serve(dst)


@app.post("/v1/audio/trim")
async def post_trim(start: float, end: float, file: UploadFile = File(...), to: Optional[str] = None) -> FileResponse:
    src = await _save_upload(file)
    fmt = (to or src.suffix.lstrip(".") or "wav").lstrip(".")
    dst = src.with_name(f"trimmed.{fmt}")
    try:
        audio_utils.trim(src, dst, start, end)
    except audio_utils.AudioToolError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _serve(dst)


@app.post("/v1/audio/normalize")
async def post_normalize(file: UploadFile = File(...), targetLufs: float = -16.0, to: Optional[str] = None) -> FileResponse:
    src = await _save_upload(file)
    fmt = (to or src.suffix.lstrip(".") or "wav").lstrip(".")
    dst = src.with_name(f"normalized.{fmt}")
    try:
        audio_utils.normalize(src, dst, target_lufs=targetLufs)
    except audio_utils.AudioToolError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _serve(dst)


@app.post("/v1/audio/probe")
async def post_probe(file: UploadFile = File(...)) -> dict:
    src = await _save_upload(file)
    try:
        return dataclasses.asdict(audio_utils.probe(src))
    except audio_utils.AudioToolError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# --- static single-page GUI --------------------------------------------------
# Browse / add / edit / versions / play, served as dependency-free static files
# by this same app. Mounted AFTER all API routes so it never shadows them, and
# BEFORE the /mcp route copy below. `/` redirects into it; the shell and its
# assets are exempt from the bearer-token middleware (every /v1 call it makes
# still carries the token). This is the ONLY static surface — no build, no CDN.
@app.get("/")
def root_redirect() -> RedirectResponse:
    return RedirectResponse("/ui/")


# The PWA manifest (master plan C5). Python's mimetypes table predates the
# .webmanifest extension, so StaticFiles would serve it as text/plain and some
# browsers refuse to install from that. Registering the type is the whole fix.
mimetypes.add_type("application/manifest+json", ".webmanifest")


class _RevalidatingStatic(StaticFiles):
    """Serve the GUI shell with ``Cache-Control: no-cache`` so the browser
    always revalidates against the ETag. Default StaticFiles sends no
    Cache-Control, letting browsers heuristically cache app.js/style.css — which
    means a deploy that ships new JS keeps running the OLD cached JS against the
    new index.html (new buttons/tabs with no handlers) until a hard refresh.
    ``no-cache`` makes each load a cheap 304 when unchanged and picks up fresh
    assets immediately after a deploy."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount(
    "/ui",
    _RevalidatingStatic(directory=str(Path(__file__).parent / "ui"), html=True),
    name="ui",
)


# --- embedded MCP route ------------------------------------------------------
# Register the MCP streamable-HTTP route (default path /mcp) onto this app,
# after all REST routes are defined. Copying the route rather than mounting the
# whole sub-app avoids a path prefix and trailing-slash mismatch, and keeps a
# single ASGI app with one lifespan. The session manager it dispatches to is
# started by _lifespan above.
for _route in _mcp_asgi_app.routes:
    app.router.routes.append(_route)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
