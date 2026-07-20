"""HTTP API — clean, typed, stateless; each pipeline step is its own endpoint.

The surface deliberately mirrors the MCP tool surface (mcp_server.py) so the
iOS app, curl, and agent callers all drive the same service layer. State
lives only in the git-backed store and the audio cache.
"""

from __future__ import annotations

import dataclasses
import os
import secrets
import tempfile
from contextlib import asynccontextmanager
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
from .audio import utils as audio_utils
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
from .schema import Song, song_json_schema
from .store import (
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


class _BearerTokenMiddleware:
    """Optional app-level static bearer token, enforced uniformly on the REST
    API and the embedded /mcp transport (this middleware wraps the whole ASGI
    app, so both surfaces share the one token — send it as
    `Authorization: Bearer <token>`).

    Active only when SNOOCLE_API_TOKEN is set; otherwise a pass-through so the
    default posture (Cloud Run IAM gates access) is unchanged. `/healthz` is
    always exempt so liveness probes work without the token. The token is read
    per request so it can be rotated/toggled without re-importing the app.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        token = settings.api_token
        # Exempt: liveness probes, and the static GUI shell (`/` redirect and
        # everything under `/ui`). The shell carries no secrets; every API call
        # it makes to `/v1/...` still requires the token.
        path = scope.get("path", "")
        if (
            not token
            or scope["type"] != "http"
            or path == "/healthz"
            or path == "/"
            or path.startswith("/ui")
        ):
            await self.app(scope, receive, send)
            return
        auth = Headers(scope=scope).get("authorization", "")
        if not (auth.startswith("Bearer ") and secrets.compare_digest(auth, f"Bearer {token}")):
            response = JSONResponse(
                {"detail": "missing or invalid bearer token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


# Wrap the entire app (REST routes + the /mcp transport appended below) so one
# token authorizes both surfaces when SNOOCLE_API_TOKEN is configured.
app.add_middleware(_BearerTokenMiddleware)


@app.exception_handler(StoreUnavailableError)
async def _store_unavailable_handler(request, exc: StoreUnavailableError) -> JSONResponse:
    # The store backend is down/misconfigured (e.g. the Firestore database
    # doesn't exist). 503, not a bare 500 — and never 404, which would falsely
    # read as "song not found".
    return JSONResponse({"detail": f"store unavailable: {exc}"}, status_code=503)


def _asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return obj


# --- health / meta ---------------------------------------------------------


@app.get("/healthz")
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
        **_reconcile_response(report.reconcile),
    }


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
        raise HTTPException(status_code=404, detail=str(e)) from e
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
        raise HTTPException(status_code=404, detail=str(e)) from e
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
    timings = [m["timingMAE"] for m in metrics if m.get("timingMAE") is not None]
    out["timingMAE"] = round(sum(timings) / len(timings), 3) if timings else None
    return out


# --- step 7: versioned store -------------------------------------------------


@app.get("/v1/songs")
def get_songs() -> dict:
    return {"songs": get_store().list_songs()}


@app.get("/v1/songs/{song_id}")
def get_song(song_id: str, version: Optional[str] = None) -> dict:
    try:
        song = get_store().get(song_id, version=version)
    except StoreError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
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
        raise HTTPException(status_code=404, detail=str(e)) from e


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


app.mount(
    "/ui",
    StaticFiles(directory=str(Path(__file__).parent / "ui"), html=True),
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
