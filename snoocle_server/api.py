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
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from . import __version__
from .audio import utils as audio_utils
from .audio.acquire import AcquisitionError, acquire
from .config import settings
from .discovery import CandidateSource, discover_sources
from .discovery.search import SearchError
from .mcp_server import mcp as _mcp
from .mcp_server import resolve_http_transport as _resolve_mcp_security
from .mir import MirAnalysis, analyze_audio
from .pipeline import PipelineStepError, get_store, run_pipeline_async
from .reconcile import ReconcileResult, provider_capabilities, reconcile
from .reconcile.engine import ReconcileError
from .reconcile.providers import ProviderError
from .schema import Song, song_json_schema
from .store import StoreError, StoreUnavailableError, VersionConflictError, backend_label

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
        if not token or scope["type"] != "http" or scope.get("path") == "/healthz":
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
    import shutil

    def has(mod: str) -> bool:
        try:
            __import__(mod)
            return True
        except Exception:  # noqa: BLE001
            return False

    return {
        "status": "ok",
        "version": __version__,
        "ffmpeg": shutil.which(settings.ffmpeg_bin) is not None,
        "mirEngines": {
            "beats": "madmom" if has("madmom") else "librosa-fallback",
            "chords": "chord-cnn-lstm" if settings.chord_cnn_lstm_dir else "chroma-template-fallback",
            "structure": "songformer" if settings.songformer_dir else "librosa-agglomerative-fallback",
        },
        "llmProviders": provider_capabilities(),
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


class AcquireRequest(BaseModel):
    title: Optional[str] = None
    artist: Optional[str] = None
    youtubeUrlOrId: Optional[str] = None


@app.post("/v1/audio/acquire")
def post_acquire(req: AcquireRequest) -> dict:
    try:
        acquired = acquire(title=req.title, artist=req.artist, video_url_or_id=req.youtubeUrlOrId)
    except AcquisitionError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return _asdict(acquired)


class AnalyzeRequest(BaseModel):
    # one of: a server-side audio path, or acquisition parameters
    audioPath: Optional[str] = None
    title: Optional[str] = None
    artist: Optional[str] = None
    youtubeUrlOrId: Optional[str] = None


@app.post("/v1/audio/analyze")
def post_analyze(req: AnalyzeRequest) -> dict:
    path = req.audioPath
    video_id = None
    if path is None:
        try:
            acquired = acquire(title=req.title, artist=req.artist, video_url_or_id=req.youtubeUrlOrId)
        except AcquisitionError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        path = acquired.path
        video_id = acquired.video_id
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"no such audio file: {path}")
    analysis = analyze_audio(path)
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
    provider: Optional[str] = None  # anthropic | openai | gemini | agent | mock
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
    title: str
    artist: str
    youtubeUrlOrId: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    attachAudio: Optional[bool] = None
    skipAudio: bool = False
    maxCandidates: Optional[int] = Field(default=None, ge=1, le=20)
    expectedVersion: Optional[str] = None  # optimistic lock for re-analyses


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
        )
    except VersionConflictError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except PipelineStepError as e:
        # A fatal step (reconcile/store) failed or timed out — name it so the
        # client shows exactly where the pipeline broke.
        raise HTTPException(status_code=502, detail=f"{e.step}: {e.message}") from e
    assert report.reconcile is not None
    return {
        "songId": report.song_id,
        "steps": report.steps,
        "storedVersion": report.stored_version,
        **_reconcile_response(report.reconcile),
    }


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
