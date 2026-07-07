"""HTTP API — clean, typed, stateless; each pipeline step is its own endpoint.

The surface deliberately mirrors the MCP tool surface (mcp_server.py) so the
iOS app, curl, and agent callers all drive the same service layer. State
lives only in the git-backed store and the audio cache.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import __version__
from .audio import utils as audio_utils
from .audio.acquire import AcquisitionError, acquire
from .config import settings
from .discovery import CandidateSource, discover_sources
from .discovery.search import SearchError
from .mir import MirAnalysis, analyze_audio
from .pipeline import get_store, run_pipeline
from .reconcile import ReconcileResult, provider_capabilities, reconcile
from .reconcile.engine import ReconcileError
from .reconcile.providers import ProviderError
from .schema import Song, song_json_schema
from .store import StoreError, VersionConflictError

app = FastAPI(
    title="Snoocle server",
    version=__version__,
    description="Audio-to-song-data foundry: web-sourced chord/lyric text + MIR analysis, "
    "reconciled by a configurable LLM into Snoocle Song JSON. Personal-use tool.",
)


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
        "store": str(settings.store_dir),
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


# --- step 5: reconciliation --------------------------------------------------


class ReconcileRequest(BaseModel):
    title: str
    artist: str
    candidates: list[CandidateSource] = Field(default_factory=list)
    mir: Optional[MirAnalysis] = None
    provider: Optional[str] = None  # anthropic | openai | gemini | mock
    model: Optional[str] = None
    audioPath: Optional[str] = None
    attachAudio: Optional[bool] = None
    youtubeVideoId: Optional[str] = None


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
def post_songs_analyze(req: PipelineRequest) -> dict:
    try:
        report = run_pipeline(
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
    except (ReconcileError, ProviderError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
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
    return dataclasses.asdict(saved)


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


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
