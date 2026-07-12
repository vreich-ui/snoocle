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
from .mir import MirAnalysis, analyze_audio as _analyze_audio
from .pipeline import get_store, run_pipeline
from .reconcile import provider_capabilities, reconcile as _reconcile
from .schema import Song, song_json_schema

mcp = FastMCP(
    "snoocle",
    instructions=(
        "Snoocle audio-to-song-data foundry (personal-use). Pipeline tools: "
        "discover_song -> acquire_audio -> analyze_audio -> reconcile_song, or "
        "analyze_and_store_song for the full flow with git-versioned persistence. "
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
    analysis = _analyze_audio(audio_path)
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
        candidates = discover_sources(title, artist)
    mir = None
    if mir_json:
        payload = json.loads(mir_json)
        mir = MirAnalysis.model_validate(payload.get("analysis", payload))
    result = _reconcile(
        title,
        artist,
        candidates,
        mir,
        provider_name=provider,
        model=model,
        audio_path=audio_path,
        attach_audio=attach_audio,
        youtube_video_id=youtube_video_id,
        media_url=media_url,
    )
    return {
        "song": result.song.model_dump(),
        "provider": result.provider,
        "model": result.model,
        "attempts": result.attempts,
        "audioAttached": result.audio_attached,
    }


@mcp.tool()
def analyze_and_store_song(
    title: str,
    artist: str,
    youtube_url_or_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    skip_audio: bool = False,
    expected_version: Optional[str] = None,
) -> dict:
    """Full pipeline: discover -> acquire -> MIR -> reconcile -> commit a new
    version to the git-backed store (never overwrites). Returns the song, the
    per-step report, and the committed version sha."""
    report = run_pipeline(
        title,
        artist,
        youtube_url_or_id=youtube_url_or_id,
        provider=provider,
        model=model,
        skip_audio=skip_audio,
        expected_version=expected_version,
    )
    assert report.reconcile is not None
    return {
        "songId": report.song_id,
        "steps": report.steps,
        "storedVersion": report.stored_version,
        "song": report.reconcile.song.model_dump(),
    }


# --- versioned store ---------------------------------------------------------


@mcp.tool()
def list_songs() -> dict:
    """List song ids present in the git-backed store."""
    return {"songs": get_store().list_songs()}


@mcp.tool()
def get_song(song_id: str, version: Optional[str] = None) -> dict:
    """Fetch a song's JSON from the store — latest, or a specific committed
    version sha."""
    return get_store().get(song_id, version=version).model_dump()


@mcp.tool()
def list_song_versions(song_id: str) -> dict:
    """List the committed versions (sha, timestamp, message) of a song,
    newest first."""
    return {
        "songId": song_id,
        "versions": [dataclasses.asdict(v) for v in get_store().versions(song_id)],
    }


@mcp.tool()
def diff_song_versions(song_id: str, version_a: str, version_b: str) -> str:
    """Unified git diff of a song between two committed versions."""
    return get_store().diff(song_id, version_a, version_b)


@mcp.tool()
def save_song(song_json: str, message: str = "Manual save", expected_version: Optional[str] = None) -> dict:
    """Validate and commit a Song JSON as a new version. expected_version
    enables optimistic locking (save-if-version-unchanged): the save is
    rejected if the stored version moved since you read it. Provenance is
    append-only — the new document must extend the stored history."""
    song = Song.model_validate_json(song_json)
    saved = get_store().save(song, message, expected_version=expected_version)
    return dataclasses.asdict(saved)


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

    return {
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


@mcp.tool()
def get_song_schema() -> dict:
    """The Song JSON schema every produced document conforms to (iOS SongStore
    compatible), including the sounding-harmony chord rule."""
    return song_json_schema()


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
