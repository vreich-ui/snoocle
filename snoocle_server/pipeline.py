"""End-to-end pipeline orchestration.

discover -> acquire (yt-dlp) -> MIR -> reconcile (LLM/agent) -> versioned store.

Each stage is independently callable (the HTTP API and MCP tools expose them
separately); this module wires them together for the one-call analyze flow.

Reliability contract (POST /v1/songs/analyze):

- **No silent hangs.** Every external step runs under its own wall-clock
  timeout (:mod:`config` ``*_timeout_seconds``). discover/acquire/mir are
  *best-effort* — a failure or timeout is recorded in ``steps`` and the
  pipeline continues from whatever it has (a song can still come from text
  sources alone, or MIR alone). reconcile/store are *fatal* — a failure or
  timeout raises :class:`PipelineStepError`, which the API turns into a
  ``502 {"detail": "<step>: <msg> [steps: ...]"}`` (the per-step outcomes so
  far) so the client sees exactly what broke — and why upstream.
- **Truthful ``steps``.** Each entry is the real per-step outcome
  (``"ok: ..."`` / ``"skipped"`` / ``"failed: ..."``).
- **Offline mock.** ``provider="mock"`` never touches the network: discovery is
  skipped and the deterministic reconciler synthesizes a small Song, so the
  whole analyze -> persist -> fetch -> versions path runs in CI with no keys.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from fastapi.concurrency import run_in_threadpool

from .audio.acquire import AcquiredAudio, acquire, extract_metadata
from .config import settings
from .discovery import CandidateSource, discover_sources
from .mir import MirAnalysis, analyze_audio
from .reconcile import ReconcileResult, reconcile
from .schema.song import slugify_song_id
from .store import SaveResult, SongRepository, VersionConflictError, get_repository

log = logging.getLogger(__name__)


class PipelineStepError(RuntimeError):
    """A fatal pipeline step failed; carries the step name for a 502 detail,
    plus the per-step outcomes so far so the client can see WHY the fatal step
    had nothing to work with (e.g. reconcile failing only because discover,
    acquire, and mir all came up empty)."""

    def __init__(self, step: str, message: str, steps: dict[str, str] | None = None):
        self.step = step
        self.message = message
        self.steps = dict(steps or {})
        detail = f"{step}: {message}"
        if self.steps:
            summary = "; ".join(f"{k}={_truncate(v)}" for k, v in self.steps.items())
            detail += f" [steps: {summary}]"
        super().__init__(detail)


def _truncate(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class PipelineReport:
    song_id: str
    steps: dict[str, str] = field(default_factory=dict)  # step -> status text
    candidates: list[CandidateSource] = field(default_factory=list)
    audio: AcquiredAudio | None = None
    mir: MirAnalysis | None = None
    reconcile: ReconcileResult | None = None
    stored_version: str | None = None
    stored_timestamp: str | None = None


def get_store() -> SongRepository:
    """The process-wide song repository (Firestore or in-memory per config)."""
    return get_repository()


# --- individual steps (pure, synchronous, blocking) -------------------------


def _step_discover(title: str, artist: str, max_candidates: int | None) -> list[CandidateSource]:
    return discover_sources(title, artist, max_candidates=max_candidates)


def _step_acquire(
    title: str, artist: str, youtube_url_or_id: str | None
) -> AcquiredAudio:
    return acquire(title=title, artist=artist, video_url_or_id=youtube_url_or_id)


def _step_mir(audio_path: str) -> MirAnalysis:
    return analyze_audio(audio_path)


def _step_reconcile(
    title: str,
    artist: str,
    song_id: str,
    candidates: list[CandidateSource],
    mir: MirAnalysis | None,
    provider: str | None,
    model: str | None,
    attach_audio: bool | None,
    audio: AcquiredAudio | None,
) -> ReconcileResult:
    return reconcile(
        title,
        artist,
        candidates,
        mir,
        provider_name=provider,
        model=model,
        audio_path=audio.path if audio else None,
        attach_audio=attach_audio,
        youtube_video_id=audio.video_id if audio else None,
        song_id=song_id,
    )


def _step_store(
    store: SongRepository,
    result: ReconcileResult,
    song_id: str,
    expected_version: str | None,
) -> SaveResult:
    prior = store.current_version(song_id)
    if prior is not None:
        # append-only provenance: extend the stored history with this run's entries
        stored = store.get(song_id)
        merged = list(stored.provenance) + list(result.song.provenance)
        result.song = result.song.model_copy(update={"provenance": merged})
    return store.save(
        result.song,
        message=(
            f"{'Re-analyze' if prior else 'Analyze'} {song_id} "
            f"[{result.provider}/{result.model}]"
        ),
        expected_version=expected_version if expected_version is not None else prior,
    )


# --- async orchestration with per-step timeouts -----------------------------


async def _timed_step(name: str, fn, timeout: float):
    """Run a blocking step in a worker thread under a wall-clock timeout.

    Returns the step result. Raises ``asyncio.TimeoutError`` on timeout or the
    step's own exception on failure — the caller decides fatal vs best-effort.
    Logs start + end (with duration) as structured key=value lines.
    """
    start = time.monotonic()
    log.info("pipeline.step start step=%s timeout=%.0fs", name, timeout)
    try:
        result = await asyncio.wait_for(run_in_threadpool(fn), timeout)
    except asyncio.TimeoutError:
        log.warning("pipeline.step timeout step=%s dur=%.1fs", name, time.monotonic() - start)
        raise
    except Exception as e:  # noqa: BLE001 — logged, then re-raised for the caller
        log.warning(
            "pipeline.step error step=%s dur=%.1fs err=%s",
            name, time.monotonic() - start, e,
        )
        raise
    log.info("pipeline.step ok step=%s dur=%.2fs", name, time.monotonic() - start)
    return result


async def run_pipeline_async(
    title: str | None,
    artist: str | None,
    youtube_url_or_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    attach_audio: bool | None = None,
    skip_audio: bool = False,
    max_candidates: int | None = None,
    expected_version: str | None = None,
    store: SongRepository | None = None,
) -> PipelineReport:
    resolved_provider = (provider or settings.llm_provider).lower()
    steps: dict[str, str] = {}

    # 0. resolve identity: title+artist may be omitted when a media URL is
    # given — derive them from the media's own metadata (no download). FATAL:
    # without an identity there is nothing to analyze or store.
    if not (title and artist):
        if not youtube_url_or_id:
            raise PipelineStepError(
                "resolve", "provide title and artist, or a youtubeUrlOrId to derive them from"
            )
        try:
            meta = await _timed_step(
                "resolve",
                lambda: extract_metadata(youtube_url_or_id),
                settings.acquire_timeout_seconds,
            )
        except asyncio.TimeoutError as e:
            raise PipelineStepError(
                "resolve", f"timed out after {settings.acquire_timeout_seconds:.0f}s"
            ) from e
        except Exception as e:  # noqa: BLE001
            raise PipelineStepError("resolve", str(e)) from e
        title = title or meta.title
        artist = artist or meta.artist
        youtube_url_or_id = youtube_url_or_id or meta.video_id
        steps["resolve"] = f"ok: title={title!r} artist={artist!r} (from {meta.video_id})"

    song_id = slugify_song_id(artist, title)
    report = PipelineReport(song_id=song_id, steps=steps)

    # 1-3. text-source discovery (best-effort). Skipped entirely for the mock
    # provider, which is the fully-offline deterministic path (no network).
    if resolved_provider == "mock":
        report.steps["discover"] = "skipped (mock: offline deterministic reconciler)"
    else:
        try:
            report.candidates = await _timed_step(
                "discover",
                lambda: _step_discover(title, artist, max_candidates),
                settings.discover_timeout_seconds,
            )
            report.steps["discover"] = f"ok: {len(report.candidates)} candidate source(s)"
        except Exception as e:  # noqa: BLE001 — best-effort (incl. timeout)
            report.steps["discover"] = _fail_text(e, settings.discover_timeout_seconds)

    # 4. audio acquisition + MIR analysis (both best-effort)
    if skip_audio:
        report.steps["acquire"] = "skipped"
        report.steps["mir"] = "skipped"
    else:
        try:
            report.audio = await _timed_step(
                "acquire",
                lambda: _step_acquire(title, artist, youtube_url_or_id),
                settings.acquire_timeout_seconds,
            )
            report.steps["acquire"] = f"ok: {report.audio.video_id} ({report.audio.video_title})"
        except Exception as e:  # noqa: BLE001 — best-effort (incl. timeout)
            report.steps["acquire"] = _fail_text(e, settings.acquire_timeout_seconds)
        if report.audio is not None:
            audio_path = report.audio.path
            try:
                report.mir = await _timed_step(
                    "mir", lambda: _step_mir(audio_path), settings.mir_timeout_seconds
                )
                report.steps["mir"] = "ok: engines=" + str(report.mir.engines)
            except Exception as e:  # noqa: BLE001 — best-effort (incl. timeout)
                report.steps["mir"] = _fail_text(e, settings.mir_timeout_seconds)
        else:
            report.steps["mir"] = "skipped (no audio)"

    # 5. reconciliation (FATAL) — uses ALL candidates + the MIR timeline
    try:
        report.reconcile = await _timed_step(
            "reconcile",
            lambda: _step_reconcile(
                title, artist, song_id, report.candidates, report.mir,
                provider, model, attach_audio, report.audio,
            ),
            settings.reconcile_timeout_seconds,
        )
    except asyncio.TimeoutError as e:
        raise PipelineStepError(
            "reconcile",
            f"timed out after {settings.reconcile_timeout_seconds:.0f}s",
            steps=report.steps,
        ) from e
    except Exception as e:  # noqa: BLE001 — ReconcileError/ProviderError/anything else
        raise PipelineStepError("reconcile", str(e), steps=report.steps) from e
    result = report.reconcile
    report.steps["reconcile"] = (
        f"ok: provider={result.provider} model={result.model} attempts={result.attempts}"
    )

    # 6-7. version-controlled persistence (FATAL, except a 409 conflict which
    # the API surfaces as-is). Every run is a new immutable version.
    store = store or get_store()
    try:
        saved = await _timed_step(
            "store",
            lambda: _step_store(store, result, song_id, expected_version),
            settings.store_timeout_seconds,
        )
    except VersionConflictError:
        raise  # -> HTTP 409, not a 502
    except asyncio.TimeoutError as e:
        raise PipelineStepError(
            "store",
            f"timed out after {settings.store_timeout_seconds:.0f}s",
            steps=report.steps,
        ) from e
    except Exception as e:  # noqa: BLE001
        raise PipelineStepError("store", str(e), steps=report.steps) from e
    report.stored_version = saved.version
    report.stored_timestamp = saved.timestamp
    report.steps["store"] = f"ok: version {saved.version}"
    return report


def _fail_text(exc: BaseException, timeout: float) -> str:
    if isinstance(exc, asyncio.TimeoutError):
        return f"failed: timed out after {timeout:.0f}s"
    return f"failed: {exc}"


def run_pipeline(
    title: str | None,
    artist: str | None,
    youtube_url_or_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    attach_audio: bool | None = None,
    skip_audio: bool = False,
    max_candidates: int | None = None,
    expected_version: str | None = None,
    store: SongRepository | None = None,
) -> PipelineReport:
    """Synchronous wrapper around :func:`run_pipeline_async` for callers that
    are not already inside an event loop (e.g. simple scripts). The API and MCP
    tool await the async form directly."""
    return asyncio.run(
        run_pipeline_async(
            title,
            artist,
            youtube_url_or_id=youtube_url_or_id,
            provider=provider,
            model=model,
            attach_audio=attach_audio,
            skip_audio=skip_audio,
            max_candidates=max_candidates,
            expected_version=expected_version,
            store=store,
        )
    )
