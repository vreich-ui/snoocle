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

from .audio.acquire import AcquiredAudio, YouTubeAuthError, acquire, extract_metadata
from .config import settings
from .discovery import CandidateSource, discover_sources
from .mir import MirAnalysis, analyze_audio
from .reconcile import ReconcileResult, provider_preflight, reconcile
from .reconcile.depth import resolve_depth
from .reconcile.trace import TraceRecorder, start_run
from .schema.song import slugify_song_id
from .store import SaveResult, SongRepository, VersionConflictError, get_repository
from .store.runs import get_run_store

log = logging.getLogger(__name__)


class PipelineStepError(RuntimeError):
    """A fatal pipeline step failed; carries the step name for a 502 detail,
    plus the per-step outcomes so far so the client can see WHY the fatal step
    had nothing to work with (e.g. reconcile failing only because discover,
    acquire, and mir all came up empty)."""

    def __init__(
        self,
        step: str,
        message: str,
        steps: dict[str, str] | None = None,
        error_code: str | None = None,
    ):
        self.step = step
        self.message = message
        self.steps = dict(steps or {})
        # Machine-readable classification for clients that offer a fix action
        # (e.g. "youtube_auth_required" -> the app's Reconnect YouTube flow).
        self.error_code = error_code
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
    error_code: str | None = None  # machine-readable cause from a failed step
    run_id: str | None = None  # id of this run's persisted step trace


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


def _step_mir(audio_path: str, accuracy: str) -> MirAnalysis:
    return analyze_audio(audio_path, accuracy=accuracy)


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
    trace: TraceRecorder | None = None,
    guidance: str | None = None,
    prior_song: dict | None = None,
    depth=None,
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
        trace=trace,
        guidance=guidance,
        prior_song=prior_song,
        depth=depth,
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
    accuracy: str | None = None,
    analysis_depth: str | None = None,
    guidance: str | None = None,
    prior_song: dict | None = None,
) -> PipelineReport:
    resolved_provider = (provider or settings.llm_provider).lower()
    # analysisDepth is the canonical control; the older `accuracy` field is
    # honored as its source when a depth isn't given explicitly. The chosen
    # profile drives MIR accuracy, agent effort, the tool budget, and syncMap.
    depth = resolve_depth(analysis_depth or accuracy)
    accuracy = depth.accuracy
    steps: dict[str, str] = {}

    # Provider preflight (FATAL, instant). A provider that can't serve ANY
    # request — unknown name or missing credential/endpoint — must fail here,
    # not minutes later at reconcile after discover/acquire/MIR have all been
    # paid for (clients retry 502s, so late failure multiplies into a loop of
    # full-price doomed runs).
    problem = provider_preflight(resolved_provider)
    if problem:
        raise PipelineStepError("reconcile", problem, error_code="provider_not_configured")

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
            code = "youtube_auth_required" if isinstance(e, YouTubeAuthError) else None
            raise PipelineStepError("resolve", str(e), error_code=code) from e
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
            if isinstance(e, YouTubeAuthError):
                report.error_code = "youtube_auth_required"
        if report.audio is not None:
            audio_path = report.audio.path
            try:
                report.mir = await _timed_step(
                    "mir", lambda: _step_mir(audio_path, accuracy), settings.mir_timeout_seconds
                )
                report.steps["mir"] = "ok: engines=" + str(report.mir.engines)
            except Exception as e:  # noqa: BLE001 — best-effort (incl. timeout)
                report.steps["mir"] = _fail_text(e, settings.mir_timeout_seconds)
        else:
            report.steps["mir"] = "skipped (no audio)"

    # 5. reconciliation (FATAL) — uses ALL candidates + the MIR timeline. The
    # run's step trace is recorded live and persisted for later replay in the
    # GUI (the agent's logic, tool calls, and repair rounds).
    recorder = start_run(song_id, resolved_provider, depth.name)
    report.run_id = recorder.trace.run_id
    try:
        report.reconcile = await _timed_step(
            "reconcile",
            lambda: _step_reconcile(
                title, artist, song_id, report.candidates, report.mir,
                provider, model, attach_audio, report.audio,
                trace=recorder, guidance=guidance, prior_song=prior_song, depth=depth,
            ),
            settings.reconcile_timeout_seconds,
        )
    except asyncio.TimeoutError as e:
        recorder.finish("error", error=f"timed out after {settings.reconcile_timeout_seconds:.0f}s")
        _persist_trace(recorder)
        raise PipelineStepError(
            "reconcile",
            f"timed out after {settings.reconcile_timeout_seconds:.0f}s",
            steps=report.steps,
        ) from e
    except Exception as e:  # noqa: BLE001 — ReconcileError/ProviderError/anything else
        recorder.finish("error", error=str(e)[:2000])
        _persist_trace(recorder)
        raise PipelineStepError(
            "reconcile", str(e), steps=report.steps, error_code=report.error_code
        ) from e
    result = report.reconcile
    recorder.finish("ok", model=result.model)
    _persist_trace(recorder)
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


def _persist_trace(recorder: TraceRecorder) -> None:
    """Durably store a run's trace (best-effort — never fail the pipeline over
    an observability write)."""
    try:
        get_run_store().save_run(recorder.trace.to_dict())
    except Exception as e:  # noqa: BLE001
        log.warning("run trace persistence failed (continuing): %s", e)


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
    accuracy: str | None = None,
    analysis_depth: str | None = None,
    guidance: str | None = None,
    prior_song: dict | None = None,
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
            accuracy=accuracy,
            analysis_depth=analysis_depth,
            guidance=guidance,
            prior_song=prior_song,
        )
    )
