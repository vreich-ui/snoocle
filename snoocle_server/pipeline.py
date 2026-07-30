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
- **Scope.** An optional :class:`~.scope.AnalysisScope` turns the two
  evidence-gathering stages off independently (``listen`` -> acquire+MIR,
  ``reconcile`` -> discover). Reconciliation itself always runs. An ABSENT
  scope is not a scope: the pipeline behaves exactly as it did before the
  field existed.
- **Timing is never lost silently.** ``listen=off`` means "reuse the existing
  audio analysis", so the run MUST have one: the prior version is resolved
  before any expensive step and a run with none fails immediately (step
  ``timing``) rather than committing a timing-less document. With one, a
  carry-forward pass (5b) replaces ``snap_chords`` and copies the prior
  version's timing onto the reconciled document. Independently of both, a
  guard refuses to store a version that empties ``audio.beats`` or nulls
  ``metadata.bpm`` when the prior had them — whatever path got there.
  ``allow_timing_loss`` is the explicit opt-out for the rare intentional case.
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

from . import __version__
from .audio.acquire import (
    AcquiredAudio,
    ResolvedMeta,
    YouTubeAuthError,
    acquire,
    extract_metadata,
)
from .config import settings
from .correction_routing import classify_correction
from .discovery import CandidateSource, discover_sources
from .discovery.cache import DiscoveryCacheInfo, discover_cached
from .identity import IdentityError, SongIdentity, resolve_identity
from .manifest import build_evidence_manifest, lrc_block
from .mir import MirAnalysis, analyze_audio
from .mir.cache import MirCacheInfo, analyze_cached
from .quality import Grade, QualityDecision, evaluate as evaluate_quality
from .quality.escalation import build_retry_feedback, search_found_better
from .quality.grader import grade_provenance_entry, timing_unreliable_provenance_entry
from .recordings import RecordingSuggestions, suggest_recordings
from .reconcile import ReconcileResult, provider_preflight, reconcile
from .reconcile.match import score_candidates
from .reconcile.depth import resolve_depth
from .reconcile.trace import TraceRecorder, start_run
from .schema.song import ProvenanceEntry, Song, slugify_song_id
from .scope import AnalysisScope
from .store import SaveResult, SongRepository, VersionConflictError, get_repository
from .store.runs import get_run_store
from .timing.carry_forward import audio_data_lost, carry_forward_timing
from .timing.collapse_guard import guard_against_collapsed_timing
from .timing.confidence import build_review_queue, score_song
from .timing.lrc import apply_lrc, fetch_lrc, match_lrc_to_lines
from .timing.snap import snap_chords

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


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cache_suffix(status: str) -> str:
    """Append the cache outcome to a step's text — and ONLY when it is
    something other than a plain recomputation, so a run with a cold cache
    reports exactly the text it always did."""
    return f" (cache {status})" if status in ("hit", "refresh") else ""


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
    # What this run reused vs recomputed, and how good the evidence was. Purely
    # descriptive — see manifest.py. Surfaced in the analyze response and on the
    # run trace so the admin UI can show it; nothing branches on it.
    evidence_manifest: dict = field(default_factory=dict)
    mir_cache: MirCacheInfo | None = None
    discovery_cache: DiscoveryCacheInfo | None = None
    # Alternative recordings found because THIS one graded as an audio fault
    # (see recordings.py). Reported, never acted on: analyzing one is an
    # explicit operator action, because it is a full second analysis.
    recording_suggestions: RecordingSuggestions | None = None


def get_store() -> SongRepository:
    """The process-wide song repository (Firestore or in-memory per config)."""
    return get_repository()


# --- individual steps (pure, synchronous, blocking) -------------------------


def _step_resolve(youtube_url_or_id: str) -> tuple[ResolvedMeta, SongIdentity]:
    """Fetch the media's metadata and resolve a song identity from it.

    One step, not two: the metadata fetch and the (occasional) disambiguation
    call are both network work on the same question, and the caller only cares
    whether an identity came out the other end.
    """
    meta = extract_metadata(youtube_url_or_id)
    identity = resolve_identity(
        video_title=meta.video_title,
        channel=meta.uploader,
        track=meta.track,
        track_artist=meta.track_artist,
    )
    return meta, identity


def _step_discover(
    title: str, artist: str, max_candidates: int | None, refresh: bool = False
) -> tuple[list[CandidateSource], DiscoveryCacheInfo]:
    # `discover_sources` stays the thing that searches — the cache wraps the
    # call rather than replacing the name, so every existing seam (and every
    # test that monkeypatches this module's `discover_sources`) is untouched.
    resolved_max = max_candidates or settings.search_max_candidates
    return discover_cached(
        title, artist,
        max_candidates=resolved_max,
        discover=lambda: discover_sources(title, artist, max_candidates=max_candidates),
        refresh=refresh,
    )


def _step_acquire(
    title: str, artist: str, youtube_url_or_id: str | None
) -> AcquiredAudio:
    return acquire(title=title, artist=artist, video_url_or_id=youtube_url_or_id)


def _step_mir(
    audio_path: str, accuracy: str, refresh: bool = False
) -> tuple[MirAnalysis, MirCacheInfo]:
    # Same shape as _step_discover: `analyze_audio` remains what computes.
    return analyze_cached(
        audio_path,
        accuracy=accuracy,
        compute=lambda: analyze_audio(audio_path, accuracy=accuracy),
        refresh=refresh,
    )


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
    guidance_origin: str | None = None,
    prior_song: dict | None = None,
    depth=None,
    scope: AnalysisScope | None = None,
    evidence_manifest: dict | None = None,
    mir_cache: MirCacheInfo | None = None,
    patch_ops_eligible: bool = False,
    quality_feedback: str | None = None,
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
        guidance_origin=guidance_origin,
        prior_song=prior_song,
        depth=depth,
        scope=scope,
        evidence_manifest=evidence_manifest,
        mir_cache=mir_cache,
        patch_ops_eligible=patch_ops_eligible,
        quality_feedback=quality_feedback,
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
    scope: AnalysisScope | None = None,
    refresh_cache: bool = False,
    allow_timing_loss: bool = False,
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
    # without an identity there is nothing to analyze or store, and the id it
    # mints is permanent (content-hash versioned store), so an ambiguous title
    # fails here rather than becoming a wrong id forever. See `identity`.
    identity: SongIdentity | None = None
    if not (title and artist):
        if not youtube_url_or_id:
            raise PipelineStepError(
                "resolve", "provide title and artist, or a youtubeUrlOrId to derive them from"
            )
        try:
            meta, identity = await _timed_step(
                "resolve",
                lambda: _step_resolve(youtube_url_or_id),
                settings.acquire_timeout_seconds,
            )
        except asyncio.TimeoutError as e:
            raise PipelineStepError(
                "resolve", f"timed out after {settings.acquire_timeout_seconds:.0f}s"
            ) from e
        except IdentityError as e:
            raise PipelineStepError(
                "resolve", str(e), error_code="identity_ambiguous"
            ) from e
        except Exception as e:  # noqa: BLE001
            code = "youtube_auth_required" if isinstance(e, YouTubeAuthError) else None
            raise PipelineStepError("resolve", str(e), error_code=code) from e
        # An explicitly-supplied half always wins over the derived one.
        title = title or identity.title
        artist = artist or identity.artist
        youtube_url_or_id = youtube_url_or_id or meta.video_id
        steps["resolve"] = f"ok: {identity.describe()} (from {meta.video_id})"

    song_id = slugify_song_id(artist, title)
    report = PipelineReport(song_id=song_id, steps=steps)

    # Reconciliation notes replay (contract §2). This has to happen HERE, not in
    # the API layer: when the caller gave only a media URL, the song id is not
    # known until the identity step above has run, and notes keyed by any other
    # id would silently never replay.
    guidance, guidance_origin = _resolve_guidance(song_id, guidance)
    if guidance:
        report.steps["notes"] = f"ok: guidance applied (from {guidance_origin})"

    # The prior version, resolved BEFORE anything expensive runs (including
    # scope inference below, which needs to know one exists). Two further
    # roles: the source this run carries timing forward from (5b) when it
    # isn't listening, and the baseline the pre-store guard compares against.
    store = store or get_store()
    stored_prior, stored_prior_version, store_problem = _load_stored_song(store, song_id)
    request_prior = _decode_prior_song(prior_song)
    # The client's edited document wins as the MATCHING source (it is the
    # document the reconciler was actually asked to correct); the stored
    # version backs it up for the audio-derived fields, which a client may
    # legitimately have never round-tripped.
    timing_prior = request_prior or stored_prior
    timing_prior_version = (
        stored_prior_version
        if timing_prior is stored_prior
        else _content_version(timing_prior)
    )

    # Route the request: a correction naming specific chords, lines, sections
    # or lyric words needs no new sources — everything it refers to is
    # already in the stored document — but without this the caller had to
    # know that and set scope explicitly. A request with none re-discovered
    # 3 sources and asked the model for a complete Song to change one chord,
    # which is what got blocked by a content filter over a single chord.
    #
    # Only ever consulted when the caller has no opinion (`scope is None`) —
    # an explicit scope always wins — and only when there is a PRIOR document
    # to correct: "notes naming a chord" has nothing to apply itself to on a
    # first-time analysis. `patch_eligible` is independent of how notes-only
    # was reached (inferred here, or given explicitly) — see reconcile.engine
    # for where it actually decides patch vs full notes-only reconcile.
    classification = None
    scope_was_inferred = False
    if guidance and (stored_prior is not None or request_prior is not None):
        classification = classify_correction(guidance)
        if scope is None and classification.is_targeted_correction:
            scope = AnalysisScope(listen=False, reconcile=False)
            scope_was_inferred = True
            log.info(
                "pipeline.scope inferred notes-only for %s: %s",
                song_id, classification.describe(),
            )

    # Scope constrains ONLY when it is present (explicit OR just inferred
    # above). `None` means no opinion was ever formed, and gets exactly the
    # historical pipeline — this is why the flags below are read through
    # `scope is None` rather than defaulting the parameter to FULL_SCOPE (the
    # report would then grow a "scope" step for every legacy caller, changing
    # their response shape).
    want_listen = True if scope is None else scope.listen
    want_sources = True if scope is None else scope.reconcile

    if scope is not None:
        report.steps["scope"] = scope.describe() + (
            f" (inferred: {classification.describe()})" if scope_was_inferred else ""
        )

    # A notes-only run needs SOMETHING to correct even when the caller didn't
    # attach one: the request's own priorSong wins when given (it's the exact
    # document the reconciler is being asked to fix), else the stored latest
    # version stands in. Scoped narrowly to notes-only, not every reconcile —
    # a full re-analysis without an explicit priorSong is unchanged.
    effective_prior_song = prior_song
    if scope is not None and scope.notes_only and effective_prior_song is None and stored_prior is not None:
        effective_prior_song = stored_prior.model_dump(mode="json")

    if not want_listen and timing_prior is None and not allow_timing_loss:
        # "Reuse the existing audio analysis" with no existing analysis to
        # reuse. Storing anyway is exactly the silent-loss bug this guard
        # exists for, so fail here — before paying for a reconciliation whose
        # result could not be stored.
        raise PipelineStepError(
            "timing",
            f"scope.listen=false reuses the prior version's audio analysis, but "
            f"{song_id!r} has no prior version to reuse "
            + (
                f"(the stored one could not be read: {store_problem})"
                if store_problem
                else "(none stored, and no priorSong was supplied)"
            )
            + ". Re-run with listen=true to analyze the audio, or set "
            "allowTimingLoss=true to store a document with no timing.",
            steps=report.steps,
            error_code="no_prior_timing_to_carry_forward",
        )

    # 1-3. text-source discovery (best-effort). Skipped entirely for the mock
    # provider, which is the fully-offline deterministic path (no network), and
    # for a run whose scope turned source-gathering off.
    if not want_sources:
        report.steps["discover"] = (
            "skipped (scope: notes only)"
            if not want_listen
            else "skipped (scope: no new sources)"
        )
    elif resolved_provider == "mock":
        report.steps["discover"] = "skipped (mock: offline deterministic reconciler)"
    else:
        try:
            report.candidates, report.discovery_cache = await _timed_step(
                "discover",
                lambda: _step_discover(title, artist, max_candidates, refresh_cache),
                settings.discover_timeout_seconds,
            )
            report.steps["discover"] = (
                f"ok: {len(report.candidates)} candidate source(s)"
                + _cache_suffix(report.discovery_cache.status)
            )
        except Exception as e:  # noqa: BLE001 — best-effort (incl. timeout)
            report.steps["discover"] = _fail_text(e, settings.discover_timeout_seconds)

    # 4. audio acquisition + MIR analysis (both best-effort). `listen=off` is
    # the same skip as the long-standing `skipAudio`, but says WHY in the
    # report — "I turned listening off" and "this caller never wants audio"
    # look identical afterwards otherwise.
    if skip_audio or not want_listen:
        skip_text = "skipped" if skip_audio else "skipped (scope: reusing the existing audio analysis)"
        report.steps["acquire"] = skip_text
        report.steps["mir"] = skip_text
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
                report.mir, report.mir_cache = await _timed_step(
                    "mir",
                    lambda: _step_mir(audio_path, accuracy, refresh_cache),
                    settings.mir_timeout_seconds,
                )
                report.steps["mir"] = (
                    "ok: engines=" + str(report.mir.engines)
                    + _cache_suffix(report.mir_cache.status)
                )
            except Exception as e:  # noqa: BLE001 — best-effort (incl. timeout)
                report.steps["mir"] = _fail_text(e, settings.mir_timeout_seconds)
        else:
            report.steps["mir"] = "skipped (no audio)"

    # 5. reconciliation (FATAL) — uses ALL candidates + the MIR timeline. The
    # run's step trace is recorded live and persisted for later replay in the
    # GUI (the agent's logic, tool calls, and repair rounds).
    #
    # The evidence manifest is assembled here, once every input's state is
    # known, and handed to the reconciler alongside the evidence itself: an
    # agent that can see "this MIR is a cache hit from three weeks ago and
    # these 3 sources are current" behaves differently from one left to guess
    # whether gathering failed. Purely descriptive — see manifest.py.
    report.evidence_manifest = build_evidence_manifest(
        mir=report.mir,
        mir_cache=report.mir_cache,
        candidates=report.candidates,
        discovery_cache=report.discovery_cache,
        prior_song=effective_prior_song,
        scope=scope,
        guidance=guidance,
        guidance_origin=guidance_origin,
    )
    recorder = start_run(song_id, resolved_provider, depth.name)
    report.run_id = recorder.trace.run_id

    # What the quality gate (5f) has already spent. Threaded into
    # `plan_escalation`, which is what makes the one-retry ceiling structural
    # rather than a promise: with these at their real values a second
    # escalation cannot be planned. See quality/escalation.py.
    quality_retries_spent = 0
    quality_searches_spent = 0

    # Steps 5-5d run as ONE unit because the quality gate (5f) may run them a
    # SECOND time: a reconciliation and every deterministic pass that reads
    # what it produced belong together. A retry that re-reconciled without
    # re-snapping, re-guarding and re-scoring would be graded on a document
    # nothing had timed — not the document a store would ever receive.
    async def _reconcile_and_time(*, quality_feedback: str | None = None) -> bool:
        """One reconciliation plus every deterministic pass that reads its output.

        Returns True when the document came from the patch path (nothing was
        regenerated, so 5b/5c/5c2/5d all skip by their own contracts). Mutates
        `report` exactly as the inline code it replaced did; FATAL failures
        raise :class:`PipelineStepError`.
        """
        nonlocal quality_retries_spent
        if quality_feedback is not None:
            quality_retries_spent += 1

        try:
            report.reconcile = await _timed_step(
                "reconcile",
                lambda: _step_reconcile(
                    title, artist, song_id, report.candidates, report.mir,
                    provider, model, attach_audio, report.audio,
                    trace=recorder, guidance=guidance, guidance_origin=guidance_origin,
                    prior_song=effective_prior_song, depth=depth, scope=scope,
                    evidence_manifest=report.evidence_manifest,
                    mir_cache=report.mir_cache,
                    patch_ops_eligible=(
                        classification.patch_eligible
                        if classification is not None and scope is not None and scope.notes_only
                        else False
                    ),
                    quality_feedback=quality_feedback,
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
            # A classified provider error (e.g. content_filtered) carries its own
            # code; otherwise fall back to any code an upstream step recorded.
            code = getattr(e, "error_code", None) or report.error_code
            raise PipelineStepError(
                "reconcile", str(e), steps=report.steps, error_code=code
            ) from e
        result = report.reconcile
        recorder.finish("ok", model=result.model)
        report.steps["reconcile"] = (
            f"ok: provider={result.provider} model={result.model} attempts={result.attempts}"
        )

        # 5a. cover attribution. `artist` is who is PLAYING on this recording, so
        # for a cover the artist who originally released the song would otherwise
        # be lost entirely. Record it as a provenance note rather than bending
        # metadata.album, which means the album this recording appears on.
        if identity is not None and identity.is_cover and identity.original_artist:
            result.song = result.song.model_copy(
                update={
                    "provenance": list(result.song.provenance)
                    + [
                        ProvenanceEntry(
                            timestamp=_now_iso(),
                            actor=f"snoocle-server/{__version__}",
                            action="cover-attribution",
                            confidence=round(identity.confidence, 3),
                            notes=(
                                f"cover: performed by {identity.artist!r}, "
                                f"originally by {identity.original_artist!r} "
                                f"(identified from the upload's own metadata via {identity.method})"
                            ),
                        )
                    ]
                }
            )

        # A patch (reconcile/patch_ops.py) names exactly what it touches and
        # copies everything else through untouched, including every timing
        # field — there is nothing here to carry forward, snap, or re-align, and
        # running any of 5b/5c/5d would mean re-deriving values a patch
        # deliberately left alone. This is what makes the timing-loss class of
        # bug impossible on this path rather than merely guarded against: none
        # of the guarding passes are consulted, because nothing was regenerated.
        patched = result.patch_ops_applied > 0

        # 5b. deterministic chord/line timing. Three mutually exclusive passes:
        #
        #   patched -> skip entirely, see above.
        #   listen=off -> carry the prior version's timing forward (FATAL on
        #     failure). This run has no MIR by construction, so snap_chords would
        #     be a no-op by its own contract and the reconciler's freshly-emitted
        #     document — which correctly leaves timing empty for a post-pass to
        #     fill — would store with its timing silently gone.
        #   otherwise  -> MIR-grounded snapping, exactly as before (best-effort —
        #     a failure here must never block storing an otherwise-good song; the
        #     pipeline continues with whatever timing the reconciler produced,
        #     which today is none). A no-op when MIR didn't run.
        carried_forward = False
        if patched:
            report.steps["timing"] = (
                f"skipped (patch: {result.patch_ops_applied} op(s) preserved the "
                f"prior version's timing)"
            )
        elif not want_listen and timing_prior is not None:
            try:
                result.song, carry_stats = carry_forward_timing(
                    result.song,
                    timing_prior,
                    audio_fallback=stored_prior,
                    prior_version=timing_prior_version,
                )
            except Exception as e:  # noqa: BLE001 — fatal: see the block comment
                log.warning("pipeline.step error step=timing err=%s", e)
                raise PipelineStepError(
                    "timing",
                    f"could not carry the prior version's timing forward: {e}",
                    steps=report.steps,
                    error_code="timing_carry_forward_failed",
                ) from e
            carried_forward = True
            report.steps["timing"] = f"ok: {carry_stats.describe()}"
        else:
            try:
                timed_song = snap_chords(result.song, report.mir)
                if report.mir is not None and report.audio is not None:
                    timed_song = timed_song.model_copy(
                        update={
                            "audio": timed_song.audio.model_copy(
                                update={"analyzedVideoId": report.audio.video_id}
                            )
                        }
                    )
                result.song = timed_song
                report.steps["timing"] = "ok" if report.mir is not None else "skipped (no MIR)"
            except Exception as e:  # noqa: BLE001 — best-effort, never fatal
                report.steps["timing"] = f"failed: {e}"
                log.warning("pipeline.step error step=timing err=%s", e)

        # 5c. LRCLIB synced-lyrics overlay (best-effort, network-dependent). LRC
        # is better LINE-timing evidence than MIR chord matching, so when it
        # matches it WINS — this must run BEFORE confidence scoring (5d), which
        # reads whatever timeSeconds ends up final, not 5b's provisional guess.
        # Skipped for provider=mock like discover — mock is the fully-offline
        # deterministic path and must make ZERO external calls of any kind.
        if patched:
            report.steps["lrc"] = "skipped (patch: nothing was regenerated to re-align)"
            report.evidence_manifest["lrcAlign"] = lrc_block("skipped")
        elif carried_forward:
            # apply_lrc REPLACES line times and then re-derives every chord time
            # by proportional redistribution — which is exactly the timing 5b just
            # restored from the prior version, only worse (with no MIR there is no
            # beat grid to snap back onto). "Reuse the existing audio analysis"
            # means reuse it, not re-derive it from a different source.
            report.steps["lrc"] = "skipped (scope: timing carried forward from the prior version)"
            report.evidence_manifest["lrcAlign"] = lrc_block("skipped")
        elif resolved_provider == "mock":
            report.steps["lrc"] = "skipped (mock: offline deterministic reconciler)"
            report.evidence_manifest["lrcAlign"] = lrc_block("skipped")
        else:
            try:
                duration = (
                    (report.mir.duration_seconds if report.mir else None)
                    or (report.audio.duration_seconds if report.audio else None)
                )
                lrc = fetch_lrc(title, artist, duration)
                if lrc:
                    matches = match_lrc_to_lines(lrc, result.song)
                    if matches:
                        result.song = apply_lrc(result.song, report.mir, matches)
                        report.steps["lrc"] = f"ok: {len(matches)}/{len(result.song.lines)} line(s) matched"
                        report.evidence_manifest["lrcAlign"] = lrc_block(
                            "hit", len(matches), len(result.song.lines)
                        )
                    else:
                        report.steps["lrc"] = "ok: no line matched closely enough"
                        report.evidence_manifest["lrcAlign"] = lrc_block(
                            "miss", 0, len(result.song.lines)
                        )
                else:
                    report.steps["lrc"] = (
                        "skipped (no LRCLIB match)" if settings.lrclib_enabled else "skipped (disabled)"
                    )
                    report.evidence_manifest["lrcAlign"] = lrc_block("skipped")
            except Exception as e:  # noqa: BLE001 — best-effort, never fatal
                report.steps["lrc"] = f"failed: {e}"
                report.evidence_manifest["lrcAlign"] = lrc_block("failed")
                log.warning("pipeline.step error step=lrc err=%s", e)

        # 5c2. timing collapse guard (best-effort). A generic safety net,
        # independent of what upstream produced it: whatever path just set
        # timeSeconds (snap, carry-forward, then possibly LRC), a run of 3+
        # consecutive lines or in-line placements sharing one identical
        # timestamp gets spread across the span to the next distinct time (on
        # the beat grid when there is one); a collapse with nothing later to
        # spread toward is left alone and reported as such. Runs BEFORE 5d so
        # confidence scoring judges the corrected times, not the collapsed
        # ones. Skipped for a patch, same as 5b/5c: nothing was regenerated.
        any_line_timed = any(line.timeSeconds is not None for line in result.song.lines)
        if patched:
            report.steps["timing-collapse-guard"] = "skipped (patch: nothing was regenerated)"
        elif not any_line_timed:
            # Nothing was ever timed (no MIR, no carry-forward, no LRC match) --
            # there is no collapse to guard against and no coverage worth
            # reporting, so stay silent rather than add a provenance entry to a
            # song this pipeline never attempted to time.
            report.steps["timing-collapse-guard"] = "skipped (no line timings to guard)"
        else:
            try:
                duration = (
                    (report.mir.duration_seconds if report.mir else None)
                    or (report.audio.duration_seconds if report.audio else None)
                    or result.song.audio.durationSeconds
                )
                guarded_song, guard_entry = guard_against_collapsed_timing(result.song, duration)
                result.song = guarded_song
                report.steps["timing-collapse-guard"] = f"ok: {guard_entry.notes}"
            except Exception as e:  # noqa: BLE001 — best-effort, never fatal
                report.steps["timing-collapse-guard"] = f"failed: {e}"
                log.warning("pipeline.step error step=timing-collapse-guard err=%s", e)

        # 5d. per-chord agreement scoring (best-effort). Must run LAST of this
        # group: the MIR-agreement signal reads placement.timeSeconds, which 5b
        # (or 5c, if LRC overrode it, or 5c2, if the collapse guard touched it)
        # is what populates. Refines confidence when both a source and MIR
        # signal exist, and records a compact "check these first" queue on the
        # run trace. Skipped for a patch: with no candidates and no MIR it would
        # only ever touch placements with NO confidence yet (a freshly inserted
        # chord) by inventing the neutral default 0.5 for them — exactly the
        # kind of un-named change this path exists to refuse.
        if patched:
            report.steps["confidence"] = "skipped (patch: nothing to re-score)"
        else:
            try:
                scored_song, scores = score_song(result.song, report.candidates, report.mir)
                result.song = scored_song
                recorder.set_review_queue(build_review_queue(scores))
                report.steps["confidence"] = f"ok: {len(scores)} placement(s) scored"
            except Exception as e:  # noqa: BLE001 — best-effort, never fatal
                report.steps["confidence"] = f"failed: {e}"
                log.warning("pipeline.step error step=confidence err=%s", e)

        return patched

    patched = await _reconcile_and_time()
    result = report.reconcile

    # 5f. the quality gate (snoocle_server/quality/). Three separate things,
    # and the order matters:
    #
    #   grade      -> deterministic, recorded in provenance on EVERY run
    #                 whatever it says. A song's grade history is the thing
    #                 that would have made ten non-converging Marley runs
    #                 visible on the second one instead of the tenth.
    #   attribute  -> MODEL vs AUDIO vs SOURCE, by comparing the document, the
    #                 candidate sheets and the MIR timeline against each other.
    #   escalate   -> at most ONE retry, and only when a retry can plausibly
    #                 change the outcome. A run whose evidence was bad pays
    #                 full price for the same result.
    #
    # Grading a patched document is fine (and its grade is still recorded);
    # escalating one is not — nothing was regenerated, so there is nothing for
    # a second attempt to do differently.
    if settings.quality_enabled:
        try:
            decision = _grade_document(
                report,
                can_search=want_sources,
                can_retry=settings.quality_retry_enabled and not patched,
                retries_spent=quality_retries_spent,
                searches_spent=quality_searches_spent,
            )
            grade, attribution, escalation = (
                decision.grade, decision.attribution, decision.escalation
            )
            report.steps["quality"] = f"{grade.describe()} | fault: {attribution.describe()}"
            recorder.step(
                "quality", "quality-grade",
                f"grade {grade.verdict}: {escalation.describe()}",
                detail={"grade": grade.to_dict(), "attribution": attribution.to_dict()},
            )

            retry_feedback: str | None = escalation.feedback if escalation.retry else None

            # SOURCE fault: the sheets this run had contradict each other, so
            # one targeted search (cache bypassed on purpose — those same
            # cached sheets are what is being escalated away from) looks for
            # better evidence. It only earns the single retry if it actually
            # found some; otherwise the document is stored with its grade.
            if escalation.search:
                quality_searches_spent += 1
                retry_feedback = await _quality_targeted_search(
                    report, title, artist, max_candidates, grade, attribution
                )

            if retry_feedback is not None:
                try:
                    patched = await _reconcile_and_time(quality_feedback=retry_feedback)
                    result = report.reconcile
                    # Re-grade what the retry produced. `retries_spent` is now
                    # 1, so plan_escalation cannot plan another retry — that is
                    # the ceiling, enforced by the same function that granted
                    # the first one.
                    decision = _grade_document(
                        report,
                        can_search=want_sources,
                        can_retry=settings.quality_retry_enabled and not patched,
                        retries_spent=quality_retries_spent,
                        searches_spent=quality_searches_spent,
                    )
                    grade, attribution, escalation = (
                        decision.grade, decision.attribution, decision.escalation
                    )
                    report.steps["quality"] = (
                        f"{grade.describe()} | fault: {attribution.describe()} "
                        f"(after 1 retry)"
                    )
                    recorder.step(
                        "quality", "quality-regrade",
                        f"after one retry: grade {grade.verdict}; {escalation.describe()}",
                        detail={"grade": grade.to_dict(), "attribution": attribution.to_dict()},
                    )
                except PipelineStepError as e:
                    # The retry failed. The first attempt's document is intact
                    # and gradeable, and losing a storable song to a failed
                    # OPTIONAL retry would be strictly worse than storing the
                    # graded original.
                    log.warning("pipeline.step error step=quality-retry err=%s", e)
                    report.steps["quality-retry"] = f"failed: {e.message}"
                    recorder.step(
                        "quality", "quality-retry-failed",
                        f"the quality retry failed; storing the first attempt: {e.message}",
                    )
                    recorder.finish("ok", model=result.model)

            entries = [grade_provenance_entry(grade, attribution=attribution)]
            if escalation.reanalyze_full_accuracy:
                # Reported, never acted on — same as the AUDIO-fault
                # alternative-recording suggestion below: a full-accuracy
                # re-analysis is a real second MIR pass, and paying for it is
                # an explicit operator decision, not something this run makes
                # for itself.
                report.steps["accuracy-escalation"] = (
                    "recommended: re-analyze at full accuracy — " + escalation.reason
                )
            if escalation.mark_timing_unreliable:
                entries.append(timing_unreliable_provenance_entry(attribution))
                report.steps["timing-reliability"] = "marked unreliable (audio fault)"
                # This recording cannot be timed reliably, so the fix is a
                # different recording (Mode B, realign.py). Look for one and
                # REPORT it: one search, no download, no analysis. Skipped for
                # the mock provider, which is the fully-offline path and must
                # make zero external calls of any kind.
                if settings.quality_suggest_recordings and resolved_provider != "mock":
                    report.recording_suggestions = await run_in_threadpool(
                        lambda: suggest_recordings(
                            result.song,
                            reason=(
                                f"{song_id!r} graded as an audio fault on its current "
                                f"recording: {attribution.reason}"
                            ),
                        )
                    )
                    report.steps["recording-suggestions"] = (
                        report.recording_suggestions.describe()
                    )
            result.song = result.song.model_copy(
                update={"provenance": list(result.song.provenance) + entries}
            )
            recorder.set_quality(
                {
                    "grade": grade.to_dict(),
                    "attribution": attribution.to_dict(),
                    "escalation": escalation.to_dict(),
                    "retriesSpent": quality_retries_spent,
                    "searchesSpent": quality_searches_spent,
                }
            )
        except Exception as e:  # noqa: BLE001 — grading must never fail a run
            report.steps["quality"] = f"failed: {e}"
            log.warning("pipeline.step error step=quality err=%s", e)
    else:
        report.steps["quality"] = "skipped (quality grading disabled)"

    # Record the manifest on the trace only NOW: 5c is what resolves its
    # lrcAlign block from "pending" to what actually happened, so attaching it
    # before this point would durably store a half-answered manifest.
    recorder.set_evidence_manifest(report.evidence_manifest)

    # Persist the trace now that timing/lrc/confidence (5b/5c/5d) have had
    # their chance to attach a review queue -- persisting right after
    # `finish()` (as the fatal-path branches above still correctly do, since
    # those never reach this group) would durably store a run missing it.
    _persist_trace(recorder)

    # 5e. audio-data guard (FATAL), independent of everything above. Losing
    # MIR-derived data must be impossible to do ACCIDENTALLY: whatever path
    # produced this document — carry-forward, snap, a future pass nobody has
    # written yet — it does not get to empty audio.beats or null metadata.bpm
    # behind a version that had them. Explicit intent has a flag.
    guard_baseline = stored_prior or request_prior
    if guard_baseline is not None:
        lost = audio_data_lost(guard_baseline, result.song)
        if lost and not allow_timing_loss:
            raise PipelineStepError(
                "timing-guard",
                f"refusing to store {song_id}: this run drops audio-derived data "
                f"the prior version had — {', '.join(lost)}. Re-run with "
                f"listen=true to re-measure it, or set allowTimingLoss=true if "
                f"dropping it is intended.",
                steps=report.steps,
                error_code="timing_data_loss",
            )
        if lost:
            report.steps["timing-guard"] = f"overridden: dropping {', '.join(lost)}"
        elif guard_baseline.audio.beats or guard_baseline.metadata.bpm is not None:
            report.steps["timing-guard"] = "ok: audio.beats and metadata.bpm preserved"

    # 6-7. version-controlled persistence (FATAL, except a 409 conflict which
    # the API surfaces as-is). Every run is a new immutable version.
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


def _grade_document(
    report: PipelineReport,
    *,
    can_search: bool,
    can_retry: bool,
    retries_spent: int,
    searches_spent: int,
) -> QualityDecision:
    """This run's document through the shared quality gate (``quality.gate``).

    Shared rather than local so Mode B (``realign.py``) grades identically —
    two callers measuring the same document differently would make their grade
    histories incomparable for no reason anyone would notice.
    """
    return evaluate_quality(
        report.reconcile.song,
        report.mir,
        report.candidates,
        can_search=can_search,
        can_retry=can_retry,
        retries_spent=retries_spent,
        searches_spent=searches_spent,
    )


async def _quality_targeted_search(
    report: PipelineReport,
    title: str,
    artist: str,
    max_candidates: int | None,
    grade: Grade,
    attribution: Attribution,
) -> str | None:
    """The SOURCE-fault escalation: one search for better evidence.

    Returns the retry feedback when the new sheets agree with the audio
    materially better than the ones this run already had (in which case
    `report.candidates` has been replaced with them), else None — a search
    that found nothing better has not changed the situation, and reconciling
    again against equivalent evidence is the full-price no-op this whole
    module exists to refuse.

    Best-effort: a failed search leaves the run exactly as it was.
    """
    try:
        candidates, cache = await _timed_step(
            "quality-search",
            lambda: _step_discover(title, artist, max_candidates, True),
            settings.discover_timeout_seconds,
        )
    except Exception as e:  # noqa: BLE001 — best-effort (incl. timeout)
        report.steps["quality-search"] = _fail_text(e, settings.discover_timeout_seconds)
        return None

    old_scores = score_candidates(report.candidates, report.mir)
    new_scores = score_candidates(candidates, report.mir)
    if not search_found_better(old_scores, new_scores):
        report.steps["quality-search"] = (
            f"ok: {len(candidates)} candidate source(s) found, none agreeing with the "
            f"audio better than the {len(report.candidates)} already gathered — "
            f"storing with the grade rather than paying for an equivalent retry"
        )
        return None

    report.steps["quality-search"] = (
        f"ok: {len(candidates)} candidate source(s) agreeing with the audio better "
        f"than the {len(report.candidates)} this run started with; reconciling once more"
    )
    report.candidates = candidates
    report.discovery_cache = cache
    return build_retry_feedback(grade, attribution)


def _resolve_guidance(song_id: str, guidance: str | None) -> tuple[str | None, str | None]:
    """Decide this run's reconciler guidance and where it came from.

    Request guidance wins and is persisted as the song's notes BEFORE any
    expensive step runs — a user's typed instruction must survive a run that
    dies at acquire or reconcile, otherwise a flaky analysis silently eats it.
    With no request guidance, stored notes replay. Store trouble degrades to
    "no guidance": notes are an assist, never a reason to fail an analysis.
    """
    from .store.song_notes import get_song_notes_store

    text = (guidance or "").strip()
    try:
        store = get_song_notes_store()
        if text:
            store.set(song_id, text)
            return text, "this request"
        stored = (store.get(song_id) or "").strip()
        if stored:
            return stored, "stored notes"
    except Exception as e:  # noqa: BLE001 — best-effort, never fatal
        log.warning("song notes unavailable for %s (continuing): %s", song_id, e)
        if text:
            return text, "this request"
    return (text or None), ("this request" if text else None)


def _load_stored_song(
    store: SongRepository, song_id: str
) -> tuple[Song | None, str | None, str | None]:
    """``(song, version, problem)`` for the stored latest version of a song.

    Best-effort by design: a song that doesn't exist yet, a backend that is
    down, and a document that no longer validates all mean the same thing to
    the callers here — "there is no prior version to reuse or compare
    against". Both callers then fail LOUDLY on their own terms (the
    listen=off precondition) or simply don't fire (the guard), so degrading
    quietly here can never turn into a silent data loss. `problem` carries
    WHY when the reason wasn't simply "no such song", so the precondition's
    error can say "your store is down" instead of "this song is new".
    """
    try:
        version = store.current_version(song_id)
        if version is None:
            return None, None, None  # a song analyzed for the first time
        return store.get(song_id), version, None
    except Exception as e:  # noqa: BLE001
        log.warning("no readable prior version for %s: %s", song_id, e)
        return None, None, str(e)


def _decode_prior_song(prior_song: dict | None) -> Song | None:
    """The caller-supplied prior document, or None if it isn't a decodable
    Song. Never fatal: `priorSong` is free-form evidence for the reconciler
    (which handles it as a dict), and a partial one must not fail a run that
    would otherwise have succeeded — it just can't be a timing source."""
    if not prior_song:
        return None
    try:
        return Song.model_validate(prior_song)
    except Exception as e:  # noqa: BLE001
        log.info("priorSong is not a decodable Song, ignoring it for timing (%s)", e)
        return None


def _content_version(song: Song | None) -> str | None:
    """The content sha a request-supplied prior song WOULD have if stored —
    so a carry-forward from an unstored document still names its source."""
    if song is None:
        return None
    from .store.base import version_sha

    return version_sha(song)


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
    scope: AnalysisScope | None = None,
    refresh_cache: bool = False,
    allow_timing_loss: bool = False,
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
            scope=scope,
            refresh_cache=refresh_cache,
            allow_timing_loss=allow_timing_loss,
        )
    )
