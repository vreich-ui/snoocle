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
  This pipeline is also the caller that DECLARES which pass will time each
  document (``TimingAuthority``, read off 5b's own branch conditions), so the
  reconciler can clear the model's timing for that pass — and puts the cleared
  fields back itself if the pass it named then skips or raises, keeping a
  best-effort failure from becoming a fatal one.
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
from .reconcile import (
    ReconcileResult,
    TimingAuthority,
    provider_preflight,
    reconcile,
)
from .reconcile.agent_config import config_version
from .reconcile.engine import TIMING_RESTORE_ACTION, _load_agent_config
from .reconcile.match import score_candidates
from .reconcile.depth import resolve_depth
from .reconcile.trace import TraceRecorder, new_run_id, start_run
from .schema.song import ProvenanceEntry, Song, slugify_song_id
from .scope import AnalysisScope
from .store import SaveResult, SongRepository, VersionConflictError, get_repository
from .store.runs import get_run_store
from .store.run_admission import (
    Admission,
    DuplicateRunError,
    get_run_admission_store,
    idempotency_key,
)
from .store.song_notes import length_error as notes_length_error
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
    guidance_withheld: str | None = None,
    prior_song: dict | None = None,
    depth=None,
    scope: AnalysisScope | None = None,
    evidence_manifest: dict | None = None,
    mir_cache: MirCacheInfo | None = None,
    patch_ops_eligible: bool = False,
    quality_feedback: str | None = None,
    timing_authority: TimingAuthority = TimingAuthority.NONE,
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
        guidance_withheld=guidance_withheld,
        prior_song=prior_song,
        depth=depth,
        scope=scope,
        evidence_manifest=evidence_manifest,
        mir_cache=mir_cache,
        patch_ops_eligible=patch_ops_eligible,
        quality_feedback=quality_feedback,
        timing_authority=timing_authority,
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
    force: bool = False,
    force_reason: str | None = None,
) -> PipelineReport:
    resolved_provider = (provider or settings.llm_provider).lower()
    # analysisDepth is the canonical control; the older `accuracy` field is
    # honored as its source when a depth isn't given explicitly. The chosen
    # profile drives MIR accuracy, agent effort, the tool budget, and syncMap.
    depth = resolve_depth(analysis_depth or accuracy)
    accuracy = depth.accuracy
    steps: dict[str, str] = {}

    if force and not (force_reason or "").strip():
        raise ValueError("forceReason is required when force=true")

    # Provider preflight (FATAL, instant). A provider that can't serve ANY
    # request — unknown name or missing credential/endpoint — must fail here,
    # not minutes later at reconcile after discover/acquire/MIR have all been
    # paid for (clients retry 502s, so late failure multiplies into a loop of
    # full-price doomed runs).
    problem = provider_preflight(resolved_provider)
    if problem:
        raise PipelineStepError("reconcile", problem, error_code="provider_not_configured")

    # `guidance` becomes this song's stored CORRECTION (`_resolve_guidance`), so
    # it is a notes write and is bounded by the same per-slot ceiling as a
    # preference write. Checked HERE, at the pipeline's door, for two reasons:
    #
    # - it must be a request-validation rejection, not a store-level raise. The
    #   store write happens inside `_resolve_guidance`'s deliberately
    #   best-effort try/except (a notes-store outage may never fail an
    #   analysis), so a `ValueError` from `set_correction` would be caught there
    #   and the run would proceed on the over-cap text — enforcement that
    #   enforces nothing.
    # - it must be EARLY. `_resolve_guidance` runs after identity resolution,
    #   which can be a model call, and a guided analyze that fails only after
    #   paying for work is worse than one that never started. Nothing above this
    #   line costs anything.
    #
    # The REST and MCP surfaces reject the same text with the same message
    # before they ever get here (api.post_songs_analyze,
    # mcp_server.analyze_and_store_song); this is the backstop that keeps a
    # direct `run_pipeline_async` caller from smuggling one past them.
    guidance_problem = notes_length_error(guidance)
    if guidance_problem:
        raise ValueError(guidance_problem)

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
    #
    # `resolved` reports each note lifetime SEPARATELY as well as combined
    # (:class:`ResolvedGuidance`). `pending_correction` is the raw, unlabeled
    # correction text (if any) this run's guidance is standing in for —
    # distinct from `guidance` itself, which may be a preference+correction
    # combination (see song_notes.combine_guidance). Two readers below need it
    # un-combined: the single-shot consumption call after storing, which must
    # compare-and-set against the correction's OWN stored text, and scope
    # inference, which must route on what the caller said about THIS run and
    # nothing replayed. `resolved.preference` has a third reader: the notes
    # STEP, which states which halves were in force and may not infer that by
    # comparing strings.
    resolved = _resolve_guidance(song_id, guidance)
    guidance, guidance_origin = resolved.text, resolved.origin
    pending_correction = resolved.correction
    # Recorded here so the step keeps its place in the report's order, then
    # REWRITTEN once the scope is settled below — what the model is actually
    # shown is not decided until then (see `model_guidance`).
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
    #
    # Guidance may only ROUTE a run when the caller attached it to THIS request.
    # A note replayed from the store says nothing about what this run is for, and
    # a request that sent no guidance at all must get the pipeline it asked for:
    # inferring notes-only from a replayed note is how a full analysis silently
    # became a re-application of a previous request's correction. Replayed
    # guidance is still classified when an EXPLICIT notes-only scope makes
    # `patch_eligible` readable below.
    #
    # A classification is computed ONLY where one of its two readers exists,
    # because its fallback tier is a real blocking `provider.complete` (see
    # correction_routing._llm_classify) and a run must not pay for an answer
    # nothing looks at. Both readers need `scope is None or scope.notes_only`:
    # inference is gated on `scope is None`, and `patch_ops_eligible` is gated
    # on `scope.notes_only`. Under an EXPLICIT non-notes-only scope neither can
    # fire, so an open-ended correction there (no deterministic rule matches,
    # straight to the model tier) bought one extra synchronous model call per
    # request for a result that was discarded.
    #
    # The same rule has to hold WITHIN one string, which is why routing reads
    # `request_correction` and never `guidance`: `guidance` may be a standing
    # preference combined with this request's correction, and
    # `correction_routing._deterministic_signals` scans whatever text it is
    # handed, so classifying the combination lets a preference's words route
    # the run. A preference saying "the bridge is Bm, not D" would then send
    # "double-check this against better sources" down the notes-only path —
    # no discovery, no listening, a re-application of the document the caller
    # asked to have re-verified — and, because a preference never expires, on
    # every guided analyze from then on. Store content is not caller intent no
    # matter which half of a string it arrives in.
    classification = None
    scope_was_inferred = False
    guidance_from_this_request = guidance_origin == "this request"
    # THIS request's own correction text, unlabeled and with nothing replayed
    # combined in — the only text that expresses intent about this run.
    request_correction = pending_correction if guidance_from_this_request else None
    # What a notes-only run acts on, and (below) all it is shown: the correction
    # alone when there is one, otherwise whatever replayed (a preference by
    # itself, which is then the only instruction such a run has).
    targeted_guidance = pending_correction or guidance
    if stored_prior is not None or request_prior is not None:
        if request_correction and (scope is None or scope.notes_only):
            classification = classify_correction(request_correction)
            if scope is None and classification.is_targeted_correction:
                scope = AnalysisScope(listen=False, reconcile=False)
                scope_was_inferred = True
                log.info(
                    "pipeline.scope inferred notes-only for %s: %s",
                    song_id, classification.describe(),
                )
        elif scope is not None and scope.notes_only and targeted_guidance:
            # Nothing to route (the caller already did), but patch eligibility
            # is still read below — and it is read about the text this run
            # applies, which is the same text the model sees.
            classification = classify_correction(targeted_guidance)

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

    # WHAT THE MODEL SEES, which is not always all of what is in force.
    #
    # A notes-only run is handed one contract, in the prompt and in the agent
    # payload both: "return this document with these notes applied and nothing
    # else changed" (reconcile/prompt.py, reconcile/anthropic_agent.py). A
    # standing preference is not part of that ask — it is an instruction about
    # how to BUILD this song, open-ended by nature ("capo-free voicings
    # please"), and a run whose whole job is one chord has neither the licence
    # to act on it nor, having gathered nothing, the evidence to. Handing it
    # over anyway asks the model to do two contradictory things at once, and
    # the patch path (a closed op set naming exact edits) has no way to express
    # it at all. So on a notes-only run the model sees the CORRECTION alone.
    #
    # Two things this deliberately is not:
    #   - not a regression: before the preference/correction split a preference
    #     only ever entered runs that supplied no guidance of their own, so a
    #     targeted correction never carried one. This restores that, it does
    #     not narrow it.
    #   - not "notes-only drops the preference": when a notes-only run has NO
    #     correction (an explicit `scope` plus a stored preference), the
    #     preference is the only instruction the run has and it stands alone,
    #     exactly as it always did — `targeted_guidance` falls back to it.
    #
    # A FULL run still sees both. There the preference is squarely in scope:
    # the document is being rebuilt from evidence, which is the moment a
    # standing instruction about how to build it applies.
    notes_only_run = scope is not None and scope.notes_only
    model_guidance = targeted_guidance if notes_only_run else guidance
    # The two halves are read from what `_resolve_guidance` says was IN FORCE,
    # never recovered by comparing `model_guidance`/`guidance`/
    # `pending_correction` against each other. String equality cannot tell "a
    # preference was combined in" from "there was no preference and the one
    # correction came back unchanged", and asserting the first when the second
    # is true puts a standing instruction into a report for a song that has
    # never had one.
    both_in_force = bool(resolved.preference and resolved.correction)
    withheld_preference = resolved.preference if (both_in_force and notes_only_run) else None
    if guidance:
        if withheld_preference:
            detail = " (correction only; standing preference held back on a notes-only run)"
        elif both_in_force:
            detail = " (preference + correction combined)"
        else:
            detail = ""
        report.steps["notes"] = f"ok: guidance applied (from {guidance_origin}){detail}"

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
        # `model_guidance`, not `guidance`: the manifest describes what this run
        # HANDS the reconciler, and it is itself part of the agent payload — a
        # manifest quoting guidance the run withheld would both misdescribe the
        # run and smuggle the withheld half to the model through the back door.
        guidance=model_guidance,
        guidance_origin=guidance_origin,
    )
    resolved_config_version = config_version(_load_agent_config())
    admission_key, admission_evidence_hash = idempotency_key(
        song_id, resolved_config_version, report.evidence_manifest
    )
    admitted_run_id = new_run_id()
    admission_store = get_run_admission_store()
    try:
        admission = admission_store.admit(
            key=admission_key,
            evidence_hash=admission_evidence_hash,
            run_id=admitted_run_id,
            lease_seconds=settings.run_lock_lease_seconds,
            completed_ttl_seconds=settings.duplicate_run_ttl_seconds,
            force=force,
            force_reason=force_reason,
        )
    except DuplicateRunError as duplicate:
        # `_resolve_guidance` deliberately persists a request correction before
        # expensive work. A client retry after a completed run writes that same
        # correction again before reaching this gate; restore its consumed
        # marker so a refused duplicate cannot reopen already-applied guidance.
        completed_version = (duplicate.summary or {}).get("storedVersion")
        if pending_correction and completed_version:
            _consume_applied_correction(song_id, pending_correction, completed_version)
        raise
    recorder = start_run(
        song_id,
        resolved_provider,
        depth.name,
        run_id=admitted_run_id,
        config_version=resolved_config_version,
        idempotency_key=admission_key,
        evidence_hash=admission_evidence_hash,
        forced=force,
        force_reason=(force_reason or "").strip() or None,
    )
    report.run_id = recorder.trace.run_id

    # A withheld preference is a DECISION, and it has to outlive the HTTP
    # response. `report.steps` is returned to the caller and then thrown away —
    # nothing persists it — so on its own it said "a standing instruction was in
    # force and this run deliberately did not apply it" to exactly one reader,
    # once. reconcile/engine.py states the opposing rule for the applied case
    # ("Guidance is never applied silently: a reader of the song's history must
    # be able to see that a human instruction shaped this run"), and guidance
    # deliberately NOT applied is the same claim about the same run. So it goes
    # in both durable places: this run's persisted trace, and (via
    # `guidance_withheld` below) the reconciled provenance entry that already
    # names the guidance origin.
    if withheld_preference:
        recorder.step(
            "inputs",
            "notes:preference-withheld",
            "standing preference held back: a notes-only run is shown the correction alone",
            detail={"preference": withheld_preference, "correction": pending_correction},
        )

    # What the quality gate (5f) has already spent. Threaded into
    # `plan_escalation`, which is what makes the one-retry ceiling structural
    # rather than a promise: with these at their real values a second
    # escalation cannot be planned. See quality/escalation.py.
    quality_retries_spent = 0
    quality_searches_spent = 0

    # WHICH deterministic pass will time this run's document. The reconciler is
    # TOLD this (reconcile/engine.py's TimingAuthority) instead of inferring it
    # from "was a MIR handed in?", because a MIR in the inputs says nothing
    # about whether anything is going to run afterwards.
    #
    # Read straight off 5b's own branch conditions below, and it has to stay
    # that way — any drift means the reconciler clears fields for a pass that
    # doesn't run:
    #
    #   `not want_listen and timing_prior is not None`  -> the carry-forward
    #     branch, verbatim. The `timing_prior is not None` half is load-bearing:
    #     `listen=off` with no prior and `allowTimingLoss=true` (the combination
    #     the guard above tells callers to use) falls through to the else-branch,
    #     where it has no MIR either — so nothing times that document and nothing
    #     may be stripped from it.
    #   `report.mir is not None`  -> the else-branch with something to snap to.
    #     `snap_chords` is a documented no-op without a MIR (timing/snap.py:250),
    #     so a MIR-less run through that branch is NONE, not SNAP.
    #
    # The patch path deliberately has no entry here: only the engine knows a
    # patch happened, and it never strips — which matches 5b's `patched -> skip
    # entirely`.
    if not want_listen and timing_prior is not None:
        timing_authority = TimingAuthority.CARRY_FORWARD
    elif report.mir is not None:
        timing_authority = TimingAuthority.SNAP
    else:
        timing_authority = TimingAuthority.NONE

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
                    trace=recorder, guidance=model_guidance, guidance_origin=guidance_origin,
                    guidance_withheld=withheld_preference,
                    prior_song=effective_prior_song, depth=depth, scope=scope,
                    evidence_manifest=report.evidence_manifest,
                    mir_cache=report.mir_cache,
                    patch_ops_eligible=(
                        classification.patch_eligible
                        if classification is not None and scope is not None and scope.notes_only
                        else False
                    ),
                    quality_feedback=quality_feedback,
                    timing_authority=timing_authority,
                ),
                settings.reconcile_timeout_seconds,
            )
        except asyncio.TimeoutError as e:
            recorder.finish("error", error=f"timed out after {settings.reconcile_timeout_seconds:.0f}s")
            _persist_trace(recorder)
            if quality_feedback is None:
                _abandon_admission(admission_store, admission)
            raise PipelineStepError(
                "reconcile",
                f"timed out after {settings.reconcile_timeout_seconds:.0f}s",
                steps=report.steps,
            ) from e
        except Exception as e:  # noqa: BLE001 — ReconcileError/ProviderError/anything else
            recorder.finish("error", error=str(e)[:2000])
            _persist_trace(recorder)
            if quality_feedback is None:
                _abandon_admission(admission_store, admission)
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
        #
        # These conditions are also what `timing_authority` above was derived
        # from, and the two must not drift: the reconciler cleared the model's
        # timing on the strength of that declaration. Where a branch can end
        # without the declared pass having written anything, this block puts back
        # what was cleared (`TimingStrip.restore`) rather than leaving a
        # best-effort failure to be re-reported as a fatal loss by 5e.
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
            # What the strip COST, reported rather than left for the grade to
            # imply. The reconciler's own times are gone (the model is not the
            # authority on timing — reconcile/engine.py), a line it added or
            # reworded has no partner in the prior version to carry from, and
            # LRC is skipped on this path (5c below), so these elements reach
            # the store with no time at all. That is the right answer — "could
            # not time this region" beats fabricated spacing, see
            # quality/escalation.py — but the operator has to be able to read it
            # here instead of discovering it as a timingCoverage drop.
            if carry_stats.lines_empty or carry_stats.placements_empty:
                untimed = (
                    f"{carry_stats.lines_empty} line(s) and "
                    f"{carry_stats.placements_empty} placement(s) left untimed "
                    f"(no match in the prior version, and nothing else times "
                    f"them on this path)"
                )
                report.steps["timing"] += f"; {untimed}"
                recorder.step(
                    "timing", "left-untimed", untimed,
                    detail={
                        "linesUntimed": carry_stats.lines_empty,
                        "placementsUntimed": carry_stats.placements_empty,
                        "linesTotal": carry_stats.lines_total,
                        "placementsTotal": carry_stats.placements_total,
                    },
                )
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
                # This pass is documented non-fatal, and it must stay that way.
                # The reconciler cleared audio.beats/metadata.bpm/audio.syncMap
                # on the promise that THIS call would rewrite them; it raised, so
                # the promise is void and the strip is undone here — otherwise a
                # best-effort failure would empty fields the prior version had
                # and trip the FATAL pre-store guard (5e) instead.
                if result.timing_strip is not None:
                    try:
                        result.song, restored = result.timing_strip.restore(
                            result.song, reason=f"timing.snap failed: {e}"
                        )
                    except Exception as undo_error:  # noqa: BLE001 — same promise
                        log.warning(
                            "pipeline.step error step=timing-restore err=%s", undo_error
                        )
                        restored = []
                    if restored:
                        report.steps["timing"] += (
                            f"; restored {', '.join(restored)} the failed pass owed"
                        )
                        recorder.step(
                            "timing", TIMING_RESTORE_ACTION,
                            f"timing.snap failed, so the fields stripped for it were "
                            f"put back as the model supplied them: {', '.join(restored)}",
                            detail={"restored": restored, "reason": str(e)[:500]},
                        )

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

            # The escalation decision is the server-side authority. Search may
            # produce useful evidence, but it may not manufacture a retry after
            # the quality gate explicitly returned retry=false.
            if retry_feedback is not None and escalation.retry:
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
            elif retry_feedback is not None:
                report.steps["quality-retry"] = "skipped: escalation.retry=false"
                recorder.step(
                    "quality",
                    "quality-retry-suppressed",
                    "internally-generated retry suppressed: escalation.retry=false",
                )

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
                # The marker states the escalation's OWN reason: two different
                # situations write it (an unusable recording, or a collapse run
                # the guard could not spread) and the fault attribution only
                # describes the first. See `timing_unreliable_provenance_entry`.
                entries.append(
                    timing_unreliable_provenance_entry(
                        attribution, reason=escalation.reason
                    )
                )
                report.steps["timing-reliability"] = (
                    f"marked unreliable ({escalation.mark_cause})"
                    if escalation.mark_cause
                    else "marked unreliable"
                )
            # Separate from the marker above, and gated on its own flag: only an
            # unusable RECORDING is fixed by a different recording. A surviving
            # collapse run marks the timing unreliable too, and searching for
            # another recording of the same song would spend a network call on a
            # region no recording this run had could time (see
            # quality/escalation.py). One search, no download, no analysis, and
            # skipped for the mock provider, which is the fully-offline path and
            # must make zero external calls of any kind.
            if (
                escalation.suggest_alternative_recording
                and settings.quality_suggest_recordings
                and resolved_provider != "mock"
            ):
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
            _abandon_admission(admission_store, admission)
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
        _abandon_admission(admission_store, admission)
        raise  # -> HTTP 409, not a 502
    except asyncio.TimeoutError as e:
        _abandon_admission(admission_store, admission)
        raise PipelineStepError(
            "store",
            f"timed out after {settings.store_timeout_seconds:.0f}s",
            steps=report.steps,
        ) from e
    except Exception as e:  # noqa: BLE001
        _abandon_admission(admission_store, admission)
        raise PipelineStepError("store", str(e), steps=report.steps) from e
    report.stored_version = saved.version
    report.stored_timestamp = saved.timestamp
    report.steps["store"] = f"ok: version {saved.version}"

    # The guidance this run applied is now IN a stored version, so a pending
    # correction has done its job exactly once. Deliberately AFTER the store:
    # a run that dies earlier leaves the note unconsumed and it replays into the
    # retry, which is the reason _resolve_guidance persists it up front.
    #
    # `pending_correction`, NOT `guidance`: `guidance` may be a preference
    # combined with the correction (see song_notes.combine_guidance), and the
    # store's compare-and-set stamps a correction by matching its OWN stored
    # text exactly — comparing the combined string would never match, and a
    # correction that can never be marked applied would replay forever.
    quality_retry = bool(
        (((recorder.trace.quality or {}).get("escalation") or {}).get("retry", False))
    )
    admission_store.complete(
        admission,
        retry=quality_retry,
        summary={
            "runId": report.run_id,
            "songId": report.song_id,
            "status": recorder.trace.status,
            "storedVersion": report.stored_version,
            "finishedAt": recorder.trace.finished_at,
            "quality": recorder.trace.quality,
            "evidenceManifest": report.evidence_manifest,
            "forced": recorder.trace.forced,
        },
    )
    # Complete admission BEFORE stamping the correction. A duplicate arriving
    # in this tiny post-store window now sees the completed version in the lock
    # and can restore the stamp after `_resolve_guidance` rewrites the same
    # correction; stamping first left a race where that retry reopened it.
    if pending_correction:
        _consume_applied_correction(song_id, pending_correction, saved.version)
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


@dataclass(frozen=True)
class ResolvedGuidance:
    """What was actually in force for one run, half by half.

    Each half is reported SEPARATELY rather than left to be recovered by
    comparing ``text`` against ``correction``: a report line that says "a
    standing preference was combined in" or "a standing preference was held
    back" is a claim about the store's state, and reconstructing it from string
    equality made it claim a preference existed on a song that had never had
    one.
    """

    #: Everything in force, as one string — a bare half, or both combined
    #: (``song_notes.combine_guidance``). Not necessarily what the model is
    #: shown; the caller decides that once the scope is settled
    #: (:func:`run_pipeline_async`'s ``model_guidance``).
    text: str | None
    #: ``"this request"`` | ``"stored notes"`` | ``None``.
    origin: str | None
    #: The standing preference in force, or ``None`` when the song has none
    #: (or the store could not be read).
    preference: str | None
    #: The pending correction's OWN unlabeled text, or ``None``.
    correction: str | None


def _resolve_guidance(song_id: str, guidance: str | None) -> ResolvedGuidance:
    """Decide this run's reconciler guidance, where it came from, and which of
    the song's two note lifetimes were actually in force.

    ``text`` is everything in force for this run, and may be a durable
    PREFERENCE and a pending CORRECTION combined into one string (see
    ``song_notes.combine_guidance`` for the combination rule and why both being
    in force is legitimate).

    ``correction`` is that correction's OWN unlabeled text with nothing
    combined in, and has TWO readers that both need it un-combined:

    - :func:`_consume_applied_correction`, because the store stamps a
      correction by matching its stored text exactly and a combined string
      would never match it; and
    - scope inference, because a preference's words are not a statement about
      what THIS run is for. Classifying the combined string is what let a
      standing "the bridge is Bm, not D" route an open-ended re-verification
      request to the notes-only path.

    Request guidance wins and is persisted as the song's notes BEFORE any
    expensive step runs — a user's typed instruction must survive a run that
    dies at acquire or reconcile, otherwise a flaky analysis silently eats it.
    It is persisted as a single-shot CORRECTION: it replays into a retry that
    supplies none (the whole reason for storing it early) and stops replaying
    once :func:`_consume_applied_correction` records the version it landed in.
    Persisting it never disturbs a standing preference already on file — they
    live in independent fields of the same record — so it is combined with
    one, if present, for THIS run's guidance.

    With no request guidance, a stored note replays via
    ``song_notes.replay_guidance``: a durable preference always does, a
    pending correction only until it has been applied, and both combine when
    both are in force. Store trouble degrades to "no guidance": notes are an
    assist, never a reason to fail an analysis. ``preference`` is ``None`` on
    that degraded path too — not "unknown": no preference was in force for this
    run, because none could be read, and the report must not claim one.

    ``guidance`` is assumed already length-checked by the caller
    (:func:`run_pipeline_async`, before anything expensive) — see
    ``song_notes.MAX_NOTES_CHARS``.
    """
    from .store.song_notes import (
        combine_guidance,
        get_song_notes_store,
        preference_text,
        replay_guidance,
    )

    text = (guidance or "").strip()
    try:
        store = get_song_notes_store()
        if text:
            # Read the standing preference BEFORE writing the correction (the
            # write never touches it anyway — independent fields — but this
            # keeps the two calls in the obvious order): a fresh correction
            # from this request still combines with whatever preference is
            # already on file, exactly as a replayed one would.
            preference = preference_text(store.get_record(song_id))
            store.set_correction(song_id, text)
            return ResolvedGuidance(
                text=combine_guidance(preference, text),
                origin="this request",
                preference=preference,
                correction=text,
            )
        record = store.get_record(song_id)
        combined, pending_correction = replay_guidance(record)
        if combined:
            return ResolvedGuidance(
                text=combined,
                origin="stored notes",
                preference=preference_text(record),
                correction=pending_correction,
            )
    except Exception as e:  # noqa: BLE001 — best-effort, never fatal
        log.warning("song notes unavailable for %s (continuing): %s", song_id, e)
    return ResolvedGuidance(
        text=(text or None),
        origin=("this request" if text else None),
        preference=None,
        correction=(text or None),
    )


def _consume_applied_correction(song_id: str, correction_notes: str, version: str) -> None:
    """Record that the song's PENDING CORRECTION landed in ``version``.

    This is what makes a correction single-shot. It is in the document the next
    run starts from now, so replaying it into a run that asked for nothing would
    silently re-apply a previous request's instruction — and (before the origin
    check at the scope-inference step) reclassify that run as notes-only.

    ``correction_notes`` must be the correction's OWN raw text — never a
    preference, and never the preference+correction combination
    ``_resolve_guidance`` may have handed the reconciler — because the store
    stamps a correction by matching its stored text exactly; a combined
    string would never match and the correction would replay forever instead
    of ever being marked applied.

    A no-op when there is no pending correction (a durable preference alone
    is never consumed) or when the caller replaced it mid-run. Best-effort:
    the run has already stored, and a failed bookkeeping write must not turn
    a successful analysis into an error.
    """
    from .store.song_notes import get_song_notes_store

    try:
        get_song_notes_store().mark_applied(song_id, correction_notes, version)
    except Exception as e:  # noqa: BLE001 — never fatal
        log.warning("could not mark notes applied for %s (continuing): %s", song_id, e)


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


def _abandon_admission(store, admission: Admission) -> None:
    """Release a failed run without hiding the original pipeline error."""
    try:
        store.abandon(admission)
    except Exception as e:  # noqa: BLE001
        # The lease remains the crash-safety backstop when Firestore itself is
        # unavailable during cleanup.
        log.warning("run admission cleanup failed (lease will expire): %s", e)


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
    force: bool = False,
    force_reason: str | None = None,
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
            force=force,
            force_reason=force_reason,
        )
    )
