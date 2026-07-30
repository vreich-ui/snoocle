"""Mode B: re-align an existing Song to a DIFFERENT recording of the same song.

A song that has been through Mode A once carries the expensive part — words
and a chord sequence, agreed across sources and reviewed by a human. A second
recording of that song needs none of that work repeated; it needs new TIMES,
and times come from audio. So Mode B is a full analysis of the new recording
and a re-timing of the document you already have, almost entirely
deterministic:

1. **Same-recording pre-check** (free, and only when it is free). Two uploads
   of the SAME recording are ``timing/offset.py``'s problem, not this one: a
   constant lag exists and ``POST /v1/songs/{id}/video-offset`` finds it for
   the price of one cross-correlation. Mode B costs a download plus a full MIR
   pass. When both audio files happen to be in the local cache already, the
   cross-correlation runs first and refuses with a pointer to the cheap tool.
2. **MIR on the new audio.** A cache miss by design — different bytes, so
   there is nothing to reuse (the same video re-aligned twice does hit).
3. **Transposition.** ``reconcile.match.score_candidate`` over all twelve
   shifts, the same search the grader uses (see ``timing/realign.py``).
   A document in G against a recording in Bb moves +3.
4. **Structural comparison.** Does the stored document still fit this
   arrangement? This is the ONLY thing that can spend model tokens, so it is
   deliberately conservative: a live version of a song already in the library
   normally costs zero.
5. **The timing passes, exactly as any run gets them**: ``snap_chords``
   against the new timeline, LRC line alignment, the collapse guard, per-chord
   confidence, then the quality gate. Nothing here is Mode-B-specific, which
   is the point — a re-aligned version is graded by the same standard as a
   freshly reconciled one.

Lyrics and the chord SEQUENCE carry over untouched. The result is stored as a
new version of the SAME song id (never a new song: it is the same song, timed
differently), recording which video supplied the timing and what was done.

The companion action lives in :mod:`snoocle_server.recordings`, which reports
alternative recordings when a version comes out timing-unreliable and never
analyzes one on its own.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from . import __version__
from .audio.acquire import AcquiredAudio, YouTubeAuthError, acquire, cached_audio_path
from .config import settings
from .mir import MirAnalysis, analyze_audio
from .mir.cache import MirCacheInfo, analyze_cached
# Shared with the analyze pipeline on purpose: one wall-clock/logging
# convention for external steps, one fatal-error shape for the API layer to
# translate, one run-trace persistence path.
from .pipeline import PipelineStepError, _fail_text, _persist_trace, _timed_step, get_store
from .quality import QualityDecision, evaluate as evaluate_quality
from .quality.grader import grade_provenance_entry, timing_unreliable_provenance_entry
from .recordings import RecordingSuggestions, suggest_recordings
from .reconcile import provider_preflight, reconcile
from .reconcile.depth import resolve_depth
from .reconcile.trace import start_run
from .schema.song import ProvenanceEntry, Song
from .scope import AnalysisScope
from .store import SaveResult, SongRepository, VersionConflictError
from .timing.carry_forward import audio_data_lost
from .timing.collapse_guard import guard_against_collapsed_timing
from .timing.confidence import build_review_queue, score_song
from .timing.lrc import apply_lrc, fetch_lrc, match_lrc_to_lines
from .timing.offset import OffsetEstimate, estimate_offset
from .timing.realign import (
    StructureComparison,
    TranspositionEstimate,
    apply_transposition,
    clear_recording_timing,
    compare_structure,
    derive_transposition,
    retime_sections,
)
from .timing.snap import snap_chords

log = logging.getLogger(__name__)

#: Provenance ``action`` for a Mode B pass, alongside ``"timing-snap"`` and
#: ``"timing-carry-forward"``.
ACTION = "timing-realign"


class RealignError(PipelineStepError):
    """A fatal Mode B step. Carries the step name and the outcomes so far, the
    same contract the analyze pipeline's failures have (-> HTTP 502/409)."""


@dataclass
class RealignReport:
    """Everything Mode B did, in the order it did it."""

    song_id: str
    video_id: str
    steps: dict[str, str] = field(default_factory=dict)
    source_version: Optional[str] = None  # the document this re-timed
    stored_version: Optional[str] = None
    stored_timestamp: Optional[str] = None
    run_id: Optional[str] = None
    transposition: Optional[TranspositionEstimate] = None
    structure: Optional[StructureComparison] = None
    model_consulted: bool = False
    quality: Optional[QualityDecision] = None
    suggestions: Optional[RecordingSuggestions] = None
    same_recording: Optional[OffsetEstimate] = None
    mir: Optional[MirAnalysis] = None
    error_code: Optional[str] = None

    def to_dict(self) -> dict:
        out: dict = {
            "songId": self.song_id,
            "videoId": self.video_id,
            "steps": self.steps,
            "sourceVersion": self.source_version,
            "storedVersion": self.stored_version,
            "storedTimestamp": self.stored_timestamp,
            "runId": self.run_id,
            "modelConsulted": self.model_consulted,
            "transposition": (
                self.transposition.to_dict() if self.transposition is not None else None
            ),
            "structure": self.structure.to_dict() if self.structure is not None else None,
            "quality": self.quality.to_dict() if self.quality is not None else None,
        }
        if self.suggestions is not None:
            out["recordingSuggestions"] = self.suggestions.to_dict()
        if self.same_recording is not None:
            out["sameRecordingCheck"] = {
                "offsetSeconds": round(self.same_recording.offset_seconds, 2),
                "confidence": round(self.same_recording.confidence, 3),
            }
        return out


# --- steps (each patchable by name, like pipeline.py's) ----------------------


def _step_acquire(video_id: str) -> AcquiredAudio:
    return acquire(video_url_or_id=video_id)


def _step_mir(audio_path: str, accuracy: str, refresh: bool) -> tuple[MirAnalysis, MirCacheInfo]:
    return analyze_cached(
        audio_path,
        accuracy=accuracy,
        compute=lambda: analyze_audio(audio_path, accuracy=accuracy),
        refresh=refresh,
    )


def same_recording_check(song: Song, video_id: str) -> Optional[OffsetEstimate]:
    """Is `video_id` the same RECORDING the document is already timed against?

    If it is, Mode B is the wrong tool: a constant lag exists and
    ``timing/offset.py`` finds it for one cross-correlation instead of a
    download plus a full MIR pass. That is the one piece of machinery the two
    paths genuinely share, and this is where they meet.

    Returns None — "not checked" — unless BOTH audio files are already in the
    local cache. The check is only worth having while it is free: downloading
    the reference audio purely to decide whether to download the other one
    inverts the saving it exists to make. Also returns None when the estimate
    cannot be computed at all (no librosa, unreadable audio); a missing check
    must never block a re-alignment.
    """
    reference = song.audio.analyzedVideoId
    if not reference or reference == video_id:
        return None
    ref_path = cached_audio_path(reference)
    other_path = cached_audio_path(video_id)
    if ref_path is None or other_path is None:
        return None
    try:
        return estimate_offset(ref_path, other_path, settings.offset_max_search_seconds)
    except Exception as e:  # noqa: BLE001 — an optional check, never a blocker
        log.info("same-recording check unavailable for %s vs %s: %s", reference, video_id, e)
        return None


def _structure_feedback(song: Song, comparison: StructureComparison, video_id: str) -> str:
    """What the model is told, when structural comparison found a difference.

    Narrow on purpose. The words and the chord vocabulary are settled — they
    came from multiple sources and (usually) a human review. What is open is
    how many times each part repeats in THIS performance. Saying that plainly
    is what keeps a structural fix from turning into a re-transcription.
    """
    return "\n".join(
        [
            "## Structural difference between this document and the new recording",
            (
                f"This document's words and chords are settled. It was timed against a "
                f"DIFFERENT recording of the same song, and it is being re-timed to "
                f"{video_id}. Deterministic comparison found a difference the document "
                f"cannot explain:"
            ),
            "",
            *(f"- {reason}" for reason in comparison.reasons),
            "",
            (
                f"For reference: the document has {comparison.stored_sections} section(s) "
                f"and {len(song.lines)} line(s); the new recording's structure analysis "
                f"found {comparison.new_segments} segment(s)"
            )
            + (
                f" across {comparison.new_duration:.0f}s"
                if comparison.new_duration
                else ""
            )
            + ".",
            "",
            "Resolve ONLY the structural difference: add or remove repeats of sections "
            "that already exist (a chorus sung three times where the document has it "
            "twice, an outro this performance cuts), and adjust section boundaries to "
            "match. Every line keeps its words by referencing the prior song, and every "
            "chord keeps the symbol the document gives it. Do not re-transcribe, do not "
            "change chord identities, do not invent lyrics, and do not fill in times — a "
            "deterministic pass measures those from the audio after you answer.",
        ]
    )


def _realign_provenance(
    *,
    video_id: str,
    source_version: Optional[str],
    transposition: TranspositionEstimate,
    comparison: StructureComparison,
    model_consulted: bool,
    model_note: str,
) -> ProvenanceEntry:
    """The entry that makes a re-aligned version legible six months later."""
    shift = (
        f"transposed {transposition.semitones:+d} semitone(s)"
        if transposition.applies
        else "no transposition applied"
    )
    return ProvenanceEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        actor=f"snoocle-server/{__version__}",
        action=ACTION,
        sources=[
            f"timing-video:{video_id}",
            f"source-document:{source_version or 'unversioned'}",
        ],
        confidence=round(transposition.score.score, 3),
        notes=(
            f"re-aligned to recording {video_id}; lyrics and chord sequence carried "
            f"from the source document unchanged; {shift} ({transposition.reason}); "
            f"structure: {comparison.describe()}; model "
            + ("consulted: " + model_note if model_consulted else "NOT consulted")
        ),
    )


async def realign_song_async(
    song_id: str,
    video_id: str,
    *,
    store: Optional[SongRepository] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    analysis_depth: Optional[str] = None,
    expected_version: Optional[str] = None,
    source_version: Optional[str] = None,
    allow_timing_loss: bool = False,
    allow_same_recording: bool = False,
    refresh_cache: bool = False,
) -> RealignReport:
    """Re-align `song_id`'s stored document to the recording in `video_id`.

    `source_version` re-times a specific stored version rather than the latest
    (the operator may want the human-reviewed one). `allow_same_recording`
    overrides the pre-check in step 1. Raises :class:`RealignError` for a fatal
    step; everything advisory degrades into ``steps``.
    """
    report = RealignReport(song_id=song_id, video_id=video_id)
    store = store or get_store()
    depth = resolve_depth(analysis_depth)

    # 1. the source document (FATAL — Mode B has nothing to re-time without it)
    try:
        stored = store.get(song_id, source_version)
        report.source_version = source_version or store.current_version(song_id)
    except Exception as e:  # noqa: BLE001 — no document, no Mode B
        raise RealignError(
            "source",
            f"no readable stored document for {song_id!r} to re-align: {e}",
            error_code="song_not_found",
        ) from e
    report.steps["source"] = (
        f"ok: {len(stored.lines)} line(s), "
        f"{sum(len(l.chordPlacements) for l in stored.lines)} chord(s) "
        f"from version {report.source_version}"
    )

    # 2. same-recording pre-check (see same_recording_check)
    if allow_same_recording:
        report.steps["same-recording-check"] = "skipped (allowSameRecording)"
    else:
        estimate = await asyncio.to_thread(same_recording_check, stored, video_id)
        report.same_recording = estimate
        if estimate is None:
            report.steps["same-recording-check"] = (
                "skipped (both audio files would have to be cached locally for this "
                "to be free)"
            )
        elif estimate.confidence >= settings.offset_min_confidence:
            raise RealignError(
                "same-recording-check",
                (
                    f"{video_id} looks like the SAME recording as {song_id!r}'s current "
                    f"reference {stored.audio.analyzedVideoId} (cross-correlation "
                    f"confidence {estimate.confidence:.2f} at "
                    f"{estimate.offset_seconds:+.2f}s). Mode B costs a download and a "
                    f"full MIR pass; POST /v1/songs/{song_id}/video-offset with "
                    f'videoId={video_id!r} handles a re-upload of the same recording '
                    f"for one cross-correlation. Pass allowSameRecording=true to "
                    f"re-align anyway."
                ),
                steps=report.steps,
                error_code="same_recording_use_video_offset",
            )
        else:
            report.steps["same-recording-check"] = (
                f"ok: a different recording (cross-correlation confidence "
                f"{estimate.confidence:.2f} < {settings.offset_min_confidence:.2f})"
            )

    recorder = start_run(song_id, provider="realign", depth=depth.name)
    report.run_id = recorder.trace.run_id
    recorder.step(
        "inputs", "realign-inputs",
        f"re-aligning {song_id!r} (version {report.source_version}) to recording {video_id}",
        detail={"videoId": video_id, "sourceVersion": report.source_version},
    )

    # 3-4. the new recording (both FATAL — the new timing IS the deliverable)
    try:
        audio = await _timed_step(
            "acquire", lambda: _step_acquire(video_id), settings.acquire_timeout_seconds
        )
    except asyncio.TimeoutError as e:
        recorder.finish("error", error="acquire timed out")
        _persist_trace(recorder)
        raise RealignError(
            "acquire", f"timed out after {settings.acquire_timeout_seconds:.0f}s",
            steps=report.steps,
        ) from e
    except Exception as e:  # noqa: BLE001
        recorder.finish("error", error=str(e)[:2000])
        _persist_trace(recorder)
        raise RealignError(
            "acquire", str(e), steps=report.steps,
            error_code="youtube_auth_required" if isinstance(e, YouTubeAuthError) else None,
        ) from e
    report.steps["acquire"] = f"ok: {audio.video_id} ({audio.video_title})"

    try:
        mir, mir_cache = await _timed_step(
            "mir",
            lambda: _step_mir(audio.path, depth.accuracy, refresh_cache),
            settings.mir_timeout_seconds,
        )
    except asyncio.TimeoutError as e:
        recorder.finish("error", error="mir timed out")
        _persist_trace(recorder)
        raise RealignError(
            "mir", f"timed out after {settings.mir_timeout_seconds:.0f}s", steps=report.steps
        ) from e
    except Exception as e:  # noqa: BLE001 — unlike Mode A, MIR is not optional here
        recorder.finish("error", error=str(e)[:2000])
        _persist_trace(recorder)
        raise RealignError(
            "mir",
            f"the new recording could not be analyzed, so there is no timing to "
            f"re-align to: {e}",
            steps=report.steps,
            error_code="mir_failed",
        ) from e
    report.mir = mir
    report.steps["mir"] = f"ok: engines={mir.engines} (cache {mir_cache.status})"
    recorder.attach_mir(mir.to_run_payload())

    # 5. transposition, and 6. structure — both read the document BEFORE its
    # old timing is cleared (stored_recording_duration needs it).
    transposition = derive_transposition(stored, mir)
    report.transposition = transposition
    report.steps["transpose"] = (
        f"{'ok' if transposition.trustworthy else 'skipped'}: {transposition.reason}"
    )
    comparison = compare_structure(stored, mir)
    report.structure = comparison
    report.steps["structure"] = f"ok: {comparison.describe()}"
    recorder.step(
        "quality", "realign-plan",
        f"{transposition.reason}; {comparison.describe()}",
        detail={
            "transposition": transposition.to_dict(),
            "structure": comparison.to_dict(),
        },
    )

    base_provenance = list(stored.provenance)
    document = clear_recording_timing(apply_transposition(stored, transposition.semitones))

    # 7. the model, and ONLY when the structure comparison found something the
    # document cannot explain. Everything above and below is deterministic.
    model_note = ""
    if comparison.explained:
        report.steps["model"] = (
            "not consulted (the stored document explains this recording's structure)"
            if comparison.comparable
            else "not consulted (no structural difference was measurable)"
        )
    else:
        resolved_provider = (provider or settings.llm_provider).lower()
        problem = provider_preflight(resolved_provider)
        if problem:
            recorder.finish("error", error=problem)
            _persist_trace(recorder)
            raise RealignError(
                "model", problem, steps=report.steps,
                error_code="provider_not_configured",
            )
        try:
            result = await _timed_step(
                "model",
                lambda: reconcile(
                    stored.metadata.title,
                    stored.metadata.artist,
                    [],
                    mir,
                    provider_name=provider,
                    model=model,
                    youtube_video_id=video_id,
                    song_id=song_id,
                    audio_path=audio.path,
                    trace=recorder,
                    prior_song=document.model_dump(mode="json"),
                    depth=depth,
                    scope=AnalysisScope(listen=True, reconcile=False),
                    structure_feedback=_structure_feedback(stored, comparison, video_id),
                ),
                settings.reconcile_timeout_seconds,
            )
        except asyncio.TimeoutError as e:
            recorder.finish("error", error="model timed out")
            _persist_trace(recorder)
            raise RealignError(
                "model", f"timed out after {settings.reconcile_timeout_seconds:.0f}s",
                steps=report.steps,
            ) from e
        except Exception as e:  # noqa: BLE001
            recorder.finish("error", error=str(e)[:2000])
            _persist_trace(recorder)
            raise RealignError(
                "model", str(e), steps=report.steps,
                error_code=getattr(e, "error_code", None),
            ) from e
        report.model_consulted = True
        model_note = f"{result.provider}/{result.model} resolved the structural difference"
        report.steps["model"] = f"ok: {model_note}"
        # The model's document carries only its OWN provenance (see
        # reconcile.engine._finalize); splice the stored history back in front
        # so the version reads as one continuous history instead of starting
        # over at this run. The deterministic path needs no such splice — its
        # document IS the stored one.
        document = clear_recording_timing(
            result.song.model_copy(
                update={"provenance": base_provenance + list(result.song.provenance)}
            )
        )

    document = document.model_copy(
        update={
            "provenance": list(document.provenance)
            + [
                _realign_provenance(
                    video_id=video_id,
                    source_version=report.source_version,
                    transposition=transposition,
                    comparison=comparison,
                    model_consulted=report.model_consulted,
                    model_note=model_note,
                )
            ]
        }
    )

    # 8. the timing passes, in the order the analyze pipeline runs them.
    duration = mir.duration_seconds or audio.duration_seconds
    try:
        document = snap_chords(document, mir)
        document = document.model_copy(
            update={
                "audio": document.audio.model_copy(
                    update={
                        "analyzedVideoId": audio.video_id,
                        "youtubeVideoId": audio.video_id,
                        "durationSeconds": duration,
                    }
                )
            }
        )
        report.steps["timing"] = f"ok: {document.provenance[-1].notes}"
    except Exception as e:  # noqa: BLE001 — without new times this is pointless
        recorder.finish("error", error=str(e)[:2000])
        _persist_trace(recorder)
        raise RealignError("timing", str(e), steps=report.steps) from e

    try:
        lrc = fetch_lrc(stored.metadata.title, stored.metadata.artist, duration)
        matches = match_lrc_to_lines(lrc, document) if lrc else []
        if matches:
            document = apply_lrc(document, mir, matches)
            report.steps["lrc"] = f"ok: {len(matches)}/{len(document.lines)} line(s) matched"
        else:
            report.steps["lrc"] = "skipped (no LRCLIB match)" if lrc is None else (
                "ok: no line matched closely enough"
            )
    except Exception as e:  # noqa: BLE001 — best-effort, never fatal
        report.steps["lrc"] = _fail_text(e, settings.fetch_timeout_seconds)

    try:
        document, guard_entry = guard_against_collapsed_timing(document, duration)
        report.steps["timing-collapse-guard"] = f"ok: {guard_entry.notes}"
    except Exception as e:  # noqa: BLE001 — best-effort, never fatal
        report.steps["timing-collapse-guard"] = f"failed: {e}"

    # Sections last of the timing passes: they are spans over the LINE times
    # every pass above may still have moved. Nothing else puts a span back on a
    # section after clear_recording_timing emptied it (see retime_sections).
    try:
        document, sections_timed = retime_sections(document, duration)
        report.steps["timing-sections"] = (
            f"ok: {sections_timed}/{len(document.sections)} section span(s) derived "
            f"from the new line times"
        )
    except Exception as e:  # noqa: BLE001 — best-effort, never fatal
        report.steps["timing-sections"] = f"failed: {e}"

    try:
        # No candidate sources in a Mode B run: the source agreement signal
        # belongs to the reconciliation that produced this document, and
        # re-deriving it from nothing would only add the neutral default.
        document, scores = score_song(document, [], mir)
        recorder.set_review_queue(build_review_queue(scores))
        report.steps["confidence"] = f"ok: {len(scores)} placement(s) scored"
    except Exception as e:  # noqa: BLE001 — best-effort, never fatal
        report.steps["confidence"] = f"failed: {e}"

    # 9. the quality gate, same standard as any run. No retry: Mode B's answer
    # to a bad grade is a different RECORDING (reported below), not another
    # attempt at the same one — and the document's chords and words came from
    # the operator's own library, so there is nothing for a model to fix.
    decision = evaluate_quality(
        document,
        mir,
        [],
        can_search=False,
        can_retry=False,
        # Mode B gathers no text sources and is not supposed to: the words and
        # chords come from the document it is re-timing. Saying so keeps an
        # empty candidate list from reading as "the sources failed" — and keeps
        # the AUDIO checks reachable, which is the verdict that matters when a
        # re-alignment lands on another unusable recording.
        sources_expected=False,
    )
    report.quality = decision
    report.steps["quality"] = decision.describe()
    entries = [grade_provenance_entry(decision.grade, attribution=decision.attribution)]
    if decision.escalation.mark_timing_unreliable:
        entries.append(timing_unreliable_provenance_entry(decision.attribution))
        report.steps["timing-reliability"] = "marked unreliable (audio fault)"
        report.suggestions = await asyncio.to_thread(
            suggest_recordings,
            document,
            reason=(
                f"re-aligning {song_id!r} to {video_id} still grades as an audio fault: "
                f"{decision.attribution.reason}"
            ),
        )
        report.steps["recording-suggestions"] = report.suggestions.describe()
    document = document.model_copy(
        update={"provenance": list(document.provenance) + entries}
    )
    recorder.set_quality(
        {**decision.to_dict(), "retriesSpent": 0, "searchesSpent": 0}
    )
    recorder.finish("ok", model=recorder.trace.model or "deterministic")
    _persist_trace(recorder)

    # 10. the audio-data guard, same rule as the analyze pipeline: a re-align
    # that ends up with LESS audio-derived data than the version it started
    # from is a downgrade, and downgrades need explicit intent.
    lost = audio_data_lost(stored, document)
    if lost and not allow_timing_loss:
        raise RealignError(
            "timing-guard",
            (
                f"refusing to store the re-aligned {song_id!r}: the new recording's "
                f"analysis produced less audio-derived data than the version it "
                f"re-times — {', '.join(lost)}. The stored version is unchanged. Pick "
                f"a different recording, or set allowTimingLoss=true if the trade is "
                f"intended."
            ),
            steps=report.steps,
            error_code="timing_data_loss",
        )
    if lost:
        report.steps["timing-guard"] = f"overridden: dropping {', '.join(lost)}"

    # 11. store — a new version of the SAME song. Never a new song id: this is
    # the same song, timed against a different recording.
    try:
        saved: SaveResult = await _timed_step(
            "store",
            lambda: store.save(
                document,
                message=(
                    f"Re-align {song_id} to {video_id}"
                    + (f" ({transposition.semitones:+d} semitones)" if transposition.applies else "")
                ),
                expected_version=(
                    expected_version if expected_version is not None
                    else store.current_version(song_id)
                ),
            ),
            settings.store_timeout_seconds,
        )
    except VersionConflictError:
        raise
    except asyncio.TimeoutError as e:
        raise RealignError(
            "store", f"timed out after {settings.store_timeout_seconds:.0f}s",
            steps=report.steps,
        ) from e
    except Exception as e:  # noqa: BLE001
        raise RealignError("store", str(e), steps=report.steps) from e

    report.stored_version = saved.version
    report.stored_timestamp = saved.timestamp
    report.steps["store"] = f"ok: version {saved.version}"
    return report


def realign_song(song_id: str, video_id: str, **kwargs) -> RealignReport:
    """Synchronous wrapper for callers not already inside an event loop."""
    return asyncio.run(realign_song_async(song_id, video_id, **kwargs))
