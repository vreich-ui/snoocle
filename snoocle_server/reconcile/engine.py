"""Reconciliation engine: prompt -> LLM -> splice -> validate -> repair -> provenance.

Schema compliance and the chord-normalization rule are ENFORCED here, not
hoped for: the LLM's JSON must validate against the Song schema (whose
validators reject shape/tab chords outright). Validation errors are fed back
to the model for up to SNOOCLE_LLM_REPAIR_ATTEMPTS repair rounds.

Lyrics are the same story. A model-backed provider does not emit them at
all: it emits REFERENCES into the sources already in this run's context, and
``lyric_refs.splice_lyrics`` substitutes the real text here, before
validation. See that module for why (a Song document IS the complete lyrics
of a copyrighted song) and for the four rules that make it a guarantee
rather than a request.

A targeted correction in notes-only scope is a THIRD path, not a smaller
version of the first two: the model doesn't reconcile or reference anything,
it names a short list of OPERATIONS against the prior document
(``patch_ops.py``), and this module applies them in local code. Unlike a
schema-validation slip or an unresolvable lyric ref, a patch failure is
never retried — see ``_attempt_patch``'s docstring for why.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .. import __version__
from ..audio.utils import probe, trim
from ..config import settings
from ..discovery.models import CandidateSource
from ..mir.base import MirAnalysis
from ..schema import Song, song_json_schema
from ..schema.song import ProvenanceEntry, slugify_song_id
from ..scope import AnalysisScope
from .agent_config import AgentConfig, config_version
from .depth import DepthProfile, resolve_depth
from .lyric_refs import (
    LyricOverride,
    LyricSpliceError,
    UnresolvableLyricRefError,
    agent_song_json_schema,
    build_ref_index,
    splice_lyrics,
)
from .patch_ops import AppliedOp, PatchError, apply_patch, parse_ops_response
from .prompt import (
    build_patch_system_prompt,
    build_patch_user_prompt,
    build_repair_prompt,
    build_system_prompt,
    build_user_prompt,
)
from .providers import AudioAttachment, LLMProvider, get_provider
from .trace import RunTrace, TraceRecorder, clock

log = logging.getLogger(__name__)


def _load_agent_config() -> AgentConfig:
    """The stored operator config, or built-in defaults if unset/unreadable."""
    try:
        from ..store.agent_config import get_agent_config_store

        doc = get_agent_config_store().get()
        return AgentConfig.model_validate(doc) if doc else AgentConfig()
    except Exception as e:  # noqa: BLE001 — never fail a run over config
        log.warning("agent config unavailable, using defaults: %s", e)
        return AgentConfig()

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class ReconcileError(RuntimeError):
    # Optional machine-readable classification the pipeline surfaces to clients.
    error_code: str | None = None


class LyricProvenanceError(ReconcileError):
    """A lyric reached the document with no valid provenance: a ref this run
    cannot resolve, or so many overrides that reference resolution is plainly
    broken. Deliberately NOT repairable — see lyric_refs.py, rules 3 and 4."""

    error_code = "lyric_provenance"


class PatchApplicationError(ReconcileError):
    """A patch op didn't apply cleanly, or the response wasn't a valid patch
    at all (unknown op, over the cap, malformed). Deliberately NOT repaired
    — see patch_ops.py's module docstring and ``_attempt_patch`` below: a
    wrong op is not the kind of mistake a retry fixes, and "try again, maybe
    you'll guess differently" is exactly the fuzziness this path exists to
    refuse. Distinct from the model explicitly declining the patch path
    (``needsFullReconcile``), which is not an error at all."""

    error_code = "patch_failed"


@dataclass
class ReconcileResult:
    song: Song
    provider: str
    model: str
    attempts: int
    audio_attached: bool
    usage: dict = field(default_factory=dict)
    trace: RunTrace | None = None  # the run's step-by-step logic (if recorded)
    # >0 only when the patch path (patch_ops.py) produced this song: the
    # pipeline reads this to skip timing carry-forward/snap/LRC entirely —
    # nothing was regenerated, so there is nothing for those passes to do
    # except risk disturbing what a patch deliberately left alone.
    patch_ops_applied: int = 0


def extract_json(text: str) -> str:
    """Pull the JSON document out of an LLM response (tolerate fences/preamble)."""
    text = _FENCE_RE.sub("", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return text[start : end + 1]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _make_snippet(audio_path: str) -> AudioAttachment | None:
    """A short mid-song excerpt (30s from 25% in) as mp3 for providers that
    accept audio input. Failure here never fails the reconciliation."""
    try:
        info = probe(audio_path)
        start = max(info.duration_seconds * 0.25, 0.0)
        end = min(start + 30.0, info.duration_seconds)
        out = Path(tempfile.mkdtemp(prefix="snoocle-snippet-")) / "snippet.mp3"
        trim(audio_path, out, start, end)
        return AudioAttachment(path=str(out))
    except Exception as e:  # noqa: BLE001
        log.warning("audio snippet preparation failed (continuing without): %s", e)
        return None


def _finalize(
    song: Song,
    *,
    song_id: str,
    title: str,
    artist: str,
    youtube_video_id: str | None,
    candidates: list[CandidateSource],
    mir: MirAnalysis | None,
    provider: LLMProvider,
    model: str,
    attempts: int,
    guidance: str | None = None,
    guidance_origin: str | None = None,
    scope: AnalysisScope | None = None,
    mir_cache=None,
    lyric_refs_resolved: int = 0,
    lyric_overrides: tuple[LyricOverride, ...] = (),
    patch_ops_applied: tuple[AppliedOp, ...] = (),
    quality_feedback: str | None = None,
    structure_feedback: str | None = None,
) -> Song:
    """Server-side guardrails + append provenance (never trusted to the LLM)."""
    updates: dict = {"id": song_id, "provenance": []}
    if song.displayPreferences.capo != 0:
        log.warning("reconciler set capo=%d; forcing display capo to 0", song.displayPreferences.capo)
        updates["displayPreferences"] = song.displayPreferences.model_copy(update={"capo": 0})
    md_updates = {}
    if song.metadata.title != title:
        md_updates["title"] = title
    if song.metadata.artist != artist:
        md_updates["artist"] = artist
    if md_updates:
        updates["metadata"] = song.metadata.model_copy(update=md_updates)
    if youtube_video_id and song.audio.youtubeVideoId != youtube_video_id:
        updates["audio"] = song.audio.model_copy(update={"youtubeVideoId": youtube_video_id})
    song = song.model_copy(update=updates)

    prov: list[ProvenanceEntry] = []
    if candidates:
        prov.append(
            ProvenanceEntry(
                timestamp=_now(),
                actor=f"snoocle-server/{__version__}",
                action="discovered-sources",
                sources=[c.url or c.sourceId for c in candidates],
                confidence=round(max(c.confidence for c in candidates), 3),
                notes=f"{len(candidates)} candidate text source(s) gathered via general web search",
            )
        )
    if mir is not None:
        # `timestamp` is when THIS version was produced; `analyzedAt` in the
        # notes is when the audio was actually listened to. On a cache hit
        # those differ, and collapsing them would make a song's history claim
        # a re-listen that never happened. Reading a song must still tell you
        # when its audio was really analyzed.
        reuse_note = ""
        if mir_cache is not None and getattr(mir_cache, "from_cache", False):
            reuse_note = f"; reused from cache (analyzedAt={mir_cache.analyzed_at})"
        prov.append(
            ProvenanceEntry(
                timestamp=_now(),
                actor=f"snoocle-server/{__version__}",
                action="mir-analysis",
                sources=[f"{slot}:{impl}" for slot, impl in mir.engines.items()],
                notes=f"audio-grounded analysis; bpm={mir.bpm}, key={mir.key}{reuse_note}",
            )
        )
    # more independent sources -> higher reconciliation confidence
    conf = min(0.45 + 0.1 * min(len(candidates), 3) + (0.15 if mir else 0.0), 0.9)
    if provider.name == "mock":
        conf = min(conf, 0.5)
    # Guidance is never applied silently: a reader of the song's history must be
    # able to see that a human instruction shaped this run, and whether it came
    # from the request or replayed from the song's stored notes.
    guidance_note = ""
    if guidance:
        guidance_note = f"; guidance applied (from {guidance_origin or 'this request'})"
    # Same reasoning for scope: a version produced from the prior song + notes
    # alone is a very different artifact from a full re-analysis, and six
    # months later the history is the only place that difference survives.
    scope_note = f"; scope: {scope.describe()}" if scope is not None else ""
    # How the words got here. A reader of the history should be able to tell
    # "every line was spliced from a source the run actually had" from "some
    # lines were written by the model", without diffing anything.
    lyric_note = ""
    overrides = lyric_overrides
    if lyric_refs_resolved or overrides:
        lyric_note = (
            f"; lyrics spliced from {lyric_refs_resolved} source reference(s)"
            + (f" with {len(overrides)} override(s)" if overrides else "")
        )
    patch_note = (
        f"; patch: {len(patch_ops_applied)} op(s) applied" if patch_ops_applied else ""
    )
    # A retry driven by the quality grader is a different artifact from a first
    # attempt, and the history is the only place that difference survives. It is
    # NOT reported as guidance: no human asked for it.
    quality_note = (
        "; quality retry: the previous attempt's grade and its specific failures "
        "were handed back to the model"
        if quality_feedback
        else ""
    )
    # Mode B consulted the model, which it only does when the deterministic
    # structural comparison found something the stored document cannot explain.
    # Worth its own clause: "this version's repeats were decided by a model"
    # is exactly the kind of thing a reader of the history needs to know.
    structure_note = (
        "; structural re-align: a measured difference between the source document "
        "and the new recording's arrangement was handed to the model"
        if structure_feedback
        else ""
    )
    prov.append(
        ProvenanceEntry(
            timestamp=_now(),
            actor=f"reconcile:{provider.name}/{model}",
            action="reconciled",
            sources=[c.sourceId for c in candidates],
            confidence=round(conf, 3),
            notes=(
                f"attempt(s)={attempts}; chord rule enforced by schema validation"
                + guidance_note
                + scope_note
                + lyric_note
                + patch_note
                + quality_note
                + structure_note
            ),
        )
    )
    # One entry per override, so the escape hatch is auditable line by line
    # rather than a count buried in a summary.
    for override in overrides:
        prov.append(
            ProvenanceEntry(
                timestamp=_now(),
                actor=f"reconcile:{provider.name}/{model}",
                action="lyric-override",
                confidence=None,
                notes=(
                    f"line {override.line_index}: written by the model instead "
                    f"of referenced — {override.reason}"
                ),
            )
        )
    # Same reasoning, one entry per op: a correction history has to be
    # readable op by op, not collapsed into a count.
    for applied in patch_ops_applied:
        prov.append(
            ProvenanceEntry(
                timestamp=_now(),
                actor=f"reconcile:{provider.name}/{model}",
                action="patch-applied",
                confidence=None,
                notes=(
                    applied.description
                    + (f"; reason: {applied.reason}" if applied.reason else "")
                ),
            )
        )
    return song.model_copy(update={"provenance": prov})


def _attempt_patch(
    provider: LLMProvider,
    model: str | None,
    title: str,
    artist: str,
    song_id: str,
    prior_song: dict,
    prior_song_obj: Song,
    guidance: str,
    trace: TraceRecorder | None,
) -> tuple[Song, list[AppliedOp], str, dict] | None:
    """One model call, patch-shaped. Returns ``(song, applied_ops,
    resolved_model, usage)`` on success.

    Returns ``None`` when the model explicitly declined the patch path
    (``needsFullReconcile`` — the correction genuinely needs different words,
    or is otherwise bigger than a targeted fix): the caller falls through to
    the normal reconcile prompt, in the SAME run, rather than failing it.
    That is the visible fallback the brief asks for — not a silent one, and
    not a failure.

    Raises :class:`PatchApplicationError` for anything that actually went
    wrong (malformed JSON, an op that doesn't apply, an unknown op, over the
    cap). Never retried: seeing a WRONG op and being asked to guess again is
    the fuzziness this whole path exists to refuse. One call, one verdict.
    """
    system_prompt = build_patch_system_prompt()
    user_prompt = build_patch_user_prompt(title, artist, song_id, prior_song, guidance)
    started = clock()
    response = provider.complete(system_prompt, [{"role": "user", "text": user_prompt}], model=model)
    usage: dict = {}
    for k, v in (response.usage or {}).items():
        if isinstance(v, (int, float)):
            usage[k] = usage.get(k, 0) + v
    resolved_model = response.model

    try:
        document = json.loads(extract_json(response.text))
    except (ValueError, json.JSONDecodeError) as e:
        if trace is not None:
            trace.step(
                "patch", "patch-parse-failed", f"patch response was not valid JSON: {e}",
                detail={"errors": str(e)[:2000]}, duration_seconds=clock() - started,
            )
        raise PatchApplicationError(f"patch response was not valid JSON: {e}") from e

    if isinstance(document, dict) and document.get("needsFullReconcile"):
        reason = document.get("reason") or "the model reported it needs a full reconcile"
        if trace is not None:
            trace.step(
                "patch", "needs-full-reconcile",
                f"falling back to full reconcile: {reason}",
                detail={"reason": reason}, duration_seconds=clock() - started,
            )
        return None

    try:
        ops = parse_ops_response(document)
        patched_song, applied = apply_patch(prior_song_obj, ops)
    except PatchError as e:
        if trace is not None:
            trace.step(
                "patch", "patch-failed", str(e),
                detail={"errors": str(e)[:2000]}, duration_seconds=clock() - started,
            )
        raise PatchApplicationError(str(e)) from e

    if trace is not None:
        trace.step(
            "patch", "patch-applied", f"{len(applied)} op(s) applied",
            detail={"ops": [a.description for a in applied]},
            duration_seconds=clock() - started,
        )
    return patched_song, applied, resolved_model, usage


def reconcile(
    title: str,
    artist: str,
    candidates: list[CandidateSource],
    mir: MirAnalysis | None,
    provider_name: str | None = None,
    model: str | None = None,
    audio_path: str | None = None,
    attach_audio: bool | None = None,
    youtube_video_id: str | None = None,
    song_id: str | None = None,
    media_url: str | None = None,
    trace: TraceRecorder | None = None,
    guidance: str | None = None,
    guidance_origin: str | None = None,
    prior_song: dict | None = None,
    depth: DepthProfile | None = None,
    scope: AnalysisScope | None = None,
    evidence_manifest: dict | None = None,
    mir_cache=None,
    patch_ops_eligible: bool = False,
    quality_feedback: str | None = None,
    structure_feedback: str | None = None,
) -> ReconcileResult:
    song_id = song_id or slugify_song_id(artist, title)
    provider = get_provider(provider_name)
    depth = depth or resolve_depth(None)

    # Operator agent config (runtime-editable instructions/tooling). A store
    # failure degrades to built-in defaults — observability config must never
    # fail a reconciliation.
    agent_config = _load_agent_config()
    if trace is not None:
        trace.trace.config_version = config_version(agent_config)

    # The mock provider is a deterministic offline reconciler: it can synthesize
    # a small Song from title/artist alone, so it never requires inputs. Every
    # other provider needs something concrete to reconcile — and a PRIOR SONG
    # counts: the notes-only scope (listen=off, reconcile=off) deliberately
    # gathers nothing and asks the model to correct what the user already has.
    # Without a prior song that scope would have the model invent a song from
    # the title alone, which is exactly the hallucination this guard exists to
    # prevent, so it still fails here.
    if not candidates and mir is None and prior_song is None and provider.name != "mock":
        raise ReconcileError(
            "nothing to reconcile: no candidate sources and no MIR analysis"
            + (" and no prior song to correct" if scope is not None else "")
        )

    if media_url is None and youtube_video_id:
        media_url = f"https://www.youtube.com/watch?v={youtube_video_id}"

    # The patch protocol (patch_ops.py): a targeted correction in notes-only
    # scope doesn't reconcile anything — it names exact changes, and this
    # replaces the WHOLE reconcile prompt with a much smaller, closed-set
    # ask. `supports_patch_ops` mirrors `emits_lyric_refs`: the contract is
    # between this server's prompt and a model actually told it, and
    # providers with their own prompt pipeline (the in-process agent, which
    # ignores the `system` argument entirely and builds its own) or none at
    # all (mock) are not party to it.
    prior_song_obj: Song | None = None
    if prior_song:
        try:
            prior_song_obj = Song.model_validate(prior_song)
        except Exception:  # noqa: BLE001 — a malformed prior can't be patched;
            prior_song_obj = None  # demote to the full-reconcile path below.
    uses_patch_ops = bool(
        scope is not None and scope.notes_only
        and patch_ops_eligible
        and prior_song_obj is not None
        and getattr(provider, "supports_patch_ops", False)
    )

    if uses_patch_ops:
        if trace is not None:
            trace.step(
                "inputs", "read-inputs",
                f"patch mode: correcting {song_id!r} in place; human corrections attached",
                detail={"provider": provider.name, "mode": "patch", "guidance": guidance},
            )
        attempt = _attempt_patch(
            provider, model, title, artist, song_id, prior_song, prior_song_obj,
            guidance or "", trace,
        )
        if attempt is not None:
            patched_song, applied_ops, resolved_model, usage = attempt
            song = _finalize(
                patched_song,
                song_id=song_id,
                title=title,
                artist=artist,
                youtube_video_id=youtube_video_id,
                candidates=[],
                mir=None,
                provider=provider,
                model=resolved_model,
                attempts=1,
                guidance=guidance,
                guidance_origin=guidance_origin,
                scope=scope,
                patch_ops_applied=tuple(applied_ops),
            )
            if trace is not None:
                trace.step(
                    "final", "reconciled",
                    f"produced Song via patch: {len(applied_ops)} op(s) applied",
                    detail={"opsApplied": len(applied_ops), "usage": usage},
                )
            return ReconcileResult(
                song=song,
                provider=provider.name,
                model=resolved_model,
                attempts=1,
                audio_attached=False,
                usage=usage,
                trace=trace.trace if trace is not None else None,
                patch_ops_applied=len(applied_ops),
            )
        # else: the model explicitly asked for a full reconcile. Falls
        # through to EXACTLY today's notes-only path below (full-Song
        # regeneration via the lyric-reference protocol) — visibly (the
        # trace step _attempt_patch already recorded), not silently.

    # The lyric-reference protocol (lyric_refs.py). `emits_lyric_refs` is a
    # PROVIDER capability, not a global switch: the contract is between this
    # server's prompt and the model that reads it, so a provider opts in only
    # once it is actually told the contract. The deterministic mock builds its
    # Song in local code and is never prompted at all.
    uses_lyric_refs = bool(getattr(provider, "emits_lyric_refs", False))
    ref_index = build_ref_index(candidates, prior_song) if uses_lyric_refs else {}
    base_schema = song_json_schema()
    schema_for_model = agent_song_json_schema(base_schema) if uses_lyric_refs else base_schema

    if trace is not None:
        # The FULL MIR snapshot rides on the run itself (un-truncated; the GUI
        # timeline renders from it). The inputs step keeps only a compact
        # summary — putting the whole timeline in a step detail would silently
        # hit the 50-item truncation cap.
        if mir is not None:
            trace.attach_mir(mir.to_run_payload())
        trace.step(
            "inputs", "read-inputs",
            f"{len(candidates)} text source(s), "
            + (f"MIR key={mir.key} bpm={mir.bpm}" if mir else "no MIR")
            + (f"; human corrections attached" if guidance else "")
            + ("; quality retry" if quality_feedback else "")
            + ("; structural re-align" if structure_feedback else ""),
            detail={
                "provider": provider.name,
                "depth": depth.name,
                "qualityRetry": bool(quality_feedback),
                "structuralRealign": bool(structure_feedback),
                "candidateSources": [c.url or c.sourceId for c in candidates],
                "mir": (
                    {
                        "engines": mir.engines,
                        "estimatedKey": mir.key,
                        "bpm": mir.bpm,
                        "durationSeconds": mir.duration_seconds,
                        "chordSegments": len(mir.chords),
                        "beatCount": len(mir.beats),
                        "analyzedWindows": [
                            {"start": w.start, "end": w.end} for w in mir.analyzed_windows
                        ],
                    }
                    if mir is not None
                    else None
                ),
                "guidance": guidance,
                "scope": scope.describe() if scope is not None else None,
            },
        )

    # Context-driven providers (mock, agent) consume the structured inputs
    # directly instead of the rendered prompt text. The trace + depth overrides
    # ride along so an agentic provider records its own turns/tool calls into
    # this run's timeline and honors the requested effort/budget.
    if getattr(provider, "wants_context", False):
        provider.context = {
            "title": title,
            "artist": artist,
            "song_id": song_id,
            "youtube_video_id": youtube_video_id,
            "media_url": media_url,
            "candidates": candidates,
            "mir": mir,
            "song_schema": schema_for_model,
            "ref_index": ref_index,
            "audio_path": audio_path,
            "guidance": guidance,
            "prior_song": prior_song,
            "depth": depth,
            "scope": scope,
            "agent_config": agent_config if not agent_config.is_default() else None,
            "evidence_manifest": evidence_manifest,
            "quality_feedback": quality_feedback,
            "structure_feedback": structure_feedback,
        }
    if hasattr(provider, "trace"):
        provider.trace = trace

    audio: AudioAttachment | None = None
    attach = settings.llm_audio_snippet if attach_audio is None else attach_audio
    if attach and audio_path and provider.supports_audio:
        audio = _make_snippet(audio_path)

    user_prompt = build_user_prompt(
        title, artist, candidates, mir, schema_for_model, song_id, youtube_video_id,
        guidance=guidance, prior_song=prior_song, time_align=depth.time_align,
        scope=scope, evidence_manifest=evidence_manifest,
        ref_index=ref_index if uses_lyric_refs else None,
        quality_feedback=quality_feedback,
        structure_feedback=structure_feedback,
    )
    system_prompt = build_system_prompt(lyric_refs=uses_lyric_refs)
    turns: list[dict] = [{"role": "user", "text": user_prompt}]

    usage: dict = {}
    resolved_model = model or settings.llm_model or provider.default_model
    last_errors = ""
    for attempt in range(1, settings.llm_repair_attempts + 2):
        started = clock()
        response = provider.complete(
            system_prompt, turns, model=model, audio=audio if attempt == 1 else None
        )
        for k, v in (response.usage or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
        resolved_model = response.model
        overrides: tuple[LyricOverride, ...] = ()
        refs_resolved = 0
        try:
            document = json.loads(extract_json(response.text))
            if uses_lyric_refs:
                # Splice BEFORE schema validation: the Song schema knows
                # nothing about refs, and its charIndex rule only means
                # something once the real text is in place.
                spliced = splice_lyrics(document, ref_index)
                document = spliced.document
                overrides = spliced.overrides
                refs_resolved = spliced.refs_resolved
            song = Song.model_validate(document)
        except UnresolvableLyricRefError as e:
            # NOT repaired. A lyric with no valid provenance must not reach
            # the store, and a retry is an invitation to supply one from
            # memory instead — see lyric_refs.py rules 3 and 4.
            if trace is not None:
                trace.step(
                    "repair", f"lyric-refs-attempt-{attempt}",
                    "unresolvable lyric reference — failing the run",
                    detail={"errors": str(e)[:2000]},
                    duration_seconds=clock() - started,
                )
            raise LyricProvenanceError(str(e)) from e
        except (ValidationError, ValueError, json.JSONDecodeError, LyricSpliceError) as e:
            last_errors = str(e)[:4000]
            log.info("reconcile attempt %d failed validation: %s", attempt, last_errors[:300])
            if trace is not None:
                trace.step(
                    "repair", f"validate-attempt-{attempt}",
                    f"attempt {attempt} failed schema validation — repairing",
                    detail={"errors": last_errors[:2000]},
                    duration_seconds=clock() - started,
                )
            turns.append({"role": "assistant", "text": response.text})
            turns.append(
                {"role": "user", "text": build_repair_prompt(last_errors, uses_lyric_refs)}
            )
            continue
        song = _finalize(
            song,
            song_id=song_id,
            title=title,
            artist=artist,
            youtube_video_id=youtube_video_id,
            candidates=candidates,
            mir=mir,
            provider=provider,
            model=resolved_model,
            attempts=attempt,
            guidance=guidance,
            guidance_origin=guidance_origin,
            scope=scope,
            mir_cache=mir_cache,
            lyric_refs_resolved=refs_resolved,
            lyric_overrides=overrides,
            quality_feedback=quality_feedback,
            structure_feedback=structure_feedback,
        )
        if trace is not None:
            trace.step(
                "final", "reconciled",
                f"produced Song: {len(song.lines)} lines, "
                f"{sum(len(l.chordPlacements) for l in song.lines)} chords, "
                f"{len(song.sections)} sections (attempt {attempt})",
                detail={
                    "attempts": attempt,
                    "lineCount": len(song.lines),
                    "chordCount": sum(len(l.chordPlacements) for l in song.lines),
                    "syncMapPoints": len(song.audio.syncMap),
                    "lyricRefsResolved": refs_resolved if uses_lyric_refs else None,
                    "lyricOverrides": [
                        {"lineIndex": o.line_index, "reason": o.reason} for o in overrides
                    ],
                    "usage": usage,
                },
                duration_seconds=clock() - started,
            )
        return ReconcileResult(
            song=song,
            provider=provider.name,
            model=resolved_model,
            attempts=attempt,
            audio_attached=audio is not None,
            usage=usage,
            trace=trace.trace if trace is not None else None,
        )

    raise ReconcileError(
        f"reconciliation failed schema validation after "
        f"{settings.llm_repair_attempts + 1} attempts; last errors: {last_errors[:1000]}"
    )
