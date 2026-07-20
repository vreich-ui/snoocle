"""Reconciliation engine: prompt -> LLM -> validate -> repair -> provenance.

Schema compliance and the chord-normalization rule are ENFORCED here, not
hoped for: the LLM's JSON must validate against the Song schema (whose
validators reject shape/tab chords outright). Validation errors are fed back
to the model for up to SNOOCLE_LLM_REPAIR_ATTEMPTS repair rounds.
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
from .agent_config import AgentConfig, config_version
from .depth import DepthProfile, resolve_depth
from .prompt import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt
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
    pass


@dataclass
class ReconcileResult:
    song: Song
    provider: str
    model: str
    attempts: int
    audio_attached: bool
    usage: dict = field(default_factory=dict)
    trace: RunTrace | None = None  # the run's step-by-step logic (if recorded)


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
        prov.append(
            ProvenanceEntry(
                timestamp=_now(),
                actor=f"snoocle-server/{__version__}",
                action="mir-analysis",
                sources=[f"{slot}:{impl}" for slot, impl in mir.engines.items()],
                notes=f"audio-grounded analysis; bpm={mir.bpm}, key={mir.key}",
            )
        )
    # more independent sources -> higher reconciliation confidence
    conf = min(0.45 + 0.1 * min(len(candidates), 3) + (0.15 if mir else 0.0), 0.9)
    if provider.name == "mock":
        conf = min(conf, 0.5)
    prov.append(
        ProvenanceEntry(
            timestamp=_now(),
            actor=f"reconcile:{provider.name}/{model}",
            action="reconciled",
            sources=[c.sourceId for c in candidates],
            confidence=round(conf, 3),
            notes=f"attempt(s)={attempts}; chord rule enforced by schema validation",
        )
    )
    return song.model_copy(update={"provenance": prov})


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
    prior_song: dict | None = None,
    depth: DepthProfile | None = None,
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
    # other provider needs something concrete to reconcile.
    if not candidates and mir is None and provider.name != "mock":
        raise ReconcileError("nothing to reconcile: no candidate sources and no MIR analysis")

    if media_url is None and youtube_video_id:
        media_url = f"https://www.youtube.com/watch?v={youtube_video_id}"

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
            + (f"; human corrections attached" if guidance else ""),
            detail={
                "provider": provider.name,
                "depth": depth.name,
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
            "song_schema": song_json_schema(),
            "audio_path": audio_path,
            "guidance": guidance,
            "prior_song": prior_song,
            "depth": depth,
            "agent_config": agent_config if not agent_config.is_default() else None,
        }
    if hasattr(provider, "trace"):
        provider.trace = trace

    audio: AudioAttachment | None = None
    attach = settings.llm_audio_snippet if attach_audio is None else attach_audio
    if attach and audio_path and provider.supports_audio:
        audio = _make_snippet(audio_path)

    user_prompt = build_user_prompt(
        title, artist, candidates, mir, song_json_schema(), song_id, youtube_video_id,
        guidance=guidance, prior_song=prior_song, time_align=depth.time_align,
    )
    turns: list[dict] = [{"role": "user", "text": user_prompt}]

    usage: dict = {}
    resolved_model = model or settings.llm_model or provider.default_model
    last_errors = ""
    for attempt in range(1, settings.llm_repair_attempts + 2):
        started = clock()
        response = provider.complete(
            SYSTEM_PROMPT, turns, model=model, audio=audio if attempt == 1 else None
        )
        for k, v in (response.usage or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
        resolved_model = response.model
        try:
            song = Song.model_validate_json(extract_json(response.text))
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
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
            turns.append({"role": "user", "text": build_repair_prompt(last_errors)})
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
