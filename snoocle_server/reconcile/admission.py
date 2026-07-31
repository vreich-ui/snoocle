"""Admission wrapper for the standalone reconciliation REST/MCP surfaces."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import settings
from ..identity import resolve_identity_from_evidence, require_resolved_song_id
from ..manifest import build_evidence_manifest
from ..mir.cache import audio_sha256
from ..schema.song import slugify_song_id
from ..store.run_admission import get_run_admission_store, idempotency_key
from ..store.runs import get_run_store
from ..store import get_repository
from ..usage import BudgetExceededError
from ..usage_summary import enforce_admission_budgets
from .agent_config import config_version
from .engine import _load_agent_config, reconcile
from .depth import require_effort_level, resolve_depth
from .trace import TraceRecorder, new_run_id, start_run

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdmittedReconcile:
    result: object
    recorder: TraceRecorder
    evidence_manifest: dict


def reconcile_admitted(
    title: str,
    artist: str,
    candidates,
    mir,
    *,
    provider: str | None = None,
    model: str | None = None,
    audio_path: str | None = None,
    attach_audio: bool | None = None,
    youtube_video_id: str | None = None,
    media_url: str | None = None,
    force: bool = False,
    force_reason: str | None = None,
    batch_id: str | None = None,
    effort_level: str | None = None,
    prior_song: dict | None = None,
) -> AdmittedReconcile:
    if force and not (force_reason or "").strip():
        raise ValueError("forceReason is required when force=true")

    identity = resolve_identity_from_evidence(
        artist=artist,
        title=title,
        candidates=candidates,
    )
    title, artist = identity.title, identity.artist
    song_id = slugify_song_id(artist, title)
    require_resolved_song_id(song_id)
    if prior_song is None:
        try:
            stored_prior = get_repository().get(song_id)
            prior_song = stored_prior.model_dump(mode="json") if stored_prior is not None else None
        except Exception:  # noqa: BLE001 — standalone reconcile may use no song store
            prior_song = None
    resolved_provider = (provider or settings.llm_provider).lower()
    require_effort_level(effort_level)
    depth = resolve_depth(effort_level)
    manifest = build_evidence_manifest(mir=mir, candidates=candidates, prior_song=prior_song)
    resolved_agent_config = _load_agent_config()
    resolved_config_version = config_version(resolved_agent_config)
    enforce_admission_budgets(
        get_run_store(), resolved_agent_config, batch_id=batch_id
    )
    key, digest = idempotency_key(song_id, resolved_config_version, manifest)
    run_id = new_run_id()
    lock_store = get_run_admission_store()
    admission = lock_store.admit(
        key=key,
        evidence_hash=digest,
        run_id=run_id,
        lease_seconds=settings.run_lock_lease_seconds,
        completed_ttl_seconds=settings.duplicate_run_ttl_seconds,
        force=force,
        force_reason=force_reason,
    )
    recorder = start_run(
        song_id,
        resolved_provider,
        depth.name,
        run_id=run_id,
        config_version=resolved_config_version,
        idempotency_key=key,
        evidence_hash=digest,
        forced=force,
        force_reason=(force_reason or "").strip() or None,
        batch_id=batch_id,
        effort_level={"fast": "low", "thorough": "high"}.get(depth.name, "standard"),
    )
    try:
        result = reconcile(
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
            song_id=song_id,
            trace=recorder,
            evidence_manifest=manifest,
            prior_song=prior_song,
            depth=depth,
        )
        if audio_path:
            try:
                result.song = result.song.model_copy(
                    update={
                        "audio": result.song.audio.model_copy(
                            update={"contentHash": audio_sha256(audio_path)}
                        )
                    }
                )
            except OSError as error:
                log.warning("standalone reconcile could not hash audio %s: %s", audio_path, error)
    except Exception as error:
        recorder.finish(
            "budget_exceeded" if isinstance(error, BudgetExceededError) else "error",
            error=str(error)[:2000],
        )
        _persist(recorder)
        try:
            lock_store.abandon(admission)
        except Exception as cleanup_error:  # noqa: BLE001
            log.warning("run admission cleanup failed (lease will expire): %s", cleanup_error)
        raise

    recorder.finish("ok", model=result.model)
    recorder.set_evidence_manifest(manifest)
    _persist(recorder)
    lock_store.complete(
        admission,
        retry=False,
        summary={
            "runId": run_id,
            "songId": song_id,
            "status": "ok",
            "finishedAt": recorder.trace.finished_at,
            "evidenceManifest": manifest,
            "forced": recorder.trace.forced,
            "usage": recorder.trace.to_dict()["usage"],
            "costUSD": recorder.trace.cost_usd,
            "usageReliable": True,
        },
    )
    return AdmittedReconcile(result=result, recorder=recorder, evidence_manifest=manifest)


def _persist(recorder: TraceRecorder) -> None:
    try:
        get_run_store().save_run(recorder.trace.to_dict())
    except Exception as error:  # noqa: BLE001
        log.warning("run trace persistence failed (continuing): %s", error)
