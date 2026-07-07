"""End-to-end pipeline orchestration.

discover -> acquire -> MIR -> reconcile -> versioned commit

Each stage is independently callable (the HTTP API and MCP tools expose them
separately); this module wires them together for the one-call flow and is
deliberately tolerant of partial failure: a song can still be produced from
text sources alone if audio/MIR is unavailable (recorded in provenance), or
from MIR alone if the web is unreachable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .audio.acquire import AcquiredAudio, AcquisitionError, acquire
from .config import settings
from .discovery import CandidateSource, discover_sources
from .discovery.search import SearchError
from .mir import MirAnalysis, analyze_audio
from .reconcile import ReconcileResult, reconcile
from .schema.song import slugify_song_id
from .store import GitSongStore

log = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    song_id: str
    steps: dict[str, str] = field(default_factory=dict)  # step -> "ok" | error text
    candidates: list[CandidateSource] = field(default_factory=list)
    audio: AcquiredAudio | None = None
    mir: MirAnalysis | None = None
    reconcile: ReconcileResult | None = None
    stored_version: str | None = None


def get_store() -> GitSongStore:
    return GitSongStore(settings.store_dir)


def run_pipeline(
    title: str,
    artist: str,
    youtube_url_or_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    attach_audio: bool | None = None,
    skip_audio: bool = False,
    max_candidates: int | None = None,
    expected_version: str | None = None,
    store: GitSongStore | None = None,
) -> PipelineReport:
    song_id = slugify_song_id(artist, title)
    report = PipelineReport(song_id=song_id)

    # 1-3. text-source discovery (gather generously; keep sources separate)
    try:
        report.candidates = discover_sources(title, artist, max_candidates=max_candidates)
        report.steps["discover"] = f"ok: {len(report.candidates)} candidate source(s)"
    except (SearchError, Exception) as e:  # noqa: BLE001
        report.steps["discover"] = f"failed: {e}"
        log.warning("discovery failed: %s", e)

    # 4. audio acquisition + MIR analysis
    if not skip_audio:
        try:
            report.audio = acquire(title=title, artist=artist, video_url_or_id=youtube_url_or_id)
            report.steps["acquire"] = f"ok: {report.audio.video_id} ({report.audio.video_title})"
        except AcquisitionError as e:
            report.steps["acquire"] = f"failed: {e}"
            log.warning("audio acquisition failed: %s", e)
        if report.audio is not None:
            try:
                report.mir = analyze_audio(report.audio.path)
                report.steps["mir"] = "ok: engines=" + str(report.mir.engines)
            except Exception as e:  # noqa: BLE001
                report.steps["mir"] = f"failed: {e}"
                log.warning("MIR analysis failed: %s", e)
    else:
        report.steps["acquire"] = "skipped"
        report.steps["mir"] = "skipped"

    # 5. reconciliation (uses ALL candidates + MIR timeline)
    result = reconcile(
        title,
        artist,
        report.candidates,
        report.mir,
        provider_name=provider,
        model=model,
        audio_path=report.audio.path if report.audio else None,
        attach_audio=attach_audio,
        youtube_video_id=report.audio.video_id if report.audio else None,
        song_id=song_id,
    )
    report.reconcile = result
    report.steps["reconcile"] = (
        f"ok: provider={result.provider} model={result.model} attempts={result.attempts}"
    )

    # 7. version-controlled persistence — every run is a new commit
    store = store or get_store()
    prior = store.current_version(song_id)
    if prior is not None:
        # append-only provenance: extend the stored history with this run's entries
        stored = store.get(song_id)
        merged = list(stored.provenance) + list(result.song.provenance)
        result.song = result.song.model_copy(update={"provenance": merged})
    saved = store.save(
        result.song,
        message=(
            f"{'Re-analyze' if prior else 'Analyze'} {song_id} "
            f"[{result.provider}/{result.model}]"
        ),
        expected_version=expected_version if expected_version is not None else prior,
    )
    report.stored_version = saved.version
    report.steps["store"] = f"ok: version {saved.version[:12]}"
    return report
