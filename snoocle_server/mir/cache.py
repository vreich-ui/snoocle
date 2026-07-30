"""MIR result cache — the same audio, analyzed once.

:class:`~snoocle_server.mir.base.MirAnalysis` is a pure function of the audio
bytes, the engine implementations that ran, the accuracy profile, and (for
windowed probes) the window. Nothing else can change the answer, so nothing
else belongs in the key — and because the key covers everything, entries never
need to expire.

The motivating evidence: one song accumulated 13 stored versions, 10 of them
inside 16 minutes on the same day, each re-running the full stack over
byte-identical audio and each returning bpm=152.0, key=A major. Ten identical
executions of a pure function.

Keyed on:

- **sha256 of the audio file's bytes** — not the video id. A re-upload under
  the same id has different bytes and must MISS; two different ids that resolve
  to identical audio should HIT.
- **the engine fingerprint** — the implementation ids that land in
  ``MirAnalysis.engines`` (``chords:chord-cnn-lstm`` vs the chroma fallback,
  ``structure:songformer`` vs librosa). Swapping a model in, or losing the
  heavy one to a fallback, must invalidate rather than silently serve the other
  engine's answer.
- **the accuracy profile** plus the settings that shape it (the analysis cap,
  the fast-window geometry), since those decide what span is analyzed.
- **the window** for :func:`~snoocle_server.mir.pipeline.analyze_window`
  probes, so a probe of 72.86-105.71s caches independently of the full-track
  run and of every other probe.

Read/write key asymmetry — deliberate, and the reason a degraded engine can
never be served as a healthy one:

    lookup key  <- the engines that are PREDICTED to run (config + what's
                   installed, via :func:`predicted_engines`)
    store  key  <- the engines that ACTUALLY ran (``analysis.engines``)

In the normal case those are identical and the cache behaves as expected. When
the heavy model is present but crashes at runtime, the stack falls back and the
result is written under the fallback's key — so the next run, still predicting
the heavy engine, misses and retries it rather than being handed the chroma
answer forever. And if an operator later removes the model entirely, the
prediction becomes the fallback and those earlier fallback-keyed entries start
hitting. Self-correcting in both directions.

**The cached value is the MirAnalysis and nothing else.** A hit deserializes to
an object byte-identical to what a miss computes; cache status and the original
analysis timestamp ride alongside in :class:`MirCacheInfo`, never inside the
analysis. That is what keeps the cache semantically invisible while still
letting provenance say truthfully when the audio was actually listened to.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..config import settings
from .base import MirAnalysis

log = logging.getLogger(__name__)

#: Engine slots that participate in the key. ``sampling`` is deliberately
#: EXCLUDED: for fast accuracy it is a human-readable rendering of the chosen
#: windows, which are themselves a pure function of the audio and the
#: fast-window settings — both already in the key. Including it would make the
#: lookup key unpredictable (the windows aren't known until the audio is read),
#: which would mean fast-accuracy runs could never hit.
_KEYED_ENGINE_SLOTS = ("beats", "chords", "structure")

#: The structure slot's value on any windowed run — set by ``_analyze_windows``.
_WINDOWED_STRUCTURE = "skipped (fast accuracy)"

#: Accuracy label used for single-window probes, which have no accuracy knob of
#: their own but must not collide with a full-track run's key.
WINDOW_ACCURACY = "window"


def audio_sha256(path: str | Path) -> str:
    """Content hash of the audio file. Chunked so a long track doesn't have to
    be held in memory just to be identified."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def predicted_engines(accuracy: str) -> dict[str, str]:
    """The engine ids that WILL run, resolved from config + what's installed.

    Mirrors the health endpoint's reporting and the per-slot resolvers the
    engines themselves use, so a lookup key matches the store key whenever the
    stack behaves as configured. A runtime fallback (model present but raising)
    is the one case where prediction and reality diverge — see the module
    docstring for why that is safe.
    """
    from .beats import _madmom_available
    from .chordrec import chord_engine_id
    from .structure import _runner_path as _songformer_runner

    if accuracy in (WINDOW_ACCURACY, "fast"):
        structure = _WINDOWED_STRUCTURE
    else:
        structure = "songformer" if _songformer_runner() is not None else "librosa-agglomerative-fallback"
    return {
        "beats": "madmom" if _madmom_available() else "librosa-fallback",
        "chords": chord_engine_id(),
        "structure": structure,
    }


def engine_fingerprint(engines: dict[str, str]) -> list[str]:
    """Stable ``["slot:impl", ...]`` over the keyed slots only."""
    return [f"{slot}:{engines.get(slot, 'none')}" for slot in _KEYED_ENGINE_SLOTS]


def _accuracy_profile(accuracy: str) -> dict:
    """The settings that change WHAT SPAN gets analyzed for this accuracy.

    Only the knobs that actually apply are included, so an unrelated setting
    change can't invalidate entries it has no effect on.
    """
    if accuracy == "fast":
        return {
            "fastWindowSeconds": float(settings.mir_fast_window_seconds),
            "fastWindowCount": int(settings.mir_fast_window_count),
        }
    if accuracy == WINDOW_ACCURACY:
        return {}
    if accuracy == "thorough":
        return {}  # thorough never clips, so the cap is irrelevant
    return {"maxAnalysisSeconds": settings.mir_max_analysis_seconds}


def mir_cache_key(
    *,
    audio_hash: str,
    engines: dict[str, str],
    accuracy: str,
    window: Optional[tuple[float, float]] = None,
) -> str:
    """A hex key over every input that can change the analysis."""
    components = {
        # key-scheme version: bump to invalidate everything at once. v2 = beats
        # carry `detected` (grid continued over fade-outs), so entries written
        # before it would decode as an all-measured grid that never was.
        "v": 2,
        "audio": audio_hash,
        "engines": engine_fingerprint(engines),
        "accuracy": accuracy,
        "profile": _accuracy_profile(accuracy),
        # Rounded: window bounds come from float arithmetic on durations, and
        # a probe of the same span shouldn't miss over a 1e-12 difference.
        "window": [round(window[0], 3), round(window[1], 3)] if window else None,
    }
    canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _decode(value: object, key: str) -> Optional[MirAnalysis]:
    """Deserialize a cached blob, or None if it isn't faithfully a MirAnalysis.

    Validation alone is not enough to trust an entry: EVERY field on
    MirAnalysis has a default, so an unrelated dict validates cleanly into an
    empty-but-valid analysis — which would then be served as if the audio had
    been listened to and found silent. So the read enforces the same invariant
    the whole cache rests on: re-serializing what we decoded must reproduce
    exactly what was stored. Anything else (foreign keys dropped, a type
    coerced) means the blob was not written by this cache, and the safe
    reading of that is "miss".
    """
    if not isinstance(value, dict):
        log.warning("MIR cache: entry is not an object key=%s", key)
        return None
    try:
        analysis = MirAnalysis.model_validate(value)
    except Exception as e:  # noqa: BLE001 — a bad entry is just a miss
        log.warning("MIR cache: unreadable entry key=%s (%s)", key, e)
        return None
    if analysis.model_dump(mode="json") != value:
        log.warning("MIR cache: entry did not round-trip faithfully key=%s", key)
        return None
    return analysis


@dataclass(frozen=True)
class MirCacheInfo:
    """Cache provenance for one analysis — deliberately OUTSIDE MirAnalysis.

    Putting ``from_cache`` on the analysis itself would break the property the
    whole design rests on: a hit and a miss must produce byte-identical
    MirAnalysis objects. This rides alongside instead, so the pipeline can
    record truthfully in provenance that a result was reused, and when the
    audio was actually listened to, without the analysis differing at all.
    """

    status: str  # "hit" | "miss" | "refresh" | "disabled"
    analyzed_at: str  # ISO-8601 of the ORIGINAL analysis (== now for a miss)
    key: str = ""

    @property
    def from_cache(self) -> bool:
        return self.status == "hit"


_store = None
_lock = threading.Lock()


def get_mir_cache():
    """Process-wide MIR cache store. No TTL — the key encodes everything."""
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                from ..store.blob_cache import build_blob_cache

                _store = build_blob_cache(settings.mir_cache_collection, ttl_seconds=None)
                log.info("MIR cache backend: %s", type(_store).__name__)
    return _store


def reset_mir_cache() -> None:
    """Drop the cached store handle (tests; reconfiguration)."""
    global _store
    with _lock:
        _store = None


def analyze_cached(
    audio_path: str | Path,
    *,
    accuracy: str,
    compute: Callable[[], MirAnalysis],
    window: Optional[tuple[float, float]] = None,
    refresh: bool = False,
) -> tuple[MirAnalysis, MirCacheInfo]:
    """Run ``compute`` unless an identical analysis is already cached.

    ``compute`` is injected rather than called by name so the caller keeps
    ownership of HOW the analysis runs — the pipeline's existing
    ``analyze_audio`` symbol stays the thing that computes, which keeps every
    existing seam (and monkeypatch) working exactly as before.

    ``refresh=True`` forces recomputation and overwrites the entry: the escape
    hatch for an engine upgraded in place without its id changing, which is the
    one change the key cannot see.

    Never raises on cache trouble. A cache is an optimization; if anything
    about it misbehaves the correct outcome is the un-cached one.
    """
    if not settings.mir_cache_enabled:
        analysis = compute()
        return analysis, MirCacheInfo(status="disabled", analyzed_at=_now_iso())

    try:
        audio_hash = audio_sha256(audio_path)
    except OSError as e:
        log.warning("MIR cache: cannot hash %s (%s) — analyzing uncached", audio_path, e)
        analysis = compute()
        return analysis, MirCacheInfo(status="disabled", analyzed_at=_now_iso())

    store = get_mir_cache()
    lookup_key = mir_cache_key(
        audio_hash=audio_hash,
        engines=predicted_engines(accuracy),
        accuracy=accuracy,
        window=window,
    )

    if not refresh:
        hit = store.get(lookup_key)
        if hit is not None:
            value, stored_at = hit
            analysis = _decode(value, lookup_key)
            if analysis is not None:
                log.info("MIR cache hit key=%s analyzedAt=%s", lookup_key, stored_at)
                return analysis, MirCacheInfo(
                    status="hit", analyzed_at=stored_at, key=lookup_key
                )

    analysis = compute()

    # Store under the engines that ACTUALLY ran. Usually identical to the
    # lookup key; when they differ the stack fell back at runtime and this
    # entry belongs to the fallback, not to the engine we predicted.
    store_key = mir_cache_key(
        audio_hash=audio_hash,
        engines=analysis.engines,
        accuracy=accuracy,
        window=window,
    )
    if store_key != lookup_key:
        log.warning(
            "MIR cache: engines fell back at runtime (predicted=%s actual=%s) — "
            "storing under the fallback's key so the heavy engine is retried next run",
            engine_fingerprint(predicted_engines(accuracy)),
            engine_fingerprint(analysis.engines),
        )
    stored_at = store.put(store_key, analysis.model_dump(mode="json")) or _now_iso()
    return analysis, MirCacheInfo(
        status="refresh" if refresh else "miss", analyzed_at=stored_at, key=store_key
    )


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
