"""LRCLIB synced-lyrics engine — the first, most reliable layer in the
timing pipeline (master plan D3: LRCLIB > forced alignment > MIR inference
> agent judgment).

lrclib.net (https://lrclib.net) is a free, no-API-key community database of
line-synced lyrics (LRC format). When a song has a match there, its line
timestamps come from an actual karaoke/lyric-sync file rather than being
inferred from chord recognition — strictly better ground truth for "when
does this line start", even though it says nothing about chords.

Both HTTP calls and the whole engine are best-effort: any failure (network,
no match, disabled via config) returns None/skips, never raises into the
pipeline. This mirrors the MIR engines' own fallback honesty (see
mir/*.py) — the pipeline must always be able to say "this evidence source
wasn't available" and continue.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Optional

import httpx

from ..config import settings
from ..mir.base import MirAnalysis
from ..schema.song import ProvenanceEntry, Song, SyncPoint
from .quantize import DEFAULT_SNAP_TOLERANCE_SECONDS, build_beat_grid, snap_time

log = logging.getLogger(__name__)

_LRCLIB_BASE = "https://lrclib.net/api"
_TIMEOUT_SECONDS = 10.0
_DURATION_TOLERANCE_SECONDS = 3.0
LINE_MATCH_THRESHOLD = 0.75

_LRC_LINE_RE = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)$")
_USER_AGENT = "snoocle-server (personal use; https://github.com/vreich-ui/snoocle)"


@dataclass
class LrcLine:
    time: float
    text: str


@dataclass(frozen=True)
class LrcMatch:
    """A lyrics match plus the catalogue identity that named it.

    Timing callers use ``lines``; identity resolution uses ``track_name`` and
    ``artist_name`` before it ever mints a song id.
    """

    track_name: str
    artist_name: str
    lines: list[LrcLine]


def _parse_synced_lyrics(raw: str) -> list[LrcLine]:
    lines: list[LrcLine] = []
    for raw_line in raw.splitlines():
        m = _LRC_LINE_RE.match(raw_line.strip())
        if not m:
            continue
        minutes, seconds, text = m.groups()
        text = text.strip()
        if text:
            lines.append(LrcLine(time=int(minutes) * 60 + float(seconds), text=text))
    lines.sort(key=lambda l: l.time)
    return lines


def fetch_lrc_match(
    title: str, artist: str, duration_seconds: Optional[float]
) -> Optional[LrcMatch]:
    """Best-effort: None on ANY failure — network error, disabled via
    config, no exact match, or no synced-lyrics fallback match either.
    Callers never need to distinguish why; they just skip this evidence."""
    if not settings.lrclib_enabled:
        return None
    headers = {"User-Agent": _USER_AGENT}
    try:
        params: dict = {"track_name": title, "artist_name": artist}
        if duration_seconds:
            params["duration"] = round(duration_seconds)
        r = httpx.get(f"{_LRCLIB_BASE}/get", params=params, timeout=_TIMEOUT_SECONDS, headers=headers)
        if r.status_code == 200:
            payload = r.json() or {}
            synced = payload.get("syncedLyrics")
            if synced:
                parsed = _parse_synced_lyrics(synced)
                if parsed:
                    return LrcMatch(
                        track_name=str(payload.get("trackName") or title).strip(),
                        artist_name=str(payload.get("artistName") or artist).strip(),
                        lines=parsed,
                    )

        # /api/get is an exact (title, artist, duration) lookup; fall back to
        # /api/search and pick the closest-duration synced-lyrics result.
        r = httpx.get(
            f"{_LRCLIB_BASE}/search",
            params={"track_name": title, "artist_name": artist},
            timeout=_TIMEOUT_SECONDS,
            headers=headers,
        )
        if r.status_code != 200:
            return None
        candidates = [c for c in (r.json() or []) if c.get("syncedLyrics")]
        if duration_seconds:
            candidates = [
                c
                for c in candidates
                if c.get("duration")
                and abs(c["duration"] - duration_seconds) <= _DURATION_TOLERANCE_SECONDS
            ]
        if not candidates:
            return None
        chosen = candidates[0]
        parsed = _parse_synced_lyrics(chosen["syncedLyrics"])
        if not parsed:
            return None
        return LrcMatch(
            track_name=str(chosen.get("trackName") or title).strip(),
            artist_name=str(chosen.get("artistName") or artist).strip(),
            lines=parsed,
        )
    except Exception as e:  # noqa: BLE001 — best-effort, must never raise
        log.info("lrclib fetch failed (continuing without it): %s", e)
        return None


def fetch_lrc(
    title: str, artist: str, duration_seconds: Optional[float]
) -> Optional[list[LrcLine]]:
    """Compatibility timing wrapper around :func:`fetch_lrc_match`."""
    match = fetch_lrc_match(title, artist, duration_seconds)
    return match.lines if match is not None else None


def _normalize(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def match_lrc_to_lines(lrc: list[LrcLine], song: Song) -> dict[int, tuple[float, float]]:
    """{lineIndex: (time, similarity)} for song lines that fuzzily match an
    LRC line at ratio >= LINE_MATCH_THRESHOLD.

    Matching is monotonic: once an LRC line is consumed, only LATER LRC
    lines are considered for subsequent song lines — a repeated chorus in
    the LRC file can't get matched out of order against the song's single
    linear lyric transcript.
    """
    result: dict[int, tuple[float, float]] = {}
    lrc_cursor = 0
    for line in song.lines:
        target = _normalize(line.lyrics)
        if not target:
            continue
        best_ratio = 0.0
        best_j: Optional[int] = None
        for j in range(lrc_cursor, len(lrc)):
            candidate = _normalize(lrc[j].text)
            if not candidate:
                continue
            ratio = SequenceMatcher(None, target, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_j = j
        if best_j is not None and best_ratio >= LINE_MATCH_THRESHOLD:
            result[line.lineIndex] = (lrc[best_j].time, best_ratio)
            lrc_cursor = best_j + 1
    return result


def _fill_line_times(n: int, known: dict[int, float]) -> list[Optional[float]]:
    """Positional linear interpolation of line times between known anchors
    (LRC matches plus whatever MIR-derived times already existed), with a
    defensive non-decreasing clamp. Mirrors timing.snap's line-time
    derivation, kept self-contained here rather than shared/refactored."""
    raw: list[Optional[float]] = [known.get(i) for i in range(n)]
    positions = sorted(known)
    if not positions:
        return raw

    result = list(raw)
    for i in range(n):
        if result[i] is not None:
            continue
        before = max((p for p in positions if p <= i), default=None)
        after = min((p for p in positions if p >= i), default=None)
        if before is not None and after is not None and before != after:
            frac = (i - before) / (after - before)
            result[i] = raw[before] + frac * (raw[after] - raw[before])
        elif before is not None:
            result[i] = raw[before]
        elif after is not None:
            result[i] = raw[after]

    running_max = float("-inf")
    for i in range(n):
        if result[i] is None:
            continue
        if result[i] < running_max:
            result[i] = running_max
        else:
            running_max = result[i]
    return result


def apply_lrc(
    song: Song, mir: Optional[MirAnalysis], matches: dict[int, tuple[float, float]]
) -> Song:
    """LRC times WIN over whatever line timing already existed. Chord
    placements are then re-anchored proportionally (by charIndex fraction)
    between their line's new start and the next line's new start, and
    snapped onto the MIR beat grid when one is available — LRC is
    excellent evidence for LINE timing, not chord timing, so chord
    positions are redistributed rather than trusted from a prior pass.

    A no-op (returns `song` unchanged) when `matches` is empty.
    """
    if not matches:
        return song

    n = len(song.lines)
    known_times: dict[int, float] = {
        line.lineIndex: line.timeSeconds for line in song.lines if line.timeSeconds is not None
    }
    line_confidence: dict[int, float] = {}
    for line_idx, (t, similarity) in matches.items():
        known_times[line_idx] = t
        line_confidence[line_idx] = similarity

    filled = _fill_line_times(n, known_times)
    beat_grid = build_beat_grid(mir) if mir else []

    new_lines = []
    for i, line in enumerate(song.lines):
        line_start = filled[i]
        next_start = filled[i + 1] if i + 1 < n else None
        if line_start is not None and (next_start is None or next_start <= line_start):
            next_start = (
                mir.duration_seconds
                if mir and mir.duration_seconds and mir.duration_seconds > line_start
                else line_start
            )

        new_placements = []
        for p in line.chordPlacements:
            if line_start is None:
                new_placements.append(p)
                continue
            frac = (p.charIndex / len(line.lyrics)) if line.lyrics else 0.0
            span = next_start - line_start if next_start and next_start > line_start else 0.0
            t = line_start + frac * span
            if beat_grid:
                t, beat_ref = snap_time(t, beat_grid, DEFAULT_SNAP_TOLERANCE_SECONDS)
            else:
                beat_ref = None
            update: dict = {"timeSeconds": t}
            if beat_ref is not None:
                update["beat"] = beat_ref
            new_placements.append(p.model_copy(update=update))

        line_update: dict = {"chordPlacements": new_placements}
        if line_start is not None:
            line_update["timeSeconds"] = line_start
        if line.lineIndex in line_confidence:
            line_update["confidence"] = line_confidence[line.lineIndex]
        new_lines.append(line.model_copy(update=line_update))

    sync_map = [
        SyncPoint(lineIndex=l.lineIndex, time=l.timeSeconds)
        for l in new_lines
        if l.timeSeconds is not None
    ]
    new_audio = song.audio.model_copy(update={"syncMap": sync_map})

    similarities = list(line_confidence.values())
    entry = ProvenanceEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        actor="snoocle-server/timing",
        action="lrc-align",
        sources=["lrclib.net"],
        confidence=(sum(similarities) / len(similarities)) if similarities else None,
        notes=f"{len(matches)}/{n} line(s) matched to LRCLIB synced lyrics",
    )

    updated = song.model_copy(
        update={
            "lines": new_lines,
            "audio": new_audio,
            "provenance": list(song.provenance) + [entry],
        }
    )
    return Song.model_validate(updated.model_dump())
