"""Deterministic, LLM-free reconciliation used by the mock provider.

Merges the highest-confidence candidate's lines with MIR-derived metadata,
section timestamps and a syncMap. Real output, zero network — used for
offline tests and as executable documentation of a valid reconciliation.

With no candidate sources (the fully-offline analyze path) it synthesizes a
small, schema-valid placeholder Song from title/artist so the whole
analyze -> persist -> fetch -> versions path is exercisable with no network.
"""

from __future__ import annotations

from ..discovery.models import CandidateSource
from ..mir.base import MirAnalysis
from ..schema.song import (
    AudioInfo,
    ChordPlacement,
    DisplayPreferences,
    Line,
    Section,
    Song,
    SongMetadata,
    SyncPoint,
)

_KIND_WORDS = {
    "intro": "intro",
    "verse": "verse",
    "pre-chorus": "prechorus",
    "prechorus": "prechorus",
    "chorus": "chorus",
    "bridge": "bridge",
    "solo": "solo",
    "instrumental": "instrumental",
    "interlude": "interlude",
    "outro": "outro",
    "ending": "outro",
    "coda": "outro",
    "breakdown": "breakdown",
    "hook": "chorus",
    "refrain": "chorus",
}


def _kind_for(name: str) -> str:
    low = name.lower()
    for word, kind in _KIND_WORDS.items():
        if word in low:
            return kind
    return "other"


def _synthesize_placeholder(
    title: str,
    artist: str,
    song_id: str,
    youtube_video_id: str | None,
    mir: MirAnalysis | None,
) -> Song:
    """A tiny deterministic Song from title/artist alone — the offline path
    (no candidate sources). Two sections, a handful of lines, a simple diatonic
    progression; MIR metadata folded in when present."""
    lines = [
        Line(lineIndex=0, lyrics=title,
             chordPlacements=[ChordPlacement(charIndex=0, chord="C")]),
        Line(lineIndex=1, lyrics=f"performed by {artist}",
             chordPlacements=[ChordPlacement(charIndex=0, chord="G")]),
        Line(lineIndex=2, lyrics="(mock reconciliation — deterministic offline placeholder)",
             chordPlacements=[ChordPlacement(charIndex=0, chord="Am"),
                              ChordPlacement(charIndex=5, chord="F")]),
    ]
    sections = [
        Section(sectionIndex=0, name="Verse", kind="verse", startLineIndex=0, endLineIndex=1),
        Section(sectionIndex=1, name="Chorus", kind="chorus", startLineIndex=2, endLineIndex=2),
    ]
    return Song(
        id=song_id,
        metadata=SongMetadata(
            title=title,
            artist=artist,
            key=(mir.key if mir else "C major"),
            bpm=mir.bpm if mir else None,
            timeSignature=mir.time_signature if mir else None,
        ),
        displayPreferences=DisplayPreferences(capo=0, tuning="standard"),
        audio=AudioInfo(
            youtubeVideoId=youtube_video_id,
            durationSeconds=mir.duration_seconds if mir else None,
        ),
        sections=sections,
        lines=lines,
        provenance=[],
    )


def reconcile_deterministically(
    title: str,
    artist: str,
    song_id: str,
    youtube_video_id: str | None,
    candidates: list[CandidateSource],
    mir: MirAnalysis | None,
) -> Song:
    if not candidates:
        return _synthesize_placeholder(title, artist, song_id, youtube_video_id, mir)
    best = max(candidates, key=lambda c: c.confidence)
    lines = [l.model_copy(update={"lineIndex": i}) for i, l in enumerate(best.lines)]

    # sections from the winning sheet's header positions
    sections: list[Section] = []
    starts = [(s.name, s.startLineIndex) for s in best.sectionStarts if s.startLineIndex < len(lines)]
    if not starts and lines:
        starts = [("Song", 0)]
    for i, (name, start) in enumerate(starts):
        end = (starts[i + 1][1] - 1) if i + 1 < len(starts) else len(lines) - 1
        if end < start:
            continue
        sections.append(
            Section(
                sectionIndex=len(sections),
                name=name,
                kind=_kind_for(name),
                startLineIndex=start,
                endLineIndex=end,
            )
        )

    # attach MIR timestamps: same count -> 1:1 by order; else proportional by line share
    sync: list[SyncPoint] = []
    if mir is not None and sections:
        mir_secs = mir.sections
        if len(mir_secs) == len(sections):
            paired = list(zip(sections, mir_secs))
        else:
            total_lines = max(len(lines), 1)
            duration = mir.duration_seconds or 0.0
            paired = []
            for s in sections:
                frac_a = s.startLineIndex / total_lines
                frac_b = (s.endLineIndex + 1) / total_lines
                paired.append(
                    (s, type("T", (), {"start": frac_a * duration, "end": frac_b * duration})())
                )
        sections = [
            s.model_copy(update={"startTime": round(t.start, 2), "endTime": round(t.end, 2)})
            for s, t in paired
        ]
        prev_time = -1.0
        for s in sections:
            t = max(float(s.startTime or 0.0), prev_time)
            sync.append(SyncPoint(lineIndex=s.startLineIndex, time=round(t, 2)))
            prev_time = t

    key = best.declaredKey or (mir.key if mir else None)
    return Song(
        id=song_id,
        metadata=SongMetadata(
            title=title,
            artist=artist,
            key=key,
            bpm=mir.bpm if mir else None,
            timeSignature=mir.time_signature if mir else None,
        ),
        displayPreferences=DisplayPreferences(capo=0, tuning="standard"),
        audio=AudioInfo(
            youtubeVideoId=youtube_video_id,
            durationSeconds=mir.duration_seconds if mir else None,
            syncMap=sync,
        ),
        sections=sections,
        lines=lines,
        provenance=[],
    )
