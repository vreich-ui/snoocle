"""Finding a BETTER recording of a song we already have.

When :mod:`quality.attribution` returns an AUDIO fault, the document is as good
as its recording allows and no amount of re-alignment or re-prompting will
improve it. The 1966 live Paint It Black is the case in point: three sources
agree on the words and chords, and the chord recognizer still only reaches
86.5s of a 220.6s track because the recording is a 60-year-old live video. What
that song needs is a recording the analysis can hear.

This module only ever REPORTS. It searches (one cheap yt-dlp query, no
download) and ranks; it never analyzes. That restraint is deliberate: a second
track is real spend — a download, a full MIR pass, and possibly a model call —
and the operator is the one who should decide whether this song is worth it.
So every suggestion carries the exact action that would spend it, and nothing
happens until someone invokes that action.

Ranking reuses ``audio.acquire.score_video``, the same judgement acquisition
uses to pick a video in the first place (official/audio/remaster up, cover/
lesson/karaoke down, implausible lengths down). A recording the song is ALREADY
timed against is never suggested as an alternative to itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .audio.acquire import score_video, search_video
from .schema.song import Song

log = logging.getLogger(__name__)

#: How many ranked alternatives to report. Small: this is a decision for a
#: human, and a list of twenty is not a decision aid.
DEFAULT_LIMIT = 5

#: How many search results to consider before ranking. Wider than the reported
#: limit so covers and lessons can be filtered out without emptying the list.
SEARCH_WIDTH = 8


@dataclass(frozen=True)
class RecordingSuggestion:
    """One alternative recording, with the action that would use it."""

    video_id: str
    title: str
    channel: str
    duration_seconds: Optional[float]
    score: float
    url: str
    action: str  # the operator-invokable instruction, in words

    def to_dict(self) -> dict:
        return {
            "videoId": self.video_id,
            "title": self.title,
            "channel": self.channel,
            "durationSeconds": self.duration_seconds,
            "score": round(self.score, 2),
            "url": self.url,
            "action": self.action,
        }


@dataclass(frozen=True)
class RecordingSuggestions:
    """The report: candidates, how to act on them, and why they were sought."""

    song_id: str
    reason: str
    suggestions: list[RecordingSuggestion] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)  # already-used video ids
    error: Optional[str] = None

    def describe(self) -> str:
        if self.error:
            return f"recording search failed: {self.error}"
        if not self.suggestions:
            return "no alternative recording found"
        best = self.suggestions[0]
        return (
            f"{len(self.suggestions)} alternative recording(s) found; best: "
            f"{best.video_id} ({best.title!r}) — not analyzed, awaiting an operator"
        )

    def to_dict(self) -> dict:
        out: dict = {
            "songId": self.song_id,
            "reason": self.reason,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "excluded": list(self.excluded),
            # Stated in the report itself so no client has to infer it from the
            # absence of a version field.
            "analyzed": False,
            "howToApply": (
                "Nothing has been analyzed. To spend on one of these, invoke the "
                "action on that suggestion: it runs a full analysis of that video "
                f"and re-aligns {self.song_id!r} to it (Mode B), storing the result "
                "as a new version of the same song."
            ),
        }
        if self.error:
            out["error"] = self.error
        return out


def realign_action(song_id: str, video_id: str) -> str:
    """The operator-invokable action, in the words the brief asks for."""
    return f"analyze {video_id} as the timing reference for {song_id}"


def suggest_recordings(
    song: Song,
    *,
    reason: str = "this version's timing is unreliable on its current recording",
    limit: int = DEFAULT_LIMIT,
    search_width: int = SEARCH_WIDTH,
) -> RecordingSuggestions:
    """Rank alternative recordings of `song`. Never analyzes, never raises.

    A search failure (no network, YouTube throttling, no results) is reported
    in the result rather than raised: this is an advisory step hanging off a
    run that has already produced a storable document, and it must not be able
    to turn that run into a failure.
    """
    already_used = [
        v for v in (song.audio.analyzedVideoId, song.audio.youtubeVideoId) if v
    ]
    title, artist = song.metadata.title, song.metadata.artist
    try:
        entries = search_video(title, artist, max_results=search_width)
    except Exception as e:  # noqa: BLE001 — advisory only, see the docstring
        log.info("recording suggestion search failed for %s: %s", song.id, e)
        return RecordingSuggestions(
            song_id=song.id, reason=reason, excluded=already_used, error=str(e)[:300]
        )

    ranked: list[RecordingSuggestion] = []
    for entry in entries:
        video_id = entry.get("id") or ""
        if not video_id or video_id in already_used:
            continue
        duration = entry.get("duration")
        ranked.append(
            RecordingSuggestion(
                video_id=video_id,
                title=entry.get("title") or "",
                channel=entry.get("channel") or entry.get("uploader") or "",
                duration_seconds=float(duration) if duration else None,
                score=score_video(entry, title, artist),
                url=entry.get("url") or f"https://www.youtube.com/watch?v={video_id}",
                action=realign_action(song.id, video_id),
            )
        )
    ranked.sort(key=lambda s: s.score, reverse=True)
    return RecordingSuggestions(
        song_id=song.id,
        reason=reason,
        suggestions=ranked[:limit],
        excluded=already_used,
    )
