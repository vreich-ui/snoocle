"""YouTube audio acquisition via yt-dlp.

PERSONAL-USE NOTICE: server-side YouTube audio extraction is a deliberate,
scoped decision for a single-user, non-distributed personal tool (see
README). Revisit before any public or shared exposure.

Given title+artist we search YouTube (yt-dlp's ytsearch) and pick the most
plausible official/album match; given an explicit video id/URL we download
directly. Audio is cached by video id so repeat analyses don't re-download.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import settings

log = logging.getLogger(__name__)

_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|^)([A-Za-z0-9_-]{11})(?:[?&]|$)")


class AcquisitionError(RuntimeError):
    pass


@dataclass
class AcquiredAudio:
    video_id: str
    video_title: str
    path: str  # audio file on disk (m4a/webm/opus as delivered)
    duration_seconds: float | None
    from_cache: bool = False


@dataclass
class ResolvedMeta:
    """Song identity derived from a media URL's own metadata (no download)."""

    video_id: str
    video_title: str
    title: str
    artist: str
    duration_seconds: float | None = None


# Trailing decoration commonly appended to music video titles — stripped before
# parsing "Artist - Title". Matches a parenthesized/bracketed group containing
# one of these words: "(Official Music Video)", "[Lyric Video]", "(Remastered
# 2009)", "(HD)", "(Audio)", etc.
_TITLE_NOISE_RE = re.compile(
    r"\s*[\(\[][^\)\]]*\b(?:official|video|audio|lyrics?|visuali[sz]er|remaster(?:ed)?|"
    r"hd|4k|hq|mv|m/v|explicit|clean|full\s+album|live|performance|version)\b[^\)\]]*[\)\]]",
    re.IGNORECASE,
)


def _strip_title_noise(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = _TITLE_NOISE_RE.sub("", text).strip()
    return text.strip(" -—–|·")


def derive_title_artist(info: dict) -> tuple[str, str]:
    """Best-effort (title, artist) from a yt-dlp info dict.

    Prefers explicit music metadata (``track``/``artist``, present on YouTube
    Music / "Provided to YouTube by..." entries); otherwise parses the video
    title ("Artist - Title", noise stripped) and falls back to the uploader
    (minus a "- Topic" suffix) for the artist.
    """
    track = (info.get("track") or "").strip()
    artist = (info.get("artist") or info.get("creator") or "").strip()
    vid_title = (info.get("title") or "").strip()
    uploader = (info.get("uploader") or info.get("channel") or "").strip()
    uploader = re.sub(r"\s*-\s*Topic$", "", uploader, flags=re.IGNORECASE).strip()

    cleaned = _strip_title_noise(vid_title)
    if not (track and artist):
        # "Artist - Title", tolerating hyphen / en-dash / em-dash separators
        parts = re.split(r"\s+[-–—]\s+", cleaned, maxsplit=1)
        if len(parts) == 2:
            left, right = parts[0].strip(), parts[1].strip()
            artist = artist or left
            track = track or right

    title = track or cleaned or vid_title or "Unknown"
    artist = artist or uploader or "Unknown"
    return title, artist


def extract_metadata(url_or_id: str) -> ResolvedMeta:
    """Resolve a song's identity from a YouTube URL/id WITHOUT downloading the
    audio (yt-dlp ``download=False``) — used when the caller gives only a URL
    and expects title/artist to be populated from the media itself."""
    import yt_dlp

    vid = extract_video_id(url_or_id)
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
    except Exception as e:  # yt_dlp raises many exception types
        raise AcquisitionError(f"metadata fetch failed for {vid}: {e}") from e
    title, artist = derive_title_artist(info or {})
    return ResolvedMeta(
        video_id=vid,
        video_title=(info or {}).get("title") or "",
        title=title,
        artist=artist,
        duration_seconds=float(info["duration"]) if (info or {}).get("duration") else None,
    )


def extract_video_id(url_or_id: str) -> str:
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    m = _VIDEO_ID_RE.search(s)
    if not m:
        raise AcquisitionError(f"cannot extract a YouTube video id from {url_or_id!r}")
    return m.group(1)


def _cache_hit(video_id: str) -> Path | None:
    cache = Path(settings.audio_cache_dir)
    if cache.exists():
        for p in sorted(cache.glob(f"*[[]{video_id}[]].*")):
            if p.suffix.lower() in {".m4a", ".webm", ".opus", ".mp3", ".ogg", ".wav"}:
                return p
    return None


def search_video(title: str, artist: str, max_results: int = 5) -> list[dict]:
    """Search YouTube for candidate videos; returns yt-dlp flat entries."""
    import yt_dlp

    query = f"{artist} {title}"
    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist", "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
    except Exception as e:  # yt_dlp raises many exception types
        raise AcquisitionError(f"YouTube search failed for {query!r}: {e}") from e
    entries = [e for e in (info.get("entries") or []) if e]
    if not entries:
        raise AcquisitionError(f"no YouTube results for {query!r}")
    return entries


def pick_best_video(entries: list[dict], title: str, artist: str) -> dict:
    """Prefer plausible studio recordings over covers/lessons/live cuts."""
    bad_words = ("cover", "lesson", "tutorial", "karaoke", "how to play", "reaction", "drum")
    good_words = ("official", "audio", "remaster", "album", "lyric")

    def score(e: dict) -> float:
        t = (e.get("title") or "").lower()
        s = 0.0
        if title.lower() in t:
            s += 2.0
        if artist.lower() in t or artist.lower() in (e.get("channel") or e.get("uploader") or "").lower():
            s += 2.0
        s += sum(0.5 for w in good_words if w in t)
        s -= sum(1.5 for w in bad_words if w in t)
        dur = e.get("duration") or 0
        if dur and not (60 <= dur <= 15 * 60):  # implausible song length
            s -= 2.0
        return s

    return max(entries, key=score)


def download_audio(video_id: str) -> AcquiredAudio:
    import yt_dlp

    cached = _cache_hit(video_id)
    if cached is not None:
        log.info("audio cache hit for %s: %s", video_id, cached)
        info = {"id": video_id, "title": cached.stem, "duration": None}
        return AcquiredAudio(
            video_id=video_id,
            video_title=cached.stem,
            path=str(cached),
            duration_seconds=None,
            from_cache=True,
        )

    cache = Path(settings.audio_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "outtmpl": str(cache / "%(title).80s [%(id)s].%(ext)s"),
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
    except Exception as e:  # yt_dlp raises many exception types
        raise AcquisitionError(f"yt-dlp failed for {video_id}: {e}") from e

    path = _cache_hit(video_id)
    if path is None:
        raise AcquisitionError(f"yt-dlp reported success but no file found for {video_id}")
    return AcquiredAudio(
        video_id=video_id,
        video_title=info.get("title") or path.stem,
        path=str(path),
        duration_seconds=float(info["duration"]) if info.get("duration") else None,
    )


def acquire(
    title: str | None = None,
    artist: str | None = None,
    video_url_or_id: str | None = None,
) -> AcquiredAudio:
    """Resolve a recording: explicit id/URL wins, else search by title+artist."""
    if video_url_or_id:
        return download_audio(extract_video_id(video_url_or_id))
    if not (title and artist):
        raise AcquisitionError("need either a video URL/id or title+artist")
    entries = search_video(title, artist)
    best = pick_best_video(entries, title, artist)
    vid = best.get("id") or extract_video_id(best.get("url") or "")
    log.info("picked video %s (%s) for %s — %s", vid, best.get("title"), artist, title)
    return download_audio(vid)
