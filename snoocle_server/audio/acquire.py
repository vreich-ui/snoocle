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
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..config import settings

log = logging.getLogger(__name__)

_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|^)([A-Za-z0-9_-]{11})(?:[?&]|$)")


class AcquisitionError(RuntimeError):
    pass


_materialized: dict[str, str] = {}  # content hash -> temp cookies.txt path


def _materialize_cookies(content: str) -> str:
    """Write cookies.txt content to a temp file, cached by content hash so a
    refreshed cookie set takes effect immediately without re-writing."""
    import hashlib

    key = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    path = _materialized.get(key)
    if path is None or not Path(path).exists():
        p = Path(tempfile.mkdtemp(prefix="snoocle-ytc-")) / "cookies.txt"
        p.write_text(content)
        _materialized[key] = path = str(p)
    return path


def _stored_cookies_txt() -> str | None:
    """Cookies uploaded at runtime (in-app sign-in / manual upload), from the
    durable store. Best-effort — never let a store hiccup break acquisition."""
    try:
        from ..store import get_repository

        return get_repository().get_youtube_cookies_txt()
    except Exception:  # noqa: BLE001
        return None


def _resolve_cookiefile() -> str | None:
    """A cookies.txt path for yt-dlp. Precedence: runtime-uploaded cookies
    (refreshable without redeploy) > SNOOCLE_YTDLP_COOKIES_FILE (mounted path) >
    SNOOCLE_YTDLP_COOKIES (env content). None when nothing is configured."""
    stored = _stored_cookies_txt()
    if stored:
        return _materialize_cookies(stored)
    if settings.ytdlp_cookies_file:
        return settings.ytdlp_cookies_file
    if settings.ytdlp_cookies:
        return _materialize_cookies(settings.ytdlp_cookies)
    return None


def _ytdlp_opts(base: dict) -> dict:
    """Merge YouTube-auth accommodations (cookies, player clients) into a base
    yt-dlp options dict — used by every yt-dlp call so they authenticate
    consistently. Off by default (no config -> base unchanged)."""
    opts = dict(base)
    cookiefile = _resolve_cookiefile()
    if cookiefile:
        opts["cookiefile"] = cookiefile
    if settings.ytdlp_proxy:
        opts["proxy"] = settings.ytdlp_proxy
    if settings.ytdlp_cache_dir:
        opts["cachedir"] = settings.ytdlp_cache_dir
    clients = [c.strip() for c in settings.ytdlp_player_clients.split(",") if c.strip()]
    if clients:
        extractor_args = dict(opts.get("extractor_args") or {})
        youtube = dict(extractor_args.get("youtube") or {})
        youtube["player_client"] = clients
        extractor_args["youtube"] = youtube
        opts["extractor_args"] = extractor_args
    return opts


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


_QUOTE_CHARS = '"“”„‟'

# 'Artist "Track" <anything>' — live/one-off uploads often quote the song name
# instead of using an "Artist - Title" separator (e.g. 'Blues Traveler "Hook"
# at Howard Stern's 1996 Birthday Show').
_QUOTED_TRACK_RE = re.compile(
    rf"^(?P<artist>[^{_QUOTE_CHARS}]{{2,60}}?)\s+[{_QUOTE_CHARS}]"
    rf"(?P<track>[^{_QUOTE_CHARS}]{{1,80}})[{_QUOTE_CHARS}]"
)


def parse_quoted_track(text: str) -> tuple[str, str] | None:
    """(artist, track) from a video title like 'Artist "Track" at X's show',
    or None when the pattern doesn't apply."""
    m = _QUOTED_TRACK_RE.match(_strip_title_noise(text))
    if m:
        return m.group("artist").strip(), m.group("track").strip()
    return None


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
            left, right = (p.strip().strip(_QUOTE_CHARS).strip() for p in parts)
            artist = artist or left
            track = track or right
        else:
            quoted = parse_quoted_track(cleaned)
            if quoted:
                artist = artist or quoted[0]
                track = track or quoted[1]

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
        with yt_dlp.YoutubeDL(_ytdlp_opts(opts)) as ydl:
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
        with yt_dlp.YoutubeDL(_ytdlp_opts(opts)) as ydl:
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
        "format": settings.ytdlp_format,
        "concurrent_fragment_downloads": max(settings.ytdlp_concurrent_fragments, 1),
        "outtmpl": str(cache / "%(title).80s [%(id)s].%(ext)s"),
        "noplaylist": True,
    }
    import time

    start = time.monotonic()
    try:
        with yt_dlp.YoutubeDL(_ytdlp_opts(opts)) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
    except Exception as e:  # yt_dlp raises many exception types
        raise AcquisitionError(f"yt-dlp failed for {video_id}: {e}") from e

    path = _cache_hit(video_id)
    if path is None:
        raise AcquisitionError(f"yt-dlp reported success but no file found for {video_id}")
    log.info(
        "yt-dlp downloaded %s in %.1fs (%.1f MB)",
        video_id, time.monotonic() - start, path.stat().st_size / 1e6,
    )
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
