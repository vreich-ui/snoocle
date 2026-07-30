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

from .. import identity as _identity
from ..config import settings

log = logging.getLogger(__name__)

_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|^)([A-Za-z0-9_-]{11})(?:[?&]|$)")


class AcquisitionError(RuntimeError):
    pass


class YouTubeAuthError(AcquisitionError):
    """YouTube refused the request for AUTH reasons — the datacenter bot-check,
    expired/invalid session cookies, or an age-restricted video. Retrying won't
    help; the fix is reconnecting the YouTube session (the in-app cookie
    upload). Surfaced to clients as errorCode "youtube_auth_required"."""


# yt-dlp error fragments that mean the YouTube SESSION is the problem, not the
# video or the network. Matched case-insensitively against the wrapped error.
_AUTH_ERROR_MARKERS = (
    "sign in to confirm you're not a bot",
    "sign in to confirm your age",
    "confirm you're not a robot",
    "use --cookies",
    "cookies are no longer valid",
    "please sign in",
    "login required",
    "not a bot",
)


def _acquisition_error(message: str, cause: Exception) -> AcquisitionError:
    """Wrap a yt-dlp failure, classifying YouTube auth problems so callers can
    tell "reconnect YouTube" apart from every other failure."""
    text = f"{message}: {cause}"
    low = str(cause).lower()
    if any(marker in low for marker in _AUTH_ERROR_MARKERS):
        return YouTubeAuthError(text)
    return AcquisitionError(text)


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
    """Song identity derived from a media URL's own metadata (no download).

    ``title``/``artist`` are the best-effort labels from
    :func:`derive_title_artist`. The raw signals they were derived from are
    carried alongside so the caller can run the refuses-rather-than-guesses
    resolver (:func:`snoocle_server.identity.resolve_identity`) without a
    second metadata fetch.
    """

    video_id: str
    video_title: str
    title: str
    artist: str
    duration_seconds: float | None = None
    # Raw identity signals, straight from yt-dlp.
    uploader: str = ""  # channel name, "- Topic" suffix already removed
    track: str = ""  # distributor-supplied song title, when present
    track_artist: str = ""  # distributor-supplied artist, when present


# Title decoration handling and separator parsing live in `identity`, which
# owns the whole "video title -> (title, artist)" problem: the deterministic
# layer below plus the escalation path for titles it can't settle. These thin
# wrappers keep the long-standing call sites here and in `discovery.service`
# pointed at that single implementation.
_strip_title_noise = _identity.strip_title_noise
_QUOTE_CHARS = _identity._QUOTE_CHARS


def parse_quoted_track(text: str) -> tuple[str, str] | None:
    """(artist, track) from a video title like 'Artist "Track" at X's show',
    or None when the pattern doesn't apply."""
    split = _identity.split_artist_title(_strip_title_noise(text))
    if split and split[2] == "quoted":
        return split[0], split[1]
    return None


def parse_dash_title(text: str) -> tuple[str, str] | None:
    """(artist, track) from an 'Artist - Track' video title (hyphen / en-dash /
    em-dash separators, decoration stripped), or None when it doesn't apply."""
    split = _identity.split_artist_title(_strip_title_noise(text))
    if split and split[2] == "dash":
        return split[0], split[1]
    return None


def derive_title_artist(info: dict) -> tuple[str, str]:
    """Best-effort (title, artist) from a yt-dlp info dict.

    Prefers explicit music metadata (``track``/``artist``, present on YouTube
    Music / "Provided to YouTube by..." entries); otherwise parses the video
    title ("Artist - Title", noise stripped) and falls back to the uploader
    (minus a "- Topic" suffix) for the artist.

    Best-effort by contract: it always returns SOMETHING, filling "Unknown"
    when the title yields nothing — which is exactly why it must not be what
    mints a song id. The pipeline resolves identity through
    :func:`snoocle_server.identity.resolve_identity`, which refuses instead of
    guessing; this stays for display/logging callers that just want a label.
    """
    track = (info.get("track") or "").strip()
    artist = (info.get("artist") or info.get("creator") or "").strip()
    vid_title = (info.get("title") or "").strip()
    uploader = _identity._strip_topic(info.get("uploader") or info.get("channel") or "")

    cleaned = _strip_title_noise(vid_title)
    if not (track and artist):
        split = _identity.split_artist_title(cleaned)
        if split is not None:
            artist = artist or split[0]
            track = track or split[1]

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
        raise _acquisition_error(f"metadata fetch failed for {vid}", e) from e
    info = info or {}
    title, artist = derive_title_artist(info)
    return ResolvedMeta(
        video_id=vid,
        video_title=info.get("title") or "",
        title=title,
        artist=artist,
        duration_seconds=float(info["duration"]) if info.get("duration") else None,
        uploader=_identity._strip_topic(info.get("uploader") or info.get("channel") or ""),
        track=(info.get("track") or "").strip(),
        track_artist=(info.get("artist") or info.get("creator") or "").strip(),
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


def cached_audio_path(video_id: str) -> Path | None:
    """The locally cached audio for `video_id`, or None.

    Public because "is this already on disk?" decides whether an optional
    check is free or expensive: ``realign.same_recording_check`` only runs its
    cross-correlation when both files are already here (see that function).
    """
    return _cache_hit(video_id)


def search_video(title: str, artist: str, max_results: int = 5) -> list[dict]:
    """Search YouTube for candidate videos; returns yt-dlp flat entries."""
    import yt_dlp

    query = f"{artist} {title}"
    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist", "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(_ytdlp_opts(opts)) as ydl:
            info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
    except Exception as e:  # yt_dlp raises many exception types
        raise _acquisition_error(f"YouTube search failed for {query!r}", e) from e
    entries = [e for e in (info.get("entries") or []) if e]
    if not entries:
        raise AcquisitionError(f"no YouTube results for {query!r}")
    return entries


_BAD_WORDS = ("cover", "lesson", "tutorial", "karaoke", "how to play", "reaction", "drum")
_GOOD_WORDS = ("official", "audio", "remaster", "album", "lyric")


def score_video(entry: dict, title: str, artist: str) -> float:
    """How plausibly this search result is a studio recording of `title`.

    Public because "which recording should we use?" is asked in two places
    now: acquisition picks the single best, and
    :mod:`snoocle_server.recordings` ranks alternatives for an operator to
    choose from (a song whose timing came out unreliable needs a BETTER
    recording, and that is the same judgement).
    """
    t = (entry.get("title") or "").lower()
    s = 0.0
    if title.lower() in t:
        s += 2.0
    if artist.lower() in t or artist.lower() in (
        entry.get("channel") or entry.get("uploader") or ""
    ).lower():
        s += 2.0
    s += sum(0.5 for w in _GOOD_WORDS if w in t)
    s -= sum(1.5 for w in _BAD_WORDS if w in t)
    dur = entry.get("duration") or 0
    if dur and not (60 <= dur <= 15 * 60):  # implausible song length
        s -= 2.0
    return s


def pick_best_video(entries: list[dict], title: str, artist: str) -> dict:
    """Prefer plausible studio recordings over covers/lessons/live cuts."""
    return max(entries, key=lambda e: score_video(e, title, artist))


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
        raise _acquisition_error(f"yt-dlp failed for {video_id}", e) from e

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
