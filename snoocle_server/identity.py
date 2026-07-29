"""Song identity (title + artist) resolution from media metadata.

The song id is a content-hash-versioned store key derived from
``slugify_song_id(artist, title)`` — it is PERMANENT. A wrong id is forever:
it can't be renamed without orphaning the version history, and every later
run keyed on the same video lands on the same wrong document. So this module
biases hard toward REFUSING over guessing.

Three tiers, cheapest first:

1. **yt-dlp structured fields** (``track``/``artist``/``creator``, present on
   YouTube Music and "Provided to YouTube by ..." uploads). These are the
   distributor's own metadata — trusted above anything parsed out of a title.
2. **Deterministic title parsing** — strip the decoration layer YouTube
   titles accumulate ("(Official Video)", "[4K]", "w/ lyrics", a remaster
   year, emoji), then match the handful of separator shapes uploaders
   actually use. No network, no model, no cost.
3. **One cheap LLM call** (:data:`Settings.identity_model`, a Haiku) — ONLY
   when 1 and 2 leave it genuinely ambiguous: no separator at all, cover
   phrasing (where the uploader is not the artist and the title is not the
   song), or a channel name that can't be trusted as the artist.

Below :data:`Settings.identity_min_confidence` the resolve step FAILS with
:class:`IdentityError` telling the caller to pass title+artist explicitly.
It never stores a guess.

**Covers.** For a cover, the stored identity is the ORIGINAL song's title and
the PERFORMING artist — the band actually playing, which is usually neither
the uploader's channel name nor the original artist. The original artist is
carried on :attr:`SongIdentity.original_artist` so the pipeline can record it
as a provenance note rather than silently losing it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, Optional

from .config import settings

log = logging.getLogger(__name__)


class IdentityError(ValueError):
    """Identity could not be resolved confidently enough to mint a song id.

    Deliberately fatal: a song id is permanent, so an ambiguous title is a
    reason to ask the caller for title+artist, never to guess.
    """


@dataclass(frozen=True)
class SongIdentity:
    title: str
    artist: str
    confidence: float  # 0-1; below settings.identity_min_confidence -> refuse
    method: str  # how it was derived — recorded in the pipeline's step trace
    is_cover: bool = False
    # For a cover: who the song is originally BY (artist is who's playing it).
    original_artist: Optional[str] = None

    def describe(self) -> str:
        s = f"title={self.title!r} artist={self.artist!r} ({self.method}, confidence={self.confidence:.2f})"
        if self.is_cover:
            s += f" cover-of={self.original_artist!r}"
        return s


# --- deterministic layer ----------------------------------------------------

# Decoration words. A bracketed group containing any of these is stripped
# whole. Deliberately ABSENT: "live", "version", "performance", "acoustic" —
# "(Live at Wembley)" and "(Acoustic Version)" name a DIFFERENT recording of
# the song, not decoration on this one, and dropping them would collapse two
# genuinely distinct songs onto one permanent id. A bare "(Live)" surviving
# into the title is the accepted cost of never making that mistake.
_NOISE_WORD = (
    r"official|video|audio|lyrics?|lyric\s+video|visuali[sz]er|remaster(?:ed)?|"
    r"hd|hq|uhd|4k|8k|1080p?|720p?|mv|m/v|explicit|clean|full\s+album|"
    r"audio\s+only|color\s+coded"
)

_BRACKETED_NOISE_RE = re.compile(
    rf"\s*[\(\[\{{][^\)\]\}}]*\b(?:{_NOISE_WORD})\b[^\)\]\}}]*[\)\]\}}]",
    re.IGNORECASE,
)

# Unbracketed decoration, only ever at the END of a title and only for phrases
# that are never part of a real song name. "Live" / "Version" are deliberately
# ABSENT here: bare "... Live" may well be "(Live)" decoration, but bare
# "... Version" is often the actual title ("Piano Version"), and getting this
# wrong writes a permanent id. Bracketed forms are still stripped above.
_TRAILING_NOISE_RE = re.compile(
    # Preceded by a separator, by whitespace, or by nothing at all — the last
    # alternative is what lets a title that is ENTIRELY decoration ("Official
    # Video") strip down to empty instead of surviving as a bogus song name.
    r"(?:\s*[-–—|:/·•]+\s*|\s+|^)"
    r"(?:official\s+(?:music\s+)?video|official\s+(?:lyric|visuali[sz]er|audio)(?:\s+video)?|"
    r"music\s+video|lyrics?\s+video|video\s+oficial|"
    r"(?:with|w/)\s+lyrics|lyrics\s+on\s+screen|lyrics?|"
    r"remaster(?:ed)?(?:\s+\d{4})?|\d{4}\s+remaster(?:ed)?|"
    r"full\s+hd|hd|hq|uhd|4k|8k|1080p|720p|audio)\s*$",
    re.IGNORECASE,
)

# Emoji and other pictographs uploaders sprinkle into titles.
_EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff️⭐❤]+"
)

_QUOTE_CHARS = "\"“”„‟'‘’"
_SEPARATOR_DEBRIS = " -—–|·•/:\t"


def strip_title_noise(text: str) -> str:
    """Remove the decoration layer from a video title.

    Bracketed noise groups, trailing unbracketed noise phrases, and emoji —
    applied repeatedly because titles stack them ("... (Official Video) [HD]
    w/ lyrics"). Returns the bare "Artist - Title"-ish core.
    """
    prev = None
    while prev != text:
        prev = text
        text = _BRACKETED_NOISE_RE.sub("", text)
        text = _EMOJI_RE.sub(" ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        text = _TRAILING_NOISE_RE.sub("", text)
        text = text.strip().strip(_SEPARATOR_DEBRIS).strip()
    return text


# Phrasing that means "this upload is somebody performing somebody else's
# song" — the uploader is not the artist and the title is not just the song.
_COVER_RE = re.compile(
    r"\b(?:covers?|covered\s+by|cover\s+of|tribute|tribute\s+to|"
    r"performed\s+by|plays?|sings?|rendition|karaoke|remix(?:ed)?\s+by)\b",
    re.IGNORECASE,
)


def looks_like_cover(text: str) -> bool:
    return bool(_COVER_RE.search(text))


_DASH_SPLIT_RE = re.compile(r"\s+[-–—]\s+")
_PIPE_SPLIT_RE = re.compile(r"\s*\|\s*")
_DOUBLE_SLASH_SPLIT_RE = re.compile(r"\s*//\s*")
_QUOTED_TRACK_RE = re.compile(
    rf"^(?P<artist>[^{_QUOTE_CHARS}]{{2,60}}?)\s+[{_QUOTE_CHARS}]"
    rf"(?P<track>[^{_QUOTE_CHARS}]{{1,80}})[{_QUOTE_CHARS}]"
)


def _clean_part(s: str) -> str:
    return s.strip().strip(_QUOTE_CHARS).strip(_SEPARATOR_DEBRIS).strip()


def split_artist_title(text: str) -> Optional[tuple[str, str, str]]:
    """(artist, title, method) from an already-noise-stripped video title.

    Handles the shapes uploaders actually use, in decreasing reliability.
    Returns None when no separator applies — which is the signal that the
    title is ambiguous and needs the LLM tier, NOT a licence to fall back to
    the channel name.
    """
    # "Title // Artist" — note the REVERSED order; the double slash is a
    # distinct convention from a single "/" (which usually joins collaborators).
    parts = _DOUBLE_SLASH_SPLIT_RE.split(text, maxsplit=1)
    if len(parts) == 2:
        title, artist = (_clean_part(p) for p in parts)
        if title and artist:
            return artist, title, "double-slash"

    # 'Artist "Track"' / "Artist 'Track'" — live and one-off uploads quote the
    # song instead of using a separator.
    m = _QUOTED_TRACK_RE.match(text)
    if m:
        artist, track = _clean_part(m.group("artist")), _clean_part(m.group("track"))
        if artist and track:
            return artist, track, "quoted"

    # "Artist - Title" (hyphen / en-dash / em-dash)
    parts = _DASH_SPLIT_RE.split(text, maxsplit=1)
    if len(parts) == 2:
        artist, title = (_clean_part(p) for p in parts)
        if artist and title:
            return artist, title, "dash"

    # "Artist | Title"
    parts = _PIPE_SPLIT_RE.split(text, maxsplit=1)
    if len(parts) == 2:
        artist, title = (_clean_part(p) for p in parts)
        if artist and title:
            return artist, title, "pipe"

    return None


def _strip_topic(channel: str) -> str:
    return re.sub(r"\s*-\s*Topic$", "", (channel or "").strip(), flags=re.IGNORECASE).strip()


# --- LLM layer --------------------------------------------------------------

_LLM_SYSTEM = """\
You identify which song a YouTube upload is, from its title and channel name.

Return ONLY a JSON object, no prose and no code fences:
{"title": str, "artist": str, "isCover": bool, "originalArtist": str|null, "confidence": float}

Rules:
- "title" is the SONG's title alone. Strip decoration ("Official Video", "HD",
  "4K", "Lyrics", "Remastered 2011", emoji) and any artist name.
- "artist" is who is PERFORMING on this recording.
- A channel name is often NOT the artist (labels, aggregators, fan channels,
  personal vlogs). Only use it as the artist when the upload is plainly that
  artist's own channel.
- If this is a cover or tribute, set isCover=true, put the PERFORMING act in
  "artist" (never the uploader's channel unless they are the performers), and
  put the artist who originally released the song in "originalArtist".
- "confidence" is your genuine 0-1 certainty that BOTH title and artist are
  right. Use a value below 0.6 whenever you are guessing, the upload is not a
  song, or the title gives you no reliable way to name the artist. A low
  score is expected and useful — do not inflate it.
"""


def _build_llm_prompt(video_title: str, channel: str) -> str:
    return (
        f"Video title: {video_title!r}\n"
        f"Channel name: {channel!r}\n\n"
        "Identify the song."
    )


def _extract_json(text: str) -> dict:
    """Parse the model's JSON, tolerating code fences and surrounding prose."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def _llm_resolve(video_title: str, channel: str) -> Optional[dict]:
    """One cheap call to the configured Haiku. None when it can't be used.

    Never raises: an unconfigured key, a network failure, or unparseable
    output all mean "no LLM opinion available", which leaves the caller with
    the deterministic confidence — and that fails the resolve step cleanly
    rather than storing a guess.
    """
    from .reconcile.providers import ProviderError, get_provider

    if not settings.provider_key("anthropic"):
        log.info("identity: no anthropic key configured; skipping LLM disambiguation")
        return None
    try:
        provider = get_provider("anthropic")
        response = provider.complete(
            _LLM_SYSTEM,
            [{"role": "user", "text": _build_llm_prompt(video_title, channel)}],
            model=settings.identity_model,
            max_tokens=settings.identity_max_tokens,
        )
        return _extract_json(response.text)
    except (ProviderError, json.JSONDecodeError, ValueError, KeyError) as e:
        log.warning("identity: LLM disambiguation failed: %s", e)
        return None
    except Exception as e:  # noqa: BLE001 — identity must never hard-fail here
        log.warning("identity: LLM disambiguation errored: %s", e)
        return None


def _identity_from_llm(payload: dict) -> Optional[SongIdentity]:
    title = str(payload.get("title") or "").strip()
    artist = str(payload.get("artist") or "").strip()
    if not title or not artist:
        return None
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        return None
    confidence = max(0.0, min(1.0, confidence))
    is_cover = bool(payload.get("isCover"))
    original = str(payload.get("originalArtist") or "").strip() or None
    return SongIdentity(
        title=title,
        artist=artist,
        confidence=confidence,
        method="llm",
        is_cover=is_cover,
        original_artist=original if is_cover else None,
    )


# --- the resolver ------------------------------------------------------------

# Deterministic confidence tiers. A separator split is good but not proof (the
# left side can still be a channel or a featuring credit), so it sits below
# the distributor's own structured metadata.
_CONF_STRUCTURED = 0.95
_CONF_SEPARATOR = 0.8
# No separator: all we have is "some words" plus a channel name. That is not
# an identity, it's a guess — scored below the refusal threshold on purpose.
_CONF_BARE = 0.3

# Above this the deterministic answer is trusted outright and no LLM call is
# made. Below it we pay for one Haiku call before deciding.
_LLM_TRIGGER = _CONF_SEPARATOR


def resolve_identity(
    *,
    video_title: str,
    channel: str = "",
    track: str = "",
    track_artist: str = "",
    llm: Optional[Callable[[str, str], Optional[dict]]] = None,
) -> SongIdentity:
    """Resolve a song identity, or raise :class:`IdentityError`.

    ``llm`` is injectable so tests can drive the ambiguous path without a
    network call; it defaults to :func:`_llm_resolve`.
    """
    raw_title = (video_title or "").strip()
    channel = _strip_topic(channel)
    track = (track or "").strip()
    track_artist = (track_artist or "").strip()
    cover_phrasing = looks_like_cover(raw_title)

    # 1. The distributor's own metadata. Trusted outright — but NOT for a
    #    cover upload, where `artist` is frequently the original artist rather
    #    than whoever is actually playing.
    if track and track_artist and not cover_phrasing:
        return SongIdentity(
            title=strip_title_noise(track) or track,
            artist=track_artist,
            confidence=_CONF_STRUCTURED,
            method="structured",
        )

    # 2. Deterministic title parsing.
    cleaned = strip_title_noise(raw_title)
    deterministic: Optional[SongIdentity] = None
    split = split_artist_title(cleaned) if cleaned else None
    if split is not None:
        artist, title, method = split
        deterministic = SongIdentity(
            title=title, artist=artist, confidence=_CONF_SEPARATOR, method=method
        )
    elif cleaned:
        # No separator. The channel is the only artist candidate and it is a
        # weak one — scored below the threshold so it can never stand alone.
        deterministic = SongIdentity(
            title=cleaned,
            artist=channel or "",
            confidence=_CONF_BARE,
            method="bare-title",
        )

    # A cover never resolves deterministically. "The best Beatles cover ever -
    # Blac Rabbit" splits cleanly on the dash, but the left side is a
    # description and the right side is the performer — the split parses and
    # means nothing. So cover phrasing both forces the escalation AND discards
    # the deterministic answer, leaving nothing to fall back on if the LLM
    # can't help. Refusing is the correct outcome there.
    if cover_phrasing:
        deterministic = None
    ambiguous = cover_phrasing or deterministic is None or deterministic.confidence < _LLM_TRIGGER

    if ambiguous:
        payload = (llm or _llm_resolve)(raw_title, channel)
        if payload is not None:
            resolved = _identity_from_llm(payload)
            if resolved is not None and resolved.confidence >= settings.identity_min_confidence:
                return resolved
            if resolved is not None:
                deterministic = None  # the model itself says it's a guess
                _refuse(raw_title, channel, resolved.confidence)

    if deterministic is None or deterministic.confidence < settings.identity_min_confidence:
        _refuse(
            raw_title,
            channel,
            deterministic.confidence if deterministic else 0.0,
        )
    return deterministic


def _refuse(video_title: str, channel: str, confidence: float) -> None:
    raise IdentityError(
        f"could not confidently identify the song from video title {video_title!r} "
        f"(channel {channel!r}); best confidence {confidence:.2f} < "
        f"{settings.identity_min_confidence:.2f}. Song ids are permanent, so this "
        "is not guessed — re-run with an explicit title and artist."
    )
