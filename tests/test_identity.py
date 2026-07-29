"""Song identity resolution: real-world YouTube titles -> (title, artist).

The song id is permanent (content-hash versioned store), so the contract
under test is two-sided:

- titles the deterministic layer CAN settle must come out clean, with no
  decoration surviving into the slug, and
- titles it can't must escalate to the LLM tier and, failing that, REFUSE —
  never fall back to the channel name and a junk title.

The table below is real YouTube titles, including the three that produced the
broken ids this module exists to fix:
``unknown--official-video``, ``epitaph--eric-clapton-tears-in-heaven-official-video``,
and ``just-kristin--the-best-beatles-cover-ever-...``.
"""

from __future__ import annotations

import pytest

from snoocle_server.identity import (
    IdentityError,
    resolve_identity,
    split_artist_title,
    strip_title_noise,
)
from snoocle_server.schema.song import slugify_song_id


def _no_llm(video_title, channel):
    """Assert the deterministic layer settled it without paying for a call."""
    raise AssertionError(
        f"unexpected LLM escalation for {video_title!r} (channel {channel!r})"
    )


# (video_title, channel, expected_title, expected_artist)
DETERMINISTIC = [
    # --- decoration that used to survive into the slug ----------------------
    (
        "Bob Marley & The Wailers - Three Little Birds (Official Video)",
        "BobMarleyVEVO",
        "Three Little Birds",
        "Bob Marley & The Wailers",
    ),
    # Unbracketed trailing noise — the case the old bracket-only regex missed.
    (
        "Guns N' Roses - Don't Cry Official Video",
        "GunsNRosesVEVO",
        "Don't Cry",
        "Guns N' Roses",
    ),
    # Stacked decoration, mixed bracket styles, and a remaster year.
    (
        "Queen - Bohemian Rhapsody (Official Video Remastered) [4K] HD",
        "Queen Official",
        "Bohemian Rhapsody",
        "Queen",
    ),
    # Spotify-style unbracketed "- Remastered <year>" suffix.
    (
        "The Beatles - Let It Be - Remastered 2009",
        "The Beatles - Topic",
        "Let It Be",
        "The Beatles",
    ),
    # Emoji, and "w/ lyrics" as trailing noise.
    (
        "Fleetwood Mac – Dreams 🌙✨ w/ lyrics",
        "Rumours Fan Channel",
        "Dreams",
        "Fleetwood Mac",
    ),
    # "Title // Artist" — note the REVERSED order.
    ("Motion Sickness // Phoebe Bridgers", "deadoceans", "Motion Sickness", "Phoebe Bridgers"),
    # 'Artist "Track" <venue/context>' — quoted, no separator.
    (
        'Blues Traveler "Hook" at Howard Stern\'s 1996 Birthday Show',
        "Howard Stern",
        "Hook",
        "Blues Traveler",
    ),
    # Pipe separator.
    ("Radiohead | Creep [Official Music Video]", "XLRecordings", "Creep", "Radiohead"),
    # Leading bracketed tag plus trailing noise.
    ("[MV] IU - Through the Night (Lyrics)", "1theK", "Through the Night", "IU"),
    # A legitimate parenthetical that is NOT decoration must survive.
    (
        "Elton John - Don't Let the Sun Go Down on Me (Live at Wembley)",
        "Elton John",
        "Don't Let the Sun Go Down on Me (Live at Wembley)",
        "Elton John",
    ),
]


@pytest.mark.parametrize(
    "video_title,channel,want_title,want_artist",
    DETERMINISTIC,
    ids=[t[0][:48] for t in DETERMINISTIC],
)
def test_deterministic_titles_resolve_without_an_llm_call(
    video_title, channel, want_title, want_artist
):
    got = resolve_identity(video_title=video_title, channel=channel, llm=_no_llm)
    assert (got.title, got.artist) == (want_title, want_artist)
    assert got.confidence >= 0.6


def test_structured_fields_beat_the_video_title():
    """yt-dlp's track/artist is the distributor's own metadata — it wins over
    anything parsed out of a decorated title."""
    got = resolve_identity(
        video_title="Tears In Heaven (Official Video) [Remastered]",
        channel="Epitaph",
        track="Tears in Heaven",
        track_artist="Eric Clapton",
        llm=_no_llm,
    )
    assert (got.title, got.artist) == ("Tears in Heaven", "Eric Clapton")
    assert got.method == "structured"


# --- the ambiguous tier ------------------------------------------------------

# (video_title, channel) pairs the deterministic layer must NOT settle on its
# own. Each previously produced a wrong permanent id.
AMBIGUOUS = [
    # -> "unknown--official-video": noise was the entire parsed title.
    ("Official Video", ""),
    # -> "epitaph--eric-clapton-tears-in-heaven-official-video": a label
    # channel became the artist because no separator was found.
    ("Eric Clapton Tears In Heaven Official Video", "Epitaph"),
    # -> "just-kristin--the-best-beatles-cover-ever-...": cover phrasing, and
    # the uploader is neither the performer nor the original artist.
    ("The best Beatles cover ever - Blac Rabbit", "Just Kristin"),
    # No separator at all; channel is a person, not a band.
    ("Wonderwall acoustic in my bedroom", "daves guitar stuff"),
    # Not a song.
    ("Top 10 Guitar Riffs of All Time", "GuitarWorld"),
]


@pytest.mark.parametrize("video_title,channel", AMBIGUOUS, ids=[t[0][:48] for t in AMBIGUOUS])
def test_ambiguous_titles_escalate_instead_of_guessing(video_title, channel):
    """The deterministic layer must hand these to the LLM tier, not settle
    them by falling back to the channel name."""
    calls = []

    def spy(vt, ch):
        calls.append((vt, ch))
        return None  # no LLM opinion available

    with pytest.raises(IdentityError):
        resolve_identity(video_title=video_title, channel=channel, llm=spy)
    assert calls == [(video_title, channel)]


def test_low_confidence_llm_answer_is_refused_not_stored():
    """A model that says it's guessing is refused. Ids are permanent, so a
    0.4-confidence answer is worse than no answer."""
    payload = {"title": "Something", "artist": "Someone", "isCover": False, "confidence": 0.4}
    with pytest.raises(IdentityError, match="not guessed"):
        resolve_identity(
            video_title="some ambiguous upload",
            channel="a channel",
            llm=lambda *_: payload,
        )


def test_confident_llm_answer_resolves_the_broken_label_channel_case():
    payload = {
        "title": "Tears in Heaven",
        "artist": "Eric Clapton",
        "isCover": False,
        "confidence": 0.95,
    }
    got = resolve_identity(
        video_title="Eric Clapton Tears In Heaven Official Video",
        channel="Epitaph",
        llm=lambda *_: payload,
    )
    assert (got.title, got.artist) == ("Tears in Heaven", "Eric Clapton")
    assert slugify_song_id(got.artist, got.title) == "eric-clapton--tears-in-heaven"


def test_cover_keeps_the_performer_and_records_the_original_artist():
    """For a cover the stored artist is who is PLAYING (here Blac Rabbit) —
    not the uploader's channel and not the original artist, which is carried
    separately so the pipeline can record it as provenance."""
    payload = {
        "title": "Tomorrow Never Knows",
        "artist": "Blac Rabbit",
        "isCover": True,
        "originalArtist": "The Beatles",
        "confidence": 0.9,
    }
    got = resolve_identity(
        video_title="The best Beatles cover ever - Blac Rabbit",
        channel="Just Kristin",
        llm=lambda *_: payload,
    )
    assert got.artist == "Blac Rabbit"
    assert got.is_cover and got.original_artist == "The Beatles"
    assert "just-kristin" not in slugify_song_id(got.artist, got.title)


def test_cover_phrasing_escalates_even_when_a_separator_parses():
    """'The best Beatles cover ever - Blac Rabbit' splits cleanly on the dash,
    but that split is meaningless — the left side is a description, not an
    artist. Cover phrasing must override a successful deterministic parse."""
    split = split_artist_title("The best Beatles cover ever - Blac Rabbit")
    assert split is not None  # the dash really does parse...

    calls = []
    resolve_identity(
        video_title="The best Beatles cover ever - Blac Rabbit",
        channel="Just Kristin",
        llm=lambda vt, ch: calls.append((vt, ch))
        or {"title": "Tomorrow Never Knows", "artist": "Blac Rabbit",
            "isCover": True, "originalArtist": "The Beatles", "confidence": 0.9},
    )
    assert len(calls) == 1  # ...and it was escalated anyway


def test_structured_fields_do_not_shortcut_a_cover():
    """A cover upload's `artist` field is routinely the ORIGINAL artist, so
    the structured-metadata shortcut must not apply to it."""
    calls = []
    resolve_identity(
        video_title="Hurt (Johnny Cash cover)",
        channel="Some Fan",
        track="Hurt",
        track_artist="Nine Inch Nails",
        llm=lambda vt, ch: calls.append((vt, ch))
        or {"title": "Hurt", "artist": "Johnny Cash", "isCover": True,
            "originalArtist": "Nine Inch Nails", "confidence": 0.9},
    )
    assert len(calls) == 1


def test_no_decoration_survives_into_a_slug():
    """The regression the broken ids all shared: a noise token reaching the
    permanent id."""
    for video_title, channel, want_title, want_artist in DETERMINISTIC:
        slug = slugify_song_id(want_artist, want_title)
        assert "official" not in slug, video_title
        assert "-hd" not in slug and "-4k" not in slug, video_title
        assert "lyrics" not in slug, video_title


@pytest.mark.parametrize(
    "raw,want",
    [
        ("Song Name (Official Music Video)", "Song Name"),
        ("Song Name [Official Audio]", "Song Name"),
        ("Song Name - Official Video", "Song Name"),
        ("Song Name | Lyrics", "Song Name"),
        ("Song Name w/ lyrics", "Song Name"),
        ("Song Name (Remastered 2011)", "Song Name"),
        ("Song Name - Remastered 2011", "Song Name"),
        ("Song Name 🔥🎸", "Song Name"),
        ("Song Name (Official Video) [HD] w/ lyrics", "Song Name"),
        ("Official Video", ""),
    ],
)
def test_strip_title_noise(raw, want):
    assert strip_title_noise(raw) == want
