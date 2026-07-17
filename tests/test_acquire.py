import pytest

from snoocle_server.audio.acquire import (
    AcquisitionError,
    YouTubeAuthError,
    download_audio,
    extract_video_id,
    parse_dash_title,
    pick_best_video,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=5", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_extract_video_id(raw, expected):
    assert extract_video_id(raw) == expected


def test_extract_video_id_rejects_garbage():
    with pytest.raises(AcquisitionError):
        extract_video_id("not a url at all!!")


def test_pick_best_video_prefers_official_studio_recording():
    entries = [
        {"id": "cover0000000", "title": "Let It Be - amazing acoustic COVER", "duration": 240},
        {"id": "lesson000000", "title": "How to play Let It Be - guitar lesson", "duration": 600},
        {"id": "official0000", "title": "The Beatles - Let It Be (Official Audio)", "channel": "The Beatles", "duration": 243},
        {"id": "tooshort0000", "title": "Let It Be The Beatles", "duration": 20},
    ]
    best = pick_best_video(entries, "Let It Be", "The Beatles")
    assert best["id"] == "official0000"


class _FailingYDL:
    def __init__(self, message):
        self._message = message

    def __call__(self, opts):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, *a, **k):
        raise RuntimeError(self._message)


@pytest.mark.parametrize(
    "message, expected",
    [
        ("ERROR: [youtube] x: Sign in to confirm you're not a bot. Use --cookies ...", YouTubeAuthError),
        ("ERROR: [youtube] x: Sign in to confirm your age", YouTubeAuthError),
        ("ERROR: [youtube] x: Requested format is not available", AcquisitionError),
        ("Unable to download webpage: connection reset", AcquisitionError),
    ],
)
def test_download_classifies_youtube_auth_failures(tmp_path, monkeypatch, message, expected):
    """Auth failures (bot-check, dead cookies, age gate) surface as
    YouTubeAuthError so clients can offer the Reconnect YouTube action;
    everything else stays a plain AcquisitionError."""
    import yt_dlp

    from snoocle_server.config import settings

    monkeypatch.setattr(settings, "audio_cache_dir", tmp_path)
    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FailingYDL(message))
    with pytest.raises(expected) as excinfo:
        download_audio("AAAAAAAAAAA")
    if expected is AcquisitionError:
        assert not isinstance(excinfo.value, YouTubeAuthError)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Amy Winehouse - Back To Black", ("Amy Winehouse", "Back To Black")),
        ("Queen – Bohemian Rhapsody (Official Video)", ("Queen", "Bohemian Rhapsody")),
        ("Some Jam", None),
    ],
)
def test_parse_dash_title(text, expected):
    assert parse_dash_title(text) == expected


def test_download_uses_cache_without_network(tmp_path, monkeypatch):
    from snoocle_server.config import settings

    monkeypatch.setattr(settings, "audio_cache_dir", tmp_path)
    cached = tmp_path / "Some Song [AAAAAAAAAAA].m4a"
    cached.write_bytes(b"fake-audio")
    got = download_audio("AAAAAAAAAAA")
    assert got.from_cache is True
    assert got.path == str(cached)
