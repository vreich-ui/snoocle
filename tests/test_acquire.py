import pytest

from snoocle_server.audio.acquire import (
    AcquisitionError,
    download_audio,
    extract_video_id,
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


def test_download_uses_cache_without_network(tmp_path, monkeypatch):
    from snoocle_server.config import settings

    monkeypatch.setattr(settings, "audio_cache_dir", tmp_path)
    cached = tmp_path / "Some Song [AAAAAAAAAAA].m4a"
    cached.write_bytes(b"fake-audio")
    got = download_audio("AAAAAAAAAAA")
    assert got.from_cache is True
    assert got.path == str(cached)
