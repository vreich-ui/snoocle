import pytest
from pydantic import ValidationError

from snoocle_server.schema import Song
from snoocle_server.schema.song import slugify_song_id


def make_song(**overrides):
    base = {
        "id": "the-beatles--let-it-be",
        "metadata": {"title": "Let It Be", "artist": "The Beatles"},
        "audio": {
            "youtubeVideoId": "QDYfEBY9NM4",
            "syncMap": [{"lineIndex": 0, "time": 13.2}, {"lineIndex": 1, "time": 18.9}],
        },
        "sections": [
            {
                "sectionIndex": 0,
                "name": "Verse 1",
                "kind": "verse",
                "startLineIndex": 0,
                "endLineIndex": 1,
                "startTime": 13.0,
                "endTime": 25.0,
            }
        ],
        "lines": [
            {
                "lineIndex": 0,
                "lyrics": "When I find myself in times of trouble",
                "chordPlacements": [
                    {"charIndex": 7, "chord": "C"},
                    {"charIndex": 23, "chord": "G"},
                    {"charIndex": 32, "chord": "Am"},
                ],
            },
            {
                "lineIndex": 1,
                "lyrics": "Mother Mary comes to me",
                "chordPlacements": [{"charIndex": 0, "chord": "F"}],
            },
        ],
        "provenance": [
            {
                "timestamp": "2026-07-06T00:00:00Z",
                "actor": "test",
                "action": "created",
                "confidence": 0.9,
            }
        ],
    }
    base.update(overrides)
    return base


def test_valid_song_roundtrip():
    song = Song.model_validate(make_song())
    dumped = song.model_dump()
    assert dumped["audio"]["youtubeVideoId"] == "QDYfEBY9NM4"
    assert Song.model_validate(dumped) == song


def test_rejects_shape_chord():
    data = make_song()
    data["lines"][0]["chordPlacements"][0]["chord"] = "x02210"
    with pytest.raises(ValidationError, match="shape"):
        Song.model_validate(data)


def test_rejects_charindex_beyond_line():
    data = make_song()
    data["lines"][1]["chordPlacements"] = [{"charIndex": 99, "chord": "F"}]
    with pytest.raises(ValidationError, match="beyond"):
        Song.model_validate(data)


def test_rejects_noncontiguous_line_indexes():
    data = make_song()
    data["lines"][1]["lineIndex"] = 5
    with pytest.raises(ValidationError, match="contiguous"):
        Song.model_validate(data)


def test_rejects_overlapping_sections():
    data = make_song()
    data["sections"].append(
        {
            "sectionIndex": 1,
            "name": "Verse 1 again",
            "kind": "verse",
            "startLineIndex": 1,
            "endLineIndex": 1,
        }
    )
    with pytest.raises(ValidationError, match="overlaps"):
        Song.model_validate(data)


def test_rejects_bad_video_id():
    data = make_song()
    data["audio"]["youtubeVideoId"] = "not-a-video-id!!"
    with pytest.raises(ValidationError, match="implausible"):
        Song.model_validate(data)


def test_empty_lyrics_line_allows_ordinal_slots():
    data = make_song()
    data["lines"].append(
        {
            "lineIndex": 2,
            "lyrics": "",
            "chordPlacements": [
                {"charIndex": 0, "chord": "C"},
                {"charIndex": 1, "chord": "G"},
            ],
        }
    )
    Song.model_validate(data)


def test_slugify():
    assert slugify_song_id("The Beatles", "Let It Be") == "the-beatles--let-it-be"
    assert slugify_song_id("AC/DC", "T.N.T.") == "ac-dc--t-n-t"
