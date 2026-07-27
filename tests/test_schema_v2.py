"""Schema v2: chord/line-level timing, beat grid, video offsets.

All v2 fields are optional — old (v1) documents must keep validating
unchanged (test_v1_song_still_validates). New invariants only bite when the
new fields are actually populated.
"""

import pytest
from pydantic import ValidationError

from snoocle_server.schema import Song
from snoocle_server.schema.song import SCHEMA_VERSION

from test_schema import make_song


def test_schema_version_is_2():
    assert SCHEMA_VERSION == 2


def test_v1_song_still_validates():
    # No schemaVersion, no v2 fields at all -- exactly what's in Firestore today.
    data = make_song()
    song = Song.model_validate(data)
    assert song.schemaVersion == 2  # default for freshly-constructed songs
    # But a document that explicitly claims to be v1 must still decode.
    data["schemaVersion"] = 1
    song_v1 = Song.model_validate(data)
    assert song_v1.lines[0].timeSeconds is None
    assert song_v1.audio.beats == []
    assert song_v1.audio.videoOffsets == {}


def test_chord_placement_time_and_confidence_roundtrip():
    data = make_song()
    data["lines"][0]["chordPlacements"][0].update(
        {"timeSeconds": 13.2, "confidence": 0.9, "beat": {"measure": 1, "beat": 1}}
    )
    song = Song.model_validate(data)
    p = song.lines[0].chordPlacements[0]
    assert p.timeSeconds == 13.2
    assert p.confidence == 0.9
    assert p.beat.measure == 1 and p.beat.beat == 1
    dumped = song.model_dump()
    assert Song.model_validate(dumped) == song


def test_voicing_hint_is_never_chord_validated():
    """A voicingHint of 'x02210' must pass -- it is a display hint, not a
    chord identity. The SAME string as `chord` must still be rejected."""
    data = make_song()
    data["lines"][0]["chordPlacements"][0]["voicingHint"] = "x02210"
    song = Song.model_validate(data)  # must not raise
    assert song.lines[0].chordPlacements[0].voicingHint == "x02210"

    data["lines"][0]["chordPlacements"][0]["chord"] = "x02210"
    with pytest.raises(ValidationError, match="shape"):
        Song.model_validate(data)


def test_chord_times_must_be_nondecreasing_within_a_line():
    data = make_song()
    data["lines"][0]["chordPlacements"][0]["timeSeconds"] = 10.0
    data["lines"][0]["chordPlacements"][1]["timeSeconds"] = 5.0  # goes backwards
    with pytest.raises(ValidationError, match="non-decreasing"):
        Song.model_validate(data)


def test_line_times_must_be_nondecreasing_across_lines():
    data = make_song()
    data["lines"][0]["timeSeconds"] = 20.0
    data["lines"][1]["timeSeconds"] = 10.0  # earlier line, later lineIndex
    with pytest.raises(ValidationError, match="non-decreasing"):
        Song.model_validate(data)


def test_beats_roundtrip_and_cap():
    data = make_song()
    data["audio"]["beats"] = [
        {"time": 0.5, "measure": 1, "beatInMeasure": 1},
        {"time": 1.0, "measure": 1, "beatInMeasure": 2},
    ]
    song = Song.model_validate(data)
    assert len(song.audio.beats) == 2

    data["audio"]["beats"] = [
        {"time": float(i), "measure": i // 4 + 1, "beatInMeasure": i % 4 + 1}
        for i in range(10_001)
    ]
    with pytest.raises(ValidationError, match="too large"):
        Song.model_validate(data)


def test_video_offsets_roundtrip_and_key_validation():
    data = make_song()
    data["audio"]["analyzedVideoId"] = "QDYfEBY9NM4"
    data["audio"]["videoOffsets"] = {"kXYiU_JCYtU": 3.2}
    song = Song.model_validate(data)
    assert song.audio.analyzedVideoId == "QDYfEBY9NM4"
    assert song.audio.videoOffsets["kXYiU_JCYtU"] == 3.2

    data["audio"]["videoOffsets"] = {"not-a-video-id!!": 3.2}
    with pytest.raises(ValidationError, match="implausible"):
        Song.model_validate(data)


def test_analyzed_video_id_validated_like_youtube_video_id():
    data = make_song()
    data["audio"]["analyzedVideoId"] = "bad!!"
    with pytest.raises(ValidationError, match="implausible"):
        Song.model_validate(data)
