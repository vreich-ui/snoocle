import pytest

from snoocle_server.chords import (
    ChordParseError,
    looks_like_shape,
    normalize_chord,
    parse_chord,
    shape_to_sounding,
    transpose_chord,
    validate_stored_chord,
)


@pytest.mark.parametrize(
    "raw,canonical",
    [
        ("C", "C"),
        ("Cmaj", "C"),
        ("Am", "Am"),
        ("Amin", "Am"),
        ("A-", "Am"),
        ("F#m7", "F#m7"),
        ("Gbmaj7", "F#maj7"),
        ("Bbm7b5", "A#m7b5"),
        ("Dsus4", "Dsus4"),
        ("Dsus", "Dsus4"),
        ("E7sus4", "E7sus4"),
        ("Cadd9", "Cadd9"),
        ("C/G", "C/G"),
        ("Am7/G", "Am7/G"),
        ("B♭", "A#"),
        ("C♯m", "C#m"),
        ("Caug", "Caug"),
        ("C+", "Caug"),
        ("Cdim7", "Cdim7"),
        ("C6/9", "C6/9"),
        ("C7#9", "C7#9"),
        ("C7(b9)", "C7b9"),
    ],
)
def test_parse_and_canonicalize(raw, canonical):
    assert normalize_chord(raw) == canonical


@pytest.mark.parametrize(
    "raw,flat",
    [
        ("Bb", "Bb"),
        ("Gbmaj7", "Gbmaj7"),
        ("A#m", "Bbm"),
    ],
)
def test_flat_spelling(raw, flat):
    assert normalize_chord(raw, prefer_flats=True) == flat


@pytest.mark.parametrize("bad", ["x02210", "0-2-2-1-0-0", "e|---3---", "H7", "?", "", "chord"])
def test_rejects_non_chords(bad):
    with pytest.raises(ChordParseError):
        parse_chord(bad)


def test_rejects_no_chord_marker_for_storage():
    with pytest.raises(ChordParseError):
        validate_stored_chord("N.C.")


def test_shape_detection():
    assert looks_like_shape("x02210")
    assert looks_like_shape("0-2-2-1-0-0")
    assert not looks_like_shape("Am7")


def test_transpose():
    assert transpose_chord("Am", 5) == "Dm"
    assert transpose_chord("C/G", 2) == "D/A"
    assert transpose_chord("B", 1) == "C"
    assert transpose_chord("Cm7", -1) == "Bm7"


def test_capo_shape_to_sounding():
    # The core normalization rule: capo 5 + Am shape sounds as Dm.
    assert shape_to_sounding("Am", 5) == "Dm"
    assert shape_to_sounding("G", 2) == "A"
    assert shape_to_sounding("Em7", 3) == "Gm7"
    assert shape_to_sounding("C", 0) == "C"
    with pytest.raises(ValueError):
        shape_to_sounding("C", 12)
