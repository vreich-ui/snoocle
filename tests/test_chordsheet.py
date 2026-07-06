from pathlib import Path

from snoocle_server.discovery.chordsheet import parse_chord_sheet
from snoocle_server.discovery.service import candidate_from_text

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return (FIXTURES / name).read_text()


def test_chord_over_lyrics_layout():
    sheet = parse_chord_sheet(load("sheet_over_lyrics.txt"))
    assert sheet.declared_capo == 0
    assert sheet.declared_key == "C"
    assert "Verse 1" in sheet.sections_hint and "Chorus" in sheet.sections_hint
    first = sheet.lines[0]
    assert first.lyrics.startswith("When I find myself")
    chords = [p.chord for p in first.chordPlacements]
    assert chords == ["C", "G", "Am", "F"]
    # placement column mapping: "C" at col 0, "G" above "myself"-ish region
    assert first.chordPlacements[0].charIndex == 0
    assert first.chordPlacements[1].charIndex == 15
    # instrumental solo lines parsed as empty-lyric ordinal slots
    solo_lines = [l for l in sheet.lines if not l.lyrics and l.chordPlacements]
    assert len(solo_lines) == 2
    assert [p.chord for p in solo_lines[0].chordPlacements] == ["C", "G", "Am", "F"]


def test_capo_sheet_normalized_to_sounding():
    sheet = parse_chord_sheet(load("sheet_capo.txt"))
    assert sheet.declared_capo == 2
    first_chords = [p.chord for p in sheet.lines[0].chordPlacements]
    # shapes Em7 G with capo 2 sound as F#m7 A
    assert first_chords == ["F#m7", "A"]
    second_chords = [p.chord for p in sheet.lines[1].chordPlacements]
    assert second_chords == ["Esus4", "B7sus4"]  # Dsus4 -> Esus4, A7sus4 -> B7sus4


def test_inline_layout():
    sheet = parse_chord_sheet(load("sheet_inline.txt"))
    first = sheet.lines[0]
    assert first.lyrics == "When I find myself in times of trouble"
    assert [p.chord for p in first.chordPlacements] == ["C", "G"]
    assert first.chordPlacements[0].charIndex == 0
    assert first.chordPlacements[1].charIndex == len("When I find myself in ")


def test_candidate_from_text_confidence_and_capo_note():
    cand = candidate_from_text(load("sheet_capo.txt"), "web-1", url="https://example.com/x")
    assert cand is not None
    assert cand.declaredCapo == 2
    assert "transposed to" in (cand.notes or "")
    assert 0 < cand.confidence < 1
    assert cand.chord_vocabulary()  # non-empty


def test_garbage_text_is_not_plausible():
    assert candidate_from_text("just some random prose\nwith nothing musical\n" * 10, "web-9") is None
