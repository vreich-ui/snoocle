"""Master plan B6: deterministic ChordPro/txt/json export, and its round trip
back through the existing generic chord-sheet parser."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server.api import app
from snoocle_server.discovery.chordsheet import parse_chord_sheet
from snoocle_server.export import to_chordpro, to_txt
from snoocle_server.schema import Song
from snoocle_server.store.memory import InMemorySongRepository

from test_schema import make_song

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch):
    store = InMemorySongRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: store)
    return store


def _song() -> Song:
    return Song.model_validate(
        make_song(metadata={"title": "Let It Be", "artist": "The Beatles", "key": "C major", "bpm": 76.0})
    )


# --- pure serializer tests ---------------------------------------------------


def test_to_txt_golden_output():
    assert to_txt(_song()) == (
        "[Verse 1]\n"
        "When I [C]find myself in t[G]imes of t[Am]rouble\n"
        "[F]Mother Mary comes to me\n"
    )


def test_to_chordpro_golden_output():
    assert to_chordpro(_song()) == (
        "{title: Let It Be}\n"
        "{artist: The Beatles}\n"
        "{key: C major}\n"
        "{tempo: 76}\n"
        "\n"
        "[Verse 1]\n"
        "When I [C]find myself in t[G]imes of t[Am]rouble\n"
        "[F]Mother Mary comes to me\n"
    )


def test_to_chordpro_omits_banner_when_no_metadata():
    song = Song.model_validate(make_song(metadata={"title": "", "artist": ""}))
    # no title/artist/key/bpm -> no header block, straight to the sheet body
    assert to_chordpro(song) == to_txt(song)


def test_instrumental_line_serializes_as_space_joined_chords():
    song = Song.model_validate(
        make_song(
            audio={},
            sections=[],
            lines=[
                {
                    "lineIndex": 0,
                    "lyrics": "",
                    "chordPlacements": [
                        {"charIndex": 0, "chord": "C"},
                        {"charIndex": 1, "chord": "Am"},
                        {"charIndex": 2, "chord": "F"},
                        {"charIndex": 3, "chord": "G"},
                    ],
                }
            ],
        )
    )
    assert to_txt(song) == "[C] [Am] [F] [G]\n"


def test_export_round_trips_through_the_generic_chord_sheet_parser():
    song = _song()
    for text in (to_txt(song), to_chordpro(song)):
        parsed = parse_chord_sheet(text)
        assert [ln.lyrics for ln in parsed.lines] == [ln.lyrics for ln in song.lines]
        assert [[(p.charIndex, p.chord) for p in ln.chordPlacements] for ln in parsed.lines] == [
            [(p.charIndex, p.chord) for p in ln.chordPlacements] for ln in song.lines
        ]
        assert parsed.sections_hint == [s.name for s in song.sections]


# --- API endpoint -------------------------------------------------------------


def _seed(store, song: Song) -> None:
    store.save(song, "seed")


def test_export_chordpro_endpoint(isolated_store):
    song = _song()
    _seed(isolated_store, song)
    r = client.get(f"/v1/songs/{song.id}/export?format=chordpro")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert r.headers["content-disposition"] == f'attachment; filename="{song.id}.cho"'
    assert r.text == to_chordpro(song)


def test_export_txt_endpoint(isolated_store):
    song = _song()
    _seed(isolated_store, song)
    r = client.get(f"/v1/songs/{song.id}/export?format=txt")
    assert r.status_code == 200
    assert r.text == to_txt(song)


def test_export_json_endpoint(isolated_store):
    song = _song()
    _seed(isolated_store, song)
    r = client.get(f"/v1/songs/{song.id}/export?format=json")
    assert r.status_code == 200
    assert r.json() == song.model_dump()


def test_export_defaults_to_chordpro(isolated_store):
    song = _song()
    _seed(isolated_store, song)
    r = client.get(f"/v1/songs/{song.id}/export")
    assert r.status_code == 200
    assert r.text == to_chordpro(song)


def test_export_404_when_song_missing():
    r = client.get("/v1/songs/nope--nothing/export")
    assert r.status_code == 404


def test_export_rejects_unknown_format(isolated_store):
    song = _song()
    _seed(isolated_store, song)
    r = client.get(f"/v1/songs/{song.id}/export?format=xml")
    assert r.status_code == 422
