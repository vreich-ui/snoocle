import httpx
import pytest

from snoocle_server import config as config_mod
from snoocle_server.mir.base import MirAnalysis
from snoocle_server.schema.song import Song
from snoocle_server.timing import lrc as lrc_mod
from snoocle_server.timing.lrc import LrcLine, apply_lrc, fetch_lrc, match_lrc_to_lines

from test_schema import make_song

_SAMPLE_LRC = """[00:12.20]When I find myself in times of trouble
[00:18.90]Mother Mary comes to me
[00:25.00]Speaking words of wisdom
"""


class _FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _lrclib_enabled(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "lrclib_enabled", True)


def test_fetch_lrc_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(config_mod.settings, "lrclib_enabled", False)

    def boom(*a, **k):  # noqa: ANN001
        raise AssertionError("must not call the network when disabled")

    monkeypatch.setattr(lrc_mod.httpx, "get", boom)
    assert fetch_lrc("X", "Y", 200.0) is None


def test_fetch_lrc_uses_exact_get_endpoint_first(monkeypatch):
    def fake_get(url, params=None, timeout=None, headers=None):
        assert url.endswith("/api/get")
        assert params["track_name"] == "Let It Be"
        assert params["artist_name"] == "The Beatles"
        return _FakeResponse(200, {"syncedLyrics": _SAMPLE_LRC})

    monkeypatch.setattr(lrc_mod.httpx, "get", fake_get)
    result = fetch_lrc("Let It Be", "The Beatles", 240.0)
    assert result is not None
    assert [l.text for l in result] == [
        "When I find myself in times of trouble",
        "Mother Mary comes to me",
        "Speaking words of wisdom",
    ]
    assert result[0].time == pytest.approx(12.2)
    assert result[1].time == pytest.approx(18.9)


def test_fetch_lrc_falls_back_to_search_within_duration_tolerance(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(url)
        if url.endswith("/api/get"):
            return _FakeResponse(404, {})
        assert url.endswith("/api/search")
        return _FakeResponse(
            200,
            [
                {"duration": 500, "syncedLyrics": _SAMPLE_LRC},  # too far off
                {"duration": 241, "syncedLyrics": _SAMPLE_LRC},  # within tolerance
            ],
        )

    monkeypatch.setattr(lrc_mod.httpx, "get", fake_get)
    result = fetch_lrc("Let It Be", "The Beatles", 240.0)
    assert calls == [f"{lrc_mod._LRCLIB_BASE}/get", f"{lrc_mod._LRCLIB_BASE}/search"]
    assert result is not None
    assert len(result) == 3


def test_fetch_lrc_search_fallback_rejects_out_of_tolerance_durations(monkeypatch):
    def fake_get(url, params=None, timeout=None, headers=None):
        if url.endswith("/api/get"):
            return _FakeResponse(404, {})
        return _FakeResponse(200, [{"duration": 500, "syncedLyrics": _SAMPLE_LRC}])

    monkeypatch.setattr(lrc_mod.httpx, "get", fake_get)
    assert fetch_lrc("Let It Be", "The Beatles", 240.0) is None


def test_fetch_lrc_returns_none_on_network_error(monkeypatch):
    def raise_error(*a, **k):  # noqa: ANN001
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(lrc_mod.httpx, "get", raise_error)
    assert fetch_lrc("X", "Y", 100.0) is None


def test_fetch_lrc_returns_none_when_neither_endpoint_has_synced_lyrics(monkeypatch):
    def fake_get(url, params=None, timeout=None, headers=None):
        if url.endswith("/api/get"):
            return _FakeResponse(200, {"plainLyrics": "no sync here"})
        return _FakeResponse(200, [])

    monkeypatch.setattr(lrc_mod.httpx, "get", fake_get)
    assert fetch_lrc("X", "Y", 100.0) is None


def test_match_lrc_to_lines_fuzzy_and_monotonic():
    song = Song.model_validate(make_song())
    parsed = [
        LrcLine(time=12.2, text="When I find myself, in times of trouble!"),
        LrcLine(time=18.9, text="Mother Mary comes to me"),
    ]
    matches = match_lrc_to_lines(parsed, song)
    assert set(matches) == {0, 1}
    assert matches[0][0] == pytest.approx(12.2)
    assert matches[0][1] >= 0.75
    assert matches[1][0] == pytest.approx(18.9)


def test_match_lrc_to_lines_below_threshold_is_unmatched():
    song = Song.model_validate(make_song())
    parsed = [LrcLine(time=1.0, text="completely unrelated text here")]
    matches = match_lrc_to_lines(parsed, song)
    assert matches == {}


def test_apply_lrc_overrides_line_times_and_redistributes_chords():
    song = Song.model_validate(make_song())
    matches = {0: (12.0, 0.9), 1: (20.0, 0.85)}
    result = apply_lrc(song, mir=None, matches=matches)

    line0 = result.lines[0]
    assert line0.timeSeconds == 12.0
    assert line0.confidence == 0.9
    # chords redistributed proportionally between 12.0 and the next line's 20.0
    lyrics_len = len(line0.lyrics)
    for p in line0.chordPlacements:
        expected = 12.0 + (p.charIndex / lyrics_len) * (20.0 - 12.0)
        assert p.timeSeconds == pytest.approx(expected)

    line1 = result.lines[1]
    assert line1.timeSeconds == 20.0
    assert [(sp.lineIndex, sp.time) for sp in result.audio.syncMap] == [(0, 12.0), (1, 20.0)]
    assert result.provenance[-1].action == "lrc-align"
    assert result.provenance[-1].sources == ["lrclib.net"]


def test_apply_lrc_is_noop_with_no_matches():
    song = Song.model_validate(make_song())
    result = apply_lrc(song, mir=None, matches={})
    assert result is song


def test_apply_lrc_snaps_redistributed_chords_to_beat_grid_when_mir_available():
    from snoocle_server.mir.base import Beat

    data = {
        "id": "x--y",
        "metadata": {"title": "Y", "artist": "X"},
        "lines": [
            {"lineIndex": 0, "lyrics": "abcd", "chordPlacements": [{"charIndex": 2, "chord": "C"}]},
            {"lineIndex": 1, "lyrics": "ef", "chordPlacements": []},
        ],
    }
    song = Song.model_validate(data)
    # frac = 2/4 = 0.5 -> t = 10.0 + 0.5*(14.0-10.0) = 12.0 exactly, matching a
    # real MIR beat at 12.0 -- must snap and attach a BeatRef.
    mir = MirAnalysis(beats=[Beat(time=12.0, position=1), Beat(time=12.5, position=0)])
    matches = {0: (10.0, 0.9), 1: (14.0, 0.85)}

    result = apply_lrc(song, mir=mir, matches=matches)
    placement = result.lines[0].chordPlacements[0]
    assert placement.timeSeconds == pytest.approx(12.0)
    assert placement.beat is not None
    assert placement.beat.measure == 1 and placement.beat.beat == 1.0
