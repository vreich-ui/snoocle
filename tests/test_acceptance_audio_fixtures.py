"""Hermetic tests for scripts/acceptance.py's online audio-fixtures mode.

The mode itself needs real YouTube egress, ffmpeg, the chord-cnn-lstm model
weights, and a configured LLM provider — none of which this suite has or
should require. What's tested here is everything that CAN be exercised
without a network call: fixture-file loading/validation, and the five
per-song assertions (resolve id / MIR engine+timeline / schema validity /
snap-match ratio / get_song) against fabricated API and run-trace payloads
shaped exactly like the real server's responses.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO / "scripts" / "acceptance.py"


def _load_acceptance_module():
    spec = importlib.util.spec_from_file_location("acceptance_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module defines dataclasses, and their field
    # resolution looks itself up via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def acc():
    return _load_acceptance_module()


# --- fixture-file loading -----------------------------------------------------


def test_load_audio_fixtures_happy_path(acc, tmp_path):
    p = tmp_path / "fixtures.json"
    p.write_text(json.dumps([
        {"url": "https://www.youtube.com/watch?v=abc12345678", "title": "Let It Be", "artist": "The Beatles"},
    ]))
    fixtures = acc.load_audio_fixtures(p)
    assert fixtures == [
        {"url": "https://www.youtube.com/watch?v=abc12345678", "title": "Let It Be", "artist": "The Beatles"}
    ]


def test_load_audio_fixtures_missing_file(acc, tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        acc.load_audio_fixtures(tmp_path / "nope.json")


def test_load_audio_fixtures_bad_json(acc, tmp_path):
    p = tmp_path / "fixtures.json"
    p.write_text("{not json")
    with pytest.raises(SystemExit, match="not valid JSON"):
        acc.load_audio_fixtures(p)


def test_load_audio_fixtures_not_a_list(acc, tmp_path):
    p = tmp_path / "fixtures.json"
    p.write_text(json.dumps({"url": "x", "title": "y", "artist": "z"}))
    with pytest.raises(SystemExit, match="non-empty JSON array"):
        acc.load_audio_fixtures(p)


def test_load_audio_fixtures_empty_list(acc, tmp_path):
    p = tmp_path / "fixtures.json"
    p.write_text("[]")
    with pytest.raises(SystemExit, match="non-empty JSON array"):
        acc.load_audio_fixtures(p)


def test_load_audio_fixtures_missing_field(acc, tmp_path):
    p = tmp_path / "fixtures.json"
    p.write_text(json.dumps([{"url": "https://x", "title": "Y"}]))  # no artist
    with pytest.raises(SystemExit, match="artist"):
        acc.load_audio_fixtures(p)


def test_load_audio_fixtures_rejects_unedited_placeholder(acc, tmp_path):
    """The shipped example file must not silently run against fake URLs."""
    example = REPO / "scripts" / "audio_fixtures.example.json"
    with pytest.raises(SystemExit, match="placeholder"):
        acc.load_audio_fixtures(example)


# --- check_resolve_id ---------------------------------------------------------


def test_check_resolve_id_pass(acc):
    fixture = {"url": "u", "title": "Tears in Heaven", "artist": "Eric Clapton"}
    body = {"songId": "eric-clapton--tears-in-heaven", "steps": {"resolve": "ok: ..."}}
    r = acc.check_resolve_id(fixture, body)
    assert r.passed


def test_check_resolve_id_fail_on_mismatch(acc):
    fixture = {"url": "u", "title": "Tears in Heaven", "artist": "Eric Clapton"}
    body = {"songId": "epitaph--eric-clapton-tears-in-heaven-official-video", "steps": {}}
    r = acc.check_resolve_id(fixture, body)
    assert not r.passed
    assert "eric-clapton--tears-in-heaven" in r.detail


# --- check_mir -----------------------------------------------------------------


def test_check_mir_pass_with_primary_engine(acc):
    run_trace = {"mir": {"engines": {"chords": "chord-cnn-lstm", "beats": "madmom"},
                          "chordTimeline": [{"start": 0.0, "end": 1.0, "chord": "C"}]}}
    r = acc.check_mir(True, run_trace)
    assert r.passed


def test_check_mir_fails_on_fallback_engine(acc):
    run_trace = {"mir": {"engines": {"chords": "chroma-template-fallback"},
                          "chordTimeline": [{"start": 0.0, "end": 1.0, "chord": "C"}]}}
    r = acc.check_mir(True, run_trace)
    assert not r.passed
    assert "FALLBACK" in r.detail


def test_check_mir_fails_on_empty_timeline(acc):
    run_trace = {"mir": {"engines": {"chords": "chord-cnn-lstm"}, "chordTimeline": []}}
    r = acc.check_mir(True, run_trace)
    assert not r.passed


def test_check_mir_fails_when_run_fetch_failed(acc):
    r = acc.check_mir(False, None)
    assert not r.passed
    assert "could not fetch" in r.detail


def test_check_mir_fails_when_mir_never_ran(acc):
    r = acc.check_mir(True, {"mir": None})
    assert not r.passed
    assert "did not run" in r.detail


# --- check_song_valid ----------------------------------------------------------


def _minimal_valid_song() -> dict:
    return {
        "id": "the-beatles--let-it-be",
        "metadata": {"title": "Let It Be", "artist": "The Beatles"},
        "lines": [
            {"lineIndex": 0, "lyrics": "let it be",
             "chordPlacements": [{"charIndex": 0, "chord": "C"}]},
        ],
        "provenance": [{"timestamp": "2026-01-01T00:00:00Z", "actor": "test", "action": "created"}],
    }


def test_check_song_valid_pass(acc):
    r = acc.check_song_valid(_minimal_valid_song())
    assert r.passed


def test_check_song_valid_fails_on_shape_chord(acc):
    song = _minimal_valid_song()
    song["lines"][0]["chordPlacements"][0]["chord"] = "x02210"  # fretboard shape, not harmony
    r = acc.check_song_valid(song)
    assert not r.passed


def test_check_song_valid_fails_on_garbage(acc):
    r = acc.check_song_valid({"not": "a song"})
    assert not r.passed


# --- check_snap_match ------------------------------------------------------------


def _song_with_snap_note(notes: str, confidence=None) -> dict:
    song = _minimal_valid_song()
    entry = {"timestamp": "2026-01-01T00:00:00Z", "actor": "snoocle-server/timing",
             "action": "timing-snap", "notes": notes}
    if confidence is not None:
        entry["confidence"] = confidence
    song["provenance"].append(entry)
    return song


def test_check_snap_match_pass_above_threshold(acc):
    song = _song_with_snap_note("matched 8/10 chord placement(s) to the MIR timeline; 5/5 line(s) timed")
    r = acc.check_snap_match(song)
    assert r.passed
    assert "8/10" in r.detail


def test_check_snap_match_fail_below_threshold(acc):
    song = _song_with_snap_note("matched 2/10 chord placement(s) to the MIR timeline; 1/5 line(s) timed")
    r = acc.check_snap_match(song)
    assert not r.passed


def test_check_snap_match_exactly_at_threshold_passes(acc):
    song = _song_with_snap_note("matched 5/10 chord placement(s) to the MIR timeline; 3/5 line(s) timed")
    r = acc.check_snap_match(song)
    assert r.passed  # >= 50%, not strictly >


def test_check_snap_match_fails_with_no_provenance_entry(acc):
    song = _minimal_valid_song()
    r = acc.check_snap_match(song)
    assert not r.passed
    assert "no timing-snap" in r.detail


def test_check_snap_match_falls_back_to_confidence_when_notes_unparseable(acc):
    song = _song_with_snap_note("some future note format the regex doesn't know", confidence=0.75)
    r = acc.check_snap_match(song)
    assert r.passed
    assert "75%" in r.detail


def test_check_snap_match_fails_when_neither_notes_nor_confidence_parse(acc):
    song = _song_with_snap_note("unparseable and no confidence either")
    r = acc.check_snap_match(song)
    assert not r.passed


def test_check_snap_match_accepts_a_carry_forward_entry(acc):
    """A run that reused the prior version's audio analysis (scope.listen=off)
    records `timing-carry-forward` instead of `timing-snap`. The check is about
    whether the placements ended up timed, not which pass timed them — and the
    real pass's real notes must parse, so this builds one rather than
    hand-writing the string."""
    from snoocle_server.timing.carry_forward import carry_forward_timing
    from tests.test_timing_carry_forward import _gold_song, _reconciled_without_timing

    carried, _ = carry_forward_timing(_reconciled_without_timing(), _gold_song())
    r = acc.check_snap_match(carried.model_dump(mode="json"))
    assert r.passed, r.detail
    assert "10/13" in r.detail


def test_check_snap_match_uses_the_last_entry_when_several_exist(acc):
    """A song re-analyzed more than once accumulates provenance across runs
    (see pipeline._step_store) — only the most recent snap pass counts."""
    song = _song_with_snap_note("matched 1/10 chord placement(s) to the MIR timeline; 1/5 line(s) timed")
    song["provenance"].append({
        "timestamp": "2026-01-02T00:00:00Z", "actor": "snoocle-server/timing",
        "action": "timing-snap",
        "notes": "matched 9/10 chord placement(s) to the MIR timeline; 5/5 line(s) timed",
    })
    r = acc.check_snap_match(song)
    assert r.passed
    assert "9/10" in r.detail


# --- SongReport ------------------------------------------------------------------


def test_song_report_passed_requires_http_ok_and_all_checks(acc):
    fixture = {"url": "u", "title": "t", "artist": "a"}
    ok_report = acc.SongReport(
        fixture=fixture, http_ok=True,
        checks=[acc.StepResult("x", True, ""), acc.StepResult("y", True, "")],
    )
    assert ok_report.passed

    one_failing = acc.SongReport(
        fixture=fixture, http_ok=True,
        checks=[acc.StepResult("x", True, ""), acc.StepResult("y", False, "")],
    )
    assert not one_failing.passed

    http_failed = acc.SongReport(fixture=fixture, http_ok=False, error="boom")
    assert not http_failed.passed


def test_song_report_total_seconds_sums_the_three_phases(acc):
    fixture = {"url": "u", "title": "t", "artist": "a"}
    r = acc.SongReport(fixture=fixture, http_ok=True,
                       analyze_seconds=10.0, run_seconds=1.5, mcp_seconds=0.5)
    assert r.total_seconds == pytest.approx(12.0)


# --- CLI wiring --------------------------------------------------------------


def test_audio_fixtures_flags_parse_on_the_real_parser(acc):
    args = acc.build_arg_parser().parse_args([
        "--audio-fixtures", "fixtures.json", "--base-url", "https://x.run.app",
        "--token", "tok123", "--timeout-seconds", "600",
    ])
    assert args.audio_fixtures == "fixtures.json"
    assert args.base_url == "https://x.run.app"
    assert args.token == "tok123"
    assert args.timeout_seconds == 600.0


def test_default_mode_flags_still_parse(acc):
    """The pre-existing --offline/--providers/--title/--artist flags are
    unaffected by the new ones."""
    args = acc.build_arg_parser().parse_args(["--offline", "--title", "X", "--artist", "Y"])
    assert args.offline is True
    assert args.title == "X"
    assert args.artist == "Y"
    assert args.audio_fixtures is None
