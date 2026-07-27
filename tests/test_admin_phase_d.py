"""Phase D admin surfaces (master plan D1, D3, D4, D5).

Server-side coverage of what the server owns: the timing-coverage metrics the
scorecard columns read, and the assets/wiring the admin shell needs for the
Review tab, the queue dashboard and the version chips. The interactive
behaviour of those panes is vanilla JS in `ui/admin-d.js`; what is asserted
here is that it is served and hooked up, and that the data it reads has the
shape it expects.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from snoocle_server.api import app
from snoocle_server.eval.metrics import score_song, timing_coverage
from snoocle_server.schema import Song
from snoocle_server.timing.confidence import build_review_queue

client = TestClient(app)


def _song(*, chord_times=None, line_times=None, confidences=None) -> Song:
    """A two-line song; each argument is a per-chord / per-line list of values
    where None means 'this one has no value'."""
    chord_times = chord_times if chord_times is not None else [None, None]
    line_times = line_times if line_times is not None else [None, None]
    confidences = confidences if confidences is not None else [None, None]

    lines = []
    for i in range(2):
        placement: dict = {"charIndex": 0, "chord": "C" if i == 0 else "G"}
        if chord_times[i] is not None:
            placement["timeSeconds"] = chord_times[i]
        line: dict = {"lineIndex": i, "lyrics": f"line {i}", "chordPlacements": [placement]}
        if line_times[i] is not None:
            line["timeSeconds"] = line_times[i]
        if confidences[i] is not None:
            line["confidence"] = confidences[i]
        lines.append(line)

    return Song.model_validate(
        {"id": "a--b", "metadata": {"title": "T", "artist": "A"}, "lines": lines}
    )


# --- D3: timing coverage ----------------------------------------------------


def test_coverage_is_none_not_zero_when_there_is_no_timing():
    """A scorecard column reading '—' means 'not measured'; 0.0 means
    'measured, and nothing landed'. Conflating them would make an unrun
    engine look like a broken one."""
    cov = timing_coverage(_song())
    assert cov["chordTimeCoverage"] == 0.0  # placements exist, none are timed
    assert cov["lineTimeCoverage"] == 0.0
    assert cov["confidentLineShare"] == 0.0

    empty = Song.model_validate(
        {"id": "a--b", "metadata": {"title": "T", "artist": "A"}, "lines": []}
    )
    none_cov = timing_coverage(empty)
    assert none_cov["chordTimeCoverage"] is None
    assert none_cov["lineTimeCoverage"] is None
    assert none_cov["confidentLineShare"] is None


def test_coverage_counts_timed_chords_and_lines():
    cov = timing_coverage(_song(chord_times=[1.0, None], line_times=[1.0, 5.0]))
    assert cov["chordTimeCoverage"] == 0.5
    assert cov["lineTimeCoverage"] == 1.0


def test_confident_line_share_uses_the_0_75_threshold():
    cov = timing_coverage(_song(confidences=[0.75, 0.74]))
    assert cov["confidentLineShare"] == 0.5


def test_legacy_syncmap_lines_count_as_timed():
    song = Song.model_validate(
        {
            "id": "a--b",
            "metadata": {"title": "T", "artist": "A"},
            "audio": {"syncMap": [{"lineIndex": 0, "time": 2.0}]},
            "lines": [
                {"lineIndex": 0, "lyrics": "x", "chordPlacements": []},
                {"lineIndex": 1, "lyrics": "y", "chordPlacements": []},
            ],
        }
    )
    assert timing_coverage(song)["lineTimeCoverage"] == 0.5


def test_score_song_carries_the_coverage_metrics():
    gold = _song(line_times=[1.0, 5.0])
    cand = _song(chord_times=[1.0, 2.0], line_times=[1.1, 5.2], confidences=[0.9, 0.9])
    metrics = score_song(cand, gold)
    assert metrics["chordTimeCoverage"] == 1.0
    assert metrics["confidentLineShare"] == 1.0
    assert metrics["timingMAE"] is not None


def test_scorecard_aggregate_skips_missing_metrics(monkeypatch):
    """Averaging must ignore songs that don't have a metric rather than
    treating them as zero."""
    from snoocle_server.api import _aggregate_scores

    base = {
        "chordSimilarity": 1.0, "chordRootSimilarity": 1.0, "lyricSimilarity": 1.0,
        "sectionSimilarity": 1.0, "overall": 1.0,
    }
    out = _aggregate_scores([
        {**base, "timingMAE": 2.0, "chordTimeCoverage": 1.0,
         "lineTimeCoverage": 1.0, "confidentLineShare": 1.0},
        {**base, "timingMAE": None, "chordTimeCoverage": None,
         "lineTimeCoverage": None, "confidentLineShare": None},
    ])
    assert out["timingMAE"] == 2.0        # not 1.0 — the None song is skipped
    assert out["chordTimeCoverage"] == 1.0

    none_only = _aggregate_scores([
        {**base, "timingMAE": None, "chordTimeCoverage": None,
         "lineTimeCoverage": None, "confidentLineShare": None},
    ])
    assert none_only["timingMAE"] is None
    assert none_only["chordTimeCoverage"] is None


# --- D1: the review queue's data shape --------------------------------------


def test_review_queue_entries_carry_what_the_ui_renders():
    from snoocle_server.timing.confidence import PlacementScore

    queue = build_review_queue([
        PlacementScore(lineIndex=1, charIndex=4, chord="Bm", confidence=0.2,
                       reasons=["no MIR agreement"]),
        PlacementScore(lineIndex=0, charIndex=0, chord="C", confidence=0.5,
                       reasons=["1 of 3 sources"]),
        PlacementScore(lineIndex=2, charIndex=0, chord="G", confidence=0.95,
                       reasons=[]),
    ])
    assert [e["confidence"] for e in queue] == [0.2, 0.5]  # ascending, thresholded
    for entry in queue:
        assert set(entry) == {"lineIndex", "charIndex", "chord", "confidence", "reasons"}


# --- the admin shell --------------------------------------------------------


def test_admin_serves_and_wires_the_phase_d_script():
    assert client.get("/ui/admin-d.js").status_code == 200
    html = client.get("/ui/").text
    assert "admin-d.js" in html
    # admin-d.js uses app.js's globals, so it must load after it.
    assert html.index("app.js") < html.index("admin-d.js")


def test_admin_shell_has_the_review_tab_and_queue_button():
    html = client.get("/ui/").text
    assert 'data-tab="review"' in html
    assert 'id="tab-review"' in html
    assert 'id="queue-btn"' in html


@pytest.mark.parametrize("selector", ["review-row", "queue-card", "heat-low", "meter"])
def test_phase_d_styles_exist(selector):
    assert selector in client.get("/ui/style.css").text
