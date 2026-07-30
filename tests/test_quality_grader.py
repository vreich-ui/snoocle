"""The deterministic grade, on hand-built documents whose defects are known.

Four shapes, each one a real production failure mode:

- healthy       — what a good run looks like; must grade "pass".
- collapsed     — every line piled onto one timestamp (the 1966 live take:
                  lines 17-20 all at 86.51755102040816).
- partial       — timing that dies a fifth of the way through a 220s track.
- theory-broken — chords no key explains: the transcription-error signal
                  nothing in the pipeline could previously see.

Plus the two rules that keep a grade honest: "not measurable" never reads as
"bad", and a grade is recorded whatever it says.
"""

from __future__ import annotations

import pytest

from snoocle_server.mir.base import ChordSegment, MirAnalysis
from snoocle_server.quality.grader import (
    GradeThresholds,
    grade_provenance_entry,
    grade_song,
)
from snoocle_server.schema import Song

_PROGRESSION = ["C", "G", "Am", "F"]
_DURATION = 40.0


def _mir(chords: list[str] | None = None, *, duration: float = _DURATION, key: str = "C major"):
    """An MIR timeline of five-second segments covering the whole track."""
    chords = chords if chords is not None else _PROGRESSION * 2
    step = duration / len(chords)
    return MirAnalysis(
        duration_seconds=duration,
        key=key,
        chords=[
            ChordSegment(start=i * step, end=(i + 1) * step, chord=c)
            for i, c in enumerate(chords)
        ],
    )


def _song(lines: list[dict], *, sections: list[dict] | None = None, key: str | None = "C major",
          duration: float | None = _DURATION, provenance: list[dict] | None = None) -> Song:
    return Song.model_validate(
        {
            "id": "test--graded-song",
            "metadata": {"title": "Graded", "artist": "Test", "key": key},
            "audio": {"durationSeconds": duration},
            "sections": sections or [],
            "lines": lines,
            "provenance": provenance or [],
        }
    )


def _line(index: int, chord: str, time: float | None, *, confidence: float = 0.9,
          lyrics: str | None = None) -> dict:
    timing = {"timeSeconds": time, "confidence": confidence} if time is not None else {}
    return {
        "lineIndex": index,
        "lyrics": lyrics if lyrics is not None else f"line {index} has real words in it",
        **timing,
        "chordPlacements": [{"charIndex": 0, "chord": chord, **timing}],
    }


def _healthy_song() -> Song:
    lines = [_line(i, _PROGRESSION[i % 4], i * 5.0) for i in range(8)]
    sections = [
        {"sectionIndex": 0, "name": "Verse 1", "kind": "verse", "startLineIndex": 0,
         "endLineIndex": 3, "startTime": 0.0, "endTime": 20.0},
        {"sectionIndex": 1, "name": "Chorus", "kind": "chorus", "startLineIndex": 4,
         "endLineIndex": 7, "startTime": 20.0, "endTime": 40.0},
    ]
    return _song(lines, sections=sections)


# --- healthy ----------------------------------------------------------------


def test_a_healthy_document_passes_with_every_metric_measured():
    grade = grade_song(_healthy_song(), _mir())

    assert grade.verdict == "pass", grade.describe()
    assert grade.overall > 0.9
    assert grade.failing == ()
    assert {m.name for m in grade.measured} == {
        "chordMatchRatio", "timingCoverage", "interpolationShare",
        "collapseRuns", "sectionCoverage", "theoryValidity", "lyricCompleteness",
    }
    assert grade.value("chordMatchRatio") == 1.0
    assert grade.value("collapseRuns") == 0.0


def test_the_grade_serializes_with_its_thresholds_and_failures_named():
    payload = grade_song(_healthy_song(), _mir()).to_dict()
    assert payload["verdict"] == "pass"
    assert payload["failing"] == []
    assert payload["metrics"]["chordMatchRatio"]["threshold"] == GradeThresholds().chord_match_ratio
    assert "detail" in payload["metrics"]["timingCoverage"]


# --- collapsed --------------------------------------------------------------


def test_a_collapsed_document_fails_and_names_the_collapsed_run():
    """The 1966 live shape: every line holds one timestamp because there was
    nothing later to interpolate toward, so the collapse guard left it alone."""
    lines = [_line(i, _PROGRESSION[i % 4], 8.0, confidence=0.3) for i in range(5)]
    grade = grade_song(_song(lines), _mir())

    assert grade.verdict == "fail", grade.describe()
    failing = {m.name for m in grade.failing}
    assert {"collapseRuns", "timingCoverage", "interpolationShare"} <= failing

    run = grade.metric("collapseRuns").offenders[0]
    assert run == {
        "kind": "lines", "fromLineIndex": 0, "toLineIndex": 4,
        "timeSeconds": 8.0, "count": 5,
    }


def test_placements_collapsed_within_one_line_are_caught_too():
    lines = [
        {
            "lineIndex": 0,
            "lyrics": "one line with four chords piled on one moment",
            "timeSeconds": 4.0,
            "chordPlacements": [
                {"charIndex": i * 4, "chord": _PROGRESSION[i], "timeSeconds": 4.0}
                for i in range(4)
            ],
        },
        _line(1, "C", 20.0),
    ]
    metric = grade_song(_song(lines), _mir()).metric("collapseRuns")
    assert metric.value == 1.0
    assert metric.offenders[0]["kind"] == "placements"
    assert metric.offenders[0]["lineIndex"] == 0


def test_a_short_run_of_two_shared_timestamps_is_not_a_collapse():
    """Two adjacent lines legitimately starting on the same beat happens; the
    threshold is the collapse guard's own, not a second opinion."""
    lines = [_line(0, "C", 0.0), _line(1, "G", 0.0), _line(2, "Am", 20.0), _line(3, "F", 35.0)]
    assert grade_song(_song(lines), _mir()).value("collapseRuns") == 0.0


# --- collapseRuns is a hard gate ---------------------------------------------


def test_a_single_collapsed_run_forces_fail_even_in_an_otherwise_perfect_document():
    """The production defect, isolated: `collapseRuns`' `ok` is a COUNT test
    (any run at all is a defect) but its `score` is a DENSITY, so one short
    run on a long, otherwise-pristine document barely dents the weighted
    overall. Before `Metric.hard_gate`, this document would have graded
    `warn`: one failing metric is nowhere near `half_of_measured`, and 0.98
    overall is nowhere near the 0.6 fail line. The hard gate is the only
    thing standing between this document and a `warn`."""
    progression = ["C", "G", "Am", "F"]
    duration = 100.0
    n = 20
    lines = [_line(i, progression[i % 4], i * 5.0) for i in range(n)]
    # A run of exactly COLLAPSE_RUN_THRESHOLD (3) consecutive lines piled onto
    # one timestamp -- everything else stays perfectly, distinctly timed.
    for i in (10, 11, 12):
        lines[i] = _line(i, progression[i % 4], 50.0)
    sections = [
        {"sectionIndex": 0, "name": "Verse 1", "kind": "verse", "startLineIndex": 0,
         "endLineIndex": n - 1, "startTime": 0.0, "endTime": duration},
    ]
    mir = _mir(progression * (n // 4), duration=duration)
    grade = grade_song(_song(lines, sections=sections, duration=duration), mir)

    collapse = grade.metric("collapseRuns")
    assert collapse.ok is False
    assert collapse.value == 1.0
    assert collapse.score > 0.9, "one short run on 20 lines barely costs the density score"

    # The two pre-existing fail routes both plainly do NOT explain this
    # verdict -- only the hard gate does.
    assert grade.overall > 0.9 > GradeThresholds().overall
    half_of_measured = max(2, -(-len(grade.measured) // 2))
    assert len(grade.failing) == 1 < half_of_measured

    assert grade.verdict == "fail", grade.describe()


def test_the_warn_0_76_incident_now_grades_fail_from_the_collapse_gate_alone():
    """The exact production shape: two hard FAILs (`collapseRuns`,
    `sectionCoverage`) commanding only 0.2 of the total weight, an overall in
    the high 0.7s, and a failure count nowhere near `half_of_measured`. This
    used to grade `warn (0.76)`. It must now grade `fail` -- and specifically
    because `collapseRuns` is gated, not because `sectionCoverage` is (that
    gate defaults OFF; see the `sectionCoverage` gate tests below)."""
    progression = ["C", "G", "Am", "F"]
    duration = 120.0
    n = 24
    chords = [progression[i % 4] for i in range(n)]
    # A handful of chords the document got wrong (chordMatchRatio ~0.62) and
    # timing that only reaches 81% of the track (timingCoverage ~0.81) --
    # imperfect, but both comfortably above their own `ok` thresholds.
    mismatched = list(chords)
    for i in (2, 5, 8, 9, 14, 15, 18, 20, 22):
        mismatched[i] = "B"
    lines = [_line(i, mismatched[i], round(i * 4.2, 3)) for i in range(n)]
    # One collapsed run of exactly COLLAPSE_RUN_THRESHOLD lines.
    for i in (10, 11, 12):
        lines[i] = _line(i, mismatched[i], 42.0)
    # Sections present but never timed -- sectionCoverage fails to 0.0, the
    # second hard FAIL in the incident, structurally NOT gated by default.
    sections = [
        {"sectionIndex": 0, "name": "Verse 1", "kind": "verse",
         "startLineIndex": 0, "endLineIndex": n - 1},
    ]
    mir = _mir(chords, duration=duration)
    grade = grade_song(_song(lines, sections=sections, duration=duration), mir)

    assert grade.metric("chordMatchRatio").ok is True
    assert grade.metric("timingCoverage").ok is True
    assert grade.metric("collapseRuns").ok is False
    assert grade.metric("sectionCoverage").ok is False
    assert {m.name for m in grade.failing} == {"collapseRuns", "sectionCoverage"}

    # The incident's own arithmetic: warn-shaped, both fail routes clear.
    assert grade.overall == pytest.approx(0.76, abs=0.02)
    assert grade.overall > GradeThresholds().overall
    half_of_measured = max(2, -(-len(grade.measured) // 2))
    assert len(grade.failing) < half_of_measured

    assert grade.verdict == "fail", grade.describe()


# --- which route made it a fail ---------------------------------------------


def _collapse_only_fail_song() -> tuple[Song, MirAnalysis]:
    """An otherwise-healthy document with one collapsed run at the tail: the
    gate is the only route to `fail`."""
    progression = ["C", "G", "Am", "F"]
    duration = 100.0
    n = 20
    lines = [_line(i, progression[i % 4], i * 5.0) for i in range(n)]
    for i in (17, 18, 19):  # held at one timestamp, nothing later to spread to
        lines[i] = _line(i, progression[i % 4], 85.0)
    sections = [
        {"sectionIndex": 0, "name": "Verse 1", "kind": "verse", "startLineIndex": 0,
         "endLineIndex": n - 1, "startTime": 0.0, "endTime": duration},
    ]
    return (
        _song(lines, sections=sections, duration=duration),
        _mir(progression * (n // 4), duration=duration),
    )


def test_a_collapse_only_fail_names_the_gate_as_its_single_reason():
    """`fail` says a document is bad; `fail_routes` says WHICH of the three
    independent routes got it there, which is what lets an escalation tell "one
    zero-tolerance metric tripped" apart from "this document is broadly bad" --
    they want different actions (see quality/escalation.py)."""
    song, mir = _collapse_only_fail_song()
    grade = grade_song(song, mir)

    assert grade.verdict == "fail", grade.describe()
    assert grade.fail_routes == ("hard-gate:collapseRuns",)
    assert grade.fails_only_on_hard_gate("collapseRuns") is True
    assert grade.fails_only_on_hard_gate("sectionCoverage") is False


def test_a_broadly_bad_document_is_not_a_collapse_only_fail():
    """The 1966 shape: every line piled onto one timestamp. The collapse gate
    trips, but so do the overall and the failing-majority routes -- the collapse
    is a symptom here, not the whole story."""
    lines = [_line(i, _PROGRESSION[i % 4], 8.0, confidence=0.3) for i in range(5)]
    grade = grade_song(_song(lines), _mir())

    assert grade.verdict == "fail", grade.describe()
    assert "hard-gate:collapseRuns" in grade.fail_routes
    assert len(grade.fail_routes) > 1
    assert grade.fails_only_on_hard_gate("collapseRuns") is False


def test_a_healthy_grade_has_no_fail_routes():
    grade = grade_song(_healthy_song(), _mir())
    assert grade.verdict == "pass"
    assert grade.fail_routes == ()
    assert grade.fails_only_on_hard_gate("collapseRuns") is False


def test_a_warn_has_no_fail_routes_either():
    grade = grade_song(_song_with_failing_section_coverage(), _mir())
    assert grade.verdict == "warn", grade.describe()
    assert grade.metric("sectionCoverage").ok is False
    assert grade.fail_routes == ()


# --- which way a threshold points -------------------------------------------


def test_the_maximum_style_metrics_declare_themselves_as_maximums():
    """`interpolationShare` and `collapseRuns` pass their `ok` test by staying
    BELOW their threshold. Anything rendering one has to know that -- printing
    "collapseRuns = 1.00 (needs >= 0.0)" states a satisfied condition for a
    metric that just failed (see quality/escalation.py's retry feedback).

    `comparator` alone is NOT enough to render every metric's threshold
    honestly, though -- `sectionCoverage`'s `ok` is a compound test (coverage
    AND no untimed section AND no excess overlap), so knowing it points ">="
    still is not the whole requirement. See `test_a_compound_ok_metric_states_
    its_real_requirement` below, and `Metric.requirement`'s docstring, for the
    gap this pins wrong if read as "comparator + threshold is always the full
    story"."""
    grade = grade_song(_healthy_song(), _mir())

    for name in ("collapseRuns", "interpolationShare"):
        assert grade.metric(name).maximum is True, name
        assert grade.metric(name).comparator == "<=", name
    for name in ("chordMatchRatio", "timingCoverage", "sectionCoverage",
                 "theoryValidity", "lyricCompleteness"):
        assert grade.metric(name).maximum is False, name
        assert grade.metric(name).comparator == ">=", name

    # `sectionCoverage` is the one metric here whose `ok` a (comparator,
    # threshold) pair cannot fully state -- it alone carries an explicit
    # `requirement`; every metric whose `ok` really is just `value` against
    # `threshold` carries none.
    assert grade.metric("sectionCoverage").requirement != ""
    for name in ("chordMatchRatio", "timingCoverage", "interpolationShare",
                 "collapseRuns", "theoryValidity", "lyricCompleteness"):
        assert grade.metric(name).requirement == "", name


# --- partial timing ---------------------------------------------------------


def test_partial_timing_fails_coverage_while_the_rest_still_measures():
    """Timing that dies early: the first three lines are timed inside 6s of a
    40s track and the remaining five carry nothing."""
    lines = [_line(i, _PROGRESSION[i % 4], i * 2.0 if i < 3 else None) for i in range(8)]
    grade = grade_song(_song(lines), _mir())

    coverage = grade.metric("timingCoverage")
    assert coverage.value == pytest.approx(0.1)
    assert coverage.ok is False
    assert "40.0s track" in coverage.detail
    # The chords themselves are fine, and the grade says so rather than
    # sweeping everything into one number.
    assert grade.value("chordMatchRatio") == 1.0
    assert grade.metric("lyricCompleteness").ok is True


def test_interpolated_placements_are_counted_at_snaps_own_confidence_tier():
    from snoocle_server.timing.snap import INTERPOLATED_CONFIDENCE

    lines = [
        _line(0, "C", 0.0, confidence=0.9),
        _line(1, "G", 10.0, confidence=INTERPOLATED_CONFIDENCE),
        _line(2, "Am", 20.0, confidence=INTERPOLATED_CONFIDENCE),
        _line(3, "F", 30.0, confidence=0.9),
    ]
    metric = grade_song(_song(lines), _mir()).metric("interpolationShare")
    assert metric.value == 0.5
    assert metric.ok is True  # exactly at the threshold, not over it
    assert [o["lineIndex"] for o in metric.offenders] == [1, 2]


# --- sections ---------------------------------------------------------------


def test_sections_with_no_times_score_zero_coverage_and_name_themselves():
    """The 1966 case again: every section has a null start/endTime. That is
    measurable and it is zero -- not "unmeasurable"."""
    lines = [_line(i, _PROGRESSION[i % 4], i * 5.0) for i in range(4)]
    sections = [
        {"sectionIndex": 0, "name": "Verse 1", "kind": "verse",
         "startLineIndex": 0, "endLineIndex": 3},
    ]
    metric = grade_song(_song(lines, sections=sections), _mir()).metric("sectionCoverage")
    assert metric.value == 0.0
    assert metric.ok is False
    assert metric.offenders[0]["reason"] == "no startTime/endTime"


def test_a_gap_between_sections_is_reported_with_its_span():
    lines = [_line(i, _PROGRESSION[i % 4], i * 5.0) for i in range(4)]
    sections = [
        {"sectionIndex": 0, "name": "Verse 1", "kind": "verse", "startLineIndex": 0,
         "endLineIndex": 1, "startTime": 0.0, "endTime": 10.0},
        {"sectionIndex": 1, "name": "Chorus", "kind": "chorus", "startLineIndex": 2,
         "endLineIndex": 3, "startTime": 30.0, "endTime": 40.0},
    ]
    metric = grade_song(_song(lines, sections=sections), _mir()).metric("sectionCoverage")
    assert metric.value == 0.5
    assert metric.ok is False
    assert {"gap": [10.0, 30.0], "reason": "no section covers this span"} in metric.offenders


def test_no_sections_at_all_is_not_measurable_rather_than_zero():
    lines = [_line(i, _PROGRESSION[i % 4], i * 5.0) for i in range(4)]
    metric = grade_song(_song(lines), _mir()).metric("sectionCoverage")
    assert metric.value is None
    assert metric.ok is None


def test_a_compound_ok_metric_states_its_real_requirement():
    """`sectionCoverage`'s `ok` is a CONJUNCTION -- coverage >= threshold AND
    no untimed section AND no excess overlap -- so a comparator/threshold pair
    alone cannot state what it needs. Mode A has no section timer, so a
    partially-timed section (one timed, one not) is the normal case, not a
    corner: the coverage NUMBER can be satisfied while `ok` is still `False`
    because of the untimed section, and `Metric.requirement` has to name the
    real, compound reason rather than leave a renderer to reconstruct a
    satisfied-looking clause from `threshold` alone."""
    lines = [_line(i, _PROGRESSION[i % 4], i * 20.0) for i in range(4)]
    sections = [
        {"sectionIndex": 0, "name": "Verse 1", "kind": "verse", "startLineIndex": 0,
         "endLineIndex": 1, "startTime": 0.0, "endTime": 60.0},
        {"sectionIndex": 1, "name": "Chorus", "kind": "chorus", "startLineIndex": 2,
         "endLineIndex": 3},
    ]
    metric = grade_song(
        _song(lines, sections=sections, duration=60.0), _mir(duration=60.0)
    ).metric("sectionCoverage")

    # The coverage NUMBER is fully satisfied -- one timed section already
    # spans the whole track -- yet `ok` is still False because of the other
    # section's missing times.
    assert metric.value == 1.0
    assert metric.ok is False
    assert metric.requirement != ""
    assert "0.75" in metric.requirement or str(GradeThresholds().section_coverage) in metric.requirement
    assert "untimed" in metric.requirement
    assert "overlap" in metric.requirement


# --- sectionCoverage's hard gate is configurable and OFF by default ---------


def _song_with_failing_section_coverage() -> Song:
    """Otherwise-healthy, no collapse anywhere -- the only failing metric is
    `sectionCoverage` (sections present, never timed)."""
    lines = [_line(i, _PROGRESSION[i % 4], i * 5.0) for i in range(8)]
    sections = [
        {"sectionIndex": 0, "name": "Verse 1", "kind": "verse",
         "startLineIndex": 0, "endLineIndex": 7},
    ]
    return _song(lines, sections=sections)


def test_section_coverage_is_not_hard_gated_by_default():
    """Mode A has no section timer yet (see `GradeThresholds.
    section_coverage_gated`'s docstring) -- gating this metric unconditionally
    today would hard-fail nearly every Mode A document, which is exactly the
    regression this default avoids."""
    song = _song_with_failing_section_coverage()
    grade = grade_song(song, _mir())

    assert GradeThresholds().section_coverage_gated is False
    assert grade.metric("sectionCoverage").hard_gate is False
    assert grade.metric("sectionCoverage").ok is False
    assert grade.verdict == "warn", grade.describe()


def test_section_coverage_gate_forces_fail_once_switched_on():
    song = _song_with_failing_section_coverage()
    mir = _mir()

    ungated = grade_song(song, mir)
    assert ungated.verdict == "warn", ungated.describe()

    gated = grade_song(song, mir, thresholds=GradeThresholds(section_coverage_gated=True))
    assert gated.metric("sectionCoverage").hard_gate is True
    assert gated.metric("sectionCoverage").ok is False
    # Same document, same overall, same failing metric NAMES -- only the gate
    # differs (each metric's own `hard_gate` flag), and that alone flips the
    # verdict.
    assert gated.overall == ungated.overall
    assert {m.name for m in gated.failing} == {m.name for m in ungated.failing}
    assert gated.verdict == "fail", gated.describe()


def test_section_coverage_gate_is_configurable_via_settings(monkeypatch):
    from snoocle_server.config import settings

    assert GradeThresholds.from_settings().section_coverage_gated is False

    monkeypatch.setattr(settings, "quality_section_coverage_gated", True)
    assert GradeThresholds.from_settings().section_coverage_gated is True

    song = _song_with_failing_section_coverage()
    grade = grade_song(song, _mir(), thresholds=GradeThresholds.from_settings())
    assert grade.verdict == "fail", grade.describe()


# --- theory validity --------------------------------------------------------


def test_a_theory_violating_document_is_caught_and_the_chords_named():
    """Chords a semitone off the key: the classic surviving transcription
    error, and the one signal here that needs neither audio nor sources."""
    pytest.importorskip("music21")
    bad = ["C#", "G#", "Bbm", "Ebm"]
    lines = [_line(i, bad[i], i * 5.0) for i in range(4)]
    metric = grade_song(_song(lines, key="C major"), _mir()).metric("theoryValidity")

    assert metric.value == 0.0
    assert metric.ok is False
    assert [o["chord"] for o in metric.offenders] == bad
    assert "outside the key" in metric.offenders[0]["reason"]


def test_diatonic_borrowings_and_secondary_dominants_are_explainable():
    """Deliberately permissive: a false "this song is broken" is worse than a
    missed one. bVII in a minor key, V/V in major, a diatonic seventh."""
    pytest.importorskip("music21")
    lines = [
        _line(0, "Am", 0.0), _line(1, "G", 5.0),  # i, bVII in A minor
        _line(2, "Dm7", 10.0), _line(3, "E", 15.0),  # iv, V (harmonic minor)
    ]
    assert grade_song(_song(lines, key="A minor"), _mir()).value("theoryValidity") == 1.0

    major = [_line(0, "C", 0.0), _line(1, "D7", 5.0), _line(2, "G", 10.0), _line(3, "Am7", 15.0)]
    assert grade_song(_song(major, key="C major"), _mir()).value("theoryValidity") == 1.0


def test_the_mir_key_stands_in_when_the_document_has_none():
    pytest.importorskip("music21")
    lines = [_line(i, _PROGRESSION[i % 4], i * 5.0) for i in range(4)]
    metric = grade_song(_song(lines, key=None), _mir(key="C major")).metric("theoryValidity")
    assert metric.value == 1.0
    assert metric.detail.endswith("(diatonic, borrowed major, or secondary dominant)")


def test_no_key_anywhere_is_not_measurable():
    lines = [_line(i, _PROGRESSION[i % 4], i * 5.0) for i in range(4)]
    metric = grade_song(_song(lines, key=None), _mir(key=None)).metric("theoryValidity")
    assert metric.value is None
    assert "no established key" in metric.detail


def test_theory_validity_degrades_to_unmeasured_without_music21(monkeypatch):
    """An optional dependency's absence must read as "not measured", never as
    a bad score -- the same contract the MIR engine fallbacks hold to."""
    monkeypatch.setattr("snoocle_server.quality.theory._load_music21", lambda: None)
    lines = [_line(i, "C#", i * 5.0) for i in range(4)]  # would fail if measured
    metric = grade_song(_song(lines, key="C major"), _mir()).metric("theoryValidity")
    assert metric.value is None
    assert metric.ok is None
    assert "music21 is not installed" in metric.detail


# --- lyric completeness -----------------------------------------------------


def test_empty_lines_are_fine_inside_an_instrumental_section():
    lines = [
        {"lineIndex": 0, "lyrics": "", "timeSeconds": 0.0,
         "chordPlacements": [{"charIndex": 0, "chord": "C", "timeSeconds": 0.0}]},
        _line(1, "G", 10.0),
    ]
    sections = [
        {"sectionIndex": 0, "name": "Intro", "kind": "intro", "startLineIndex": 0,
         "endLineIndex": 0, "startTime": 0.0, "endTime": 10.0},
        {"sectionIndex": 1, "name": "Verse 1", "kind": "verse", "startLineIndex": 1,
         "endLineIndex": 1, "startTime": 10.0, "endTime": 40.0},
    ]
    assert grade_song(_song(lines, sections=sections), _mir()).value("lyricCompleteness") == 1.0


def test_an_unexplained_empty_line_is_a_gap():
    lines = [
        _line(0, "C", 0.0),
        {"lineIndex": 1, "lyrics": "", "timeSeconds": 10.0,
         "chordPlacements": [{"charIndex": 0, "chord": "G", "timeSeconds": 10.0}]},
    ]
    metric = grade_song(_song(lines), _mir()).metric("lyricCompleteness")
    assert metric.value == 0.5
    assert metric.ok is False
    assert metric.offenders[0]["lineIndex"] == 1


def test_words_no_candidate_source_contains_count_against_completeness():
    """Lyrics with no visible provenance. The lyric-reference protocol makes
    this impossible on providers that speak it -- which is why it is worth
    measuring on the ones that don't."""
    from snoocle_server.discovery.models import CandidateSource
    from snoocle_server.schema.song import ChordPlacement, Line

    candidate = CandidateSource(
        sourceId="web-1",
        lines=[
            Line(lineIndex=0, lyrics="line 0 has real words in it",
                 chordPlacements=[ChordPlacement(charIndex=0, chord="C")]),
        ],
    )
    lines = [_line(0, "C", 0.0), _line(1, "G", 10.0, lyrics="a line invented out of thin air")]
    grade = grade_song(_song(lines), _mir(), [candidate])
    metric = grade.metric("lyricCompleteness")
    assert metric.value == 0.5
    assert "no candidate source contains these words" in metric.offenders[0]["reason"]


def test_model_written_lyric_overrides_count_against_completeness():
    lines = [_line(0, "C", 0.0), _line(1, "G", 10.0)]
    provenance = [
        {"timestamp": "2026-07-30T00:00:00Z", "actor": "reconcile:test/x",
         "action": "lyric-override", "notes": "line 1: written by the model"},
    ]
    grade = grade_song(_song(lines, provenance=provenance), _mir())
    assert grade.value("lyricCompleteness") == 0.5


# --- "not measurable" is not "bad" ------------------------------------------


def test_a_document_with_no_mir_is_unknown_not_failing():
    """No audio analysis means most metrics have no input. A grade must not
    escalate a document for evidence it was never given."""
    lines = [_line(i, _PROGRESSION[i % 4], None) for i in range(4)]
    grade = grade_song(_song(lines, key=None, duration=None))

    assert grade.verdict == "unknown"
    assert grade.metric("chordMatchRatio").value is None
    assert "no MIR analysis" in grade.metric("chordMatchRatio").detail


def test_half_the_measurable_metrics_failing_is_a_failing_document():
    """A document with perfect chords, key and words but no usable timing at
    all sits above the weighted-overall line -- and is not one anybody can
    play along to."""
    lines = [_line(i, _PROGRESSION[i % 4], 8.0, confidence=0.3) for i in range(5)]
    grade = grade_song(_song(lines), _mir())
    assert grade.overall > GradeThresholds().overall
    assert len(grade.failing) >= 3
    assert grade.verdict == "fail"


# --- thresholds + provenance ------------------------------------------------


def test_thresholds_are_configurable_and_change_the_verdict():
    lines = [_line(i, _PROGRESSION[i % 4], i * 2.0 if i < 3 else None) for i in range(8)]
    song, mir = _song(lines), _mir()

    strict = grade_song(song, mir, thresholds=GradeThresholds(chord_match_ratio=1.01))
    assert strict.metric("chordMatchRatio").ok is False

    forgiving = grade_song(song, mir, thresholds=GradeThresholds(timing_coverage=0.05, overall=0.1))
    assert forgiving.metric("timingCoverage").ok is True
    assert forgiving.verdict in ("pass", "warn")


def test_thresholds_from_settings_read_the_quality_prefix(monkeypatch):
    from snoocle_server.config import settings

    monkeypatch.setattr(settings, "quality_chord_match_ratio", 0.99)
    assert GradeThresholds.from_settings().chord_match_ratio == 0.99


def test_every_grade_produces_a_provenance_entry_whatever_the_verdict():
    """The grade is valuable as history even when no action follows: ten runs
    at ~0.5 are only visible as non-convergence if each wrote down its score."""
    for song in (_healthy_song(), _song([_line(i, "C", 8.0) for i in range(5)])):
        grade = grade_song(song, _mir())
        entry = grade_provenance_entry(grade)
        assert entry.action == "quality-grade"
        assert entry.actor == "snoocle-server/quality"
        assert entry.confidence == pytest.approx(round(grade.overall, 3))
        assert grade.verdict in entry.notes


def test_the_grade_is_deterministic():
    song, mir = _healthy_song(), _mir()
    assert grade_song(song, mir).to_dict() == grade_song(song, mir).to_dict()
