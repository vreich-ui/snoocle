"""reconcile/patch_ops.py: apply a closed set of operations to a prior Song.

The whole point of the patch path: a targeted correction changes exactly what
it names and nothing else, by construction — there is no regeneration step
here for a timing-loss bug to hide behind. These tests pin that directly, at
the byte level, alongside every closed-set failure mode being a hard,
unambiguous, un-fuzzy-matched error.
"""

from __future__ import annotations

import pytest

from snoocle_server.reconcile.patch_ops import (
    MAX_OPS,
    PatchError,
    apply_patch,
    parse_ops_response,
)
from snoocle_server.schema import Song


def _song(**overrides) -> Song:
    base = {
        "id": "the-rolling-stones--paint-it-black",
        "metadata": {"title": "Paint It Black", "artist": "The Rolling Stones", "bpm": 128.0},
        "audio": {
            "youtubeVideoId": "flSmiIne4k1",
            "beats": [{"time": 0.0, "measure": 1, "beatInMeasure": 1}],
            "syncMap": [{"lineIndex": 0, "time": 0.0}, {"lineIndex": 1, "time": 4.0}],
        },
        "sections": [
            {"sectionIndex": 0, "name": "Verse 1", "kind": "verse",
             "startLineIndex": 0, "endLineIndex": 1, "startTime": 0.0, "endTime": 8.0},
        ],
        "lines": [
            {"lineIndex": 0, "lyrics": "I see a red door", "timeSeconds": 0.0, "confidence": 0.9,
             "chordPlacements": [
                 {"charIndex": 0, "chord": "C", "timeSeconds": 0.0, "confidence": 0.9},
                 {"charIndex": 9, "chord": "G", "timeSeconds": 2.0, "confidence": 0.8},
             ]},
            {"lineIndex": 1, "lyrics": "and I want it painted black", "timeSeconds": 4.0,
             "chordPlacements": [{"charIndex": 21, "chord": "C", "timeSeconds": 6.0}]},
        ],
        "provenance": [
            {"timestamp": "2026-07-01T00:00:00Z", "actor": "reconcile:test/gold", "action": "reconciled"},
        ],
    }
    base.update(overrides)
    return Song.model_validate(base)


def _op(prior_lines_index, char_index, from_chord, to_chord, reason=None):
    op = {"op": "replace_chord", "lineIndex": prior_lines_index, "charIndex": char_index,
          "from": from_chord, "to": to_chord}
    if reason is not None:
        op["reason"] = reason
    return op


# --- the primary case: one chord, nothing else changes ---------------------------


def test_replace_chord_changes_only_that_field():
    """The exact reproduced scenario: change one chord (line 12's real-world
    analogue is line 1 here), everything else byte-identical, timing included."""
    prior = _song()
    ops = [{"op": "replace_chord", "lineIndex": 1, "charIndex": 21, "from": "C", "to": "B",
            "reason": "should be B, not C"}]

    patched, applied = apply_patch(prior, ops)

    before = prior.model_dump(mode="json")
    after = patched.model_dump(mode="json")
    before["lines"][1]["chordPlacements"][0]["chord"] = "B"  # the ONE expected diff
    assert after == before, "every field but the one named chord must be untouched"

    assert len(applied) == 1
    assert applied[0].op_type == "replace_chord"
    assert applied[0].reason == "should be B, not C"
    assert "line 1 charIndex 21: C -> B" in applied[0].description


def test_replace_chord_preserves_the_placements_own_timing():
    prior = _song()
    patched, _ = apply_patch(prior, [_op(1, 21, "C", "B")])
    placement = patched.lines[1].chordPlacements[0]
    assert placement.chord == "B"
    assert placement.timeSeconds == 6.0  # untouched


def test_replace_chord_mismatched_from_fails_and_changes_nothing():
    prior = _song()
    with pytest.raises(PatchError) as exc:
        apply_patch(prior, [_op(1, 21, "G", "B")])  # actually C, not G
    assert "op 0" in str(exc.value)
    assert "line 1" in str(exc.value) and "charIndex 21" in str(exc.value)
    assert "'C'" in str(exc.value) and "'G'" in str(exc.value)


def test_replace_chord_unknown_charindex_fails():
    prior = _song()
    with pytest.raises(PatchError, match="no chord placement"):
        apply_patch(prior, [_op(1, 999, "C", "B")])


def test_replace_chord_unknown_line_fails():
    prior = _song()
    with pytest.raises(PatchError, match="no line with lineIndex 99"):
        apply_patch(prior, [_op(99, 21, "C", "B")])


def test_replace_chord_rejects_a_shape_not_a_harmony():
    prior = _song()
    with pytest.raises(PatchError):
        apply_patch(prior, [_op(1, 21, "C", "x02210")])


# --- insert / remove / move --------------------------------------------------------


def test_insert_chord_adds_without_disturbing_neighbors():
    prior = _song()
    patched, applied = apply_patch(
        prior, [{"op": "insert_chord", "lineIndex": 0, "charIndex": 5, "chord": "Am"}]
    )
    chords = [(p.charIndex, p.chord) for p in patched.lines[0].chordPlacements]
    assert chords == [(0, "C"), (5, "Am"), (9, "G")]
    assert applied[0].reason is None


def test_insert_chord_on_an_occupied_charindex_fails():
    prior = _song()
    with pytest.raises(PatchError, match="already has a chord"):
        apply_patch(prior, [{"op": "insert_chord", "lineIndex": 0, "charIndex": 9, "chord": "Am"}])


def test_insert_chord_past_the_end_of_the_line_fails():
    prior = _song()
    with pytest.raises(PatchError, match="beyond the line"):
        apply_patch(prior, [{"op": "insert_chord", "lineIndex": 0, "charIndex": 500, "chord": "Am"}])


def test_remove_chord_drops_the_placement():
    prior = _song()
    patched, applied = apply_patch(
        prior, [{"op": "remove_chord", "lineIndex": 0, "charIndex": 9}]
    )
    assert [p.charIndex for p in patched.lines[0].chordPlacements] == [0]
    assert "G" in applied[0].description


def test_remove_chord_with_a_from_mismatch_fails():
    prior = _song()
    with pytest.raises(PatchError, match="not 'Am'"):
        apply_patch(prior, [{"op": "remove_chord", "lineIndex": 0, "charIndex": 9, "from": "Am"}])


def test_move_chord_changes_only_charindex():
    prior = _song()
    patched, _ = apply_patch(
        prior, [{"op": "move_chord", "lineIndex": 0, "fromCharIndex": 9, "toCharIndex": 7}]
    )
    chords = [(p.charIndex, p.chord) for p in patched.lines[0].chordPlacements]
    assert chords == [(0, "C"), (7, "G")]
    assert patched.lines[0].chordPlacements[1].timeSeconds == 2.0  # preserved


def test_move_chord_onto_an_occupied_charindex_fails():
    prior = _song()
    with pytest.raises(PatchError, match="already has a chord"):
        apply_patch(prior, [{"op": "move_chord", "lineIndex": 0, "fromCharIndex": 9, "toCharIndex": 0}])


# --- section ops ---------------------------------------------------------------


def test_set_section_name_renames_only_that_section():
    prior = _song()
    patched, applied = apply_patch(
        prior, [{"op": "set_section_name", "sectionIndex": 0, "to": "Bridge"}]
    )
    assert patched.sections[0].name == "Bridge"
    assert patched.sections[0].kind == "verse"  # untouched
    assert patched.sections[0].startTime == 0.0  # untouched
    assert "'Verse 1' -> 'Bridge'" in applied[0].description


def test_set_section_name_unknown_section_fails():
    prior = _song()
    with pytest.raises(PatchError, match="no section with sectionIndex 9"):
        apply_patch(prior, [{"op": "set_section_name", "sectionIndex": 9, "to": "X"}])


def test_set_section_bounds_changes_the_range():
    prior = _song(lines=[l for l in _song().model_dump(mode="json")["lines"]] * 1)
    # extend the song to 4 lines so a wider bound is legal
    doc = prior.model_dump(mode="json")
    doc["lines"] = doc["lines"] + [
        {"lineIndex": 2, "lyrics": "x", "chordPlacements": []},
        {"lineIndex": 3, "lyrics": "y", "chordPlacements": []},
    ]
    prior = Song.model_validate(doc)

    patched, _ = apply_patch(
        prior, [{"op": "set_section_bounds", "sectionIndex": 0,
                 "startLineIndex": 0, "endLineIndex": 3}]
    )
    assert (patched.sections[0].startLineIndex, patched.sections[0].endLineIndex) == (0, 3)
    assert patched.sections[0].startTime == 0.0  # untouched


def test_set_section_bounds_past_the_end_fails():
    prior = _song()
    with pytest.raises(PatchError, match="beyond"):
        apply_patch(prior, [{"op": "set_section_bounds", "sectionIndex": 0,
                              "startLineIndex": 0, "endLineIndex": 50}])


def test_set_section_bounds_end_before_start_fails():
    prior = _song()
    with pytest.raises(PatchError, match="endLineIndex"):
        apply_patch(prior, [{"op": "set_section_bounds", "sectionIndex": 0,
                              "startLineIndex": 1, "endLineIndex": 0}])


# --- split / merge --------------------------------------------------------------


def test_split_line_creates_two_lines_from_one():
    prior = _song()
    patched, applied = apply_patch(
        prior, [{"op": "split_line", "lineIndex": 0, "charIndex": 5}]
    )
    assert len(patched.lines) == 3
    assert patched.lines[0].lyrics == "I see"
    assert patched.lines[1].lyrics == " a red door"
    assert patched.lines[1].timeSeconds is None  # no invented timing
    assert patched.lines[0].timeSeconds == 0.0  # unchanged
    # the chord at charIndex 0 stays on the first half
    assert [p.charIndex for p in patched.lines[0].chordPlacements] == [0]
    # the chord that was at charIndex 9 moves to the second half, re-based
    assert [p.charIndex for p in patched.lines[1].chordPlacements] == [4]
    # every later line shifted up by one
    assert patched.lines[2].lyrics == "and I want it painted black"
    assert "lines 0 and 1" in applied[0].description


def test_split_line_extends_the_containing_section():
    prior = _song()
    patched, _ = apply_patch(prior, [{"op": "split_line", "lineIndex": 0, "charIndex": 5}])
    # section 0 originally covered lines 0-1; splitting line 0 must extend
    # its end (the new line inherits the same section) without moving start
    assert (patched.sections[0].startLineIndex, patched.sections[0].endLineIndex) == (0, 2)


def test_split_line_regenerates_syncmap():
    prior = _song()
    patched, _ = apply_patch(prior, [{"op": "split_line", "lineIndex": 0, "charIndex": 5}])
    assert [(p.lineIndex, p.time) for p in patched.audio.syncMap] == [
        (line.lineIndex, line.timeSeconds) for line in patched.lines if line.timeSeconds is not None
    ]


def test_split_line_on_an_instrumental_line_fails():
    doc = _song().model_dump(mode="json")
    doc["lines"][0]["lyrics"] = ""
    doc["lines"][0]["chordPlacements"] = []
    prior = Song.model_validate(doc)
    with pytest.raises(PatchError, match="no lyrics to split"):
        apply_patch(prior, [{"op": "split_line", "lineIndex": 0, "charIndex": 2}])


def test_split_line_at_the_very_start_or_end_fails():
    prior = _song()
    with pytest.raises(PatchError, match="strictly between"):
        apply_patch(prior, [{"op": "split_line", "lineIndex": 0, "charIndex": 0}])
    with pytest.raises(PatchError, match="strictly between"):
        apply_patch(prior, [{"op": "split_line", "lineIndex": 0, "charIndex": len("I see a red door")}])


def test_merge_line_is_the_inverse_of_split():
    prior = _song()
    split, _ = apply_patch(prior, [{"op": "split_line", "lineIndex": 0, "charIndex": 5}])
    merged, applied = apply_patch(split, [{"op": "merge_line", "lineIndex": 0}])

    assert len(merged.lines) == 2
    assert merged.lines[0].lyrics == prior.lines[0].lyrics
    assert [p.charIndex for p in merged.lines[0].chordPlacements] == \
        [p.charIndex for p in prior.lines[0].chordPlacements]
    assert merged.lines[1].lyrics == prior.lines[1].lyrics
    assert (merged.sections[0].startLineIndex, merged.sections[0].endLineIndex) == (0, 1)
    assert "merged into line 0" in applied[0].description


def test_merge_line_with_no_following_line_fails():
    prior = _song()
    with pytest.raises(PatchError, match="no following line"):
        apply_patch(prior, [{"op": "merge_line", "lineIndex": 1}])


def test_merge_line_keeps_the_first_lines_timing():
    prior = _song()
    merged, _ = apply_patch(prior, [{"op": "merge_line", "lineIndex": 0}])
    assert merged.lines[0].timeSeconds == 0.0  # line 0's own, not line 1's 4.0


# --- provenance-relevant reasons ---------------------------------------------------


def test_reason_is_optional_and_carried_when_given():
    prior = _song()
    _, applied = apply_patch(prior, [
        {"op": "replace_chord", "lineIndex": 1, "charIndex": 21, "from": "C", "to": "B"},
    ])
    assert applied[0].reason is None

    _, applied = apply_patch(prior, [
        {"op": "replace_chord", "lineIndex": 1, "charIndex": 21, "from": "C", "to": "B",
         "reason": "matches the recording"},
    ])
    assert applied[0].reason == "matches the recording"


# --- parse_ops_response: shape + closed-set + cap -------------------------------


def test_parse_ops_response_requires_the_ops_key():
    with pytest.raises(PatchError, match='"ops"'):
        parse_ops_response({"notOps": []})
    with pytest.raises(PatchError, match='"ops"'):
        parse_ops_response({"ops": "not a list"})
    with pytest.raises(PatchError):
        parse_ops_response(["not", "a", "dict"])


def test_parse_ops_response_rejects_an_empty_list():
    with pytest.raises(PatchError, match="empty"):
        parse_ops_response({"ops": []})


def test_parse_ops_response_rejects_unknown_op_type():
    with pytest.raises(PatchError) as exc:
        parse_ops_response({"ops": [{"op": "delete_song", "lineIndex": 0}]})
    assert "unknown op" in str(exc.value)
    assert "delete_song" in str(exc.value)
    assert "replace_chord" in str(exc.value)  # the closed set is named


def test_parse_ops_response_rejects_a_malformed_op():
    with pytest.raises(PatchError):
        parse_ops_response({"ops": ["replace_chord"]})
    with pytest.raises(PatchError):
        parse_ops_response({"ops": [{"lineIndex": 0}]})  # no "op" field


def test_parse_ops_response_enforces_the_cap():
    ops = [{"op": "set_section_name", "sectionIndex": 0, "to": f"X{i}"} for i in range(MAX_OPS + 1)]
    with pytest.raises(PatchError) as exc:
        parse_ops_response({"ops": ops})
    assert "exceeds the cap" in str(exc.value)
    assert str(MAX_OPS) in str(exc.value)
    assert "full re-analysis" in str(exc.value)


def test_parse_ops_response_allows_exactly_the_cap():
    ops = [{"op": "set_section_name", "sectionIndex": 0, "to": f"X{i}"} for i in range(MAX_OPS)]
    assert parse_ops_response({"ops": ops}) == ops


# --- applying multiple ops in one patch ------------------------------------------


def test_multiple_ops_apply_in_order_and_each_is_recorded():
    prior = _song()
    ops = [
        {"op": "replace_chord", "lineIndex": 1, "charIndex": 21, "from": "C", "to": "B"},
        {"op": "set_section_name", "sectionIndex": 0, "to": "Intro"},
    ]
    patched, applied = apply_patch(prior, ops)
    assert patched.lines[1].chordPlacements[0].chord == "B"
    assert patched.sections[0].name == "Intro"
    assert len(applied) == 2
    assert applied[0].index == 0 and applied[1].index == 1


def test_a_failing_op_leaves_nothing_applied():
    """No partial application: the exception propagates before any Song is
    returned, so a caller can never mistake a half-applied patch for success."""
    prior = _song()
    ops = [
        {"op": "set_section_name", "sectionIndex": 0, "to": "Intro"},
        {"op": "replace_chord", "lineIndex": 1, "charIndex": 21, "from": "WRONG", "to": "B"},
    ]
    with pytest.raises(PatchError):
        apply_patch(prior, ops)
    # the prior object itself was never mutated (model_dump makes a fresh dict)
    assert prior.sections[0].name == "Verse 1"
