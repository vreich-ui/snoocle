"""The patch protocol: notes-only corrections as OPERATIONS, not a Song.

A valid Song document is the WHOLE song. Regenerating the whole document for
"change one chord" means the model reproduces every lyric verbatim to answer
a one-word question — which is what got a real correction blocked by
Anthropic's content filter over a single chord (see reconcile/lyric_refs.py
for the same story, one layer up: that fix stopped the model from writing
lyrics it should never need to write; this one stops it from being ASKED to
regenerate anything at all for a change that names exactly what it touches).

So a notes-only correction gets a different contract: the model returns a
short list of OPERATIONS against the prior document —

    {"ops": [
        {"op": "replace_chord", "lineIndex": 12, "charIndex": 21,
         "from": "C", "to": "B", "reason": "..."},
        {"op": "set_section_name", "sectionIndex": 3, "to": "Bridge"},
    ]}

— and this module applies them deterministically, in local code, never by
another model call. The properties that matters follow directly from that:

- **Nothing not named by an op changes.** There is no regeneration step to
  guard against timing loss on this path — the class of bug is structurally
  impossible, not defended against. Every op mutates exactly the field(s) it
  names and nothing else; the result is re-validated against the full Song
  schema as a whole, but never rebuilt.
- **Lyric text never appears in an op.** The closed op set below has no
  "replace lyric" — chords, section names/boundaries, and line split/merge
  (which only ever recombines lyric text ALREADY in the document, never
  writes new words) are the whole vocabulary. A correction that needs
  different words cannot be expressed here; the caller (reconcile/engine.py)
  must route it to the full reconcile path instead, visibly.
- **An op that doesn't apply is a hard failure, not a soft one.** No
  fuzzy-matching a charIndex that's "close enough", no silently dropping an
  op that doesn't validate. Every failure here is FATAL — see
  :class:`PatchError` — never retried through the repair loop the way a
  schema-validation slip is: a wrong op is not the kind of mistake retrying
  fixes, and "try again, maybe you'll guess differently" is exactly the
  fuzziness this path exists to refuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..chords import ChordParseError, validate_stored_chord
from ..schema.song import Song

#: More than this many ops is not a correction — see PatchError message.
MAX_OPS = 20


class PatchError(ValueError):
    """An op could not be applied, or the response didn't shape up as a
    patch at all. Always names the offending op (by index and type) so the
    caller sees exactly what didn't take. FATAL — never fed back through a
    repair round; see the module docstring for why."""


@dataclass(frozen=True)
class AppliedOp:
    """One op that was successfully applied — enough to write a provenance
    entry naming what changed and why."""

    index: int
    op_type: str
    description: str
    reason: Optional[str] = None


def _op_label(op: dict, idx: int) -> str:
    return f"op {idx} ({op.get('op', '?')})"


def _require_int(op: dict, key: str, idx: int) -> int:
    v = op.get(key)
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise PatchError(
            f"{_op_label(op, idx)}: {key!r} must be a non-negative integer, got {v!r}"
        )
    return v


def _require_str(op: dict, key: str, idx: int) -> str:
    v = op.get(key)
    if not isinstance(v, str) or not v:
        raise PatchError(
            f"{_op_label(op, idx)}: {key!r} must be a non-empty string, got {v!r}"
        )
    return v


def _optional_str(op: dict, key: str, idx: int) -> Optional[str]:
    v = op.get(key)
    if v is None:
        return None
    if not isinstance(v, str) or not v:
        raise PatchError(f"{_op_label(op, idx)}: {key!r} must be a non-empty string if given")
    return v


def _validate_chord_symbol(chord: str, op: dict, idx: int) -> None:
    try:
        validate_stored_chord(chord)
    except ChordParseError as e:
        raise PatchError(f"{_op_label(op, idx)}: {e}") from e


def _find_line(lines: list[dict], line_index: int, op: dict, idx: int) -> dict:
    for line in lines:
        if line["lineIndex"] == line_index:
            return line
    raise PatchError(f"{_op_label(op, idx)}: no line with lineIndex {line_index}")


def _line_position(lines: list[dict], line_index: int, op: dict, idx: int) -> int:
    for i, line in enumerate(lines):
        if line["lineIndex"] == line_index:
            return i
    raise PatchError(f"{_op_label(op, idx)}: no line with lineIndex {line_index}")


def _find_placement(line: dict, char_index: int, op: dict, idx: int) -> dict:
    for p in line["chordPlacements"]:
        if p["charIndex"] == char_index:
            return p
    raise PatchError(
        f"{_op_label(op, idx)}: line {line['lineIndex']} has no chord placement "
        f"at charIndex {char_index}"
    )


def _find_section(sections: list[dict], section_index: int, op: dict, idx: int) -> dict:
    for s in sections:
        if s["sectionIndex"] == section_index:
            return s
    raise PatchError(f"{_op_label(op, idx)}: no section with sectionIndex {section_index}")


def _check_charindex_bounds(lyrics: str, char_index: int, op: dict, idx: int) -> None:
    if lyrics and char_index > len(lyrics):
        raise PatchError(
            f"{_op_label(op, idx)}: charIndex {char_index} is beyond the line's "
            f"lyrics length {len(lyrics)}"
        )


# --- chord ops -----------------------------------------------------------------


def _apply_replace_chord(doc: dict, op: dict, idx: int) -> str:
    line_index = _require_int(op, "lineIndex", idx)
    char_index = _require_int(op, "charIndex", idx)
    from_chord = _require_str(op, "from", idx)
    to_chord = _require_str(op, "to", idx)
    _validate_chord_symbol(to_chord, op, idx)

    line = _find_line(doc["lines"], line_index, op, idx)
    placement = _find_placement(line, char_index, op, idx)
    if placement["chord"] != from_chord:
        raise PatchError(
            f"{_op_label(op, idx)}: line {line_index} charIndex {char_index} is "
            f"{placement['chord']!r}, not {from_chord!r} as the op expected — "
            f"nothing was changed"
        )
    placement["chord"] = to_chord
    return f"replace_chord: line {line_index} charIndex {char_index}: {from_chord} -> {to_chord}"


def _apply_insert_chord(doc: dict, op: dict, idx: int) -> str:
    line_index = _require_int(op, "lineIndex", idx)
    char_index = _require_int(op, "charIndex", idx)
    chord = _require_str(op, "chord", idx)
    _validate_chord_symbol(chord, op, idx)

    line = _find_line(doc["lines"], line_index, op, idx)
    if any(p["charIndex"] == char_index for p in line["chordPlacements"]):
        raise PatchError(
            f"{_op_label(op, idx)}: line {line_index} already has a chord at "
            f"charIndex {char_index} — use replace_chord instead"
        )
    _check_charindex_bounds(line["lyrics"], char_index, op, idx)
    line["chordPlacements"].append({"charIndex": char_index, "chord": chord})
    line["chordPlacements"].sort(key=lambda p: p["charIndex"])
    return f"insert_chord: line {line_index} charIndex {char_index}: {chord}"


def _apply_remove_chord(doc: dict, op: dict, idx: int) -> str:
    line_index = _require_int(op, "lineIndex", idx)
    char_index = _require_int(op, "charIndex", idx)
    expected = _optional_str(op, "from", idx)

    line = _find_line(doc["lines"], line_index, op, idx)
    placement = _find_placement(line, char_index, op, idx)
    if expected is not None and placement["chord"] != expected:
        raise PatchError(
            f"{_op_label(op, idx)}: line {line_index} charIndex {char_index} is "
            f"{placement['chord']!r}, not {expected!r} as the op expected — "
            f"nothing was changed"
        )
    removed = placement["chord"]
    line["chordPlacements"] = [p for p in line["chordPlacements"] if p["charIndex"] != char_index]
    return f"remove_chord: line {line_index} charIndex {char_index} ({removed})"


def _apply_move_chord(doc: dict, op: dict, idx: int) -> str:
    line_index = _require_int(op, "lineIndex", idx)
    from_char = _require_int(op, "fromCharIndex", idx)
    to_char = _require_int(op, "toCharIndex", idx)
    expected = _optional_str(op, "chord", idx)

    line = _find_line(doc["lines"], line_index, op, idx)
    placement = _find_placement(line, from_char, op, idx)
    if expected is not None and placement["chord"] != expected:
        raise PatchError(
            f"{_op_label(op, idx)}: line {line_index} charIndex {from_char} is "
            f"{placement['chord']!r}, not {expected!r} as the op expected — "
            f"nothing was changed"
        )
    if from_char != to_char and any(p["charIndex"] == to_char for p in line["chordPlacements"]):
        raise PatchError(
            f"{_op_label(op, idx)}: line {line_index} already has a chord at "
            f"charIndex {to_char}"
        )
    _check_charindex_bounds(line["lyrics"], to_char, op, idx)
    chord = placement["chord"]
    placement["charIndex"] = to_char
    line["chordPlacements"].sort(key=lambda p: p["charIndex"])
    return f"move_chord: line {line_index} {chord} charIndex {from_char} -> {to_char}"


# --- section ops -----------------------------------------------------------------


def _apply_set_section_name(doc: dict, op: dict, idx: int) -> str:
    section_index = _require_int(op, "sectionIndex", idx)
    to_name = _require_str(op, "to", idx)
    section = _find_section(doc["sections"], section_index, op, idx)
    old_name = section["name"]
    section["name"] = to_name
    return f"set_section_name: section {section_index} {old_name!r} -> {to_name!r}"


def _apply_set_section_bounds(doc: dict, op: dict, idx: int) -> str:
    section_index = _require_int(op, "sectionIndex", idx)
    start = _require_int(op, "startLineIndex", idx)
    end = _require_int(op, "endLineIndex", idx)
    if end < start:
        raise PatchError(f"{_op_label(op, idx)}: endLineIndex ({end}) < startLineIndex ({start})")
    n = len(doc["lines"])
    if n and (start >= n or end >= n):
        raise PatchError(
            f"{_op_label(op, idx)}: section {section_index} would reference lines "
            f"beyond {n - 1} (the song has {n} line(s))"
        )
    section = _find_section(doc["sections"], section_index, op, idx)
    old = (section["startLineIndex"], section["endLineIndex"])
    section["startLineIndex"] = start
    section["endLineIndex"] = end
    return f"set_section_bounds: section {section_index} lines {old[0]}-{old[1]} -> {start}-{end}"


# --- line ops --------------------------------------------------------------------


def _apply_split_line(doc: dict, op: dict, idx: int) -> str:
    line_index = _require_int(op, "lineIndex", idx)
    char_index = _require_int(op, "charIndex", idx)
    lines = doc["lines"]
    position = _line_position(lines, line_index, op, idx)
    line = lines[position]
    lyrics = line["lyrics"]
    if not lyrics:
        raise PatchError(
            f"{_op_label(op, idx)}: line {line_index} has no lyrics to split "
            f"(instrumental line)"
        )
    if not (0 < char_index < len(lyrics)):
        raise PatchError(
            f"{_op_label(op, idx)}: charIndex {char_index} must be strictly between "
            f"0 and {len(lyrics)} to split line {line_index} into two non-empty lines"
        )

    first_text, second_text = lyrics[:char_index], lyrics[char_index:]
    first_placements = [p for p in line["chordPlacements"] if p["charIndex"] < char_index]
    second_placements = [
        dict(p, charIndex=p["charIndex"] - char_index)
        for p in line["chordPlacements"] if p["charIndex"] >= char_index
    ]

    line["lyrics"] = first_text
    line["chordPlacements"] = first_placements
    # The new second half starts with no timing of its own — it did not
    # exist as a line a moment ago, so there is nothing honest to carry onto
    # it; the first half's timeSeconds/confidence are untouched (still the
    # earlier of the two, which they still are).
    new_line = {
        "lineIndex": line_index + 1,
        "lyrics": second_text,
        "chordPlacements": second_placements,
        "timeSeconds": None,
        "confidence": None,
    }
    for later in lines[position + 1 :]:
        later["lineIndex"] += 1
    lines.insert(position + 1, new_line)

    for section in doc["sections"]:
        if section["startLineIndex"] > line_index:
            section["startLineIndex"] += 1
        if section["endLineIndex"] >= line_index:
            section["endLineIndex"] += 1

    return f"split_line: line {line_index} split at charIndex {char_index} into lines {line_index} and {line_index + 1}"


def _apply_merge_line(doc: dict, op: dict, idx: int) -> str:
    """Merge line N with line N+1 into one line at position N.

    Pure concatenation, no separator invented between them: ops never carry
    lyric text, so any space this function added would itself be new
    content the model never supplied. This makes merge the exact algebraic
    inverse of split_line (which is itself a pure substring split — a
    boundary space, if any, already travels with whichever half it fell on)
    rather than a best-effort guess at how the two lines ought to join.
    """
    line_index = _require_int(op, "lineIndex", idx)
    lines = doc["lines"]
    position = _line_position(lines, line_index, op, idx)
    if position + 1 >= len(lines):
        raise PatchError(
            f"{_op_label(op, idx)}: line {line_index} has no following line to merge with"
        )
    first, second = lines[position], lines[position + 1]

    offset = len(first["lyrics"])
    joined = first["lyrics"] + second["lyrics"]
    merged_placements = list(first["chordPlacements"]) + [
        dict(p, charIndex=p["charIndex"] + offset) for p in second["chordPlacements"]
    ]

    first["lyrics"] = joined
    first["chordPlacements"] = merged_placements
    # first's timeSeconds/confidence are kept as-is: the earlier line's
    # timing anchors the merged line, same as snap_chords does for any line
    # with no chord of its own.

    del lines[position + 1]
    for later in lines[position + 1 :]:
        later["lineIndex"] -= 1

    for section in doc["sections"]:
        if section["startLineIndex"] > line_index + 1:
            section["startLineIndex"] -= 1
        elif section["startLineIndex"] == line_index + 1:
            section["startLineIndex"] = line_index
        if section["endLineIndex"] > line_index:
            section["endLineIndex"] -= 1

    return f"merge_line: lines {line_index} and {line_index + 1} merged into line {line_index}"


_OP_HANDLERS: dict[str, Callable[[dict, dict, int], str]] = {
    "replace_chord": _apply_replace_chord,
    "insert_chord": _apply_insert_chord,
    "remove_chord": _apply_remove_chord,
    "move_chord": _apply_move_chord,
    "set_section_name": _apply_set_section_name,
    "set_section_bounds": _apply_set_section_bounds,
    "split_line": _apply_split_line,
    "merge_line": _apply_merge_line,
}

#: The lines/sections structural ops — the only ones that renumber anything,
#: which is what makes audio.syncMap need regenerating afterward.
_STRUCTURAL_OPS = frozenset({"split_line", "merge_line"})


def describe_op_set() -> str:
    """The closed op vocabulary, for the patch system prompt."""
    return ", ".join(sorted(_OP_HANDLERS))


def parse_ops_response(document: Any) -> list[dict]:
    """Validate the top-level shape (``{"ops": [...]}``) and every op's
    ``op`` field before applying anything. Raises :class:`PatchError` naming
    exactly what's wrong — an unknown op type included."""
    if not isinstance(document, dict):
        raise PatchError(
            f'expected a JSON object {{"ops": [...]}}, got {type(document).__name__}'
        )
    ops = document.get("ops")
    if not isinstance(ops, list):
        raise PatchError('expected {"ops": [...]} — "ops" is missing or not an array')
    if not ops:
        raise PatchError('"ops" is empty — a patch must contain at least one change')
    if len(ops) > MAX_OPS:
        raise PatchError(
            f"{len(ops)} ops exceeds the cap of {MAX_OPS} — this is not a targeted "
            f"correction; re-run with a full re-analysis instead"
        )
    for i, op in enumerate(ops):
        if not isinstance(op, dict) or not isinstance(op.get("op"), str):
            raise PatchError(f'op {i}: expected an object with an "op" field, got {op!r}')
        if op["op"] not in _OP_HANDLERS:
            raise PatchError(
                f"op {i}: unknown op {op['op']!r} — the closed op set is "
                f"{describe_op_set()}"
            )
    return ops


def apply_patch(song: Song, ops: list[dict]) -> tuple[Song, list[AppliedOp]]:
    """Apply ``ops`` to `song` (the prior document) in order.

    Every field the ops don't name is copied through verbatim — there is no
    regeneration step here to lose anything, by construction. Raises
    :class:`PatchError` (never partially applies: the exception propagates
    before anything is returned) naming the exact op that failed.
    """
    doc = song.model_dump(mode="json")
    applied: list[AppliedOp] = []
    structural = False
    for i, op in enumerate(ops):
        handler = _OP_HANDLERS[op["op"]]
        description = handler(doc, op, i)
        if op["op"] in _STRUCTURAL_OPS:
            structural = True
        applied.append(
            AppliedOp(
                index=i, op_type=op["op"], description=description,
                reason=_optional_str(op, "reason", i),
            )
        )

    if structural:
        # Only split_line/merge_line can change which lineIndex a time
        # belongs to — regenerate the map exactly the way timing.snap and
        # timing.carry_forward do, so it can never disagree with the lines.
        # A pure chord/section patch never reaches this: its syncMap is
        # untouched, matching every other field it didn't name.
        doc["audio"]["syncMap"] = [
            {"lineIndex": line["lineIndex"], "time": line["timeSeconds"]}
            for line in doc["lines"]
            if line.get("timeSeconds") is not None
        ]

    return Song.model_validate(doc), applied
