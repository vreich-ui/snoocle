"""The lyric-reference protocol: the model points at lyrics, it never retypes them.

Why this exists
---------------
A valid Song document IS the complete lyrics of a copyrighted song. So every
successful reconciliation used to ask the model for a full verbatim
reproduction — and two live runs today ("Yesterday", and the Creedence song)
were blocked by Anthropic's content filter with FOUR candidate sources
successfully fetched. The old error copy blamed "no chord/lyric sources were
found"; the evidence contradicts that. The cause is structural, not a
retrieval failure, and no amount of prompt wording changes it. Removing
lyrics from the model's OUTPUT does.

So the agent-facing line format is a reference:

    {"lineIndex": 7,
     "lyricRef": {"sourceId": "ultimate-guitar-1087597", "line": 12},
     "chordPlacements": [{"charIndex": 0, "chord": "F"}]}

and :func:`splice_lyrics` — deterministic local code, no model — substitutes
the real text out of the candidate that is already in the run's context,
before validation and storage.

What this is NOT
----------------
A schema change. ``lyricRef`` never reaches the stored Song: this is an
internal protocol between the server and the model it prompts, and the
document that lands in the store is byte-identical in shape to what landed
there before. ``schema/song.py`` is untouched.

The four rules that make it a guarantee rather than a request
-------------------------------------------------------------
1. **Retyped lyrics are rejected.** A line carrying non-empty ``lyrics``
   violates the contract and goes back through the repair loop. Instrumental
   lines are the one exception and must say so explicitly with ``lyrics: ""``
   and no ref — an empty line is a deliberate statement, not a missing one.
2. **charIndex is validated AFTER splicing** (:func:`splice_lyrics` again),
   against the resolved text. The model places chords into text it can see
   but is not retyping, so an index past the end of the real line is the
   predictable failure — it is named, per line, and repaired. Never clamped:
   a silently moved chord is a wrong transcription that looks right.
3. **Unresolvable refs are fatal, never a fallback.** A sourceId that is not
   in context, or a line index out of range, fails the run. The entire point
   is that a lyric with no valid provenance cannot reach the store, and
   "just let the model write that one from memory" is exactly the hole this
   closes.
4. **Overrides are audited and capped.** ``lyricOverride`` exists for the
   genuine cases — merging two partial sources, fixing an obvious source
   typo, a line no source covers — and every one lands in provenance with
   its line index and reason. Past ~15% of lines it is not an escape hatch
   any more, it is reference resolution being broken, and the run fails.

The prior song is a source
--------------------------
It is registered under the reserved id ``prior-song``. Without it a
notes-only run (``listen=off, reconcile=off``) — which deliberately gathers
no candidates and asks the model to return the user's own document with
their notes applied — could only comply by retyping every line as an
override, which would trip rule 4 and fail every time.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Optional

from ..discovery.models import CandidateSource

#: Reserved source id for the prior version of the song being corrected.
PRIOR_SONG_SOURCE_ID = "prior-song"

#: Share of a song's lines that may carry a ``lyricOverride`` before the run
#: is treated as broken rather than exceptional. One override is always
#: allowed — the cap is aimed at systemic resolution failure, not at a single
#: legitimately merged line.
MAX_OVERRIDE_SHARE = 0.15

#: How much of a resolved line to quote back in a repair message. Enough to
#: locate a chord, short enough not to bloat the retry.
_QUOTE_CHARS = 80


class LyricSpliceError(ValueError):
    """The model's document violates the reference contract in a way IT can
    fix: retyped lyrics, a missing/duplicated lyric source, a malformed ref,
    a charIndex past the end of the resolved line. Repairable — the engine
    feeds these back through the existing repair loop."""


class UnresolvableLyricRefError(RuntimeError):
    """A ref that cannot be resolved against this run's context: an unknown
    sourceId, or a line index outside that source. NOT repairable by design
    (see rule 3) — the run fails rather than inventing the line."""


@dataclass(frozen=True)
class LyricOverride:
    """One line the model wrote itself instead of referencing, and why."""

    line_index: int
    reason: str


@dataclass(frozen=True)
class SpliceResult:
    """A spliced document plus what it took to get there."""

    document: dict
    refs_resolved: int = 0            # lines whose words came from a source
    overrides: tuple[LyricOverride, ...] = ()
    instrumental: int = 0             # lines that legitimately have no words


def build_ref_index(
    candidates: list[CandidateSource] | None,
    prior_song: dict | None = None,
) -> dict[str, list[str]]:
    """``sourceId -> that source's lyric lines, by line index``.

    Every source the model may point at, and nothing else. Built from the
    same objects the prompt hands it, so "in the prompt" and "resolvable"
    cannot drift apart.
    """
    index: dict[str, list[str]] = {}
    for cand in candidates or []:
        index[cand.sourceId] = [line.lyrics for line in cand.lines]
    if prior_song:
        lines = prior_song.get("lines")
        if isinstance(lines, list):
            index[PRIOR_SONG_SOURCE_ID] = [
                str((line or {}).get("lyrics", "")) if isinstance(line, dict) else ""
                for line in lines
            ]
    return index


def describe_ref_sources(index: dict[str, list[str]]) -> str:
    """One line per referenceable source: its id and its valid line range.
    Goes in the prompt so the model never has to guess either."""
    if not index:
        return "(none — no sources were gathered for this run)"
    return "\n".join(
        f"- {source_id!r}: lines 0-{len(lines) - 1}" if lines
        else f"- {source_id!r}: (no lines)"
        for source_id, lines in index.items()
    )


def agent_song_json_schema(base_schema: dict) -> dict:
    """The Song schema as the MODEL sees it: `lyrics` optional, `lyricRef` /
    `lyricOverride` allowed on a line.

    Derived from the real schema rather than hand-written, so it cannot drift
    from what the splice output must validate against.
    """
    schema = copy.deepcopy(base_schema)
    line = (schema.get("$defs") or {}).get("Line")
    if line is None:  # pragma: no cover — the Song schema always defines Line
        return schema
    line["required"] = [r for r in line.get("required", []) if r != "lyrics"]
    line["properties"]["lyrics"] = {
        "type": "string",
        "const": "",
        "description": (
            "ONLY for an instrumental line with no words. Must be the empty "
            "string, and the line must then carry no lyricRef. Never retype "
            "a song's words here — use lyricRef."
        ),
    }
    line["properties"]["lyricRef"] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["sourceId", "line"],
        "properties": {
            "sourceId": {
                "type": "string",
                "description": "id of a source in this request (a candidate's "
                f"sourceId, or {PRIOR_SONG_SOURCE_ID!r} for the prior song).",
            },
            "line": {
                "type": "integer",
                "minimum": 0,
                "description": "lineIndex WITHIN that source. The server "
                "substitutes that line's text verbatim.",
            },
        },
        "description": "Where this line's words come from. The normal case: "
        "every line with words carries one of these.",
    }
    line["properties"]["lyricOverride"] = {
        "type": "string",
        "description": (
            "Escape hatch — the line's text, written out, ONLY when no single "
            "source line is right (merging two partial sources, fixing an "
            "obvious source typo, a line no source covers). Requires "
            "lyricOverrideReason. Rare by design: too many fails the run."
        ),
    }
    line["properties"]["lyricOverrideReason"] = {
        "type": "string",
        "description": "Why a reference was not usable for this line. "
        "Required with lyricOverride; recorded in the song's provenance.",
    }
    line["description"] = (
        "Exactly ONE of: lyricRef (the normal case), lyricOverride + "
        "lyricOverrideReason (rare), or lyrics:\"\" (instrumental line)."
    )
    return schema


def _resolve_ref(ref: Any, index: dict[str, list[str]], line_index: int) -> str:
    if not isinstance(ref, dict):
        raise LyricSpliceError(
            f"line {line_index}: lyricRef must be an object "
            f'{{"sourceId": ..., "line": ...}}, got {type(ref).__name__}'
        )
    source_id = ref.get("sourceId")
    line_no = ref.get("line")
    if not isinstance(source_id, str) or not isinstance(line_no, int) or isinstance(line_no, bool):
        raise LyricSpliceError(
            f"line {line_index}: lyricRef needs a string sourceId and an "
            f"integer line, got {ref!r}"
        )
    if source_id not in index:
        raise UnresolvableLyricRefError(
            f"line {line_index}: lyricRef points at source {source_id!r}, which "
            f"is not part of this run. Available: {sorted(index) or '(none)'}"
        )
    lines = index[source_id]
    if not 0 <= line_no < len(lines):
        extent = f"{len(lines)} line(s) (0-{len(lines) - 1})" if lines else "no lines"
        raise UnresolvableLyricRefError(
            f"line {line_index}: lyricRef {source_id!r} line {line_no} is out of "
            f"range — that source has {extent}"
        )
    return lines[line_no]


def _resolve_line_text(
    line: dict, line_index: int, index: dict[str, list[str]]
) -> tuple[str, Optional[LyricOverride]]:
    """The text for one drafted line, plus an override record if it was one.

    Enforces "exactly one of" here rather than in the schema: JSON Schema can
    express it, but a model that gets it wrong needs a sentence, not a
    oneOf-branch dump.
    """
    has_ref = "lyricRef" in line and line["lyricRef"] is not None
    has_override = "lyricOverride" in line and line["lyricOverride"] is not None
    raw_lyrics = line.get("lyrics")
    has_lyrics = isinstance(raw_lyrics, str)

    if sum((has_ref, has_override, has_lyrics and raw_lyrics != "")) > 1:
        raise LyricSpliceError(
            f"line {line_index}: give exactly ONE of lyricRef, lyricOverride, "
            f'or lyrics:"" — this line has more than one'
        )

    if has_ref:
        if has_lyrics and raw_lyrics != "":
            raise LyricSpliceError(
                f"line {line_index}: a line with a lyricRef must not also carry "
                f"lyrics text — the server substitutes it"
            )
        return _resolve_ref(line["lyricRef"], index, line_index), None

    if has_override:
        text = line["lyricOverride"]
        if not isinstance(text, str):
            raise LyricSpliceError(
                f"line {line_index}: lyricOverride must be a string, got "
                f"{type(text).__name__}"
            )
        reason = line.get("lyricOverrideReason")
        if not isinstance(reason, str) or not reason.strip():
            raise LyricSpliceError(
                f"line {line_index}: lyricOverride requires a non-empty "
                f'lyricOverrideReason (e.g. "no source covers this line") — '
                f"every override is recorded in the song's provenance"
            )
        return text, LyricOverride(line_index=line_index, reason=reason.strip())

    if has_lyrics:
        if raw_lyrics != "":
            raise LyricSpliceError(
                f"line {line_index}: lyrics must not be written out. Reference "
                f"the source line instead: "
                f'"lyricRef": {{"sourceId": ..., "line": ...}}. Only an '
                f'instrumental line may carry lyrics, and only as ""'
            )
        return "", None

    raise LyricSpliceError(
        f"line {line_index}: no lyricRef, no lyricOverride, and no explicit "
        f'lyrics:"" — every line must say where its words come from (or that '
        f"it has none)"
    )


def _check_placements(line: dict, line_index: int, lyrics: str, source: str) -> None:
    """charIndex validity against the RESOLVED text. Never clamps: a chord
    moved to fit is a wrong transcription that looks right."""
    placements = line.get("chordPlacements") or []
    if not isinstance(placements, list):
        raise LyricSpliceError(
            f"line {line_index}: chordPlacements must be an array, got "
            f"{type(placements).__name__}"
        )
    previous = -1
    for placement in placements:
        if not isinstance(placement, dict):
            raise LyricSpliceError(
                f"line {line_index}: each chordPlacement must be an object"
            )
        char_index = placement.get("charIndex")
        if not isinstance(char_index, int) or isinstance(char_index, bool) or char_index < 0:
            raise LyricSpliceError(
                f"line {line_index}: chordPlacements.charIndex must be a "
                f"non-negative integer, got {char_index!r}"
            )
        if char_index <= previous:
            raise LyricSpliceError(
                f"line {line_index}: chordPlacements must be strictly ascending "
                f"by charIndex (got {char_index} after {previous})"
            )
        previous = char_index
        # An empty-lyric (instrumental) line uses ordinal slots 0,1,2,... and
        # has no text to run past — the schema allows that and so does this.
        if lyrics and char_index > len(lyrics):
            raise LyricSpliceError(
                f"line {line_index}: charIndex {char_index} is past the end of "
                f"the resolved line, which is {len(lyrics)} character(s) long "
                f"({source}): {lyrics[:_QUOTE_CHARS]!r}. Re-place this chord "
                f"within the real line, or reference a different source line"
            )


def splice_lyrics(document: Any, index: dict[str, list[str]]) -> SpliceResult:
    """Resolve every ``lyricRef`` into real text and return a Song-shaped dict.

    The returned document carries no protocol fields — ``lyricRef``,
    ``lyricOverride`` and ``lyricOverrideReason`` are consumed here and never
    reach the stored Song.

    Raises :class:`LyricSpliceError` for anything the model can fix (the
    engine repairs), :class:`UnresolvableLyricRefError` for a ref this run
    cannot honour (the engine fails).
    """
    if not isinstance(document, dict):
        raise LyricSpliceError(
            f"expected a JSON object for the Song, got {type(document).__name__}"
        )
    lines = document.get("lines")
    if lines is None:
        # Not our error to raise: a document with no `lines` at all fails
        # Song validation next, with a message about the field it is missing.
        return SpliceResult(document=document)
    if not isinstance(lines, list):
        raise LyricSpliceError(f"lines must be an array, got {type(lines).__name__}")

    spliced: list[dict] = []
    overrides: list[LyricOverride] = []
    refs_resolved = 0
    instrumental = 0
    for position, line in enumerate(lines):
        if not isinstance(line, dict):
            raise LyricSpliceError(
                f"line at position {position}: expected an object, got "
                f"{type(line).__name__}"
            )
        line_index = line.get("lineIndex")
        if not isinstance(line_index, int) or isinstance(line_index, bool):
            line_index = position
        text, override = _resolve_line_text(line, line_index, index)
        if override is not None:
            overrides.append(override)
            source = "your lyricOverride for this line"
        elif "lyricRef" in line and isinstance(line["lyricRef"], dict):
            refs_resolved += 1
            source = (
                f"resolved from {line['lyricRef'].get('sourceId')!r} line "
                f"{line['lyricRef'].get('line')}"
            )
        else:
            instrumental += 1
            source = "instrumental line"
        resolved = {k: v for k, v in line.items()
                    if k not in ("lyricRef", "lyricOverride", "lyricOverrideReason")}
        resolved["lyrics"] = text
        _check_placements(resolved, line_index, text, source)
        spliced.append(resolved)

    allowance = max(1, math.ceil(MAX_OVERRIDE_SHARE * len(spliced)))
    if len(overrides) > allowance:
        raise UnresolvableLyricRefError(
            f"{len(overrides)} of {len(spliced)} line(s) were written out as "
            f"lyricOverride, over the {MAX_OVERRIDE_SHARE:.0%} ceiling "
            f"({allowance} allowed). That many overrides means reference "
            f"resolution is broken, not that the sources were bad — failing "
            f"rather than storing lyrics with no source. Overridden lines: "
            + ", ".join(str(o.line_index) for o in overrides[:20])
        )

    out = dict(document)
    out["lines"] = spliced
    return SpliceResult(
        document=out,
        refs_resolved=refs_resolved,
        overrides=tuple(overrides),
        instrumental=instrumental,
    )
