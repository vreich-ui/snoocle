"""Preserve a prior version's audio-derived timing across a run that did not
listen to the audio.

Why this exists
---------------
``snap_chords`` is the ONLY pass that populates timing, and by its own
contract it is a no-op when ``mir is None``. A re-analysis with
``scope.listen=false`` passes ``mir=None``, so on that path the reconciler's
freshly-emitted document — which correctly leaves ``timeSeconds``/``beat``/
section times empty, since a post-pass normally fills them — went straight to
the store with its timing gone: ``audio.beats`` emptied, ``metadata.bpm``
nulled, every ``ChordPlacement.beat`` null, every section time null, and the
lines collapsed onto whatever the reconciler happened to emit. It validated,
it stored, and the loss was silent.

This module is the post-pass that runs INSTEAD of ``snap_chords`` on that
path. It invents nothing: every timestamp it writes is copied from the prior
version of the same song. The reconciler is free to change the document
(this is what it was asked to do) — added lines, added placements, renamed
sections — so the pass is a MATCHING problem, and an element with no
confident match keeps empty timing rather than stealing a neighbour's.

Matching, in order:

1. **Lines** — a ``difflib`` alignment over normalized lyrics. Equal blocks
   pair one-to-one; within a replaced block, same-position lines pair only
   when they are still ≥75% similar (a typo fix keeps its time, a rewritten
   line does not). Insertions and deletions shift everything after them
   without breaking the pairing, and the alignment is order-preserving, so
   carried line times stay non-decreasing exactly as the schema requires.
2. **Placements**, within a matched line pair — exact ``(charIndex, chord)``
   first, then reading order within the line among the leftovers, restricted
   to the same chord symbol and to the gap between the neighbouring exact
   matches (so a fallback match can never cross one). A placement the
   reconciler ADDED — the Am/G and C/B walkdowns of the bug report — has no
   same-chord partner left and keeps empty timing.
3. **Sections** — by ``sectionIndex`` plus line range: either the range is
   unchanged, or the section's first line still aligns into the prior
   section's range. Times are carried as a PAIR (both or neither), so a
   half-carried section can never trip the ``endTime >= startTime`` rule.

``audio.beats``, ``metadata.bpm`` and ``audio.analyzedVideoId`` are copied
wholesale — they describe the recording, not the document, and nothing about
re-reading the chord sheet can have changed them. ``audio.syncMap`` is
REGENERATED from the resulting line times (never merged), the same way
``snap_chords`` does it, so the two can never disagree.

A ProvenanceEntry documents the pass (``action="timing-carry-forward"``)
with counts of what was preserved vs left empty, in the same notes shape as
``timing-snap`` so the acceptance script and the scorecard can read either.

:func:`audio_data_lost` is the independent backstop — see its docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Optional

from ..schema.song import ChordPlacement, Line, Section, Song, SyncPoint, ProvenanceEntry

#: Provenance ``action`` for this pass. The acceptance script and the evidence
#: manifest read it alongside ``"timing-snap"``.
ACTION = "timing-carry-forward"

#: How similar two lines' lyrics must still be, inside a replaced block, for
#: the later one to inherit the earlier one's time. High on purpose: this is
#: the "the agent fixed a typo" allowance, not a fuzzy re-match.
LINE_EDIT_THRESHOLD = 0.75


def _normalize(text: str) -> str:
    """Lyrics reduced to what makes two lines "the same line": case and
    punctuation are the agent's to change freely, words are not."""
    return " ".join("".join(c for c in text.lower() if c.isalnum() or c.isspace()).split())


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def align_lines(prior_lines: list[Line], new_lines: list[Line]) -> dict[int, int]:
    """Map ``new position -> prior position`` for lines that are the same line.

    Order-preserving by construction (``SequenceMatcher`` opcodes never
    cross), which is what keeps the carried line times non-decreasing.
    """
    a = [_normalize(line.lyrics) for line in prior_lines]
    b = [_normalize(line.lyrics) for line in new_lines]
    mapping: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[j1 + k] = i1 + k
        elif tag == "replace":
            for k in range(min(i2 - i1, j2 - j1)):
                if _similarity(a[i1 + k], b[j1 + k]) >= LINE_EDIT_THRESHOLD:
                    mapping[j1 + k] = i1 + k
    return mapping


def match_placements(
    prior_placements: list[ChordPlacement], new_placements: list[ChordPlacement]
) -> dict[int, int]:
    """Map ``new position -> prior position`` within one matched line pair.

    Never crosses: pass 1 pairs on ``(charIndex, chord)``, which is already
    order-preserving because both lists ascend by charIndex; pass 2 only ever
    looks between the exact matches that bracket it. A new placement with no
    same-chord partner left is deliberately absent from the result.
    """
    matched: dict[int, int] = {}
    claimed: set[int] = set()

    for j, p in enumerate(new_placements):
        for i, q in enumerate(prior_placements):
            if i in claimed:
                continue
            if q.charIndex == p.charIndex and q.chord == p.chord:
                matched[j] = i
                claimed.add(i)
                break

    for j, p in enumerate(new_placements):
        if j in matched:
            continue
        lower = max((matched[k] for k in matched if k < j), default=-1)
        upper = min((matched[k] for k in matched if k > j), default=len(prior_placements))
        for i in range(lower + 1, upper):
            if i in claimed or prior_placements[i].chord != p.chord:
                continue
            matched[j] = i
            claimed.add(i)
            break

    return matched


@dataclass(frozen=True)
class CarryForwardStats:
    """What the pass preserved and what it left empty. Reported in the step
    text, the provenance notes, and the pass's own tests."""

    lines_total: int = 0
    lines_carried: int = 0       # got a time FROM the prior version
    lines_empty: int = 0         # still have no time afterwards
    placements_total: int = 0
    placements_carried: int = 0
    placements_empty: int = 0
    sections_total: int = 0
    sections_carried: int = 0
    beats_carried: bool = False
    bpm_carried: bool = False
    analyzed_video_id_carried: bool = False

    @property
    def match_ratio(self) -> Optional[float]:
        if not self.placements_total:
            return None
        return self.placements_carried / self.placements_total

    def notes(self) -> str:
        """Same shape as ``timing-snap``'s notes (``matched N/M chord
        placement(s) ...``; ``N/M line(s) timed``) so every existing reader
        parses it, plus the left-empty counts this pass is accountable for."""
        return (
            f"carried timing forward from the prior version (no new audio "
            f"analysis this run); matched "
            f"{self.placements_carried}/{self.placements_total} chord "
            f"placement(s) to the prior timeline "
            f"({self.placements_empty} left empty); "
            f"{self.lines_total - self.lines_empty}/{self.lines_total} line(s) "
            f"timed ({self.lines_empty} left empty); "
            f"{self.sections_carried}/{self.sections_total} section time(s) "
            f"carried; beats={'carried' if self.beats_carried else 'kept/empty'}, "
            f"bpm={'carried' if self.bpm_carried else 'kept/empty'}"
        )

    def describe(self) -> str:
        """Compact one-liner for the pipeline's step report."""
        return (
            f"{self.placements_carried}/{self.placements_total} placement time(s), "
            f"{self.lines_carried}/{self.lines_total} line time(s), "
            f"{self.sections_carried}/{self.sections_total} section time(s) "
            f"carried forward"
        )


def _carry_sections(
    new_sections: list[Section], prior_sections: list[Section], line_map: dict[int, int]
) -> tuple[list[Section], int]:
    prior_by_index = {s.sectionIndex: s for s in prior_sections}
    out: list[Section] = []
    carried = 0
    for section in new_sections:
        prior = prior_by_index.get(section.sectionIndex)
        take = (
            prior is not None
            and section.startTime is None
            and section.endTime is None
            and (prior.startTime is not None or prior.endTime is not None)
            and _same_line_range(section, prior, line_map)
        )
        if take:
            out.append(
                section.model_copy(
                    update={"startTime": prior.startTime, "endTime": prior.endTime}
                )
            )
            carried += 1
        else:
            out.append(section)
    return out, carried


def _same_line_range(section: Section, prior: Section, line_map: dict[int, int]) -> bool:
    if (section.startLineIndex, section.endLineIndex) == (
        prior.startLineIndex,
        prior.endLineIndex,
    ):
        return True
    # Lines were added or removed: the section is still "the same section" if
    # its first line is one the alignment recognized, and that line lived
    # inside the prior section's range.
    aligned = line_map.get(section.startLineIndex)
    return aligned is not None and prior.startLineIndex <= aligned <= prior.endLineIndex


def carry_forward_timing(
    song: Song,
    prior: Song,
    *,
    audio_fallback: Optional[Song] = None,
    prior_version: Optional[str] = None,
) -> tuple[Song, CarryForwardStats]:
    """Pure: returns a NEW Song with `prior`'s timing carried onto `song`,
    plus the counts of what was preserved.

    `audio_fallback` supplies ``beats``/``bpm``/``analyzedVideoId`` when
    `prior` itself lacks them — the caller may hold two priors (the
    human-edited document the client sent, and the stored latest version),
    and the audio-derived fields should come from whichever actually has
    them. `prior_version` is recorded in the provenance entry's sources.
    """
    line_map = align_lines(prior.lines, song.lines)

    new_lines: list[Line] = []
    lines_carried = 0
    lines_empty = 0
    placements_total = 0
    placements_carried = 0
    placements_empty = 0

    for position, line in enumerate(song.lines):
        prior_line = (
            prior.lines[line_map[position]] if position in line_map else None
        )

        placement_map = (
            match_placements(prior_line.chordPlacements, line.chordPlacements)
            if prior_line is not None
            else {}
        )
        new_placements: list[ChordPlacement] = []
        for pi, placement in enumerate(line.chordPlacements):
            placements_total += 1
            update: dict = {}
            if pi in placement_map and placement.timeSeconds is None:
                source = prior_line.chordPlacements[placement_map[pi]]
                if source.timeSeconds is not None:
                    update["timeSeconds"] = source.timeSeconds
                    update["confidence"] = source.confidence
                    update["beat"] = source.beat
            if update:
                placements_carried += 1
                new_placements.append(placement.model_copy(update=update))
            else:
                if placement.timeSeconds is None:
                    placements_empty += 1
                new_placements.append(placement)

        line_update: dict = {"chordPlacements": new_placements}
        if (
            prior_line is not None
            and line.timeSeconds is None
            and prior_line.timeSeconds is not None
        ):
            line_update["timeSeconds"] = prior_line.timeSeconds
            line_update["confidence"] = prior_line.confidence
            lines_carried += 1
        elif line.timeSeconds is None:
            lines_empty += 1
        new_lines.append(line.model_copy(update=line_update))

    new_sections, sections_carried = _carry_sections(song.sections, prior.sections, line_map)

    # Regenerated, never merged — the syncMap is a projection of the line
    # times and must not be able to disagree with them.
    sync_map = [
        SyncPoint(lineIndex=line.lineIndex, time=line.timeSeconds)
        for line in new_lines
        if line.timeSeconds is not None
    ]

    audio_sources = [s for s in (prior, audio_fallback) if s is not None]
    audio_update: dict = {"syncMap": sync_map}

    beats = next((s.audio.beats for s in audio_sources if s.audio.beats), [])
    if not song.audio.beats and beats:
        audio_update["beats"] = list(beats)

    analyzed_video_id = next(
        (s.audio.analyzedVideoId for s in audio_sources if s.audio.analyzedVideoId), None
    )
    if song.audio.analyzedVideoId is None and analyzed_video_id:
        audio_update["analyzedVideoId"] = analyzed_video_id

    bpm = next((s.metadata.bpm for s in audio_sources if s.metadata.bpm is not None), None)
    metadata_update: dict = {}
    if song.metadata.bpm is None and bpm is not None:
        metadata_update["bpm"] = bpm

    stats = CarryForwardStats(
        lines_total=len(song.lines),
        lines_carried=lines_carried,
        lines_empty=lines_empty,
        placements_total=placements_total,
        placements_carried=placements_carried,
        placements_empty=placements_empty,
        sections_total=len(song.sections),
        sections_carried=sections_carried,
        beats_carried="beats" in audio_update,
        bpm_carried="bpm" in metadata_update,
        analyzed_video_id_carried="analyzedVideoId" in audio_update,
    )

    entry = ProvenanceEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        actor="snoocle-server/timing",
        action=ACTION,
        sources=[f"prior-version:{prior_version}" if prior_version else "prior-version"],
        confidence=stats.match_ratio,
        notes=stats.notes(),
    )

    updated = song.model_copy(
        update={
            "lines": new_lines,
            "sections": new_sections,
            "audio": song.audio.model_copy(update=audio_update),
            "metadata": (
                song.metadata.model_copy(update=metadata_update)
                if metadata_update
                else song.metadata
            ),
            "provenance": list(song.provenance) + [entry],
        }
    )
    # model_copy(update=...) does not re-run validators (pydantic v2). Same
    # rule as timing.snap: this pass must never hand the pipeline an invalid
    # Song, so re-validate explicitly and fail loudly instead of persisting.
    return Song.model_validate(updated.model_dump()), stats


def audio_data_lost(prior: Song, candidate: Song) -> list[str]:
    """Names of audio-derived fields `candidate` drops that `prior` had.

    Both fields are audio-DERIVED: nothing that happens while re-reading a
    chord sheet can legitimately empty them, so an empty one next to a
    populated prior is a bug in whatever path produced it — not a judgement
    call.

    Independent of :func:`carry_forward_timing` on purpose. That pass fixes
    the one path we know about (``scope.listen=false``); this answers the
    broader question the bug actually posed — "could a run empty these
    silently again, by some path nobody has thought of yet?" — with "no,
    whatever path got there". Returns ``[]`` when nothing was lost.
    """
    lost: list[str] = []
    if prior.audio.beats and not candidate.audio.beats:
        lost.append(f"audio.beats ({len(prior.audio.beats)} entries -> 0)")
    if prior.metadata.bpm is not None and candidate.metadata.bpm is None:
        lost.append(f"metadata.bpm ({prior.metadata.bpm} -> null)")
    return lost
