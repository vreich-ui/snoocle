"""Deterministic Song export serializers (master plan B6).

Two plain-text formats share one inline-bracket layout — `[Chord]` markers
spliced into lyric lines at their stored charIndex, section names on their
own bracketed line (`[Verse 1]`) — which is exactly the layout the
EXISTING generic chord-sheet parser (discovery/chordsheet.py) already
understands. That's deliberate: `to_chordpro`/`to_txt`'s output round-trips
through `parse_chord_sheet` back to an equivalent line/chord set (see
tests/test_export.py) — the same bracket-text contract the web Edit tab's
"paste a sheet" box relies on.

`to_chordpro` additionally prefixes ChordPro metadata directives
(`{title: ...}` etc.) for interop with other ChordPro-reading tools. Those
directives are NOT understood by our own parser — nor do they need to be:
only the lyric/chord LINES have to round-trip, not the metadata banner — so
`to_txt` omits them for a leaner "just the sheet" export.

Instrumental lines (no lyrics, chords only) serialize as space-joined
bracketed chords, e.g. `[C] [Am] [F] [G]`, and parse back the same way via
the generic parser's inline-chord path.
"""

from __future__ import annotations

from .schema.song import ChordPlacement, Section, Song


def _inline_chord_line(lyrics: str, placements: list[ChordPlacement]) -> str:
    if not placements:
        return lyrics
    if not lyrics:
        return " ".join(f"[{p.chord}]" for p in placements)
    out: list[str] = []
    prev = 0
    for p in placements:
        idx = min(p.charIndex, len(lyrics))
        out.append(lyrics[prev:idx])
        out.append(f"[{p.chord}]")
        prev = idx
    out.append(lyrics[prev:])
    return "".join(out)


def _section_boundaries(song: Song) -> tuple[dict[int, Section], dict[int, Section]]:
    starts: dict[int, Section] = {}
    ends: dict[int, Section] = {}
    for s in song.sections:
        starts.setdefault(s.startLineIndex, s)
        ends.setdefault(s.endLineIndex, s)
    return starts, ends


def _body_lines(song: Song) -> list[str]:
    starts, ends = _section_boundaries(song)
    out: list[str] = []
    open_section: Section | None = None
    for line in song.lines:
        sec = starts.get(line.lineIndex)
        if sec is not None:
            out.append(f"[{sec.name}]")
            open_section = sec
        out.append(_inline_chord_line(line.lyrics, line.chordPlacements))
        end_sec = ends.get(line.lineIndex)
        if end_sec is not None and open_section is end_sec:
            open_section = None
    return out


def to_txt(song: Song) -> str:
    """Just the sheet: section headers + inline-bracket chords-over-lyrics,
    no metadata banner. Round-trips through `parse_chord_sheet`."""
    return "\n".join(_body_lines(song)) + "\n"


def to_chordpro(song: Song) -> str:
    """The same sheet body as `to_txt`, preceded by a ChordPro metadata
    banner (`{title: ...}` etc.) for interop with other ChordPro tools."""
    m = song.metadata
    header: list[str] = []
    if m.title:
        header.append(f"{{title: {m.title}}}")
    if m.artist:
        header.append(f"{{artist: {m.artist}}}")
    if m.key:
        header.append(f"{{key: {m.key}}}")
    if m.bpm:
        header.append(f"{{tempo: {int(round(m.bpm))}}}")
    body = _body_lines(song)
    parts = (header + [""] if header else []) + body
    return "\n".join(parts) + "\n"
