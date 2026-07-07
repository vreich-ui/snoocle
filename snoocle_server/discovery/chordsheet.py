"""Generic chord-sheet text parser.

Takes raw text from ANY web source (no site-specific scraping) and produces
schema-shaped lines with chordPlacements keyed by charIndex. Understands the
two common layouts:

  1. chord-over-lyrics:   C        G         Am
                          When I find myself in times of trouble
  2. inline:              [C]When I find my[G]self in times of trouble

plus section headers ([Verse 1], Chorus:), capo/key declarations, and
instrumental chord runs. A declared capo is transposed away at ingestion so
every stored chord is the sounding harmony; the declaration is preserved on
the candidate for provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..chords import ChordParseError, is_no_chord, parse_chord, shape_to_sounding
from ..schema.song import ChordPlacement, Line

_SECTION_WORDS = (
    "intro|verse|chorus|pre-?chorus|post-?chorus|bridge|outro|solo|instrumental|"
    "interlude|breakdown|refrain|hook|ending|coda"
)
_SECTION_RE = re.compile(
    rf"^\s*\[?\s*(?P<name>(?:{_SECTION_WORDS})(?:\s*\d+)?(?:\s*[:x]\s*\d+)?)\s*\]?\s*:?\s*$",
    re.IGNORECASE,
)
_BRACKET_HEADER_RE = re.compile(r"^\s*\[(?P<name>[^\]\[]{1,40})\]\s*$")
_CAPO_RE = re.compile(r"\bcapo\b[:\s]*(?:on\s*)?(\d{1,2})(?:st|nd|rd|th)?", re.IGNORECASE)
_KEY_RE = re.compile(r"^\s*key\b[:\s]*([A-G][#b♯♭]?\s*(?:major|minor|maj|min|m)?)\s*$", re.IGNORECASE)
_TAB_LINE_RE = re.compile(r"^\s*[eEBGDAd]?\|?[-0-9hpbrx/\\~|]{6,}\s*$")
_INLINE_CHORD_RE = re.compile(r"\[([^\]\s]{1,12})\]")

_BAR_TOKENS = {"|", "||", "-", "–", "/", "//"}
_REPEAT_RE = re.compile(r"^[x(]?\s*x?\d+[x)]?$", re.IGNORECASE)


@dataclass
class ParsedSheet:
    lines: list[Line] = field(default_factory=list)
    sections_hint: list[str] = field(default_factory=list)
    section_starts: list[tuple[str, int]] = field(default_factory=list)  # (name, lineIndex)
    declared_capo: int = 0
    declared_key: str | None = None
    chord_line_count: int = 0
    lyric_line_count: int = 0
    placement_count: int = 0

    @property
    def is_plausible(self) -> bool:
        return self.placement_count >= 4 and len(self.lines) >= 4


def _token_is_chordish(tok: str) -> bool:
    if tok in _BAR_TOKENS or is_no_chord(tok) or _REPEAT_RE.match(tok):
        return True
    try:
        parse_chord(tok)
        return True
    except ChordParseError:
        return False


def _is_chord_line(line: str) -> bool:
    tokens = line.split()
    if not tokens:
        return False
    chordish = sum(1 for t in tokens if _token_is_chordish(t))
    real_chords = 0
    for t in tokens:
        try:
            parse_chord(t)
            real_chords += 1
        except ChordParseError:
            pass
    return real_chords >= 1 and chordish / len(tokens) >= 0.8


def _chord_tokens_with_columns(line: str) -> list[tuple[int, str]]:
    out = []
    for m in re.finditer(r"\S+", line):
        tok = m.group(0)
        try:
            parse_chord(tok)
        except ChordParseError:
            continue
        out.append((m.start(), tok))
    return out


def _section_header(line: str) -> str | None:
    m = _SECTION_RE.match(line)
    if m:
        return m.group("name").strip()
    m = _BRACKET_HEADER_RE.match(line)
    if m:
        name = m.group("name").strip()
        # bracketed inline-chord lines are not headers
        try:
            parse_chord(name)
            return None
        except ChordParseError:
            return name
    return None


def _parse_inline_line(text: str) -> tuple[str, list[tuple[int, str]]] | None:
    """Parse '[C]When I find my[G]self' -> (lyrics, [(charIndex, chord), ...]).
    Returns None if the line has no valid inline chords."""
    placements: list[tuple[int, str]] = []
    lyrics_parts: list[str] = []
    pos = 0
    out_len = 0
    found = False
    for m in _INLINE_CHORD_RE.finditer(text):
        try:
            parse_chord(m.group(1))
        except ChordParseError:
            continue
        found = True
        lyrics_parts.append(text[pos : m.start()])
        out_len += m.start() - pos
        placements.append((out_len, m.group(1)))
        pos = m.end()
    if not found:
        return None
    lyrics_parts.append(text[pos:])
    lyrics = "".join(lyrics_parts).rstrip()
    return lyrics, placements


def parse_chord_sheet(text: str) -> ParsedSheet:
    sheet = ParsedSheet()
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    m = _CAPO_RE.search(text)
    if m:
        capo = int(m.group(1))
        if 0 <= capo <= 11:
            sheet.declared_capo = capo

    def add_line(lyrics: str, placements: list[tuple[int, str]]) -> None:
        cps = []
        prev = -1
        for char_index, tok in placements:
            try:
                sounding = shape_to_sounding(tok, sheet.declared_capo)
            except (ChordParseError, ValueError):
                continue
            if char_index <= prev:
                char_index = prev + 1
            cps.append(ChordPlacement(charIndex=char_index, chord=sounding))
            prev = char_index
        # a chord column can sit past the end of a shorter lyric line; pad with
        # spaces rather than clamp so alignment survives exactly
        if lyrics and cps and cps[-1].charIndex > len(lyrics):
            lyrics = lyrics.ljust(cps[-1].charIndex)
        sheet.lines.append(
            Line(lineIndex=len(sheet.lines), lyrics=lyrics, chordPlacements=cps)
        )
        sheet.placement_count += len(cps)
        if lyrics.strip():
            sheet.lyric_line_count += 1

    i = 0
    seen_musical = False  # plain text before any chords/section header is preamble junk
    while i < len(raw_lines):
        line = raw_lines[i].rstrip("\n")
        stripped = line.strip()

        if not stripped:
            i += 1
            continue
        km = _KEY_RE.match(stripped)
        if km:
            sheet.declared_key = km.group(1).strip()
            i += 1
            continue
        if _CAPO_RE.search(stripped) and len(stripped) < 40:
            i += 1
            continue
        header = _section_header(stripped)
        if header is not None:
            sheet.sections_hint.append(header)
            if sheet.section_starts and sheet.section_starts[-1][1] == len(sheet.lines):
                sheet.section_starts[-1] = (header, len(sheet.lines))  # empty section: keep last
            else:
                sheet.section_starts.append((header, len(sheet.lines)))
            seen_musical = True
            i += 1
            continue
        if _TAB_LINE_RE.match(stripped):
            i += 1
            continue

        if _is_chord_line(stripped):
            seen_musical = True
            sheet.chord_line_count += 1
            placements = _chord_tokens_with_columns(line)
            # find what follows: a lyric line -> pair them; else instrumental
            j = i + 1
            nxt = raw_lines[j].rstrip("\n") if j < len(raw_lines) else ""
            nxt_stripped = nxt.strip()
            if (
                nxt_stripped
                and not _is_chord_line(nxt_stripped)
                and _section_header(nxt_stripped) is None
                and not _TAB_LINE_RE.match(nxt_stripped)
            ):
                add_line(nxt.rstrip(), placements)
                i = j + 1
            else:
                # instrumental run: ordinal slots on an empty-lyric line
                add_line("", [(k, tok) for k, (_, tok) in enumerate(placements)])
                i += 1
            continue

        inline = _parse_inline_line(line)
        if inline is not None:
            lyrics, placements = inline
            seen_musical = True
            sheet.chord_line_count += 1
            add_line(lyrics, placements)
            i += 1
            continue

        # plain lyric line with no chords above it; drop preamble junk that
        # appears before any musical content (titles, author credits, ads)
        if seen_musical:
            add_line(line.rstrip(), [])
        i += 1

    return sheet
