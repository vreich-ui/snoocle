"""Chord symbol parsing, normalization, and transposition.

The Snoocle schema rule: a stored chord is the actual *sounding harmony*,
never a fretboard shape. Capo/tuning live in displayPreferences as
display-only transforms. Concretely:

- A source sheet that says "capo 5" with an Am shape is storing the sound
  of D minor -> we transpose shape chords UP by the capo before storage.
- Validation rejects anything that does not parse as a harmony symbol
  (tab fragments, fingering like "x02210", fret annotations).

Parsing is tolerant (unicode accidentals, `min`/`-` for minor, parens),
canonical output is conservative and stable so diffs stay meaningful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PITCH_CLASSES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH_CLASSES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

_NOTE_TO_PC = {}
for i, n in enumerate(PITCH_CLASSES_SHARP):
    _NOTE_TO_PC[n] = i
for i, n in enumerate(PITCH_CLASSES_FLAT):
    _NOTE_TO_PC[n] = i
_NOTE_TO_PC.update({"E#": 5, "B#": 0, "Cb": 11, "Fb": 4})

# Qualities we canonicalize to. Keys are lowercase match tokens; values canonical.
_QUALITY_ALIASES = {
    "": "",
    "maj": "",
    "major": "",
    "m": "m",
    "min": "m",
    "minor": "m",
    "-": "m",
    "dim": "dim",
    "o": "dim",
    "°": "dim",
    "dim7": "dim7",
    "aug": "aug",
    "+": "aug",
    "sus": "sus4",
    "sus2": "sus2",
    "sus4": "sus4",
    "5": "5",
    "6": "6",
    "m6": "m6",
    "min6": "m6",
    "69": "6/9",
    "6/9": "6/9",
    "7": "7",
    "maj7": "maj7",
    "ma7": "maj7",
    "M7": "maj7",
    "Δ": "maj7",
    "Δ7": "maj7",
    "m7": "m7",
    "min7": "m7",
    "-7": "m7",
    "mmaj7": "mMaj7",
    "mM7": "mMaj7",
    "minmaj7": "mMaj7",
    "m7b5": "m7b5",
    "ø": "m7b5",
    "ø7": "m7b5",
    "7sus2": "7sus2",
    "7sus4": "7sus4",
    "9": "9",
    "maj9": "maj9",
    "m9": "m9",
    "11": "11",
    "m11": "m11",
    "13": "13",
    "m13": "m13",
    "add9": "add9",
    "madd9": "madd9",
    "add11": "add11",
    "add2": "add9",
}

# Trailing alterations we accept (possibly several, possibly parenthesized).
_ALTERATION_RE = re.compile(r"^(?:\(?(?:[#b♯♭+-]?(?:5|9|11|13))\)?)+$")

_ROOT_RE = re.compile(r"^([A-G])([#b♯♭]?)")

_NO_CHORD_TOKENS = {"n.c.", "nc", "n.c", "x", "-", "%"}

# Things that look like fingerings/tab, not harmony: e.g. x02210, 3-2-0-0-0-3, e|---
_SHAPE_LIKE_RE = re.compile(r"^[xX0-9oO](?:[-xX0-9oO]){3,}$")


class ChordParseError(ValueError):
    pass


@dataclass(frozen=True)
class Chord:
    """A parsed sounding-harmony symbol."""

    root_pc: int  # pitch class 0-11 (C=0)
    quality: str  # canonical quality token ("", "m", "maj7", ...)
    alterations: str  # canonical trailing alterations, e.g. "b5#9" (may be "")
    bass_pc: int | None = None  # slash-chord bass pitch class

    def symbol(self, prefer_flats: bool = False) -> str:
        names = PITCH_CLASSES_FLAT if prefer_flats else PITCH_CLASSES_SHARP
        s = names[self.root_pc] + self.quality + self.alterations
        if self.bass_pc is not None and self.bass_pc != self.root_pc:
            s += "/" + names[self.bass_pc]
        return s

    def transposed(self, semitones: int) -> "Chord":
        return Chord(
            root_pc=(self.root_pc + semitones) % 12,
            quality=self.quality,
            alterations=self.alterations,
            bass_pc=None if self.bass_pc is None else (self.bass_pc + semitones) % 12,
        )


def _normalize_accidentals(s: str) -> str:
    return s.replace("♯", "#").replace("♭", "b").replace("𝄫", "bb")


def is_no_chord(token: str) -> bool:
    return token.strip().lower() in _NO_CHORD_TOKENS


def looks_like_shape(token: str) -> bool:
    """True for fingering/tab-like tokens (x02210, 0-2-2-1-0-0) that must never
    be stored as a chord identity."""
    t = token.strip()
    return bool(_SHAPE_LIKE_RE.match(t))


def parse_chord(symbol: str) -> Chord:
    """Parse a chord symbol. Raises ChordParseError on anything that is not a
    plausible harmony symbol."""
    raw = symbol.strip()
    if not raw:
        raise ChordParseError("empty chord symbol")
    if looks_like_shape(raw):
        raise ChordParseError(f"fretboard shape/fingering is not a chord identity: {raw!r}")
    s = _normalize_accidentals(raw)

    bass_pc: int | None = None
    if "/" in s:
        s, _, bass = s.partition("/")
        bass = bass.strip()
        m = _ROOT_RE.match(bass)
        if not m or m.end() != len(bass):
            # not a note bass: "C6/9"-style compound quality, rejoin and let
            # quality matching handle it
            if bass == "9" and s.endswith("6"):
                s = s + "/" + bass
            else:
                raise ChordParseError(f"unparseable bass note in {raw!r}")
        else:
            bass_pc = _NOTE_TO_PC[m.group(1) + m.group(2)]

    m = _ROOT_RE.match(s)
    if not m:
        raise ChordParseError(f"no root note in {raw!r}")
    root_pc = _NOTE_TO_PC[m.group(1) + m.group(2)]
    rest = s[m.end() :].strip()

    # Split rest into quality + trailing alterations.
    quality = None
    alterations = ""
    if rest in _QUALITY_ALIASES:
        quality = _QUALITY_ALIASES[rest]
    else:
        # longest-prefix quality match, remainder must be alterations
        for cut in range(len(rest), -1, -1):
            head, tail = rest[:cut], rest[cut:]
            if head in _QUALITY_ALIASES and (not tail or _ALTERATION_RE.match(tail)):
                quality = _QUALITY_ALIASES[head]
                alterations = tail.replace("(", "").replace(")", "").replace("+", "#")
                break
    if quality is None:
        raise ChordParseError(f"unrecognized chord quality in {raw!r}")

    return Chord(root_pc=root_pc, quality=quality, alterations=alterations, bass_pc=bass_pc)


def normalize_chord(symbol: str, prefer_flats: bool = False) -> str:
    """Canonical sounding-harmony spelling of `symbol` (no transposition)."""
    return parse_chord(symbol).symbol(prefer_flats=prefer_flats)


def transpose_chord(symbol: str, semitones: int, prefer_flats: bool = False) -> str:
    return parse_chord(symbol).transposed(semitones).symbol(prefer_flats=prefer_flats)


def shape_to_sounding(symbol: str, capo: int, prefer_flats: bool = False) -> str:
    """Convert a capo'd fretboard-shape chord symbol to the sounding harmony.

    A shape played with a capo sounds `capo` semitones HIGHER than its symbol:
    capo 5 + Am shape -> Dm sounding.
    """
    if capo < 0 or capo > 11:
        raise ValueError(f"implausible capo value: {capo}")
    return transpose_chord(symbol, capo, prefer_flats=prefer_flats)


def validate_stored_chord(symbol: str) -> None:
    """Gate for anything about to be persisted in chordPlacements.

    Raises ChordParseError when the symbol is not a valid sounding-harmony
    symbol (unparseable, tab/fingering, or a no-chord marker).
    """
    if is_no_chord(symbol):
        raise ChordParseError("no-chord markers are not storable chord identities")
    parse_chord(symbol)
