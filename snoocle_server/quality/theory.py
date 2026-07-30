"""Is every stored chord explainable in the song's established key?

A chord no key explains is usually not an exotic reharmonization — it is a
transcription error that survived reconciliation: a root read a semitone off a
badly-OCR'd sheet, a source in a different key spliced into one that wasn't, a
minor written where the audio plays major. That is a quality signal nothing in
this pipeline currently sees, and it needs no audio and no model to compute.

The judgement, deliberately permissive (a false "this song is broken" is
worse than a missed one), in order:

1. **Diatonic** — every pitch class of the chord belongs to the key's scale.
   For a minor key the raised seventh counts too (harmonic minor: the V, V7
   and vii° that real minor-key songs are full of). This covers the vast
   majority of a well-transcribed pop/rock song.
2. **Borrowed or secondary dominant** — the ROOT is diatonic and the chord is
   a plain major triad or a dominant seventh. This is the D7-in-C-major case
   (V/V) and the borrowed-major case, both extremely common and both
   perfectly explainable.
3. Anything else is unexplained, and reported with its line/char index so a
   retry can be told exactly where to look.

music21 does the theory: it owns the key -> scale mapping and chord-symbol ->
pitch-class parsing, and reimplementing either here would be both worse and
untrustworthy. It is an OPTIONAL dependency (extra ``theory``) — without it
this metric reports ``None`` with the reason, exactly like an MIR engine that
isn't installed. "Not measured" and "measured as bad" must never look alike.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..chords import Chord, ChordParseError, parse_chord

log = logging.getLogger(__name__)

#: Qualities whose canonical spelling music21's ChordSymbol parser rejects
#: (e.g. "maj9", "mMaj7", "6/9"). For these the pitch classes are unknown, so
#: the check falls back to rule 2 (root membership) alone rather than guessing
#: an interval set — recorded per chord so the fallback is never invisible.
_ROOT_ONLY_FALLBACK = "quality not parseable by music21; judged on its root alone"

_KEY_RE = re.compile(r"^\s*([A-Ga-g])\s*([#b♯♭]?)\s*(.*?)\s*$")

_MODE_ALIASES = {
    "": "major",
    "major": "major",
    "maj": "major",
    "ionian": "major",
    "minor": "minor",
    "min": "minor",
    "m": "minor",
    "aeolian": "minor",
    "dorian": "dorian",
    "phrygian": "phrygian",
    "lydian": "lydian",
    "mixolydian": "mixolydian",
    "locrian": "locrian",
}

#: Chord qualities that count as "dominant/major function" for rule 2.
_MAJOR_FUNCTION_QUALITIES = {"", "7", "9", "13", "maj7", "maj9", "add9", "6", "6/9"}


@dataclass(frozen=True)
class TheoryReport:
    """The outcome of the key-explainability check."""

    share: Optional[float]  # explained / total, or None when not measurable
    explained: int
    total: int
    key: Optional[str]  # the key actually used, as music21 resolved it
    unexplained: list[dict] = field(default_factory=list)
    unavailable_reason: Optional[str] = None

    def to_dict(self) -> dict:
        out: dict = {
            "share": round(self.share, 3) if self.share is not None else None,
            "explained": self.explained,
            "total": self.total,
            "key": self.key,
        }
        if self.unexplained:
            out["unexplained"] = self.unexplained[:12]
        if self.unavailable_reason:
            out["unavailable"] = self.unavailable_reason
        return out


def _load_music21():
    """The music21 modules this check needs, or None when the extra is absent."""
    try:
        from music21 import harmony, key as m21key  # noqa: PLC0415 — optional dep

        return harmony, m21key
    except Exception as e:  # noqa: BLE001 — an optional dep, never a failure
        log.info("music21 unavailable, skipping the theory-validity metric (%s)", e)
        return None


def parse_key_name(name: str | None) -> Optional[tuple[str, str]]:
    """``("Bb", "minor")`` from "Bb minor" / "bbm" / "Bb Minor". None if unusable."""
    if not name:
        return None
    m = _KEY_RE.match(name.replace("♯", "#").replace("♭", "b"))
    if not m:
        return None
    tonic = m.group(1).upper() + m.group(2)
    mode = _MODE_ALIASES.get(m.group(3).strip().lower())
    if mode is None:
        return None
    return tonic, mode


def _explained_pitch_classes(m21key, tonic: str, mode: str) -> Optional[set[int]]:
    try:
        k = m21key.Key(tonic, mode)
        pcs = {p.pitchClass for p in k.getPitches()}
        if mode == "minor":
            # The harmonic-minor seventh: V/V7/vii° in a minor key are not
            # borrowings worth flagging, they are how minor-key songs work.
            pcs.add((k.tonic.pitchClass + 11) % 12)
        return pcs
    except Exception as e:  # noqa: BLE001 — an unusable key is "not measurable"
        log.info("music21 could not build key %s %s (%s)", tonic, mode, e)
        return None


def _chord_pitch_classes(harmony, chord: Chord) -> Optional[set[int]]:
    """Pitch classes of `chord`, ignoring any slash bass.

    The bass is excluded on purpose: this metric is about harmony, and a
    slash bass outside the key is a different (and much rarer) error than a
    wrong chord. Canonical sharp spelling is used because that is what
    music21's parser accepts — it reads "Bb" as B natural plus a quality
    "bmaj7"-style token.
    """
    figure = Chord(
        root_pc=chord.root_pc, quality=chord.quality, alterations=chord.alterations
    ).symbol(prefer_flats=False)
    try:
        return set(harmony.ChordSymbol(figure).pitchClasses)
    except Exception:  # noqa: BLE001 — see _ROOT_ONLY_FALLBACK
        return None


def theory_validity(
    chords: Sequence[tuple[int, int, str]], key_name: str | None
) -> TheoryReport:
    """Share of `chords` explainable in `key_name`.

    `chords` is ``(lineIndex, charIndex, symbol)`` in reading order — the
    indexes ride along so an unexplained chord can be named precisely rather
    than counted.
    """
    if not chords:
        return TheoryReport(
            share=None, explained=0, total=0, key=None,
            unavailable_reason="the document has no chords to check",
        )

    parsed = parse_key_name(key_name)
    if parsed is None:
        return TheoryReport(
            share=None, explained=0, total=len(chords), key=None,
            unavailable_reason=(
                f"no usable established key (metadata/MIR key={key_name!r})"
                if key_name
                else "no established key on the song or in the MIR analysis"
            ),
        )
    tonic, mode = parsed

    modules = _load_music21()
    if modules is None:
        return TheoryReport(
            share=None, explained=0, total=len(chords), key=f"{tonic} {mode}",
            unavailable_reason=(
                "music21 is not installed (pip install 'snoocle-server[theory]')"
            ),
        )
    harmony, m21key = modules

    in_key = _explained_pitch_classes(m21key, tonic, mode)
    if in_key is None:
        return TheoryReport(
            share=None, explained=0, total=len(chords), key=f"{tonic} {mode}",
            unavailable_reason=f"music21 could not interpret the key {tonic} {mode!r}",
        )

    explained = 0
    unexplained: list[dict] = []
    for line_index, char_index, symbol in chords:
        try:
            chord = parse_chord(symbol)
        except ChordParseError:
            continue  # the schema forbids these; never counted either way
        pcs = _chord_pitch_classes(harmony, chord)
        root_in_key = chord.root_pc in in_key
        major_function = chord.quality in _MAJOR_FUNCTION_QUALITIES

        if pcs is not None and pcs <= in_key:
            explained += 1  # rule 1: fully diatonic
        elif root_in_key and major_function:
            explained += 1  # rule 2: borrowed major / secondary dominant
        else:
            unexplained.append(
                {
                    "lineIndex": line_index,
                    "charIndex": char_index,
                    "chord": symbol,
                    "reason": (
                        _ROOT_ONLY_FALLBACK
                        if pcs is None
                        else (
                            f"no function in {tonic} {mode}: "
                            + (
                                "root is outside the key"
                                if not root_in_key
                                else "chord tones fall outside the key"
                            )
                        )
                    ),
                }
            )

    total = explained + len(unexplained)
    return TheoryReport(
        share=(explained / total) if total else None,
        explained=explained,
        total=total,
        key=f"{tonic} {mode}",
        unexplained=unexplained,
    )
