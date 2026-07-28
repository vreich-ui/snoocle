"""Forced alignment: known lyrics + audio -> word and line times (B2).

This is not transcription. We already know exactly what is sung — the lyrics
came out of reconciliation and a human may have corrected them. Asking an ASR
model to guess the words again would throw that away and introduce errors we'd
then have to detect. WhisperX's *alignment-only* path takes the text as given
and answers a much narrower question: when is each word spoken?

That narrowness is why this is worth doing. LRCLIB (the first timing layer) has
excellent coverage of popular music and none at all of everything else; when it
misses, this fills the gap from the audio itself, and it produces WORD times
where LRCLIB only ever gives LINE times — which is what a karaoke highlight
needs.

## Align against the vocals stem when there is one

Alignment quality depends almost entirely on how audible the voice is. A dense
mix with loud guitars is genuinely hard; the isolated vocals stem from B4 is
close to a studio a cappella. So if stems exist for the song, use them, and
record which one was used — a caller comparing two runs deserves to know that
the difference was the input, not the model.

## The line mapping is the part that can silently rot

WhisperX returns a flat list of words. Getting them back onto lines is
positional: the Nth word of the flattened lyrics is the Nth word returned. That
holds only if we tokenize the text the same way going in and coming out, so the
tokenizer is one function used for both, and the mapping refuses to guess when
the counts disagree rather than producing a plausible-looking off-by-three
alignment that nobody notices until the highlight drifts a line behind.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Words are compared only for counting, so this is deliberately crude: split on
# whitespace, drop anything with no letters or digits (bare punctuation, the
# "(x2)" repeat markers that survive from chord sheets).
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


class AlignmentUnavailable(RuntimeError):
    """whisperx isn't installed here — the same posture as the MIR fallbacks
    and the stems engine: report it, don't crash the pipeline."""


def engine_available() -> bool:
    try:
        import whisperx  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def tokenize(text: str) -> list[str]:
    """The single definition of "a word", used for both the text we send and
    the words we get back. Two definitions would mean a mapping that is right
    on English and wrong on anything with an apostrophe."""
    return _WORD.findall(text or "")


@dataclass
class WordTime:
    lineIndex: int
    word: str
    start: Optional[float] = None
    end: Optional[float] = None

    def to_json(self) -> dict:
        return {"lineIndex": self.lineIndex, "word": self.word,
                "start": self.start, "end": self.end}


@dataclass
class LineTime:
    lineIndex: int
    start: Optional[float] = None
    end: Optional[float] = None
    # Fraction of this line's words the aligner actually placed. A line whose
    # words were mostly dropped got a time, but not one worth trusting, and the
    # review queue should be able to tell the difference.
    coverage: float = 0.0

    def to_json(self) -> dict:
        return {"lineIndex": self.lineIndex, "start": self.start,
                "end": self.end, "coverage": round(self.coverage, 3)}


@dataclass
class Alignment:
    lines: list[LineTime] = field(default_factory=list)
    words: list[WordTime] = field(default_factory=list)
    source: str = "mix"       # "vocals" when a stem was used
    engine: str = "whisperx"
    language: str = "en"

    @property
    def coverage(self) -> float:
        """Fraction of all words that got a time. The single number worth
        gating on: below ~0.5 the result is noise wearing a timestamp."""
        if not self.words:
            return 0.0
        return sum(1 for w in self.words if w.start is not None) / len(self.words)

    def to_json(self) -> dict:
        return {
            "engine": self.engine,
            "source": self.source,
            "language": self.language,
            "coverage": round(self.coverage, 3),
            "lines": [line.to_json() for line in self.lines],
            "words": [word.to_json() for word in self.words],
        }


def choose_audio(audio_path: str | Path, song_id: Optional[str] = None,
                 model: Optional[str] = None) -> tuple[Path, str]:
    """Prefer the isolated vocals stem; fall back to the full mix.

    Returns (path, label) so the label can travel into the result — the input
    is the biggest single lever on alignment quality, and a run that silently
    used a different one is a run you cannot compare.
    """
    if song_id:
        from ..audio import stems as stems_module

        vocals = stems_module.stem_path(
            song_id, "vocals", model or stems_module.DEFAULT_MODEL
        )
        if vocals is not None:
            return vocals, "vocals"
    return Path(audio_path), "mix"


def map_words_to_lines(lines: list[str], timed: list[dict]) -> tuple[list[LineTime], list[WordTime]]:
    """Fold a flat word list back onto the lines it came from.

    Positional, and strict about it: if the aligner returned a different number
    of words than we sent, every assignment after the discrepancy is wrong, and
    a silently shifted karaoke highlight is worse than an honest failure.
    """
    per_line = [tokenize(line) for line in lines]
    expected = sum(len(w) for w in per_line)
    if len(timed) != expected:
        raise ValueError(
            f"aligner returned {len(timed)} words for {expected} sent — refusing "
            "to guess the mapping"
        )

    words: list[WordTime] = []
    line_times: list[LineTime] = []
    cursor = 0
    for index, tokens in enumerate(per_line):
        slice_ = timed[cursor:cursor + len(tokens)]
        cursor += len(tokens)

        placed = [w for w in slice_ if w.get("start") is not None]
        for token, entry in zip(tokens, slice_):
            words.append(WordTime(
                lineIndex=index, word=token,
                start=entry.get("start"), end=entry.get("end"),
            ))
        line_times.append(LineTime(
            lineIndex=index,
            # A line's start is its first PLACED word, not its first word:
            # whisperx drops words it can't find, and taking the first
            # regardless would leave the line untimed for no reason.
            start=min((w["start"] for w in placed), default=None),
            end=max((w.get("end") or w["start"] for w in placed), default=None),
            coverage=(len(placed) / len(tokens)) if tokens else 0.0,
        ))
    return line_times, words


def _default_aligner(audio_path: Path, text: str, language: str,
                     duration: float) -> list[dict]:
    """Run WhisperX's alignment-only path. The only function that imports it."""
    try:
        import whisperx
    except Exception as e:  # noqa: BLE001
        raise AlignmentUnavailable(
            "whisperx is not installed here — install the [align] extra "
            "(pip install 'snoocle-server[align]') on the machine that runs "
            "alignment"
        ) from e

    device = "cpu"
    audio = whisperx.load_audio(str(audio_path))
    model, metadata = whisperx.load_align_model(language_code=language, device=device)
    # One segment spanning the track: we are not segmenting, we are aligning
    # text we already have against audio we already have.
    result = whisperx.align(
        [{"start": 0.0, "end": duration, "text": text}],
        model, metadata, audio, device, return_char_alignments=False,
    )
    out: list[dict] = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            out.append({
                "word": word.get("word", ""),
                "start": word.get("start"),
                "end": word.get("end"),
            })
    return out


def align_lyrics(
    audio_path: str | Path,
    lines: list[str],
    song_id: Optional[str] = None,
    language: str = "en",
    duration: Optional[float] = None,
    aligner: Optional[Callable[[Path, str, str, float], list[dict]]] = None,
) -> Alignment:
    """Time the given lines against the audio.

    `lines` is the lyric text exactly as stored — empty lines (instrumental
    slots) are kept in place so `lineIndex` stays a real index into the song,
    and they simply get no times.
    """
    sung = [line for line in lines if tokenize(line)]
    if not sung:
        raise ValueError("no lyric text to align")

    source_path, source_label = choose_audio(audio_path, song_id)
    if not Path(source_path).exists():
        raise FileNotFoundError(f"no such audio file: {source_path}")

    if duration is None:
        from ..audio import utils as audio_utils

        duration = audio_utils.probe(source_path).duration_seconds

    run = aligner or _default_aligner
    text = " ".join(" ".join(tokenize(line)) for line in lines)
    timed = run(Path(source_path), text, language, float(duration))

    line_times, words = map_words_to_lines(lines, timed)
    return Alignment(lines=line_times, words=words,
                     source=source_label, language=language)
