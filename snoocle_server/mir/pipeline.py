"""MIR pipeline: audio file -> beats + chords + structure + key, one call."""

from __future__ import annotations

import logging
import tempfile
from collections import defaultdict
from pathlib import Path

from ..audio.utils import probe, to_analysis_wav, trim
from ..chords import PITCH_CLASSES_SHARP, ChordParseError, parse_chord
from ..config import settings
from .base import Beat, ChordSegment, MirAnalysis
from .beats import track_beats
from .chordrec import recognize_chords
from .structure import segment_structure

log = logging.getLogger(__name__)


def estimate_key(chords: list[ChordSegment]) -> str | None:
    """Duration-weighted diatonic vote across the chord timeline."""
    weights: dict[int, float] = defaultdict(float)  # (pitch class) -> seconds, split maj/min
    minor_weights: dict[int, float] = defaultdict(float)
    for seg in chords:
        if seg.chord == "N":
            continue
        try:
            c = parse_chord(seg.chord)
        except ChordParseError:
            continue
        dur = max(seg.end - seg.start, 0.0)
        # each chord votes for keys it is diatonic to; tonic vote weighted more
        if c.quality.startswith("m") and not c.quality.startswith("maj"):
            minor_weights[c.root_pc] += dur * 1.5  # as minor tonic
            weights[(c.root_pc + 3) % 12] += dur  # as vi of relative major
            weights[(c.root_pc + 10) % 12] += dur * 0.5  # as ii
        else:
            weights[c.root_pc] += dur * 1.5  # as major tonic
            weights[(c.root_pc + 5) % 12] += dur  # as V of key a fourth up
            weights[(c.root_pc + 7) % 12] += dur * 0.75  # as IV
            minor_weights[(c.root_pc + 9) % 12] += dur  # as III of relative minor
    if not weights and not minor_weights:
        return None
    best_major = max(weights.items(), key=lambda kv: kv[1], default=(0, 0.0))
    best_minor = max(minor_weights.items(), key=lambda kv: kv[1], default=(0, 0.0))
    if best_minor[1] > best_major[1]:
        return f"{PITCH_CLASSES_SHARP[best_minor[0]]} minor"
    return f"{PITCH_CLASSES_SHARP[best_major[0]]} major"


def analyze_audio(audio_path: str | Path) -> MirAnalysis:
    """Run the full MIR stack over an audio file (any ffmpeg-readable format)."""
    audio_path = Path(audio_path)
    info = probe(audio_path)
    duration = info.duration_seconds

    with tempfile.TemporaryDirectory(prefix="snoocle-mir-") as td:
        src: Path = audio_path
        if settings.mir_max_analysis_seconds and duration > settings.mir_max_analysis_seconds:
            clipped = Path(td) / "clip.wav"
            trim(audio_path, clipped, 0.0, float(settings.mir_max_analysis_seconds))
            src = clipped
            duration = float(settings.mir_max_analysis_seconds)
        wav = Path(td) / "analysis.wav"
        to_analysis_wav(src, wav)

        beats, bpm, time_signature, beats_engine = track_beats(str(wav))
        beat_times = [t for t, _ in beats]
        chord_segments, chords_engine = recognize_chords(str(wav), beat_times)
        sections, structure_engine = segment_structure(str(wav), beat_times)

    return MirAnalysis(
        engines={
            "beats": beats_engine,
            "chords": chords_engine,
            "structure": structure_engine,
        },
        duration_seconds=duration,
        bpm=bpm,
        time_signature=time_signature,
        key=estimate_key(chord_segments),
        beats=[Beat(time=t, position=p) for t, p in beats],
        chords=chord_segments,
        sections=sections,
    )
