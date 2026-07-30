"""MIR pipeline: audio file -> beats + chords + structure + key, one call."""

from __future__ import annotations

import logging
import tempfile
from collections import defaultdict
from pathlib import Path

from ..audio.utils import probe, to_analysis_wav, trim
from ..chords import PITCH_CLASSES_SHARP, ChordParseError, parse_chord
from ..config import settings
from .base import AnalyzedWindow, Beat, ChordSegment, MirAnalysis
from .beat_fill import fill_beat_gaps
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


def detect_music_start(wav_path: str | Path, max_lead_seconds: float = 120.0) -> float:
    """Seconds into the file where sustained music begins (0.0 when it starts
    immediately). YouTube uploads often open with talking, applause, or slates;
    fast-accuracy windows should skip that lead-in. Heuristic: the first moment
    whose following ~2s of RMS energy reaches a fraction of the loud part of
    the opening minutes. Never returns more than ``max_lead_seconds``."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(wav_path), sr=None, mono=True, duration=max_lead_seconds + 10.0)
    hop = 2048
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    if rms.size == 0:
        return 0.0
    threshold = float(np.percentile(rms, 90)) * 0.3
    sustain = max(int(2.0 * sr / hop), 1)  # ~2 seconds of frames
    for i in range(rms.size - sustain):
        if float(rms[i : i + sustain].mean()) >= threshold:
            return min(float(librosa.frames_to_time(i, sr=sr, hop_length=hop)), max_lead_seconds)
    return 0.0


def fast_windows(
    duration: float,
    music_start: float,
    window_seconds: float,
    count: int,
) -> list[tuple[float, float]]:
    """Sampling windows for fast accuracy: a few spots spread across the
    musical span — just after the opening, mid-song, and toward (not at) the
    end, where the core sections live. Falls back to one window covering the
    whole musical span when the song is short. Overlapping windows merge."""
    music_start = min(max(music_start, 0.0), duration)
    span = duration - music_start
    if span <= 0:
        return [(0.0, duration)]
    if span <= window_seconds * count:
        return [(round(music_start, 2), round(duration, 2))]
    anchors = [0.08, 0.42, 0.72][:max(count, 1)]
    windows: list[tuple[float, float]] = []
    for a in anchors:
        start = music_start + span * a
        end = min(start + window_seconds, duration)
        windows.append((round(start, 2), round(end, 2)))
    merged: list[tuple[float, float]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _analyze_windows(
    audio_path: Path, duration: float, windows: list[tuple[float, float]], td: str
) -> MirAnalysis:
    """Run beats+chords per window, shifting results back into the original
    track's time coordinates so the timeline still aligns with the video.
    Structure segmentation is skipped — text sources carry section names, and
    fragments would only produce misleading labels."""
    all_beats: list[Beat] = []
    all_chords: list[ChordSegment] = []
    bpms: list[tuple[int, float]] = []  # (beat count, bpm) per window
    time_signature = None
    beats_engine = chords_engine = "none"
    for n, (start, end) in enumerate(windows):
        clip = Path(td) / f"window{n}.clip.wav"
        wav = Path(td) / f"window{n}.wav"
        trim(audio_path, clip, start, end)
        to_analysis_wav(clip, wav)
        beats, bpm, ts, beats_engine = track_beats(str(wav))
        # In clip coordinates: the window is its own audio file, and a window
        # that ends at the track's end fades out like the track does.
        window_beats, fill = fill_beat_gaps(
            [Beat(time=t, position=p) for t, p in beats], end - start, ts
        )
        if fill.inferred:
            log.info("window %.0f-%.0fs beat grid: %s", start, end, fill.describe())
        chords, chords_engine = recognize_chords(str(wav), [b.time for b in window_beats])
        all_beats.extend(b.model_copy(update={"time": b.time + start}) for b in window_beats)
        all_chords.extend(
            c.model_copy(update={"start": c.start + start, "end": c.end + start}) for c in chords
        )
        if bpm:
            bpms.append((fill.detected, bpm))
        time_signature = time_signature or ts
    best_bpm = max(bpms, default=(0, None))[1]
    return MirAnalysis(
        engines={
            "beats": beats_engine,
            "chords": chords_engine,
            "structure": "skipped (fast accuracy)",
            "sampling": "fast: " + ", ".join(f"{s:.0f}-{e:.0f}s" for s, e in windows),
        },
        duration_seconds=duration,
        bpm=best_bpm,
        time_signature=time_signature,
        key=estimate_key(all_chords),
        beats=all_beats,
        chords=all_chords,
        sections=[],
        analyzed_windows=[AnalyzedWindow(start=s, end=e) for s, e in windows],
    )


def analyze_window(audio_path: str | Path, start: float, end: float) -> MirAnalysis:
    """Windowed analysis with timestamps in the ORIGINAL track's coordinates."""
    duration = probe(audio_path).duration_seconds
    with tempfile.TemporaryDirectory(prefix="snoocle-mir-") as td:
        return _analyze_windows(Path(audio_path), duration, [(max(start, 0.0), min(end, duration))], td)


def analyze_audio(audio_path: str | Path, accuracy: str = "standard") -> MirAnalysis:
    """Run the MIR stack over an audio file (any ffmpeg-readable format).

    accuracy: "fast" analyzes a few short windows across the musical span
    (cheap, quick, timeline stays in original time coordinates); "standard"
    honors SNOOCLE_MIR_MAX_ANALYSIS_SECONDS; "thorough" always analyzes the
    full track.
    """
    audio_path = Path(audio_path)
    info = probe(audio_path)
    duration = info.duration_seconds

    with tempfile.TemporaryDirectory(prefix="snoocle-mir-") as td:
        if accuracy == "fast":
            lead_clip = Path(td) / "lead.clip.wav"
            lead_wav = Path(td) / "lead.wav"
            lead_end = min(duration, 130.0)
            music_start = 0.0
            if lead_end > 5.0:
                trim(audio_path, lead_clip, 0.0, lead_end)
                to_analysis_wav(lead_clip, lead_wav)
                music_start = detect_music_start(lead_wav)
            windows = fast_windows(
                duration, music_start,
                float(settings.mir_fast_window_seconds), settings.mir_fast_window_count,
            )
            return _analyze_windows(audio_path, duration, windows, td)

        src: Path = audio_path
        cap = settings.mir_max_analysis_seconds
        if accuracy != "thorough" and cap and duration > cap:
            clipped = Path(td) / "clip.wav"
            trim(audio_path, clipped, 0.0, float(cap))
            src = clipped
            duration = float(cap)
        wav = Path(td) / "analysis.wav"
        to_analysis_wav(src, wav)

        beats, bpm, time_signature, beats_engine = track_beats(str(wav))
        # Onset-driven trackers lose lock on fade-outs and near-silent
        # intros, leaving the ends of the track with no grid at all — and,
        # because chord recognition is beat-synchronous, no chord timeline
        # either. Continue the established tempo across those spans BEFORE
        # the downstream engines read the beat times. See mir/beat_fill.py.
        beat_list, fill = fill_beat_gaps(
            [Beat(time=t, position=p) for t, p in beats], duration, time_signature
        )
        if fill.inferred:
            log.info("beat grid extended by tempo continuation: %s", fill.describe())
        beat_times = [b.time for b in beat_list]
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
        beats=beat_list,
        chords=chord_segments,
        sections=sections,
        # duration was already reduced to the cap when the clip was trimmed, so
        # this honestly reports the analyzed span either way
        analyzed_windows=[AnalyzedWindow(start=0.0, end=duration)],
    )
