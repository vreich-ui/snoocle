"""Chord recognition over time: Chord-CNN-LSTM primary, chroma-template fallback.

Primary engine: the ISMIR2019 large-vocabulary Chord-CNN-LSTM (music-x-lab),
the same model ChordMiniApp ships. It is research code with git-lfs
checkpoints, so integration is via an external-runner contract rather than an
import: point SNOOCLE_CHORD_CNN_LSTM_DIR at a checkout containing
`snoocle_runner.py` which must accept `<in.wav> <out.lab>` and write MIREX
.lab lines (`start\tend\tlabel`, labels like `C:maj`, `A:min7`, `N`). The
ChordMiniApp python_backend shows exactly how to wire that runner around
`chord_recognition()`.

Fallback: beat-synchronous chroma template matching (maj/min/7/maj7/min7)
with an energy gate for no-chord. Far smaller vocabulary but audio-grounded,
which is what reconciliation needs from this input.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

from ..chords import PITCH_CLASSES_SHARP
from ..config import settings
from .base import ChordSegment

log = logging.getLogger(__name__)

_MIREX_QUALITY = {
    "maj": "",
    "min": "m",
    "maj7": "maj7",
    "min7": "m7",
    "7": "7",
    "maj6": "6",
    "min6": "m6",
    "dim": "dim",
    "dim7": "dim7",
    "hdim7": "m7b5",
    "aug": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "9": "9",
    "maj9": "maj9",
    "min9": "m9",
    "11": "11",
    "13": "13",
}

_NOTE_RE = re.compile(r"^[A-G][#b]?$")


def mirex_to_symbol(label: str) -> str:
    """'C:maj7' -> 'Cmaj7'; 'N'/'X' -> 'N'. Inversions by scale degree are
    dropped (degree->note needs key context the .lab doesn't carry)."""
    label = label.strip()
    if label in ("N", "X", ""):
        return "N"
    root, _, rest = label.partition(":")
    rest, _, bass = rest.partition("/")
    quality = _MIREX_QUALITY.get(rest, None)
    if quality is None:
        # unknown extended quality: keep the seventh-ness if hinted, else plain
        quality = "m" if rest.startswith("min") else ""
    symbol = root + quality
    if bass and _NOTE_RE.match(bass):
        symbol += "/" + bass
    return symbol


def _runner_path() -> Path | None:
    d = settings.chord_cnn_lstm_dir
    if d and Path(d).exists():
        runner = Path(d) / "snoocle_runner.py"
        if runner.exists():
            return runner
    return None


def chord_engine_id() -> str:
    """The chord engine that will ACTUALLY run — checks the runner exists, not
    just that the setting is set (a configured-but-empty dir silently falls
    back, and health reporting must not claim otherwise)."""
    return "chord-cnn-lstm" if _runner_path() is not None else "chroma-template-fallback"


def chord_model_status() -> dict:
    """Diagnosable health detail for the heavy chord model mount."""
    d = settings.chord_cnn_lstm_dir
    try:
        import torch  # noqa: F401

        torch_ok = True
    except Exception:  # noqa: BLE001
        torch_ok = False
    return {
        "dirConfigured": bool(d),
        "runnerPresent": _runner_path() is not None,
        "torchImportable": torch_ok,
    }


def recognize_chords_cnn_lstm(wav_path: str) -> list[ChordSegment]:
    runner = _runner_path()
    assert runner is not None
    # subprocess runs with cwd=runner.parent (the model resolves its data/ and
    # cache_data/ relative to cwd) — so both paths must be absolute or a
    # relative SNOOCLE_CHORD_CNN_LSTM_DIR would re-resolve against the new cwd.
    runner = runner.resolve()
    lab_path = Path(wav_path).resolve().with_suffix(".lab")
    wav_path = str(Path(wav_path).resolve())
    proc = subprocess.run(
        [sys.executable, str(runner), wav_path, str(lab_path)],
        capture_output=True,
        text=True,
        cwd=str(runner.parent),
        timeout=1800,
    )
    if proc.returncode != 0 or not lab_path.exists():
        raise RuntimeError(f"chord-cnn-lstm runner failed: {proc.stderr[-500:]}")
    segments = []
    for line in lab_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            segments.append(
                ChordSegment(start=float(parts[0]), end=float(parts[1]), chord=mirex_to_symbol(parts[2]))
            )
    return segments


def recognize_chords_chroma(wav_path: str, beat_times: list[float]) -> list[ChordSegment]:
    import librosa
    import numpy as np

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]

    # analysis frames: beat-synchronous when we have beats, else 0.5s grid
    duration = len(y) / sr
    if len(beat_times) >= 4:
        bounds = [0.0, *beat_times, duration]
    else:
        bounds = list(np.arange(0.0, duration, 0.5)) + [duration]
    frames = librosa.time_to_frames(bounds, sr=sr, hop_length=hop)
    frames = np.clip(frames, 0, chroma.shape[1] - 1)

    # chord templates
    names: list[str] = []
    templates: list[np.ndarray] = []

    def add(root: int, name_suffix: str, intervals: list[int]) -> None:
        t = np.zeros(12)
        for iv in intervals:
            t[(root + iv) % 12] = 1.0
        templates.append(t / np.linalg.norm(t))
        names.append(PITCH_CLASSES_SHARP[root] + name_suffix)

    for root in range(12):
        add(root, "", [0, 4, 7])
        add(root, "m", [0, 3, 7])
        add(root, "7", [0, 4, 7, 10])
        add(root, "maj7", [0, 4, 7, 11])
        add(root, "m7", [0, 3, 7, 10])
    T = np.stack(templates)  # (60, 12)

    energy_gate = float(np.percentile(rms, 20)) * 0.5
    labels: list[str] = []
    for a, b in zip(frames[:-1], frames[1:]):
        b = max(int(b), int(a) + 1)
        seg_chroma = chroma[:, int(a) : b].mean(axis=1)
        seg_rms = rms[int(a) : min(b, len(rms))].mean() if int(a) < len(rms) else 0.0
        norm = np.linalg.norm(seg_chroma)
        if norm < 1e-6 or seg_rms < energy_gate:
            labels.append("N")
            continue
        sims = T @ (seg_chroma / norm)
        labels.append(names[int(np.argmax(sims))])

    # mode-filter singletons: X Y X -> X X X
    for i in range(1, len(labels) - 1):
        if labels[i - 1] == labels[i + 1] != labels[i]:
            labels[i] = labels[i - 1]

    segments: list[ChordSegment] = []
    for label, start, end in zip(labels, bounds[:-1], bounds[1:]):
        if segments and segments[-1].chord == label:
            segments[-1] = segments[-1].model_copy(update={"end": float(end)})
        else:
            segments.append(ChordSegment(start=float(start), end=float(end), chord=label))
    return segments


def recognize_chords(wav_path: str, beat_times: list[float]) -> tuple[list[ChordSegment], str]:
    """Returns (segments, engine_id)."""
    if _runner_path() is not None:
        try:
            return recognize_chords_cnn_lstm(wav_path), "chord-cnn-lstm"
        except Exception as e:  # noqa: BLE001
            log.warning("chord-cnn-lstm failed, falling back to chroma templates: %s", e)
    return recognize_chords_chroma(wav_path, beat_times), "chroma-template-fallback"
