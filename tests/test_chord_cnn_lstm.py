"""Chord-CNN-LSTM integration — runs ONLY when the real model is present.

The model checkout (scripts/setup_chord_model.sh) is found via
SNOOCLE_CHORD_CNN_LSTM_DIR or the default models/chord-cnn-lstm; when absent
(or torch isn't installed) the whole module skips, so the base suite stays
fast and dependency-light. Ground truth is the same synthesized C-G-Am-F
progression test_mir.py uses — the real model must name all four triads
exactly, which the chroma fallback is not held to.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from snoocle_server.config import settings
from snoocle_server.mir.chordrec import recognize_chords


def _model_dir() -> Path | None:
    for cand in (os.environ.get("SNOOCLE_CHORD_CNN_LSTM_DIR"), "models/chord-cnn-lstm"):
        if cand and (Path(cand) / "snoocle_runner.py").exists():
            return Path(cand)
    return None


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = [
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed"),
    pytest.mark.skipif(_model_dir() is None, reason="chord-cnn-lstm checkout not present"),
    pytest.mark.skipif(not _has_torch(), reason="torch not installed"),
]

_CHORDS = {
    "C": (261.63, 329.63, 392.00),
    "G": (196.00, 246.94, 392.00),
    "Am": (220.00, 261.63, 329.63),
    "F": (174.61, 220.00, 349.23),
}


@pytest.fixture(scope="module")
def progression_wav(tmp_path_factory):
    """One cycle of | C | G | Am | F | at 2s per chord = 8s."""
    d = tmp_path_factory.mktemp("cnnlstm")
    parts = []
    for i, name in enumerate(["C", "G", "Am", "F"]):
        f1, f2, f3 = _CHORDS[name]
        p = d / f"part{i}.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", f"sine=frequency={f1}:duration=2",
                "-f", "lavfi", "-i", f"sine=frequency={f2}:duration=2",
                "-f", "lavfi", "-i", f"sine=frequency={f3}:duration=2",
                "-filter_complex", "amix=inputs=3:normalize=1",
                "-c:a", "pcm_s16le", "-ar", "22050", str(p),
            ],
            check=True, capture_output=True,
        )
        parts.append(p)
    concat = d / "list.txt"
    concat.write_text("".join(f"file '{p}'\n" for p in parts))
    out = d / "progression.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
         "-c:a", "pcm_s16le", str(out)],
        check=True, capture_output=True,
    )
    return out


def test_real_model_names_all_four_triads_exactly(progression_wav, monkeypatch):
    monkeypatch.setattr(settings, "chord_cnn_lstm_dir", _model_dir())
    segments, engine = recognize_chords(str(progression_wav), beat_times=[])
    assert engine == "chord-cnn-lstm"
    # exact triads, in order, ignoring no-chord padding
    played = [s.chord for s in segments if s.chord != "N"]
    assert played == ["C", "G", "Am", "F"], f"got {played}"
    # boundaries land on the 2s grid within a beat's tolerance
    starts = [s.start for s in segments if s.chord != "N"]
    for got, expected in zip(starts, [0.0, 2.0, 4.0, 6.0]):
        assert abs(got - expected) < 0.35, f"boundary {got} vs {expected}"
