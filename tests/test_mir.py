"""MIR tests against a synthesized chord progression (no network, no real song).

We render C - G - Am - F as sine-triads at 120bpm with a click on beat 1, so
the expected chords, tempo, and duration are known ground truth.
"""

import shutil
import subprocess

import pytest

from snoocle_server.mir import analyze_audio
from snoocle_server.mir.chordrec import mirex_to_symbol

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

# triads as frequency triples (Hz)
_CHORDS = {
    "C": (261.63, 329.63, 392.00),
    "G": (196.00, 246.94, 392.00),
    "Am": (220.00, 261.63, 329.63),
    "F": (174.61, 220.00, 349.23),
}


@pytest.fixture(scope="module")
def progression_wav(tmp_path_factory):
    """16 bars of | C | G | Am | F | x4 at 120bpm (2s per bar) = 32s."""
    d = tmp_path_factory.mktemp("mir")
    parts = []
    for i, name in enumerate(["C", "G", "Am", "F"] * 4):
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
    concat_list = d / "list.txt"
    concat_list.write_text("".join(f"file '{p}'\n" for p in parts))
    out = d / "progression.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat_list),
         "-c:a", "pcm_s16le", str(out)],
        check=True, capture_output=True,
    )
    return out


def test_mirex_label_conversion():
    assert mirex_to_symbol("C:maj") == "C"
    assert mirex_to_symbol("A:min") == "Am"
    assert mirex_to_symbol("G:7") == "G7"
    assert mirex_to_symbol("B:min7") == "Bm7"
    assert mirex_to_symbol("N") == "N"
    assert mirex_to_symbol("D:maj/A") == "D/A"


def test_analyze_synthesized_progression(progression_wav):
    analysis = analyze_audio(progression_wav)

    assert analysis.duration_seconds == pytest.approx(32.0, abs=0.5)
    assert analysis.engines["chords"]  # some engine ran
    assert analysis.beats, "no beats detected"

    # chord timeline should be dominated by the four real chords
    named = [c.chord for c in analysis.chords if c.chord != "N"]
    assert named, "no chords recognized"
    hits = sum(1 for c in named if c in {"C", "G", "Am", "F", "Cmaj7", "Am7"})
    assert hits / len(named) >= 0.5, f"chord timeline implausible: {named}"

    # key of C major / A minor expected
    assert analysis.key in ("C major", "A minor")

    # structure: at least one section, covering the track
    assert analysis.sections
    assert analysis.sections[0].start == pytest.approx(0.0, abs=0.1)
    assert analysis.sections[-1].end == pytest.approx(32.0, abs=1.0)

    payload = analysis.to_prompt_payload()
    assert payload["chordTimeline"]
    assert payload["beatCount"] == len(analysis.beats)


def test_analyze_fast_accuracy_samples_windows(progression_wav):
    """Fast accuracy on a short song: one window over the musical span,
    structure skipped, timestamps still in the original track's coordinates."""
    analysis = analyze_audio(progression_wav, accuracy="fast")

    assert analysis.duration_seconds == pytest.approx(32.0, abs=0.5)
    assert analysis.engines["structure"].startswith("skipped")
    assert analysis.engines["sampling"].startswith("fast:")
    assert analysis.sections == []
    assert analysis.beats and analysis.chords
    # absolute time coordinates: the timeline must not exceed track length
    assert all(0.0 <= c.start < c.end <= 32.5 for c in analysis.chords)
    named = [c.chord for c in analysis.chords if c.chord != "N"]
    hits = sum(1 for c in named if c in {"C", "G", "Am", "F", "Cmaj7", "Am7"})
    assert named and hits / len(named) >= 0.5
    assert analysis.key in ("C major", "A minor")
