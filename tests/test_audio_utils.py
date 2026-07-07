import shutil

import pytest

from snoocle_server.audio import utils as au

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def tone_wav(tmp_path_factory):
    """A synthesized 8-second 440Hz tone — no external audio files needed."""
    path = tmp_path_factory.mktemp("audio") / "tone.wav"
    au._run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=8:sample_rate=44100",
            "-c:a", "pcm_s16le", str(path),
        ]
    )
    return path


def test_probe(tone_wav):
    info = au.probe(tone_wav)
    assert info.duration_seconds == pytest.approx(8.0, abs=0.1)
    assert info.sample_rate == 44100
    assert info.codec == "pcm_s16le"


def test_convert_wav_to_mp3_and_m4a(tone_wav, tmp_path):
    mp3 = au.convert(tone_wav, tmp_path / "tone.mp3")
    assert mp3.codec == "mp3"
    assert mp3.duration_seconds == pytest.approx(8.0, abs=0.2)
    m4a = au.convert(tone_wav, tmp_path / "tone.m4a")
    assert m4a.codec == "aac"
    # and back to wav
    wav = au.convert(tmp_path / "tone.mp3", tmp_path / "back.wav")
    assert wav.codec == "pcm_s16le"


def test_trim(tone_wav, tmp_path):
    out = au.trim(tone_wav, tmp_path / "cut.wav", start=2.0, end=5.0)
    assert out.duration_seconds == pytest.approx(3.0, abs=0.05)


def test_trim_rejects_bad_range(tone_wav, tmp_path):
    with pytest.raises(au.AudioToolError):
        au.trim(tone_wav, tmp_path / "bad.wav", start=5.0, end=2.0)


def test_normalize(tone_wav, tmp_path):
    out = au.normalize(tone_wav, tmp_path / "norm.wav", target_lufs=-16.0)
    assert out.duration_seconds == pytest.approx(8.0, abs=0.3)


def test_to_analysis_wav(tone_wav, tmp_path):
    out = au.to_analysis_wav(tone_wav, tmp_path / "mono.wav")
    assert out.channels == 1
    assert out.sample_rate == 22050


def test_unsupported_format(tone_wav, tmp_path):
    with pytest.raises(au.AudioToolError, match="unsupported"):
        au.convert(tone_wav, tmp_path / "tone.xyz")
