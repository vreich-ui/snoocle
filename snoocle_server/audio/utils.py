"""Deterministic audio utilities — ffmpeg only, no AI anywhere.

Local-first routing principle (same philosophy as pdf-tool): format
conversion, trimming, and loudness normalization are purely mechanical, so
they are handled by ffmpeg; AI is reserved for genuinely ambiguous
reconciliation work elsewhere in the pipeline.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..config import settings

SUPPORTED_FORMATS = {"mp3", "wav", "m4a", "flac", "ogg", "opus"}

# Container/codec pairs where ffmpeg's default pick is wrong or suboptimal.
_FORMAT_ARGS = {
    "m4a": ["-c:a", "aac", "-f", "ipod"],
    "mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
    "wav": ["-c:a", "pcm_s16le"],
    "flac": ["-c:a", "flac"],
    "ogg": ["-c:a", "libvorbis"],
    "opus": ["-c:a", "libopus"],
}


class AudioToolError(RuntimeError):
    pass


@dataclass
class AudioProbe:
    path: str
    format_name: str
    duration_seconds: float
    sample_rate: int
    channels: int
    codec: str


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        raise AudioToolError(f"{cmd[0]} failed ({proc.returncode}): " + " | ".join(tail))
    return proc


def probe(path: str | Path) -> AudioProbe:
    path = Path(path)
    if not path.exists():
        raise AudioToolError(f"no such file: {path}")
    proc = _run(
        [
            settings.ffprobe_bin,
            "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ]
    )
    info = json.loads(proc.stdout)
    audio_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise AudioToolError(f"no audio stream in {path}")
    st = audio_streams[0]
    fmt = info.get("format", {})
    return AudioProbe(
        path=str(path),
        format_name=fmt.get("format_name", ""),
        duration_seconds=float(fmt.get("duration") or st.get("duration") or 0.0),
        sample_rate=int(st.get("sample_rate") or 0),
        channels=int(st.get("channels") or 0),
        codec=st.get("codec_name", ""),
    )


def _out_args(dst: Path) -> list[str]:
    fmt = dst.suffix.lstrip(".").lower()
    if fmt not in SUPPORTED_FORMATS:
        raise AudioToolError(f"unsupported output format .{fmt} (supported: {sorted(SUPPORTED_FORMATS)})")
    return _FORMAT_ARGS.get(fmt, [])


def convert(src: str | Path, dst: str | Path) -> AudioProbe:
    """Convert between audio formats; output format inferred from dst suffix."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run([settings.ffmpeg_bin, "-y", "-v", "error", "-i", str(src), "-vn", *_out_args(dst), str(dst)])
    return probe(dst)


def trim(src: str | Path, dst: str | Path, start: float, end: float) -> AudioProbe:
    """Crop [start, end) seconds into dst (format from dst suffix).

    Re-encodes rather than stream-copying so cut points are exact instead of
    snapping to the nearest keyframe/frame boundary.
    """
    if end <= start or start < 0:
        raise AudioToolError(f"invalid trim range: start={start} end={end}")
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            settings.ffmpeg_bin, "-y", "-v", "error",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-i", str(src), "-vn", *_out_args(dst), str(dst),
        ]
    )
    return probe(dst)


def normalize(src: str | Path, dst: str | Path, target_lufs: float = -16.0) -> AudioProbe:
    """EBU R128 loudness normalization to target integrated LUFS."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            settings.ffmpeg_bin, "-y", "-v", "error",
            "-i", str(src), "-vn",
            "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
            *_out_args(dst), str(dst),
        ]
    )
    return probe(dst)


def to_analysis_wav(src: str | Path, dst: str | Path, sample_rate: int = 22050) -> AudioProbe:
    """Mono 16-bit WAV at a fixed sample rate — canonical input for MIR engines."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            settings.ffmpeg_bin, "-y", "-v", "error",
            "-i", str(src), "-vn",
            "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
            str(dst),
        ]
    )
    return probe(dst)
