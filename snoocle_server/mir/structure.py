"""Structural segmentation: SongFormer primary, librosa-novelty fallback.

Primary engine contract mirrors chordrec: point SNOOCLE_SONGFORMER_DIR at a
checkout containing `snoocle_runner.py <in.wav> <out.json>` writing
`[{"start": s, "end": e, "label": "verse"}, ...]` (SongFormer ships as a
Docker service in ChordMiniApp; the runner wraps one inference call).

Fallback: agglomerative segmentation over beat-synchronous chroma+MFCC,
then repetition-based labeling — the most-repeated cluster is called
"chorus", the runner-up "verse", one-off mid-song segments "bridge",
first/last segments "intro"/"outro" when they're short. Approximate by
design: these labels are reconciliation EVIDENCE (aligned to real
timestamps), not final section names; text sources usually carry the names.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from ..config import settings
from .base import StructureSegment

log = logging.getLogger(__name__)


def _runner_path() -> Path | None:
    d = settings.songformer_dir
    if d and Path(d).exists():
        runner = Path(d) / "snoocle_runner.py"
        if runner.exists():
            return runner
    return None


def segment_songformer(wav_path: str) -> list[StructureSegment]:
    runner = _runner_path()
    assert runner is not None
    out_path = Path(wav_path).with_suffix(".sections.json")
    proc = subprocess.run(
        [sys.executable, str(runner), wav_path, str(out_path)],
        capture_output=True,
        text=True,
        cwd=str(runner.parent),
        timeout=1800,
    )
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"songformer runner failed: {proc.stderr[-500:]}")
    data = json.loads(out_path.read_text())
    return [StructureSegment(**seg) for seg in data]


def segment_librosa(wav_path: str, beat_times: list[float]) -> list[StructureSegment]:
    import librosa
    import numpy as np

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = len(y) / sr
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)
    feats = np.vstack([librosa.util.normalize(chroma, axis=0), librosa.util.normalize(mfcc, axis=0)])

    if len(beat_times) >= 8:
        frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop)
        frames = np.unique(np.clip(frames, 0, feats.shape[1] - 1))
        sync = librosa.util.sync(feats, frames)
        grid_times = [0.0, *beat_times]
    else:
        sync = feats
        grid_times = list(librosa.frames_to_time(np.arange(feats.shape[1] + 1), sr=sr, hop_length=hop))

    # ~one boundary per 20s of audio, clamped to a sane section count
    k = int(np.clip(duration // 20, 3, 12))
    if sync.shape[1] <= k:
        return [StructureSegment(start=0.0, end=duration, label="other")]
    bound_idx = librosa.segment.agglomerative(sync, k)
    bounds = [grid_times[min(i, len(grid_times) - 1)] for i in bound_idx] + [duration]
    bounds[0] = 0.0

    # cluster segments by mean feature similarity -> letters
    seg_means = []
    for a, b in zip(bound_idx, list(bound_idx[1:]) + [sync.shape[1]]):
        b = max(int(b), int(a) + 1)
        seg_means.append(sync[:, int(a) : b].mean(axis=1))
    letters: list[int] = []
    reps: list[np.ndarray] = []
    for m in seg_means:
        n = m / (np.linalg.norm(m) + 1e-9)
        for li, r in enumerate(reps):
            if float(n @ r) > 0.88:
                letters.append(li)
                break
        else:
            reps.append(n)
            letters.append(len(reps) - 1)

    counts = {li: letters.count(li) for li in set(letters)}
    most_repeated = max(counts, key=lambda li: counts[li]) if counts else 0
    second = None
    for li, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        if li != most_repeated:
            second = li
            break

    segments: list[StructureSegment] = []
    n_seg = len(letters)
    for idx, (li, start, end) in enumerate(zip(letters, bounds[:-1], bounds[1:])):
        if idx == 0 and (end - start) < 25:
            label = "intro"
        elif idx == n_seg - 1 and (end - start) < 25:
            label = "outro"
        elif counts[li] >= 2 and li == most_repeated:
            label = "chorus"
        elif li == second and counts.get(second, 0) >= 2:
            label = "verse"
        elif counts[li] == 1 and 0 < idx < n_seg - 1:
            label = "bridge"
        else:
            label = "other"
        segments.append(StructureSegment(start=float(start), end=float(end), label=label))
    # merge consecutive same-label segments
    merged: list[StructureSegment] = []
    for seg in segments:
        if merged and merged[-1].label == seg.label:
            merged[-1] = merged[-1].model_copy(update={"end": seg.end})
        else:
            merged.append(seg)
    return merged


def segment_structure(wav_path: str, beat_times: list[float]) -> tuple[list[StructureSegment], str]:
    if _runner_path() is not None:
        try:
            return segment_songformer(wav_path), "songformer"
        except Exception as e:  # noqa: BLE001
            log.warning("songformer failed, falling back to librosa novelty: %s", e)
    return segment_librosa(wav_path, beat_times), "librosa-agglomerative-fallback"
