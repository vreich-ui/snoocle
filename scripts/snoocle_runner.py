"""Snoocle external-runner adapter for Chord-CNN-LSTM (ISMIR2019, music-x-lab).

Copied into the model checkout by scripts/setup_chord_model.sh; invoked by
snoocle_server.mir.chordrec as `python snoocle_runner.py <in.wav> <out.lab>`
with cwd = the checkout (the model resolves its data/ and cache_data/ paths
relative to cwd). Writes MIREX .lab lines: `start\tend\tlabel`.

The upstream research code predates numpy 1.24 (uses the removed np.int /
np.float aliases) and CUDA-only checkpoint loading; this adapter shims both
so the pristine upstream checkout runs unmodified on modern CPU-only
environments.
"""

from __future__ import annotations

import sys


def _shim_numpy() -> None:
    import numpy as np

    for alias, typ in {"int": int, "float": float, "bool": bool, "object": object, "complex": complex}.items():
        if not hasattr(np, alias):
            setattr(np, alias, typ)


def _shim_torch_cpu() -> None:
    """Force checkpoint loads onto CPU when no GPU is present."""
    import torch

    if torch.cuda.is_available():
        return
    _orig_load = torch.load

    def _cpu_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        # research-era checkpoints contain pickled numpy objects; they are our
        # own vendored files, so full unpickling is intended
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _cpu_load


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: snoocle_runner.py <in.wav> <out.lab>", file=sys.stderr)
        raise SystemExit(2)
    _shim_numpy()
    _shim_torch_cpu()
    from chord_recognition import chord_recognition

    chord_recognition(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
