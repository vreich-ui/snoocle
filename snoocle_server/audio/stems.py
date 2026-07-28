"""Stem separation (demucs) and the practice mixes derived from it.

Why this exists: a play-along is far more useful when you can take yourself out
of the recording. Mute the vocals and sing it; mute the guitar and play it. Both
need the song split into sources first, which is what demucs does.

Three things are deliberately separate here:

- **Separation** needs demucs + torch — gigabytes of wheels and minutes of CPU
  per song. It is imported lazily and only inside `separate()`, so this module
  imports cleanly on a machine that has neither, and `engine_available()` is an
  honest answer rather than a guess from configuration.
- **Mixing** is pure ffmpeg (`audio/utils.py`'s philosophy: mechanical work
  stays mechanical). Summing four stems is arithmetic, not intelligence.
- **Reading the cache** needs nothing at all. That split matters: separation
  runs on a worker with the ML extras installed, while the API process only
  ever has to find files on disk and serve them.

## Where the files live

`{SNOOCLE_STEMS_DIR}/{songId}/{model}/{name}.wav` — content-addressed by
nothing, because the inputs are already identified by songId + model, and a
re-separation of the same song with the same model is the same audio. Existing
output is reused unless `force=True`.

WAV, not a compressed format: these get summed together into derived mixes, and
stacking lossy artefacts through a second encode is audibly worse for no real
saving at the point where they are still intermediates.

## The `backing_no_guitar` approximation

htdemucs' four-stem model separates vocals / drums / bass / other. There is no
guitar stem — guitar lives in `other`, alongside keys, strings, horns and
anything else that isn't a voice, a drum or a bass. So "no guitar" is really
"vocals + drums + bass", which also removes the piano and everything else in
that bucket.

That is a real limitation and it is named in the mix's own description rather
than hidden, because a user who mutes the guitar and loses the piano too should
be able to find out why. The upgrade path is `htdemucs_6s`, which does emit a
guitar stem; it is slower and a little less clean on the other sources, so it is
selectable per job rather than the default.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..config import settings
from . import utils as audio_utils

# The four sources htdemucs emits, in a fixed order so directory listings and
# API responses don't reshuffle between calls.
STEM_NAMES: tuple[str, ...] = ("vocals", "drums", "bass", "other")

# htdemucs_6s additionally splits guitar and piano out of `other`.
STEM_NAMES_6S: tuple[str, ...] = ("vocals", "drums", "bass", "guitar", "piano", "other")

MODELS: dict[str, tuple[str, ...]] = {
    "htdemucs": STEM_NAMES,
    "htdemucs_ft": STEM_NAMES,
    "htdemucs_6s": STEM_NAMES_6S,
}
DEFAULT_MODEL = "htdemucs"

# name -> (stems to sum, human description). Descriptions travel to the client
# because the no-guitar approximation is not self-evident from the name.
DERIVED_MIXES: dict[str, tuple[tuple[str, ...], str]] = {
    "backing_no_vocals": (
        ("drums", "bass", "other"),
        "Everything except the lead vocal — sing over it.",
    ),
    "backing_no_guitar": (
        ("vocals", "drums", "bass"),
        "Vocals, drums and bass only. With the 4-stem model this also removes "
        "keys, strings and anything else that isn't voice/drums/bass, because "
        "they share the 'other' stem with the guitar.",
    ),
}

# With htdemucs_6s the guitar is its own stem, so no-guitar stops being an
# approximation and everything else survives.
DERIVED_MIXES_6S: dict[str, tuple[tuple[str, ...], str]] = {
    "backing_no_vocals": (
        ("drums", "bass", "guitar", "piano", "other"),
        "Everything except the lead vocal — sing over it.",
    ),
    "backing_no_guitar": (
        ("vocals", "drums", "bass", "piano", "other"),
        "Everything except the guitar.",
    ),
}


class StemsUnavailable(RuntimeError):
    """demucs isn't installed here. Not an error in the pipeline sense — the
    caller is expected to report it and carry on, exactly like the MIR
    fallbacks."""


def engine_available() -> bool:
    """Proven by import, never assumed from configuration."""
    try:
        import demucs  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def known_model(model: str) -> bool:
    return model in MODELS


def mixes_for(model: str) -> dict[str, tuple[tuple[str, ...], str]]:
    return DERIVED_MIXES_6S if model == "htdemucs_6s" else DERIVED_MIXES


def stems_root() -> Path:
    return Path(settings.stems_dir)


def stems_dir(song_id: str, model: str = DEFAULT_MODEL) -> Path:
    """Where one song's stems live. `song_id` is a slug from our own store, but
    it still goes through a containment check — a path that escapes the root is
    a bug worth catching loudly rather than a file written somewhere odd."""
    root = stems_root().resolve()
    target = (root / song_id / model).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"unsafe stems path for song {song_id!r} / model {model!r}")
    return target


@dataclass
class StemSet:
    """What exists on disk for one (song, model) pair."""

    song_id: str
    model: str
    stems: dict[str, Path] = field(default_factory=dict)
    mixes: dict[str, Path] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Every stem the model promises is present.

        A partial set means separation died halfway; treating it as usable
        would produce a "backing track" silently missing its drums.
        """
        return bool(self.stems) and all(n in self.stems for n in MODELS.get(self.model, ()))

    def to_json(self) -> dict:
        descriptions = {n: d for n, (_, d) in mixes_for(self.model).items()}
        return {
            "songId": self.song_id,
            "model": self.model,
            "complete": self.complete,
            "stems": [
                {"name": n, "bytes": p.stat().st_size if p.exists() else 0}
                for n, p in sorted(self.stems.items())
            ],
            "mixes": [
                {
                    "name": n,
                    "bytes": p.stat().st_size if p.exists() else 0,
                    "description": descriptions.get(n, ""),
                }
                for n, p in sorted(self.mixes.items())
            ],
        }


def available(song_id: str, model: str = DEFAULT_MODEL) -> Optional[StemSet]:
    """Read the cache. Needs no demucs, no torch, no ffmpeg — which is what
    lets an API process with none of them installed still serve what a worker
    produced."""
    directory = stems_dir(song_id, model)
    if not directory.is_dir():
        return None
    found = StemSet(song_id=song_id, model=model)
    for path in sorted(directory.glob("*.wav")):
        name = path.stem
        if name in MODELS.get(model, ()):
            found.stems[name] = path
        elif name in mixes_for(model):
            found.mixes[name] = path
    return found if (found.stems or found.mixes) else None


def stem_path(song_id: str, name: str, model: str = DEFAULT_MODEL) -> Optional[Path]:
    """One named stem or derived mix, or None. `name` is checked against the
    known sets rather than joined onto a path, so a caller cannot walk out of
    the directory with it."""
    if name not in MODELS.get(model, ()) and name not in mixes_for(model):
        return None
    path = stems_dir(song_id, model) / f"{name}.wav"
    return path if path.exists() else None


# --- separation --------------------------------------------------------------


def _default_separator(audio_path: Path, model: str, out_dir: Path) -> dict[str, Path]:
    """Run demucs. This is the only function that touches it.

    Kept behind a seam so the tests can exercise every path around it —
    caching, mixing, partial results, cleanup — without a multi-gigabyte
    dependency and minutes of CPU per case.
    """
    try:
        from demucs.api import Separator, save_audio
    except Exception as e:  # noqa: BLE001
        raise StemsUnavailable(
            "demucs is not installed here — install the [stems] extra "
            "(pip install 'snoocle-server[stems]') on the machine that runs "
            "separation"
        ) from e

    separator = Separator(model=model)
    _origin, sources = separator.separate_audio_file(str(audio_path))

    written: dict[str, Path] = {}
    for name, tensor in sources.items():
        destination = out_dir / f"{name}.wav"
        save_audio(tensor, str(destination), samplerate=separator.samplerate)
        written[name] = destination
    return written


def render_mix(sources: list[Path], dst: Path) -> Path:
    """Sum stems back together with ffmpeg.

    `normalize=0` is the load-bearing argument. amix divides by the number of
    inputs by default, so the obvious version of this produces a backing track
    roughly a third the volume of the original and everyone reaches for the
    volume knob instead of reporting a bug.
    """
    if not sources:
        raise ValueError("no sources to mix")
    dst.parent.mkdir(parents=True, exist_ok=True)

    if len(sources) == 1:
        shutil.copyfile(sources[0], dst)
        return dst

    inputs: list[str] = []
    for path in sources:
        inputs += ["-i", str(path)]
    audio_utils._run([
        settings.ffmpeg_bin, "-y", "-v", "error",
        *inputs,
        "-filter_complex", f"amix=inputs={len(sources)}:normalize=0",
        "-c:a", "pcm_s16le",
        str(dst),
    ])
    return dst


def separate(
    audio_path: str | Path,
    song_id: str,
    model: str = DEFAULT_MODEL,
    force: bool = False,
    separator: Optional[Callable[[Path, str, Path], dict[str, Path]]] = None,
) -> StemSet:
    """Split a song into stems and render the practice mixes.

    Returns the cached set unmodified when one is already complete and `force`
    is false — separation is minutes of CPU, and the inputs (this song, this
    model) fully determine the output.
    """
    if not known_model(model):
        raise ValueError(f"unknown model {model!r} (known: {sorted(MODELS)})")

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"no such audio file: {audio_path}")

    cached = available(song_id, model)
    if cached and cached.complete and not force:
        return _ensure_mixes(cached)

    directory = stems_dir(song_id, model)
    directory.mkdir(parents=True, exist_ok=True)

    run = separator or _default_separator
    produced = run(audio_path, model, directory)

    result = StemSet(
        song_id=song_id,
        model=model,
        stems={n: Path(p) for n, p in produced.items() if n in MODELS[model]},
    )
    if not result.complete:
        # Half a separation is worse than none: it would render a backing mix
        # silently missing a source. Say so instead.
        missing = [n for n in MODELS[model] if n not in result.stems]
        raise StemsUnavailable(
            f"separation produced an incomplete set for {song_id} — missing: {missing}"
        )
    return _ensure_mixes(result)


def _ensure_mixes(stem_set: StemSet) -> StemSet:
    """Render any derived mix that isn't already on disk.

    A missing mix is cheap to rebuild (seconds of ffmpeg) and a stale one is
    impossible, since the stems it sums are immutable for a given song+model.
    """
    directory = stems_dir(stem_set.song_id, stem_set.model)
    for name, (parts, _description) in mixes_for(stem_set.model).items():
        destination = directory / f"{name}.wav"
        if destination.exists():
            stem_set.mixes[name] = destination
            continue
        sources = [stem_set.stems[p] for p in parts if p in stem_set.stems]
        if len(sources) != len(parts):
            continue  # can't build it honestly; leave it absent
        stem_set.mixes[name] = render_mix(sources, destination)
    return stem_set


def clear(song_id: str, model: Optional[str] = None) -> int:
    """Delete cached stems. Returns how many directories were removed."""
    root = stems_root()
    targets = [stems_dir(song_id, model)] if model else sorted(
        (root / song_id).glob("*")
    ) if (root / song_id).is_dir() else []
    removed = 0
    for target in targets:
        if Path(target).is_dir():
            shutil.rmtree(target)
            removed += 1
    parent = root / song_id
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
    return removed
