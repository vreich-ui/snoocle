"""MIR analysis output model — the audio-grounded half of reconciliation.

Every engine slot (beats / chords / structure) has a primary heavy model
(madmom / Chord-CNN-LSTM / SongFormer, ChordMiniApp-style) and a pure-librosa
fallback so the pipeline always produces a timeline; `engines` records which
implementation actually ran, and that lands in the song's provenance.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

_TIME_SIGNATURE_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
DEFAULT_BEATS_PER_MEASURE = 4


def beats_per_measure(time_signature: str | None, default: int = DEFAULT_BEATS_PER_MEASURE) -> int:
    """Numerator of a "n/d" time signature, or `default` when absent/unparseable."""
    if time_signature:
        m = _TIME_SIGNATURE_RE.match(time_signature)
        if m:
            numerator = int(m.group(1))
            if numerator > 0:
                return numerator
    return default


class Beat(BaseModel):
    time: float  # seconds
    position: int = 0  # beat within bar (1 = downbeat); 0 = unknown
    # False when this beat was never heard in the audio: the tracker lost lock
    # (a fade-out, a near-silent intro) and the grid was continued at the
    # established tempo instead — see mir.beats.extend_beat_grid. An inferred
    # beat must never be presented as a measurement; everything downstream
    # (the schema's BeatMark, snap.py, provenance) carries the flag through.
    detected: bool = True


class ChordSegment(BaseModel):
    start: float
    end: float
    chord: str  # normalized sounding harmony, or "N" for no-chord


class StructureSegment(BaseModel):
    start: float
    end: float
    label: str  # intro/verse/chorus/bridge/outro/... or cluster letter


class AnalyzedWindow(BaseModel):
    """A span of the ORIGINAL track (seconds) that analysis actually covered."""

    start: float
    end: float


class MirAnalysis(BaseModel):
    engines: dict[str, str] = Field(default_factory=dict)  # slot -> implementation id
    duration_seconds: float = 0.0
    bpm: float | None = None
    time_signature: str | None = None
    key: str | None = None
    beats: list[Beat] = Field(default_factory=list)
    chords: list[ChordSegment] = Field(default_factory=list)
    sections: list[StructureSegment] = Field(default_factory=list)
    # Which parts of the track the analysis covered — the whole track for
    # standard/thorough, the sampled windows for fast accuracy. Empty means
    # unknown/legacy (treat as full-track).
    analyzed_windows: list[AnalyzedWindow] = Field(default_factory=list)

    @property
    def detected_beat_count(self) -> int:
        """Beats actually heard in the audio (as opposed to tempo-continued)."""
        return sum(1 for b in self.beats if b.detected)

    @property
    def inferred_beat_count(self) -> int:
        return len(self.beats) - self.detected_beat_count

    @property
    def analyzed_partially(self) -> bool:
        """True when `analyzed_windows` covers only PART of the track —
        fast accuracy's sampled windows (a few short spans, not the whole
        song), rather than a single continuous pass over it. Empty
        `analyzed_windows` predates window tracking and was always
        full-track, so it reads as fully covered (see the field's own
        docstring)."""
        if not self.analyzed_windows:
            return False
        if len(self.analyzed_windows) != 1:
            return True
        w = self.analyzed_windows[0]
        covers_all = w.start <= 1e-6 and w.end >= self.duration_seconds - 1e-6
        return not covers_all

    def to_prompt_payload(self, max_beats: int = 64) -> dict:
        """Compact JSON for the reconciliation prompt: full chord/section
        timeline, beats summarized (they matter as tempo/downbeat evidence,
        not individually)."""
        beats = self.beats
        stride = max(1, len(beats) // max_beats)
        return {
            "source": "mir-audio-analysis",
            "engines": self.engines,
            "durationSeconds": round(self.duration_seconds, 2),
            "bpm": round(self.bpm, 1) if self.bpm else None,
            "timeSignature": self.time_signature,
            "estimatedKey": self.key,
            "beatCount": len(beats),
            "beatsDetected": self.detected_beat_count,
            "beatsInferred": self.inferred_beat_count,
            "beatsSampled": [
                {"time": round(b.time, 2), "position": b.position, "detected": b.detected}
                for b in beats[::stride]
            ],
            "chordTimeline": [
                {"start": round(c.start, 2), "end": round(c.end, 2), "chord": c.chord}
                for c in self.chords
            ],
            "structure": [
                {"start": round(s.start, 2), "end": round(s.end, 2), "label": s.label}
                for s in self.sections
            ],
        }

    def to_run_payload(self, max_beats: int = 500, max_chords: int = 3000) -> dict:
        """Bounded, storage-safe snapshot for the run trace (GUI timeline).

        Unlike :meth:`to_prompt_payload` this keeps the FULL chord timeline
        (the caps are pathological guards, not working limits — a 5-minute song
        is a few hundred segments) plus the analyzed windows, so the GUI can
        show exactly what the audio said and which spans were examined.
        Worst-case size stays well under Firestore's 1 MB document limit.
        """
        chords = self.chords
        beats = self.beats
        truncated = False
        if len(chords) > max_chords:
            chords = chords[:: -(-len(chords) // max_chords)]  # ceil stride -> <= cap
            truncated = True
        if len(beats) > max_beats:
            beats = beats[:: -(-len(beats) // max_beats)]
            truncated = True
        return {
            "engines": self.engines,
            "durationSeconds": round(self.duration_seconds, 2),
            "bpm": round(self.bpm, 1) if self.bpm else None,
            "timeSignature": self.time_signature,
            "estimatedKey": self.key,
            "beatCount": len(self.beats),
            "beatsDetected": self.detected_beat_count,
            "beatsInferred": self.inferred_beat_count,
            "beatsSampled": [
                {"time": round(b.time, 2), "position": b.position, "detected": b.detected}
                for b in beats
            ],
            "chordTimeline": [
                {"start": round(c.start, 2), "end": round(c.end, 2), "chord": c.chord}
                for c in chords
            ],
            "structure": [
                {"start": round(s.start, 2), "end": round(s.end, 2), "label": s.label}
                for s in self.sections
            ],
            "analyzedWindows": [
                {"start": round(w.start, 2), "end": round(w.end, 2)}
                for w in self.analyzed_windows
            ],
            "truncated": truncated,
        }
