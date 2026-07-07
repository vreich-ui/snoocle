"""MIR analysis output model — the audio-grounded half of reconciliation.

Every engine slot (beats / chords / structure) has a primary heavy model
(madmom / Chord-CNN-LSTM / SongFormer, ChordMiniApp-style) and a pure-librosa
fallback so the pipeline always produces a timeline; `engines` records which
implementation actually ran, and that lands in the song's provenance.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Beat(BaseModel):
    time: float  # seconds
    position: int = 0  # beat within bar (1 = downbeat); 0 = unknown


class ChordSegment(BaseModel):
    start: float
    end: float
    chord: str  # normalized sounding harmony, or "N" for no-chord


class StructureSegment(BaseModel):
    start: float
    end: float
    label: str  # intro/verse/chorus/bridge/outro/... or cluster letter


class MirAnalysis(BaseModel):
    engines: dict[str, str] = Field(default_factory=dict)  # slot -> implementation id
    duration_seconds: float = 0.0
    bpm: float | None = None
    time_signature: str | None = None
    key: str | None = None
    beats: list[Beat] = Field(default_factory=list)
    chords: list[ChordSegment] = Field(default_factory=list)
    sections: list[StructureSegment] = Field(default_factory=list)

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
            "beatsSampled": [
                {"time": round(b.time, 2), "position": b.position} for b in beats[::stride]
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
