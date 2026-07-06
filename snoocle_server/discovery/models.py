"""Candidate text-source model.

Candidates stay SEPARATE until reconciliation — each keeps its own
confidence and provenance. Chords are normalized to sounding pitch at
ingestion (declared capo transposed away, recorded in `notes`).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ..schema.song import Line


class CandidateSource(BaseModel):
    sourceId: str  # e.g. "web-1"
    url: Optional[str] = None
    title: Optional[str] = None
    retrievedAt: Optional[str] = None  # ISO-8601
    declaredCapo: int = 0  # capo the sheet declared; chords BELOW are already sounding-pitch
    declaredKey: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0, le=1)
    sectionsHint: list[str] = Field(default_factory=list)  # e.g. ["Verse 1", "Chorus"]
    lines: list[Line] = Field(default_factory=list)
    notes: Optional[str] = None

    def chord_vocabulary(self) -> list[str]:
        seen: dict[str, None] = {}
        for line in self.lines:
            for p in line.chordPlacements:
                seen.setdefault(p.chord)
        return list(seen)
