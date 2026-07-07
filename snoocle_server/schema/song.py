"""The Snoocle `Song` schema.

Field names are camelCase on purpose: this JSON is consumed as-is by the
Snoocle iOS app's SongStore with no transformation step. Structure follows
the app's conventions:

- metadata, displayPreferences (capo/tuning are DISPLAY-ONLY transforms),
- audio (youtubeVideoId + syncMap of lineIndex->seconds),
- sections referencing lines by index range, with optional MIR timestamps,
- lines: lineIndex + lyrics + chordPlacements keyed by charIndex,
- provenance: append-only history with confidence scores.

Invariants enforced here (not just documented):
- every chordPlacements.chord parses as a sounding harmony symbol
  (fretboard shapes / tab fingerings are rejected outright),
- lineIndex is contiguous from 0,
- charIndex within [0, len(lyrics)] (== len(lyrics) means "after the last
  character"; empty-lyric instrumental lines may use ordinal slots),
- placements per line strictly ascending by charIndex,
- sections reference valid, non-overlapping, ascending line ranges,
- syncMap times non-decreasing and lineIndexes valid.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..chords import ChordParseError, validate_stored_chord

SCHEMA_VERSION = 1

SectionKind = Literal[
    "intro",
    "verse",
    "prechorus",
    "chorus",
    "postchorus",
    "bridge",
    "solo",
    "instrumental",
    "interlude",
    "breakdown",
    "outro",
    "other",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChordPlacement(_Model):
    charIndex: int = Field(ge=0)
    # Sounding harmony, never a fretboard shape. Enforced by validator.
    chord: str

    @field_validator("chord")
    @classmethod
    def _chord_is_sounding_harmony(cls, v: str) -> str:
        try:
            validate_stored_chord(v)
        except ChordParseError as e:
            raise ValueError(str(e)) from e
        return v


class Line(_Model):
    lineIndex: int = Field(ge=0)
    lyrics: str
    chordPlacements: list[ChordPlacement] = Field(default_factory=list)

    @model_validator(mode="after")
    def _placements_valid(self) -> "Line":
        prev = -1
        for p in self.chordPlacements:
            if p.charIndex <= prev:
                raise ValueError(
                    f"line {self.lineIndex}: chordPlacements must be strictly "
                    f"ascending by charIndex (got {p.charIndex} after {prev})"
                )
            prev = p.charIndex
        if self.lyrics:  # empty-lyric lines use ordinal slots, no upper bound
            for p in self.chordPlacements:
                if p.charIndex > len(self.lyrics):
                    raise ValueError(
                        f"line {self.lineIndex}: charIndex {p.charIndex} beyond "
                        f"lyrics length {len(self.lyrics)}"
                    )
        return self


class Section(_Model):
    sectionIndex: int = Field(ge=0)
    name: str  # display name, e.g. "Verse 1"
    kind: SectionKind = "other"
    startLineIndex: int = Field(ge=0)
    endLineIndex: int = Field(ge=0)  # inclusive
    startTime: Optional[float] = Field(default=None, ge=0)  # seconds, MIR-derived
    endTime: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _range_valid(self) -> "Section":
        if self.endLineIndex < self.startLineIndex:
            raise ValueError(f"section {self.name!r}: endLineIndex < startLineIndex")
        if self.startTime is not None and self.endTime is not None and self.endTime < self.startTime:
            raise ValueError(f"section {self.name!r}: endTime < startTime")
        return self


class SyncPoint(_Model):
    lineIndex: int = Field(ge=0)
    time: float = Field(ge=0)  # seconds into the recording


class AudioInfo(_Model):
    youtubeVideoId: Optional[str] = None
    durationSeconds: Optional[float] = Field(default=None, ge=0)
    syncMap: list[SyncPoint] = Field(default_factory=list)

    @field_validator("youtubeVideoId")
    @classmethod
    def _plausible_video_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not re.fullmatch(r"[A-Za-z0-9_-]{11}", v):
            raise ValueError(f"implausible YouTube video id: {v!r}")
        return v


class DisplayPreferences(_Model):
    # Display-only transforms. NEVER baked into stored chord identities.
    capo: int = Field(default=0, ge=0, le=11)
    tuning: str = "standard"


class SongMetadata(_Model):
    title: str
    artist: str
    album: Optional[str] = None
    year: Optional[int] = None
    key: Optional[str] = None  # e.g. "D minor"
    bpm: Optional[float] = Field(default=None, gt=0)
    timeSignature: Optional[str] = None  # e.g. "4/4"


class ProvenanceEntry(_Model):
    timestamp: str  # ISO-8601 UTC
    actor: str  # e.g. "snoocle-server/0.1.0", "reconcile:anthropic/claude-..."
    action: str  # e.g. "discovered-sources", "mir-analysis", "reconciled"
    sources: list[str] = Field(default_factory=list)  # URLs / engine ids / model ids
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    notes: Optional[str] = None


class Song(_Model):
    schemaVersion: int = SCHEMA_VERSION
    id: str  # stable slug, e.g. "the-beatles--let-it-be"
    metadata: SongMetadata
    displayPreferences: DisplayPreferences = Field(default_factory=DisplayPreferences)
    audio: AudioInfo = Field(default_factory=AudioInfo)
    sections: list[Section] = Field(default_factory=list)
    lines: list[Line] = Field(default_factory=list)
    provenance: list[ProvenanceEntry] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_is_slug(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z0-9]+(?:[a-z0-9-]*[a-z0-9])?", v):
            raise ValueError(f"song id must be a lowercase slug, got {v!r}")
        return v

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> "Song":
        for i, line in enumerate(self.lines):
            if line.lineIndex != i:
                raise ValueError(
                    f"lines must be contiguous from 0: position {i} has lineIndex {line.lineIndex}"
                )
        n = len(self.lines)
        prev_end = -1
        prev_idx = -1
        for s in sorted(self.sections, key=lambda s: s.sectionIndex):
            if s.sectionIndex == prev_idx:
                raise ValueError(f"duplicate sectionIndex {s.sectionIndex}")
            prev_idx = s.sectionIndex
            if n and (s.startLineIndex >= n or s.endLineIndex >= n):
                raise ValueError(f"section {s.name!r} references lines beyond {n - 1}")
            if s.startLineIndex <= prev_end:
                raise ValueError(f"section {s.name!r} overlaps a previous section")
            prev_end = s.endLineIndex
        prev_t = 0.0
        for p in self.audio.syncMap:
            if n and p.lineIndex >= n:
                raise ValueError(f"syncMap references lineIndex {p.lineIndex} beyond {n - 1}")
            if p.time < prev_t:
                raise ValueError("syncMap times must be non-decreasing")
            prev_t = p.time
        return self


def song_json_schema() -> dict[str, Any]:
    return Song.model_json_schema()


def slugify_song_id(artist: str, title: str) -> str:
    def slug(s: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
        return s or "unknown"

    return f"{slug(artist)}--{slug(title)}"
