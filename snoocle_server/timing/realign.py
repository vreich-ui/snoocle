"""Mode B's deterministic core: re-point an existing document at a DIFFERENT
recording of the same song.

The library-economics argument: a song already reconciled once carries the
expensive part — the words and the chord sequence, agreed across sources and
reviewed by a human. A second recording of that song (a live take, an acoustic
session, a different studio master) needs none of that work repeated. It needs
new TIMES, and times come from audio, deterministically.

It is also the answer to the AUDIO verdict :mod:`quality.attribution` now
produces. The 1966 live Paint It Black has correct lyrics and chords from three
sources, a 0.48 chord match ratio, and timing that dies at 86.5s of 220.6s. No
better alignment onto that recording exists — the recording is what is
unreadable. Borrowing timing from a recording the analysis can actually hear is
the only fix, and it costs one MIR pass and no model tokens.

What this module does, all pure:

- :func:`derive_transposition` — which key the new recording is in, relative to
  the document. Delegated to ``reconcile.match.score_candidate``, the same
  12-transposition search the grader uses: a document in G against a recording
  in Bb scores 1.0 at +3. A best score at the coincidence floor means the two
  are not the same song, and the caller must not transpose on that.
- :func:`apply_transposition` — the chord SYMBOLS and ``metadata.key`` move;
  lyrics, charIndexes, line and section structure do not.
- :func:`clear_recording_timing` — every field that describes the OLD
  recording is emptied before the timing passes run. This is the load-bearing
  step: a placement that keeps its old time because nothing in the new
  timeline matched it is a time from a different performance, presented as if
  it were measured. An honest empty is what the collapse guard and the grader
  are built to read.
- :func:`compare_structure` — does the stored document still fit this
  recording, or has the arrangement changed (an extra chorus, a truncated
  outro)? This is the ONLY gate on invoking a model, so it is deliberately
  conservative: a live version of a song already in the library normally costs
  zero model tokens.

What this module does NOT do: shift stored times by a constant. That is
``timing/offset.py``'s job and it applies to a different problem — the SAME
recording re-uploaded with different intro padding. Two different performances
have no constant lag between them (different tempo, different fills, different
repeat counts), which is exactly why Mode B re-snaps against a fresh timeline
instead. See :func:`snoocle_server.realign.same_recording_check`, which uses
that module to catch an operator reaching for the expensive tool when the cheap
one applies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..chords import PITCH_CLASSES_FLAT, PITCH_CLASSES_SHARP, ChordParseError, transpose_chord
from ..discovery.models import CandidateSource
from ..mir.base import MirAnalysis
from ..reconcile.match import CandidateScore, score_candidate
from ..schema.song import Song

#: Below this chord-match score, the best-fitting transposition is not
#: distinguishable from coincidence (the shared root matcher scores ~0.38 on
#: unrelated sequences — see AttributionThresholds.mir_agreement) and must not
#: be applied. The document stays in its own key and the run says so.
MIN_TRANSPOSITION_SCORE = 0.5

#: How much longer or shorter the new recording may be before its arrangement
#: is considered structurally different from the one the document was timed
#: against. 10% of a 3:30 song is 21 seconds — comfortably more than a fade
#: difference, comfortably less than an extra chorus.
DURATION_TOLERANCE = 0.1

#: Structure-segment count difference that counts as a real divergence on its
#: own. Generous on purpose: the stored section list and the MIR structure
#: timeline come from different estimators (a human-named sheet vs SongFormer
#: or a librosa novelty fallback), so a small disagreement says nothing about
#: the arrangement. Expressed as a share of the stored section count, floored.
SEGMENT_COUNT_TOLERANCE = 0.5
MIN_SEGMENT_COUNT_DELTA = 2


@dataclass(frozen=True)
class TranspositionEstimate:
    """How far the new recording sits from the document's own key."""

    semitones: int
    score: CandidateScore  # the full 12-transposition search result
    trustworthy: bool
    reason: str
    #: The same answer derived from the two KEY strings, when both exist. Not
    #: authoritative — key estimation is a single noisy guess per recording,
    #: where the chord search compares whole sequences — but a disagreement
    #: between them is worth recording rather than hiding.
    key_semitones: Optional[int] = None

    @property
    def applies(self) -> bool:
        return self.trustworthy and self.semitones != 0

    def to_dict(self) -> dict:
        return {
            "semitones": self.semitones,
            "trustworthy": self.trustworthy,
            "reason": self.reason,
            "chordMatch": self.score.to_dict(),
            "keySemitones": self.key_semitones,
        }


@dataclass(frozen=True)
class StructureComparison:
    """Whether the stored document's shape still fits this recording."""

    explained: bool  # False -> the difference needs a model to resolve
    comparable: bool  # False -> nothing measurable; never a reason to spend
    reasons: list[str] = field(default_factory=list)
    stored_duration: Optional[float] = None
    new_duration: Optional[float] = None
    duration_delta: Optional[float] = None  # signed fraction of stored duration
    stored_sections: int = 0
    new_segments: int = 0

    def describe(self) -> str:
        if not self.comparable:
            return "not comparable: " + "; ".join(self.reasons)
        if self.explained:
            return "the stored document explains this recording's structure"
        return "structural difference: " + "; ".join(self.reasons)

    def to_dict(self) -> dict:
        return {
            "explained": self.explained,
            "comparable": self.comparable,
            "reasons": list(self.reasons),
            "storedDurationSeconds": self.stored_duration,
            "newDurationSeconds": self.new_duration,
            "durationDelta": (
                round(self.duration_delta, 3) if self.duration_delta is not None else None
            ),
            "storedSections": self.stored_sections,
            "newSegments": self.new_segments,
        }


def song_as_candidate(song: Song) -> CandidateSource:
    """The document, shaped as a candidate source.

    Not a trick: ``CandidateSource.lines`` IS ``schema.song.Line``, so a Song's
    lines are already exactly what ``score_candidate`` reads. This is what lets
    the transposition search be the one in ``reconcile/match.py`` rather than a
    second implementation of the same twelve comparisons.
    """
    return CandidateSource(sourceId=f"song:{song.id}", lines=list(song.lines))


def _key_delta(stored_key: Optional[str], new_key: Optional[str]) -> Optional[int]:
    """Semitones from `stored_key`'s tonic to `new_key`'s, or None."""
    from ..quality.theory import parse_key_name

    a, b = parse_key_name(stored_key), parse_key_name(new_key)
    if a is None or b is None:
        return None
    names = {n: i for i, n in enumerate(PITCH_CLASSES_SHARP)}
    names.update({n: i for i, n in enumerate(PITCH_CLASSES_FLAT)})
    if a[0] not in names or b[0] not in names:
        return None
    delta = (names[b[0]] - names[a[0]]) % 12
    return delta - 12 if delta > 6 else delta


def derive_transposition(song: Song, mir: MirAnalysis) -> TranspositionEstimate:
    """How many semitones to move `song`'s chords to reach `mir`'s recording."""
    score = score_candidate(song_as_candidate(song), mir)
    key_semitones = _key_delta(song.metadata.key, mir.key)

    if score.total == 0:
        return TranspositionEstimate(
            semitones=0, score=score, trustworthy=False,
            reason="the document has no chords, so there is nothing to transpose",
            key_semitones=key_semitones,
        )
    if score.score < MIN_TRANSPOSITION_SCORE:
        return TranspositionEstimate(
            semitones=0, score=score, trustworthy=False,
            reason=(
                f"the best of the 12 transpositions matches only "
                f"{score.matched}/{score.total} chord(s) ({score.score:.0%}, below "
                f"{MIN_TRANSPOSITION_SCORE:.0%}) — indistinguishable from coincidence, "
                f"so the document keeps its own key"
            ),
            key_semitones=key_semitones,
        )

    reason = (
        f"{score.matched}/{score.total} chord(s) match the new recording at "
        f"{score.transposition:+d} semitone(s) ({score.score:.0%})"
    )
    if key_semitones is not None and key_semitones != score.transposition:
        reason += (
            f"; the stored/estimated key strings suggest {key_semitones:+d} instead, "
            f"and the chord-sequence search wins (one key guess per recording vs "
            f"{score.total} chords compared)"
        )
    return TranspositionEstimate(
        semitones=score.transposition, score=score, trustworthy=True, reason=reason,
        key_semitones=key_semitones,
    )


#: Conventional tonic spelling per pitch class — the key signature a musician
#: would actually write. Mode matters: pitch class 6 is Gb major (6 flats) but
#: F# minor (3 sharps), and spelling it "Gb minor" would be technically
#: readable and obviously wrong to anyone reading the chart. Display only,
#: never an identity (see chords.py).
_MAJOR_TONICS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_MINOR_TONICS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]


def transpose_key_name(key_name: str, semitones: int) -> str:
    """"C major" +3 -> "Eb major"; "E minor" +2 -> "F# minor". Mode preserved.

    An unparseable key string is returned unchanged — a key nobody can read is
    not improved by being rewritten.
    """
    from ..quality.theory import parse_key_name

    parsed = parse_key_name(key_name)
    if parsed is None:
        return key_name
    tonic, mode = parsed
    names = {n: i for i, n in enumerate(PITCH_CLASSES_SHARP)}
    names.update({n: i for i, n in enumerate(PITCH_CLASSES_FLAT)})
    if tonic not in names:
        return key_name
    pc = (names[tonic] + semitones) % 12
    table = _MINOR_TONICS if mode in ("minor", "dorian", "phrygian", "aeolian", "locrian") else _MAJOR_TONICS
    return f"{table[pc]} {mode}"


def apply_transposition(song: Song, semitones: int) -> Song:
    """A NEW Song with every chord symbol moved by `semitones`.

    Lyrics, charIndexes, line indexes, section ranges and provenance are
    untouched — this changes what is played, never what is sung or where a
    chord sits in a line. A symbol that somehow fails to parse is left exactly
    as it is rather than dropped: the schema already guarantees it parses, and
    silently losing a chord would be worse than an untransposed one.
    """
    if semitones % 12 == 0:
        return song
    prefer_flats = semitones < 0
    new_lines = []
    for line in song.lines:
        placements = []
        for p in line.chordPlacements:
            try:
                placements.append(
                    p.model_copy(
                        update={
                            "chord": transpose_chord(p.chord, semitones, prefer_flats=prefer_flats)
                        }
                    )
                )
            except ChordParseError:
                placements.append(p)
        new_lines.append(line.model_copy(update={"chordPlacements": placements}))

    metadata_update: dict = {}
    if song.metadata.key:
        metadata_update["key"] = transpose_key_name(song.metadata.key, semitones)
    updated = song.model_copy(
        update={
            "lines": new_lines,
            "metadata": (
                song.metadata.model_copy(update=metadata_update)
                if metadata_update
                else song.metadata
            ),
        }
    )
    return Song.model_validate(updated.model_dump())


def clear_recording_timing(song: Song) -> Song:
    """A NEW Song with every field that describes the OLD recording emptied.

    Placement and line times, their confidences and beat refs, section times,
    the syncMap, the beat grid, bpm, ``analyzedVideoId`` and every
    ``videoOffsets`` entry — all of it measured against a recording this
    document is about to stop being timed by. ``videoOffsets`` especially: an
    offset is a correction relative to the OLD reference, so keeping one would
    silently mis-shift a playback that is now anchored somewhere else.

    Everything the document actually owns — lyrics, chord symbols, charIndexes,
    line and section structure, provenance — is preserved exactly.
    """
    new_lines = [
        line.model_copy(
            update={
                "timeSeconds": None,
                "confidence": None,
                "chordPlacements": [
                    p.model_copy(update={"timeSeconds": None, "confidence": None, "beat": None})
                    for p in line.chordPlacements
                ],
            }
        )
        for line in song.lines
    ]
    new_sections = [
        s.model_copy(update={"startTime": None, "endTime": None}) for s in song.sections
    ]
    updated = song.model_copy(
        update={
            "lines": new_lines,
            "sections": new_sections,
            "audio": song.audio.model_copy(
                update={
                    "syncMap": [],
                    "beats": [],
                    "analyzedVideoId": None,
                    "videoOffsets": {},
                    "durationSeconds": None,
                }
            ),
            "metadata": song.metadata.model_copy(update={"bpm": None}),
        }
    )
    return Song.model_validate(updated.model_dump())


def retime_sections(song: Song, duration: Optional[float] = None) -> tuple[Song, int]:
    """Fill section start/endTimes from the document's own re-measured line times.

    Mode B clears the old recording's section times along with everything else,
    and the passes that follow (``snap_chords``, LRC, the collapse guard) time
    LINES and PLACEMENTS — nothing puts a span back on a section. On Mode A
    that gap does not exist because the reconciler emits section times from the
    MIR structure timeline; on a deterministic Mode B run there is no
    reconciler, so a re-aligned document would arrive with every section
    untimed and be graded down for it.

    Nothing is invented: a section starts when its first timed line starts and
    ends where the next section starts (or, for the last one, at the track's
    end). A section with no timed lines at all is left untimed rather than
    given a guessed span. Times are set as a PAIR, so the schema's
    ``endTime >= startTime`` rule can never be tripped half-way.
    """
    line_times = {line.lineIndex: line.timeSeconds for line in song.lines}
    ordered = sorted(song.sections, key=lambda s: (s.startLineIndex, s.sectionIndex))

    starts: dict[int, Optional[float]] = {}
    for section in ordered:
        times = [
            t
            for i in range(section.startLineIndex, section.endLineIndex + 1)
            if (t := line_times.get(i)) is not None
        ]
        starts[section.sectionIndex] = min(times) if times else None

    updated: dict[int, tuple[float, float]] = {}
    for position, section in enumerate(ordered):
        start = starts[section.sectionIndex]
        if start is None:
            continue
        end: Optional[float] = None
        for later in ordered[position + 1 :]:
            later_start = starts[later.sectionIndex]
            if later_start is not None and later_start > start:
                end = later_start
                break
        if end is None:
            # The last timed section: run to the end of the track when we know
            # it, else to the last time this section itself carries.
            times = [
                t
                for i in range(section.startLineIndex, section.endLineIndex + 1)
                if (t := line_times.get(i)) is not None
            ]
            end = duration if duration and duration > start else max(times)
        updated[section.sectionIndex] = (start, max(start, end))

    new_sections = [
        s.model_copy(update={"startTime": updated[s.sectionIndex][0],
                             "endTime": updated[s.sectionIndex][1]})
        if s.sectionIndex in updated
        else s
        for s in song.sections
    ]
    out = Song.model_validate(song.model_copy(update={"sections": new_sections}).model_dump())
    return out, len(updated)


def stored_recording_duration(song: Song) -> Optional[float]:
    """The length of the recording this document was timed against.

    ``audio.durationSeconds`` when it is there; otherwise the span the
    document's own timing covers, which is a floor rather than the true
    duration — so it is only ever used to detect a NEW recording being
    materially LONGER, never shorter. Returns None when the document carries
    no timing at all.
    """
    if song.audio.durationSeconds:
        return song.audio.durationSeconds
    times = [line.timeSeconds for line in song.lines if line.timeSeconds is not None]
    section_ends = [s.endTime for s in song.sections if s.endTime is not None]
    candidates = times + section_ends
    return max(candidates) if candidates else None


def compare_structure(
    song: Song,
    mir: MirAnalysis,
    *,
    duration_tolerance: float = DURATION_TOLERANCE,
) -> StructureComparison:
    """Does the stored document still explain this recording's arrangement?

    The gate on spending model tokens, so the bar for "different" is a real
    difference, not an estimator disagreement:

    1. **Duration.** The most robust arrangement signal there is. An extra
       chorus makes a recording longer; a truncated outro makes it shorter.
       Compared against the length of the recording the document was timed
       against (:func:`stored_recording_duration`), which is why the document's
       own timing has to be read BEFORE it is cleared.
    2. **Segment count.** The MIR structure timeline vs the stored section
       list, with a wide tolerance — these are different estimators and a
       small disagreement is expected. Only a gross difference (half again as
       many, and at least two) counts.

    ``comparable=False`` means neither check had inputs (an untimed document,
    or a recording whose analysis produced no structure and no duration).
    That is reported, and it is NOT a difference: an unmeasurable comparison
    must never be the reason a model gets called.
    """
    stored_duration = stored_recording_duration(song)
    new_duration = mir.duration_seconds or None
    stored_sections = len(song.sections)
    new_segments = len(mir.sections)

    reasons: list[str] = []
    checks_run = 0
    delta: Optional[float] = None

    if stored_duration and new_duration:
        checks_run += 1
        delta = (new_duration - stored_duration) / stored_duration
        if abs(delta) > duration_tolerance:
            longer = delta > 0
            reasons.append(
                f"the new recording is {abs(delta):.0%} "
                f"{'longer' if longer else 'shorter'} than the {stored_duration:.0f}s "
                f"recording this document was timed against "
                f"({new_duration:.0f}s) — "
                + (
                    "repeats this document does not have"
                    if longer
                    else "a truncated or shortened arrangement"
                )
            )
    if stored_sections and new_segments:
        checks_run += 1
        allowed = max(MIN_SEGMENT_COUNT_DELTA, round(SEGMENT_COUNT_TOLERANCE * stored_sections))
        if abs(new_segments - stored_sections) > allowed:
            reasons.append(
                f"the new recording's structure timeline has {new_segments} segment(s) "
                f"against the document's {stored_sections} section(s) (more than the "
                f"{allowed} this comparison tolerates between two different estimators)"
            )

    if not checks_run:
        return StructureComparison(
            explained=True,
            comparable=False,
            reasons=[
                "no comparable structure signal (the document carries no duration or "
                "timing, or the analysis produced no structure timeline) — reported, "
                "not treated as a difference"
            ],
            stored_duration=stored_duration,
            new_duration=new_duration,
            stored_sections=stored_sections,
            new_segments=new_segments,
        )

    return StructureComparison(
        explained=not reasons,
        comparable=True,
        reasons=reasons or ["duration and section count both fit the stored document"],
        stored_duration=stored_duration,
        new_duration=new_duration,
        duration_delta=delta,
        stored_sections=stored_sections,
        new_segments=new_segments,
    )
