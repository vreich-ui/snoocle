"""Reconciliation prompt construction.

The baseline input is STRUCTURED DATA ONLY — every candidate text source
plus the MIR timeline as JSON — so the exact same request works across
Claude, GPT, and Gemini regardless of their audio-input support. An audio
snippet is an optional, provider-conditional attachment handled in
providers.py, never required for the baseline.

The chord-normalization rule is baked in here AND enforced post-hoc by the
Song schema validator (engine.py repair loop) — prompt-level instruction
alone is hope, not a guarantee.
"""

from __future__ import annotations

import json

from ..discovery.models import CandidateSource
from ..mir.base import MirAnalysis
from ..scope import AnalysisScope

SYSTEM_PROMPT = """You are a music transcription reconciliation engine for the Snoocle song-practice app.

You receive:
1. Multiple candidate chord/lyric sheets discovered on the web (parsed into lines with chord placements by character index). Sources vary in quality and may disagree.
2. An independent music-information-retrieval (MIR) analysis of the actual studio recording: beat grid, bpm, chord timeline with timestamps, structural sections, estimated key.

Your job: produce the single best reconciled song document as JSON conforming EXACTLY to the schema provided in the user message.

Reconciliation principles:
- USE ALL CANDIDATE SOURCES. Agreement between independent sources is strong evidence. Where sources disagree, prefer the reading consistent with the MIR chord timeline and key.
- The MIR analysis is audio ground truth for TIMING (bpm, section boundaries, syncMap times) and a strong signal for HARMONY, but its chord vocabulary may be coarser than the sheets (e.g. it may report C where the true chord is Cmaj7). Prefer a text source's richer chord quality when the root and family agree with the audio.
- Lyrics come from the text sources; pick the most complete, correctly-ordered lyric set.
- Sections: name and order them from the text sources' hints plus MIR structure; attach startTime/endTime from the MIR structure timeline where alignment is clear.
- audio.syncMap: map line indexes to seconds using the MIR section boundaries and beat grid; include an entry at least for the first line of each section. Times must be non-decreasing.

CHORD NORMALIZATION RULE (non-negotiable):
Every chordPlacements.chord value MUST be the actual sounding harmony — never a fretboard shape, never a capo'd shape name, never tablature or fingering. Candidate sources have already been transposed to sounding pitch at ingestion (their declaredCapo is metadata about the original sheet, not an instruction to transpose again). displayPreferences.capo is a display-only preference — set it to 0; never bake a capo into chord identities. Valid chord symbols look like: C, F#m, Bbmaj7, Dm7/G, Esus4, A7, Gdim7, Cadd9. Invalid: x02210, 3-2-0-0-0-3, "Am shape", N.C. (omit no-chord positions instead).

Output rules:
- Output ONLY the JSON document. No markdown fences, no commentary.
- lines: lineIndex contiguous from 0; chordPlacements strictly ascending by charIndex, charIndex within [0, len(lyrics)] (empty-lyric instrumental lines may use ordinal slots 0,1,2,...).
- sections reference lines by inclusive [startLineIndex, endLineIndex]; sections must not overlap and must be in ascending line order.
- Leave "provenance" as an empty array — the server appends provenance entries itself.
- Do not invent lyrics or chords absent from all evidence."""


#: The `listen=off, reconcile=off` contract, stated plainly. This run is a
#: targeted EDIT of an existing document, not a transcription task, and the
#: prompt has to say so — the model is otherwise being handed a page with two
#: empty evidence sections and every incentive to fill them from memory.
_NOTES_ONLY_BANNER = (
    "## Scope of this run: apply the user's notes only\n"
    "No new sources were gathered and the audio was not re-analyzed — both"
    " were deliberately switched off for this run. Correct the prior song"
    " below using the user's notes, and change NOTHING the notes do not ask"
    " about. Do not re-derive lyrics, chords, sections, or timings from"
    " memory, and do not invent a song: if the notes do not mention"
    " something, the prior song's value for it is the right answer."
)


def build_user_prompt(
    title: str,
    artist: str,
    candidates: list[CandidateSource],
    mir: MirAnalysis | None,
    song_schema: dict,
    song_id: str,
    youtube_video_id: str | None,
    guidance: str | None = None,
    prior_song: dict | None = None,
    time_align: bool = False,
    scope: AnalysisScope | None = None,
) -> str:
    parts: list[str] = []
    parts.append(
        f"Reconcile the song {title!r} by {artist!r} into a single schema-compliant JSON document.\n"
        f"Use song id {song_id!r}"
        + (f" and audio.youtubeVideoId {youtube_video_id!r}." if youtube_video_id else ".")
    )
    # The scope banner goes FIRST, before the schema and the (possibly empty)
    # evidence sections. A model that reads "Candidate text sources (0 found —
    # use ALL of them)" and no MIR with no explanation concludes the evidence
    # failed to load and helpfully reconstructs the song from memory. Saying up
    # front that gathering was deliberately switched off is what turns an
    # apparent failure into an instruction.
    if scope is not None and scope.notes_only:
        parts.append(_NOTES_ONLY_BANNER)
    elif scope is not None and not scope.reconcile:
        parts.append(
            "## Scope of this run: no new sources\n"
            "Web search for chord/lyric sheets was deliberately switched off"
            " for this run. Work from the audio analysis, the prior song, and"
            " the user's notes below. The absence of candidate sources is"
            " intentional — do not treat it as missing evidence to fill in"
            " from memory."
        )
    parts.append("## Song JSON schema (output must validate against this)\n" + json.dumps(song_schema, indent=1))

    if candidates:
        parts.append(f"## Candidate text sources ({len(candidates)} found — use ALL of them)")
        for cand in candidates:
            payload = cand.model_dump(exclude_none=True)
            parts.append(f"### {cand.sourceId}\n" + json.dumps(payload, indent=1))
    else:
        parts.append(
            "## Candidate text sources\n"
            "NONE were gathered for this run. Do not invent sources and do not"
            " recall lyrics or chords from memory to stand in for them."
        )

    if mir is not None:
        parts.append(
            "## MIR audio analysis (independent, audio-grounded)\n"
            + json.dumps(mir.to_prompt_payload(), indent=1)
        )
    elif scope is not None and not scope.listen:
        parts.append(
            "## MIR audio analysis\n"
            "NOT re-run for this run — listening was deliberately switched off."
            " The prior song below already carries the timings from its last"
            " audio analysis: KEEP them as they are rather than recomputing or"
            " dropping them."
        )
    else:
        parts.append("## MIR audio analysis\nUNAVAILABLE for this run — reconcile from text sources alone; omit timestamps you cannot support.")

    if prior_song is not None:
        header = (
            # In notes-only scope the prior song is not "evidence" among
            # others — it IS the document, and the answer is a copy of it with
            # the notes applied. Saying "strong evidence" there invites a
            # rewrite, which is precisely what the user asked not to happen.
            "## The song to correct (this is your starting document)\n"
            "Return THIS document with the user's notes below applied and"
            " nothing else changed. Copy every field you are not explicitly"
            " asked to change, verbatim — lyrics, chord placements, section"
            " boundaries, and all existing times.\n"
            if scope is not None and scope.notes_only
            else "## Prior result the user corrected\n"
            "A previous reconciliation was reviewed and edited by a human. Treat"
            " their version as strong evidence — preserve their lyrics, chord"
            " placements, and section boundaries unless the audio flatly"
            " contradicts them.\n"
        )
        parts.append(header + json.dumps(prior_song, indent=1))
    if guidance:
        parts.append(
            "## Human correction notes (highest priority)\n"
            "Apply these explicit instructions from the user; they override"
            " conflicting evidence:\n" + guidance
        )
    if time_align:
        parts.append(
            "## Time alignment (thorough analysis)\n"
            "Populate audio.syncMap: map line indexes to seconds using the MIR"
            " section boundaries and beat grid, with at least one entry for the"
            " first line of every section. Times must be non-decreasing."
        )

    parts.append("Now output the reconciled Song JSON only.")
    return "\n\n".join(parts)


def build_repair_prompt(errors: str) -> str:
    return (
        "Your previous JSON output failed schema validation with these errors:\n"
        f"{errors}\n\n"
        "Fix ALL of these and output the corrected, complete Song JSON only — "
        "no commentary, no markdown fences. Remember the chord-normalization rule: "
        "sounding harmony only, no shapes/tab/N.C., placements strictly ascending."
    )
