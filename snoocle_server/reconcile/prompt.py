"""Reconciliation prompt construction.

The baseline input is STRUCTURED DATA ONLY — every candidate text source
plus the MIR timeline as JSON — so the exact same request works across
Claude, GPT, and Gemini regardless of their audio-input support. An audio
snippet is an optional, provider-conditional attachment handled in
providers.py, never required for the baseline.

Two rules here are baked into the prompt AND enforced post-hoc, because a
prompt-level instruction alone is hope, not a guarantee:

- chord normalization -> the Song schema validators (engine.py repair loop);
- the LYRIC REFERENCE PROTOCOL -> lyric_refs.splice_lyrics. The model never
  writes a song's words; it points at the source line it read them in, and
  deterministic code substitutes the text. See lyric_refs.py for why.
"""

from __future__ import annotations

import json

from ..discovery.models import CandidateSource
from ..mir.base import MirAnalysis
from ..scope import AnalysisScope
from .lyric_refs import PRIOR_SONG_SOURCE_ID, describe_ref_sources

_SYSTEM_BASE = """You are a music transcription reconciliation engine for the Snoocle song-practice app.

You receive:
1. Multiple candidate chord/lyric sheets discovered on the web (parsed into lines with chord placements by character index). Sources vary in quality and may disagree.
2. An independent music-information-retrieval (MIR) analysis of the actual studio recording: beat grid, bpm, chord timeline with timestamps, structural sections, estimated key.

Your job: produce the single best reconciled song document as JSON conforming EXACTLY to the schema provided in the user message.

Reconciliation principles:
- USE ALL CANDIDATE SOURCES. Agreement between independent sources is strong evidence. Where sources disagree, prefer the reading consistent with the MIR chord timeline and key.
- The MIR analysis is audio ground truth for TIMING (bpm, section boundaries, syncMap times) and a strong signal for HARMONY, but its chord vocabulary may be coarser than the sheets (e.g. it may report C where the true chord is Cmaj7). Prefer a text source's richer chord quality when the root and family agree with the audio.
- Sections: name and order them from the text sources' hints plus MIR structure; attach startTime/endTime from the MIR structure timeline where alignment is clear.
- audio.syncMap: map line indexes to seconds using the MIR section boundaries and beat grid; include an entry at least for the first line of each section. Times must be non-decreasing.

CHORD NORMALIZATION RULE (non-negotiable):
Every chordPlacements.chord value MUST be the actual sounding harmony — never a fretboard shape, never a capo'd shape name, never tablature or fingering. Candidate sources have already been transposed to sounding pitch at ingestion (their declaredCapo is metadata about the original sheet, not an instruction to transpose again). displayPreferences.capo is a display-only preference — set it to 0; never bake a capo into chord identities. Valid chord symbols look like: C, F#m, Bbmaj7, Dm7/G, Esus4, A7, Gdim7, Cadd9. Invalid: x02210, 3-2-0-0-0-3, "Am shape", N.C. (omit no-chord positions instead)."""

#: The lyric-reference contract, stated once and shared by every prompt that
#: carries it. Structural, not a plea: the server resolves these refs and
#: rejects retyped text — see lyric_refs.py.
LYRIC_REF_CONTRACT = """LYRIC REFERENCE PROTOCOL (non-negotiable):
You do NOT write out the song's words. For each line you say WHERE its words are, and the server substitutes them verbatim. A line carries exactly ONE of:
- "lyricRef": {"sourceId": <a source id from this request>, "line": <lineIndex within that source>} — the normal case, for every line that has words.
- "lyricOverride": <text> together with "lyricOverrideReason": <why> — rare. Only when no single source line is right: merging two partial sources, fixing an obvious typo in a source, or a line no source covers. Every one is recorded in the song's provenance, and too many fail the run.
- "lyrics": "" — an instrumental line, which has no words at all. Empty string, and no lyricRef.

Worked example of two lines:
{"lineIndex": 7, "lyricRef": {"sourceId": "web-2", "line": 12}, "chordPlacements": [{"charIndex": 0, "chord": "F"}, {"charIndex": 18, "chord": "C"}]}
{"lineIndex": 8, "lyrics": "", "chordPlacements": [{"charIndex": 0, "chord": "Am"}, {"charIndex": 1, "chord": "G"}]}

charIndex is an index into the REFERENCED line's text as it appears in the source — count characters there, do not estimate. The server checks every index against the resolved line and sends mismatches back to you."""

_OUTPUT_RULES = """Output rules:
- Output ONLY the JSON document. No markdown fences, no commentary.
- lines: lineIndex contiguous from 0; chordPlacements strictly ascending by charIndex, charIndex within [0, len(lyrics)] (empty-lyric instrumental lines may use ordinal slots 0,1,2,...).
- sections reference lines by inclusive [startLineIndex, endLineIndex]; sections must not overlap and must be in ascending line order.
- Leave "provenance" as an empty array — the server appends provenance entries itself.
- Do not invent chords absent from all evidence."""


def build_system_prompt(lyric_refs: bool = True) -> str:
    """The system prompt, with or without the lyric-reference contract.

    Providers that are not prompted by this server (the deterministic mock)
    produce their Song directly and never see either version — see
    ``LLMProvider.emits_lyric_refs``.
    """
    parts = [_SYSTEM_BASE]
    if lyric_refs:
        parts.append(LYRIC_REF_CONTRACT)
    else:
        parts.append(
            "- Lyrics come from the text sources; pick the most complete, "
            "correctly-ordered lyric set.\n- Do not invent lyrics."
        )
    parts.append(_OUTPUT_RULES)
    return "\n\n".join(parts)


#: Back-compatible module constant: the prompt as sent on the normal path.
SYSTEM_PROMPT = build_system_prompt(lyric_refs=True)


#: The `listen=off, reconcile=off` contract, stated plainly. This run is a
#: targeted EDIT of an existing document, not a transcription task, and the
#: prompt has to say so — the model is otherwise being handed a page with two
#: empty evidence sections and every incentive to fill them from memory.
_NOTES_ONLY_BANNER = (
    "## Scope of this run: apply the user's notes only\n"
    "No new sources were gathered and the audio was not re-analyzed — both"
    " were deliberately switched off for this run. Correct the prior song"
    " below using the user's notes, and change NOTHING the notes do not ask"
    " about. Do not re-derive chords, sections, or timings from memory, and"
    " do not invent a song: if the notes do not mention something, the prior"
    " song's value for it is the right answer. Every line keeps its words by"
    f" referencing the prior song itself — {PRIOR_SONG_SOURCE_ID!r} is a"
    " referenceable source, so an unchanged line is"
    f' {{"sourceId": "{PRIOR_SONG_SOURCE_ID}", "line": <its lineIndex>}}.'
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
    evidence_manifest: dict | None = None,
    ref_index: dict[str, list[str]] | None = None,
) -> str:
    parts: list[str] = []
    parts.append(
        f"Reconcile the song {title!r} by {artist!r} into a single schema-compliant JSON document.\n"
        f"Use song id {song_id!r}"
        + (f" and audio.youtubeVideoId {youtube_video_id!r}." if youtube_video_id else ".")
    )
    # The manifest is the general form of the scope banner below: rather than
    # leaving the model to infer evidence quality from what happens to be
    # present, state it. It is DESCRIPTIVE — nothing is parsed back out of the
    # response and nothing downstream depends on it.
    if evidence_manifest:
        parts.append(
            "## Evidence manifest (what you are being handed, and where it came from)\n"
            "Context only — the evidence itself is below. Cached evidence is as"
            " valid as freshly gathered evidence: a cache hit means this exact"
            " work was already done, not that it is stale or second-hand. Use"
            " this to judge what is worth re-deriving, never as a source of"
            " song content.\n"
            + json.dumps({"evidenceManifest": evidence_manifest}, indent=1)
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

    # The exact ids and line ranges a lyricRef may name. Stated explicitly
    # rather than left to be inferred from the evidence sections below: an
    # unresolvable ref fails the run outright (see lyric_refs.py), so the
    # model must never have to guess what is addressable.
    if ref_index is not None:
        parts.append(
            "## Referenceable lyric sources (lyricRef.sourceId -> valid line range)\n"
            "These, and only these, are addressable. A ref to anything else"
            " fails the run — it is not repaired and it does not fall back to"
            " your own recollection of the song.\n"
            + describe_ref_sources(ref_index)
        )

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


def build_repair_prompt(errors: str, lyric_refs: bool = True) -> str:
    return (
        "Your previous JSON output failed validation with these errors:\n"
        f"{errors}\n\n"
        "Fix ALL of these and output the corrected, complete Song JSON only — "
        "no commentary, no markdown fences. Remember the chord-normalization rule: "
        "sounding harmony only, no shapes/tab/N.C., placements strictly ascending."
        + (
            " And the lyric protocol: every line with words carries a lyricRef"
            " into a source in this request — never the words themselves."
            if lyric_refs
            else ""
        )
    )
