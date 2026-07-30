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

A THIRD prompt lives here too, for a different kind of run entirely: a
targeted correction ("change the C to a B in line 12") doesn't need the
model to reconcile anything, and asking it to regenerate the whole Song for
a one-word fix is exactly what got a real correction blocked by a content
filter over a single chord. build_patch_system_prompt/build_patch_user_prompt
ask for a short list of OPERATIONS instead — see reconcile/patch_ops.py,
which is what actually applies them. Neither the reconciliation prompt above
nor the lyric-reference contract belongs in a patch run: there is no
reconciling to do, and an op never carries lyric text at all.
"""

from __future__ import annotations

import json

from ..discovery.models import CandidateSource
from ..mir.base import MirAnalysis
from ..scope import AnalysisScope
from .lyric_refs import PRIOR_SONG_SOURCE_ID, describe_ref_sources
from .patch_ops import MAX_OPS, describe_op_set

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
    quality_feedback: str | None = None,
    structure_feedback: str | None = None,
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
    # A MODEL-fault retry (see quality/escalation.py). Kept separate from
    # `guidance` on purpose: this is the server's own measurement of the
    # previous attempt, not a human instruction, and provenance must not
    # report it as one. It carries named metrics and named indexes — the one
    # retry this run allows is worth nothing if it only says "try harder".
    if quality_feedback:
        parts.append(quality_feedback)
    # The one thing that can make Mode B (realign.py) spend model tokens: a
    # deterministic comparison found a structural difference between the stored
    # document and the recording it is being re-timed to. Its own channel for
    # the same reason as above — it is a measurement, not a human instruction —
    # and a narrow one: the words and chords are settled, only the repeats are
    # open. See realign._structure_feedback.
    if structure_feedback:
        parts.append(structure_feedback)
    if time_align:
        parts.append(
            "## Time alignment (thorough analysis)\n"
            "Populate audio.syncMap: map line indexes to seconds using the MIR"
            " section boundaries and beat grid, with at least one entry for the"
            " first line of every section. Times must be non-decreasing."
        )

    parts.append("Now output the reconciled Song JSON only.")
    return "\n\n".join(parts)


# --- the patch protocol (notes-only, targeted corrections) ------------------

_PATCH_SYSTEM_PROMPT = f"""You are a targeted-correction engine for the Snoocle song-practice app.

You are given a song document that already exists and a short correction note naming specific things to fix. Your ENTIRE job is to say what changes — nothing else. You do not reconcile, re-transcribe, or reproduce the song. Everything the note does not mention stays exactly as it already is; you never restate it.

Answer with a JSON object of the shape:
{{"ops": [ <op>, <op>, ... ]}}

Each op is one of ({describe_op_set()}):
- {{"op": "replace_chord", "lineIndex": int, "charIndex": int, "from": "<chord now there>", "to": "<chord it should be>", "reason": "<why, optional>"}}
- {{"op": "insert_chord", "lineIndex": int, "charIndex": int, "chord": "<chord>", "reason": "<optional>"}} — charIndex must not already have a chord.
- {{"op": "remove_chord", "lineIndex": int, "charIndex": int, "from": "<chord expected there, optional>", "reason": "<optional>"}}
- {{"op": "move_chord", "lineIndex": int, "fromCharIndex": int, "toCharIndex": int, "chord": "<expected chord, optional>", "reason": "<optional>"}} — moves a chord within the SAME line only.
- {{"op": "set_section_name", "sectionIndex": int, "to": "<new name>", "reason": "<optional>"}}
- {{"op": "set_section_bounds", "sectionIndex": int, "startLineIndex": int, "endLineIndex": int, "reason": "<optional>"}} — give the FULL new range, both ends.
- {{"op": "split_line", "lineIndex": int, "charIndex": int, "reason": "<optional>"}} — splits the line's EXISTING lyrics at that character position into two lines; every later line shifts down to make room.
- {{"op": "merge_line", "lineIndex": int, "reason": "<optional>"}} — merges this line with the one right after it.

Worked example — fixing one chord and renaming a section, in one patch:
{{"ops": [
  {{"op": "replace_chord", "lineIndex": 12, "charIndex": 21, "from": "C", "to": "B", "reason": "sheet has a walkdown here, not a straight C"}},
  {{"op": "set_section_name", "sectionIndex": 3, "to": "Bridge"}}
]}}

Non-negotiable rules:
- NEVER write out any of the song's words, in an op or anywhere else. There is no "replace lyric" op and there never will be — that is not an oversight, it is the whole point of this format. If the correction genuinely needs different words, do not attempt a patch: respond with exactly {{"ops": [], "needsFullReconcile": true, "reason": "<why>"}} instead of forcing the words into some other field.
- charIndex is exact, not an estimate: it is the position in the LINE'S OWN lyrics as given to you below, character by character.
- Reference only lines, sections, and chords that already exist in the document below. An op the server can't apply — wrong "from" chord, a charIndex that isn't there, a section that doesn't exist — fails the whole run. There is no retry: get it right the first time, or say plainly that you can't.
- At most {MAX_OPS} ops. If the correction genuinely needs more than that, it is not a targeted correction — respond with {{"ops": [], "needsFullReconcile": true, "reason": "<why>"}}.
- Output ONLY the JSON object. No markdown fences, no commentary, no prose before or after."""


def build_patch_system_prompt() -> str:
    return _PATCH_SYSTEM_PROMPT


def build_patch_user_prompt(
    title: str,
    artist: str,
    song_id: str,
    prior_song: dict,
    guidance: str,
) -> str:
    return "\n\n".join(
        [
            f"Correcting the song {title!r} by {artist!r} (id {song_id!r}).",
            "## The document as it exists now\n" + json.dumps(prior_song, indent=1),
            "## Correction note\n" + guidance,
            "Now output the ops JSON only.",
        ]
    )


def build_patch_repair_prompt(errors: str) -> str:
    # NOT currently used by the engine — see engine.py's docstring on why a
    # patch failure is fatal rather than repaired — but kept as a real,
    # working function rather than a stub: a future retry policy for
    # clearly-malformed-JSON-only failures (as opposed to a wrong "from")
    # can use this without inventing prompt text under time pressure.
    return (
        "Your previous response did not parse as a valid patch:\n"
        f"{errors}\n\n"
        'Output the corrected {"ops": [...]} JSON only — no commentary, no '
        "markdown fences, no lyric text anywhere in it."
    )


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
