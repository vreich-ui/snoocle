"""In-process agentic reconciler (provider "anthropic-agent").

Unlike the "agent" provider (which delegates to an EXTERNAL agent workspace
over MCP), this runs the agentic loop INSIDE this server: the Anthropic SDK
drives Claude through server-side web search + web fetch and two local tools
(chord-sheet fetch/parse, windowed MIR) until it emits the final Song JSON.
No external agent service, no MCP hop — one Cloud Run container.

The loop is written by hand rather than using the SDK's beta tool runner:
server-side tools (web_search/web_fetch) can pause a turn (stop_reason
"pause_turn"), and the manual loop resumes that work explicitly.
"""

from __future__ import annotations

import json
import logging
import time

from ..config import settings
from ..discovery.fetch import extract_sheet_text, fetch_page
from ..discovery.service import candidate_from_text
from ..mir.cache import WINDOW_ACCURACY, analyze_cached
from ..mir.pipeline import analyze_window
from .providers import ContentFilterError, LLMProvider, LLMResponse, ProviderError
from .trace import TraceRecorder

log = logging.getLogger(__name__)


# The system prompt is assembled from named sections so an operator's AgentConfig
# can swap the theory-rules / retrieval-recipe sections, append extra rules, or
# (dangerously) replace the whole base — while _OUTPUT_CONTRACT is ALWAYS
# appended and never editable (schema validation is the real guardrail).

_PROMPT_ROLE = """\
You are an expert music transcriber. Your job is to produce one Snoocle Song \
JSON object that faithfully captures a song's chords, lyrics, and sections."""

_PROMPT_EVIDENCE = """\
Evidence rules:
- The MIR chord timeline (derived from the ACTUAL audio recording) is the \
primary evidence for each chord's ROOT and its major/minor QUALITY.
- Text sources (chord sheets, lyric pages) are the primary evidence for \
lyrics, section structure, and chord EXTENSIONS (7ths, 9ths, sus, add).
- A source that declares a capo is written N semitones BELOW sounding pitch. \
Transpose it up N semitones before comparing it to the audio or to other \
sources."""

_PROMPT_THEORY = """\
Music theory:
- Prefer chord readings that are diatonic to the established key, or that are \
classically explainable (secondary dominants, borrowed iv or bVII, the \
relative major/minor). When the audio is ambiguous, let theory break the tie.
- Spell enharmonics according to the key signature: F#m in A major, never Gbm."""

_PROMPT_CHORD_RULE = """\
Hard chord rule:
- Every chord symbol you emit is the SOUNDING harmony. NEVER write fretboard \
shapes or tab fingerings, and NEVER bake a capo into the chord names. \
displayPreferences.capo MUST be 0."""

_PROMPT_RECIPE = """\
Retrieval recipe (follow it; do not improvise a research plan):
1. Read the provided candidates and MIR timeline FIRST. If two or more \
candidates agree with each other and with the MIR timeline on the key and \
the core progression, SKIP the web entirely and write the Song now.
2. Otherwise run ONE web_search: `<title> <artist> chords lyrics`. From the \
results pick the 2-3 most promising chord pages and call `fetch_chord_sheet` \
on each (never web_fetch a chord page — fetch_chord_sheet parses and \
capo-normalizes in one step).
3. At most ONE more web_search, only if the lyrics are still incomplete.
4. Call `analyze_audio_window` only when text sources disagree about a \
specific passage AND the provided MIR timeline does not cover it."""

_PROMPT_BUDGET = """\
Hard budget: at most 2 web_search calls, 4 page fetches, and 2 \
analyze_audio_window calls per song. Disagreements are settled by the MIR \
timeline and music theory — NOT by more searching. When you have enough \
information to act, act: produce the Song instead of continuing to verify. \
Two agreeing sources plus the provided MIR is always enough."""

# NON-EDITABLE. Always the last section of the system prompt. The lyric
# protocol lives here rather than in an editable section on purpose: it is
# enforced by lyric_refs.splice_lyrics, so an operator who edited it away
# would only be making every run fail a rule the model was no longer told.
_OUTPUT_CONTRACT = """\
Output contract:
- Your FINAL message must be EXACTLY ONE JSON object — the Song — that \
validates against the provided songSchema. No markdown fences, no commentary, \
no prose before or after. The schema is strict: only its keys are allowed. \
Set id, title, artist, and youtubeVideoId from the request.

Lyric reference protocol (non-negotiable):
- You do NOT write out the song's words. For each line you say WHERE its \
words are, and the server substitutes them verbatim. Each line carries \
exactly ONE of:
  * "lyricRef": {"sourceId": <id of a source in this request>, "line": \
<lineIndex within that source>} — the normal case, every line that has words.
  * "lyricOverride": <text> plus "lyricOverrideReason": <why> — rare, only \
when no single source line is right (merging two partial sources, fixing an \
obvious source typo, a line no source covers). Each one is recorded in the \
song's provenance, and too many fail the run.
  * "lyrics": "" — an instrumental line, which has no words. Empty string, \
and no lyricRef.
- Worked example of two lines:
  {"lineIndex": 7, "lyricRef": {"sourceId": "web-2", "line": 12}, \
"chordPlacements": [{"charIndex": 0, "chord": "F"}, {"charIndex": 18, \
"chord": "C"}]}
  {"lineIndex": 8, "lyrics": "", "chordPlacements": [{"charIndex": 0, \
"chord": "Am"}, {"charIndex": 1, "chord": "G"}]}
- Referenceable sourceIds are the candidates in this request, "prior-song" \
when one is given, and any sheet you fetch with fetch_chord_sheet (it \
returns the sourceId to use). A ref to anything else, or to a line outside \
that source, FAILS the run — it is not repaired and it does not fall back \
to your own recollection of the song.
- charIndex is an index into the REFERENCED line's text as it appears in \
that source. Count characters there; do not estimate."""


def build_system_blocks(cfg=None) -> list[dict]:
    """Assemble the cached system block from the default sections plus any
    operator overrides in ``cfg`` (an AgentConfig). ``_OUTPUT_CONTRACT`` is
    always appended last, regardless of overrides."""
    if cfg is not None and cfg.instructions_override:
        base = cfg.instructions_override
    else:
        theory = (cfg.theory_rules if cfg and cfg.theory_rules else _PROMPT_THEORY)
        recipe = (cfg.retrieval_recipe if cfg and cfg.retrieval_recipe else _PROMPT_RECIPE)
        base = "\n\n".join(
            [_PROMPT_ROLE, _PROMPT_EVIDENCE, theory, _PROMPT_CHORD_RULE, recipe, _PROMPT_BUDGET]
        )
    parts = [base]
    if cfg is not None and cfg.instructions_extra:
        parts.append(cfg.instructions_extra)
    parts.append(_OUTPUT_CONTRACT)  # always, never editable
    return [{"type": "text", "text": "\n\n".join(parts), "cache_control": {"type": "ephemeral"}}]


# Backward-compatible defaults (no config): the original prompt + block.
SYSTEM_BLOCKS = build_system_blocks(None)
SYSTEM_PROMPT = SYSTEM_BLOCKS[0]["text"]

_FETCH_TOOL = {
    "name": "fetch_chord_sheet",
    "description": "Fetch a URL and parse it as a chord sheet. Returns a structured candidate (lines with chord placements at sounding pitch, declared key/capo, confidence) or an error if the page has no usable transcription. Call this for chord/tab pages found via web_search; prefer it over web_fetch for chord sites because it parses and capo-normalizes.",
    "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"], "additionalProperties": False},
}
_WINDOW_TOOL = {
    "name": "analyze_audio_window",
    "description": "Run audio chord/beat analysis on a specific window of the actual recording. Returns a chord timeline with ABSOLUTE timestamps (seconds in the video), bpm and beats for that window. Use when text sources disagree about a passage and the provided MIR timeline does not cover it. Windows are capped at 60 seconds.",
    "input_schema": {"type": "object", "properties": {"start_seconds": {"type": "number"}, "end_seconds": {"type": "number"}}, "required": ["start_seconds", "end_seconds"], "additionalProperties": False},
}


def _build_tools(max_web_search: int, max_fetch: int, disabled=frozenset()) -> list[dict]:
    """Tools with the server-tool budget set by the analysis-depth profile;
    any tool named in ``disabled`` (an operator AgentConfig) is omitted — an
    undeclared tool simply cannot be called."""
    candidates = [
        {"type": "web_search_20260209", "name": "web_search", "max_uses": max_web_search},
        {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": max_fetch},
        _FETCH_TOOL,
        _WINDOW_TOOL,
    ]
    return [t for t in candidates if t["name"] not in disabled]


# Default budget (used when no depth profile is injected) mirrors "standard".
TOOLS = _build_tools(2, 3)

# Cap for windowed on-demand analysis (seconds) — see the tool description.
_MAX_WINDOW_SECONDS = 60.0


def _classify_api_error(e: Exception) -> ProviderError:
    """Turn a raw Anthropic SDK error into a clean, actionable ProviderError.

    The content-filtering block (a 400 whose message names the policy) is the
    common one for songs with no text sources — surface it as its own class so
    the app can show a real message instead of a JSON dump."""
    msg = str(e)
    lowered = msg.lower()
    if "content filtering" in lowered or "content_filter" in lowered:
        # The old copy blamed "no chord/lyric sources were found". The two
        # live failures that motivated the lyric-reference protocol both had
        # FOUR sources successfully fetched, so that claim was not just
        # unhelpful, it pointed at the wrong thing. The cause is what the
        # model was asked to WRITE — see reconcile/lyric_refs.py.
        return ContentFilterError(
            "anthropic-agent: the model's output was blocked by Anthropic's "
            "content-filtering policy for this song. This happens when the "
            "response reproduces long verbatim lyrics, independently of how "
            "many sources the run gathered. Reconciliation normally emits "
            "lyric REFERENCES instead of lyric text (the server splices them "
            "in), so a block here usually means the model wrote lyrics out "
            "anyway — retrying often clears it, since the filter is not "
            "deterministic. A lower analysis depth also helps."
        )
    return ProviderError(f"anthropic-agent: model API error: {msg}")


def _block_field(block, key: str, default=None):
    """Read one field from a content block.

    Blocks arrive from the SDK as objects (``.type``/``.id``), but the ones this
    module writes into the history — and the ones tests script — are plain
    dicts. Both shapes have to be readable, and ``.model_dump()`` is not
    guaranteed to exist on either.
    """
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _is_tool_result_block(block_type: str) -> bool:
    """True for a client ``tool_result`` and for every server-side result block
    (``web_search_tool_result``, ``text_editor_code_execution_tool_result``...)."""
    return block_type == "tool_result" or block_type.endswith("_tool_result")


def _tool_use_kind(block_type: str) -> str | None:
    """``"client"`` for a local ``tool_use``, ``"server"`` for one the API runs
    itself, ``None`` for anything that is not a tool use.

    Server-side uses are ``server_tool_use`` (web_search / web_fetch) plus the
    blocks their API-managed code-execution container emits, whose types carry
    ``code_execution`` (e.g. ``text_editor_code_execution``). Result blocks are
    excluded first: their type names contain the same words.
    """
    if _is_tool_result_block(block_type):
        return None
    if block_type == "tool_use":
        return "client"
    if block_type == "server_tool_use" or "code_execution" in block_type:
        return "server"
    return None


_TRUNCATED_TURN_NOTE = "[turn truncated at the token cap]"
_ORPHANED_RESULT_NOTE = "[tool results dropped: the calls they answered were truncated away]"
_TRUNCATED_TOOL_ERROR = (
    "the assistant turn was cut off before this tool could run "
    "(stop_reason max_tokens); no result was produced"
)


def _result_ids(blocks) -> set:
    """The tool-use ids that the ``*_tool_result`` blocks in one content list
    answer. A non-list content (a plain string user message) answers nothing."""
    if not isinstance(blocks, (list, tuple)):
        return set()
    return {
        _block_field(block, "tool_use_id")
        for block in blocks
        if _is_tool_result_block(str(_block_field(block, "type", "")))
    }


def _answered_ids(messages) -> set:
    """Every tool-use id answered anywhere in ``messages``."""
    answered: set = set()
    for message in messages:
        answered |= _result_ids(message.get("content"))
    return answered


def _tool_use_ids(blocks) -> set:
    """The ids of every tool use (client or server) in one content list."""
    return {
        _block_field(block, "id")
        for block in blocks
        if _tool_use_kind(str(_block_field(block, "type", ""))) is not None
    }


def _open_client_ids(blocks, answered: set) -> list:
    """Client ``tool_use`` ids in ``blocks`` that ``answered`` does not cover,
    in the order the model emitted them."""
    return [
        _block_field(block, "id")
        for block in blocks
        if _tool_use_kind(str(_block_field(block, "type", ""))) == "client"
        and _block_field(block, "id") not in answered
    ]


def _first_unpairable(content, answered_later: set) -> int:
    """Index of the first block in ``content`` that can never be paired — a
    SERVER tool use no ``*_tool_result`` answers — or ``len(content)`` when
    every server use is answered.

    ``answered_later`` are the ids answered by the messages AFTER this one (a
    paused turn's result arrives with the resumption); results inside this same
    message count too, because server results are inline.
    """
    cut = len(content)
    while True:
        answered = answered_later | _result_ids(content[:cut])
        first = next(
            (
                i
                for i, block in enumerate(content[:cut])
                if _tool_use_kind(str(_block_field(block, "type", ""))) == "server"
                and _block_field(block, "id") not in answered
            ),
            None,
        )
        if first is None:
            return cut
        # Everything from here on goes with it — and a dropped tail can take an
        # inline result with it, which re-opens a block that looked paired a
        # moment ago, so the kept region has to be re-checked.
        cut = first


def _tool_summary(name: str, tool_input: dict, result: dict, is_error: bool) -> str:
    """One human-readable line describing a tool call and its outcome."""
    if is_error:
        return f"{name} failed: {result.get('error')}"
    if name == "fetch_chord_sheet":
        lines = result.get("lines")
        n = len(lines) if isinstance(lines, list) else "?"
        key = result.get("declaredKey") or result.get("key") or "?"
        return f"fetched {tool_input.get('url', '?')} → {n} lines, key={key}"
    if name == "analyze_audio_window":
        chords = result.get("chords") or []
        return (
            f"analyzed {tool_input.get('start_seconds')}–{tool_input.get('end_seconds')}s "
            f"→ {len(chords)} chord segment(s), bpm={result.get('bpm')}"
        )
    return f"{name} ok"


def fetch_chord_sheet(url: str, source_id: str = "agent-1") -> dict:
    """Fetch a URL, extract chord-sheet text, and parse it into a candidate.

    Returns the candidate's serialized dict, or an ``{"error": ...}`` object
    (never raises) when the page can't be fetched or isn't a plausible sheet.
    """
    try:
        page = fetch_page(url)
    except Exception as e:  # noqa: BLE001 — a dead/blocked page is a tool error, not a crash
        return {"error": f"fetch failed: {e}"}
    text = extract_sheet_text(page)
    candidate = candidate_from_text(text, source_id=source_id, url=url)
    if candidate is None:
        return {"error": "page is not a plausible chord sheet"}
    return candidate.model_dump(exclude_none=True)


def analyze_audio_window(audio_path: str | None, start_seconds: float, end_seconds: float) -> dict:
    """Windowed MIR on the actual recording, clamped to <= 60s and the track.

    Returns ``{"chords": [...], "beats": N, "bpm": ...}`` with absolute
    timestamps, or an ``{"error": ...}`` object when no audio is available or
    analysis fails.
    """
    if not audio_path:
        return {"error": "no audio available"}
    try:
        start = max(float(start_seconds), 0.0)
        end = float(end_seconds)
        if end <= start:
            return {"error": "end_seconds must be greater than start_seconds"}
        end = min(end, start + _MAX_WINDOW_SECONDS)  # analyze_window clamps to track duration
        # Cached per (audio bytes, engines, window): an agent that probes the
        # same span twice in a run — or across re-analyses of the same song —
        # pays for it once. `analyze_window` stays what computes.
        analysis, _cache_info = analyze_cached(
            audio_path,
            accuracy=WINDOW_ACCURACY,
            window=(start, end),
            compute=lambda: analyze_window(audio_path, start, end),
        )
        # Report the span that was ACTUALLY analyzed (post-clamp) so both the
        # model and the run trace see the real coverage, not the request.
        if analysis.analyzed_windows:
            start = analysis.analyzed_windows[0].start
            end = analysis.analyzed_windows[-1].end
        return {
            "window": {"start": round(start, 2), "end": round(end, 2)},
            "chords": [
                {"start": round(c.start, 2), "end": round(c.end, 2), "chord": c.chord}
                for c in analysis.chords
            ],
            "beats": len(analysis.beats),
            "bpm": round(analysis.bpm, 1) if analysis.bpm else None,
        }
    except Exception as e:  # noqa: BLE001 — surface any failure to the model as a tool error
        return {"error": str(e)}


class AnthropicAgentProvider(LLMProvider):
    """Claude runs the reconciliation agent loop inside this process."""

    name = "anthropic-agent"
    default_model = "claude-opus-4-8"
    supports_audio = False  # audio is reached through analyze_audio_window, not attached
    wants_context = True
    emits_lyric_refs = True  # see _OUTPUT_CONTRACT and reconcile/lyric_refs.py

    # engine.py injects the structured inputs (incl. audio_path) here before complete()
    context: dict | None = None
    # engine.py injects the run's trace recorder here so each model turn and
    # tool call lands in the same timeline as the engine's inputs/repair steps.
    trace: TraceRecorder | None = None

    def __init__(self) -> None:
        # The real Anthropic-format conversation, including tool_use/tool_result
        # and thinking blocks — kept across repair rounds so the full history
        # (and its cached prefix) carries forward.
        self._messages: list[dict] = []
        self._fetch_count = 0

    def _create_client(self):
        # Lazy import + isolated in a method so tests can monkeypatch it.
        import anthropic

        return anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def _build_first_user_message(self) -> dict:
        ctx = self.context or {}
        mir = ctx.get("mir")
        depth = ctx.get("depth")
        payload = {
            "songId": ctx.get("song_id"),
            "title": ctx.get("title"),
            "artist": ctx.get("artist"),
            "youtubeVideoId": ctx.get("youtube_video_id"),
            "mediaUrl": ctx.get("media_url"),
            "mir": mir.to_prompt_payload() if mir is not None else None,
            "candidates": [c.model_dump(exclude_none=True) for c in ctx.get("candidates") or []],
            "songSchema": ctx.get("song_schema"),
        }
        # Exactly which ids a lyricRef may name, and how many lines each has.
        # Stated rather than left to be inferred from `candidates`, because an
        # unresolvable ref fails the run outright (lyric_refs.py rule 3).
        ref_index = ctx.get("ref_index")
        if ref_index is not None:
            payload["referenceableLyricSources"] = {
                source_id: {"lines": len(lines)} for source_id, lines in ref_index.items()
            }
        # Descriptive context: what this run reused vs recomputed, and how good
        # each input is. See manifest.py — never a source of song content.
        if ctx.get("evidence_manifest"):
            payload["evidenceManifest"] = ctx["evidence_manifest"]
        if getattr(self, "_effective_budget", None):
            payload["toolBudget"] = self._effective_budget
        if depth is not None and depth.time_align:
            payload["fillSyncMap"] = (
                "Thorough analysis: also populate audio.syncMap (lineIndex -> "
                "seconds) from the MIR section boundaries and beat grid, at "
                "least one entry per section. Times must be non-decreasing."
            )
        if ctx.get("prior_song") is not None:
            payload["priorHumanEditedSong"] = ctx["prior_song"]
        if ctx.get("guidance"):
            payload["humanCorrectionNotes"] = ctx["guidance"]
        # The pipeline can only decline to CALL discover/acquire; this provider
        # has its own fetch/analyze tools and would happily re-gather exactly
        # the evidence the user just switched off. The scope has to reach the
        # agent as an instruction, not just as an empty candidates list.
        scope = ctx.get("scope")
        if scope is not None:
            payload["scope"] = {"listen": scope.listen, "reconcile": scope.reconcile}
            if scope.notes_only:
                payload["scopeInstruction"] = (
                    "APPLY-NOTES-ONLY run: do not fetch chord sheets and do not "
                    "analyze audio windows. Return priorHumanEditedSong with "
                    "humanCorrectionNotes applied and nothing else changed."
                )
            elif not scope.reconcile:
                payload["scopeInstruction"] = (
                    "Do not fetch new chord sheets for this run — source "
                    "gathering was switched off. Work from the audio analysis, "
                    "the prior song, and the user's notes."
                )
            elif not scope.listen:
                payload["scopeInstruction"] = (
                    "Do not analyze audio windows for this run — listening was "
                    "switched off. Keep the prior song's existing times."
                )
        return {"role": "user", "content": json.dumps(payload)}

    def _register_ref_source(self, source_id: str, fetched: dict) -> None:
        """Make a just-fetched sheet addressable by ``lyricRef``.

        The index is the engine's dict (injected via context) — mutating it
        here is deliberate: "what the model can see" and "what the server can
        resolve" must be the same set at every moment of the run.
        """
        index = (self.context or {}).get("ref_index")
        if index is None or "error" in fetched:
            return
        lines = fetched.get("lines")
        if isinstance(lines, list):
            index[source_id] = [
                str(line.get("lyrics", "")) if isinstance(line, dict) else ""
                for line in lines
            ]

    def _run_tool(self, block) -> dict:
        name = block.name
        tool_input = block.input or {}
        if name == "fetch_chord_sheet":
            self._fetch_count += 1
            source_id = f"agent-{self._fetch_count}"
            result = fetch_chord_sheet(tool_input.get("url", ""), source_id=source_id)
            # A sheet fetched mid-run is evidence like any other, so its lines
            # have to become REFERENCEABLE — otherwise the agent finds a better
            # source than the ones it was handed and then cannot point at it,
            # and an unresolvable ref fails the run (lyric_refs.py rule 3).
            self._register_ref_source(source_id, result)
        elif name == "analyze_audio_window":
            audio_path = (self.context or {}).get("audio_path")
            result = analyze_audio_window(
                audio_path, tool_input.get("start_seconds"), tool_input.get("end_seconds")
            )
        else:
            result = {"error": f"unknown tool {name!r}"}
        is_error = isinstance(result, dict) and "error" in result
        if self.trace is not None:
            self.trace.step(
                "tool", f"tool:{name}",
                _tool_summary(name, tool_input, result, is_error),
                detail={"tool": name, "input": tool_input, "result": result},
            )
            if name == "analyze_audio_window" and not is_error:
                # The probe also lands on the run's MIR record (un-truncated)
                # so the GUI timeline can shade exactly what the agent examined.
                self.trace.add_mir_window(
                    {
                        "window": result.get("window"),
                        "chords": result.get("chords"),
                        "bpm": result.get("bpm"),
                        "beats": result.get("beats"),
                    }
                )
        tool_result: dict = {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": json.dumps(result),
        }
        if is_error:
            tool_result["is_error"] = True
        return tool_result

    def _drop_orphaned_results(self, start: int, dropped_ids: set) -> int:
        """Delete the ``*_tool_result`` blocks that ``dropped_ids`` orphaned.

        Truncating a NON-trailing assistant turn breaks the pairing in the other
        direction: a client ``tool_use`` inside the dropped region was answered
        by a ``tool_result`` in a LATER message, and that result now names a
        tool use the request no longer contains — which the API rejects just as
        it rejects the unanswered use. A message left with nothing but dropped
        results keeps a note, because empty content is rejected too.
        """
        removed = 0
        for i in range(start, len(self._messages)):
            message = self._messages[i]
            blocks = message.get("content")
            if not isinstance(blocks, (list, tuple)):
                continue
            kept = [
                block
                for block in blocks
                if not (
                    _is_tool_result_block(str(_block_field(block, "type", "")))
                    and _block_field(block, "tool_use_id") in dropped_ids
                )
            ]
            if len(kept) == len(blocks):
                continue
            removed += len(blocks) - len(kept)
            self._messages[i] = {
                "role": message.get("role"),
                "content": kept or [{"type": "text", "text": _ORPHANED_RESULT_NOTE}],
            }
        return removed

    def _stub_client_results(self, idx: int, tool_use_ids: list) -> None:
        """Answer the assistant turn at ``idx``'s open client tool uses."""
        stubs = [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": json.dumps({"error": _TRUNCATED_TOOL_ERROR}),
                "is_error": True,
            }
            for tool_use_id in tool_use_ids
        ]
        following = self._messages[idx + 1] if idx + 1 < len(self._messages) else None
        if following is not None and following.get("role") == "user":
            existing = following.get("content")
            # tool_result blocks lead the user turn they answer.
            if isinstance(existing, list):
                following["content"] = stubs + existing
                return
            if isinstance(existing, str) and existing:
                # A plain-text user turn (the engine's repair prompt) becomes a
                # text block so the results can join it, rather than opening a
                # second user message between the two.
                following["content"] = [*stubs, {"type": "text", "text": existing}]
                return
        self._messages.insert(idx + 1, {"role": "user", "content": stubs})

    def _close_open_assistant_turn(self) -> None:
        """Make EVERY assistant turn in the history self-consistent again.

        A turn the API cut short (stop_reason ``max_tokens``) can end ON a
        tool-use block: the model asked for a tool and the turn died before the
        result existed. Sending that turn back — which every later turn and
        every repair round does — is a 400: "tool use with id ... was found
        without a corresponding ..._tool_result block", which kills the whole
        reconciliation. Nothing else in this loop closes such a turn, so:

        - an unanswered CLIENT ``tool_use`` gets an is_error ``tool_result``
          (we own that side of the protocol, so the pair can be completed);
        - an unanswered SERVER tool use is DROPPED, along with everything after
          it, back to the last complete pair. Its ``*_tool_result`` is generated
          by the API inside its own container; a client cannot synthesise one,
          so not carrying the open block is the only valid repair. Dropping it
          can orphan a later ``tool_result`` (see ``_drop_orphaned_results``),
          which is broken pairing too and goes with it.

        Every assistant message is scanned, not just the last one: the open
        block that produced the live 400 was at ``messages.1``, and a turn stops
        being the last one as soon as anything is appended after it — its client
        tool results, the resumption of a paused turn, or the repair prompt.

        The one block that is legitimately pending is a SERVER tool use in a
        GENUINELY TRAILING assistant message: that is the ``pause_turn`` resume
        shape, where re-sending the still-open block is how the paused work
        continues. "Genuinely trailing" means last in the history AND staying
        there — a turn that also has an open client tool use does not qualify,
        because answering that appends a user turn right after it. Once a
        resumption HAS happened, the paused turn is no longer trailing and its
        result is inline in the message that follows, so it stays paired without
        the exemption; if the resumption was itself truncated before that result
        arrived, the block is an orphan and goes.

        A turn with no open blocks is left exactly as it is — same dicts, same
        lists — so this is inert on the normal path.
        """
        closed = dropped = stubbed = orphaned = 0
        i = 0
        while i < len(self._messages):
            message = self._messages[i]
            content = message.get("content")
            if message.get("role") != "assistant" or not isinstance(content, (list, tuple)):
                i += 1
                continue
            # Answered ids: server results are inline in the same assistant
            # message, client results (and a paused turn's result) arrive in the
            # message(s) that follow it.
            answered_later = _answered_ids(self._messages[i + 1:])
            trailing = (
                i == len(self._messages) - 1
                and not _open_client_ids(content, answered_later | _result_ids(content))
            )
            cut = len(content) if trailing else _first_unpairable(content, answered_later)
            kept = list(content[:cut])
            unanswered_client = _open_client_ids(kept, answered_later | _result_ids(kept))
            if cut == len(content) and not unanswered_client:
                i += 1
                continue  # nothing open in this turn — leave its stored dicts alone
            closed += 1
            if cut < len(content):
                # An assistant message may not be empty, so a turn that was
                # nothing but an open server block keeps a marker instead of
                # vanishing.
                self._messages[i] = {
                    "role": "assistant",
                    "content": kept or [{"type": "text", "text": _TRUNCATED_TURN_NOTE}],
                }
                dropped += len(content) - cut
                orphaned += self._drop_orphaned_results(i + 1, _tool_use_ids(content[cut:]))
            if unanswered_client:
                self._stub_client_results(i, unanswered_client)
                stubbed += len(unanswered_client)
            i += 1
        if closed:
            log.warning(
                "anthropic-agent: closed %d open assistant turn(s) (dropped %d "
                "unpairable block(s), stubbed %d tool result(s), dropped %d "
                "orphaned tool result(s))",
                closed, dropped, stubbed, orphaned,
            )

    def complete(self, system, turns, model=None, max_tokens=None, audio=None):
        if not settings.anthropic_api_key:
            raise ProviderError("anthropic-agent: SNOOCLE_ANTHROPIC_API_KEY is not configured")
        if not self.context:
            raise ProviderError("anthropic-agent provider requires engine-injected context")

        # Precedence for every knob: explicit request > operator AgentConfig >
        # analysis-depth profile > server default.
        cfg = (self.context or {}).get("agent_config")
        depth = (self.context or {}).get("depth")

        def _pick(cfg_val, depth_attr, default):
            if cfg_val is not None:
                return cfg_val
            if depth is not None:
                return getattr(depth, depth_attr)
            return default

        resolved_model = (
            model or (cfg.model if cfg else None)
            or settings.llm_model or settings.anthropic_agent_model
        )
        effort = (
            (cfg.effort if cfg else None) or (depth.effort if depth is not None else None)
            or settings.anthropic_agent_effort
        )
        max_turns = (cfg.max_turns if cfg else None) or settings.anthropic_agent_max_turns
        web = _pick(cfg.max_web_search if cfg else None, "max_web_search", 2)
        fetch = _pick(cfg.max_fetch if cfg else None, "max_fetch", 3)
        windows = _pick(cfg.max_windows if cfg else None, "max_windows", 2)
        disabled = set(cfg.disabled_tools) if cfg else set()
        # Scope must REMOVE tools, not merely ask the model not to use them.
        # The payload carries a scopeInstruction too, but an instruction is a
        # request: the model kept searching and re-analyzing audio on runs where
        # the user had switched exactly that off, then ground into the 900s
        # reconcile timeout. An undeclared tool cannot be called, so the scope
        # is enforced here and merely explained in the prompt.
        scope = (self.context or {}).get("scope")
        if scope is not None:
            if not scope.listen:
                disabled.add("analyze_audio_window")
            if not scope.reconcile:
                disabled.update(("web_search", "web_fetch", "fetch_chord_sheet"))
        tools = _build_tools(web, fetch, frozenset(disabled))
        system_blocks = build_system_blocks(cfg)
        # Budget the first user message advertises to the model (see _build...).
        self._effective_budget = {"webSearch": web, "pageFetch": fetch, "audioWindow": windows}

        if len(turns) == 1:
            # First attempt: build a fresh conversation from the injected context.
            self._messages = [self._build_first_user_message()]
            self._fetch_count = 0
        else:
            # Repair round: the engine passed [user, assistant, repair-user, ...].
            # The assistant's previous final answer is already in self._messages;
            # append only the new repair prompt and continue the same loop.
            # That previous answer may be a turn the API cut short — appending
            # this prompt is what turns its open tool use into an unpairable one,
            # so the loop below sanitises the history AFTER the append, not here.
            last = self._messages[-1] if self._messages else None
            if last is not None and last.get("role") == "user" and isinstance(
                last.get("content"), list
            ):
                # Closing the turn left stub tool_results in a trailing user
                # message; the repair prompt joins that same turn (results
                # first, then text) rather than opening a second user message.
                last["content"] = [*last["content"], {"type": "text", "text": turns[-1]["text"]}]
            else:
                self._messages.append({"role": "user", "content": turns[-1]["text"]})

        client = self._create_client()
        usage: dict = {}
        response = None
        # The server-side tools (web_search / web_fetch) execute in an
        # API-managed container. Once one has run, EVERY subsequent turn of the
        # same conversation must name that container, or the API rejects the
        # request with "container_id is required when there are pending tool
        # uses generated by code execution with tools" — a 400 that kills the
        # whole reconciliation mid-loop. The id only appears on the response
        # that allocated it, so it has to be carried forward by hand.
        container_id: str | None = None
        for turn in range(1, max_turns + 1):
            # Every request has to be internally consistent, and this loop is
            # what breaks that: a turn that stopped with `tool_use` while also
            # carrying an open server block stops being trailing the moment its
            # client results are appended below, and a paused turn whose
            # resumption was truncated is left open behind the repair prompt.
            # Sanitising here — after every append, before every send — is the
            # only point that sees the history exactly as the API will.
            self._close_open_assistant_turn()
            turn_start = time.monotonic()
            request: dict = {
                "model": resolved_model,
                "max_tokens": 16000,
                "thinking": {"type": "adaptive"},
                # effort is the dominant wall-clock lever for this loop; the
                # analysis-depth profile sets it (see reconcile/depth.py).
                "output_config": {"effort": effort},
                # auto-cache the latest prefix so each turn reuses the whole
                # prior conversation (system block carries its own breakpoint).
                "cache_control": {"type": "ephemeral"},
                "system": system_blocks,
                # no temperature/top_p/top_k: sampling params are rejected here
                "messages": self._messages,
            }
            # A notes-only run disables every tool. Send no `tools` key at all
            # rather than an empty list — the two are not equivalent, and an
            # empty list is rejected.
            if tools:
                request["tools"] = tools
            # Omitted entirely rather than sent as null: the parameter is typed
            # Optional[str] and an explicit None is not the same as absent.
            if container_id:
                request["container"] = container_id
            try:
                response = client.messages.create(**request)
            except Exception as e:  # noqa: BLE001 — classify API errors cleanly
                raise _classify_api_error(e) from e
            allocated = getattr(response, "container", None)
            if allocated is not None and getattr(allocated, "id", None):
                container_id = allocated.id
            u = getattr(response, "usage", None)
            if u is not None:
                usage["input_tokens"] = usage.get("input_tokens", 0) + (getattr(u, "input_tokens", 0) or 0)
                usage["output_tokens"] = usage.get("output_tokens", 0) + (getattr(u, "output_tokens", 0) or 0)
            tool_names = [b.name for b in response.content if getattr(b, "type", "") == "tool_use"]
            turn_dur = time.monotonic() - turn_start
            log.info(
                "anthropic-agent turn=%d stop=%s tools=%s dur=%.1fs in=%s out=%s cached=%s",
                turn, response.stop_reason, ",".join(tool_names) or "-",
                turn_dur,
                getattr(u, "input_tokens", "?"), getattr(u, "output_tokens", "?"),
                getattr(u, "cache_read_input_tokens", "?"),
            )
            if self.trace is not None:
                thinking = "".join(
                    getattr(b, "thinking", "") for b in response.content
                    if getattr(b, "type", "") == "thinking"
                )
                self.trace.step(
                    "model", f"turn-{turn}",
                    (
                        f"thinking + requested {', '.join(tool_names)}"
                        if tool_names else f"stop={response.stop_reason}"
                    ),
                    detail={
                        "turn": turn,
                        "stopReason": response.stop_reason,
                        "toolsRequested": tool_names,
                        "reasoning": thinking[:2000] or None,
                        "inputTokens": getattr(u, "input_tokens", None),
                        "outputTokens": getattr(u, "output_tokens", None),
                        "cachedInputTokens": getattr(u, "cache_read_input_tokens", None),
                    },
                    duration_seconds=turn_dur,
                )
            if response.stop_reason == "refusal":
                raise ProviderError("anthropic-agent: model refused the request")
            if response.stop_reason == "pause_turn":
                # Server-side tool work paused this turn; re-send to resume it.
                self._messages.append({"role": "assistant", "content": response.content})
                continue
            if response.stop_reason == "tool_use":
                self._messages.append({"role": "assistant", "content": response.content})
                results = [self._run_tool(b) for b in response.content if b.type == "tool_use"]
                self._messages.append({"role": "user", "content": results})
                continue
            if response.stop_reason == "max_tokens":
                # Handled apart from end_turn: this turn is INCOMPLETE, not
                # finished. The partial text is still the best diagnostic there
                # is (the engine's repair round re-asks for the whole Song and
                # usually gets it), so it falls through — but the turn can end
                # on a tool-use block whose result will never exist, which is
                # what _close_open_assistant_turn below removes.
                log.warning(
                    "anthropic-agent turn=%d was truncated at the max_tokens cap (%s); "
                    "the partial output goes to the repair round",
                    turn, request["max_tokens"],
                )
                break
            break  # end_turn (or an unknown reason): fall through with the text we have
        else:
            raise ProviderError("anthropic-agent: exceeded max turns without a final answer")

        # Keep the assistant's final answer in the history for any repair round,
        # then close it: a turn truncated at the token cap can carry a tool-use
        # block with no result, and every later request would be rejected. Doing
        # it here rather than only at the top of the next round is what answers a
        # client tool use in the SAME user turn the repair prompt joins; an open
        # server block in this now-trailing turn is still pause_turn-shaped, so
        # it waits for the append above to make it unpairable.
        self._messages.append({"role": "assistant", "content": response.content})
        self._close_open_assistant_turn()
        final_text = "".join(b.text for b in response.content if b.type == "text")
        return LLMResponse(
            text=final_text,
            provider=self.name,
            model=resolved_model,
            usage=usage,
        )
