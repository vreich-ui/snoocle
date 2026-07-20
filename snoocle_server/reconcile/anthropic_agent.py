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
from ..mir.pipeline import analyze_window
from .providers import LLMProvider, LLMResponse, ProviderError
from .trace import TraceRecorder

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are an expert music transcriber. Your job is to produce one Snoocle Song \
JSON object that faithfully captures a song's chords, lyrics, and sections.

Evidence rules:
- The MIR chord timeline (derived from the ACTUAL audio recording) is the \
primary evidence for each chord's ROOT and its major/minor QUALITY.
- Text sources (chord sheets, lyric pages) are the primary evidence for \
lyrics, section structure, and chord EXTENSIONS (7ths, 9ths, sus, add).
- A source that declares a capo is written N semitones BELOW sounding pitch. \
Transpose it up N semitones before comparing it to the audio or to other \
sources.

Music theory:
- Prefer chord readings that are diatonic to the established key, or that are \
classically explainable (secondary dominants, borrowed iv or bVII, the \
relative major/minor). When the audio is ambiguous, let theory break the tie.
- Spell enharmonics according to the key signature: F#m in A major, never Gbm.

Hard chord rule:
- Every chord symbol you emit is the SOUNDING harmony. NEVER write fretboard \
shapes or tab fingerings, and NEVER bake a capo into the chord names. \
displayPreferences.capo MUST be 0.

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
specific passage AND the provided MIR timeline does not cover it.

Hard budget: at most 2 web_search calls, 4 page fetches, and 2 \
analyze_audio_window calls per song. Disagreements are settled by the MIR \
timeline and music theory — NOT by more searching. When you have enough \
information to act, act: produce the Song instead of continuing to verify. \
Two agreeing sources plus the provided MIR is always enough.

Output contract:
- Your FINAL message must be EXACTLY ONE JSON object — the Song — that \
validates against the provided songSchema. No markdown fences, no commentary, \
no prose before or after. The schema is strict: only its keys are allowed. \
Set id, title, artist, and youtubeVideoId from the request.
"""

# The system block is stable across loop turns and repair rounds; caching it
# lets every subsequent request reuse the prefix.
SYSTEM_BLOCKS = [
    {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
]

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


def _build_tools(max_web_search: int, max_fetch: int) -> list[dict]:
    """Tools with the server-tool budget set by the analysis-depth profile."""
    return [
        {"type": "web_search_20260209", "name": "web_search", "max_uses": max_web_search},
        {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": max_fetch},
        _FETCH_TOOL,
        _WINDOW_TOOL,
    ]


# Default budget (used when no depth profile is injected) mirrors "standard".
TOOLS = _build_tools(2, 3)

# Cap for windowed on-demand analysis (seconds) — see the tool description.
_MAX_WINDOW_SECONDS = 60.0


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
        analysis = analyze_window(audio_path, start, end)
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
        if depth is not None:
            payload["toolBudget"] = {
                "webSearch": depth.max_web_search,
                "pageFetch": depth.max_fetch,
                "audioWindow": depth.max_windows,
            }
            if depth.time_align:
                payload["fillSyncMap"] = (
                    "Thorough analysis: also populate audio.syncMap (lineIndex -> "
                    "seconds) from the MIR section boundaries and beat grid, at "
                    "least one entry per section. Times must be non-decreasing."
                )
        if ctx.get("prior_song") is not None:
            payload["priorHumanEditedSong"] = ctx["prior_song"]
        if ctx.get("guidance"):
            payload["humanCorrectionNotes"] = ctx["guidance"]
        return {"role": "user", "content": json.dumps(payload)}

    def _run_tool(self, block) -> dict:
        name = block.name
        tool_input = block.input or {}
        if name == "fetch_chord_sheet":
            self._fetch_count += 1
            result = fetch_chord_sheet(
                tool_input.get("url", ""), source_id=f"agent-{self._fetch_count}"
            )
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

    def complete(self, system, turns, model=None, max_tokens=None, audio=None):
        if not settings.anthropic_api_key:
            raise ProviderError("anthropic-agent: SNOOCLE_ANTHROPIC_API_KEY is not configured")
        if not self.context:
            raise ProviderError("anthropic-agent provider requires engine-injected context")

        # explicit model arg -> SNOOCLE_LLM_MODEL -> SNOOCLE_ANTHROPIC_AGENT_MODEL
        resolved_model = model or settings.llm_model or settings.anthropic_agent_model

        # Depth profile (injected by the engine) sets effort + the tool budget;
        # fall back to the server defaults when absent.
        depth = (self.context or {}).get("depth")
        effort = depth.effort if depth is not None else settings.anthropic_agent_effort
        tools = (
            _build_tools(depth.max_web_search, depth.max_fetch) if depth is not None else TOOLS
        )

        if len(turns) == 1:
            # First attempt: build a fresh conversation from the injected context.
            self._messages = [self._build_first_user_message()]
            self._fetch_count = 0
        else:
            # Repair round: the engine passed [user, assistant, repair-user, ...].
            # The assistant's previous final answer is already in self._messages;
            # append only the new repair prompt and continue the same loop.
            self._messages.append({"role": "user", "content": turns[-1]["text"]})

        client = self._create_client()
        usage: dict = {}
        response = None
        for turn in range(1, settings.anthropic_agent_max_turns + 1):
            turn_start = time.monotonic()
            response = client.messages.create(
                model=resolved_model,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                # effort is the dominant wall-clock lever for this loop; the
                # analysis-depth profile sets it (see reconcile/depth.py).
                output_config={"effort": effort},
                # auto-cache the latest prefix so each turn reuses the whole
                # prior conversation (system block carries its own breakpoint).
                cache_control={"type": "ephemeral"},
                system=SYSTEM_BLOCKS,
                tools=tools,
                # no temperature/top_p/top_k: sampling params are rejected here
                messages=self._messages,
            )
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
            break  # end_turn / max_tokens: fall through with whatever text we have
        else:
            raise ProviderError("anthropic-agent: exceeded max turns without a final answer")

        # Keep the assistant's final answer in the history for any repair round.
        self._messages.append({"role": "assistant", "content": response.content})
        final_text = "".join(b.text for b in response.content if b.type == "text")
        return LLMResponse(
            text=final_text,
            provider=self.name,
            model=resolved_model,
            usage=usage,
        )
