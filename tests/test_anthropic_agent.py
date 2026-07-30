"""The in-process "anthropic-agent" reconciliation provider.

The agent loop runs INSIDE the server (Anthropic SDK + server-side search +
local chord-sheet/MIR tools). These tests drive it with a fully faked Anthropic
client — no network, no real `anthropic` objects — by scripting responses as
`types.SimpleNamespace` blocks and monkeypatching
`AnthropicAgentProvider._create_client`. They verify:
- the happy path flows a tool call and returns a validated Song,
- repair rounds continue the SAME conversation (full tool history preserved),
- a local tool that errors is fed back as an is_error tool_result,
- an unconfigured key and an over-long loop fail with clear ProviderErrors,
- provider_capabilities() tracks the key.
"""

from __future__ import annotations

import json
import pathlib
import types

import pytest

from snoocle_server.config import settings
from snoocle_server.mir.base import Beat, ChordSegment, MirAnalysis, StructureSegment
from snoocle_server.reconcile import reconcile
from snoocle_server.reconcile import anthropic_agent as agent_mod
from snoocle_server.reconcile.anthropic_agent import AnthropicAgentProvider
from snoocle_server.reconcile.providers import ProviderError, provider_capabilities

# The canonical valid Song DRAFT: the reference-protocol shape this provider
# must now emit (no lyric text). See snoocle_server/reconcile/lyric_refs.py.
from tests.fake_agent_mcp import _SONG_DRAFT

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"


# --- fake Anthropic client -------------------------------------------------


def _text(text: str):
    return types.SimpleNamespace(type="text", text=text)


def _tool_use(tool_id: str, name: str, tool_input: dict):
    return types.SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def _server_tool_use(tool_id: str, name: str = "web_search", block_type: str = "server_tool_use"):
    """A tool use the API runs itself (ids prefixed `srvtoolu_`).

    web_search / web_fetch execute in an API-managed code-execution container,
    so a turn can also carry that container's own blocks — block types with
    `code_execution` in them, e.g. `text_editor_code_execution`. Only the API
    can emit the matching `*_tool_result`, which is why an open one has to be
    dropped rather than answered.
    """
    return types.SimpleNamespace(type=block_type, id=tool_id, name=name, input={})


def _server_tool_result(tool_use_id: str, block_type: str = "web_search_tool_result"):
    """The API's own result for a server tool use. The loop treats these as
    arriving INLINE in an assistant turn — the same turn, or the turn that
    resumes a paused one."""
    return types.SimpleNamespace(type=block_type, tool_use_id=tool_use_id, content=[])


def _field(block, key: str, default=None):
    """Blocks are SimpleNamespace when scripted and dicts when the provider
    wrote them; read either shape."""
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _unpaired_tool_use_ids(messages: list) -> list:
    """Tool-use ids in an outgoing request that no `*_tool_result` answers.

    The API rejects the WHOLE request in that case — the live 400 was
    "text_editor_code_execution tool use with id srvtoolu_... was found without
    a corresponding text_editor_code_execution_tool_result block". A SERVER tool
    use in the TRAILING assistant message is exempt: that is the pause_turn
    shape, where re-sending the still-pending block is how the work resumes.
    """
    answered: set = set()
    uses: list = []
    for i, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, (list, tuple)):
            continue
        trailing_assistant = i == len(messages) - 1 and message.get("role") == "assistant"
        for block in content:
            btype = str(_field(block, "type", ""))
            if btype == "tool_result" or btype.endswith("_tool_result"):
                answered.add(_field(block, "tool_use_id"))
            elif btype == "tool_use":
                uses.append(_field(block, "id"))
            elif btype == "server_tool_use" or "code_execution" in btype:
                if not trailing_assistant:
                    uses.append(_field(block, "id"))
    return [tool_use_id for tool_use_id in uses if tool_use_id not in answered]


def _orphaned_tool_result_ids(messages: list) -> list:
    """`*_tool_result` blocks in an outgoing request that answer no tool use.

    The other direction of the same break, and the API rejects it too. Dropping
    an unpairable server tool use from a NON-trailing assistant turn is what can
    create one: the client `tool_use` that followed it goes with it, and the
    `tool_result` in a later message that answered THAT use is left naming
    nothing.
    """
    uses: set = set()
    orphans: list = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, (list, tuple)):
            continue
        for block in content:
            btype = str(_field(block, "type", ""))
            if btype == "tool_result" or btype.endswith("_tool_result"):
                if _field(block, "tool_use_id") not in uses:
                    orphans.append(_field(block, "tool_use_id"))
            elif btype == "tool_use" or btype == "server_tool_use" or "code_execution" in btype:
                uses.add(_field(block, "id"))
    return orphans


def _assert_pairable(messages: list) -> None:
    """Both directions of the tool-use/tool_result pairing, asserted on an
    outgoing request. Either break is a 400 that kills the whole run."""
    unpaired = _unpaired_tool_use_ids(messages)
    assert not unpaired, f"outgoing request carries unpaired tool-use id(s): {unpaired}"
    orphaned = _orphaned_tool_result_ids(messages)
    assert not orphaned, f"outgoing request carries orphaned tool_result id(s): {orphaned}"


def _response(stop_reason: str, content: list, in_tok: int = 100, out_tok: int = 40,
              container_id: str | None = None):
    return types.SimpleNamespace(
        stop_reason=stop_reason,
        content=content,
        usage=types.SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
        # The API returns this only on the turn that allocated the container for
        # the server-side tools; None on every other turn.
        container=(
            types.SimpleNamespace(id=container_id, expires_at=None)
            if container_id else None
        ),
    )


class _FakeClient:
    """Returns scripted responses from a SHARED queue (so repair rounds — which
    create a fresh client — keep consuming the same script in order). Snapshots
    the `messages` passed to each create() so tests can inspect them."""

    def __init__(self, queue: list, captured: dict):
        self._queue = queue
        self._captured = captured
        self.messages = self  # so client.messages.create(...) resolves here

    def create(self, **kwargs):
        messages = list(kwargs.get("messages") or [])
        # Invariant on EVERY request, in every scenario: the API 400s on an
        # assistant turn whose tool use nothing answers — and on a tool_result
        # that answers no tool use — so no scripted flow may ever produce one.
        _assert_pairable(messages)
        self._captured.setdefault("calls", []).append(messages)
        self._captured["last_kwargs"] = kwargs
        if not self._queue:
            raise AssertionError("fake Anthropic client ran out of scripted responses")
        return self._queue.pop(0)


def _install(monkeypatch, queue: list) -> dict:
    """Monkeypatch _create_client to hand out fake clients backed by `queue`.

    Returns a `captured` dict that ends up holding the provider instance and the
    per-create() message snapshots.
    """
    captured: dict = {}

    def _create(self):
        captured["provider"] = self
        return _FakeClient(queue, captured)

    monkeypatch.setattr(AnthropicAgentProvider, "_create_client", _create)
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    return captured


def _mir() -> MirAnalysis:
    return MirAnalysis(
        engines={"beats": "madmom", "chords": "chord-cnn-lstm", "structure": "songformer"},
        duration_seconds=243.0,
        bpm=73.5,
        time_signature="4/4",
        key="C major",
        beats=[Beat(time=0.8, position=1), Beat(time=1.6, position=2)],
        chords=[
            ChordSegment(start=13.1, end=15.2, chord="C"),
            ChordSegment(start=15.2, end=17.3, chord="G"),
        ],
        sections=[StructureSegment(start=13.0, end=25.0, label="verse")],
    )


def _tool_result_user_messages(msgs: list) -> list:
    """User messages whose content is a list of tool_result blocks."""
    out = []
    for m in msgs:
        content = m.get("content")
        if m.get("role") == "user" and isinstance(content, list):
            if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                out.append(m)
    return out


# --- scenario 1: happy path -------------------------------------------------


def _referencing_draft(source_id: str, line: int = 0) -> dict:
    """The `_SONG_DRAFT` shape with its first line pointing at a source
    instead of being instrumental — the normal case of the protocol."""
    draft = json.loads(json.dumps(_SONG_DRAFT))
    draft["lines"][0] = {
        "lineIndex": 0,
        "lyricRef": {"sourceId": source_id, "line": line},
        "chordPlacements": [
            {"charIndex": 0, "chord": "C"},
            {"charIndex": 15, "chord": "G"},
            {"charIndex": 32, "chord": "Am"},
        ],
    }
    return draft


def test_happy_path_tool_call_then_valid_song(monkeypatch):
    # fetch_chord_sheet reaches no network: fetch_page returns a real fixture
    # sheet, then the real extract/parse turn it into a candidate.
    sheet = (_FIXTURES / "sheet_over_lyrics.txt").read_text()
    fetched = {}

    def _fake_fetch_page(url: str) -> str:
        fetched["url"] = url
        return sheet

    monkeypatch.setattr(agent_mod, "fetch_page", _fake_fetch_page)

    queue = [
        _response("tool_use", [_tool_use("t1", "fetch_chord_sheet", {"url": "https://ex/let-it-be"})]),
        # The agent references the sheet it just fetched — mid-run sources are
        # registered as referenceable, or this would fail the run.
        _response("end_turn", [_text(json.dumps(_referencing_draft("agent-1")))]),
    ]
    captured = _install(monkeypatch, queue)

    result = reconcile(
        "Let It Be",
        "The Beatles",
        candidates=[],
        mir=_mir(),
        provider_name="anthropic-agent",
        youtube_video_id="QDYfEBY9NM4",
    )

    assert result.provider == "anthropic-agent"
    assert result.attempts == 1
    assert result.song.id == "the-beatles--let-it-be"
    assert result.song.metadata.title == "Let It Be"
    # accumulated token usage surfaced from response.usage
    assert result.usage.get("input_tokens", 0) > 0

    # the lyric the model never wrote: spliced from the sheet it fetched
    assert result.song.lines[0].lyrics.strip() == "When I find myself in times of trouble"

    # the fetch tool was actually invoked with the model's URL
    assert fetched["url"] == "https://ex/let-it-be"
    # and its tool_result went back in a user message
    assert _tool_result_user_messages(captured["provider"]._messages)

    # request shape: consolidated effort + per-turn prefix caching — the two
    # wall-clock levers for the loop (see config.anthropic_agent_effort)
    kwargs = captured["last_kwargs"]
    assert kwargs["output_config"] == {"effort": settings.anthropic_agent_effort}
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    assert kwargs["thinking"] == {"type": "adaptive"}
    for banned in ("temperature", "top_p", "top_k"):
        assert banned not in kwargs


# --- scenario 2: repair round continues the same conversation ---------------


def test_repair_round_continues_same_conversation(monkeypatch):
    # Uses analyze_audio_window (no audio -> harmless error result, no network)
    # so the focus is purely the repair-round conversation continuation.
    queue = [
        _response("tool_use", [_tool_use("t1", "analyze_audio_window", {"start_seconds": 5, "end_seconds": 15})]),
        _response("end_turn", [_text(json.dumps({"bad": True}))]),  # attempt 1: invalid
        _response("end_turn", [_text(json.dumps(_SONG_DRAFT))]),          # attempt 2 (repair): valid
    ]
    captured = _install(monkeypatch, queue)

    result = reconcile(
        "Let It Be",
        "The Beatles",
        candidates=[],
        mir=_mir(),
        provider_name="anthropic-agent",
        youtube_video_id="QDYfEBY9NM4",
    )

    assert result.attempts == 2
    msgs = captured["provider"]._messages
    # same conversation carried across rounds: two assistant answers + the
    # round-1 tool history are all still present (not a reset per attempt)
    assistant_turns = [m for m in msgs if m.get("role") == "assistant"]
    assert len(assistant_turns) >= 2
    assert _tool_result_user_messages(msgs)
    # the final create() saw the full prior history (grew beyond a single turn)
    assert len(captured["calls"][-1]) >= 4


# --- scenario 3: local tool error is fed back, loop still completes ----------


def test_analyze_audio_window_without_audio_is_error_but_completes(monkeypatch):
    queue = [
        _response("tool_use", [_tool_use("t1", "analyze_audio_window", {"start_seconds": 10, "end_seconds": 20})]),
        _response("end_turn", [_text(json.dumps(_SONG_DRAFT))]),
    ]
    captured = _install(monkeypatch, queue)

    result = reconcile(
        "Let It Be",
        "The Beatles",
        candidates=[],
        mir=_mir(),
        provider_name="anthropic-agent",
        youtube_video_id="QDYfEBY9NM4",
        # no audio_path -> analyze_audio_window returns an error object
    )

    assert result.song.id == "the-beatles--let-it-be"
    tr_msgs = _tool_result_user_messages(captured["provider"]._messages)
    assert tr_msgs
    block = tr_msgs[0]["content"][0]
    assert block["is_error"] is True
    assert "no audio available" in block["content"]


# --- scenario 4: unconfigured key -------------------------------------------


def test_unconfigured_key_is_clear_error(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    with pytest.raises(ProviderError, match="SNOOCLE_ANTHROPIC_API_KEY"):
        reconcile(
            "Let It Be", "The Beatles", candidates=[], mir=_mir(),
            provider_name="anthropic-agent",
        )


# --- scenario 5: max turns exceeded -----------------------------------------


class _AlwaysToolUse:
    def __init__(self):
        self.messages = self

    def create(self, **kwargs):
        _assert_pairable(list(kwargs.get("messages") or []))
        # never emits a final answer — always asks for another tool call
        return _response(
            "tool_use",
            [_tool_use("t", "analyze_audio_window", {"start_seconds": 0, "end_seconds": 5})],
        )


def test_max_turns_exceeded_raises(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(settings, "anthropic_agent_max_turns", 3)
    monkeypatch.setattr(AnthropicAgentProvider, "_create_client", lambda self: _AlwaysToolUse())

    with pytest.raises(ProviderError, match="exceeded max turns"):
        reconcile(
            "Let It Be", "The Beatles", candidates=[], mir=_mir(),
            provider_name="anthropic-agent",
        )


# --- scenario 6: capabilities track the key ---------------------------------


def test_anthropic_agent_in_provider_capabilities(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    caps = provider_capabilities()
    assert "anthropic-agent" in caps
    assert caps["anthropic-agent"]["configured"] is False
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    assert provider_capabilities()["anthropic-agent"]["configured"] is True


# --- server-side tool containers ---------------------------------------------
#
# web_search / web_fetch run in an API-managed container. The id is handed back
# ONCE, on the turn that allocated it, and every later turn of the same
# conversation must repeat it. Getting this wrong produced a live 400 —
# "container_id is required when there are pending tool uses generated by code
# execution with tools" — that killed a reconciliation after the notes and
# scope steps had already succeeded.


def _create_kwargs(captured: dict) -> list[dict]:
    return captured.setdefault("kwargs_log", [])


class _RecordingClient(_FakeClient):
    """Like _FakeClient but keeps EVERY create() kwargs, not just the last."""

    def create(self, **kwargs):
        self._captured.setdefault("kwargs_log", []).append(kwargs)
        return super().create(**kwargs)


def _install_recording(monkeypatch, queue: list) -> dict:
    captured: dict = {}

    def _create(self):
        captured["provider"] = self
        return _RecordingClient(queue, captured)

    monkeypatch.setattr(AnthropicAgentProvider, "_create_client", _create)
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    return captured


def test_container_id_is_carried_into_every_later_turn(monkeypatch):
    """The turn after a container is allocated must name it, and so must the
    one after that — the id is not re-sent by the API."""
    sheet = (_FIXTURES / "sheet_over_lyrics.txt").read_text()
    monkeypatch.setattr(agent_mod, "fetch_page", lambda url: sheet)

    queue = [
        # turn 1 allocates the container while requesting a tool
        _response("tool_use", [_tool_use("t1", "fetch_chord_sheet", {"url": "https://ex/a"})],
                  container_id="ctr_abc123"),
        # turn 2 asks for another tool; no container echoed back this time
        _response("tool_use", [_tool_use("t2", "fetch_chord_sheet", {"url": "https://ex/b"})]),
        _response("end_turn", [_text(json.dumps(_SONG_DRAFT))]),
    ]
    captured = _install_recording(monkeypatch, queue)

    reconcile("Let It Be", "The Beatles", candidates=[], mir=_mir(),
              provider_name="anthropic-agent", youtube_video_id="QDYfEBY9NM4")

    calls = _create_kwargs(captured)
    assert len(calls) == 3
    assert "container" not in calls[0], "nothing to name before one is allocated"
    assert calls[1]["container"] == "ctr_abc123"
    assert calls[2]["container"] == "ctr_abc123", (
        "the id must persist across turns — the API only sends it once"
    )


def test_no_container_key_is_sent_when_none_was_allocated(monkeypatch):
    """An explicit null is not the same as an absent field: the parameter is
    typed Optional[str] and sending None would be a different request."""
    queue = [_response("end_turn", [_text(json.dumps(_SONG_DRAFT))])]
    captured = _install_recording(monkeypatch, queue)

    reconcile("Let It Be", "The Beatles", candidates=[], mir=_mir(),
              provider_name="anthropic-agent", youtube_video_id="QDYfEBY9NM4")

    for call in _create_kwargs(captured):
        assert "container" not in call


# --- scope must remove tools, not just ask ------------------------------------
#
# The payload carries a scopeInstruction, but an instruction is a request. On a
# live run with listen=off the model went on calling analyze_audio_window and
# web_search anyway, then ground into the 900s reconcile timeout. Tools the
# scope switched off must be UNDECLARED, which makes them uncallable.


def _tool_names(captured: dict) -> list[str]:
    tools = _create_kwargs(captured)[0].get("tools", [])
    return [t["name"] for t in tools]


def _reconcile_with_scope(monkeypatch, scope, queue):
    from snoocle_server.scope import AnalysisScope

    captured = _install_recording(monkeypatch, queue)
    reconcile("Let It Be", "The Beatles", candidates=[], mir=_mir(),
              provider_name="anthropic-agent", youtube_video_id="QDYfEBY9NM4",
              scope=scope)
    return captured


def test_listen_off_removes_the_audio_window_tool(monkeypatch):
    from snoocle_server.scope import AnalysisScope

    captured = _reconcile_with_scope(
        monkeypatch, AnalysisScope(listen=False, reconcile=True),
        [_response("end_turn", [_text(json.dumps(_SONG_DRAFT))])],
    )
    names = _tool_names(captured)
    assert "analyze_audio_window" not in names, "listening was switched off"
    assert "web_search" in names, "source gathering was left on"


def test_reconcile_off_removes_every_source_gathering_tool(monkeypatch):
    from snoocle_server.scope import AnalysisScope

    captured = _reconcile_with_scope(
        monkeypatch, AnalysisScope(listen=True, reconcile=False),
        [_response("end_turn", [_text(json.dumps(_SONG_DRAFT))])],
    )
    names = _tool_names(captured)
    for gone in ("web_search", "web_fetch", "fetch_chord_sheet"):
        assert gone not in names, f"{gone} should be undeclared"
    assert "analyze_audio_window" in names, "listening was left on"


def test_notes_only_sends_no_tools_key_at_all(monkeypatch):
    """Every tool is off, and an empty list is not the same as no list."""
    from snoocle_server.scope import AnalysisScope

    captured = _reconcile_with_scope(
        monkeypatch, AnalysisScope(listen=False, reconcile=False),
        [_response("end_turn", [_text(json.dumps(_SONG_DRAFT))])],
    )
    call = _create_kwargs(captured)[0]
    assert "tools" not in call


# --- a turn the API cut short must not be carried open -------------------------
#
# stop_reason "max_tokens" ends a turn MID-generation: the model may already have
# emitted a tool-use block whose result will never exist. Committing that turn to
# history and then appending the engine's repair prompt after it produced a live
# 400 — "text_editor_code_execution tool use with id srvtoolu_... was found
# without a corresponding text_editor_code_execution_tool_result block" — at
# messages.1, i.e. the first assistant turn of the first attempt.
#
# messages.1 is not the LAST assistant turn once anything follows it, so every
# assistant message has to be checked, not just the last. The single exemption is
# a server tool use in a genuinely trailing turn: that is the pause_turn resume
# shape, where re-sending the open block is how the paused work continues.


def test_truncated_turn_is_not_carried_open_into_the_repair_round(monkeypatch):
    queue = [
        # attempt 1: cut off mid-JSON, on a server-side tool use with no result
        _response(
            "max_tokens",
            [
                _text('{"lines": ['),
                _server_tool_use(
                    "srvtoolu_1", "text_editor_code_execution",
                    block_type="text_editor_code_execution",
                ),
            ],
        ),
        # attempt 2 (the engine's repair round): a complete Song
        _response("end_turn", [_text(json.dumps(_SONG_DRAFT))]),
    ]
    captured = _install(monkeypatch, queue)

    result = reconcile("Let It Be", "The Beatles", candidates=[], mir=_mir(),
                       provider_name="anthropic-agent", youtube_video_id="QDYfEBY9NM4")

    # truncation is repaired, not fatal: the repair round produced the Song
    assert result.attempts == 2
    for msgs in captured["calls"]:
        assert not _unpaired_tool_use_ids(msgs)

    stored = captured["provider"]._messages
    truncated_turn = [m for m in stored if m.get("role") == "assistant"][0]
    # the partial text survives (it is the engine's diagnostic); the block only
    # the API could have answered is gone
    assert [_field(b, "type") for b in truncated_turn["content"]] == ["text"]


def test_truncated_turn_with_an_open_client_tool_use_is_answered_not_dropped(monkeypatch):
    """A LOCAL tool use is ours to answer, so the block stays and gets a
    tool_result — dropping it would throw away the model's reasoning."""
    queue = [
        _response(
            "max_tokens",
            [
                _text('{"lines": ['),
                _tool_use("t1", "analyze_audio_window", {"start_seconds": 0, "end_seconds": 5}),
            ],
        ),
        _response("end_turn", [_text(json.dumps(_SONG_DRAFT))]),
    ]
    captured = _install(monkeypatch, queue)

    result = reconcile("Let It Be", "The Beatles", candidates=[], mir=_mir(),
                       provider_name="anthropic-agent", youtube_video_id="QDYfEBY9NM4")

    assert result.attempts == 2
    for msgs in captured["calls"]:
        assert not _unpaired_tool_use_ids(msgs)

    stored = captured["provider"]._messages
    truncated_turn = [m for m in stored if m.get("role") == "assistant"][0]
    assert [_field(b, "type") for b in truncated_turn["content"]] == ["text", "tool_use"]
    stubs = [
        b for m in stored if isinstance(m.get("content"), list)
        for b in m["content"] if _field(b, "type") == "tool_result"
    ]
    assert [b["tool_use_id"] for b in stubs] == ["t1"]
    assert stubs[0]["is_error"] is True
    assert "cut off" in stubs[0]["content"]
    # results lead the user turn they answer; the repair prompt joins it as text
    answering_turn = stored[stored.index(truncated_turn) + 1]
    assert answering_turn["role"] == "user"
    assert [_field(b, "type") for b in answering_turn["content"]] == ["tool_result", "text"]


def test_an_open_client_use_is_answered_inside_the_user_turn_that_follows_it():
    """A stub result has to LEAD the user turn that answers its assistant
    message — including when that turn is a plain-text one (the engine's repair
    prompt), which becomes a text block rather than a second user message."""
    provider = AnthropicAgentProvider()
    provider._messages = [
        {"role": "user", "content": "{}"},
        {
            "role": "assistant",
            "content": [
                _text("partial"),
                _tool_use("t1", "analyze_audio_window", {"start_seconds": 0, "end_seconds": 5}),
            ],
        },
        {"role": "user", "content": "that JSON failed validation; send the whole Song"},
    ]

    provider._close_open_assistant_turn()

    assert len(provider._messages) == 3, "no second user message was opened"
    assert [_field(b, "type") for b in provider._messages[2]["content"]] == [
        "tool_result", "text",
    ]
    assert not _unpaired_tool_use_ids(provider._messages)


def test_closing_an_open_turn_is_inert_on_a_clean_conversation(monkeypatch):
    """Nothing was left open, so the stored history must come out identical —
    same dicts, same lists, same bytes — however many times it runs."""
    sheet = (_FIXTURES / "sheet_over_lyrics.txt").read_text()
    monkeypatch.setattr(agent_mod, "fetch_page", lambda url: sheet)
    queue = [
        _response("tool_use", [_tool_use("t1", "fetch_chord_sheet", {"url": "https://ex/a"})]),
        _response("end_turn", [_text(json.dumps(_referencing_draft("agent-1")))]),
    ]
    captured = _install(monkeypatch, queue)

    reconcile("Let It Be", "The Beatles", candidates=[], mir=_mir(),
              provider_name="anthropic-agent", youtube_video_id="QDYfEBY9NM4")

    provider = captured["provider"]
    before = repr(provider._messages)
    identities = [(id(m), id(m["content"])) for m in provider._messages]

    provider._close_open_assistant_turn()
    provider._close_open_assistant_turn()  # idempotent

    assert repr(provider._messages) == before
    assert [(id(m), id(m["content"])) for m in provider._messages] == identities


def test_a_server_tool_use_answered_inline_is_left_alone():
    """The drop rule is keyed on PAIRING, not on the block being server-side:
    web_search's result comes back inline in the same assistant turn."""
    provider = AnthropicAgentProvider()
    provider._messages = [
        {"role": "user", "content": "{}"},
        {
            "role": "assistant",
            "content": [
                _server_tool_use("srvtoolu_1", "web_search"),
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []},
                _text("...and here is the Song"),
            ],
        },
    ]
    before = repr(provider._messages)

    provider._close_open_assistant_turn()

    assert repr(provider._messages) == before


def test_a_turn_that_was_only_an_open_server_block_keeps_a_marker():
    """Dropping the block must not leave empty assistant content, which the API
    also rejects. The turn is followed by the engine's repair prompt here, which
    is what makes its open block unpairable — the same block in a TRAILING turn
    is the pause_turn shape and is kept (next test)."""
    provider = AnthropicAgentProvider()
    provider._messages = [
        {"role": "user", "content": "{}"},
        {"role": "assistant", "content": [_server_tool_use("srvtoolu_1", "web_fetch")]},
        {"role": "user", "content": "that JSON failed validation; send the whole Song"},
    ]

    provider._close_open_assistant_turn()

    assert provider._messages[1]["content"] == [
        {"type": "text", "text": agent_mod._TRUNCATED_TURN_NOTE}
    ]
    assert not _unpaired_tool_use_ids(provider._messages)


def test_a_pending_server_block_in_the_trailing_turn_is_kept_for_the_resume():
    """The one legitimately pending block. A pause_turn turn is committed with
    its server tool use still open and re-sent AS IS — that is how the API
    resumes the work — so nothing may drop it while it is still trailing."""
    provider = AnthropicAgentProvider()
    provider._messages = [
        {"role": "user", "content": "{}"},
        {"role": "assistant", "content": [_server_tool_use("srvtoolu_1", "web_search")]},
    ]
    before = repr(provider._messages)

    provider._close_open_assistant_turn()

    assert repr(provider._messages) == before


def test_a_paused_turn_is_resumed_and_its_inline_result_keeps_it_paired(monkeypatch):
    """End to end: the paused turn goes back out with its open block, and when
    the resumption answers it inline — the shape this loop assumes — the turn is
    still paired once it is no longer trailing, so the paused work is kept
    rather than dropped."""
    queue = [
        _response("pause_turn", [_server_tool_use("srvtoolu_1", "web_search")],
                  container_id="ctr_1"),
        _response(
            "end_turn",
            [_server_tool_result("srvtoolu_1"), _text(json.dumps(_SONG_DRAFT))],
        ),
    ]
    captured = _install(monkeypatch, queue)

    result = reconcile("Let It Be", "The Beatles", candidates=[], mir=_mir(),
                       provider_name="anthropic-agent", youtube_video_id="QDYfEBY9NM4")

    assert result.attempts == 1
    # the resume request re-sent the still-open block
    resume = captured["calls"][1]
    assert [_field(b, "id") for b in resume[1]["content"]] == ["srvtoolu_1"]
    for msgs in captured["calls"]:
        _assert_pairable(msgs)
    # and the paused turn survives in the history, block intact
    stored = captured["provider"]._messages
    assert [_field(b, "type") for b in stored[1]["content"]] == ["server_tool_use"]


def test_a_paused_turn_left_open_by_a_truncated_resumption_is_closed(monkeypatch):
    """The reported incident's shape. The paused turn is committed with an open
    server block, the RESUMPTION is cut at the token cap before the container's
    result arrives, and the repair round then re-sends the paused turn as
    messages.1 — where it is neither trailing nor answerable."""
    queue = [
        _response("pause_turn", [_server_tool_use("srvtoolu_1", "web_search")],
                  container_id="ctr_1"),
        _response("max_tokens", [_text('{"lines": [')]),
        _response("end_turn", [_text(json.dumps(_SONG_DRAFT))]),
    ]
    captured = _install(monkeypatch, queue)

    result = reconcile("Let It Be", "The Beatles", candidates=[], mir=_mir(),
                       provider_name="anthropic-agent", youtube_video_id="QDYfEBY9NM4")

    assert result.attempts == 2
    assert len(captured["calls"]) == 3
    # the repair request really does carry the paused turn at messages.1
    assert captured["calls"][2][1]["role"] == "assistant"
    for msgs in captured["calls"]:
        _assert_pairable(msgs)
    stored = captured["provider"]._messages
    assert stored[1]["content"] == [{"type": "text", "text": agent_mod._TRUNCATED_TURN_NOTE}]


def test_an_open_server_block_on_a_tool_use_turn_is_closed_before_the_next_send(monkeypatch):
    """A turn can stop with `tool_use` while ALSO carrying an open server block.
    The loop appends that turn and then the client results, so it stops being
    trailing immediately and no later step revisits it — the next request has to
    be the one that is already clean."""
    queue = [
        _response(
            "tool_use",
            [
                _server_tool_use("srvtoolu_9", "text_editor_code_execution",
                                 block_type="text_editor_code_execution"),
                _tool_use("t1", "analyze_audio_window", {"start_seconds": 0, "end_seconds": 5}),
            ],
        ),
        _response("end_turn", [_text(json.dumps(_SONG_DRAFT))]),
    ]
    captured = _install(monkeypatch, queue)

    result = reconcile("Let It Be", "The Beatles", candidates=[], mir=_mir(),
                       provider_name="anthropic-agent", youtube_video_id="QDYfEBY9NM4")

    assert result.song.id == "the-beatles--let-it-be"
    for msgs in captured["calls"]:
        _assert_pairable(msgs)
    stored = captured["provider"]._messages
    # the open server block took the client tool_use after it — and that use's
    # result, which would otherwise name a tool_use the request no longer has
    assert stored[1]["content"] == [{"type": "text", "text": agent_mod._TRUNCATED_TURN_NOTE}]
    assert stored[2]["content"] == [{"type": "text", "text": agent_mod._ORPHANED_RESULT_NOTE}]


def test_truncating_a_non_trailing_turn_drops_only_the_results_it_orphans():
    """Truncating a turn in the MIDDLE of the history changes what follows it:
    a tool_result answering a client tool_use inside the dropped region becomes
    an orphan in the other direction. Results for kept uses stay."""
    provider = AnthropicAgentProvider()
    provider._messages = [
        {"role": "user", "content": "{}"},
        {
            "role": "assistant",
            "content": [
                _text("partial"),
                _tool_use("t0", "fetch_chord_sheet", {"url": "https://ex/a"}),
                _server_tool_use("srvtoolu_1", "web_search"),
                _tool_use("t1", "analyze_audio_window", {"start_seconds": 0, "end_seconds": 5}),
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t0", "content": "{}"},
                {"type": "tool_result", "tool_use_id": "t1", "content": "{}"},
            ],
        },
        {"role": "assistant", "content": [_text("...")]},
    ]

    provider._close_open_assistant_turn()

    assert [_field(b, "type") for b in provider._messages[1]["content"]] == ["text", "tool_use"]
    assert [b["tool_use_id"] for b in provider._messages[2]["content"]] == ["t0"]
    assert not _unpaired_tool_use_ids(provider._messages)
    assert not _orphaned_tool_result_ids(provider._messages)


def test_absent_scope_leaves_every_tool_declared(monkeypatch):
    captured = _install_recording(monkeypatch, [_response("end_turn", [_text(json.dumps(_SONG_DRAFT))])])
    reconcile("Let It Be", "The Beatles", candidates=[], mir=_mir(),
              provider_name="anthropic-agent", youtube_video_id="QDYfEBY9NM4")
    names = _tool_names(captured)
    assert {"web_search", "web_fetch", "fetch_chord_sheet", "analyze_audio_window"} <= set(names)
