"""The lyric-reference protocol: the model points, the server splices.

A valid Song document IS the complete lyrics of a copyrighted song, so asking
a model to emit one is asking for a full verbatim reproduction — which is what
blocked two live runs today under Anthropic's content filter, both with four
sources successfully fetched. These tests pin the structural fix: the model
emits references into sources the run already has, deterministic code
substitutes the text, and a lyric with no valid provenance cannot reach the
store.

The four rules being enforced (see snoocle_server/reconcile/lyric_refs.py):
retyped lyrics are rejected, charIndex is validated against the RESOLVED text,
unresolvable refs fail the run outright, and overrides are audited and capped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snoocle_server.discovery.service import candidate_from_text
from snoocle_server.reconcile import reconcile
from snoocle_server.reconcile.engine import LyricProvenanceError, ReconcileError
from snoocle_server.reconcile.lyric_refs import (
    MAX_OVERRIDE_SHARE,
    PRIOR_SONG_SOURCE_ID,
    LyricSpliceError,
    UnresolvableLyricRefError,
    agent_song_json_schema,
    build_ref_index,
    splice_lyrics,
)
from snoocle_server.reconcile.providers import _PROVIDERS, LLMProvider, LLMResponse
from snoocle_server.schema import song_json_schema

FIXTURES = Path(__file__).parent / "fixtures"

LINE_0 = "When I find myself in times of trouble    "
LINE_1 = "Mother Mary comes to me            "


@pytest.fixture()
def candidate():
    c = candidate_from_text(
        (FIXTURES / "sheet_over_lyrics.txt").read_text(), "web-1", url="https://a.example/x"
    )
    assert c is not None
    return c


@pytest.fixture()
def index(candidate):
    return build_ref_index([candidate])


def _draft(*lines: dict, song_id: str = "the-beatles--let-it-be") -> dict:
    return {
        "id": song_id,
        "metadata": {"title": "Let It Be", "artist": "The Beatles"},
        "lines": [dict(line, lineIndex=i) for i, line in enumerate(lines)],
        "provenance": [],
    }


def _ref_line(line: int, *placements: tuple[int, str], source: str = "web-1") -> dict:
    return {
        "lyricRef": {"sourceId": source, "line": line},
        "chordPlacements": [{"charIndex": c, "chord": ch} for c, ch in placements],
    }


# --- the normal case -------------------------------------------------------------


def test_a_reference_becomes_the_sources_text_verbatim(index):
    result = splice_lyrics(_draft(_ref_line(0, (0, "C")), _ref_line(1, (0, "C"))), index)

    assert [line["lyrics"] for line in result.document["lines"]] == [LINE_0, LINE_1]
    assert result.refs_resolved == 2
    assert result.overrides == ()


def test_protocol_fields_never_reach_the_stored_document(index):
    result = splice_lyrics(
        _draft(
            _ref_line(0, (0, "C")),
            {"lyricOverride": "a line no source has", "lyricOverrideReason": "gap"},
        ),
        index,
    )
    serialized = json.dumps(result.document)
    for field in ("lyricRef", "lyricOverride", "lyricOverrideReason", "sourceId"):
        assert field not in serialized, f"{field} leaked into the stored document"


def test_an_instrumental_line_keeps_its_deliberate_empty_lyrics(index):
    """`lyrics: ""` is a statement, not a missing reference — and its ordinal
    charIndex slots have no text to run past."""
    result = splice_lyrics(
        _draft({"lyrics": "", "chordPlacements": [
            {"charIndex": 0, "chord": "Am"}, {"charIndex": 1, "chord": "G"},
            {"charIndex": 2, "chord": "F"}, {"charIndex": 3, "chord": "C"}]}),
        index,
    )
    assert result.document["lines"][0]["lyrics"] == ""
    assert result.instrumental == 1
    assert result.refs_resolved == 0


def test_the_prior_song_is_a_referenceable_source(candidate):
    """Without this, a notes-only run — which gathers no candidates by design
    and is asked to return the user's own document — could only comply by
    retyping every line, tripping the override cap on every run."""
    prior = {"lines": [{"lineIndex": 0, "lyrics": "the user's own words"}]}
    index = build_ref_index([candidate], prior)
    assert PRIOR_SONG_SOURCE_ID in index

    result = splice_lyrics(
        _draft(_ref_line(0, (0, "C"), source=PRIOR_SONG_SOURCE_ID)), index
    )
    assert result.document["lines"][0]["lyrics"] == "the user's own words"


# --- rule 1: retyped lyrics are rejected ------------------------------------------


def test_retyped_lyrics_are_rejected_and_repairable(index):
    with pytest.raises(LyricSpliceError) as exc:
        splice_lyrics(
            _draft({"lyrics": "When I find myself in times of trouble",
                    "chordPlacements": [{"charIndex": 0, "chord": "C"}]}),
            index,
        )
    assert "line 0" in str(exc.value)
    assert "lyricRef" in str(exc.value), "the error must say what to do instead"


def test_a_line_that_says_nothing_about_its_words_is_rejected(index):
    with pytest.raises(LyricSpliceError, match="line 0"):
        splice_lyrics(_draft({"chordPlacements": []}), index)


def test_a_line_cannot_carry_two_of_the_three(index):
    with pytest.raises(LyricSpliceError, match="exactly ONE"):
        splice_lyrics(
            _draft({
                "lyricRef": {"sourceId": "web-1", "line": 0},
                "lyricOverride": "something else",
                "lyricOverrideReason": "why",
                "chordPlacements": [],
            }),
            index,
        )


# --- rule 2: charIndex is validated against the RESOLVED text ---------------------


def test_a_charindex_past_the_resolved_line_is_named_and_repairable(index):
    with pytest.raises(LyricSpliceError) as exc:
        splice_lyrics(_draft(_ref_line(1, (0, "C"), (99, "G"))), index)
    message = str(exc.value)
    assert "line 0" in message
    assert "99" in message and str(len(LINE_1)) in message
    assert "'web-1' line 1" in message, "the error must name the resolved source"


def test_a_charindex_error_is_never_silently_clamped(index):
    """A chord moved to fit is a wrong transcription that looks right."""
    with pytest.raises(LyricSpliceError):
        splice_lyrics(_draft(_ref_line(1, (500, "C"))), index)


def test_non_ascending_charindexes_are_repairable(index):
    with pytest.raises(LyricSpliceError, match="strictly ascending"):
        splice_lyrics(_draft(_ref_line(0, (10, "C"), (4, "G"))), index)


def test_a_charindex_at_exactly_the_line_length_is_allowed(index):
    """== len(lyrics) means "after the last character" — the Song schema
    allows it and so must this pass, or valid documents get bounced."""
    result = splice_lyrics(_draft(_ref_line(1, (len(LINE_1), "C"))), index)
    assert result.document["lines"][0]["lyrics"] == LINE_1


# --- rule 3: unresolvable refs fail the run --------------------------------------


def test_an_unknown_source_id_fails_rather_than_repairs(index):
    with pytest.raises(UnresolvableLyricRefError) as exc:
        splice_lyrics(_draft(_ref_line(0, source="ultimate-guitar-1087597")), index)
    message = str(exc.value)
    assert "ultimate-guitar-1087597" in message, "the error must identify the ref"
    assert "web-1" in message, "...and say what WAS available"


def test_a_line_index_out_of_range_fails_rather_than_repairs(index):
    with pytest.raises(UnresolvableLyricRefError) as exc:
        splice_lyrics(_draft(_ref_line(999)), index)
    assert "999" in str(exc.value) and "web-1" in str(exc.value)


def test_a_malformed_ref_is_repairable_rather_than_fatal(index):
    """Malformed is not the same as unresolvable: the model can fix a shape
    mistake, and the repair loop is cheaper than failing the whole run."""
    with pytest.raises(LyricSpliceError):
        splice_lyrics(_draft({"lyricRef": "web-1:0", "chordPlacements": []}), index)
    with pytest.raises(LyricSpliceError):
        splice_lyrics(
            _draft({"lyricRef": {"sourceId": "web-1"}, "chordPlacements": []}), index
        )


# --- rule 4: overrides are audited and capped ------------------------------------


def test_an_override_is_recorded_with_its_line_and_reason(index):
    result = splice_lyrics(
        _draft(
            _ref_line(0, (0, "C")),
            _ref_line(1, (0, "C")),
            _ref_line(2, (0, "C")),
            _ref_line(3, (0, "C")),
            _ref_line(4, (0, "C")),
            _ref_line(5, (0, "C")),
            {"lyricOverride": "merged from two partial sheets",
             "lyricOverrideReason": "web-1 and web-2 each cover half of this line",
             "chordPlacements": [{"charIndex": 0, "chord": "G"}]},
        ),
        index,
    )
    assert result.document["lines"][6]["lyrics"] == "merged from two partial sheets"
    assert len(result.overrides) == 1
    assert result.overrides[0].line_index == 6
    assert "each cover half" in result.overrides[0].reason
    assert result.refs_resolved == 6


def test_an_override_without_a_reason_is_rejected(index):
    with pytest.raises(LyricSpliceError, match="lyricOverrideReason"):
        splice_lyrics(
            _draft({"lyricOverride": "some text", "chordPlacements": []}), index
        )
    with pytest.raises(LyricSpliceError, match="lyricOverrideReason"):
        splice_lyrics(
            _draft({"lyricOverride": "t", "lyricOverrideReason": "  ",
                    "chordPlacements": []}),
            index,
        )


def test_too_many_overrides_fail_the_run(index):
    """Past the ceiling it is not an escape hatch, it is reference resolution
    being broken — and storing that is storing lyrics with no source."""
    lines = [
        {"lyricOverride": f"line {i}", "lyricOverrideReason": "no source",
         "chordPlacements": []}
        for i in range(10)
    ]
    with pytest.raises(UnresolvableLyricRefError) as exc:
        splice_lyrics(_draft(*lines), index)
    assert f"{MAX_OVERRIDE_SHARE:.0%}" in str(exc.value)
    assert "10 of 10" in str(exc.value)


def test_one_override_is_always_allowed_even_in_a_short_song(index):
    """The cap targets systemic failure, not a single legitimately merged
    line — 15% of a 3-line song would otherwise be zero."""
    result = splice_lyrics(
        _draft(
            _ref_line(0, (0, "C")),
            _ref_line(1, (0, "C")),
            {"lyricOverride": "one merged line", "lyricOverrideReason": "typo in source",
             "chordPlacements": []},
        ),
        index,
    )
    assert len(result.overrides) == 1


# --- the agent-facing schema -----------------------------------------------------


def test_the_agent_schema_allows_refs_and_the_stored_schema_does_not():
    base = song_json_schema()
    agent = agent_song_json_schema(base)

    line = agent["$defs"]["Line"]
    assert "lyrics" not in line["required"]
    assert "lyricRef" in line["properties"]
    assert "lyricOverride" in line["properties"]

    # The stored artifact is unchanged — this is a protocol, not a schema change.
    assert "lyricRef" not in json.dumps(base)
    assert base["$defs"]["Line"]["required"] == ["lineIndex", "lyrics"]
    assert base is not agent


# --- wired into the engine -------------------------------------------------------


class _RefProvider(LLMProvider):
    """A provider on the protocol, replaying scripted documents."""

    name = "test-refs"
    default_model = "test-refs-1"
    emits_lyric_refs = True

    def __init__(self, payloads: list[dict]):
        self.payloads = payloads
        self.calls = 0
        self.systems: list[str] = []

    def complete(self, system, turns, model=None, max_tokens=None, audio=None):
        self.calls += 1
        self.systems.append(system)
        payload = self.payloads[min(self.calls - 1, len(self.payloads) - 1)]
        return LLMResponse(
            text=json.dumps(payload), provider=self.name, model=self.default_model
        )


def _install(monkeypatch, provider):
    monkeypatch.setattr(
        "snoocle_server.reconcile.engine.get_provider", lambda name=None: provider
    )
    return provider


def test_a_reconcile_run_splices_references_and_records_them(monkeypatch, candidate):
    provider = _install(
        monkeypatch,
        _RefProvider([
            _draft(
                _ref_line(0, (0, "C"), (15, "G")),
                {"lyricOverride": "words no sheet has", "lyricOverrideReason": "outro ad-lib",
                 "chordPlacements": [{"charIndex": 0, "chord": "F"}]},
            )
        ]),
    )

    result = reconcile("Let It Be", "The Beatles", [candidate], None,
                       provider_name="test-refs")

    assert provider.calls == 1
    assert result.song.lines[0].lyrics == LINE_0
    assert result.song.lines[1].lyrics == "words no sheet has"

    # the escape hatch is auditable line by line, not a silent bypass
    overrides = [p for p in result.song.provenance if p.action == "lyric-override"]
    assert len(overrides) == 1
    assert "line 1" in overrides[0].notes and "outro ad-lib" in overrides[0].notes
    reconciled = [p for p in result.song.provenance if p.action == "reconciled"][-1]
    assert "spliced from 1 source reference(s)" in reconciled.notes
    assert "1 override(s)" in reconciled.notes

    # the contract actually reached the model
    assert "LYRIC REFERENCE PROTOCOL" in provider.systems[0]


def test_a_charindex_mismatch_goes_through_the_repair_loop(monkeypatch, candidate):
    """The model places chords into text it can see but is not retyping, so an
    index past the end of the real line is the predictable failure. It is
    repaired, named, and never clamped."""
    _install(
        monkeypatch,
        _RefProvider([
            _draft(_ref_line(1, (0, "C"), (400, "G"))),   # attempt 1: past the end
            _draft(_ref_line(1, (0, "C"), (15, "G"))),    # attempt 2: inside it
        ]),
    )

    result = reconcile("Let It Be", "The Beatles", [candidate], None,
                       provider_name="test-refs")

    assert result.attempts == 2
    assert result.song.lines[0].lyrics == LINE_1
    assert [p.charIndex for p in result.song.lines[0].chordPlacements] == [0, 15]


def test_an_unresolvable_ref_fails_the_run_without_a_repair_round(monkeypatch, candidate):
    """Never a fallback to model memory: a retry here is an invitation to
    supply the line from recollection instead of from a source."""
    provider = _install(
        monkeypatch,
        _RefProvider([
            _draft(_ref_line(0, (0, "C"), source="a-source-that-was-never-fetched")),
            _draft(_ref_line(0, (0, "C"))),  # would succeed — must never be reached
        ]),
    )

    with pytest.raises(LyricProvenanceError) as exc:
        reconcile("Let It Be", "The Beatles", [candidate], None, provider_name="test-refs")

    assert provider.calls == 1, "an unresolvable ref must not be retried"
    assert "a-source-that-was-never-fetched" in str(exc.value)
    assert isinstance(exc.value, ReconcileError)
    assert exc.value.error_code == "lyric_provenance"


def test_the_stored_song_is_shaped_exactly_as_before(monkeypatch, candidate):
    """This is an internal protocol between the server and the model, not a
    change to the artifact: the stored document must still validate against
    the unmodified Song schema and carry no trace of the protocol."""
    _install(monkeypatch, _RefProvider([_draft(_ref_line(0, (0, "C")))]))
    result = reconcile(
        "Let It Be", "The Beatles", [candidate], None, provider_name="test-refs"
    )

    dumped = result.song.model_dump(mode="json")
    assert set(dumped["lines"][0]) == {
        "lineIndex", "lyrics", "chordPlacements", "timeSeconds", "confidence"
    }


def test_the_document_the_model_emits_carries_no_lyric_text(index):
    """The property the whole change exists for, stated directly: the model's
    response contains none of the song's words, and the server puts them back."""
    draft = _draft(_ref_line(0, (0, "C")), _ref_line(1, (0, "C")))

    emitted = json.dumps(draft)
    assert LINE_0.strip() not in emitted
    assert LINE_1.strip() not in emitted

    stored = json.dumps(splice_lyrics(draft, index).document)
    assert LINE_0 in stored and LINE_1 in stored


# --- what the model is actually told ---------------------------------------------


def test_the_prompt_states_the_contract_with_a_worked_example(candidate, index):
    from snoocle_server.reconcile.prompt import build_system_prompt, build_user_prompt

    system = build_system_prompt(lyric_refs=True)
    assert "LYRIC REFERENCE PROTOCOL" in system
    assert '"lyricRef": {"sourceId"' in system
    # one referenced line and one instrumental line, spelled out
    assert '"lineIndex": 7, "lyricRef"' in system
    assert '"lineIndex": 8, "lyrics": ""' in system
    # the weaker restatement is gone: the format enforces it structurally, and
    # a plea alongside a hard constraint is just tokens.
    assert "Do not invent lyrics" not in system
    assert "Do not invent chords absent from all evidence." in system

    user = build_user_prompt(
        "Let It Be", "The Beatles", [candidate], None,
        agent_song_json_schema(song_json_schema()), "the-beatles--let-it-be", None,
        ref_index=index,
    )
    assert "Referenceable lyric sources" in user
    assert f"'web-1': lines 0-{len(candidate.lines) - 1}" in user


def test_the_agent_output_contract_carries_the_same_protocol():
    from snoocle_server.reconcile.anthropic_agent import build_system_blocks

    text = build_system_blocks(None)[0]["text"]
    assert "Lyric reference protocol" in text
    assert '"lyricRef"' in text and '"lyricOverride"' in text
    assert '"lineIndex": 7, "lyricRef"' in text
    assert "prior-song" in text
    assert "search_and_fetch_sheet" in text, "parsed sheets found mid-run are referenceable too"
    assert "fetch_chord_sheet" not in text, "the agent must never choose a raw URL"


def test_the_content_filter_message_no_longer_blames_missing_sources():
    """Both live failures had FOUR sources fetched. The old copy sent the
    reader looking for a retrieval problem that was not there."""
    from snoocle_server.reconcile.anthropic_agent import _classify_api_error

    err = _classify_api_error(Exception("400 ... blocked by content filtering policy"))
    assert err.error_code == "content_filtered"
    assert "no chord/lyric sources were found" not in str(err)
    assert "independently of how many sources" in str(err)


# --- which providers are party to the contract -----------------------------------


def test_every_model_backed_provider_is_on_the_protocol():
    """A new provider must not quietly inherit "not my contract" — the whole
    point is that no model-backed path can emit lyrics."""
    on_protocol = {
        name for name, cls in _PROVIDERS.items() if cls.emits_lyric_refs
    }
    assert on_protocol == {"anthropic", "anthropic-agent", "openai", "gemini"}
    # The deterministic mock builds its Song in local code and is never
    # prompted, so it is not party to a contract it was never given.
    assert _PROVIDERS["mock"].emits_lyric_refs is False
    # The external agent workspace is a third-party system on its own release
    # cycle; it adopts the protocol by setting this flag once it emits refs.
    assert _PROVIDERS["agent"].emits_lyric_refs is False


def test_the_mock_path_is_untouched_by_the_protocol(candidate):
    result = reconcile("Let It Be", "The Beatles", [candidate], None, provider_name="mock")
    assert result.song.lines[0].lyrics == LINE_0
    assert not [p for p in result.song.provenance if p.action == "lyric-override"]
