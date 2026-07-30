"""reconcile/engine.py's patch path, end to end through `reconcile()`.

The reported incident: a request to change one chord (C -> B) in Paint It
Black ran with scope listen=off/reconcile=on, re-discovered 3 sources, asked
the model for a complete Song, and was blocked by the content filter — over
ONE chord. These tests pin the fix at the engine boundary: a notes-only,
patch-eligible correction gets the patch prompt (never the reconciliation
prompt, never the lyric-reference contract), the model answers with ops, and
the server applies them — never regenerating anything.
"""

from __future__ import annotations

import json

import pytest

from snoocle_server.reconcile import reconcile
from snoocle_server.reconcile.engine import PatchApplicationError, ReconcileError
from snoocle_server.reconcile.patch_ops import MAX_OPS
from snoocle_server.reconcile.providers import LLMProvider, LLMResponse
from snoocle_server.schema import Song
from snoocle_server.scope import AnalysisScope

NOTES_ONLY = AnalysisScope(listen=False, reconcile=False)


def _gold_song() -> dict:
    """13 lines (0-12) so the reported scenario's own line number is real."""
    lines = []
    for i in range(12):
        lines.append({
            "lineIndex": i, "lyrics": f"la la la number {i}",
            "chordPlacements": [{"charIndex": 0, "chord": "G", "confidence": 0.7}],
        })
    lines.append({
        "lineIndex": 12,
        "lyrics": "there's a calm before the storm, I know",
        "timeSeconds": 42.0,
        "confidence": 0.85,
        "chordPlacements": [
            {"charIndex": 0, "chord": "F", "timeSeconds": 42.0, "confidence": 0.9},
            {"charIndex": 21, "chord": "C", "timeSeconds": 44.5, "confidence": 0.8,
             "beat": {"measure": 11, "beat": 1}},
        ],
    })
    return {
        "id": "the-rolling-stones--paint-it-black",
        "metadata": {"title": "Paint It Black", "artist": "The Rolling Stones", "bpm": 128.0},
        "audio": {
            "youtubeVideoId": "flSmiIne4k1",
            "beats": [{"time": float(i), "measure": i // 4 + 1, "beatInMeasure": i % 4 + 1} for i in range(8)],
            "syncMap": [{"lineIndex": i, "time": float(i) * 3.5} for i in range(13)],
        },
        "sections": [
            {"sectionIndex": 0, "name": "Verse 1", "kind": "verse",
             "startLineIndex": 0, "endLineIndex": 5},
            {"sectionIndex": 1, "name": "Verse 2", "kind": "verse",
             "startLineIndex": 6, "endLineIndex": 12},
        ],
        "lines": lines,
        "provenance": [
            {"timestamp": "2026-07-01T00:00:00Z", "actor": "reconcile:test/gold", "action": "reconciled"},
        ],
    }


class _ScriptedProvider(LLMProvider):
    """Replays scripted responses; records every system prompt it was sent
    so tests can assert which contract actually reached the model."""

    name = "test-patch"
    default_model = "test-patch-1"
    supports_patch_ops = True
    # Deliberately False: the fallback-to-full-reconcile tests below use
    # plain Song JSON (real lyrics), not lyricRef-formatted responses — the
    # lyric-reference protocol itself is covered end-to-end in
    # test_lyric_refs.py. What THIS file tests is patch-vs-full routing.
    emits_lyric_refs = False

    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = 0
        self.systems: list[str] = []
        self.user_prompts: list[str] = []

    def complete(self, system, turns, model=None, max_tokens=None, audio=None):
        self.calls += 1
        self.systems.append(system)
        self.user_prompts.append(turns[-1]["text"])
        payload = self.payloads[min(self.calls - 1, len(self.payloads) - 1)]
        return LLMResponse(text=json.dumps(payload), provider=self.name, model=self.default_model)


def _reconcile_with(monkeypatch, payloads, guidance, patch_ops_eligible=True, prior=None):
    provider = _ScriptedProvider(payloads)
    monkeypatch.setattr(
        "snoocle_server.reconcile.engine.get_provider", lambda name=None: provider
    )
    result = reconcile(
        "Paint It Black", "The Rolling Stones", [], None,
        provider_name="test-patch", song_id="the-rolling-stones--paint-it-black",
        guidance=guidance, prior_song=prior or _gold_song(), scope=NOTES_ONLY,
        patch_ops_eligible=patch_ops_eligible,
    )
    return provider, result


# --- the primary reported scenario -----------------------------------------------


def test_change_one_chord_produces_one_op_and_is_byte_identical_otherwise(monkeypatch):
    provider, result = _reconcile_with(
        monkeypatch,
        [{"ops": [{"op": "replace_chord", "lineIndex": 12, "charIndex": 21,
                    "from": "C", "to": "B", "reason": "walkdown, not a straight C"}]}],
        guidance="change the C to a B in line 12",
    )
    assert provider.calls == 1
    assert result.patch_ops_applied == 1

    gold = Song.model_validate(_gold_song())
    song = result.song
    assert song.lines[12].chordPlacements[1].chord == "B"

    before = gold.model_dump(mode="json")
    after = song.model_dump(mode="json")
    before["lines"][12]["chordPlacements"][1]["chord"] = "B"  # the ONE expected diff
    before["provenance"] = after["provenance"]  # provenance growth is expected
    assert after == before, "every field but the one named chord must be untouched"

    # the correction history is auditable op by op
    patch_entries = [p for p in song.provenance if p.action == "patch-applied"]
    assert len(patch_entries) == 1
    assert "line 12" in patch_entries[0].notes and "C -> B" in patch_entries[0].notes
    assert "walkdown" in patch_entries[0].notes


def test_the_patch_prompt_is_sent_not_the_reconciliation_or_lyric_prompt(monkeypatch):
    provider, _ = _reconcile_with(
        monkeypatch,
        [{"ops": [{"op": "replace_chord", "lineIndex": 12, "charIndex": 21, "from": "C", "to": "B"}]}],
        guidance="change the C to a B in line 12",
    )
    system = provider.systems[0]
    assert "targeted-correction engine" in system
    assert "ops" in system
    assert "LYRIC REFERENCE PROTOCOL" not in system
    assert "reconciliation engine" not in system.lower() or "targeted" in system.lower()
    # the closed op set is explicit
    for op_name in ("replace_chord", "insert_chord", "remove_chord", "move_chord",
                     "set_section_name", "set_section_bounds", "split_line", "merge_line"):
        assert op_name in system
    # never a lyric line in the user prompt beyond what the prior doc already had
    assert "NEVER write out" in system


def test_multiple_ops_in_one_patch_all_land_in_provenance(monkeypatch):
    provider, result = _reconcile_with(
        monkeypatch,
        [{"ops": [
            {"op": "replace_chord", "lineIndex": 12, "charIndex": 21, "from": "C", "to": "B"},
            {"op": "set_section_name", "sectionIndex": 1, "to": "Bridge"},
        ]}],
        guidance="fix the chord in line 12 and rename section 2 to Bridge",
    )
    assert result.patch_ops_applied == 2
    assert result.song.sections[1].name == "Bridge"
    patch_entries = [p for p in result.song.provenance if p.action == "patch-applied"]
    assert len(patch_entries) == 2


# --- failure modes: fatal, not repaired --------------------------------------------


def test_a_mismatched_from_fails_clearly_and_is_not_repaired(monkeypatch):
    provider = _ScriptedProvider([
        {"ops": [{"op": "replace_chord", "lineIndex": 12, "charIndex": 21, "from": "G", "to": "B"}]},
        {"ops": [{"op": "replace_chord", "lineIndex": 12, "charIndex": 21, "from": "C", "to": "B"}]},
    ])
    monkeypatch.setattr(
        "snoocle_server.reconcile.engine.get_provider", lambda name=None: provider
    )
    with pytest.raises(PatchApplicationError) as exc:
        reconcile(
            "Paint It Black", "The Rolling Stones", [], None,
            provider_name="test-patch", song_id="the-rolling-stones--paint-it-black",
            guidance="change the C to a B in line 12", prior_song=_gold_song(),
            scope=NOTES_ONLY, patch_ops_eligible=True,
        )
    assert provider.calls == 1, "a wrong op must not be retried"
    assert "charIndex 21" in str(exc.value)
    assert "'C'" in str(exc.value) and "'G'" in str(exc.value)
    assert isinstance(exc.value, ReconcileError)
    assert exc.value.error_code == "patch_failed"


def test_an_unknown_op_fails_the_run(monkeypatch):
    with pytest.raises(PatchApplicationError, match="unknown op"):
        _reconcile_with(
            monkeypatch,
            [{"ops": [{"op": "delete_the_song", "lineIndex": 12}]}],
            guidance="change the C to a B in line 12",
        )


def test_over_the_op_cap_fails_the_run(monkeypatch):
    ops = [{"op": "set_section_name", "sectionIndex": 0, "to": f"X{i}"} for i in range(MAX_OPS + 1)]
    with pytest.raises(PatchApplicationError, match="exceeds the cap"):
        _reconcile_with(monkeypatch, [{"ops": ops}], guidance="lots of small renames")


def test_malformed_json_response_fails_the_run(monkeypatch):
    class _Broken(_ScriptedProvider):
        def complete(self, system, turns, model=None, max_tokens=None, audio=None):
            self.calls += 1
            return LLMResponse(text="not json at all", provider=self.name, model=self.default_model)

    broken = _Broken([])
    monkeypatch.setattr(
        "snoocle_server.reconcile.engine.get_provider", lambda name=None: broken
    )
    with pytest.raises(PatchApplicationError, match="not valid JSON"):
        reconcile(
            "Paint It Black", "The Rolling Stones", [], None,
            provider_name="test-patch", song_id="the-rolling-stones--paint-it-black",
            guidance="change the C to a B in line 12", prior_song=_gold_song(),
            scope=NOTES_ONLY, patch_ops_eligible=True,
        )
    assert broken.calls == 1


# --- lyric changes route to full reconcile, visibly --------------------------------


def test_patch_ops_eligible_false_skips_the_patch_prompt_entirely(monkeypatch):
    """The pipeline decides eligibility from guidance classification BEFORE
    engine.py ever runs — a lyric-targeting correction never even attempts
    a patch. Falls through to the existing notes-only full reconcile."""
    full_song = _gold_song()
    full_song["lines"][12]["lyrics"] = "there's a lull before the storm, I know"
    provider, result = _reconcile_with(
        monkeypatch,
        [full_song], guidance="line 12 should say 'lull' not 'calm'",
        patch_ops_eligible=False,
    )
    assert "targeted-correction engine" not in provider.systems[0]
    assert result.patch_ops_applied == 0
    assert "lull" in result.song.lines[12].lyrics


def test_the_model_can_explicitly_decline_the_patch_and_fall_back(monkeypatch):
    """Even when the pipeline THOUGHT a correction was patch-eligible, the
    model itself can say "this needs real words" — visibly, via a declared
    response shape, not by attempting and failing a patch."""
    full_song = _gold_song()
    full_song["lines"][12]["lyrics"] = "a rewritten line entirely"
    provider = _ScriptedProvider([
        {"ops": [], "needsFullReconcile": True, "reason": "the correction needs new words"},
        full_song,
    ])
    monkeypatch.setattr(
        "snoocle_server.reconcile.engine.get_provider", lambda name=None: provider
    )
    result = reconcile(
        "Paint It Black", "The Rolling Stones", [], None,
        provider_name="test-patch", song_id="the-rolling-stones--paint-it-black",
        guidance="line 12 needs completely different words",
        prior_song=_gold_song(), scope=NOTES_ONLY, patch_ops_eligible=True,
    )
    assert provider.calls == 2, "one patch attempt, then one full-reconcile call"
    assert result.patch_ops_applied == 0
    assert result.song.lines[12].lyrics == "a rewritten line entirely"
    # the second call got the REAL reconciliation prompt, not the patch one
    assert "targeted-correction engine" in provider.systems[0]
    assert "targeted-correction engine" not in provider.systems[1]


# --- provider opt-in --------------------------------------------------------------


def test_a_provider_without_patch_support_gets_the_full_reconcile_prompt(monkeypatch):
    full_song = _gold_song()
    full_song["lines"][12]["chordPlacements"][1]["chord"] = "B"

    class _NoPatchProvider(_ScriptedProvider):
        supports_patch_ops = False

    provider = _NoPatchProvider([full_song])
    monkeypatch.setattr(
        "snoocle_server.reconcile.engine.get_provider", lambda name=None: provider
    )
    result = reconcile(
        "Paint It Black", "The Rolling Stones", [], None,
        provider_name="no-patch", song_id="the-rolling-stones--paint-it-black",
        guidance="change the C to a B in line 12", prior_song=_gold_song(),
        scope=NOTES_ONLY, patch_ops_eligible=True,
    )
    assert provider.calls == 1
    assert result.patch_ops_applied == 0
    assert "targeted-correction engine" not in provider.systems[0]
    assert result.song.lines[12].chordPlacements[1].chord == "B"


def test_full_pipeline_scope_not_notes_only_never_attempts_a_patch(monkeypatch):
    """Patch is a notes-only-scope-only mechanism: even a chord-mentioning
    guidance must not trigger it when the run isn't notes-only."""
    full_song = _gold_song()
    full_song["lines"][12]["chordPlacements"][1]["chord"] = "B"
    provider = _ScriptedProvider([full_song])
    monkeypatch.setattr(
        "snoocle_server.reconcile.engine.get_provider", lambda name=None: provider
    )
    result = reconcile(
        "Paint It Black", "The Rolling Stones", [], None,
        provider_name="test-patch", song_id="the-rolling-stones--paint-it-black",
        guidance="change the C to a B in line 12", prior_song=_gold_song(),
        scope=AnalysisScope(listen=True, reconcile=True),  # NOT notes-only
        patch_ops_eligible=True,
    )
    assert "targeted-correction engine" not in provider.systems[0]
    assert result.patch_ops_applied == 0
