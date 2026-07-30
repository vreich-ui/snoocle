"""correction_routing.py: decide whether a guidance note is a targeted
correction (-> infer notes-only scope) and whether it can be a PATCH rather
than a full reconcile (-> never when it targets lyrics).

Deterministic rules run first and are free; the LLM tier only ever grants
`is_targeted_correction`, never `patch_eligible` — see the module docstring
for why a generic yes/no isn't reliable enough for "lyrics never leak into
an op".
"""

from __future__ import annotations

import pytest

from snoocle_server.correction_routing import classify_correction


def _classify(text, llm_result=None):
    """llm=lambda that never gets called unless the deterministic tier finds
    nothing — passing a sentinel result lets tests assert that."""
    calls = []

    def fake_llm(t):
        calls.append(t)
        return llm_result

    result = classify_correction(text, llm=fake_llm)
    return result, calls


# --- the reported scenario, verbatim -----------------------------------------


def test_the_reported_correction_is_targeted_and_patch_eligible():
    result, llm_calls = _classify("change the C to a B in line 12")
    assert result.is_targeted_correction is True
    assert result.targets_lyrics is False
    assert result.patch_eligible is True
    assert llm_calls == [], "a deterministic rule fired; the LLM tier must not run"
    assert set(result.matched_rules) == {"chord-symbol", "line-reference"}


# --- deterministic rules, one at a time --------------------------------------


@pytest.mark.parametrize("text,rule", [
    ("change Bm7 to C#dim in the intro", "chord-symbol"),
    ("the chord in bar 4 is wrong", "chord-symbol"),
    ("go back and check line 5", "line-reference"),
    ("fix line42 please", "line-reference"),
    ("rename section 3 to Bridge", "section-reference"),
    ("the bridge repeats twice, not once", "section-reference"),
    ("the pre-chorus should be its own section", "section-reference"),
])
def test_each_deterministic_rule_fires_alone(text, rule):
    result, llm_calls = _classify(text)
    assert result.is_targeted_correction is True
    assert rule in result.matched_rules
    assert llm_calls == []


def test_lyric_word_signal_sets_targets_lyrics_and_is_still_targeted():
    result, _ = _classify("the chorus lyrics say love but should say lust")
    assert result.is_targeted_correction is True
    assert result.targets_lyrics is True
    assert result.patch_eligible is False, "lyrics can never go through the patch path"
    assert "lyric-word" in result.matched_rules


def test_a_bare_word_mention_also_blocks_patch_eligibility():
    result, _ = _classify("fix the word in line 4")
    assert result.is_targeted_correction is True
    assert result.targets_lyrics is True
    assert result.patch_eligible is False
    assert "line-reference" in result.matched_rules
    assert "lyric-word" in result.matched_rules


# --- the false-positive guards ------------------------------------------------


@pytest.mark.parametrize("text", [
    "Do you want this changed?",
    "Go ahead and publish it",
    "Am I reading this correctly?",
    "the whole song feels off, please redo it",
    "can you double check the sources for this one",
    "",
    "   ",
])
def test_ordinary_sentences_are_not_targeted(text):
    result, llm_calls = _classify(text, llm_result=None)
    assert result.is_targeted_correction is False
    assert result.patch_eligible is False
    if text.strip():
        assert llm_calls == [text.strip()]


def test_a_stoplisted_word_still_counts_in_an_explicit_replacement():
    """"Do" and "Go" are each ambiguous with an ordinary word alone; "Do to
    Go" as an explicit replacement is real support, not just the bare words."""
    result, _ = _classify("change Do to Go")
    assert result.is_targeted_correction is True


def test_the_word_chord_alone_is_sufficient_support():
    result, _ = _classify("the chord sounds wrong here")
    assert result.is_targeted_correction is True
    assert "chord-symbol" in result.matched_rules


# --- the LLM fallback tier ----------------------------------------------------


def test_llm_tier_only_runs_when_no_deterministic_rule_fires():
    _, llm_calls = _classify("everything about this feels wrong", llm_result={"isTargetedCorrection": True, "confidence": 0.9})
    assert llm_calls == ["everything about this feels wrong"]


def test_llm_tier_can_grant_is_targeted_but_never_patch_eligible():
    result, _ = _classify(
        "please fix the specific issue with this recording",
        llm_result={"isTargetedCorrection": True, "confidence": 0.9},
    )
    assert result.is_targeted_correction is True
    assert result.via_llm is True
    assert result.patch_eligible is False, (
        "a generic yes/no from the LLM tier is not reliable enough to trust "
        "with the one property that must never be wrong"
    )


def test_llm_tier_below_confidence_threshold_does_not_infer():
    result, _ = _classify(
        "not sure what's wrong, take a look",
        llm_result={"isTargetedCorrection": True, "confidence": 0.2},
    )
    assert result.is_targeted_correction is False


def test_llm_tier_unavailable_defaults_to_not_inferring():
    """An inconclusive classification must default to NOT constraining
    scope — never to guessing."""
    result, _ = _classify("not sure what's wrong here", llm_result=None)
    assert result.is_targeted_correction is False


def test_llm_tier_malformed_payload_does_not_crash():
    result, _ = _classify("some ambiguous note", llm_result={"garbage": True})
    assert result.is_targeted_correction is False


# --- describe() is used for step/provenance text ------------------------------


def test_describe_names_the_rule():
    result, _ = _classify("change the C to a B in line 12")
    assert "targeted" in result.describe()
    assert "chord-symbol" in result.describe()
