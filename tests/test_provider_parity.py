"""Acceptance-step-2 evidence (offline half): the baseline reconciliation
input is IDENTICAL across providers (structured data only), and it
demonstrably contains ALL discovered candidate sources, not just the first.
"""

import json
from pathlib import Path

import pytest

from snoocle_server.discovery.service import candidate_from_text
from snoocle_server.mir.base import ChordSegment, MirAnalysis, StructureSegment
from snoocle_server.reconcile import reconcile
from snoocle_server.reconcile.prompt import SYSTEM_PROMPT, build_user_prompt
from snoocle_server.reconcile.providers import LLMProvider, LLMResponse
from snoocle_server.schema import song_json_schema

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def candidates():
    c1 = candidate_from_text((FIXTURES / "sheet_over_lyrics.txt").read_text(), "web-1", url="https://a.example/x")
    c2 = candidate_from_text((FIXTURES / "sheet_inline.txt").read_text(), "web-2", url="https://b.example/y")
    c3 = candidate_from_text((FIXTURES / "sheet_capo.txt").read_text(), "web-3", url="https://c.example/z")
    return [c for c in (c1, c2, c3) if c]


@pytest.fixture()
def mir():
    return MirAnalysis(
        engines={"beats": "madmom", "chords": "chroma-template-fallback", "structure": "librosa-agglomerative-fallback"},
        duration_seconds=180.0,
        bpm=120.0,
        key="C major",
        chords=[ChordSegment(start=0, end=4, chord="C")],
        sections=[StructureSegment(start=0, end=180, label="other")],
    )


def test_prompt_contains_all_candidates_and_mir(candidates, mir):
    assert len(candidates) == 3
    prompt = build_user_prompt(
        "Let It Be", "The Beatles", candidates, mir, song_json_schema(), "the-beatles--let-it-be", None
    )
    for c in candidates:
        assert c.sourceId in prompt, f"{c.sourceId} missing from reconciliation input"
        assert (c.url or "") in prompt
    assert f"{len(candidates)} found — use ALL of them" in prompt
    assert "mir-audio-analysis" in prompt
    assert "chordTimeline" in prompt


def test_chord_rule_is_in_the_system_prompt():
    assert "sounding harmony" in SYSTEM_PROMPT
    assert "never a fretboard shape" in SYSTEM_PROMPT


class RecordingProvider(LLMProvider):
    name = "recorder"
    default_model = "recorder-1"
    recordings: list = []

    def complete(self, system, turns, model=None, max_tokens=None, audio=None):
        RecordingProvider.recordings.append(
            {"system": system, "turns": [dict(t) for t in turns], "audio": audio}
        )
        song = {
            "id": "the-beatles--let-it-be",
            "metadata": {"title": "Let It Be", "artist": "The Beatles"},
            "lines": [{"lineIndex": 0, "lyrics": "x", "chordPlacements": [{"charIndex": 0, "chord": "C"}]}],
            "provenance": [],
        }
        return LLMResponse(text=json.dumps(song), provider=self.name, model=self.default_model)


def test_baseline_request_identical_across_providers(candidates, mir, monkeypatch):
    """Simulate two different provider selections; the engine must hand them
    byte-identical baseline input (system + turns, no audio)."""
    RecordingProvider.recordings = []
    monkeypatch.setattr(
        "snoocle_server.reconcile.engine.get_provider", lambda name=None: RecordingProvider()
    )
    for pretend_provider in ("anthropic", "gemini"):
        reconcile("Let It Be", "The Beatles", candidates, mir, provider_name=pretend_provider)
    a, b = RecordingProvider.recordings
    assert a["system"] == b["system"]
    assert a["turns"] == b["turns"]
    assert a["audio"] is None and b["audio"] is None  # baseline: structured data only
