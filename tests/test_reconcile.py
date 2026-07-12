import json
from pathlib import Path

import pytest

from snoocle_server.discovery.service import candidate_from_text
from snoocle_server.mir.base import Beat, ChordSegment, MirAnalysis, StructureSegment
from snoocle_server.reconcile import reconcile
from snoocle_server.reconcile.engine import ReconcileError, extract_json
from snoocle_server.reconcile.providers import (
    LLMProvider,
    LLMResponse,
    get_provider,
    provider_capabilities,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def candidates():
    c1 = candidate_from_text(
        (FIXTURES / "sheet_over_lyrics.txt").read_text(), "web-1", url="https://a.example/x"
    )
    c2 = candidate_from_text(
        (FIXTURES / "sheet_inline.txt").read_text(), "web-2", url="https://b.example/y"
    )
    assert c1 and c2
    return [c1, c2]


@pytest.fixture()
def mir():
    return MirAnalysis(
        engines={"beats": "madmom", "chords": "chroma-template-fallback", "structure": "librosa-agglomerative-fallback"},
        duration_seconds=240.0,
        bpm=72.0,
        time_signature="4/4",
        key="C major",
        beats=[Beat(time=t * 0.833, position=(t % 4) + 1) for t in range(64)],
        chords=[
            ChordSegment(start=0.0, end=4.0, chord="C"),
            ChordSegment(start=4.0, end=8.0, chord="G"),
            ChordSegment(start=8.0, end=12.0, chord="Am"),
            ChordSegment(start=12.0, end=16.0, chord="F"),
        ],
        sections=[
            StructureSegment(start=0.0, end=60.0, label="verse"),
            StructureSegment(start=60.0, end=120.0, label="chorus"),
            StructureSegment(start=120.0, end=240.0, label="outro"),
        ],
    )


def test_extract_json_tolerates_fences_and_preamble():
    assert json.loads(extract_json('Here you go:\n```json\n{"a": 1}\n```\nDone.')) == {"a": 1}
    assert json.loads(extract_json('{"a": {"b": 2}}')) == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_provider_registry():
    caps = provider_capabilities()
    assert set(caps) == {"anthropic", "openai", "gemini", "agent", "mock"}
    assert caps["anthropic"]["supportsAudioInput"] is False  # baseline is structured-only
    assert caps["openai"]["supportsAudioInput"] is True
    assert caps["gemini"]["supportsAudioInput"] is True
    assert get_provider("anthropic").default_model == "claude-opus-4-8"


def test_mock_reconcile_end_to_end(candidates, mir):
    result = reconcile(
        "Let It Be",
        "The Beatles",
        candidates,
        mir,
        provider_name="mock",
        youtube_video_id="QDYfEBY9NM4",
    )
    song = result.song
    assert result.provider == "mock"
    assert song.id == "the-beatles--let-it-be"
    assert song.metadata.bpm == 72.0
    assert song.metadata.key  # from sheet or MIR
    assert song.audio.youtubeVideoId == "QDYfEBY9NM4"
    assert song.lines and song.lines[0].chordPlacements
    # sections derived from sheet headers, timestamps from MIR
    assert song.sections
    assert song.sections[0].startTime is not None
    assert song.audio.syncMap
    # provenance appended server-side: discovery + mir + reconcile
    actions = [p.action for p in song.provenance]
    assert actions == ["discovered-sources", "mir-analysis", "reconciled"]
    assert "2 candidate" in song.provenance[0].notes
    # display capo forced to 0; chords are sounding harmonies (validated by schema)
    assert song.displayPreferences.capo == 0


class FlakyProvider(LLMProvider):
    """Emits a shape chord first (violating the chord rule), then valid JSON
    after the repair prompt — exercises the validate/repair loop."""

    name = "flaky"
    default_model = "flaky-1"

    def __init__(self):
        self.calls = 0

    def complete(self, system, turns, model=None, max_tokens=None, audio=None):
        self.calls += 1
        bad = {
            "id": "x--y",
            "metadata": {"title": "X", "artist": "Y"},
            "lines": [
                {"lineIndex": 0, "lyrics": "la la", "chordPlacements": [{"charIndex": 0, "chord": "x02210"}]}
            ],
            "provenance": [],
        }
        good = json.loads(json.dumps(bad))
        good["lines"][0]["chordPlacements"][0]["chord"] = "Am"
        payload = bad if self.calls == 1 else good
        return LLMResponse(text=json.dumps(payload), provider=self.name, model=self.default_model)


def test_repair_loop_fixes_shape_chord(candidates, mir, monkeypatch):
    flaky = FlakyProvider()
    monkeypatch.setattr("snoocle_server.reconcile.engine.get_provider", lambda name=None: flaky)
    result = reconcile("X", "Y", candidates, mir, provider_name="flaky")
    assert result.attempts == 2  # first attempt rejected by schema, second passes
    assert flaky.calls == 2
    assert result.song.lines[0].chordPlacements[0].chord == "Am"


class HopelessProvider(FlakyProvider):
    def complete(self, system, turns, model=None, max_tokens=None, audio=None):
        self.calls += 1
        return LLMResponse(text="not json at all", provider="hopeless", model="h-1")


def test_reconcile_gives_up_after_repair_budget(candidates, mir, monkeypatch):
    hopeless = HopelessProvider()
    monkeypatch.setattr("snoocle_server.reconcile.engine.get_provider", lambda name=None: hopeless)
    with pytest.raises(ReconcileError, match="failed schema validation"):
        reconcile("X", "Y", candidates, mir, provider_name="hopeless")


def test_reconcile_without_mir(candidates):
    result = reconcile("Let It Be", "The Beatles", candidates, None, provider_name="mock")
    assert result.song.lines
    actions = [p.action for p in result.song.provenance]
    assert "mir-analysis" not in actions
