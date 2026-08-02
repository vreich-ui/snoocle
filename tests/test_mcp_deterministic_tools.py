from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from snoocle_server import mcp_server
from snoocle_server.deterministic import MAX_CANDIDATES, MAX_JSON_BYTES, MAX_LINES
from snoocle_server.discovery.models import CandidateSource, SectionStart
from snoocle_server.mir.base import ChordSegment, MirAnalysis
from snoocle_server.schema import ChordPlacement, Line


def _candidate(source_id: str = "sheet-a", chord: str = "C") -> CandidateSource:
    return CandidateSource(
        sourceId=source_id,
        retrievedAt="2026-08-02T00:00:00Z",
        confidence=0.9,
        parseConfidence=0.9,
        coverage="full-song",
        sectionsHint=["Verse 1"],
        sectionStarts=[SectionStart(name="Verse 1", startLineIndex=0)],
        lines=[
            Line(
                lineIndex=index,
                lyrics=lyrics,
                chordPlacements=[ChordPlacement(charIndex=char_index, chord=chord)],
            )
            for index, (lyrics, char_index) in enumerate(
                [(" one", 1), ("two ", 0), ("three", 2), ("four", 3)]
            )
        ],
    )


def _mir(chord: str = "C") -> MirAnalysis:
    return MirAnalysis(
        engines={"chords": "test"},
        duration_seconds=4,
        chords=[
            ChordSegment(start=float(index), end=float(index + 1), chord=chord)
            for index in range(4)
        ],
    )


def _json(model) -> str:
    return json.dumps(model.model_dump(mode="json"))


def _assert_observation(response: dict, *, ok: bool = True) -> None:
    assert response["ok"] is ok
    assert isinstance(response["elapsedMs"], int)
    assert response["elapsedMs"] >= 0
    assert response["cacheStatus"] == "not_applicable"
    assert response["modelCalls"] == 0
    assert response["modelCostUSD"] == 0
    assert isinstance(response["inputSummary"], dict)
    assert isinstance(response["outputSummary"], dict)
    assert isinstance(response["warnings"], list)


def test_parse_candidate_text_is_deterministic_and_model_free():
    sheet = "[Verse 1]\n[C] one\n[G]two \n[Am]three\n[F]four"

    first = mcp_server.parse_candidate_text(sheet, "caller-sheet")
    second = mcp_server.parse_candidate_text(sheet, "caller-sheet")

    _assert_observation(first)
    assert first["result"] == second["result"]
    candidate = first["result"]["candidate"]
    assert candidate["retrievedAt"] == "1970-01-01T00:00:00+00:00"
    assert [line["lyrics"] for line in candidate["lines"]] == [" one", "two", "three", "four"]
    assert [
        [(item["charIndex"], item["chord"]) for item in line["chordPlacements"]]
        for line in candidate["lines"]
    ] == [[(0, "C")], [(0, "G")], [(0, "Am")], [(0, "F")]]


def test_score_rank_and_select_are_bounded_observed_service_results():
    candidate = _candidate()
    alternate = _candidate("sheet-b", "G")
    mir_json = _json(_mir("D"))

    scored = mcp_server.score_candidate_against_mir(_json(candidate), mir_json)
    ranked = mcp_server.rank_candidates_deterministically(
        json.dumps([candidate.model_dump(mode="json"), alternate.model_dump(mode="json")]),
        mir_json,
    )
    selected = mcp_server.select_candidate_deterministically(
        json.dumps([candidate.model_dump(mode="json")]), "strict", mir_json
    )

    for response in (scored, ranked, selected):
        _assert_observation(response)
    assert scored["result"]["transposition"] == 2
    assert scored["result"]["matched"] == 4
    assert ranked["result"]["ranked"][0]["sourceId"] == "sheet-a"
    assert selected["result"]["status"] == "selected"
    assert selected["result"]["selectedSourceId"] == "sheet-a"


def test_baseline_and_validation_preserve_source_content_without_timing():
    candidate = _candidate()
    candidate.lines[0].chordPlacements = [
        ChordPlacement(charIndex=0, chord="C"),
        ChordPlacement(charIndex=4, chord="G"),
    ]

    baseline = mcp_server.build_song_baseline(
        _json(candidate), "artist--title", "Title", "Artist"
    )
    _assert_observation(baseline)
    song = baseline["result"]["song"]

    assert [line["lyrics"] for line in song["lines"]] == [
        line.lyrics for line in candidate.lines
    ]
    assert [
        [(item["charIndex"], item["chord"]) for item in line["chordPlacements"]]
        for line in song["lines"]
    ] == [
        [(item.charIndex, item.chord) for item in line.chordPlacements]
        for line in candidate.lines
    ]
    assert all(line["timeSeconds"] is None for line in song["lines"])
    assert all(
        item["timeSeconds"] is None
        for line in song["lines"]
        for item in line["chordPlacements"]
    )
    assert song["displayPreferences"]["capo"] == 0

    validated = mcp_server.validate_song_json(json.dumps(song))
    _assert_observation(validated)
    assert validated["result"]["valid"] is True
    assert validated["result"]["song"] == song


@pytest.mark.parametrize(
    "call,code",
    [
        (lambda: mcp_server.validate_song_json("{"), "invalid_json"),
        (
            lambda: mcp_server.rank_candidates_deterministically(
                json.dumps([{}] * (MAX_CANDIDATES + 1))
            ),
            "too_many_candidates",
        ),
        (
            lambda: mcp_server.parse_candidate_text(
                "\n".join("[C]line" for _ in range(MAX_LINES + 1)), "too-many-lines"
            ),
            "too_many_lines",
        ),
        (
            lambda: mcp_server.validate_song_json(" " * (MAX_JSON_BYTES + 1)),
            "payload_too_large",
        ),
    ],
)
def test_payload_and_list_bounds_return_structured_errors(call, code):
    response = call()

    _assert_observation(response, ok=False)
    assert response["error"]["code"] == code
    assert "traceback" not in json.dumps(response).casefold()


def test_song_schema_failure_is_structured_and_does_not_echo_lyrics():
    song = mcp_server.build_song_baseline(
        _json(_candidate()), "artist--title", "Title", "Artist"
    )["result"]["song"]
    song["lines"][0]["chordPlacements"][0]["charIndex"] = 999
    song["lines"][0]["lyrics"] = "sensitive lyric"

    response = mcp_server.validate_song_json(json.dumps(song))

    _assert_observation(response, ok=False)
    assert response["error"]["code"] == "schema_validation_failed"
    assert "sensitive lyric" not in json.dumps(response)

    song["lines"] = None
    wrong_type = mcp_server.validate_song_json(json.dumps(song))
    _assert_observation(wrong_type, ok=False)
    assert wrong_type["error"]["code"] == "schema_validation_failed"


def test_each_wrapper_routes_only_to_its_intended_service(monkeypatch):
    candidate = _candidate()
    song = mcp_server._build_song_from_candidate(
        candidate, song_id="artist--title", title="Title", artist="Artist"
    )
    candidate_json = _json(candidate)
    candidates_json = json.dumps([candidate.model_dump(mode="json")])
    mir_json = _json(_mir())
    calls: list[str] = []

    monkeypatch.setattr(
        mcp_server,
        "_candidate_from_text",
        lambda *args, **kwargs: calls.append("parse") or candidate,
    )
    monkeypatch.setattr(
        mcp_server,
        "_score_candidate",
        lambda *args, **kwargs: calls.append("score")
        or SimpleNamespace(
            source_id="sheet-a", score=1.0, transposition=0,
            matched=4, total=4, conflicts=[]
        ),
    )
    monkeypatch.setattr(
        mcp_server,
        "_rank_candidates_deterministically",
        lambda *args, **kwargs: calls.append("rank")
        or [SimpleNamespace(to_dict=lambda: {"sourceId": "sheet-a"})],
    )
    monkeypatch.setattr(
        mcp_server,
        "_select_candidate_deterministically",
        lambda *args, **kwargs: calls.append("select")
        or SimpleNamespace(
            to_dict=lambda: {
                "status": "selected", "reason": None,
                "selectedSourceId": "sheet-a", "ranked": [], "conflicts": []
            }
        ),
    )
    monkeypatch.setattr(
        mcp_server,
        "_build_song_from_candidate",
        lambda *args, **kwargs: calls.append("baseline") or song,
    )
    monkeypatch.setattr(
        mcp_server.Song,
        "model_validate",
        classmethod(lambda cls, value: calls.append("validate") or song),
    )

    responses = [
        mcp_server.parse_candidate_text(
            "[C]one\n[G]two\n[Am]three\n[F]four", "sheet-a"
        ),
        mcp_server.score_candidate_against_mir(candidate_json, mir_json),
        mcp_server.rank_candidates_deterministically(candidates_json, mir_json),
        mcp_server.select_candidate_deterministically(candidates_json, "strict", mir_json),
        mcp_server.build_song_baseline(
            candidate_json, "artist--title", "Title", "Artist"
        ),
        mcp_server.validate_song_json(json.dumps(song.model_dump(mode="json"))),
    ]

    assert calls == ["parse", "score", "rank", "select", "baseline", "validate"]
    for response in responses:
        _assert_observation(response)


def test_deterministic_wrappers_do_not_touch_pipeline_model_or_store(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("model, pipeline, or persistence boundary was called")

    monkeypatch.setattr(mcp_server, "run_pipeline_async", forbidden)
    monkeypatch.setattr(mcp_server, "reconcile_admitted", forbidden)
    monkeypatch.setattr(mcp_server, "get_store", forbidden)
    monkeypatch.setattr(mcp_server, "provider_capabilities", forbidden)

    candidate = _candidate()
    baseline = mcp_server.build_song_baseline(
        _json(candidate), "artist--title", "Title", "Artist"
    )
    calls = [
        mcp_server.parse_candidate_text(
            "[C]one\n[G]two\n[Am]three\n[F]four", "sheet-a"
        ),
        mcp_server.score_candidate_against_mir(_json(candidate), _json(_mir())),
        mcp_server.rank_candidates_deterministically(
            json.dumps([candidate.model_dump(mode="json")])
        ),
        mcp_server.select_candidate_deterministically(
            json.dumps([candidate.model_dump(mode="json")]), "best"
        ),
        baseline,
        mcp_server.validate_song_json(json.dumps(baseline["result"]["song"])),
    ]

    assert all(response["ok"] for response in calls)
    assert all(response["modelCalls"] == 0 for response in calls)
    assert all(response["modelCostUSD"] == 0 for response in calls)
