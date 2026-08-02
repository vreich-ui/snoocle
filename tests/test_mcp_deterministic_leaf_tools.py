from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from snoocle_server import mcp_server
from snoocle_server.mir.base import Beat, MirAnalysis
from snoocle_server.reconcile.patch_ops import AppliedOp
from snoocle_server.schema import ProvenanceEntry, Song
from snoocle_server.timing.carry_forward import CarryForwardStats
from snoocle_server.timing.confidence import PlacementScore
from snoocle_server.timing.lrc import LrcLine, LrcMatch
from snoocle_server.timing.offset import OffsetEstimate

from test_schema import make_song


def _song() -> Song:
    return Song.model_validate(make_song())


def _json(value) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value)


def _assert_leaf(response: dict, *, network: str = "none", ok: bool = True) -> None:
    assert response["ok"] is ok
    assert response["modelCalls"] == 0
    assert response["modelCostUSD"] == 0
    assert response["cacheStatus"] == "not_applicable"
    assert response["access"] == {
        "network": network,
        "cache": "none",
        "persistence": "none",
    }
    assert isinstance(response["elapsedMs"], int)
    assert isinstance(response["inputSummary"], dict)
    assert isinstance(response["outputSummary"], dict)


def test_leaf_wrappers_route_to_the_existing_services(monkeypatch, tmp_path):
    song = _song()
    song_json = _json(song)
    mir = MirAnalysis(duration_seconds=12, beats=[Beat(time=1, position=1)])
    mir_json = _json(mir)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    calls: list[str] = []

    @dataclass
    class Decision:
        def to_dict(self):
            return {
                "grade": {"verdict": "unknown"},
                "attribution": {"fault": "unknown"},
                "escalation": {"retry": False, "search": False},
            }

    monkeypatch.setattr(mcp_server, "_analyze_audio", lambda *a, **k: calls.append("full_mir") or mir)
    monkeypatch.setattr(mcp_server, "_analyze_window", lambda *a, **k: calls.append("window_mir") or mir)
    monkeypatch.setattr(mcp_server, "_extend_beat_grid", lambda *a, **k: calls.append("beats") or a[0])
    monkeypatch.setattr(mcp_server, "_snap_chords", lambda *a, **k: calls.append("snap") or song)
    monkeypatch.setattr(
        mcp_server,
        "_carry_forward_timing",
        lambda *a, **k: calls.append("carry") or (song, CarryForwardStats()),
    )
    monkeypatch.setattr(
        mcp_server,
        "_fetch_lrc_match",
        lambda *a, **k: calls.append("lookup")
        or LrcMatch("Title", "Artist", [LrcLine(1.0, "line")]),
    )
    monkeypatch.setattr(
        mcp_server, "_match_lrc_to_lines", lambda *a, **k: calls.append("match") or {0: (1.0, 1.0)}
    )
    monkeypatch.setattr(mcp_server, "_apply_lrc", lambda *a, **k: calls.append("apply_lrc") or song)
    monkeypatch.setattr(mcp_server, "_retime_sections", lambda *a, **k: calls.append("sections") or (song, 1))
    monkeypatch.setattr(
        mcp_server,
        "_guard_collapsed",
        lambda *a, **k: calls.append("collapse")
        or (
            song,
            ProvenanceEntry(
                timestamp="2026-08-02T00:00:00Z", actor="test", action="guard", confidence=1
            ),
        ),
    )
    monkeypatch.setattr(
        mcp_server,
        "_score_song_confidence",
        lambda *a, **k: calls.append("confidence")
        or (song, [PlacementScore(0, 7, "C", 0.5, ["review"])]),
    )
    monkeypatch.setattr(mcp_server, "_evaluate_quality", lambda *a, **k: calls.append("quality") or Decision())
    monkeypatch.setattr(
        mcp_server,
        "_theory_validity",
        lambda *a, **k: calls.append("theory")
        or SimpleNamespace(to_dict=lambda: {"share": 1.0, "explained": 4, "total": 4, "key": "C major"}),
    )
    monkeypatch.setattr(
        mcp_server,
        "_estimate_offset",
        lambda *a, **k: calls.append("offset") or OffsetEstimate(1.25, 0.9),
    )
    monkeypatch.setattr(mcp_server, "_parse_ops_response", lambda *a, **k: calls.append("parse_patch") or [{}])
    monkeypatch.setattr(
        mcp_server,
        "_apply_patch",
        lambda *a, **k: calls.append("patch")
        or (song, [AppliedOp(0, "replace_chord", "changed")]),
    )
    monkeypatch.setattr(
        mcp_server,
        "_build_evidence_manifest",
        lambda *a, **k: calls.append("manifest")
        or {"mir": {"status": "fresh"}, "sources": {"count": 0}, "lrcAlign": {"status": "pending"}},
    )

    responses = [
        mcp_server.analyze_full_track_mir(str(audio)),
        mcp_server.analyze_mir_window(str(audio), 0, 2),
        mcp_server.extend_mir_beat_grid(_json([{"time": 1, "position": 1}]), 12),
        mcp_server.snap_song_to_mir(song_json, mir_json),
        mcp_server.carry_forward_song_timing(song_json, song_json),
        mcp_server.lookup_lrc("Title", "Artist", 12),
        mcp_server.match_lrc_to_song(_json([{"time": 1, "text": "line"}]), song_json),
        mcp_server.apply_lrc_to_song(song_json, _json([{"lineIndex": 0, "timeSeconds": 1, "similarity": 1}])),
        mcp_server.retime_song_sections(song_json, 12),
        mcp_server.guard_song_timing_collapse(song_json, 12),
        mcp_server.score_song_confidence(song_json),
        mcp_server.evaluate_song_quality(song_json),
        mcp_server.validate_song_theory(song_json, "C major"),
        mcp_server.calculate_recording_offset(str(audio), str(audio)),
        mcp_server.apply_deterministic_song_patch(song_json, _json({"ops": [{}]})),
        mcp_server.build_song_evidence_manifest(),
    ]

    assert calls == [
        "full_mir", "window_mir", "beats", "snap", "carry", "lookup", "match",
        "apply_lrc", "sections", "collapse", "confidence", "quality", "theory",
        "offset", "parse_patch", "patch", "manifest",
    ]
    for response in responses:
        _assert_leaf(response, network="lrclib" if response is responses[5] else "none")


def test_native_lrc_match_apply_patch_and_manifest_are_callable():
    song_json = _json(_song())
    lrc = _json(
        [
            {"time": 2.0, "text": "When I find myself in times of trouble"},
            {"time": 6.0, "text": "Mother Mary comes to me"},
        ]
    )

    matched = mcp_server.match_lrc_to_song(lrc, song_json)
    _assert_leaf(matched)
    assert len(matched["result"]["matches"]) == 2

    applied = mcp_server.apply_lrc_to_song(song_json, _json(matched["result"]["matches"]))
    _assert_leaf(applied)
    assert [line["timeSeconds"] for line in applied["result"]["song"]["lines"]] == [2.0, 6.0]

    patched = mcp_server.apply_deterministic_song_patch(
        song_json,
        _json({"ops": [{"op": "replace_chord", "lineIndex": 0, "charIndex": 7, "from": "C", "to": "D"}]}),
    )
    _assert_leaf(patched)
    assert patched["result"]["song"]["lines"][0]["chordPlacements"][0]["chord"] == "D"

    manifest = mcp_server.build_song_evidence_manifest(prior_song_json=song_json)
    _assert_leaf(manifest)
    assert manifest["result"]["manifest"]["priorSong"]["exists"] is True


@pytest.mark.parametrize(
    "response,code",
    [
        (lambda: mcp_server.analyze_mir_window("missing.wav", 2, 1), "file_not_found"),
        (
            lambda: mcp_server.extend_mir_beat_grid(
                _json([{"time": index} for index in range(mcp_server.MAX_BEATS + 1)]), 100
            ),
            "too_many_beats",
        ),
        (
            lambda: mcp_server.apply_lrc_to_song(
                _json(_song()), _json([{"lineIndex": 0, "timeSeconds": -1, "similarity": 2}])
            ),
            "invalid_lrc_matches",
        ),
        (
            lambda: mcp_server.apply_deterministic_song_patch(
                _json(_song()), _json({"ops": [{"op": "unknown"}]})
            ),
            "invalid_patch",
        ),
    ],
)
def test_leaf_bounds_and_failures_are_structured(response, code):
    result = response()
    _assert_leaf(result, ok=False)
    assert result["error"]["code"] == code
    assert "traceback" not in json.dumps(result).lower()


def test_leaf_tools_never_touch_pipeline_model_cache_or_store(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden boundary called")

    monkeypatch.setattr(mcp_server, "run_pipeline_async", forbidden)
    monkeypatch.setattr(mcp_server, "reconcile_admitted", forbidden)
    monkeypatch.setattr(mcp_server, "get_store", forbidden)
    monkeypatch.setattr(mcp_server, "provider_capabilities", forbidden)

    song_json = _json(_song())
    responses = [
        mcp_server.snap_song_to_mir(song_json),
        mcp_server.retime_song_sections(song_json),
        mcp_server.guard_song_timing_collapse(song_json),
        mcp_server.validate_song_theory(song_json),
        mcp_server.build_song_evidence_manifest(prior_song_json=song_json),
    ]
    for response in responses:
        _assert_leaf(response)
