"""Identity admission, collision, and legacy atomic-rename contracts."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server.api import app
from snoocle_server.discovery import CandidateSource
from snoocle_server.identity import resolve_identity_from_evidence
from snoocle_server.schema import Song
from snoocle_server.store import IdentityCollisionError, StoreError
from snoocle_server.store.identity_rename import IdentityRenameError, rename_song_identity
from snoocle_server.store.memory import InMemorySongRepository
from snoocle_server.store.runs import InMemoryRunRepository


def _song(song_id: str, *, audio_hash: str, chord: str = "C") -> Song:
    return Song.model_validate(
        {
            "id": song_id,
            "metadata": {"artist": "Splean", "title": "Romans"},
            "audio": {"contentHash": audio_hash},
            "lines": [
                {
                    "lineIndex": 0,
                    "lyrics": "hello",
                    "chordPlacements": [{"charIndex": 0, "chord": chord}],
                }
            ],
            "provenance": [],
        }
    )


def test_unresolved_identity_is_refused_with_missing_fields_and_evidence(monkeypatch):
    store = InMemorySongRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: store)
    monkeypatch.setattr("snoocle_server.pipeline.get_store", lambda: store)

    response = TestClient(app).post(
        "/v1/songs/analyze",
        json={
            "artist": "Splean",
            "title": "unknown",
            "provider": "mock",
            "skipAudio": True,
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert body["errorCode"] == "identity_unresolved"
    assert body["missing"] == ["title"]
    assert body["evidenceTried"] == ["caller-supplied identity"]
    assert body["needsIdentity"] is True
    assert store.list_songs() == []


def test_mcp_reconcile_refuses_an_unknown_id_before_admission():
    from snoocle_server.mcp_server import reconcile_song

    result = reconcile_song(title="unknown", artist="Splean", provider="mock")

    assert result["errorCode"] == "identity_unresolved"
    assert result["missing"] == ["title"]
    assert result["needsIdentity"] is True


def test_gathered_amdm_slug_resolves_missing_title_before_admission():
    identity = resolve_identity_from_evidence(
        artist="Splean",
        title="unknown",
        candidates=[
            CandidateSource(
                sourceId="amdm-1",
                url="https://amdm.ru/akkordi/splean/romans/",
            )
        ],
    )

    assert (identity.artist, identity.title, identity.method) == ("Splean", "romans", "source-slug")


def test_lrc_catalogue_match_resolves_missing_identity_before_source_fallback():
    identity = resolve_identity_from_evidence(
        artist="unknown",
        title="unknown",
        lrc_match={"trackName": "Vyhoda net", "artistName": "Splean"},
    )

    assert (identity.artist, identity.title, identity.method) == ("Splean", "Vyhoda net", "lrc-match")


def test_audio_hash_collision_refuses_to_append_another_song_to_same_id():
    store = InMemorySongRepository()
    store.save(_song("splean--romans", audio_hash="a" * 64), "first")

    with pytest.raises(IdentityCollisionError) as error:
        store.save(_song("splean--romans", audio_hash="b" * 64), "different recording")

    assert error.value.code == "identity_collision"
    assert store.current_version("splean--romans") is not None
    assert len(store.versions("splean--romans")) == 1


def test_rest_save_reports_identity_collision_code(monkeypatch):
    store = InMemorySongRepository()
    store.save(_song("splean--romans", audio_hash="a" * 64), "first")
    monkeypatch.setattr(api_mod, "get_store", lambda: store)

    response = TestClient(app).post(
        "/v1/songs/splean--romans",
        json={"song": _song("splean--romans", audio_hash="b" * 64).model_dump(), "message": "collision"},
    )

    assert response.status_code == 409
    assert response.json()["errorCode"] == "identity_collision"


def test_set_song_identity_moves_versions_and_runs_atomically():
    songs = InMemorySongRepository()
    runs = InMemoryRunRepository()
    first = songs.save(_song("splean--unknown", audio_hash="a" * 64), "first")
    second_payload = _song("splean--unknown", audio_hash="a" * 64, chord="Am").model_dump()
    second_payload["provenance"] = [
        {"timestamp": "2026-07-31T00:00:00Z", "actor": "test", "action": "rerun"}
    ]
    second_song = Song.model_validate(second_payload)
    second = songs.save(second_song, "second", expected_version=first.version)
    runs.save_run({"runId": "run-1", "songId": "splean--unknown", "startedAt": "2026-07-31T00:00:00Z"})

    result = rename_song_identity(
        songs, runs, "splean--unknown", artist="Splean", title="Bog ustal nas lyubit"
    )

    assert result.new_song_id == "splean--bog-ustal-nas-lyubit"
    assert set(result.version_map) == {first.version, second.version}
    assert songs.get(result.new_song_id).metadata.title == "Bog ustal nas lyubit"
    assert [v.version for v in songs.versions(result.new_song_id)] == [
        result.version_map[second.version], result.version_map[first.version]
    ]
    with pytest.raises(StoreError):
        songs.get("splean--unknown")
    moved_run = runs.get_run("run-1")
    assert moved_run["songId"] == result.new_song_id
    assert moved_run["migratedFromSongId"] == "splean--unknown"


def test_rename_refuses_a_target_conflict_without_moving_anything():
    songs = InMemorySongRepository()
    runs = InMemoryRunRepository()
    songs.save(_song("splean--unknown", audio_hash="a" * 64), "legacy")
    songs.save(_song("splean--romans", audio_hash="b" * 64), "target")
    runs.save_run({"runId": "run-2", "songId": "splean--unknown"})

    with pytest.raises(IdentityRenameError):
        rename_song_identity(songs, runs, "splean--unknown", artist="Splean", title="Romans")

    assert songs.current_version("splean--unknown") is not None
    assert runs.get_run("run-2")["songId"] == "splean--unknown"
