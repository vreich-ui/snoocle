"""Dry-run grouping for the three-song Splean legacy collision."""

from __future__ import annotations

from snoocle_server.schema import Song
from snoocle_server.store.memory import InMemorySongRepository, _Record, _Version
from snoocle_server.store.runs import InMemoryRunRepository
from snoocle_server.store.splean_migration import plan_splean_identity_migration


def _legacy_song(song_id: str, url: str, audio_hash: str) -> Song:
    return Song.model_validate(
        {
            "id": song_id,
            "metadata": {"artist": "Unknown", "title": "Unknown"},
            "audio": {"contentHash": audio_hash},
            "lines": [],
            "provenance": [
                {
                    "timestamp": "2026-07-31T07:03:13Z",
                    "actor": "test",
                    "action": "discovered-sources",
                    "sources": [url],
                }
            ],
        }
    )


def _seed(repo: InMemorySongRepository, song_id: str, version: str, song: Song) -> None:
    repo._songs[song_id] = _Record(
        title=song.metadata.title,
        artist=song.metadata.artist,
        latest_version=version,
        updated_at="2026-07-31T07:03:13Z",
        versions={version: _Version(song=song.model_dump(mode="json"), message="legacy", timestamp="2026-07-31T07:03:13Z", parent=None)},
        order=[version],
    )


def test_dry_run_groups_the_three_splean_songs_by_urls_and_legacy_id():
    songs = InMemorySongRepository()
    runs = InMemoryRunRepository()
    # Add two immutable snapshots under the colliding id — exactly the state
    # that cannot be repaired by the normal whole-document rename endpoint.
    romans = _legacy_song("unknown--unknown", "https://amdm.ru/akkordi/splean/romans/", "a" * 64)
    vyhoda = _legacy_song("unknown--unknown", "https://amdm.ru/akkordi/splean/vyhoda-net/", "b" * 64)
    songs._songs["unknown--unknown"] = _Record(
        title="Unknown",
        artist="Unknown",
        latest_version="v-vyhoda",
        updated_at="2026-07-31T07:03:13Z",
        versions={
            "v-romans": _Version(song=romans.model_dump(mode="json"), message="legacy", timestamp="2026-07-31T07:03:13Z", parent=None),
            "v-vyhoda": _Version(song=vyhoda.model_dump(mode="json"), message="legacy", timestamp="2026-07-31T07:03:14Z", parent="v-romans"),
        },
        order=["v-romans", "v-vyhoda"],
    )
    bog = _legacy_song("splean--unknown", "https://amdm.ru/akkordi/splean/bog-ustal-nas-lyubit/", "c" * 64)
    _seed(songs, "splean--unknown", "v-bog", bog)
    runs.save_run({"runId": "r-romans", "songId": "unknown--unknown", "source": "https://amdm.ru/akkordi/splean/romans/"})
    runs.save_run({"runId": "r-vyhoda", "songId": "unknown--unknown", "evidence": {"url": "https://amdm.ru/akkordi/splean/vyhoda-net/"}})
    runs.save_run({"runId": "r-bog", "songId": "splean--unknown"})

    plan = plan_splean_identity_migration(songs, runs)
    got = {move.target_song_id: (move.versions, move.run_ids) for move in plan.moves}

    assert got == {
        "splean--romans": (["v-romans"], ["r-romans"]),
        "splean--vyhoda-net": (["v-vyhoda"], ["r-vyhoda"]),
        "splean--bog-ustal-nas-lyubit": (["v-bog"], ["r-bog"]),
    }
    assert plan.unresolved_versions == []
    assert plan.unresolved_runs == []
    rendered = plan.describe()
    assert "DRY RUN — no writes" in rendered
    assert "unknown--unknown -> splean--romans" in rendered
    assert "unknown--unknown -> splean--vyhoda-net" in rendered
    assert "splean--unknown -> splean--bog-ustal-nas-lyubit" in rendered
