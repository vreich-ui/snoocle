import pytest

from snoocle_server.schema import Song
from snoocle_server.store import GitSongStore, StoreError, VersionConflictError


def make_song(chord="C", prov_extra=None):
    prov = [
        {"timestamp": "2026-07-06T00:00:00Z", "actor": "test", "action": "created"},
    ]
    if prov_extra:
        prov.extend(prov_extra)
    return Song.model_validate(
        {
            "id": "tester--song",
            "metadata": {"title": "Song", "artist": "Tester"},
            "lines": [
                {
                    "lineIndex": 0,
                    "lyrics": "hello world",
                    "chordPlacements": [{"charIndex": 0, "chord": chord}],
                }
            ],
            "provenance": prov,
        }
    )


@pytest.fixture()
def store(tmp_path):
    return GitSongStore(tmp_path / "songstore")


def test_save_and_get_roundtrip(store):
    res = store.save(make_song(), "initial analysis")
    assert res.version
    got = store.get("tester--song")
    assert got.lines[0].chordPlacements[0].chord == "C"


def test_rerun_creates_new_version_keeps_old(store):
    r1 = store.save(make_song("C"), "run 1")
    r2 = store.save(
        make_song("Am", prov_extra=[{"timestamp": "2026-07-06T01:00:00Z", "actor": "test", "action": "re-analyzed"}]),
        "run 2",
    )
    assert r1.version != r2.version
    versions = store.versions("tester--song")
    assert [v.version for v in versions] == [r2.version, r1.version]
    # old version still fully readable
    old = store.get("tester--song", version=r1.version)
    assert old.lines[0].chordPlacements[0].chord == "C"
    new = store.get("tester--song")
    assert new.lines[0].chordPlacements[0].chord == "Am"
    # and diffable via ordinary git
    d = store.diff("tester--song", r1.version, r2.version)
    assert '-          "chord": "C"' in d
    assert '+          "chord": "Am"' in d


def test_optimistic_lock_rejects_stale_writer(store):
    r1 = store.save(make_song("C"), "run 1")
    extra = [{"timestamp": "2026-07-06T01:00:00Z", "actor": "test", "action": "re-analyzed"}]
    # writer A read r1 and saves — fine
    store.save(make_song("Am", prov_extra=extra), "run 2", expected_version=r1.version)
    # writer B also read r1, tries to save — must be rejected
    with pytest.raises(VersionConflictError):
        store.save(make_song("F", prov_extra=extra), "run 3", expected_version=r1.version)


def test_enforce_expected_requires_none_for_new_song(store):
    store.save(make_song("C"), "run 1")
    with pytest.raises(VersionConflictError):
        store.save(make_song("Am"), "conflicting create", expected_version=None, enforce_expected=True)


def test_provenance_is_append_only(store):
    store.save(make_song("C"), "run 1")
    # new version that REWRITES provenance history instead of extending it
    bad = make_song("Am")
    bad = bad.model_copy(
        update={
            "provenance": [
                bad.provenance[0].model_copy(update={"actor": "revisionist"}),
            ]
        }
    )
    with pytest.raises(StoreError, match="append-only"):
        store.save(bad, "history rewrite")


def test_noop_save_rejected(store):
    store.save(make_song("C"), "run 1")
    with pytest.raises(StoreError, match="no changes"):
        store.save(make_song("C"), "identical")


def test_list_songs(store):
    store.save(make_song(), "run 1")
    assert store.list_songs() == ["tester--song"]
