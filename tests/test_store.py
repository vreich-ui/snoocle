"""SongRepository behavior — one suite, run against every backend.

The in-memory backend runs everywhere (hermetic). The Firestore backend runs
only when a Firestore emulator is reachable (FIRESTORE_EMULATOR_HOST set) and
google-cloud-firestore is installed; otherwise those params skip. Both must
behave identically: content-hash versions, optimistic locking (409),
append-only provenance, newest-first history, and JSON diffs.

Run the Firestore params locally with:

    gcloud emulators firestore start --host-port=127.0.0.1:8080
    export FIRESTORE_EMULATOR_HOST=127.0.0.1:8080 GOOGLE_CLOUD_PROJECT=snoocle-test
    pytest tests/test_store.py
"""

from __future__ import annotations

import os
import uuid

import pytest

from snoocle_server.schema import Song
from snoocle_server.store.base import version_sha
from snoocle_server.store.memory import InMemorySongRepository
from snoocle_server.store import StoreError, VersionConflictError


def make_song(chord: str = "C", prov_extra: list[dict] | None = None) -> Song:
    prov = [{"timestamp": "2026-07-06T00:00:00Z", "actor": "test", "action": "created"}]
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


_RE = [{"timestamp": "2026-07-06T01:00:00Z", "actor": "test", "action": "re-analyzed"}]


def _wipe_firestore(repo) -> None:
    for song_ref in repo._collection.list_documents():
        for v in song_ref.collection("versions").list_documents():
            v.delete()
        song_ref.delete()


@pytest.fixture(params=["memory", "firestore"])
def repo(request):
    if request.param == "memory":
        yield InMemorySongRepository()
        return
    if not os.environ.get("FIRESTORE_EMULATOR_HOST"):
        pytest.skip("Firestore emulator not running (set FIRESTORE_EMULATOR_HOST)")
    pytest.importorskip("google.cloud.firestore")
    from snoocle_server.store.firestore_store import FirestoreSongRepository

    r = FirestoreSongRepository(
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "snoocle-test"),
        collection="songs_test_" + uuid.uuid4().hex[:8],
    )
    try:
        yield r
    finally:
        _wipe_firestore(r)


def test_save_and_get_roundtrip(repo):
    res = repo.save(make_song(), "initial analysis")
    assert res.version and res.timestamp and res.message == "initial analysis"
    got = repo.get("tester--song")
    assert got.lines[0].chordPlacements[0].chord == "C"
    assert repo.current_version("tester--song") == res.version


def test_version_is_content_hash(repo):
    res = repo.save(make_song("C"), "run 1")
    assert res.version == version_sha(make_song("C"))  # backend-independent


def test_rerun_creates_new_version_keeps_old(repo):
    r1 = repo.save(make_song("C"), "run 1")
    r2 = repo.save(make_song("Am", prov_extra=_RE), "run 2", expected_version=r1.version)
    assert r1.version != r2.version

    versions = repo.versions("tester--song")
    assert [v.version for v in versions] == [r2.version, r1.version]  # newest first

    assert repo.get("tester--song", version=r1.version).lines[0].chordPlacements[0].chord == "C"
    assert repo.get("tester--song").lines[0].chordPlacements[0].chord == "Am"

    d = repo.diff("tester--song", r1.version, r2.version)
    assert any(ln.startswith("-") and '"chord": "C"' in ln for ln in d.splitlines())
    assert any(ln.startswith("+") and '"chord": "Am"' in ln for ln in d.splitlines())


def test_optimistic_lock_rejects_stale_writer(repo):
    r1 = repo.save(make_song("C"), "run 1")
    # writer A read r1 and saves — fine
    repo.save(make_song("Am", prov_extra=_RE), "run 2", expected_version=r1.version)
    # writer B also read r1, tries to save — must be rejected
    with pytest.raises(VersionConflictError):
        repo.save(make_song("F", prov_extra=_RE), "run 3", expected_version=r1.version)


def test_enforce_expected_requires_none_for_new_song(repo):
    repo.save(make_song("C"), "run 1")
    with pytest.raises(VersionConflictError):
        repo.save(make_song("Am"), "conflicting create", expected_version=None, enforce_expected=True)


def test_provenance_is_append_only(repo):
    repo.save(make_song("C"), "run 1")
    bad = make_song("Am")
    bad = bad.model_copy(
        update={"provenance": [bad.provenance[0].model_copy(update={"actor": "revisionist"})]}
    )
    with pytest.raises(StoreError, match="append-only"):
        repo.save(bad, "history rewrite")


def test_identical_content_save_is_idempotent(repo):
    r1 = repo.save(make_song("C"), "run 1")
    r2 = repo.save(make_song("C"), "identical")  # same content -> same sha
    assert r2.version == r1.version
    assert [v.version for v in repo.versions("tester--song")] == [r1.version]


def test_list_songs(repo):
    assert repo.list_songs() == []
    repo.save(make_song(), "run 1")
    assert repo.list_songs() == ["tester--song"]


def test_get_missing_song_raises(repo):
    with pytest.raises(StoreError):
        repo.get("nope--nothing")
    assert repo.versions("nope--nothing") == []
    assert repo.current_version("nope--nothing") is None


def _seed_raw(repo, song_id: str, raw_song: dict, version: str = "legacy0000") -> None:
    """Inject a raw (unvalidated) song blob straight into the backend, the
    way a document written before the Song schema existed -- or before some
    invariant was added -- would already sit in the database. `save()` always
    takes a validated `Song`, so it can't produce this state; real legacy rows
    can.
    """
    from snoocle_server.store.memory import InMemorySongRepository, _Record, _Version

    if isinstance(repo, InMemorySongRepository):
        repo._songs[song_id] = _Record(
            title=raw_song.get("metadata", {}).get("title", ""),
            artist=raw_song.get("metadata", {}).get("artist", ""),
            latest_version=version,
            updated_at="2026-01-01T00:00:00+00:00",
            versions={version: _Version(song=raw_song, message="legacy seed", timestamp="2026-01-01T00:00:00+00:00", parent=None)},
            order=[version],
        )
        return
    # Firestore: write the same shape save() writes, skipping Song validation.
    ts = "2026-01-01T00:00:00+00:00"
    repo._version_ref(song_id, version).set(
        {"song": raw_song, "message": "legacy seed", "timestamp": ts, "parent": None}
    )
    repo._song_ref(song_id).set(
        {
            "song": raw_song,
            "title": raw_song.get("metadata", {}).get("title", ""),
            "artist": raw_song.get("metadata", {}).get("artist", ""),
            "latestVersion": version,
            "updatedAt": ts,
            "youtubeVideoId": raw_song.get("audio", {}).get("youtubeVideoId"),
            "hasTiming": False,
        }
    )


# A v1-era document exactly as it would have sat in Firestore before schema
# v2 existed: no `schemaVersion` key, none of the v2-only optional fields
# (ChordPlacement.timeSeconds/confidence/beat/voicingHint, Line.timeSeconds/
# confidence, AudioInfo.analyzedVideoId/videoOffsets/beats) present at all.
_V1_ERA_DOC = {
    "id": "bob-marley--three-little-birds",
    "metadata": {"title": "Three Little Birds", "artist": "Bob Marley"},
    "displayPreferences": {"capo": 0, "tuning": "standard"},
    "audio": {"youtubeVideoId": "HHYSSSAKfMU", "durationSeconds": 180.0, "syncMap": []},
    "sections": [],
    "lines": [
        {
            "lineIndex": 0,
            "lyrics": "Don't worry about a thing",
            "chordPlacements": [{"charIndex": 0, "chord": "A"}],
        }
    ],
    "provenance": [
        {
            "timestamp": "2025-01-01T00:00:00Z",
            "actor": "snoocle-server/0.1.0",
            "action": "reconciled",
            "sources": [],
            "confidence": 0.9,
        }
    ],
}


def test_v1_era_document_round_trips_through_read_path(repo):
    """A stored document from before schema v2 existed -- no schemaVersion,
    none of the v2-only fields -- must still come back from get() exactly
    like any other song. This is the read path get_song/list_songs actually
    use, not just Song.model_validate() in isolation."""
    _seed_raw(repo, "bob-marley--three-little-birds", _V1_ERA_DOC)

    assert "bob-marley--three-little-birds" in repo.list_songs()

    got = repo.get("bob-marley--three-little-birds")
    assert got.metadata.title == "Three Little Birds"
    assert got.metadata.artist == "Bob Marley"
    assert got.lines[0].lyrics == "Don't worry about a thing"
    assert got.lines[0].chordPlacements[0].chord == "A"
    # v2-only fields fill in their defaults for a v1-era document.
    assert got.schemaVersion == 2
    assert got.audio.beats == []
    assert got.audio.videoOffsets == {}
    assert got.lines[0].timeSeconds is None


def test_genuinely_invalid_document_reports_clearly_without_being_rewritten(repo):
    """A stored document that predates the sounding-harmony chord rule (e.g.
    an old "N.C." no-chord placeholder -- see docs/ARCHITECTURE.md's N.C.
    policy) is genuinely invalid under the current schema. get() must not
    crash with a bare pydantic ValidationError, and must not silently drop
    or rewrite the bad placement -- it should raise a clear, typed error
    naming the song so an operator can decide what to do with it."""
    bad_doc = {
        "id": "guns-n-roses--don-t-cry",
        "metadata": {"title": "Don't Cry", "artist": "Guns N' Roses"},
        "lines": [
            {
                "lineIndex": 0,
                "lyrics": "Talk to me softly",
                "chordPlacements": [{"charIndex": 0, "chord": "N.C."}],
            }
        ],
        "provenance": [],
    }
    _seed_raw(repo, "guns-n-roses--don-t-cry", bad_doc)

    # list_songs / list_song_summaries never fully validate, so the id still
    # shows up there even though the document itself can't be read.
    assert "guns-n-roses--don-t-cry" in repo.list_songs()

    from snoocle_server.store import CorruptSongError

    with pytest.raises(CorruptSongError, match="guns-n-roses--don-t-cry"):
        repo.get("guns-n-roses--don-t-cry")

    # The stored document itself is untouched -- no auto-migration.
    if hasattr(repo, "_songs"):
        raw = repo._songs["guns-n-roses--don-t-cry"].versions["legacy0000"].song
    else:
        raw = repo._version_ref("guns-n-roses--don-t-cry", "legacy0000").get().to_dict()["song"]
    assert raw["lines"][0]["chordPlacements"][0]["chord"] == "N.C."


def test_youtube_cookies_persist(repo):
    assert repo.youtube_cookies_status() is None
    assert repo.get_youtube_cookies_txt() is None

    txt = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tabc\n"
    rec = repo.set_youtube_cookies(txt, source="app")
    assert rec.source == "app" and rec.line_count == 1 and rec.updated_at

    assert repo.get_youtube_cookies_txt() == txt
    status = repo.youtube_cookies_status()
    assert status.source == "app" and status.line_count == 1

    repo.clear_youtube_cookies()
    assert repo.youtube_cookies_status() is None
    assert repo.get_youtube_cookies_txt() is None


def test_firestore_database_config(monkeypatch):
    """FIRESTORE_DATABASE targets a NAMED database; the project id (a different
    thing) stays in GOOGLE_CLOUD_PROJECT. Defaults to Firestore's "(default)"."""
    from snoocle_server.config import Settings

    monkeypatch.delenv("FIRESTORE_DATABASE", raising=False)
    monkeypatch.delenv("SNOOCLE_FIRESTORE_DATABASE", raising=False)
    assert Settings(_env_file=None).firestore_database == "(default)"

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0481163044")
    monkeypatch.setenv("FIRESTORE_DATABASE", "snoocle-db")
    s = Settings(_env_file=None)
    assert s.firestore_database == "snoocle-db"  # the named database
    assert s.google_cloud_project == "gen-lang-client-0481163044"  # still the project id

    # legacy SNOOCLE_-prefixed name still honored
    monkeypatch.delenv("FIRESTORE_DATABASE", raising=False)
    monkeypatch.setenv("SNOOCLE_FIRESTORE_DATABASE", "legacy-db")
    assert Settings(_env_file=None).firestore_database == "legacy-db"


def test_firestore_infra_errors_translate_to_store_unavailable():
    """Firestore backend failures (DB missing, permission denied, unavailable)
    surface as StoreUnavailableError; our own errors and real bugs pass through."""
    from google.api_core import exceptions as gexc

    from snoocle_server.store import StoreUnavailableError, VersionConflictError
    from snoocle_server.store.firestore_store import _is_infra_error, _translate_infra_errors

    assert _is_infra_error(gexc.NotFound("db missing")) is True
    assert _is_infra_error(gexc.PermissionDenied("nope")) is True
    assert _is_infra_error(gexc.ServiceUnavailable("down")) is True
    assert _is_infra_error(ValueError("real bug")) is False

    class Dummy:
        @_translate_infra_errors
        def db_missing(self):
            raise gexc.NotFound("the database (default) does not exist")

        @_translate_infra_errors
        def conflict(self):
            raise VersionConflictError("stale")

        @_translate_infra_errors
        def real_bug(self):
            raise ValueError("a genuine bug we must not mask")

    d = Dummy()
    with pytest.raises(StoreUnavailableError):
        d.db_missing()
    with pytest.raises(VersionConflictError):  # our own error passes through
        d.conflict()
    with pytest.raises(ValueError):  # non-infra bug is not masked
        d.real_bug()
