"""Auditing and repairing stored song ids from failed identity resolution.

The motivating state: eight stored documents (unknown--official-video,
wil-per--rolling-stones-paint-it-black-live-1966, and others) were minted
under an id derived from a bad guess, before PR #45/#51 tightened identity
resolution. Ids are content-hash permanent, so they cannot be corrected in
place — this module only detects the mismatch and, on explicit request,
re-analyzes under today's correct answer and points the old document
forward. Everything here runs against an in-memory store with injected
resolve/pipeline/gold callables: no network, no LLM, ever.
"""

from __future__ import annotations

import pytest

from snoocle_server.identity import IdentityError, SongIdentity
from snoocle_server.schema.song import Song
from snoocle_server.store.identity_audit import (
    check_song_identity,
    find_mismatched_identities,
    reanalyze_and_supersede,
    supersede_song,
)
from snoocle_server.store.memory import InMemorySongRepository

from test_schema import make_song

_OLD_ID = "the-beatles--let-it-be"  # make_song()'s id, video QDYfEBY9NM4


def _stored(store: InMemorySongRepository, **overrides) -> Song:
    song = Song.model_validate(make_song(**overrides))
    store.save(song, message="seed")
    return song


def _matching_identity(*_a, **_k) -> SongIdentity:
    return SongIdentity(title="Let It Be", artist="The Beatles", confidence=0.95, method="structured")


def _mismatched_identity(*_a, **_k) -> SongIdentity:
    return SongIdentity(title="Hook", artist="Blues Traveler", confidence=0.8, method="dash")


def _refusing_identity(*_a, **_k):
    raise IdentityError("no separator, channel is not a trustworthy artist")


class _RaisingPipeline:
    """A run_pipeline stand-in that fails the test if it is ever called —
    for asserting a refusal short-circuits before the heavy pipeline runs."""

    def __call__(self, *a, **k):
        raise AssertionError("run_pipeline must not be called")


class _FakePipelineReport:
    def __init__(self, song_id: str, stored_version: str, steps: dict):
        self.song_id = song_id
        self.stored_version = stored_version
        self.steps = steps


# --- check_song_identity: detection ------------------------------------------


def test_check_song_identity_reports_a_match():
    song = Song.model_validate(make_song())
    check = check_song_identity(song, resolve=_matching_identity)
    assert check.matches is True
    assert check.status == "match"
    assert check.resolved_id == _OLD_ID
    assert check.confidence == 0.95


def test_check_song_identity_reports_a_mismatch():
    song = Song.model_validate(make_song())
    check = check_song_identity(song, resolve=_mismatched_identity)
    assert check.matches is False
    assert check.status == "mismatch"
    assert check.resolved_id == "blues-traveler--hook"
    assert check.confidence == 0.8
    assert check.song_id == _OLD_ID  # current id preserved for the report


def test_check_song_identity_no_video_id_is_reported_but_not_a_mismatch():
    song = Song.model_validate(make_song(audio={"syncMap": []}))  # no youtubeVideoId
    check = check_song_identity(song, resolve=_mismatched_identity)
    assert check.status == "no_video_id"
    assert check.matches is True
    assert check.resolved_id is None


def test_check_song_identity_still_unresolvable_today():
    song = Song.model_validate(make_song())
    check = check_song_identity(song, resolve=_refusing_identity)
    assert check.status == "unresolvable"
    assert check.matches is False
    assert check.resolved_id is None
    assert "no separator" in check.detail


# --- find_mismatched_identities: the store-wide report -----------------------


def test_find_mismatched_identities_reports_only_the_mismatch():
    store = InMemorySongRepository()
    _stored(store)  # the-beatles--let-it-be, will match
    _stored(store, id="unknown--hook-at-howard-stern", metadata={"title": "Unknown", "artist": "Unknown"})

    # Both songs share the same fixture video id (QDYfEBY9NM4) since make_song()
    # doesn't vary it; resolve the first call to match and the rest to mismatch,
    # standing in for two different videos resolving two different ways.
    def resolve_by_call(video_id: str, _seen=[]) -> SongIdentity:
        _seen.append(video_id)
        return _matching_identity() if len(_seen) == 1 else _mismatched_identity()

    checks = find_mismatched_identities(store, resolve=resolve_by_call)
    assert [c.song_id for c in checks] == ["unknown--hook-at-howard-stern"]
    assert checks[0].status == "mismatch"


def test_a_song_whose_id_already_matches_is_skipped():
    store = InMemorySongRepository()
    _stored(store)
    checks = find_mismatched_identities(store, resolve=_matching_identity)
    assert checks == []


def test_find_mismatched_identities_skips_songs_with_no_video_id():
    store = InMemorySongRepository()
    _stored(store, audio={"syncMap": []})
    checks = find_mismatched_identities(store, resolve=_mismatched_identity)
    assert checks == []


def test_find_mismatched_identities_can_scope_to_one_song():
    store = InMemorySongRepository()
    _stored(store)
    _stored(store, id="unknown--other", metadata={"title": "Unknown", "artist": "Unknown"})
    checks = find_mismatched_identities(store, resolve=_mismatched_identity, song_ids=[_OLD_ID])
    assert [c.song_id for c in checks] == [_OLD_ID]


# --- supersede_song: the forward pointer --------------------------------------


def test_supersede_song_writes_a_pointer_and_leaves_the_original_readable():
    store = InMemorySongRepository()
    _stored(store)
    original_version = store.current_version(_OLD_ID)

    result = supersede_song(store, _OLD_ID, "blues-traveler--hook")

    updated = store.get(_OLD_ID)
    assert updated.supersededBy == "blues-traveler--hook"
    assert updated.lines == Song.model_validate(make_song()).lines  # content untouched
    assert updated.provenance[-1].action == "superseded"
    assert updated.provenance[-1].sources == ["blues-traveler--hook"]

    # nothing was deleted: the version BEFORE the supersede is still readable,
    # and it does NOT carry the pointer -- history is append-only.
    original = store.get(_OLD_ID, version=original_version)
    assert original.supersededBy is None
    assert store.current_version(_OLD_ID) == result.version
    assert result.version != original_version


def test_supersede_song_is_idempotent():
    store = InMemorySongRepository()
    _stored(store)
    supersede_song(store, _OLD_ID, "blues-traveler--hook")
    v1 = store.current_version(_OLD_ID)
    prov_len = len(store.get(_OLD_ID).provenance)

    supersede_song(store, _OLD_ID, "blues-traveler--hook")  # again, same target
    assert store.current_version(_OLD_ID) == v1
    assert len(store.get(_OLD_ID).provenance) == prov_len


# --- reanalyze_and_supersede: dry run by default, gated refusals -------------


def test_dry_run_writes_nothing():
    store = InMemorySongRepository()
    _stored(store)
    original_version = store.current_version(_OLD_ID)

    report = reanalyze_and_supersede(
        store, _OLD_ID, confirm=False, resolve=_mismatched_identity,
        run_pipeline=_RaisingPipeline(), gold_getter=lambda sid: None,
    )

    assert report.dry_run is True
    assert report.refused is None
    assert report.resolved_id == "blues-traveler--hook"
    assert report.confidence == 0.8
    assert report.superseded_version is None
    # store literally unchanged
    assert store.current_version(_OLD_ID) == original_version
    assert store.get(_OLD_ID).supersededBy is None


def test_confirm_runs_the_pipeline_and_supersedes():
    store = InMemorySongRepository()
    _stored(store)

    def fake_run_pipeline(*, title, artist, youtube_url_or_id, provider, model, store):
        assert title is None and artist is None
        assert youtube_url_or_id == "QDYfEBY9NM4"
        return _FakePipelineReport("blues-traveler--hook", "abc123def456", {"reconcile": "ok: provider=mock"})

    report = reanalyze_and_supersede(
        store, _OLD_ID, confirm=True, resolve=_mismatched_identity,
        run_pipeline=fake_run_pipeline, gold_getter=lambda sid: None,
    )

    assert report.dry_run is False
    assert report.refused is None
    assert report.new_id == "blues-traveler--hook"
    assert report.new_song_version == "abc123def456"
    assert report.superseded_version is not None
    assert report.steps == {"reconcile": "ok: provider=mock"}

    updated = store.get(_OLD_ID)
    assert updated.supersededBy == "blues-traveler--hook"


def test_refuses_when_id_already_matches_and_never_calls_the_pipeline():
    store = InMemorySongRepository()
    _stored(store)
    report = reanalyze_and_supersede(
        store, _OLD_ID, confirm=True, resolve=_matching_identity,
        run_pipeline=_RaisingPipeline(), gold_getter=lambda sid: None,
    )
    assert "already matches" in report.refused
    assert report.superseded_version is None


def test_refuses_when_gold_marked_even_with_confirm():
    store = InMemorySongRepository()
    _stored(store)
    report = reanalyze_and_supersede(
        store, _OLD_ID, confirm=True, resolve=_mismatched_identity,
        run_pipeline=_RaisingPipeline(), gold_getter=lambda sid: "goldversion123",
    )
    assert "gold" in report.refused.lower()
    assert "goldversion123" in report.refused
    assert report.superseded_version is None
    assert store.get(_OLD_ID).supersededBy is None


def test_refuses_when_no_video_id_to_reresolve_from():
    store = InMemorySongRepository()
    _stored(store, audio={"syncMap": []})
    report = reanalyze_and_supersede(
        store, _OLD_ID, confirm=True, resolve=_mismatched_identity,
        run_pipeline=_RaisingPipeline(), gold_getter=lambda sid: None,
    )
    assert "no audio.youtubeVideoId" in report.refused


def test_refuses_when_already_superseded():
    store = InMemorySongRepository()
    _stored(store)
    supersede_song(store, _OLD_ID, "some-other--song")
    report = reanalyze_and_supersede(
        store, _OLD_ID, confirm=True, resolve=_mismatched_identity,
        run_pipeline=_RaisingPipeline(), gold_getter=lambda sid: None,
    )
    assert "already superseded" in report.refused


def test_refuses_when_identity_still_unresolvable_today():
    store = InMemorySongRepository()
    _stored(store)
    report = reanalyze_and_supersede(
        store, _OLD_ID, confirm=True, resolve=_refusing_identity,
        run_pipeline=_RaisingPipeline(), gold_getter=lambda sid: None,
    )
    assert "still can't resolve" in report.refused


def test_reanalyze_unknown_song_id_refuses_cleanly():
    store = InMemorySongRepository()
    report = reanalyze_and_supersede(store, "nope--not-stored", confirm=True)
    assert "cannot read" in report.refused
