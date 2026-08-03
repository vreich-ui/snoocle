"""Per-song reconciliation notes: the store, the REST + MCP surfaces, and the
BOUNDED replay.

Replay is load-bearing but not unconditional, because two instructions that look
alike want opposite lifetimes: a durable preference ("capo-free voicings please",
curated through the notes surface) must shape every later analysis of that song,
while a pending correction ("change the C to a B in line 12", attached to one
analyze request) must be applied once — it survives a run that dies before
storing, and stops replaying as soon as it has landed in a stored version.

The two lifetimes are stored INDEPENDENTLY (two fields of one record, not one
field with a `kind` discriminator): writing one must never clobber the other,
and both may be in force at once, in which case a run's guidance is the two
combined (see `song_notes.combine_guidance`). Either way it is keyed by the
same song id the pipeline resolves and is visible in provenance rather than
applied silently.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server import pipeline as pipeline_mod
from snoocle_server.api import app
from snoocle_server.reconcile.engine import ReconcileError
from snoocle_server.schema.song import slugify_song_id
from snoocle_server.store.memory import InMemorySongRepository
from snoocle_server.store.runs import InMemoryRunRepository, reset_run_store
from snoocle_server.store.song_notes import (
    MAX_NOTES_CHARS,
    InMemorySongNotesStore,
    build_song_notes_store,
    combine_guidance,
    get_song_notes_store,
    replay_guidance,
    reset_song_notes_store,
)

SONG_ID = slugify_song_id("Tester", "Fixme")  # what analyze below resolves to


# --- store contract (both backends) -----------------------------------------


@pytest.fixture(params=["memory", "firestore"])
def notes_store(request):
    if request.param == "memory":
        yield InMemorySongNotesStore()
        return
    if not os.environ.get("FIRESTORE_EMULATOR_HOST"):
        pytest.skip("Firestore emulator not running (set FIRESTORE_EMULATOR_HOST)")
    pytest.importorskip("google.cloud.firestore")
    from snoocle_server.store.song_notes import FirestoreSongNotesStore

    store = FirestoreSongNotesStore(project=os.environ.get("GOOGLE_CLOUD_PROJECT", "snoocle-test"))
    store._COLLECTION = "song_notes_test_" + uuid.uuid4().hex[:8]
    try:
        yield store
    finally:
        for doc in store._client.collection(store._COLLECTION).list_documents():
            doc.delete()


def test_notes_store_roundtrip(notes_store):
    assert notes_store.get("nobody--nothing") == ""
    assert notes_store.get_record("nobody--nothing") is None

    assert notes_store.set_preference(SONG_ID, "  the bridge is Bm  ") == "the bridge is Bm"
    assert notes_store.get(SONG_ID) == "the bridge is Bm"  # stored trimmed
    rec = notes_store.get_record(SONG_ID)
    assert rec["preference"]["notes"] == "the bridge is Bm" and rec["preference"]["updated_at"]
    assert rec["correction"] is None

    notes_store.set_preference(SONG_ID, "second take")
    assert notes_store.get(SONG_ID) == "second take"


def test_notes_store_empty_deletes(notes_store):
    notes_store.set_preference(SONG_ID, "temporary")
    assert notes_store.set_preference(SONG_ID, "   ") == ""
    assert notes_store.get(SONG_ID) == ""
    assert notes_store.get_record(SONG_ID) is None


def test_preference_and_correction_are_stored_independently(notes_store):
    """The whole fix: setting one lifetime must never disturb the other, in
    either direction, and both may be in force at once."""
    notes_store.set_preference(SONG_ID, "capo-free voicings please")
    notes_store.set_correction(SONG_ID, "change the C to a B")

    rec = notes_store.get_record(SONG_ID)
    assert rec["preference"]["notes"] == "capo-free voicings please"
    assert rec["correction"]["notes"] == "change the C to a B"
    assert rec["correction"]["applied_to_version"] is None  # not applied yet

    # consuming the correction must not touch the preference
    notes_store.mark_applied(SONG_ID, "change the C to a B", "v1")
    rec = notes_store.get_record(SONG_ID)
    assert rec["preference"]["notes"] == "capo-free voicings please"
    assert rec["correction"]["applied_to_version"] == "v1"

    # a FRESH correction replaces the spent one (still single-shot: only one
    # may ever be outstanding) but the preference survives untouched — this is
    # the exact regression: the old design's `set()` replaced the WHOLE
    # record, so this call used to erase "capo-free voicings please"
    notes_store.set_correction(SONG_ID, "a fresh instruction")
    rec = notes_store.get_record(SONG_ID)
    assert rec["preference"]["notes"] == "capo-free voicings please"
    assert rec["correction"]["notes"] == "a fresh instruction"
    assert rec["correction"]["applied_to_version"] is None

    # and the reverse: replacing the preference must not touch the correction
    notes_store.set_preference(SONG_ID, "actually, capo on 2")
    rec = notes_store.get_record(SONG_ID)
    assert rec["preference"]["notes"] == "actually, capo on 2"
    assert rec["correction"]["notes"] == "a fresh instruction"


def test_clearing_one_lifetime_leaves_the_other_in_place(notes_store):
    """Clearing ONE slot is done by writing it empty — the same call the REST
    PUT and `set_song_notes` make with an empty body, and the only per-slot
    clear the store offers (there is no `delete_preference`/`delete_correction`:
    both surfaces state a partial delete is deliberately not exposed, and
    `delete` below is the whole-slate one). Either direction must leave the
    other lifetime untouched."""
    notes_store.set_preference(SONG_ID, "capo-free voicings please")
    notes_store.set_correction(SONG_ID, "change the C to a B")

    assert notes_store.set_correction(SONG_ID, "  ") == ""
    rec = notes_store.get_record(SONG_ID)
    assert rec["preference"]["notes"] == "capo-free voicings please"
    assert rec["correction"] is None

    notes_store.set_correction(SONG_ID, "change the C to a B")
    assert notes_store.set_preference(SONG_ID, "") == ""
    rec = notes_store.get_record(SONG_ID)
    assert rec["preference"] is None
    assert rec["correction"]["notes"] == "change the C to a B"

    # and the whole slate goes in one call, the only delete there is
    assert notes_store.delete(SONG_ID) is True
    assert notes_store.get_record(SONG_ID) is None


def test_mark_applied_consumes_only_the_pending_correction_it_applied(notes_store):
    # a durable preference is never consumed — that is what makes it durable,
    # and it has nothing for mark_applied to even compare against
    notes_store.set_preference(SONG_ID, "capo-free voicings please")
    assert notes_store.mark_applied(SONG_ID, "capo-free voicings please", "v1") is False
    assert notes_store.get_record(SONG_ID)["preference"]["notes"] == "capo-free voicings please"

    notes_store.set_correction(SONG_ID, "change the C to a B")
    # a note the caller replaced mid-run belongs to the NEXT run, not to the run
    # that never saw it
    assert notes_store.mark_applied(SONG_ID, "an older instruction", "v1") is False
    assert notes_store.mark_applied(SONG_ID, "change the C to a B", "v1") is True
    assert notes_store.get_record(SONG_ID)["correction"]["applied_to_version"] == "v1"
    # and only ever once
    assert notes_store.mark_applied(SONG_ID, "change the C to a B", "v2") is False
    assert notes_store.get_record(SONG_ID)["correction"]["applied_to_version"] == "v1"
    assert notes_store.mark_applied("nobody--nothing", "anything", "v1") is False


def test_replay_guidance_bounds_a_correction_but_never_a_preference():
    assert replay_guidance(None) == (None, None)
    assert replay_guidance({"preference": None, "correction": None}) == (None, None)
    assert replay_guidance({"preference": {"notes": "   "}, "correction": None}) == (None, None)

    combined, pending = replay_guidance({"preference": {"notes": "x"}, "correction": None})
    assert combined == "x" and pending is None  # preference alone: bare, nothing pending

    combined, pending = replay_guidance(
        {"preference": None, "correction": {"notes": "y", "applied_to_version": None}}
    )
    assert combined == "y" and pending == "y"  # unspent correction alone: bare too

    combined, pending = replay_guidance(
        {"preference": None, "correction": {"notes": "y", "applied_to_version": "v1"}}
    )
    assert combined is None and pending is None  # spent — nothing left to replay


def test_replay_guidance_combines_both_when_both_are_in_force():
    combined, pending = replay_guidance(
        {
            "preference": {"notes": "capo-free voicings please"},
            "correction": {"notes": "change the C to a B", "applied_to_version": None},
        }
    )
    assert pending == "change the C to a B"  # the raw text, for mark_applied
    assert combined == (
        "Standing preference: capo-free voicings please\n\n"
        "Requested correction: change the C to a B"
    )

    # a spent correction combines with nothing — the preference replays bare
    combined, pending = replay_guidance(
        {
            "preference": {"notes": "capo-free voicings please"},
            "correction": {"notes": "change the C to a B", "applied_to_version": "v9"},
        }
    )
    assert combined == "capo-free voicings please" and pending is None


def test_combine_guidance_orders_deterministically_and_labels_each_half():
    # a single value is returned bare — nothing to disambiguate yet, and every
    # existing single-note behavior is unchanged
    assert combine_guidance("only a preference", None) == "only a preference"
    assert combine_guidance(None, "only a correction") == "only a correction"
    assert combine_guidance(None, None) is None

    # both present: fixed order (preference, then correction), each labeled so
    # a later reader can never mistake one for the other
    assert combine_guidance("pref text", "corr text") == (
        "Standing preference: pref text\n\nRequested correction: corr text"
    )


def test_combine_guidance_never_cuts_either_half():
    """MAX_NOTES_CHARS is a PER-WRITE ceiling on each slot, enforced where each
    slot is written, so combining two already-bounded strings needs no cut and
    makes none.

    This replaces a test that asserted the opposite (the combination was cut to
    MAX_NOTES_CHARS, spending the cut on the correction). Cutting HERE was a
    read-time trim of text that stays whole in the store: the run showed the
    model a slice, and `mark_applied` then stamped `applied_to_version` against
    the WHOLE stored correction, so the unseen remainder was never replayed by
    anything — characters the caller typed, discarded permanently. Both halves
    are refused at their write surfaces now (see the boundary tests below), so
    nothing arrives that would need cutting.
    """
    preference = "P" * 5000
    correction = "C" * 5000
    combined = combine_guidance(preference, correction)

    # both halves survive WHOLE — this is the assertion that inverted
    assert combined.count("P") == 5000 and combined.count("C") == 5000
    assert combined == (
        f"Standing preference: {preference}\n\nRequested correction: {correction}"
    )
    assert "truncated" not in combined

    # under the ceiling nothing is touched, exactly as before
    plain = combine_guidance("short pref", "short corr")
    assert plain == "Standing preference: short pref\n\nRequested correction: short corr"

    # each half at its own maximum: the combination is TWO slots plus the
    # labels, which is what MAX_NOTES_CHARS now documents it bounds. Stated as a
    # real number so a future change to either the cap or the labels has to come
    # back here rather than silently widening what reaches a prompt.
    labels = len("Standing preference: \n\nRequested correction: ")
    both_max = combine_guidance("P" * MAX_NOTES_CHARS, "C" * MAX_NOTES_CHARS)
    assert len(both_max) == 2 * MAX_NOTES_CHARS + labels
    assert both_max.count("P") == both_max.count("C") == MAX_NOTES_CHARS

    # a single half is still returned bare, and still whole
    assert combine_guidance(None, "C" * MAX_NOTES_CHARS) == "C" * MAX_NOTES_CHARS
    assert combine_guidance("P" * MAX_NOTES_CHARS, None) == "P" * MAX_NOTES_CHARS

    # replay goes through the same rule, and `pending_correction` is the RAW
    # text — the compare-and-set in mark_applied matches the stored text
    replayed, pending = replay_guidance(
        {
            "preference": {"notes": preference},
            "correction": {"notes": correction, "applied_to_version": None},
        }
    )
    assert replayed == combined
    assert pending == correction


def test_a_maximal_correction_replays_whole_and_is_consumed_once(notes_store):
    """A correction at the very top of its cap, beside a preference at the top
    of its own: the whole of it replays (nothing is trimmed on the way to the
    prompt) and it is still stamped by its own stored text, so it is consumed
    exactly once instead of replaying forever.

    This replaces `test_a_capped_correction_is_still_consumable_after_
    truncation`, whose premise (the replayed string is a truncation of the
    stored one) no longer exists. The single-shot property it guarded is kept
    verbatim; what changed is that the text the run sees and the text
    `mark_applied` matches are now the SAME string, which is the point.
    """
    notes_store.set_preference(SONG_ID, "P" * MAX_NOTES_CHARS)
    long_correction = "C" * MAX_NOTES_CHARS
    notes_store.set_correction(SONG_ID, long_correction)

    combined, pending = replay_guidance(notes_store.get_record(SONG_ID))
    assert pending == long_correction
    assert combined.count("C") == MAX_NOTES_CHARS, "the whole instruction replays"
    assert combined.count("P") == MAX_NOTES_CHARS
    assert notes_store.mark_applied(SONG_ID, pending, "v1") is True
    assert notes_store.get_record(SONG_ID)["correction"]["applied_to_version"] == "v1"


def test_notes_store_delete_reports_whether_anything_was_there(notes_store):
    assert notes_store.delete(SONG_ID) is False
    notes_store.set_preference(SONG_ID, "kill me")
    assert notes_store.delete(SONG_ID) is True
    assert notes_store.get(SONG_ID) == ""


def test_notes_store_delete_clears_both_slots(notes_store):
    notes_store.set_preference(SONG_ID, "capo-free voicings please")
    notes_store.set_correction(SONG_ID, "change the C to a B")
    assert notes_store.delete(SONG_ID) is True
    assert notes_store.get_record(SONG_ID) is None


def test_notes_store_singleton_resets(monkeypatch):
    reset_song_notes_store()
    first = get_song_notes_store()
    assert first is get_song_notes_store()
    assert isinstance(build_song_notes_store(), InMemorySongNotesStore)  # default backend
    reset_song_notes_store()
    assert get_song_notes_store() is not first


# --- REST surface ------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    songs = InMemorySongRepository()
    runs = InMemoryRunRepository()
    notes = InMemorySongNotesStore()
    monkeypatch.setattr(api_mod, "get_store", lambda: songs)
    monkeypatch.setattr("snoocle_server.pipeline.get_store", lambda: songs)
    monkeypatch.setattr("snoocle_server.pipeline.get_run_store", lambda: runs)
    monkeypatch.setattr("snoocle_server.store.runs.get_run_store", lambda: runs)
    monkeypatch.setattr("snoocle_server.store.song_notes.get_song_notes_store", lambda: notes)
    reset_run_store()
    reset_song_notes_store()
    return TestClient(app)


def _empty_notes_doc(song_id: str) -> dict:
    """The whole "no notes" response — both lifetime objects null AND the
    deprecated flat compatibility keys at their pre-two-lifetime empty values
    (`notes: ""`, `updatedAt: null`), which is exactly what the shape looked
    like before this endpoint grew the two objects. Spelled out in one place
    because every test that asserts the empty document has to assert the WHOLE
    document: a key silently disappearing from it is the defect this pins."""
    return {
        "songId": song_id,
        "notes": "",
        "updatedAt": None,
        "preference": None,
        "correction": None,
    }


def test_get_notes_of_unknown_song_is_an_empty_document_not_404(client):
    r = client.get("/v1/songs/never--seen/notes")
    assert r.status_code == 200
    assert r.json() == _empty_notes_doc("never--seen")


def test_put_get_delete_notes(client):
    put = client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "capo-free voicings please"})
    assert put.status_code == 200
    body = put.json()
    assert body["songId"] == SONG_ID
    # written through the notes surface -> a standing preference, never consumed
    assert body["preference"]["notes"] == "capo-free voicings please"
    assert body["preference"]["updatedAt"]
    assert body["correction"] is None

    got = client.get(f"/v1/songs/{SONG_ID}/notes").json()
    assert got["preference"]["notes"] == "capo-free voicings please"

    assert client.delete(f"/v1/songs/{SONG_ID}/notes").json() == {"deleted": True}
    assert client.delete(f"/v1/songs/{SONG_ID}/notes").json() == {"deleted": False}
    assert client.get(f"/v1/songs/{SONG_ID}/notes").json() == _empty_notes_doc(SONG_ID)


def test_put_empty_notes_deletes_the_preference(client):
    client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "something"})
    cleared = client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "   "})
    assert cleared.status_code == 200
    assert cleared.json() == _empty_notes_doc(SONG_ID)


def test_put_does_not_disturb_a_pending_correction(client):
    """PUT writes only the durable preference — a pending correction left
    behind by an earlier analyze must survive it, same as the store contract
    test above, exercised through the REST surface."""
    from snoocle_server.store.song_notes import get_song_notes_store

    get_song_notes_store().set_correction(SONG_ID, "change the C to a B")
    put = client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "capo-free voicings please"})
    assert put.status_code == 200
    body = put.json()
    assert body["preference"]["notes"] == "capo-free voicings please"
    assert body["correction"]["notes"] == "change the C to a B"
    assert body["correction"]["appliedToVersion"] is None


def test_delete_clears_both_preference_and_correction(client):
    from snoocle_server.store.song_notes import get_song_notes_store

    client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "capo-free voicings please"})
    get_song_notes_store().set_correction(SONG_ID, "change the C to a B")

    assert client.delete(f"/v1/songs/{SONG_ID}/notes").json() == {"deleted": True}
    assert get_song_notes_store().get_record(SONG_ID) is None
    assert client.get(f"/v1/songs/{SONG_ID}/notes").json() == _empty_notes_doc(SONG_ID)


def test_the_flat_notes_keys_stay_as_the_preference_compatibility_view(client):
    """The shipped iOS client reads `notes`/`updatedAt` on every song it opens
    and after every save. Those keys predate the two-lifetime shape, which was
    introduced as ADDITIVE — a strict `Decodable` with `let notes: String`
    throws `keyNotFound` on every read and every save the moment they go away.

    They mirror `preference`, the slot this surface writes: GET-after-PUT must
    still show the caller what it just saved.
    """
    put = client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "capo-free voicings please"})
    assert put.status_code == 200
    body = put.json()
    assert body["notes"] == "capo-free voicings please"
    assert body["updatedAt"] == body["preference"]["updatedAt"]
    assert body["updatedAt"]

    got = client.get(f"/v1/songs/{SONG_ID}/notes").json()
    assert got["notes"] == got["preference"]["notes"] == "capo-free voicings please"
    assert got["updatedAt"] == got["preference"]["updatedAt"]

    # a pending CORRECTION is not surfaced through the flat view: an old client
    # never wrote one and would read it as its own durable note having changed
    # (local import: the `client` fixture patches the store getter on its own
    # module, so the name imported at the top of this file is the unpatched one)
    from snoocle_server.store.song_notes import get_song_notes_store

    get_song_notes_store().set_correction(SONG_ID, "change the C to a B")
    got = client.get(f"/v1/songs/{SONG_ID}/notes").json()
    assert got["notes"] == "capo-free voicings please"
    assert got["correction"]["notes"] == "change the C to a B"

    # cleared preference -> the pre-two-lifetime empty values, not a missing key
    client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": ""})
    got = client.get(f"/v1/songs/{SONG_ID}/notes").json()
    assert got["notes"] == "" and got["updatedAt"] is None


def test_mcp_notes_tools_carry_the_same_flat_compatibility_view(client):
    from snoocle_server import mcp_server as mcp_mod

    doc = mcp_mod.set_song_notes(SONG_ID, "capo-free voicings please")
    assert doc["notes"] == "capo-free voicings please"
    assert doc["updatedAt"] == doc["preference"]["updatedAt"]
    read = mcp_mod.get_song_notes(SONG_ID)
    assert read["notes"] == "capo-free voicings please"
    assert read["updatedAt"] == read["preference"]["updatedAt"]
    # one store, two surfaces, one shape
    assert set(read) == set(client.get(f"/v1/songs/{SONG_ID}/notes").json())


def test_put_notes_rejects_over_the_length_cap(client):
    r = client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "x" * 8001})
    assert r.status_code == 400
    assert "8000" in r.json()["detail"]
    # and nothing was stored
    assert client.get(f"/v1/songs/{SONG_ID}/notes").json()["preference"] is None
    # the cap itself is inclusive
    assert client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "y" * 8000}).status_code == 200


# --- the ceiling binds on the CORRECTION slot too, at every door -------------
# MAX_NOTES_CHARS is a per-WRITE ceiling on each slot independently. A
# request's `guidance` is a correction write, so it is refused the same way and
# with the same message as a preference write — at the door, before any
# expensive step, rather than being stored whole, shown to the model in part,
# and then marked applied in full.


def test_analyze_rejects_over_cap_guidance_at_the_rest_boundary(client, monkeypatch):
    def never(*a, **k):  # noqa: ANN001
        raise AssertionError("an over-cap guidance must be refused before any work")

    # restored by hand rather than monkeypatch.undo(): `client` shares this same
    # function-scoped monkeypatch, so undo() would also drop its store patches
    real_reconcile, real_discover = pipeline_mod.reconcile, pipeline_mod.discover_sources
    monkeypatch.setattr(pipeline_mod, "reconcile", never)
    monkeypatch.setattr(pipeline_mod, "discover_sources", never)

    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Fixme", "artist": "Tester", "provider": "mock",
              "skipAudio": True, "guidance": "x" * (MAX_NOTES_CHARS + 1)},
    )
    # the SAME shape the preference path uses: 400, same message
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert str(MAX_NOTES_CHARS) in detail and str(MAX_NOTES_CHARS + 1) in detail
    assert detail == client.put(
        f"/v1/songs/{SONG_ID}/notes", json={"notes": "x" * (MAX_NOTES_CHARS + 1)}
    ).json()["detail"]

    # nothing was persisted — not the correction, not a song version
    assert client.get(f"/v1/songs/{SONG_ID}/notes").json()["correction"] is None
    assert client.get(f"/v1/songs/{SONG_ID}").status_code == 404

    # and the cap is inclusive: exactly at it still runs
    monkeypatch.setattr(pipeline_mod, "reconcile", real_reconcile)
    monkeypatch.setattr(pipeline_mod, "discover_sources", real_discover)
    ok = client.post(
        "/v1/songs/analyze",
        json={"title": "Fixme", "artist": "Tester", "provider": "mock",
              "skipAudio": True, "guidance": "y" * MAX_NOTES_CHARS,
              "allowTestOutput": True},
    )
    assert ok.status_code == 200, ok.text
    assert client.get(f"/v1/songs/{SONG_ID}/notes").json()["correction"]["notes"] == (
        "y" * MAX_NOTES_CHARS
    )


def test_mcp_analyze_rejects_over_cap_guidance(client, monkeypatch):
    import asyncio

    from snoocle_server import mcp_server as mcp_mod

    # local import: the `client` fixture patches the store getter on its own
    # module, so the name bound at the top of this file is the unpatched one
    from snoocle_server.store.song_notes import get_song_notes_store as notes_store_getter

    def never(*a, **k):  # noqa: ANN001
        raise AssertionError("an over-cap guidance must be refused before any work")

    monkeypatch.setattr(pipeline_mod, "reconcile", never)

    with pytest.raises(ValueError) as excinfo:
        asyncio.run(
            mcp_mod.analyze_and_store_song(
                title="Fixme", artist="Tester", provider="mock", skip_audio=True,
                guidance="x" * (MAX_NOTES_CHARS + 1),
            )
        )
    # same message as set_song_notes, the preference slot's own refusal
    with pytest.raises(ValueError) as pref_exc:
        mcp_mod.set_song_notes(SONG_ID, "x" * (MAX_NOTES_CHARS + 1))
    assert str(excinfo.value) == str(pref_exc.value)
    assert notes_store_getter().get_record(SONG_ID) is None


def test_the_pipeline_refuses_over_cap_guidance_before_anything_expensive(monkeypatch):
    """The backstop behind the two surfaces: a direct `run_pipeline_async`
    caller cannot smuggle an over-cap correction past them.

    Checked at the pipeline's door specifically, NOT as a `ValueError` out of
    `store.set_correction`: that write happens inside `_resolve_guidance`'s
    best-effort try/except (which exists so a notes-store outage never fails an
    analysis), so a raise from the store would be swallowed there and the run
    would carry on with the over-cap text.
    """
    import asyncio

    def never(*a, **k):  # noqa: ANN001
        raise AssertionError("work was done before the guidance was validated")

    # every step that costs anything, including identity resolution
    monkeypatch.setattr(pipeline_mod, "_step_resolve", never)
    monkeypatch.setattr(pipeline_mod, "discover_sources", never)
    monkeypatch.setattr(pipeline_mod, "acquire", never)
    monkeypatch.setattr(pipeline_mod, "analyze_audio", never)
    monkeypatch.setattr(pipeline_mod, "reconcile", never)
    monkeypatch.setattr(pipeline_mod, "classify_correction", never)
    notes = InMemorySongNotesStore()
    monkeypatch.setattr("snoocle_server.store.song_notes.get_song_notes_store", lambda: notes)

    with pytest.raises(ValueError, match=str(MAX_NOTES_CHARS)):
        asyncio.run(
            pipeline_mod.run_pipeline_async(
                "Fixme", "Tester", provider="mock", skip_audio=True,
                guidance="x" * (MAX_NOTES_CHARS + 1),
            )
        )
    # refused before the correction was persisted, so a rejected request leaves
    # nothing behind for a later run to replay
    assert notes.get_record(SONG_ID) is None


# --- replay on analyze -------------------------------------------------------


@pytest.fixture()
def reconcile_calls(monkeypatch):
    """Capture the kwargs the reconciler is actually called with, while still
    running the real engine (so provenance is the real thing)."""
    calls: list[dict] = []
    real = pipeline_mod.reconcile

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "reconcile", spy)
    return calls


def _analyze(client, **extra) -> dict:
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Fixme", "artist": "Tester", "provider": "mock",
              "skipAudio": True, "allowTestOutput": True, **extra},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _reconciled_notes(body: dict) -> str:
    # LAST match, not first: provenance accumulates across every analyze of
    # the same song id, so a test that analyzes more than once (to exercise a
    # preference surviving alongside a spent correction, say) must read the
    # entry THIS run appended, not an earlier run's.
    entries = [p for p in body["song"]["provenance"] if p["action"] == "reconciled"]
    assert entries, "no reconciled provenance entry"
    return entries[-1]["notes"]


def test_stored_notes_replay_as_guidance_when_the_request_omits_it(client, reconcile_calls):
    client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "the bridge is Bm, not D"})
    body = _analyze(client)
    assert body["songId"] == SONG_ID  # the id the notes were stored under
    assert reconcile_calls[-1]["guidance"] == "the bridge is Bm, not D"
    assert "stored notes" in _reconciled_notes(body)


def test_a_durable_preference_keeps_replaying_across_runs(client, reconcile_calls):
    """A note curated through the notes surface is a STANDING instruction: it
    replays on every later analyze that sends none, and no run consumes it."""
    client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "capo-free voicings please"})
    for attempt in range(3):
        body = _analyze(
            client,
            **(
                {"force": True, "forceReason": "standing-preference replay test"}
                if attempt
                else {}
            ),
        )
        assert reconcile_calls[-1]["guidance"] == "capo-free voicings please"
        assert "stored notes" in _reconciled_notes(body)

    doc = client.get(f"/v1/songs/{SONG_ID}/notes").json()
    assert doc["preference"]["notes"] == "capo-free voicings please"
    assert doc["correction"] is None


def test_no_notes_means_no_guidance(client, reconcile_calls):
    body = _analyze(client)
    assert reconcile_calls[-1]["guidance"] is None
    assert "guidance" not in _reconciled_notes(body)


def test_request_guidance_is_persisted_as_a_single_shot_correction(client, reconcile_calls):
    """Request guidance is persisted the moment it arrives — as a SINGLE-SHOT
    correction, spent once it has landed in a stored version."""
    body = _analyze(client, guidance="NEW instruction")
    assert reconcile_calls[-1]["guidance"] == "NEW instruction"
    assert "this request" in _reconciled_notes(body)

    # persisted, and visibly spent: stamped with the version it landed in
    doc = client.get(f"/v1/songs/{SONG_ID}/notes").json()
    assert doc["preference"] is None
    assert doc["correction"]["notes"] == "NEW instruction"
    assert doc["correction"]["appliedToVersion"] == body["storedVersion"]

    # so the next run, which expressed no instruction of its own, gets none
    next_body = _analyze(client)
    assert reconcile_calls[-1]["guidance"] is None
    assert "notes" not in next_body["steps"]


def test_a_fresh_correction_combines_with_a_surviving_preference(client, reconcile_calls):
    """THE regression this store shape exists to fix: a standing preference
    ("capo-free voicings please") must survive an analyze that carries its
    own guidance ("change the C in line 12 to a B") — and this run's actual
    guidance is the two combined, not one clobbering the other."""
    client.put(f"/v1/songs/{SONG_ID}/notes", json={"notes": "capo-free voicings please"})
    body = _analyze(client, guidance="change the C in line 12 to a B")
    assert reconcile_calls[-1]["guidance"] == (
        "Standing preference: capo-free voicings please\n\n"
        "Requested correction: change the C in line 12 to a B"
    )
    assert "this request" in _reconciled_notes(body)
    assert "preference + correction combined" in body["steps"]["notes"]

    # both persisted independently: the preference untouched, the correction
    # stored and stamped as applied to the version this run stored
    doc = client.get(f"/v1/songs/{SONG_ID}/notes").json()
    assert doc["preference"]["notes"] == "capo-free voicings please"
    assert doc["correction"]["notes"] == "change the C in line 12 to a B"
    assert doc["correction"]["appliedToVersion"] == body["storedVersion"]

    # the next run, with no guidance of its own, replays ONLY the surviving
    # preference — the correction already landed and does not replay again,
    # so the song is never left with NO effective note (the reported defect)
    next_body = _analyze(client)
    assert reconcile_calls[-1]["guidance"] == "capo-free voicings please"
    assert "stored notes" in _reconciled_notes(next_body)
    assert "combined" not in next_body["steps"]["notes"]  # only one lifetime in play now


def test_a_pending_correction_replays_until_it_lands_then_never_again(
    client, monkeypatch, reconcile_calls
):
    """The reason request guidance is persisted BEFORE any expensive step: a run
    that dies must not eat the user's typed instruction, so it replays into the
    retry. Once it has landed in a stored version it is in the document the next
    run starts from, and replaying it there is the reported defect."""
    spy = pipeline_mod.reconcile

    def boom(*args, **kwargs):
        raise ReconcileError("provider exploded")

    monkeypatch.setattr(pipeline_mod, "reconcile", boom)
    dead = client.post(
        "/v1/songs/analyze",
        json={"title": "Fixme", "artist": "Tester", "provider": "mock",
              "skipAudio": True, "guidance": "change the C to a B in line 12"},
    )
    assert dead.status_code == 502
    monkeypatch.setattr(pipeline_mod, "reconcile", spy)

    # the retry sends nothing, and the unspent correction stands in for it
    retry = _analyze(client)
    assert reconcile_calls[-1]["guidance"] == "change the C to a B in line 12"
    assert "stored notes" in _reconciled_notes(retry)

    doc = client.get(f"/v1/songs/{SONG_ID}/notes").json()
    assert doc["correction"]["appliedToVersion"] == retry["storedVersion"]

    # exactly once: the run after that is not a correction run at all
    _analyze(client)
    assert reconcile_calls[-1]["guidance"] is None


def test_notes_survive_a_failing_analysis(client, monkeypatch):
    def boom(*args, **kwargs):
        raise ReconcileError("provider exploded")

    monkeypatch.setattr(pipeline_mod, "reconcile", boom)
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Fixme", "artist": "Tester", "provider": "mock",
              "skipAudio": True, "guidance": "keep me"},
    )
    assert r.status_code == 502
    # the typed instruction is not lost with the run that failed
    assert client.get(f"/v1/songs/{SONG_ID}/notes").json()["correction"]["notes"] == "keep me"


def test_notes_store_outage_does_not_fail_an_analysis(client, monkeypatch, reconcile_calls):
    class _Broken(InMemorySongNotesStore):
        # The replay path reads the RECORD now (it needs each half's own
        # state), so that is the read to break here.
        def get_record(self, song_id: str) -> dict | None:
            raise RuntimeError("firestore down")

        def get(self, song_id: str) -> str:
            raise RuntimeError("firestore down")

    monkeypatch.setattr("snoocle_server.store.song_notes.get_song_notes_store", lambda: _Broken())
    body = _analyze(client)
    assert body["songId"] == SONG_ID
    assert reconcile_calls[-1]["guidance"] is None


def test_a_failing_consumption_stamp_does_not_fail_a_stored_analysis(
    client, monkeypatch, reconcile_calls
):
    """The single-shot stamp is bookkeeping AFTER the store: a backend that dies
    on the way to it must not turn a stored analysis into a 502."""

    class _Broken(InMemorySongNotesStore):
        def mark_applied(self, song_id: str, notes: str, version: str) -> bool:
            raise RuntimeError("firestore down")

    monkeypatch.setattr("snoocle_server.store.song_notes.get_song_notes_store", lambda: _Broken())
    body = _analyze(client, guidance="the bridge is Bm, not D")
    assert body["storedVersion"]
    assert reconcile_calls[-1]["guidance"] == "the bridge is Bm, not D"


# --- MCP surface -------------------------------------------------------------
# This server is driven from MCP in production, and notes an MCP caller can
# neither see nor set nor clear are notes it silently inherits from whichever
# HTTP client wrote last.


def test_mcp_notes_tools_round_trip(client):
    from snoocle_server import mcp_server as mcp_mod

    assert mcp_mod.get_song_notes(SONG_ID) == _empty_notes_doc(SONG_ID)

    doc = mcp_mod.set_song_notes(SONG_ID, "  capo-free voicings please  ")
    assert doc["preference"]["notes"] == "capo-free voicings please"  # stored trimmed
    assert doc["preference"]["updatedAt"]
    assert doc["correction"] is None  # this surface curates preferences only
    assert mcp_mod.get_song_notes(SONG_ID)["preference"]["notes"] == "capo-free voicings please"
    # one store, two surfaces
    rest = client.get(f"/v1/songs/{SONG_ID}/notes").json()
    assert rest["preference"]["notes"] == "capo-free voicings please"

    assert mcp_mod.clear_song_notes(SONG_ID) == {"songId": SONG_ID, "deleted": True}
    assert mcp_mod.clear_song_notes(SONG_ID)["deleted"] is False
    assert mcp_mod.get_song_notes(SONG_ID)["preference"] is None

    with pytest.raises(ValueError):
        mcp_mod.set_song_notes(SONG_ID, "x" * (8000 + 1))


def test_mcp_set_song_notes_does_not_disturb_a_pending_correction(client):
    from snoocle_server import mcp_server as mcp_mod
    from snoocle_server.store.song_notes import get_song_notes_store

    get_song_notes_store().set_correction(SONG_ID, "change the C to a B")
    doc = mcp_mod.set_song_notes(SONG_ID, "capo-free voicings please")
    assert doc["preference"]["notes"] == "capo-free voicings please"
    assert doc["correction"]["notes"] == "change the C to a B"


def test_mcp_clear_song_notes_clears_both(client):
    from snoocle_server import mcp_server as mcp_mod
    from snoocle_server.store.song_notes import get_song_notes_store

    mcp_mod.set_song_notes(SONG_ID, "capo-free voicings please")
    get_song_notes_store().set_correction(SONG_ID, "change the C to a B")

    assert mcp_mod.clear_song_notes(SONG_ID) == {"songId": SONG_ID, "deleted": True}
    assert mcp_mod.get_song_notes(SONG_ID) == _empty_notes_doc(SONG_ID)


def test_mcp_analyze_takes_guidance_and_exposes_what_it_left_behind(client, reconcile_calls):
    """The gap that made stored notes unmanageable from MCP: analyze could not
    SET guidance, and no tool could show or clear what a previous run left."""
    import asyncio

    from snoocle_server import mcp_server as mcp_mod

    out = asyncio.run(
        mcp_mod.analyze_and_store_song(
            title="Fixme", artist="Tester", provider="mock", skip_audio=True,
            guidance="change the C to a B in line 12",
            allow_test_output=True,
        )
    )
    assert out["songId"] == SONG_ID
    assert reconcile_calls[-1]["guidance"] == "change the C to a B in line 12"

    doc = mcp_mod.get_song_notes(SONG_ID)
    assert doc["preference"] is None
    assert doc["correction"]["notes"] == "change the C to a B in line 12"
    assert doc["correction"]["appliedToVersion"] == out["storedVersion"]  # already spent

    assert mcp_mod.clear_song_notes(SONG_ID)["deleted"] is True
    assert mcp_mod.get_song_notes(SONG_ID)["correction"] is None
