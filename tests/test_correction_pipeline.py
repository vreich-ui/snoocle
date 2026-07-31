"""Pipeline-level routing: a targeted correction gets notes-only scope
inferred, skips discovery, and (when it produced a patch) skips the
timing/LRC/confidence passes that exist to guard a REGENERATED document —
none of which apply when nothing was regenerated.

Uses a stand-in `reconcile()` (matching tests/test_timing_carry_forward.py's
`_StaticReconciler` pattern) so these tests are about pipeline.py's OWN
wiring — scope inference, the discovery skip, threading patch_ops_eligible
through, reading patch_ops_applied back — independent of the LLM call
mechanics, which tests/test_patch_reconcile.py covers directly against
reconcile/engine.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server import pipeline as pipeline_mod
from snoocle_server.api import app
from snoocle_server.config import settings
from snoocle_server.reconcile.engine import ReconcileResult
from snoocle_server.schema import Song
from snoocle_server.store.memory import InMemorySongRepository
from snoocle_server.store.song_notes import (
    MAX_NOTES_CHARS,
    get_song_notes_store,
    reset_song_notes_store,
)

client = TestClient(app)

SONG_ID = "the-rolling-stones--paint-it-black"


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch):
    store = InMemorySongRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: store)
    monkeypatch.setattr(pipeline_mod, "get_store", lambda: store)
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(pipeline_mod, "fetch_lrc", lambda *a, **k: None)
    # Every test here re-analyzes the SAME song id — without resetting this
    # (a separate, process-wide singleton) a guidance note set by one test
    # replays as a later test's "no guidance given" case, per
    # _resolve_guidance's stored-notes-replay contract. That isolation is also
    # what hid the cross-REQUEST version of the same leak from these tests for
    # so long; test_a_stale_applied_correction_* below sets its own note and
    # asserts the leak is closed rather than reset away.
    reset_song_notes_store()
    yield store
    reset_song_notes_store()


def _gold_song() -> Song:
    return Song.model_validate(
        {
            "id": SONG_ID,
            "metadata": {"title": "Paint It Black", "artist": "The Rolling Stones", "bpm": 128.0},
            "audio": {
                "youtubeVideoId": "flSmiIne4k1",
                "beats": [{"time": 0.0, "measure": 1, "beatInMeasure": 1}],
                "syncMap": [{"lineIndex": 0, "time": 0.0}],
            },
            "sections": [
                {"sectionIndex": 0, "name": "Verse 1", "kind": "verse",
                 "startLineIndex": 0, "endLineIndex": 0, "startTime": 0.0, "endTime": 4.0},
            ],
            "lines": [
                {"lineIndex": 0, "lyrics": "I see a red door", "timeSeconds": 0.0, "confidence": 0.9,
                 "chordPlacements": [{"charIndex": 0, "chord": "C", "timeSeconds": 0.0, "confidence": 0.9}]},
            ],
            "provenance": [
                {"timestamp": "2026-07-01T00:00:00Z", "actor": "reconcile:test/gold", "action": "reconciled"},
            ],
        }
    )


class _Recorder:
    """Stands in for `reconcile`; returns `song` unchanged and records every
    keyword it was called with, including the new `patch_ops_eligible`."""

    def __init__(self, song: Song, patch_ops_applied: int = 0):
        self.song = song
        self.patch_ops_applied = patch_ops_applied
        self.calls: list[dict] = []

    def __call__(self, title, artist, candidates, mir, **kwargs):
        self.calls.append({"candidates": candidates, "mir": mir, **kwargs})
        return ReconcileResult(
            song=self.song.model_copy(deep=True), provider="test-patch", model="fake",
            attempts=1, audio_attached=False, usage={},
            patch_ops_applied=self.patch_ops_applied,
        )

    @property
    def last(self) -> dict:
        assert self.calls, "reconcile was never called"
        return self.calls[-1]


def _analyze(**extra):
    body = {"title": "Paint It Black", "artist": "The Rolling Stones", "provider": "anthropic"}
    body.update(extra)
    return client.post("/v1/songs/analyze", json=body)


def _no_discovery(monkeypatch) -> None:
    def boom(*a, **k):  # noqa: ANN001
        raise AssertionError("discovery ran despite an inferred notes-only scope")

    monkeypatch.setattr(pipeline_mod, "discover_sources", boom)


def _no_audio(monkeypatch) -> None:
    def boom(*a, **k):  # noqa: ANN001
        raise AssertionError("audio was acquired despite an inferred notes-only scope")

    monkeypatch.setattr(pipeline_mod, "acquire", boom)
    monkeypatch.setattr(pipeline_mod, "analyze_audio", boom)


def _classified(monkeypatch) -> list[str]:
    """Every text the router is asked to classify, in order — the real
    classifier still runs. WHAT gets classified is the whole routing decision:
    a preference's words inside a combined string classify as the caller's own
    intent, which is precisely the leak."""
    seen: list[str] = []
    real = pipeline_mod.classify_correction

    def spy(text, *args, **kwargs):  # noqa: ANN001
        seen.append(text)
        return real(text, *args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "classify_correction", spy)
    return seen


def _real_reconcile_spy(monkeypatch) -> list[dict]:
    """Capture the reconciler's kwargs while still running the REAL engine, so
    provenance is the real thing (`_Recorder` above never builds any)."""
    calls: list[dict] = []
    real = pipeline_mod.reconcile

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "reconcile", spy)
    return calls


def _llm_tier_spy(monkeypatch) -> list[str]:
    """Every text that reached the router's LLM fallback tier, which returns
    "inconclusive" here — no network call, and no accidental pass because a
    real model was generous about a vague note."""
    from snoocle_server import correction_routing

    seen: list[str] = []
    monkeypatch.setattr(
        correction_routing, "_llm_classify", lambda text: seen.append(text) or None
    )
    return seen


# --- the primary reported scenario: scope inference + no rediscovery -------------


def test_a_targeted_correction_infers_notes_only_and_skips_discovery(monkeypatch, isolated_store):
    isolated_store.save(_gold_song(), message="gold")
    _no_discovery(monkeypatch)
    _no_audio(monkeypatch)
    recorder = _Recorder(_gold_song(), patch_ops_applied=1)
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(guidance="change the C to a B in line 12")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["steps"]["discover"].startswith("skipped (scope: notes only)")
    assert body["steps"]["acquire"].startswith("skipped")
    assert "inferred" in body["steps"]["scope"]
    assert "chord-symbol" in body["steps"]["scope"]
    assert "listen=off, reconcile=off" in body["steps"]["scope"]

    call = recorder.last
    assert call["scope"].notes_only
    assert call["patch_ops_eligible"] is True
    # the stored version stood in as the prior since none was attached
    assert call["prior_song"]["id"] == SONG_ID


def test_a_lyric_targeting_correction_infers_notes_only_but_is_not_patch_eligible(
    monkeypatch, isolated_store
):
    isolated_store.save(_gold_song(), message="gold")
    _no_discovery(monkeypatch)
    _no_audio(monkeypatch)
    recorder = _Recorder(_gold_song())
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(guidance="line 0's lyrics should say 'window' not 'door'")
    assert r.status_code == 200, r.text
    assert "inferred" in r.json()["steps"]["scope"]

    call = recorder.last
    assert call["scope"].notes_only
    assert call["patch_ops_eligible"] is False


# --- explicit caller scope always wins ---------------------------------------


def test_an_explicit_scope_is_never_overridden_by_inference(monkeypatch, isolated_store):
    isolated_store.save(_gold_song(), message="gold")
    ran: list[str] = []
    monkeypatch.setattr(
        pipeline_mod, "discover_sources", lambda *a, **k: ran.append("discover") or []
    )
    recorder = _Recorder(_gold_song())
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(
        guidance="change the C to a B in line 12",  # would otherwise infer notes-only
        scope={"listen": True, "reconcile": True},
        skipAudio=True,
    )
    assert r.status_code == 200, r.text
    assert ran == ["discover"], "the caller's explicit scope must still run discovery"
    assert "inferred" not in r.json()["steps"]["scope"]
    assert r.json()["steps"]["scope"] == "listen=on, reconcile=on"
    assert recorder.last["patch_ops_eligible"] is False, (
        "patch eligibility is meaningless outside notes-only scope"
    )


# --- no prior document, nothing to infer against ------------------------------


def test_scope_inference_never_fires_with_no_prior_to_correct(monkeypatch):
    """"Notes naming a chord" has nothing to apply itself to on a song's
    first-ever analysis — inference must not fire, and the run proceeds as
    an ordinary full analysis."""
    ran: list[str] = []
    monkeypatch.setattr(
        pipeline_mod, "discover_sources", lambda *a, **k: ran.append("discover") or []
    )
    recorder = _Recorder(_gold_song())
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(guidance="change the C to a B in line 12", skipAudio=True)
    assert r.status_code == 200, r.text
    assert ran == ["discover"]
    assert "scope" not in r.json()["steps"]
    assert recorder.last["scope"] is None
    assert recorder.last["patch_ops_eligible"] is False


def test_no_guidance_at_all_is_unaffected(monkeypatch, isolated_store):
    isolated_store.save(_gold_song(), message="gold")
    ran: list[str] = []
    monkeypatch.setattr(
        pipeline_mod, "discover_sources", lambda *a, **k: ran.append("discover") or []
    )
    recorder = _Recorder(_gold_song())
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(skipAudio=True)
    assert r.status_code == 200, r.text
    assert ran == ["discover"]
    assert "scope" not in r.json()["steps"]


# --- a note left behind by a PREVIOUS request never routes a later run ----------


def test_a_stale_applied_correction_neither_replays_nor_infers_notes_only(
    monkeypatch, isolated_store
):
    """The compound production failure, both halves in one run: request 1
    corrects one chord and stores; request 2 sends no guidance and no scope.

    Nothing about request 2 is a targeted correction, so it must be an ordinary
    full analysis: the previous request's note must neither be applied to it nor
    narrow it to notes-only. Storage is the only thing connecting the two
    requests, which is exactly what made "a client that sends no guidance is
    unaffected" untrue.
    """
    isolated_store.save(_gold_song(), message="gold")
    ran: list[str] = []
    monkeypatch.setattr(
        pipeline_mod, "discover_sources", lambda *a, **k: ran.append("discover") or []
    )
    recorder = _Recorder(_gold_song(), patch_ops_applied=1)
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    first = _analyze(guidance="change the C to a B in line 12")
    assert first.status_code == 200, first.text
    assert "inferred" in first.json()["steps"]["scope"]  # routed on its OWN guidance
    assert ran == [], "a notes-only run gathers nothing"

    second = _analyze(skipAudio=True)
    assert second.status_code == 200, second.text
    steps = second.json()["steps"]
    assert recorder.last["guidance"] is None, "a spent correction must not be re-applied"
    assert "notes" not in steps
    assert "scope" not in steps, "a stored note must not narrow a run to notes-only"
    assert ran == ["discover"], "a run that expressed nothing still gathers sources"


def test_a_replayed_note_never_infers_scope_even_before_it_has_been_applied(
    monkeypatch, isolated_store
):
    """Bounding the replay is one fix; the origin check is the other, and it does
    not depend on it. A correction still awaiting a run (its own run died before
    storing) DOES replay as guidance — but it still may not route a request that
    said nothing about scope, because scope is the caller's decision about THIS
    run."""
    isolated_store.save(_gold_song(), message="gold")
    get_song_notes_store().set_correction(SONG_ID, "change the C to a B in line 12")
    ran: list[str] = []
    monkeypatch.setattr(
        pipeline_mod, "discover_sources", lambda *a, **k: ran.append("discover") or []
    )
    recorder = _Recorder(_gold_song())
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(skipAudio=True)
    assert r.status_code == 200, r.text
    assert recorder.last["guidance"] == "change the C to a B in line 12"
    assert "stored notes" in r.json()["steps"]["notes"]
    assert "scope" not in r.json()["steps"]
    assert recorder.last["scope"] is None
    assert ran == ["discover"]


def test_an_explicit_notes_only_scope_still_patches_a_replayed_correction(
    monkeypatch, isolated_store
):
    """The carve-out kept deliberately: a replayed note may not ROUTE a run, but a
    caller who explicitly asked for notes-only has already routed it, and patch
    eligibility (what keeps a one-chord fix off the regeneration path) still comes
    from classifying the note."""
    isolated_store.save(_gold_song(), message="gold")
    get_song_notes_store().set_correction(SONG_ID, "change the C to a B in line 12")
    _no_discovery(monkeypatch)
    _no_audio(monkeypatch)
    recorder = _Recorder(_gold_song(), patch_ops_applied=1)
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(scope={"listen": False, "reconcile": False})
    assert r.status_code == 200, r.text
    assert "inferred" not in r.json()["steps"]["scope"]
    assert recorder.last["guidance"] == "change the C to a B in line 12"
    assert recorder.last["patch_ops_eligible"] is True


# --- a durable preference must not regress scope inference -------------------
# The one-slot design's origin check ("guidance replayed from the store never
# routes a run") already protects a preference the same way it protects a
# stale correction; these tests pin that down explicitly for the two-lifetime
# store, including the case classify_correction has to get right: a
# preference's words sitting right next to a real correction's in one
# combined string.


def test_a_standing_preference_alone_never_infers_notes_only_scope(monkeypatch, isolated_store):
    """A preference is a standing instruction, not an expressed intent about
    THIS run — replaying it must not narrow scope, and unlike a correction it
    never expires, so this has to hold on every run forever, not just once."""
    isolated_store.save(_gold_song(), message="gold")
    get_song_notes_store().set_preference(SONG_ID, "capo-free voicings please")
    ran: list[str] = []
    monkeypatch.setattr(
        pipeline_mod, "discover_sources", lambda *a, **k: ran.append("discover") or []
    )
    recorder = _Recorder(_gold_song())
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    for attempt in range(2):
        r = _analyze(
            skipAudio=True,
            **(
                {"force": True, "forceReason": "preference replay test requires a second run"}
                if attempt
                else {}
            ),
        )
        assert r.status_code == 200, r.text
        assert recorder.last["guidance"] == "capo-free voicings please"
        assert "scope" not in r.json()["steps"]
        assert recorder.last["scope"] is None
    assert ran == ["discover", "discover"]


def test_a_targeted_preference_never_routes_a_vaguely_guided_request(
    monkeypatch, isolated_store
):
    """The reported blocker. A standing preference whose words ARE targeted
    ("the bridge is Bm, not D" — this module's own example) plus a request whose
    own guidance is open-ended ("double-check this against better sources").

    The caller asked for re-verification against sources. Classifying the
    COMBINED string handed the preference's chord and section words to the
    router as if the caller had typed them on this request, and the run came
    back as a notes-only re-application of the document it was asked to
    re-verify: discovery skipped, nothing listened to. A preference never
    expires, so that fired on every guided analyze of the song from then on.
    """
    isolated_store.save(_gold_song(), message="gold")
    get_song_notes_store().set_preference(SONG_ID, "the bridge is Bm, not D")
    classified = _classified(monkeypatch)
    llm_seen = _llm_tier_spy(monkeypatch)
    ran: list[str] = []
    monkeypatch.setattr(
        pipeline_mod, "discover_sources", lambda *a, **k: ran.append("discover") or []
    )

    def acquire_attempted(*a, **k):  # noqa: ANN001
        ran.append("acquire")
        raise RuntimeError("no audio in this test; being ASKED for it is the point")

    monkeypatch.setattr(pipeline_mod, "acquire", acquire_attempted)
    recorder = _Recorder(_gold_song())
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    vague = "please double-check this against better sources"
    r = _analyze(guidance=vague)
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]

    # the router only ever saw the caller's OWN words — not the preference's,
    # and not the two combined
    assert classified == [vague]
    assert llm_seen == [vague], "no deterministic rule may fire on the caller's own text"

    assert "scope" not in steps, "a preference's words must not route this run"
    assert recorder.last["scope"] is None
    assert ran == ["discover", "acquire"], (
        "the caller asked to re-check against sources, and re-verification listens"
    )
    assert not steps["discover"].startswith("skipped")
    assert not steps["acquire"].startswith("skipped")

    # the run is still GUIDED by both — only the routing decision is the
    # caller's alone (a full run sees the combination; see the exposure tests)
    assert recorder.last["guidance"] == (
        "Standing preference: the bridge is Bm, not D\n\nRequested correction: " + vague
    )
    assert "preference + correction combined" in steps["notes"]


def test_a_preference_next_to_a_fresh_targeted_correction_still_infers_notes_only(
    monkeypatch, isolated_store
):
    """The inverse of the leak above, and it must keep working: a VAGUE standing
    preference next to a fresh, TARGETED correction still routes notes-only, on
    the correction's own words.

    The preference is deliberately one no deterministic rule matches, so the
    inference can only be coming from the correction — the old version of this
    test asserted the classifier saw the two COMBINED, which is the same string
    that let a targeted preference route a vague request (see the test above).
    """
    isolated_store.save(_gold_song(), message="gold")
    get_song_notes_store().set_preference(SONG_ID, "capo-free voicings please")
    _no_discovery(monkeypatch)
    _no_audio(monkeypatch)
    classified = _classified(monkeypatch)
    seen = _llm_tier_spy(monkeypatch)
    recorder = _Recorder(_gold_song(), patch_ops_applied=1)
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(guidance="change the C to a B in line 12")
    assert r.status_code == 200, r.text
    scope_step = r.json()["steps"]["scope"]
    assert "inferred" in scope_step
    assert "chord-symbol+line-reference" in scope_step  # the correction's rules
    assert classified == ["change the C to a B in line 12"]
    assert seen == [], "a deterministic rule fired; the LLM tier is not needed"
    assert recorder.last["scope"].notes_only
    assert recorder.last["patch_ops_eligible"] is True

    # and the preference is still there afterward — resolving THIS run's
    # guidance never wrote the preference back over
    assert get_song_notes_store().get_record(SONG_ID)["preference"]["notes"] == (
        "capo-free voicings please"
    )


# --- what the MODEL is shown, which is not always all that is in force -------
# A notes-only run is told "return this document with these notes applied and
# nothing else changed". A standing rendering preference is not part of a
# one-chord fix, and the patch path's closed op set cannot express one at all,
# so such a run is shown the CORRECTION alone. A full run — the document being
# rebuilt from evidence — is shown both. See pipeline.py's `model_guidance`.


def test_a_notes_only_run_is_shown_the_correction_without_the_preference(
    monkeypatch, isolated_store
):
    isolated_store.save(_gold_song(), message="gold")
    get_song_notes_store().set_preference(SONG_ID, "capo-free voicings please")
    _no_discovery(monkeypatch)
    _no_audio(monkeypatch)
    recorder = _Recorder(_gold_song(), patch_ops_applied=1)
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(guidance="change the C to a B in line 12")
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert recorder.last["scope"].notes_only
    assert recorder.last["guidance"] == "change the C to a B in line 12"
    assert "capo-free" not in (recorder.last["guidance"] or "")
    # and not through the back door either: the manifest travels IN the agent
    # payload, so it may not quote guidance the run withheld
    assert recorder.last["evidence_manifest"]["request"]["notes"] == (
        "change the C to a B in line 12"
    )
    # the report says so rather than claiming a combination it did not hand over
    assert "correction only" in steps["notes"]
    assert "combined" not in steps["notes"]


def test_a_full_run_is_shown_the_preference_and_the_correction_together(
    monkeypatch, isolated_store
):
    """Same two notes, a run that is NOT notes-only (the caller said so
    explicitly): the document is being rebuilt from evidence, which is exactly
    when a standing instruction about how to build it applies."""
    isolated_store.save(_gold_song(), message="gold")
    get_song_notes_store().set_preference(SONG_ID, "capo-free voicings please")
    monkeypatch.setattr(pipeline_mod, "discover_sources", lambda *a, **k: [])
    recorder = _Recorder(_gold_song())
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(
        guidance="change the C to a B in line 12",
        scope={"listen": True, "reconcile": True},
        skipAudio=True,
    )
    assert r.status_code == 200, r.text
    combined = (
        "Standing preference: capo-free voicings please\n\n"
        "Requested correction: change the C to a B in line 12"
    )
    assert recorder.last["guidance"] == combined
    assert recorder.last["evidence_manifest"]["request"]["notes"] == combined
    assert "preference + correction combined" in r.json()["steps"]["notes"]


def test_a_notes_only_run_with_only_a_preference_still_applies_it(
    monkeypatch, isolated_store
):
    """The carve-out that keeps the rule from becoming "notes-only ignores
    preferences": with no correction in force, the preference is the only
    instruction such a run has, and withholding it would leave the model asked
    to change nothing at all."""
    isolated_store.save(_gold_song(), message="gold")
    get_song_notes_store().set_preference(SONG_ID, "the bridge is Bm, not D")
    _no_discovery(monkeypatch)
    _no_audio(monkeypatch)
    recorder = _Recorder(_gold_song(), patch_ops_applied=1)
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(scope={"listen": False, "reconcile": False})
    assert r.status_code == 200, r.text
    assert recorder.last["guidance"] == "the bridge is Bm, not D"
    # classified for PATCH eligibility only — the caller had already routed
    # this run, so this is not the inference path
    assert "inferred" not in r.json()["steps"]["scope"]
    assert recorder.last["patch_ops_eligible"] is True


# --- what the model is shown is BOUNDED, on both paths -----------------------
# The store's own cap tests exercise `combine_guidance`/`replay_guidance`
# directly, which is why the notes-only path shipped with no bound at all: the
# pipeline hands the model `pending_correction or guidance` there, the caller's
# RAW text, and only `guidance` had ever been through the capping code. These
# assert on the string the reconciler is actually called with.


def test_the_guidance_the_reconciler_receives_is_bounded_on_the_notes_only_path(
    monkeypatch, isolated_store
):
    """The reported bypass: a targeted correction infers notes-only, and a
    notes-only run is shown the correction alone — the one string that used to
    reach the model without passing through any ceiling at all. A 40k-char
    guidance arrived at the prompt whole while the same note under an explicit
    full scope came back capped, so the bound depended on the scope."""
    isolated_store.save(_gold_song(), message="gold")
    _no_discovery(monkeypatch)
    _no_audio(monkeypatch)
    recorder = _Recorder(_gold_song(), patch_ops_applied=1)
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)
    get_song_notes_store().set_preference(SONG_ID, "P" * MAX_NOTES_CHARS)

    # the largest correction the door accepts
    correction = "change the C to a B in line 12. " + "x" * (MAX_NOTES_CHARS - 32)
    assert len(correction) == MAX_NOTES_CHARS
    r = _analyze(guidance=correction)
    assert r.status_code == 200, r.text
    assert "inferred" in r.json()["steps"]["scope"]

    shown = recorder.last["guidance"]
    # the RIGHT text: the correction, whole, and no preference smuggled in
    assert shown == correction
    assert "P" not in shown
    # bounded, and by the per-write ceiling alone — nothing was trimmed
    assert len(shown) == MAX_NOTES_CHARS
    assert "truncated" not in shown
    # and the manifest, which travels inside the agent payload, agrees
    assert recorder.last["evidence_manifest"]["request"]["notes"] == correction


def test_the_guidance_the_reconciler_receives_is_bounded_on_the_full_path(
    monkeypatch, isolated_store
):
    """The same two notes under a scope that is NOT notes-only: the model sees
    both, so the bound is two slots plus the labels — stated, and the ceiling a
    reader of MAX_NOTES_CHARS is promised."""
    isolated_store.save(_gold_song(), message="gold")
    monkeypatch.setattr(pipeline_mod, "discover_sources", lambda *a, **k: [])
    recorder = _Recorder(_gold_song())
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)
    preference = "P" * MAX_NOTES_CHARS
    get_song_notes_store().set_preference(SONG_ID, preference)

    correction = "C" * MAX_NOTES_CHARS
    r = _analyze(
        guidance=correction, scope={"listen": True, "reconcile": True}, skipAudio=True
    )
    assert r.status_code == 200, r.text

    shown = recorder.last["guidance"]
    labels = len("Standing preference: \n\nRequested correction: ")
    assert shown == f"Standing preference: {preference}\n\nRequested correction: {correction}"
    assert len(shown) == 2 * MAX_NOTES_CHARS + labels
    # neither half was cut on the way — the whole of each is what the caller
    # wrote and the whole of each is what the store still holds
    assert shown.count("P") == shown.count("C") == MAX_NOTES_CHARS
    assert "truncated" not in shown
    assert get_song_notes_store().get_record(SONG_ID)["correction"]["notes"] == correction


# --- steps["notes"] states what was IN FORCE, not what two strings looked like ---
# The detail used to be derived by comparing `model_guidance`/`guidance`/
# `pending_correction`, which cannot tell "a preference was combined in" from
# "there was no preference and one correction came back unchanged". With no
# preference set at all, both branches fired falsely.


def test_the_notes_step_claims_no_preference_when_none_is_set(monkeypatch, isolated_store):
    isolated_store.save(_gold_song(), message="gold")
    monkeypatch.setattr(pipeline_mod, "discover_sources", lambda *a, **k: [])
    recorder = _Recorder(_gold_song(), patch_ops_applied=1)
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)
    assert get_song_notes_store().get_record(SONG_ID) is None  # nothing standing

    # full scope
    full = _analyze(
        guidance="rebuild this from scratch",
        scope={"listen": True, "reconcile": True},
        skipAudio=True,
    )
    assert full.status_code == 200, full.text
    assert full.json()["steps"]["notes"] == "ok: guidance applied (from this request)"

    # notes-only (inferred), same song, still no preference
    _no_discovery(monkeypatch)
    _no_audio(monkeypatch)
    only = _analyze(guidance="change the C to a B in line 12")
    assert only.status_code == 200, only.text
    assert only.json()["steps"]["notes"] == "ok: guidance applied (from this request)"
    assert get_song_notes_store().get_record(SONG_ID)["preference"] is None


def test_the_notes_step_names_the_preference_only_when_one_is_in_force(
    monkeypatch, isolated_store
):
    isolated_store.save(_gold_song(), message="gold")
    get_song_notes_store().set_preference(SONG_ID, "capo-free voicings please")
    monkeypatch.setattr(pipeline_mod, "discover_sources", lambda *a, **k: [])
    recorder = _Recorder(_gold_song(), patch_ops_applied=1)
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    full = _analyze(
        guidance="rebuild this from scratch",
        scope={"listen": True, "reconcile": True},
        skipAudio=True,
    )
    assert full.status_code == 200, full.text
    assert "preference + correction combined" in full.json()["steps"]["notes"]

    _no_discovery(monkeypatch)
    _no_audio(monkeypatch)
    only = _analyze(guidance="change the C to a B in line 12")
    assert only.status_code == 200, only.text
    assert "standing preference held back" in only.json()["steps"]["notes"]

    # a notes-only run with the preference alone claims neither: nothing was
    # combined and nothing was withheld — it IS the run's only instruction
    alone = _analyze(scope={"listen": False, "reconcile": False})
    assert alone.status_code == 200, alone.text
    assert alone.json()["steps"]["notes"] == "ok: guidance applied (from stored notes)"


def test_a_withheld_preference_is_recorded_where_it_outlives_the_response(
    monkeypatch, isolated_store
):
    """`report.steps` is returned in the HTTP response and never persisted, so
    on its own "standing preference held back" reached exactly one reader once.
    reconcile/engine.py states the opposing rule for guidance that IS applied
    ("a reader of the song's history must be able to see that a human
    instruction shaped this run"); withholding one is the same claim."""
    isolated_store.save(_gold_song(), message="gold")
    get_song_notes_store().set_preference(SONG_ID, "capo-free voicings please")
    _no_discovery(monkeypatch)
    _no_audio(monkeypatch)
    monkeypatch.setattr(settings, "llm_provider", "mock")
    seen = _real_reconcile_spy(monkeypatch)

    r = _analyze(guidance="change the C to a B in line 12", provider="mock")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "standing preference held back" in body["steps"]["notes"]
    # the preference really was withheld from the model — that is what makes
    # the two records below a statement about something the run actually did
    assert seen[-1]["guidance"] == "change the C to a B in line 12"

    # 1. the run trace, which IS persisted
    trace = client.get(f"/v1/runs/{body['runId']}").json()
    withheld = [s for s in trace["steps"] if s["label"] == "notes:preference-withheld"]
    assert len(withheld) == 1
    assert withheld[0]["detail"]["preference"] == "capo-free voicings please"
    assert withheld[0]["detail"]["correction"] == "change the C to a B in line 12"

    # 2. the song's own provenance, beside the clause that names where the
    # applied guidance came from
    entry = [p for p in body["song"]["provenance"] if p["action"] == "reconciled"][-1]
    assert "guidance applied (from this request)" in entry["notes"]
    assert "standing preference held back" in entry["notes"]


def test_no_withholding_is_claimed_when_a_full_run_applied_both(monkeypatch, isolated_store):
    isolated_store.save(_gold_song(), message="gold")
    get_song_notes_store().set_preference(SONG_ID, "capo-free voicings please")
    monkeypatch.setattr(pipeline_mod, "discover_sources", lambda *a, **k: [])
    monkeypatch.setattr(settings, "llm_provider", "mock")
    seen = _real_reconcile_spy(monkeypatch)

    r = _analyze(
        guidance="change the C to a B in line 12",
        scope={"listen": True, "reconcile": True},
        skipAudio=True,
        provider="mock",
        # the mock reconciler emits no timing, and this song's prior version has
        # some; irrelevant to what is under test here, so opt out of the guard
        allowTimingLoss=True,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "capo-free" in seen[-1]["guidance"], "a full run is shown both"

    trace = client.get(f"/v1/runs/{body['runId']}").json()
    assert not [s for s in trace["steps"] if s["label"] == "notes:preference-withheld"]
    entry = [p for p in body["song"]["provenance"] if p["action"] == "reconciled"][-1]
    assert "held back" not in entry["notes"]


# --- no classification where nothing would read its answer -------------------


def test_an_explicit_non_notes_only_scope_pays_for_no_classification(
    monkeypatch, isolated_store
):
    """`classify_correction`'s fallback tier is a real blocking
    `provider.complete` against `settings.identity_model`. Under an explicit
    non-notes-only scope, `scope is None` blocks inference and `scope.notes_only`
    blocks `patch_ops_eligible`, so its answer is dead — one extra synchronous
    model call per request, for nothing."""
    isolated_store.save(_gold_song(), message="gold")
    monkeypatch.setattr(pipeline_mod, "discover_sources", lambda *a, **k: [])
    recorder = _Recorder(_gold_song())
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)
    classified = _classified(monkeypatch)
    llm_seen = _llm_tier_spy(monkeypatch)

    r = _analyze(
        guidance="please double-check this against better sources",  # no rule fires
        scope={"listen": True, "reconcile": True},
        skipAudio=True,
    )
    assert r.status_code == 200, r.text
    assert classified == [], "nothing would read the answer, so nothing was classified"
    assert llm_seen == [], "and so no model call was made"
    assert recorder.last["patch_ops_eligible"] is False


def test_an_explicit_notes_only_scope_still_classifies_for_patch_eligibility(
    monkeypatch, isolated_store
):
    """The other half of the same guard: under an explicit NOTES-ONLY scope the
    answer does have a reader (`patch_ops_eligible`), so the classification must
    still happen. Pinned so the skip above can never widen into it."""
    isolated_store.save(_gold_song(), message="gold")
    _no_discovery(monkeypatch)
    _no_audio(monkeypatch)
    recorder = _Recorder(_gold_song(), patch_ops_applied=1)
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)
    classified = _classified(monkeypatch)

    r = _analyze(
        guidance="change the C to a B in line 12", scope={"listen": False, "reconcile": False}
    )
    assert r.status_code == 200, r.text
    assert classified == ["change the C to a B in line 12"]
    assert recorder.last["patch_ops_eligible"] is True


# --- a patched result skips the passes that guard a REGENERATED document ---------


def test_a_patched_result_skips_timing_lrc_and_confidence(monkeypatch, isolated_store):
    isolated_store.save(_gold_song(), message="gold")
    _no_discovery(monkeypatch)
    _no_audio(monkeypatch)

    def boom_lrc(*a, **k):  # noqa: ANN001
        raise AssertionError("LRC ran on a patched result")

    monkeypatch.setattr(pipeline_mod, "fetch_lrc", boom_lrc)
    recorder = _Recorder(_gold_song(), patch_ops_applied=1)
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(guidance="change the C to a B in line 12")
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert steps["timing"].startswith("skipped (patch:")
    assert steps["lrc"].startswith("skipped (patch:")
    assert steps["confidence"].startswith("skipped (patch:")


def test_an_unpatched_notes_only_result_still_runs_the_guarding_passes(
    monkeypatch, isolated_store
):
    """The model declining the patch (full-reconcile fallback within notes-
    only scope) must still go through carry-forward — patch_ops_applied=0
    there, so this is the pre-existing path, unchanged."""
    isolated_store.save(_gold_song(), message="gold")
    _no_discovery(monkeypatch)
    _no_audio(monkeypatch)
    recorder = _Recorder(_gold_song(), patch_ops_applied=0)
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = _analyze(guidance="change the C to a B in line 12")
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert steps["timing"].startswith("ok:")
    assert "preserved" in steps["timing"] or "carried" in steps["timing"].lower()
