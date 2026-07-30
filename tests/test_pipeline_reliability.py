"""Analyze-pipeline reliability: never hang, fail loudly with the step name.

Covers the §3/§4 contract: a fatal step failure or timeout becomes a
502 whose detail names the step; best-effort steps (discover/acquire/mir) are
recorded as failed but don't sink the request; and the mock path runs the whole
analyze -> persist -> fetch -> versions flow with no external calls.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from snoocle_server import api as api_mod
from snoocle_server import pipeline as pipeline_mod
from snoocle_server.api import app
from snoocle_server.config import settings
from snoocle_server.discovery.search import SearchError
from snoocle_server.reconcile.engine import ReconcileError, ReconcileResult
from snoocle_server.schema import Song
from snoocle_server.store.memory import InMemorySongRepository

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch):
    store = InMemorySongRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: store)
    monkeypatch.setattr("snoocle_server.pipeline.get_store", lambda: store)
    # The 'anthropic' provider tests below reach reconcile on purpose; give the
    # provider preflight a credential so it doesn't short-circuit them. Tests
    # that assert the misconfigured-provider path set their own empty value.
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    # No test in this file should make a real network call to lrclib.net —
    # the mock-provider path already skips it (see pipeline.py), but a
    # non-mock provider that reaches the LRC step (e.g. a monkeypatched
    # reconcile) otherwise would. Tests that specifically want to exercise
    # the LRC merge override this back with their own monkeypatch.
    monkeypatch.setattr(pipeline_mod, "fetch_lrc", lambda *a, **k: None)
    return store


def _fake_song(song_id="anon--x") -> Song:
    return Song.model_validate(
        {
            "id": song_id,
            "metadata": {"title": "X", "artist": "Anon"},
            "lines": [{"lineIndex": 0, "lyrics": "la", "chordPlacements": [{"charIndex": 0, "chord": "C"}]}],
            "provenance": [{"timestamp": "2026-07-09T00:00:00Z", "actor": "reconcile:test/fake", "action": "reconciled"}],
        }
    )


def _fake_result(song_id="anon--x") -> ReconcileResult:
    return ReconcileResult(
        song=_fake_song(song_id), provider="anthropic", model="fake", attempts=1,
        audio_attached=False, usage={},
    )


def test_mock_analyze_is_fully_offline_and_persists(monkeypatch):
    """provider=mock must make ZERO external calls: any attempt to discover,
    acquire, or reconcile-over-network would raise here."""
    def boom(*a, **k):  # noqa: ANN001
        raise AssertionError("external call attempted in the mock path")

    monkeypatch.setattr(pipeline_mod, "discover_sources", boom)
    monkeypatch.setattr(pipeline_mod, "acquire", boom)

    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Offline", "artist": "Tester", "provider": "mock", "skipAudio": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["steps"]["discover"].startswith("skipped")
    assert body["storedVersion"]
    # persisted + fetchable + versioned
    sid = body["songId"]
    assert sid in client.get("/v1/songs").json()["songs"]
    assert client.get(f"/v1/songs/{sid}").status_code == 200
    assert client.get(f"/v1/songs/{sid}/versions").json()["versions"]

    # plan A5 acceptance: even the fully-offline mock path produces a run
    # record with a reviewQueue -- no candidates and no MIR means every
    # chord the mock reconciler invented has zero independent evidence, so
    # all of them land in the queue at the neutral default confidence.
    run = client.get(f"/v1/runs/{body['runId']}").json()
    assert isinstance(run.get("reviewQueue"), list)
    assert len(run["reviewQueue"]) == 4  # mock song: C, G, Am, F
    assert all(e["confidence"] == 0.5 for e in run["reviewQueue"])
    assert all(e["confidence"] <= run["reviewQueue"][i + 1]["confidence"]
               for i, e in enumerate(run["reviewQueue"][:-1]))


def test_fatal_reconcile_failure_returns_502_naming_step(monkeypatch):
    def fail(*a, **k):  # noqa: ANN001
        raise ReconcileError("provider exploded")

    monkeypatch.setattr(pipeline_mod, "reconcile", fail)
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Boom", "artist": "Tester", "provider": "mock", "skipAudio": True},
    )
    assert r.status_code == 502
    assert r.json()["detail"].startswith("reconcile: ")
    assert "provider exploded" in r.json()["detail"]


def test_fatal_502_detail_includes_step_outcomes(monkeypatch):
    """The 502 detail carries the per-step outcomes so the client can see WHY
    the fatal step had nothing to work with (not just that it failed)."""
    def fail(*a, **k):  # noqa: ANN001
        raise ReconcileError("nothing to reconcile: no candidate sources and no MIR analysis")

    monkeypatch.setattr(pipeline_mod, "reconcile", fail)
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Boom", "artist": "Tester", "provider": "mock", "skipAudio": True},
    )
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail.startswith("reconcile: nothing to reconcile")
    assert "[steps:" in detail
    assert "discover=" in detail and "mir=skipped" in detail


def test_youtube_auth_failure_gets_error_code_and_reason(monkeypatch):
    """When the run dies because the YouTube session is dead (bot-check /
    expired cookies), the 502 body carries a machine-readable errorCode and an
    action-oriented reason so the app can show a Reconnect YouTube action
    instead of a wall of diagnostics."""
    from snoocle_server.audio.acquire import YouTubeAuthError
    from snoocle_server.discovery.search import SearchError

    def fail_acquire(*a, **k):  # noqa: ANN001
        raise YouTubeAuthError(
            "yt-dlp failed for TJAfLE39ZZ8: Sign in to confirm you're not a bot."
        )

    def fail_discover(*a, **k):  # noqa: ANN001
        raise SearchError("all search backends failed: duckduckgo: 0 results")

    monkeypatch.setattr(pipeline_mod, "acquire", fail_acquire)
    monkeypatch.setattr(pipeline_mod, "discover_sources", fail_discover)
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Back To Black", "artist": "Amy Winehouse", "provider": "anthropic"},
    )
    assert r.status_code == 502
    body = r.json()
    assert body["errorCode"] == "youtube_auth_required"
    assert "Reconnect YouTube" in body["reason"]
    # the diagnostic detail is unchanged for humans/logs
    assert body["detail"].startswith("reconcile: ")
    assert "[steps:" in body["detail"]


def test_content_filter_failure_gets_error_code_and_reason(monkeypatch):
    """A provider content-filter block surfaces as a clean errorCode + reason,
    not a raw API 400."""
    from snoocle_server.reconcile.providers import ContentFilterError

    def fail(*a, **k):  # noqa: ANN001
        raise ContentFilterError("anthropic-agent: output blocked by content-filtering policy")

    monkeypatch.setattr(pipeline_mod, "reconcile", fail)
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Get Back", "artist": "Corey Heuvel", "provider": "mock", "skipAudio": True},
    )
    assert r.status_code == 502
    body = r.json()
    assert body["errorCode"] == "content_filtered"
    assert "content-filtering" in body["reason"] or "content filter" in body["reason"].lower()


def test_non_auth_failures_have_no_error_code(monkeypatch):
    def fail(*a, **k):  # noqa: ANN001
        raise ReconcileError("provider exploded")

    monkeypatch.setattr(pipeline_mod, "reconcile", fail)
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Boom", "artist": "Tester", "provider": "mock", "skipAudio": True},
    )
    assert r.status_code == 502
    assert "errorCode" not in r.json()


def test_misconfigured_provider_fails_before_expensive_steps(monkeypatch):
    """A provider that can't serve ANY request (missing credential/endpoint)
    must be rejected instantly — discover/acquire/mir never run, so a client
    that retries the 502 doesn't loop over minutes of doomed work each time."""
    def boom(*a, **k):  # noqa: ANN001
        raise AssertionError("expensive step ran despite misconfigured provider")

    monkeypatch.setattr(pipeline_mod, "discover_sources", boom)
    monkeypatch.setattr(pipeline_mod, "acquire", boom)
    monkeypatch.setattr(pipeline_mod, "analyze_audio", boom)
    monkeypatch.setattr(settings, "agent_mcp_url", "")  # 'agent' provider unusable

    r = client.post(
        "/v1/songs/analyze",
        json={"title": "X", "artist": "Anon", "provider": "agent"},
    )
    assert r.status_code == 502
    body = r.json()
    assert body["errorCode"] == "provider_not_configured"
    assert "not configured" in body["detail"]
    assert "SNOOCLE_AGENT_MCP_URL" in body["detail"]
    assert "reason" in body


def test_unknown_provider_fails_before_expensive_steps(monkeypatch):
    def boom(*a, **k):  # noqa: ANN001
        raise AssertionError("expensive step ran despite unknown provider")

    monkeypatch.setattr(pipeline_mod, "discover_sources", boom)
    monkeypatch.setattr(pipeline_mod, "acquire", boom)

    r = client.post(
        "/v1/songs/analyze",
        json={"title": "X", "artist": "Anon", "provider": "does-not-exist"},
    )
    assert r.status_code == 502
    assert r.json()["errorCode"] == "provider_not_configured"
    assert "unknown LLM provider" in r.json()["detail"]


def test_reconcile_timeout_returns_502(monkeypatch):
    monkeypatch.setattr(settings, "reconcile_timeout_seconds", 0.2)

    def slow(*a, **k):  # noqa: ANN001
        time.sleep(1.0)
        return _fake_result()

    monkeypatch.setattr(pipeline_mod, "reconcile", slow)
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Slow", "artist": "Tester", "provider": "mock", "skipAudio": True},
    )
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail.startswith("reconcile: ") and "timed out" in detail


def test_best_effort_discover_failure_does_not_sink_request(monkeypatch):
    """A non-mock provider whose discovery fails still succeeds if reconcile
    can proceed — the failure is recorded, not fatal."""
    def fail_discover(*a, **k):  # noqa: ANN001
        raise SearchError("all search backends down")

    monkeypatch.setattr(pipeline_mod, "discover_sources", fail_discover)
    monkeypatch.setattr(pipeline_mod, "reconcile", lambda *a, **k: _fake_result("anon--x"))

    r = client.post(
        "/v1/songs/analyze",
        json={"title": "X", "artist": "Anon", "provider": "anthropic", "skipAudio": True},
    )
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert steps["discover"].startswith("failed")
    assert steps["reconcile"].startswith("ok")
    assert r.json()["storedVersion"]


def test_timing_snap_populates_chord_times_end_to_end(monkeypatch, isolated_store):
    """After a successful reconcile+MIR, the stored song gains chord/line
    times and a regenerated syncMap -- the plan-A2 deterministic pass wired
    into the pipeline, not just unit-tested in isolation."""
    from snoocle_server.audio.acquire import AcquiredAudio
    from snoocle_server.mir.base import ChordSegment, MirAnalysis

    def fake_song(song_id="anon--x") -> Song:
        return Song.model_validate(
            {
                "id": song_id,
                "metadata": {"title": "X", "artist": "Anon"},
                "lines": [
                    {
                        "lineIndex": 0,
                        "lyrics": "la la",
                        "chordPlacements": [{"charIndex": 0, "chord": "C"}],
                    }
                ],
                "provenance": [
                    {"timestamp": "2026-07-09T00:00:00Z", "actor": "reconcile:test/fake", "action": "reconciled"}
                ],
            }
        )

    monkeypatch.setattr(pipeline_mod, "discover_sources", lambda *a, **k: [])
    monkeypatch.setattr(
        pipeline_mod,
        "acquire",
        lambda *a, **k: AcquiredAudio(
            video_id="dQw4w9WgXcQ", video_title="X", path="/tmp/fake.wav", duration_seconds=3.0
        ),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "analyze_audio",
        lambda *a, **k: MirAnalysis(
            engines={"chords": "chord-cnn-lstm"},
            chords=[ChordSegment(start=1.5, end=3.0, chord="C")],
        ),
    )
    monkeypatch.setattr(
        pipeline_mod, "reconcile", lambda *a, **k: _fake_result_from(fake_song("anon--x"))
    )

    r = client.post(
        "/v1/songs/analyze",
        json={"title": "X", "artist": "Anon", "provider": "anthropic"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["steps"]["timing"].startswith("ok")

    stored = isolated_store.get("anon--x")
    assert stored.lines[0].chordPlacements[0].timeSeconds == 1.5
    assert stored.lines[0].timeSeconds == 1.5
    assert [(sp.lineIndex, sp.time) for sp in stored.audio.syncMap] == [(0, 1.5)]
    assert stored.audio.analyzedVideoId == "dQw4w9WgXcQ"
    assert any(p.action == "timing-snap" for p in stored.provenance)


def test_timing_collapse_guard_runs_after_snap_and_reports_coverage(monkeypatch, isolated_store):
    """The generic collapse guard (5c2) is wired in after snap_chords: when
    the MIR chord timeline runs out early -- exactly the lo-fi-audio shape,
    not the fade-out one already fixed at its source -- 3+ trailing lines
    hold the same interpolated time with nothing later to spread toward, and
    the guard reports that (rather than inventing spacing) plus a timing
    coverage figure, end to end through the real pipeline."""
    from snoocle_server.audio.acquire import AcquiredAudio
    from snoocle_server.mir.base import ChordSegment, MirAnalysis

    def fake_song(song_id="anon--y") -> Song:
        chords = ["C", "G", "Am", "F", "C"]
        return Song.model_validate(
            {
                "id": song_id,
                "metadata": {"title": "Y", "artist": "Anon"},
                "lines": [
                    {
                        "lineIndex": i,
                        "lyrics": f"line {i}",
                        "chordPlacements": [{"charIndex": 0, "chord": c}],
                    }
                    for i, c in enumerate(chords)
                ],
                "provenance": [
                    {"timestamp": "2026-07-09T00:00:00Z", "actor": "reconcile:test/fake", "action": "reconciled"}
                ],
            }
        )

    monkeypatch.setattr(pipeline_mod, "discover_sources", lambda *a, **k: [])
    monkeypatch.setattr(
        pipeline_mod,
        "acquire",
        lambda *a, **k: AcquiredAudio(
            video_id="dQw4w9WgXcQ", video_title="Y", path="/tmp/fake.wav", duration_seconds=60.0
        ),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "analyze_audio",
        lambda *a, **k: MirAnalysis(
            duration_seconds=60.0,
            engines={"chords": "chord-cnn-lstm"},
            # Only C and G are ever heard -- like a chord timeline that runs
            # out on lo-fi audio, everything after the second match has
            # nothing later to interpolate toward.
            chords=[
                ChordSegment(start=1.0, end=2.0, chord="C"),
                ChordSegment(start=2.0, end=3.0, chord="G"),
            ],
        ),
    )
    monkeypatch.setattr(
        pipeline_mod, "reconcile", lambda *a, **k: _fake_result_from(fake_song("anon--y"))
    )

    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Y", "artist": "Anon", "provider": "anthropic"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["steps"]["timing-collapse-guard"].startswith("ok")

    stored = isolated_store.get("anon--y")
    times = [line.timeSeconds for line in stored.lines]
    # the 3 trailing lines (Am, F, C) all held the last match's time (2.0) --
    # left exactly as found, since nothing later exists to spread toward
    assert times[2:] == [2.0, 2.0, 2.0]

    guard_entries = [p for p in stored.provenance if p.action == "timing-collapse-guard"]
    assert len(guard_entries) == 1
    assert "left alone" in guard_entries[0].notes
    assert "no later anchor" in guard_entries[0].notes
    assert "timing coverage:" in guard_entries[0].notes


def _fake_result_from(song: Song) -> ReconcileResult:
    return ReconcileResult(
        song=song, provider="anthropic", model="fake", attempts=1, audio_attached=False, usage={}
    )


# --- re-analysis scope -------------------------------------------------------
# Two independent evidence-gathering stages, each switchable: `listen` (acquire
# + MIR) and `reconcile` (web source discovery). The LLM reconciliation itself
# ALWAYS runs — the flags decide what evidence it gets. An ABSENT scope is not
# a scope: it must reproduce the pre-scope pipeline exactly.


class _Recorder:
    """Stands in for `reconcile`, capturing what the pipeline handed it."""

    def __init__(self, song_id="anon--x"):
        self.calls: list[dict] = []
        self.song_id = song_id

    def __call__(self, title, artist, candidates, mir, **kwargs):
        # candidates/mir ride positionally through `_step_reconcile`; normalize
        # everything into one dict so the assertions read the same either way.
        self.calls.append({"candidates": candidates, "mir": mir, **kwargs})
        return _fake_result(self.song_id)

    @property
    def last(self) -> dict:
        assert self.calls, "reconcile was never called"
        return self.calls[-1]


def _no_scope_analyze_body(**extra) -> dict:
    body = {"title": "X", "artist": "Anon", "provider": "anthropic"}
    body.update(extra)
    return body


def test_absent_scope_leaves_the_pipeline_exactly_as_before(monkeypatch):
    """The compatibility guarantee: a request with no `scope` key runs every
    stage and its report grows no new entries. Every pre-scope client depends
    on this, so it is asserted positively rather than by omission."""
    discovered: list[str] = []
    monkeypatch.setattr(
        pipeline_mod, "discover_sources", lambda *a, **k: discovered.append("ran") or []
    )
    recorder = _Recorder()
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = client.post("/v1/songs/analyze", json=_no_scope_analyze_body(skipAudio=True))
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert discovered == ["ran"], "discovery must still run without a scope"
    assert steps["discover"].startswith("ok")
    assert "scope" not in steps, "an absent scope must not add a step to the report"
    assert recorder.last["scope"] is None


def test_scope_reconcile_off_skips_discovery_and_says_so(monkeypatch):
    def boom(*a, **k):  # noqa: ANN001
        raise AssertionError("discovery ran despite reconcile=false")

    monkeypatch.setattr(pipeline_mod, "discover_sources", boom)
    recorder = _Recorder()
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    r = client.post(
        "/v1/songs/analyze",
        json=_no_scope_analyze_body(skipAudio=True, scope={"listen": True, "reconcile": False}),
    )
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert steps["discover"] == "skipped (scope: no new sources)"
    assert steps["scope"] == "listen=on, reconcile=off"
    assert recorder.last["candidates"] == []


def test_scope_listen_off_skips_acquire_and_mir(monkeypatch):
    def boom(*a, **k):  # noqa: ANN001
        raise AssertionError("audio was acquired/analyzed despite listen=false")

    monkeypatch.setattr(pipeline_mod, "acquire", boom)
    monkeypatch.setattr(pipeline_mod, "analyze_audio", boom)
    monkeypatch.setattr(pipeline_mod, "discover_sources", lambda *a, **k: [])
    recorder = _Recorder()
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    # listen=off means "reuse the existing audio analysis", so the run needs
    # one to reuse — see tests/test_timing_carry_forward.py for the pass that
    # carries it forward and for the failure when there is nothing to carry.
    r = client.post(
        "/v1/songs/analyze",
        json=_no_scope_analyze_body(
            scope={"listen": False, "reconcile": True},
            priorSong=_fake_song().model_dump(mode="json"),
        ),
    )
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert steps["acquire"].startswith("skipped (scope:")
    assert steps["mir"].startswith("skipped (scope:")
    assert steps["scope"] == "listen=off, reconcile=on"
    assert recorder.last["mir"] is None


def test_scope_both_off_gathers_nothing_but_still_reconciles(monkeypatch):
    """The owner's "no checkbox" case: apply my notes to the song I already
    have. Nothing is fetched, but the reconciler still runs and still receives
    the prior song AND the guidance — without both it would have nothing to
    correct and nothing to correct it with."""
    def boom(*a, **k):  # noqa: ANN001
        raise AssertionError("evidence was gathered despite an all-off scope")

    monkeypatch.setattr(pipeline_mod, "discover_sources", boom)
    monkeypatch.setattr(pipeline_mod, "acquire", boom)
    monkeypatch.setattr(pipeline_mod, "analyze_audio", boom)
    recorder = _Recorder()
    monkeypatch.setattr(pipeline_mod, "reconcile", recorder)

    prior = _fake_song("anon--x").model_dump()
    r = client.post(
        "/v1/songs/analyze",
        json=_no_scope_analyze_body(
            scope={"listen": False, "reconcile": False},
            guidance="the bridge is in D",
            priorSong=prior,
        ),
    )
    assert r.status_code == 200, r.text
    steps = r.json()["steps"]
    assert steps["discover"] == "skipped (scope: notes only)"
    assert steps["acquire"].startswith("skipped")
    assert steps["mir"].startswith("skipped")
    assert steps["reconcile"].startswith("ok"), "reconciliation must still run"

    call = recorder.last
    assert call["prior_song"] == prior
    assert call["guidance"] == "the bridge is in D"
    assert call["scope"].notes_only
    assert call["candidates"] == [] and call["mir"] is None, "nothing was gathered"


def test_partial_scope_object_only_turns_off_what_it_names(monkeypatch):
    """`{"listen": false}` must not silently disable discovery too — a missing
    flag defaults ON, so a partial object can only ever narrow what it says."""
    ran: list[str] = []
    monkeypatch.setattr(
        pipeline_mod, "discover_sources", lambda *a, **k: ran.append("discover") or []
    )
    monkeypatch.setattr(pipeline_mod, "reconcile", _Recorder())

    r = client.post(
        "/v1/songs/analyze",
        json=_no_scope_analyze_body(
            scope={"listen": False},
            priorSong=_fake_song().model_dump(mode="json"),  # listen=off needs one
        ),
    )
    assert r.status_code == 200, r.text
    assert ran == ["discover"]
    assert r.json()["steps"]["scope"] == "listen=off, reconcile=on"


def test_scope_is_recorded_in_the_reconcile_provenance():
    """A version built from the prior song plus notes is a different artifact
    from a full re-analysis; six months later the provenance is the only place
    that difference survives. Runs through the REAL reconciler (mock provider)
    so the provenance is the one the engine actually writes."""
    r = client.post(
        "/v1/songs/analyze",
        json={
            "title": "Scoped", "artist": "Tester", "provider": "mock",
            "scope": {"listen": False, "reconcile": True},
            # listen=off reuses the prior version's audio analysis, so it needs
            # one to reuse (tests/test_timing_carry_forward.py).
            "priorSong": _fake_song("tester--scoped").model_dump(mode="json"),
        },
    )
    assert r.status_code == 200, r.text
    entries = [p for p in r.json()["song"]["provenance"] if p["action"] == "reconciled"]
    assert entries, "no reconciled provenance entry"
    notes = entries[-1]["notes"]
    assert "scope: listen=off, reconcile=on" in notes
    # …alongside, not instead of, the existing text.
    assert "chord rule enforced by schema validation" in notes


def test_absent_scope_writes_no_scope_provenance():
    r = client.post(
        "/v1/songs/analyze",
        json={"title": "Unscoped", "artist": "Tester", "provider": "mock", "skipAudio": True},
    )
    assert r.status_code == 200, r.text
    entries = [p for p in r.json()["song"]["provenance"] if p["action"] == "reconciled"]
    assert "scope:" not in (entries[-1]["notes"] or "")


@pytest.mark.parametrize(
    "scope_arg, expected",
    [
        (None, None),
        ({"listen": False, "reconcile": False}, (False, False)),
        ({"listen": True, "reconcile": False}, (True, False)),
        ({"reconcile": False}, (True, False)),  # missing flag defaults ON
    ],
)
def test_mcp_tool_accepts_the_same_scope_shape(monkeypatch, scope_arg, expected):
    """Parity: the MCP tool takes the wire object verbatim, and an omitted
    scope stays omitted all the way down (None, not a full-scope object)."""
    import asyncio

    from snoocle_server import mcp_server as mcp_mod

    seen: dict = {}

    async def fake_pipeline(*a, **k):
        seen.update(k)
        report = pipeline_mod.PipelineReport(song_id="anon--x")
        report.reconcile = _fake_result()
        return report

    monkeypatch.setattr(mcp_mod, "run_pipeline_async", fake_pipeline)
    asyncio.run(mcp_mod.analyze_and_store_song(title="X", artist="Anon", scope=scope_arg))

    scope = seen["scope"]
    if expected is None:
        assert scope is None, "an omitted MCP scope must not become a full-scope object"
    else:
        assert (scope.listen, scope.reconcile) == expected


def test_notes_only_prompt_says_nothing_was_gathered():
    """With zero candidates and no MIR the prompt must SAY so. A model handed
    two empty evidence sections and no explanation concludes the evidence
    failed to load and reconstructs the song from memory — the exact
    hallucination this scope exists to avoid."""
    from snoocle_server.reconcile.prompt import build_user_prompt
    from snoocle_server.schema import song_json_schema
    from snoocle_server.scope import AnalysisScope

    prior = _fake_song("anon--x").model_dump()
    prompt = build_user_prompt(
        "X", "Anon", [], None, song_json_schema(), "anon--x", None,
        guidance="the bridge is in D",
        prior_song=prior,
        scope=AnalysisScope(listen=False, reconcile=False),
    )
    assert "apply the user's notes only" in prompt
    assert "No new sources were gathered" in prompt
    assert "do not invent a song" in prompt
    # The empty candidate section can no longer read "0 found — use ALL of them".
    assert "use ALL of them" not in prompt
    assert "NONE were gathered for this run" in prompt
    # The prior song is the starting document, not one opinion among several.
    assert "this is your starting document" in prompt
    assert "the bridge is in D" in prompt


def test_reconcile_accepts_a_prior_song_as_its_only_input():
    """The guard that stops the reconciler inventing a song from nothing has to
    let the notes-only scope through — a prior song IS something to work on —
    while still rejecting a run that has neither."""
    from snoocle_server.reconcile.engine import reconcile as real_reconcile
    from snoocle_server.scope import AnalysisScope

    scope = AnalysisScope(listen=False, reconcile=False)
    with pytest.raises(ReconcileError) as exc:
        real_reconcile(
            "X", "Anon", [], None, provider_name="anthropic", prior_song=None, scope=scope
        )
    assert "nothing to reconcile" in str(exc.value)
    assert "no prior song to correct" in str(exc.value)

    # With a prior song the guard no longer fires — the run gets as far as the
    # provider (which has no real credential here, so it fails LATER and with a
    # different error).
    with pytest.raises(Exception) as later:
        real_reconcile(
            "X", "Anon", [], None, provider_name="anthropic",
            prior_song=_fake_song("anon--x").model_dump(), scope=scope,
        )
    assert "nothing to reconcile" not in str(later.value)
