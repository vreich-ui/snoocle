"""Mode B end to end: re-align a stored document to a different recording.

The claims worth pinning, in the brief's own terms:

- a transposed recording gets the right transposition, and the lyrics come
  through byte-identically;
- a structurally identical recording invokes NO model — a live version of a
  song already in the library costs zero model tokens;
- a structurally different one does invoke it, and gets told what differs;
- the stored version records which video supplied its timing;
- the result is stored under the ORIGINAL song id, as a new version.

Nothing here touches the network or a model: acquisition, MIR, LRC and
reconciliation are all stood in for, so these tests are about realign.py's own
wiring.
"""

from __future__ import annotations

import pytest

from snoocle_server import realign as realign_mod
from snoocle_server.audio.acquire import AcquiredAudio
from snoocle_server.chords import transpose_chord
from snoocle_server.config import settings
from snoocle_server.mir.base import Beat, ChordSegment, MirAnalysis, StructureSegment
from snoocle_server.mir.cache import MirCacheInfo
from snoocle_server.realign import ACTION, RealignError, realign_song
from snoocle_server.reconcile.engine import ReconcileResult
from snoocle_server.schema import Song
from snoocle_server.store.memory import InMemorySongRepository
from snoocle_server.timing.offset import OffsetEstimate

SONG_ID = "the-rolling-stones--paint-it-black"
NEW_VIDEO = "newvideo123"
OLD_VIDEO = "oldvideo123"
DURATION = 120.0
PROGRESSION = ["Em", "B", "Em", "B", "D", "A", "Em", "G", "D", "A", "Em", "B"]
LYRICS = [f"line {i} of the song, words that must survive verbatim" for i in range(12)]


def _mir(
    chords: list[str],
    *,
    duration: float = DURATION,
    span: float | None = None,
    sections: int = 4,
    key: str = "E minor",
) -> MirAnalysis:
    """An analysis of a recording `duration` long.

    `span` shorter than the duration is the lo-fi signature: the chord
    recognizer stopped producing segments partway through the track.
    """
    span = span if span is not None else duration
    step = span / len(chords)
    seg = duration / max(sections, 1)
    return MirAnalysis(
        engines={"chords": "chord-cnn-lstm", "beats": "madmom", "structure": "songformer"},
        duration_seconds=duration,
        bpm=160.0,
        key=key,
        beats=[Beat(time=i * 0.5, position=(i % 4) + 1) for i in range(int(duration / 0.5))],
        chords=[
            ChordSegment(start=i * step, end=(i + 1) * step, chord=c)
            for i, c in enumerate(chords)
        ],
        sections=[
            StructureSegment(start=i * seg, end=(i + 1) * seg, label="verse")
            for i in range(sections)
        ],
    )


def _stored_song(chords: list[str] = tuple(PROGRESSION)) -> Song:
    """A finished document: words and chords settled, timed against OLD_VIDEO."""
    lines = [
        {
            "lineIndex": i,
            "lyrics": LYRICS[i],
            "timeSeconds": i * 10.0,
            "confidence": 0.9,
            "chordPlacements": [
                {"charIndex": 5, "chord": chord, "timeSeconds": i * 10.0, "confidence": 0.9}
            ],
        }
        for i, chord in enumerate(chords)
    ]
    return Song.model_validate(
        {
            "id": SONG_ID,
            "metadata": {
                "title": "Paint It Black", "artist": "The Rolling Stones",
                "key": "E minor", "bpm": 160.0,
            },
            "audio": {
                "youtubeVideoId": OLD_VIDEO,
                "analyzedVideoId": OLD_VIDEO,
                "durationSeconds": DURATION,
                "beats": [{"time": 0.0, "measure": 1, "beatInMeasure": 1}],
                "syncMap": [{"lineIndex": i, "time": i * 10.0} for i in range(len(chords))],
            },
            "sections": [
                {"sectionIndex": 0, "name": "Verse 1", "kind": "verse",
                 "startLineIndex": 0, "endLineIndex": 5,
                 "startTime": 0.0, "endTime": 60.0},
                {"sectionIndex": 1, "name": "Chorus", "kind": "chorus",
                 "startLineIndex": 6, "endLineIndex": 11,
                 "startTime": 60.0, "endTime": DURATION},
            ],
            "lines": lines,
            "provenance": [
                {"timestamp": "2026-07-01T00:00:00Z", "actor": "reconcile:test/gold",
                 "action": "reconciled", "confidence": 0.85},
            ],
        }
    )


class _NoModel:
    """A `reconcile` stand-in that fails the test if it is ever called.

    This is how "zero model tokens" gets asserted: not by counting tokens, but
    by making the call itself impossible to make quietly.
    """

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        raise AssertionError(
            "the model was consulted on a structurally identical recording — "
            "Mode B must cost zero model tokens there"
        )


class _Model:
    """A `reconcile` stand-in that records its call and returns `song`."""

    def __init__(self, song: Song):
        self.song = song
        self.calls: list[dict] = []

    def __call__(self, title, artist, candidates, mir, **kwargs):
        self.calls.append({"title": title, "artist": artist,
                           "candidates": candidates, "mir": mir, **kwargs})
        return ReconcileResult(
            song=self.song.model_copy(deep=True), provider="test", model="fake-model",
            attempts=1, audio_attached=False, usage={},
            # The engine gives a freshly-finalized document only its OWN
            # provenance; realign.py splices the stored history back in.
            trace=None,
        )


@pytest.fixture
def store(monkeypatch):
    repo = InMemorySongRepository()
    monkeypatch.setattr(realign_mod, "get_store", lambda: repo)
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    # No network: no LRCLIB lookup, and no model unless a test installs one.
    monkeypatch.setattr(realign_mod, "fetch_lrc", lambda *a, **k: None)
    monkeypatch.setattr(realign_mod, "reconcile", _NoModel())
    return repo


def _wire(monkeypatch, mir: MirAnalysis, *, video_id: str = NEW_VIDEO) -> None:
    monkeypatch.setattr(
        realign_mod, "_step_acquire",
        lambda vid: AcquiredAudio(
            video_id=video_id, video_title="Live 1966", path="/dev/null",
            duration_seconds=mir.duration_seconds,
        ),
    )
    monkeypatch.setattr(
        realign_mod, "_step_mir",
        lambda *a, **k: (mir, MirCacheInfo(status="miss", analyzed_at="2026-07-30T00:00:00Z")),
    )


def _model_song(chords: list[str], lyrics: list[str]) -> Song:
    """What a structural fix looks like coming back from the model: the same
    words and chords, re-grouped (here: one extra repeat of the chorus)."""
    song = _stored_song().model_dump()
    song["lines"] = [
        {"lineIndex": i, "lyrics": lyrics[i],
         "chordPlacements": [{"charIndex": 5, "chord": chord}]}
        for i, chord in enumerate(chords)
    ]
    song["sections"] = [
        {"sectionIndex": 0, "name": "Verse 1", "kind": "verse",
         "startLineIndex": 0, "endLineIndex": 5},
        {"sectionIndex": 1, "name": "Chorus", "kind": "chorus",
         "startLineIndex": 6, "endLineIndex": len(chords) - 1},
    ]
    song["audio"]["syncMap"] = []
    song["provenance"] = [
        {"timestamp": "2026-07-30T00:00:00Z", "actor": "reconcile:test/fake-model",
         "action": "reconciled"}
    ]
    return Song.model_validate(song)


# --- the transposed case -----------------------------------------------------


def test_a_transposed_recording_gets_the_right_transposition_and_keeps_the_lyrics(
    monkeypatch, store
):
    """The document is in E minor; this recording is a whole tone up. The chords
    move +2 and not one character of a lyric changes."""
    store.save(_stored_song(), message="gold")
    recording = [transpose_chord(c, 2) for c in PROGRESSION]
    _wire(monkeypatch, _mir(recording, key="F# minor"))

    report = realign_song(SONG_ID, NEW_VIDEO)

    assert report.transposition.semitones == 2
    assert report.transposition.applies is True
    assert report.model_consulted is False

    stored = store.get(SONG_ID)
    assert [p.chord for line in stored.lines for p in line.chordPlacements] == recording
    assert [line.lyrics for line in stored.lines] == LYRICS  # byte-identical
    assert stored.metadata.key == "F# minor"
    # And it is genuinely timed to the new recording, not carrying old times.
    assert [line.timeSeconds for line in stored.lines] == [
        pytest.approx(i * 10.0) for i in range(12)
    ]


def test_an_untrustworthy_transposition_is_not_applied(monkeypatch, store):
    """A recording that does not match the document at any shift must not have
    its best coincidence applied to the chords."""
    store.save(_stored_song(), message="gold")
    unrelated = ["Dm", "Eb", "Dm", "Bb", "Gm", "Db", "Dm", "Ab", "Dm", "Eb", "Cm", "Dm"]
    _wire(monkeypatch, _mir(unrelated))

    report = realign_song(SONG_ID, NEW_VIDEO)

    assert report.transposition.trustworthy is False
    assert report.transposition.semitones == 0
    assert [p.chord for line in store.get(SONG_ID).lines for p in line.chordPlacements] == list(
        PROGRESSION
    )
    assert "indistinguishable from coincidence" in report.steps["transpose"]


# --- the model gate ----------------------------------------------------------


def test_a_structurally_identical_recording_costs_zero_model_tokens(monkeypatch, store):
    """The library-economics claim, asserted: `reconcile` raises if called."""
    store.save(_stored_song(), message="gold")
    _wire(monkeypatch, _mir(PROGRESSION))

    report = realign_song(SONG_ID, NEW_VIDEO)

    assert report.model_consulted is False
    assert report.structure.explained is True
    assert report.steps["model"].startswith("not consulted")
    assert report.stored_version, "a deterministic re-align still stores a version"


def test_a_structurally_different_recording_consults_the_model_with_the_difference(
    monkeypatch, store
):
    """A live take with an extra chorus: the document cannot explain the extra
    time, so the one thing that can spend does."""
    store.save(_stored_song(), message="gold")
    extended = list(PROGRESSION) + ["Em", "G", "D", "A", "Em", "B"]
    extended_lyrics = LYRICS + LYRICS[6:]
    _wire(monkeypatch, _mir(extended, duration=DURATION * 1.5, sections=6))
    model = _Model(_model_song(extended, extended_lyrics))
    monkeypatch.setattr(realign_mod, "reconcile", model)

    report = realign_song(SONG_ID, NEW_VIDEO)

    assert report.model_consulted is True
    assert report.structure.explained is False
    assert len(model.calls) == 1

    call = model.calls[0]
    # It is told what differs, and only that.
    feedback = call["structure_feedback"]
    assert "50% longer" in feedback
    assert "add or remove repeats" in feedback
    assert "Do not re-transcribe" in feedback
    # And it works from the document, with no new sources gathered.
    assert call["candidates"] == []
    assert call["scope"].reconcile is False
    assert call["scope"].listen is True
    assert call["prior_song"]["id"] == SONG_ID
    # The prior song handed over is already transposed and already stripped of
    # the old recording's times — nothing for the model to re-derive or copy.
    assert all(line["timeSeconds"] is None for line in call["prior_song"]["lines"])

    stored = store.get(SONG_ID)
    assert len(stored.lines) == len(extended)
    assert [line.lyrics for line in stored.lines] == extended_lyrics


def test_an_unmeasurable_structure_does_not_consult_the_model(monkeypatch, store):
    """"I cannot tell whether the structure differs" is not a reason to spend."""
    bare = _stored_song().model_dump()
    bare["audio"]["durationSeconds"] = None
    for line in bare["lines"]:
        line["timeSeconds"] = None
        line["confidence"] = None
        for p in line["chordPlacements"]:
            p["timeSeconds"] = None
            p["confidence"] = None
    bare["audio"]["syncMap"] = []
    for s in bare["sections"]:
        s["startTime"] = s["endTime"] = None
    store.save(Song.model_validate(bare), message="gold")
    _wire(monkeypatch, MirAnalysis(engines={"chords": "x"}, chords=[
        ChordSegment(start=0.0, end=5.0, chord="Em"),
    ]))

    report = realign_song(SONG_ID, NEW_VIDEO, allow_timing_loss=True)

    assert report.structure.comparable is False
    assert report.model_consulted is False
    assert "no structural difference was measurable" in report.steps["model"]


# --- what the stored version records -----------------------------------------


def test_the_stored_version_records_its_timing_source(monkeypatch, store):
    store.save(_stored_song(), message="gold")
    recording = [transpose_chord(c, 2) for c in PROGRESSION]
    _wire(monkeypatch, _mir(recording, key="F# minor"))

    report = realign_song(SONG_ID, NEW_VIDEO)
    stored = store.get(SONG_ID)

    # The schema field that exists for exactly this: which upload the times
    # were measured against. Playback follows it.
    assert stored.audio.analyzedVideoId == NEW_VIDEO
    assert stored.audio.youtubeVideoId == NEW_VIDEO
    assert stored.audio.durationSeconds == DURATION

    entry = next(p for p in stored.provenance if p.action == ACTION)
    assert f"timing-video:{NEW_VIDEO}" in entry.sources
    assert f"source-document:{report.source_version}" in entry.sources
    assert "transposed +2 semitone(s)" in entry.notes
    assert "model NOT consulted" in entry.notes
    assert "lyrics and chord sequence carried" in entry.notes

    # The prior history survives, and this run's passes append to it.
    actions = [p.action for p in stored.provenance]
    assert actions[0] == "reconciled"  # the original run
    assert ACTION in actions
    assert "timing-snap" in actions
    assert "timing-collapse-guard" in actions
    assert "quality-grade" in actions


def test_a_model_consulted_realign_says_so_in_provenance(monkeypatch, store):
    store.save(_stored_song(), message="gold")
    extended = list(PROGRESSION) + ["Em", "G", "D", "A", "Em", "B"]
    _wire(monkeypatch, _mir(extended, duration=DURATION * 1.5, sections=6))
    monkeypatch.setattr(
        realign_mod, "reconcile", _Model(_model_song(extended, LYRICS + LYRICS[6:]))
    )

    realign_song(SONG_ID, NEW_VIDEO)
    stored = store.get(SONG_ID)

    entry = next(p for p in stored.provenance if p.action == ACTION)
    assert "model consulted: test/fake-model" in entry.notes
    # The stored history is continuous, not restarted by the model's document.
    assert [p.action for p in stored.provenance][0] == "reconciled"
    assert [p.actor for p in stored.provenance][0] == "reconcile:test/gold"


def test_the_result_is_a_new_version_of_the_same_song_never_a_new_song(monkeypatch, store):
    first = store.save(_stored_song(), message="gold")
    _wire(monkeypatch, _mir(PROGRESSION))

    report = realign_song(SONG_ID, NEW_VIDEO)

    assert store.list_songs() == [SONG_ID], "Mode B must never mint a second song"
    assert report.stored_version != first.version
    assert len(store.versions(SONG_ID)) == 2
    # The earlier version stays readable, still timed to the old recording.
    assert store.get(SONG_ID, first.version).audio.analyzedVideoId == OLD_VIDEO


def test_re_aligning_a_specific_source_version_reads_that_one(monkeypatch, store):
    """The operator may want the human-reviewed version as the source, not
    whatever landed last."""
    reviewed = store.save(_stored_song(), message="reviewed")
    drifted = _stored_song(["C"] * 12)
    store.save(drifted, message="a later, worse version")
    _wire(monkeypatch, _mir(PROGRESSION))

    report = realign_song(SONG_ID, NEW_VIDEO, source_version=reviewed.version)

    assert report.source_version == reviewed.version
    assert [
        p.chord for line in store.get(SONG_ID).lines for p in line.chordPlacements
    ] == list(PROGRESSION)


# --- the refusals ------------------------------------------------------------


def test_the_same_recording_is_refused_and_points_at_the_cheap_path(monkeypatch, store):
    """The one place Mode B and the video-offset endpoint meet: if the two
    videos are the same RECORDING, a constant shift exists and costs one
    cross-correlation instead of a download plus a full MIR pass."""
    store.save(_stored_song(), message="gold")
    _wire(monkeypatch, _mir(PROGRESSION))
    monkeypatch.setattr(
        realign_mod, "same_recording_check",
        lambda song, video_id: OffsetEstimate(offset_seconds=2.5, confidence=0.93),
    )

    with pytest.raises(RealignError) as excinfo:
        realign_song(SONG_ID, NEW_VIDEO)

    error = excinfo.value
    assert error.error_code == "same_recording_use_video_offset"
    assert "video-offset" in error.message
    assert "allowSameRecording" in error.message
    assert len(store.versions(SONG_ID)) == 1, "nothing stored on a refusal"


def test_the_same_recording_refusal_can_be_overridden(monkeypatch, store):
    store.save(_stored_song(), message="gold")
    _wire(monkeypatch, _mir(PROGRESSION))
    monkeypatch.setattr(
        realign_mod, "same_recording_check",
        lambda song, video_id: OffsetEstimate(offset_seconds=2.5, confidence=0.93),
    )

    report = realign_song(SONG_ID, NEW_VIDEO, allow_same_recording=True)
    assert report.stored_version
    assert report.steps["same-recording-check"] == "skipped (allowSameRecording)"


def test_a_low_confidence_offset_confirms_a_different_recording(monkeypatch, store):
    store.save(_stored_song(), message="gold")
    _wire(monkeypatch, _mir(PROGRESSION))
    monkeypatch.setattr(
        realign_mod, "same_recording_check",
        lambda song, video_id: OffsetEstimate(offset_seconds=0.0, confidence=0.11),
    )

    report = realign_song(SONG_ID, NEW_VIDEO)
    assert "a different recording" in report.steps["same-recording-check"]
    assert report.same_recording.confidence == 0.11


def test_a_realign_that_would_lose_audio_data_is_refused(monkeypatch, store):
    """Same rule the analyze pipeline holds: a version with LESS audio-derived
    data than the one it replaces is a downgrade, and downgrades are explicit."""
    store.save(_stored_song(), message="gold")
    # An analysis that produced no beats and no bpm at all.
    _wire(monkeypatch, MirAnalysis(engines={"chords": "x"}, duration_seconds=DURATION, chords=[
        ChordSegment(start=i * 10.0, end=(i + 1) * 10.0, chord=c)
        for i, c in enumerate(PROGRESSION)
    ]))

    with pytest.raises(RealignError) as excinfo:
        realign_song(SONG_ID, NEW_VIDEO)

    assert excinfo.value.error_code == "timing_data_loss"
    assert "metadata.bpm" in excinfo.value.message
    assert len(store.versions(SONG_ID)) == 1

    # ... and stores when the trade is stated.
    report = realign_song(SONG_ID, NEW_VIDEO, allow_timing_loss=True)
    assert report.stored_version
    assert "overridden" in report.steps["timing-guard"]


def test_a_missing_song_is_a_404_shaped_failure(monkeypatch, store):
    _wire(monkeypatch, _mir(PROGRESSION))
    with pytest.raises(RealignError) as excinfo:
        realign_song("no-such--song", NEW_VIDEO)
    assert excinfo.value.error_code == "song_not_found"
    assert excinfo.value.step == "source"


def test_an_unanalyzable_recording_fails_rather_than_storing_nothing(monkeypatch, store):
    """Mode A treats MIR as best-effort because a song can come from text
    alone. Mode B cannot: the new timing IS the deliverable."""
    store.save(_stored_song(), message="gold")
    monkeypatch.setattr(
        realign_mod, "_step_acquire",
        lambda vid: AcquiredAudio(video_id=NEW_VIDEO, video_title="X", path="/dev/null",
                                  duration_seconds=DURATION),
    )

    def boom(*a, **k):
        raise RuntimeError("librosa fell over")

    monkeypatch.setattr(realign_mod, "_step_mir", boom)

    with pytest.raises(RealignError) as excinfo:
        realign_song(SONG_ID, NEW_VIDEO)

    assert excinfo.value.step == "mir"
    assert excinfo.value.error_code == "mir_failed"
    assert len(store.versions(SONG_ID)) == 1


# --- grading and the run trace ----------------------------------------------


def test_a_realigned_version_is_graded_by_the_same_standard_as_any_run(monkeypatch, store):
    store.save(_stored_song(), message="gold")
    _wire(monkeypatch, _mir(PROGRESSION))

    report = realign_song(SONG_ID, NEW_VIDEO)

    assert report.quality is not None
    assert report.quality.grade.verdict in ("pass", "warn", "fail", "unknown")
    assert report.quality.escalation.retry is False, "Mode B never retries a model"

    from snoocle_server.store.runs import get_run_store

    trace = get_run_store().get_run(report.run_id)
    assert trace["provider"] == "realign"
    assert trace["quality"]["grade"]["verdict"] == report.quality.grade.verdict
    assert any(step["label"] == "realign-plan" for step in trace["steps"])
    assert "quality-grade" in [p.action for p in store.get(SONG_ID).provenance]


def test_no_text_sources_is_not_blamed_on_the_sources(monkeypatch, store):
    """Mode B gathers no candidates by design. An empty candidate set must not
    read as "the sources failed" — that would blame the wrong thing and mask
    the audio verdict that actually matters here."""
    store.save(_stored_song(), message="gold")
    # A timeline covering a third of the track: the lo-fi signature.
    _wire(monkeypatch, _mir(list(PROGRESSION)[:4], duration=DURATION, span=DURATION / 3))

    report = realign_song(SONG_ID, NEW_VIDEO, allow_timing_loss=True)

    assert report.quality.attribution.fault.value in ("audio", "unknown", "none")
    assert report.quality.attribution.fault.value != "source"


def test_a_realign_that_is_still_audio_bound_suggests_another_recording(monkeypatch, store):
    """The loop closes: if the new recording is ALSO unusable, the operator
    gets candidates rather than a silent bad version."""
    store.save(_stored_song(), message="gold")
    # Same length, but a chord timeline that dies a quarter of the way in: the
    # lo-fi signature, and NOT a structural difference (so still no model).
    _wire(monkeypatch, _mir(list(PROGRESSION)[:3], duration=DURATION, span=DURATION * 0.25))
    captured: list[dict] = []

    def fake_suggest(song, **kwargs):
        from snoocle_server.recordings import RecordingSuggestion, RecordingSuggestions

        captured.append({"song": song, **kwargs})
        return RecordingSuggestions(
            song_id=song.id,
            reason=kwargs.get("reason", ""),
            suggestions=[
                RecordingSuggestion(
                    video_id="studio12345", title="Paint It Black (Official Audio)",
                    channel="TheRollingStonesVEVO", duration_seconds=DURATION,
                    score=4.5, url="https://youtu.be/studio12345",
                    action="analyze studio12345 as the timing reference for " + SONG_ID,
                )
            ],
        )

    monkeypatch.setattr(realign_mod, "suggest_recordings", fake_suggest)

    report = realign_song(SONG_ID, NEW_VIDEO, allow_timing_loss=True)

    # Gated on `suggest_alternative_recording`, not `mark_timing_unreliable`:
    # BOTH an audio fault and a collapse the guard could not spread set
    # `mark_timing_unreliable`, but only an audio fault also looks for a
    # different recording (see `test_a_mode_b_collapse_only_fail_marks_but_
    # does_not_search_for_a_recording` below, where `mark_timing_unreliable`
    # is True and this flag is not). This fixture's chord timeline dies a
    # quarter of the way in, which fails `chordMatchRatio` and `timingCoverage`
    # in their own right (not merely a tail collapse), so it is a genuine,
    # broad AUDIO fault and must still search.
    if report.quality.escalation.suggest_alternative_recording:
        assert captured, "an audio fault must look for a better recording"
        assert report.suggestions is not None
        assert report.suggestions.suggestions[0].video_id == "studio12345"
        assert report.suggestions.to_dict()["analyzed"] is False
        stored = store.get(SONG_ID)
        assert "timing-unreliable" in [p.action for p in stored.provenance]
    else:  # pragma: no cover — the fixture is chosen to fail; keep the claim honest
        pytest.fail(
            f"expected an audio-fault verdict from a partial timeline, got "
            f"{report.quality.attribution.describe()}"
        )


def test_a_mode_b_collapse_only_fail_marks_but_does_not_search_for_a_recording(
    monkeypatch, store
):
    """The other half of the split, exercised through Mode B specifically
    (`realign.py` wires `mark_timing_unreliable` and `suggest_alternative_
    recording` from two separate escalation flags -- see the block right
    after the quality gate there). A tail collapse that is the ONLY thing
    wrong with the grade must mark the timing unreliable and make NO
    `suggest_recordings` call: the guard already refused to invent spacing for
    those lines, and no other recording of the song supplies the span it
    looked for -- unlike the broad audio fault above, where a different
    recording genuinely might do better.
    """
    store.save(_stored_song(), message="gold")
    # The new recording's chord timeline matches the stored document for its
    # first 9 chords (comfortably clearing `chordMatchRatio` and
    # `timingCoverage`) and diverges only for the last 3 -- exactly
    # `COLLAPSE_RUN_THRESHOLD`, with nothing later to spread the tail into.
    mismatched_tail = list(PROGRESSION[:9]) + ["Dm", "Dm", "Dm"]
    _wire(monkeypatch, _mir(mismatched_tail, duration=DURATION))

    captured: list[dict] = []

    def spy_suggest(song, **kwargs):
        captured.append({"song": song, **kwargs})
        from snoocle_server.recordings import RecordingSuggestions

        return RecordingSuggestions(song_id=song.id, reason="spy")

    monkeypatch.setattr(realign_mod, "suggest_recordings", spy_suggest)

    report = realign_song(SONG_ID, NEW_VIDEO, allow_timing_loss=True)

    assert report.quality.grade.failing == (report.quality.grade.metric("collapseRuns"),), (
        "the collapse must be the grade's only failing metric for this to be "
        "a genuine collapse-only fail"
    )
    assert report.quality.escalation.mark_timing_unreliable is True
    assert report.quality.escalation.suggest_alternative_recording is False

    assert captured == [], "no other recording of the song supplies the collapsed span"
    assert report.suggestions is None
    assert "recording-suggestions" not in report.steps
    assert report.steps["timing-reliability"] == (
        "marked unreliable (collapsed timing the guard could not spread)"
    )

    stored = store.get(SONG_ID)
    assert "timing-unreliable" in [p.action for p in stored.provenance]


# --- the HTTP surface --------------------------------------------------------


def test_the_endpoint_re_aligns_and_reports_what_it_did(monkeypatch, store):
    from fastapi.testclient import TestClient

    from snoocle_server import api as api_mod
    from snoocle_server.api import app

    monkeypatch.setattr(api_mod, "get_store", lambda: store)
    store.save(_stored_song(), message="gold")
    recording = [transpose_chord(c, 2) for c in PROGRESSION]
    _wire(monkeypatch, _mir(recording, key="F# minor"))

    r = TestClient(app).post(
        f"/v1/songs/{SONG_ID}/realign", json={"videoId": NEW_VIDEO}
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["songId"] == SONG_ID
    assert body["videoId"] == NEW_VIDEO
    assert body["modelConsulted"] is False
    assert body["transposition"]["semitones"] == 2
    assert body["structure"]["explained"] is True
    assert body["storedVersion"]
    assert body["quality"]["grade"]["verdict"]


def test_the_endpoint_maps_each_refusal_to_the_status_that_fits_it(monkeypatch, store):
    from fastapi.testclient import TestClient

    from snoocle_server import api as api_mod
    from snoocle_server.api import app

    monkeypatch.setattr(api_mod, "get_store", lambda: store)
    client = TestClient(app)
    _wire(monkeypatch, _mir(PROGRESSION))

    # An unknown song is the caller naming something that isn't there.
    missing = client.post("/v1/songs/no-such--song/realign", json={"videoId": NEW_VIDEO})
    assert missing.status_code == 404
    assert missing.json()["errorCode"] == "song_not_found"

    # "You want the cheap tool" is a conflict the caller can resolve.
    store.save(_stored_song(), message="gold")
    monkeypatch.setattr(
        realign_mod, "same_recording_check",
        lambda song, video_id: OffsetEstimate(offset_seconds=1.0, confidence=0.95),
    )
    same = client.post(f"/v1/songs/{SONG_ID}/realign", json={"videoId": NEW_VIDEO})
    assert same.status_code == 409
    assert same.json()["errorCode"] == "same_recording_use_video_offset"
    assert "video-offset" in same.json()["detail"]
