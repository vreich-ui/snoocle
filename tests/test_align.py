"""Forced alignment of known lyrics to audio (B2).

whisperx is not installed here and never will be in CI. What is tested is
everything around it — the tokenizer both sides share, the positional mapping
from a flat word list back onto lines, the vocals-stem preference, and the
refusal to guess when the counts disagree.

The mapping is the part worth this much attention. It is the one place where a
bug produces output that looks entirely reasonable — every line has a plausible
time — while the karaoke highlight sits three words behind for the whole song.
"""

from __future__ import annotations

import wave

import pytest

from snoocle_server.audio import stems as S
from snoocle_server.config import settings
from snoocle_server.timing import align as A

LINES = ["Hello darkness my old friend", "I've come to talk with you again"]


@pytest.fixture()
def audio(tmp_path):
    path = tmp_path / "mix.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(8000)
        fh.writeframes(b"\x00\x00" * 16000)
    return path


def timed_words(lines=LINES, start=0.0, step=0.5, drop=()):
    """A stand-in for whisperx: one evenly spaced word per token."""
    out = []
    clock = start
    for index, line in enumerate(lines):
        for token in A.tokenize(line):
            if (index, token) in drop:
                out.append({"word": token, "start": None, "end": None})
            else:
                out.append({"word": token, "start": clock, "end": clock + step})
            clock += step
    return out


# --- the shared tokenizer ----------------------------------------------------


def test_the_tokenizer_keeps_contractions_whole():
    """"I've" is one sung word. Splitting it there would shift every
    subsequent word by one and silently break the whole mapping."""
    assert A.tokenize("I've come to talk") == ["I've", "come", "to", "talk"]
    assert A.tokenize("don’t stop") == ["don’t", "stop"]


def test_the_tokenizer_drops_punctuation_and_markers():
    assert A.tokenize("Hello, darkness! (x2)") == ["Hello", "darkness", "x2"]
    assert A.tokenize("   ") == []
    assert A.tokenize("--- ***") == []


# --- the mapping -------------------------------------------------------------


def test_words_land_on_the_lines_they_came_from():
    lines, words = A.map_words_to_lines(LINES, timed_words())
    assert [line.lineIndex for line in lines] == [0, 1]
    assert [w.word for w in words if w.lineIndex == 0] == A.tokenize(LINES[0])
    assert [w.word for w in words if w.lineIndex == 1] == A.tokenize(LINES[1])


def test_line_boundaries_come_from_its_first_and_last_word():
    lines, _ = A.map_words_to_lines(LINES, timed_words(step=1.0))
    assert lines[0].start == 0.0
    assert lines[0].end == pytest.approx(5.0)   # five words, one second each
    assert lines[1].start == pytest.approx(5.0)


def test_a_count_mismatch_is_refused_rather_than_guessed():
    """Every assignment after a discrepancy is wrong. A shifted highlight that
    looks fine is worse than a failure that says what happened."""
    short = timed_words()[:-3]
    with pytest.raises(ValueError) as excinfo:
        A.map_words_to_lines(LINES, short)
    assert "refusing" in str(excinfo.value)


def test_an_untimed_word_does_not_untime_its_line():
    """whisperx drops words it cannot place. Taking the line's start from its
    first word regardless would leave the line untimed for no reason."""
    dropped = timed_words(drop=((0, "Hello"),))
    lines, words = A.map_words_to_lines(LINES, dropped)
    assert words[0].start is None
    assert lines[0].start == pytest.approx(0.5)   # the second word
    assert lines[0].coverage == pytest.approx(4 / 5)


def test_empty_lines_keep_their_index_and_get_no_times():
    """Instrumental slots are real lines in the song. Dropping them would make
    every lineIndex after them point at the wrong lyric."""
    with_gap = [LINES[0], "", LINES[1]]
    lines, words = A.map_words_to_lines(with_gap, timed_words())
    assert [line.lineIndex for line in lines] == [0, 1, 2]
    assert lines[1].start is None and lines[1].coverage == 0.0
    assert {w.lineIndex for w in words} == {0, 2}


# --- coverage ----------------------------------------------------------------


def test_coverage_reports_how_much_was_actually_placed():
    lines, words = A.map_words_to_lines(LINES, timed_words(drop=((0, "Hello"), (1, "again"))))
    result = A.Alignment(lines=lines, words=words)
    assert result.coverage == pytest.approx(10 / 12)
    assert result.to_json()["coverage"] == pytest.approx(0.833, abs=0.001)


def test_coverage_of_nothing_is_zero_not_a_crash():
    assert A.Alignment().coverage == 0.0


# --- input selection ---------------------------------------------------------


def test_the_full_mix_is_used_when_there_are_no_stems(audio, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "stems_dir", tmp_path / "stems")
    path, label = A.choose_audio(audio, song_id="artist--song")
    assert label == "mix" and path == audio


def test_the_vocals_stem_is_preferred_when_it_exists(audio, tmp_path, monkeypatch):
    """Alignment quality is dominated by how audible the voice is, so an
    isolated vocal is worth reaching for whenever B4 has produced one."""
    monkeypatch.setattr(settings, "stems_dir", tmp_path / "stems")
    directory = S.stems_dir("artist--song", S.DEFAULT_MODEL)
    directory.mkdir(parents=True)
    (directory / "vocals.wav").write_bytes(audio.read_bytes())

    path, label = A.choose_audio(audio, song_id="artist--song")
    assert label == "vocals"
    assert path.name == "vocals.wav"


def test_which_input_was_used_travels_into_the_result(audio, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "stems_dir", tmp_path / "stems")
    result = A.align_lyrics(audio, LINES, duration=12.0,
                            aligner=lambda *a: timed_words())
    assert result.to_json()["source"] == "mix"


# --- the whole call ----------------------------------------------------------


def test_align_lyrics_returns_lines_and_words(audio, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "stems_dir", tmp_path / "stems")
    result = A.align_lyrics(audio, LINES, duration=12.0, aligner=lambda *a: timed_words())
    body = result.to_json()
    assert len(body["lines"]) == 2
    assert len(body["words"]) == 12
    assert body["engine"] == "whisperx"


def test_the_text_sent_to_the_aligner_is_tokenizer_normalised(audio, monkeypatch, tmp_path):
    """It has to be, or the words coming back won't correspond one-to-one with
    the words we counted."""
    monkeypatch.setattr(settings, "stems_dir", tmp_path / "stems")
    seen = {}

    short = ["Hello, darkness!", "I've come."]

    def capture(path, text, language, duration):
        seen["text"] = text
        return timed_words(lines=short)

    A.align_lyrics(audio, short, duration=5.0, aligner=capture)
    assert seen["text"] == "Hello darkness I've come"


def test_aligning_nothing_is_an_error(audio):
    with pytest.raises(ValueError):
        A.align_lyrics(audio, ["", "   ", "(instrumental)"][:2])


def test_a_missing_audio_file_is_reported_as_such(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "stems_dir", tmp_path / "stems")
    with pytest.raises(FileNotFoundError):
        A.align_lyrics(tmp_path / "nope.wav", LINES, duration=1.0,
                       aligner=lambda *a: [])


# --- graceful absence --------------------------------------------------------


def test_the_module_imports_without_whisperx():
    assert isinstance(A.engine_available(), bool)


def test_the_real_aligner_says_what_is_missing(audio):
    if A.engine_available():
        pytest.skip("whisperx is installed in this environment")
    with pytest.raises(A.AlignmentUnavailable) as excinfo:
        A._default_aligner(audio, "hello", "en", 2.0)
    assert "align" in str(excinfo.value)


# --- the HTTP surface --------------------------------------------------------


@pytest.fixture()
def api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from snoocle_server import api as api_mod
    from snoocle_server.api import app
    from snoocle_server.store.memory import InMemorySongRepository
    from snoocle_server.schema import Song
    from snoocle_server.schema.song import Line, SongMetadata

    monkeypatch.setattr(settings, "stems_dir", tmp_path / "stems")
    # Own the repository outright rather than writing into the process-wide
    # one: other modules swap it per test, and a fixture that depends on
    # whichever store happens to be installed fails only in a full run.
    store = InMemorySongRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: store)
    monkeypatch.setattr("snoocle_server.pipeline.get_store", lambda: store)
    song = Song(
        id="align--song",
        metadata=SongMetadata(title="Song", artist="Artist"),
        lines=[Line(lineIndex=0, lyrics=LINES[0]), Line(lineIndex=1, lyrics=LINES[1])],
    )
    song.audio.analyzedVideoId = "dQw4w9WgXcQ"
    store.save(song, message="fixture: a song with lyrics and no times")
    return TestClient(app)


def test_requesting_alignment_queues_an_align_job(api):
    r = api.post("/v1/songs/align--song/align")
    assert r.status_code == 202
    body = r.json()
    job = body["jobs"][0]
    assert job["kind"] == "align"
    assert job["wants"] == ["align"]
    assert job["targetSongId"] == "align--song"
    # Worth reporting: it is the number that says whether this run has anything
    # to do at all.
    assert body["untimedLines"] == 2


def test_aligning_a_song_with_no_lyrics_is_refused(api):
    from snoocle_server import api as api_mod
    from snoocle_server.schema import Song
    from snoocle_server.schema.song import SongMetadata

    api_mod.get_store().save(
        Song(id="empty--song", metadata=SongMetadata(title="E", artist="A")),
        message="fixture: instrumental",
    )
    r = api.post("/v1/songs/empty--song/align")
    assert r.status_code == 409
    assert "lyrics" in r.json()["detail"]


def test_aligning_an_unknown_song_is_404(api):
    assert api.post("/v1/songs/nope--nope/align").status_code == 404


# --- worker dispatch ---------------------------------------------------------


def test_the_worker_routes_by_kind(monkeypatch):
    """A worker that ran the reconcile pipeline for a stems job would produce a
    whole new version of the song as a side effect of asking for a backing
    track. The dispatch is what stops that."""
    from snoocle_server import worker

    monkeypatch.setattr(worker, "run_stems_job", lambda job: {"ran": "stems"})
    monkeypatch.setattr(worker, "run_align_job", lambda job: {"ran": "align"})

    assert worker.run_pipeline_job({"kind": "stems"})["ran"] == "stems"
    assert worker.run_pipeline_job({"kind": "align"})["ran"] == "align"
