from snoocle_server.discovery.models import CandidateSource
from snoocle_server.mir.base import ChordSegment, MirAnalysis
from snoocle_server.schema.song import Song
from snoocle_server.timing.confidence import build_review_queue, score_song

from test_schema import make_song


def _line(song: Song, idx: int):
    return next(l for l in song.lines if l.lineIndex == idx)


def _candidate(sourceId: str, lyrics: str, chord: str, char_index: int = 7) -> CandidateSource:
    return CandidateSource(
        sourceId=sourceId,
        lines=[{"lineIndex": 0, "lyrics": lyrics, "chordPlacements": [{"charIndex": char_index, "chord": chord}]}],
    )


def test_agreeing_sources_and_mir_score_higher_than_disagreeing():
    data = make_song()
    # Give the first chord (C at charIndex 7) a MIR-agreed time.
    data["lines"][0]["chordPlacements"][0]["timeSeconds"] = 0.0
    song_agree = Song.model_validate(data)

    candidates_agree = [
        _candidate("web-1", "When I find myself in times of trouble", "C"),
        _candidate("web-2", "When I find myself in times of trouble", "C"),
    ]
    mir = MirAnalysis(chords=[ChordSegment(start=0.0, end=3.0, chord="C")])

    scored_agree, scores_agree = score_song(song_agree, candidates_agree, mir)
    c_score_agree = next(s for s in scores_agree if s.chord == "C")
    assert c_score_agree.confidence == 1.0
    assert c_score_agree.reasons == []
    assert _line(scored_agree, 0).chordPlacements[0].confidence == 1.0

    # Same setup but sources + MIR both disagree on the root.
    candidates_disagree = [
        _candidate("web-1", "When I find myself in times of trouble", "F"),
        _candidate("web-2", "When I find myself in times of trouble", "F"),
    ]
    mir_disagree = MirAnalysis(chords=[ChordSegment(start=0.0, end=3.0, chord="F")])
    scored_disagree, scores_disagree = score_song(song_agree, candidates_disagree, mir_disagree)
    c_score_disagree = next(s for s in scores_disagree if s.chord == "C")
    assert c_score_disagree.confidence == 0.0
    assert len(c_score_disagree.reasons) == 2

    assert c_score_agree.confidence > c_score_disagree.confidence


def test_confidence_unchanged_when_only_one_signal_available():
    data = make_song()
    data["lines"][0]["chordPlacements"][0]["timeSeconds"] = 0.0
    data["lines"][0]["chordPlacements"][0]["confidence"] = 0.7
    song = Song.model_validate(data)

    # No candidates at all -- source signal unavailable, MIR signal present.
    mir = MirAnalysis(chords=[ChordSegment(start=0.0, end=3.0, chord="C")])
    scored, scores = score_song(song, [], mir)
    c_score = next(s for s in scores if s.chord == "C")
    # Only one signal (MIR) -- original snap-time confidence must be kept,
    # not silently halved by a naive 0.5*mir computation.
    assert c_score.confidence == 0.7
    assert "no independent signal" not in " ".join(c_score.reasons)


def test_no_signals_at_all_keeps_confidence_and_notes_it():
    data = make_song()
    song = Song.model_validate(data)  # no timeSeconds, no candidates, no mir
    scored, scores = score_song(song, [], None)
    c_score = next(s for s in scores if s.chord == "C")
    assert c_score.confidence == 0.5  # placement had no prior confidence -> neutral default
    assert any("no independent signal" in r for r in c_score.reasons)


def test_review_queue_sorted_ascending_and_thresholded():
    data = make_song()
    song = Song.model_validate(data)
    # Force one placement to a known low confidence, others stay default (0.5).
    scored, scores = score_song(song, [], None)
    for s in scores:
        if s.chord == "G":
            s.confidence = 0.1
        elif s.chord == "Am":
            s.confidence = 0.9  # above threshold -- should NOT appear

    queue = build_review_queue(scores, threshold=0.6)
    chords_in_queue = [e["chord"] for e in queue]
    assert "Am" not in chords_in_queue
    assert chords_in_queue.index("G") == 0  # lowest confidence sorts first
    assert all(queue[i]["confidence"] <= queue[i + 1]["confidence"] for i in range(len(queue) - 1))
    assert all("reasons" in e and "lineIndex" in e and "charIndex" in e for e in queue)


def test_fuzzy_line_matching_tolerates_spelling_differences():
    data = make_song()
    data["lines"][0]["chordPlacements"][0]["timeSeconds"] = 0.0
    song = Song.model_validate(data)
    # Candidate's lyric line has minor spelling/punctuation differences.
    candidates = [_candidate("web-1", "When i find myself, in times of trouble!", "C")]
    scored, scores = score_song(song, candidates, None)
    c_score = next(s for s in scores if s.chord == "C")
    assert "0/1" not in " ".join(c_score.reasons)  # it DID find and match the fuzzy line
