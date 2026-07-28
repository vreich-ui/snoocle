"""Stem separation and the practice mixes derived from it (B4).

demucs is never installed here. That is the point: this suite pins the contract
*around* the separator — caching, completeness, mixing, path safety, honest
unavailability — so all of it stays correct on machines that could never run
the real thing, which includes CI and the Cloud Run image.

The one thing not tested is whether demucs separates music well. That is
upstream's problem and no assertion of ours could check it anyway.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from snoocle_server.audio import stems as S
from snoocle_server.config import settings


@pytest.fixture()
def stems_root(tmp_path, monkeypatch):
    root = tmp_path / "stems"
    monkeypatch.setattr(settings, "stems_dir", root)
    return root


@pytest.fixture()
def source_audio(tmp_path):
    """A real (silent) WAV, so ffmpeg has something valid to chew on."""
    path = tmp_path / "song.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(8000)
        fh.writeframes(b"\x00\x00" * 8000)
    return path


def fake_separator(names=S.STEM_NAMES):
    """Stands in for demucs: writes a valid WAV per source."""

    def run(audio_path: Path, model: str, out_dir: Path) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        written = {}
        for name in names:
            destination = out_dir / f"{name}.wav"
            with wave.open(str(destination), "wb") as fh:
                fh.setnchannels(1)
                fh.setsampwidth(2)
                fh.setframerate(8000)
                fh.writeframes(b"\x00\x00" * 8000)
            written[name] = destination
        return written

    return run


# --- availability is proven, not configured ---------------------------------


def test_the_module_imports_without_demucs():
    """The whole file must be importable on a machine with no ML stack —
    otherwise the API process cannot even serve cached stems."""
    assert S.DEFAULT_MODEL == "htdemucs"
    assert isinstance(S.engine_available(), bool)


def test_separating_without_the_engine_says_so_clearly(stems_root, source_audio):
    """A missing optional dependency must produce a named, actionable error —
    the same posture as the MIR fallbacks — not an ImportError from deep
    inside a library."""
    if S.engine_available():
        pytest.skip("demucs is installed in this environment")
    with pytest.raises(S.StemsUnavailable) as excinfo:
        S.separate(source_audio, "artist--song")
    assert "stems" in str(excinfo.value)


# --- separation and caching --------------------------------------------------


def test_separate_writes_every_stem_and_both_mixes(stems_root, source_audio):
    result = S.separate(source_audio, "artist--song", separator=fake_separator())
    assert result.complete
    assert set(result.stems) == set(S.STEM_NAMES)
    assert set(result.mixes) == set(S.DERIVED_MIXES)
    for path in list(result.stems.values()) + list(result.mixes.values()):
        assert path.exists() and path.stat().st_size > 0


def test_a_second_call_reuses_the_cache(stems_root, source_audio):
    """Separation is minutes of CPU and the inputs fully determine the output,
    so running it twice for the same song+model is pure waste."""
    S.separate(source_audio, "artist--song", separator=fake_separator())

    calls = []

    def counting(audio_path, model, out_dir):
        calls.append(model)
        return fake_separator()(audio_path, model, out_dir)

    S.separate(source_audio, "artist--song", separator=counting)
    assert calls == [], "cached stems should not be regenerated"


def test_force_bypasses_the_cache(stems_root, source_audio):
    S.separate(source_audio, "artist--song", separator=fake_separator())
    calls = []

    def counting(audio_path, model, out_dir):
        calls.append(model)
        return fake_separator()(audio_path, model, out_dir)

    S.separate(source_audio, "artist--song", force=True, separator=counting)
    assert calls == ["htdemucs"]


def test_a_partial_separation_is_refused_not_served(stems_root, source_audio):
    """Half a separation would render a 'backing track' silently missing its
    drums. Failing loudly is the only honest option."""
    with pytest.raises(S.StemsUnavailable) as excinfo:
        S.separate(source_audio, "artist--song",
                   separator=fake_separator(names=("vocals", "drums")))
    assert "incomplete" in str(excinfo.value)
    assert "bass" in str(excinfo.value)


def test_models_are_validated(stems_root, source_audio):
    with pytest.raises(ValueError):
        S.separate(source_audio, "artist--song", model="not-a-model",
                   separator=fake_separator())


# --- the derived mixes -------------------------------------------------------


def test_no_vocals_mix_sums_everything_but_the_voice(stems_root, source_audio):
    S.separate(source_audio, "artist--song", separator=fake_separator())
    parts, _ = S.DERIVED_MIXES["backing_no_vocals"]
    assert "vocals" not in parts
    assert set(parts) == {"drums", "bass", "other"}


def test_the_no_guitar_approximation_is_documented_not_hidden():
    """With the 4-stem model this also removes keys and strings, because they
    share the `other` stem. A user who notices should be able to find out why
    from the API response, not from reading our source."""
    _parts, description = S.DERIVED_MIXES["backing_no_guitar"]
    assert "other" in description.lower()
    assert "keys" in description.lower() or "strings" in description.lower()


def test_the_six_stem_model_makes_no_guitar_exact():
    """htdemucs_6s emits a real guitar stem, so the approximation goes away and
    the piano survives — which is the entire reason to pay for the slower
    model."""
    parts, description = S.DERIVED_MIXES_6S["backing_no_guitar"]
    assert "guitar" not in parts
    assert "piano" in parts
    assert "approximation" not in description.lower()


def test_mixing_a_single_source_still_produces_a_file(stems_root, tmp_path, source_audio):
    dst = tmp_path / "one.wav"
    assert S.render_mix([source_audio], dst).exists()


def test_mixing_nothing_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        S.render_mix([], tmp_path / "empty.wav")


# --- reading the cache needs nothing -----------------------------------------


def test_available_reads_the_cache_with_no_engine(stems_root, source_audio):
    """The API process may have neither demucs nor torch. It must still be able
    to report and serve what a worker produced."""
    S.separate(source_audio, "artist--song", separator=fake_separator())
    found = S.available("artist--song")
    assert found is not None and found.complete
    assert found.to_json()["model"] == "htdemucs"
    assert {m["name"] for m in found.to_json()["mixes"]} == set(S.DERIVED_MIXES)


def test_available_is_none_for_an_unseparated_song(stems_root):
    assert S.available("never--touched") is None


def test_stem_path_resolves_stems_and_mixes(stems_root, source_audio):
    S.separate(source_audio, "artist--song", separator=fake_separator())
    assert S.stem_path("artist--song", "vocals") is not None
    assert S.stem_path("artist--song", "backing_no_vocals") is not None
    assert S.stem_path("artist--song", "nonexistent") is None


def test_stem_path_refuses_names_that_are_not_stems(stems_root, source_audio):
    """The name is matched against a known set rather than joined onto a path,
    so it cannot be used to walk out of the directory."""
    S.separate(source_audio, "artist--song", separator=fake_separator())
    for evil in ("../../etc/passwd", "..", "vocals/../../secret", ""):
        assert S.stem_path("artist--song", evil) is None


def test_a_song_id_cannot_escape_the_stems_root(stems_root):
    with pytest.raises(ValueError):
        S.stems_dir("../../etc", "htdemucs")


# --- cleanup -----------------------------------------------------------------


def test_clear_removes_a_songs_stems(stems_root, source_audio):
    S.separate(source_audio, "artist--song", separator=fake_separator())
    assert S.clear("artist--song") == 1
    assert S.available("artist--song") is None


def test_clearing_a_song_that_was_never_separated_is_zero(stems_root):
    assert S.clear("never--touched") == 0


# --- the HTTP surface --------------------------------------------------------
#
# These matter because the API process is the one machine guaranteed NOT to
# have demucs: it serves what a worker made. If serving needed the engine, the
# whole worker split would be pointless.


@pytest.fixture()
def api(stems_root, monkeypatch):
    from fastapi.testclient import TestClient

    from snoocle_server import api as api_mod
    from snoocle_server.api import app
    from snoocle_server.store.memory import InMemorySongRepository
    from snoocle_server.schema import Song
    from snoocle_server.schema.song import SongMetadata

    # Own the repository outright rather than writing into the process-wide
    # one: other modules swap it per test, and a fixture that depends on
    # whichever store happens to be installed fails only in a full run.
    store = InMemorySongRepository()
    monkeypatch.setattr(api_mod, "get_store", lambda: store)
    monkeypatch.setattr("snoocle_server.pipeline.get_store", lambda: store)
    song = Song(id="artist--song", metadata=SongMetadata(title="Song", artist="Artist"))
    song.audio.analyzedVideoId = "dQw4w9WgXcQ"
    store.save(song, message="fixture: a song to attach stems to")
    return TestClient(app)


def test_listing_stems_for_an_unseparated_song_is_empty_not_404(api):
    """A player asks this on every song page. An absent set is a normal answer,
    not an error worth logging."""
    body = api.get("/v1/songs/artist--song/stems").json()
    assert body["complete"] is False
    assert body["stems"] == [] and body["mixes"] == []


def test_listing_and_streaming_what_a_worker_produced(api, stems_root, source_audio):
    S.separate(source_audio, "artist--song", separator=fake_separator())

    body = api.get("/v1/songs/artist--song/stems").json()
    assert body["complete"] is True
    assert {s["name"] for s in body["stems"]} == set(S.STEM_NAMES)

    audio = api.get("/v1/songs/artist--song/stems/backing_no_vocals")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert len(audio.content) > 0


def test_streaming_supports_range_requests(api, stems_root, source_audio):
    """A backing track is tens of megabytes; seeking must not refetch it."""
    S.separate(source_audio, "artist--song", separator=fake_separator())
    r = api.get("/v1/songs/artist--song/stems/vocals", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206
    assert r.headers["content-range"].startswith("bytes 0-99/")
    assert len(r.content) == 100


def test_streaming_an_unknown_stem_is_404(api, stems_root, source_audio):
    S.separate(source_audio, "artist--song", separator=fake_separator())
    assert api.get("/v1/songs/artist--song/stems/tuba").status_code == 404


def test_requesting_separation_queues_a_stems_job(api):
    r = api.post("/v1/songs/artist--song/stems")
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"
    job = body["jobs"][0]
    # The job must be routable: kind tells the worker what to run, wants tells
    # the broker which workers may claim it, targetSongId says where it lands.
    assert job["kind"] == "stems"
    assert job["wants"] == ["stems"]
    assert job["targetSongId"] == "artist--song"


def test_requesting_separation_twice_returns_the_cache(api, stems_root, source_audio):
    S.separate(source_audio, "artist--song", separator=fake_separator())
    body = api.post("/v1/songs/artist--song/stems").json()
    assert body["status"] == "cached"
    assert body["complete"] is True


def test_an_unknown_model_is_refused_before_queueing(api):
    assert api.post("/v1/songs/artist--song/stems?model=htdemucs_9000").status_code == 400


def test_separating_an_unknown_song_is_404(api):
    assert api.post("/v1/songs/nope--nope/stems").status_code == 404


def test_deleting_stems_reclaims_the_disk(api, stems_root, source_audio):
    S.separate(source_audio, "artist--song", separator=fake_separator())
    assert api.delete("/v1/songs/artist--song/stems").json()["removed"] == 1
    assert api.get("/v1/songs/artist--song/stems").json()["complete"] is False
