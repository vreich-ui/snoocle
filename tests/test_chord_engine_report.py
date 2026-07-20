"""Health reporting tells the truth about the chord engine.

`chord_engine_id()` must reflect whether the runner file actually exists, not
merely whether SNOOCLE_CHORD_CNN_LSTM_DIR is set — a configured-but-empty mount
silently falls back to the chroma engine, and /healthz must say so.
"""

from __future__ import annotations

from snoocle_server.config import settings
from snoocle_server.mir.chordrec import chord_engine_id, chord_model_status


def test_reports_fallback_when_dir_unset(monkeypatch):
    monkeypatch.setattr(settings, "chord_cnn_lstm_dir", None)
    assert chord_engine_id() == "chroma-template-fallback"
    assert chord_model_status()["dirConfigured"] is False
    assert chord_model_status()["runnerPresent"] is False


def test_reports_fallback_when_runner_missing(monkeypatch, tmp_path):
    # dir configured but no snoocle_runner.py inside -> honest fallback
    monkeypatch.setattr(settings, "chord_cnn_lstm_dir", tmp_path)
    assert chord_engine_id() == "chroma-template-fallback"
    st = chord_model_status()
    assert st["dirConfigured"] is True
    assert st["runnerPresent"] is False


def test_reports_cnn_lstm_when_runner_present(monkeypatch, tmp_path):
    (tmp_path / "snoocle_runner.py").write_text("# stub runner\n")
    monkeypatch.setattr(settings, "chord_cnn_lstm_dir", tmp_path)
    assert chord_engine_id() == "chord-cnn-lstm"
    assert chord_model_status()["runnerPresent"] is True


def test_healthz_reports_chord_model(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from snoocle_server.api import app

    monkeypatch.setattr(settings, "chord_cnn_lstm_dir", tmp_path)  # no runner -> fallback
    body = TestClient(app).get("/healthz").json()
    assert body["mirEngines"]["chords"] == "chroma-template-fallback"
    assert body["chordModel"]["runnerPresent"] is False
