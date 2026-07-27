"""Deterministic timing engines: MIR-grounded chord/line timestamps, beat
grid quantization, and (later phases) LRCLIB/forced-alignment/video-offset.

Everything in this package is pure, deterministic, and dependency-injected
(no network, no LLM) -- the reconciliation agent never invents a timestamp
these modules can compute directly from the audio.
"""
