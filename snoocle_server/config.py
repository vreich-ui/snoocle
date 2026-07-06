"""Runtime configuration.

Everything is env-driven (SNOOCLE_* / provider API keys) so the service stays
stateless and deployable anywhere; a local .env is honored for development.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SNOOCLE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- storage ---
    data_dir: Path = Path("data")
    # Git-backed artifact store. Its own repository, separate from the code repo.
    store_dir: Path = Path("data/songstore")
    audio_cache_dir: Path = Path("data/audio-cache")

    # --- LLM reconciliation ---
    # Provider is a runtime choice: anthropic | openai | gemini | mock.
    llm_provider: str = "anthropic"
    llm_model: str = ""  # empty -> provider default
    llm_max_tokens: int = 16000
    llm_temperature: float = 0.2
    llm_repair_attempts: int = 2
    # Optional provider-conditional enhancement: attach a short audio snippet to
    # the reconciliation request. Baseline never depends on this (see docs).
    llm_audio_snippet: bool = False

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    anthropic_base_url: str = "https://api.anthropic.com"
    openai_base_url: str = "https://api.openai.com"
    gemini_base_url: str = "https://generativelanguage.googleapis.com"

    # --- text-source discovery ---
    # Comma-separated ordered preference of search backends: brave, serpapi, duckduckgo.
    search_backends: str = "duckduckgo"
    brave_api_key: str = ""
    serpapi_api_key: str = ""
    search_max_candidates: int = 8  # gather generously; reconciliation uses all of them
    fetch_timeout_seconds: float = 20.0

    # --- audio / MIR ---
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    # Optional checkpoints/vendor dirs for the heavy MIR models (ChordMiniApp-style
    # layout). When absent, engines fall back to the librosa implementations.
    chord_cnn_lstm_dir: Path | None = None
    songformer_dir: Path | None = None
    mir_max_analysis_seconds: int = 0  # 0 = analyze full track

    # --- API ---
    host: str = "127.0.0.1"
    port: int = 8765

    def provider_key(self, provider: str) -> str:
        return {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "gemini": self.gemini_api_key,
        }.get(provider, "")


settings = Settings()
