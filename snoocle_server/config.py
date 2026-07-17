"""Runtime configuration.

Everything is env-driven (SNOOCLE_* / provider API keys) so the service stays
stateless and deployable anywhere; a local .env is honored for development.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SNOOCLE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- storage ---
    data_dir: Path = Path("data")
    audio_cache_dir: Path = Path("data/audio-cache")

    # Song persistence backend. "auto" (default) picks Firestore when a GCP
    # project or the Firestore emulator is configured, else an in-process
    # in-memory store (fast, hermetic — used by tests and local dev). Force one
    # with SNOOCLE_STORE_BACKEND=firestore|memory.
    store_backend: str = "auto"  # auto | firestore | memory
    # Firestore (Native mode) is the durable store on Cloud Run. Project comes
    # from GOOGLE_CLOUD_PROJECT (Application Default Credentials — no key files).
    google_cloud_project: str = Field(
        default="",
        validation_alias=AliasChoices("GOOGLE_CLOUD_PROJECT", "SNOOCLE_GOOGLE_CLOUD_PROJECT"),
    )
    # Firestore database ID. Defaults to Firestore's "(default)" database; set
    # FIRESTORE_DATABASE to target a NAMED database (e.g. "snoocle-db"). This is
    # the database *within* the project — GOOGLE_CLOUD_PROJECT stays the project
    # ID and is unrelated. (Read from FIRESTORE_DATABASE, not the SNOOCLE_ prefix,
    # so it lines up with the conventional GCP-style variable name.)
    firestore_database: str = Field(
        default="(default)",
        validation_alias=AliasChoices("FIRESTORE_DATABASE", "SNOOCLE_FIRESTORE_DATABASE"),
    )
    firestore_collection: str = "songs"

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

    # --- agent-delegated reconciliation (provider "agent") ---
    # Snoocle holds no LLM keys in this mode: reconciliation is delegated to an
    # external agent workspace (e.g. Claude Agent SDK with specialty agents)
    # reachable as an MCP server over streamable HTTP. Snoocle is the MCP
    # CLIENT: it calls one tool there, passing title/artist, the media URL, and
    # the timestamped MIR chord timeline, and expects Song JSON back.
    agent_mcp_url: str = ""  # e.g. https://my-agent.example.run.app/mcp
    agent_mcp_tool: str = "reconcile_song"
    agent_mcp_auth_token: str = ""  # sent as Authorization: Bearer <token>
    agent_mcp_timeout_seconds: float = 600.0  # agent runs can be slow
    # CMS-Agent-style workspaces expose a node graph instead of one bespoke
    # tool. Set a comma-separated ordered list of node ids to drive that graph
    # via the workspace's generic `node_execute` tool — each node's output is
    # fed forward as dependencyOutputs and the LAST node's output must be the
    # Song JSON. Empty (default) -> call SNOOCLE_AGENT_MCP_TOOL once.
    # e.g. "snoocle_source_search,snoocle_source_compare,snoocle_reconciler"
    agent_mcp_nodes: str = ""

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

    # --- YouTube acquisition (yt-dlp) ---
    # YouTube blocks datacenter IPs (Cloud Run) with a "confirm you're not a
    # bot" challenge. Supply cookies from a signed-in browser to authenticate:
    #   * SNOOCLE_YTDLP_COOKIES       — the cookies.txt CONTENT (e.g. a Secret
    #     Manager value injected as an env var); written to a temp file for yt-dlp.
    #   * SNOOCLE_YTDLP_COOKIES_FILE  — a path to a mounted cookies.txt instead.
    # And/or try alternate player clients (sometimes bypasses the check without
    # cookies): SNOOCLE_YTDLP_PLAYER_CLIENTS="default,android,ios,tv,web_safari".
    ytdlp_cookies: str = ""
    ytdlp_cookies_file: str = ""
    ytdlp_player_clients: str = ""
    # Route ONLY yt-dlp traffic through a proxy so YouTube sees a residential
    # IP instead of Cloud Run's datacenter IP — e.g. a Tailscale exit node at
    # home (socks5://localhost:1055) or a commercial residential proxy
    # (http://user:pass@proxy.example:8080). Empty = direct.
    ytdlp_proxy: str = ""

    # --- audio / MIR ---
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    # Optional checkpoints/vendor dirs for the heavy MIR models (ChordMiniApp-style
    # layout). When absent, engines fall back to the librosa implementations.
    chord_cnn_lstm_dir: Path | None = None
    songformer_dir: Path | None = None
    mir_max_analysis_seconds: int = 0  # 0 = analyze full track (standard accuracy)
    # "fast" accuracy: instead of the whole track, analyze a few short windows
    # spread across the MUSICAL span (after the detected music onset — YouTube
    # videos often open with talking/intros). Enough audio evidence to anchor
    # key/bpm and arbitrate between text sources at a fraction of the CPU time;
    # window timestamps stay in the original track's time coordinates.
    mir_fast_window_seconds: int = 40
    mir_fast_window_count: int = 3

    # --- pipeline reliability ---
    # Per-step wall-clock ceilings (seconds) for POST /v1/songs/analyze so no
    # single external step can hang the request forever. discover/acquire/mir
    # are best-effort (a timeout is recorded and the pipeline continues from
    # whatever it has); reconcile/store are fatal (a timeout -> HTTP 502 naming
    # the step). Cloud Run's own request timeout must be >= the sum that a real
    # run can take (deploy with --timeout=3600; see README/DEPLOY docs).
    discover_timeout_seconds: float = 90.0
    acquire_timeout_seconds: float = 600.0
    mir_timeout_seconds: float = 1500.0
    reconcile_timeout_seconds: float = 900.0
    store_timeout_seconds: float = 60.0

    # --- API ---
    host: str = "127.0.0.1"
    port: int = 8765
    # Optional app-level static bearer token, enforced uniformly on the REST API
    # AND the embedded /mcp transport. When set, every request (except /healthz)
    # must send `Authorization: Bearer <token>`. Leave empty to rely solely on
    # Cloud Run IAM (the default posture). Store it in Secret Manager, not here.
    api_token: str = ""

    def provider_key(self, provider: str) -> str:
        """The credential/endpoint whose presence makes a provider usable."""
        return {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "gemini": self.gemini_api_key,
            "agent": self.agent_mcp_url,
        }.get(provider, "")


settings = Settings()
