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
    # Separated stems and the practice mixes derived from them (B4). Sits
    # beside the audio cache and is equally disposable — everything here can be
    # regenerated from the source audio, it is just expensive to. NOTE on Cloud
    # Run this is tmpfs (RAM, wiped on restart), which is why separation is a
    # worker job: the machine that produces stems is the machine that keeps
    # them. See docs/STEMS.md.
    stems_dir: Path = Path("data/stems")

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
    # Provider is a runtime choice: anthropic | anthropic-agent | openai |
    # gemini | agent | mock.
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

    # --- in-process Anthropic agent (provider "anthropic-agent") ---
    # Agentic reconciliation inside this server: Claude + server-side web search
    # + local tools (chord-sheet fetch/parse, windowed MIR). Uses the same
    # SNOOCLE_ANTHROPIC_API_KEY as the plain "anthropic" provider.
    anthropic_agent_model: str = "claude-opus-4-8"
    anthropic_agent_max_turns: int = 12  # hard cap on agent loop iterations
    # Reasoning effort for the agent loop. "medium" keeps tool use consolidated
    # and turns fast — reconciliation is structured evidence-fusion, not
    # open-ended research; raise to "high" only if quality visibly suffers.
    # (Wall-clock is dominated by effort x turns; SNOOCLE_LLM_MODEL=
    # claude-sonnet-5 is the other big speed lever.)
    anthropic_agent_effort: str = "medium"

    # --- song identity resolution (title/artist from a media URL) ---
    # When only a YouTube URL is given, title+artist come from the video's own
    # metadata. Deterministic parsing handles most titles; a genuinely
    # ambiguous one (no separator, cover phrasing, untrustworthy channel) gets
    # ONE cheap Haiku call. Song ids are permanent (content-hash versioned
    # store), so below identity_min_confidence the resolve step FAILS rather
    # than storing a guess — see snoocle_server/identity.py.
    identity_model: str = "claude-haiku-4-5"
    identity_max_tokens: int = 512  # a small JSON object; nothing more
    identity_min_confidence: float = 0.6

    # --- deterministic-step caches ---
    # Both are pure optimizations: same outputs, fewer executions. Disabling
    # either only costs time.
    #
    # MIR has NO TTL on purpose. Its key is a content hash of the audio bytes
    # plus the engine ids plus the accuracy profile — everything that can
    # change the answer is in the key, so an entry cannot go stale. The one
    # change the key can't see is an engine upgraded IN PLACE without its id
    # changing; `refreshCache` on a request (or flipping this flag off) is the
    # escape hatch for that.
    mir_cache_enabled: bool = True
    mir_cache_collection: str = "mir_cache"
    # Discovery DOES need a TTL: its key is (title, artist, backends) and the
    # web behind it changes. Slowly — hence a generous window — but it changes.
    discovery_cache_enabled: bool = True
    discovery_cache_collection: str = "discovery_cache"
    discovery_cache_ttl_days: float = 30.0

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
    # Comma-separated ordered preference of search backends: brave, serpapi,
    # duckduckgo, static.
    #
    # The keyed backends lead. They cost nothing to have in the list when no key
    # is set — each raises immediately on the missing key and web_search() falls
    # through to the next — but the old "duckduckgo"-only default meant setting
    # SNOOCLE_BRAVE_API_KEY did precisely nothing until you ALSO knew to edit
    # this variable. Configuring a key should be sufficient to be used.
    #
    # duckduckgo stays last as the keyless fallback. It is dependable from a
    # residential IP and heavily throttled from datacenter ranges — measured 2
    # successes in 8 attempts, which the retry in _search_duckduckgo lifts to
    # roughly 4 in 6. Usable on Cloud Run, not something to depend on there.
    search_backends: str = "brave,serpapi,duckduckgo"
    brave_api_key: str = ""
    serpapi_api_key: str = ""
    search_max_candidates: int = 8  # gather generously; reconciliation uses all of them
    fetch_timeout_seconds: float = 20.0
    # Ultimate Guitar discovery source (master plan B5). Scrapes UG's
    # undocumented js-store JSON (see discovery/sources/ultimate_guitar.py);
    # personal-use, one flag to kill it instantly if the endpoint changes shape.
    #
    # ON by default since 2026-07. It shipped OFF because it was an extra,
    # opt-in source alongside a general web search that worked; that assumption
    # no longer holds. The keyless general backend is now throttled from
    # datacenter IPs (DuckDuckGo answers Cloud Run with HTTP 202 and an empty
    # shell most of the time), so discovery kept returning nothing and the
    # reconciler had no chord text to work from. UG's endpoint answers those
    # same IPs reliably, which makes it the only dependable keyless source.
    # Set SNOOCLE_SOURCE_UG=0 to switch it off — but configure a keyed search
    # backend if you do, or discovery is back to depending on a coin flip.
    #
    # Tests force this back OFF (tests/conftest.py) — the suite is hermetic and
    # must not reach the live endpoint.
    source_ug_enabled: bool = True
    source_ug_max_candidates: int = 3

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
    # (http://user:pass@proxy.example:8080). Empty = direct. NOTE: YouTube
    # binds stream URLs to the requesting IP, so the media bytes must flow
    # through the proxy too — its bandwidth is then the download ceiling.
    ytdlp_proxy: str = ""
    # --- download speed ---
    # Audio-only preference with a sane bitrate cap; "best" only as the last
    # resort when no audio-only format exists (that may include video and
    # download far more bytes). MIR analyzes at 22.05 kHz mono, so >160 kbps
    # sources buy nothing.
    ytdlp_format: str = "bestaudio[abr<=160]/bestaudio/best"
    # Parallel fragment downloads for HLS/DASH formats — the main lever
    # against YouTube's per-connection throttling on segmented streams.
    ytdlp_concurrent_fragments: int = 4
    # Persist yt-dlp's player/challenge-solver cache (e.g. /data/ytdlp-cache)
    # so repeat acquisitions on the same instance skip re-solving JS
    # challenges. Empty = yt-dlp's default (~/.cache).
    ytdlp_cache_dir: str = ""

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

    # --- deterministic timing enrichment (Phase B) ---
    # LRCLIB (https://lrclib.net) is a free, no-key community synced-lyrics
    # database — the first, most reliable layer for LINE timing (see
    # timing/lrc.py). Best-effort and network-dependent; set to false for a
    # fully offline/hermetic deployment or to skip it deliberately.
    lrclib_enabled: bool = True

    # --- cross-video offset alignment (Phase B) ---
    # POST /v1/songs/{id}/video-offset cross-correlates onset-strength
    # envelopes between a song's already-analyzed reference audio and a
    # DIFFERENT video's audio to find the constant number of seconds to add
    # to every stored time when that OTHER video is playing
    # (AudioInfo.videoOffsets) -- see timing/offset.py.
    # The confidence is a documented heuristic (peak Normalized
    # Cross-Correlation over a bounded lag search), calibrated against
    # synthetic aligned/unrelated fixtures in tests/test_offset.py -- not a
    # statistical guarantee. Below this threshold the endpoint refuses (409)
    # rather than store a guess; raise it for stricter gating, or the caller
    # can always override by supplying offsetSeconds directly.
    offset_min_confidence: float = 0.5
    # How far (seconds, either direction) to search for the best-aligning
    # lag. Offsets between two uploads of the same song are essentially
    # always small; keeping this bounded also avoids spurious far-lag
    # matches (see timing/offset.py's module docstring).
    offset_max_search_seconds: float = 30.0

    # --- quality grading + escalation (snoocle_server/quality/) ---
    # The deterministic grader runs on every reconciled document and records
    # its grade in provenance whatever the outcome. These thresholds decide
    # where "good enough" sits per metric, and the overall below which the
    # verdict is "fail" — the only verdict that can escalate at all.
    #
    # Defaults are read off production evidence, not guessed (see
    # quality/grader.py). Lower a threshold to make the gate more forgiving;
    # `quality_enabled=False` turns grading off entirely (the pipeline then
    # stores exactly what it stored before this existed).
    quality_enabled: bool = True
    quality_chord_match_ratio: float = 0.5
    quality_timing_coverage: float = 0.6
    quality_interpolation_share: float = 0.5  # MAXIMUM
    quality_collapse_runs: int = 0  # MAXIMUM
    quality_section_coverage: float = 0.75
    quality_theory_validity: float = 0.85
    quality_lyric_completeness: float = 0.95
    quality_overall: float = 0.6
    # Fault attribution (quality/attribution.py): where each comparison tips
    # over between MODEL, AUDIO and SOURCE. Attribution decides whether a
    # retry can help at all, so these are the knobs that control spend.
    quality_source_agreement: float = 0.6
    quality_mir_agreement: float = 0.5
    quality_mir_timeline_coverage: float = 0.5
    quality_model_margin: float = 0.15
    # A MODEL-fault retry is ONE extra full-price reconciliation. Set false to
    # keep the grade (still recorded, still acted on for AUDIO/SOURCE) without
    # ever paying for a second attempt.
    quality_retry_enabled: bool = True
    # An AUDIO fault means no re-alignment or re-prompting of THIS recording can
    # help (see quality/attribution.py), so the run searches for an alternative
    # recording and REPORTS it — one cheap yt-dlp query, no download, no
    # analysis. Acting on a suggestion is an explicit operator action (Mode B,
    # see realign.py) because a second track is real spend. Set false to skip
    # the search entirely.
    quality_suggest_recordings: bool = True

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
    # The externally visible base URL (e.g. https://snoocle-x.run.app). Only
    # needed when the forwarded headers can't be trusted to reconstruct it;
    # every OAuth metadata document is built from this, so a wrong value makes
    # discovery advertise URLs the client cannot reach.
    public_url: str = ""

    def provider_key(self, provider: str) -> str:
        """The credential/endpoint whose presence makes a provider usable."""
        return {
            "anthropic": self.anthropic_api_key,
            "anthropic-agent": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "gemini": self.gemini_api_key,
            "agent": self.agent_mcp_url,
        }.get(provider, "")


settings = Settings()
