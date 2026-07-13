# Snoocle server — architecture

One Python service (FastAPI + an MCP server sharing the same service layer).
Durable state lives in **Firestore** (the song store); the audio cache is
disposable local disk. Everything else is stateless and env-configured
(`.env.example`). Deployed to Cloud Run as a **single** service: the MCP
streamable-HTTP transport is embedded into the FastAPI app at `/mcp` (one ASGI
app, one lifespan). Firestore transactions provide the write serialization
(optimistic locking), so correctness no longer depends on `--concurrency=1`.
See `docs/DEPLOY_CLOUD_RUN.md`.

```
snoocle_server/
├── config.py        env-driven Settings (SNOOCLE_*)
├── chords.py        chord parsing/normalization/transposition — the sounding-
│                    harmony rule lives here (shapes/tab rejected, capo helper)
├── schema/song.py   the Song schema (camelCase, iOS SongStore-compatible),
│                    invariants enforced by validators, JSON Schema export
├── discovery/       step 2-3: pluggable web search (brave/serpapi/duckduckgo/
│                    static) -> site-agnostic sheet extraction -> generic
│                    chord-sheet parser -> CandidateSource (kept separate,
│                    capo transposed away at ingestion, per-source confidence)
├── audio/
│   ├── utils.py     deterministic ffmpeg ops: convert/trim/normalize/probe
│   │                (never AI — pdf-tool local-first routing)
│   └── acquire.py   step 4a: yt-dlp search+download, video-id cache
│                    (personal-use only, see README)
├── mir/             step 4b: beats (madmom ACTIVE / librosa fallback),
│                    chords (Chord-CNN-LSTM runner adapter / chroma-template
│                    fallback), structure (SongFormer runner adapter /
│                    agglomerative fallback), key vote; engines recorded in
│                    provenance. Heavy-model contract: point
│                    SNOOCLE_CHORD_CNN_LSTM_DIR / SNOOCLE_SONGFORMER_DIR at a
│                    checkout containing snoocle_runner.py (see module docs).
├── reconcile/       step 5: providers.py (anthropic/openai/gemini/agent/mock
│                    as a RUNTIME choice; audio-input capability map),
│                    prompt.py (baseline = ALL candidates + MIR timeline as
│                    JSON, identical across providers), engine.py (validate ->
│                    repair loop -> server-side provenance/guardrails),
│                    mock_reconciler.py (deterministic offline reconciler)
├── store/           step 6-7: SongRepository interface (base.py) with
│                    Firestore (firestore_store.py, durable) and in-memory
│                    (memory.py, hermetic) backends; content-hash versions,
│                    expected_version optimistic locking via a Firestore
│                    transaction, append-only provenance, JSON diffs
├── pipeline.py      orchestration: per-step timeouts, best-effort
│                    discover/acquire/mir + fatal reconcile/store (502 names
│                    the failed step), truthful per-step status report
├── api.py           HTTP surface (one endpoint per step + full pipeline);
│                    ALSO embeds the MCP transport at /mcp (single-service
│                    topology) — imports the FastMCP instance, runs its
│                    session manager in the app lifespan, registers the route
└── mcp_server.py    MCP surface (16 step-scoped tools; base64 fallback for
                     binary; save-if-version-unchanged exposed). Defaults to
                     stdio (local subprocess use); SNOOCLE_MCP_TRANSPORT=
                     streamable-http serves it as a standalone HTTP process.
                     resolve_http_transport() is the shared, unit-tested
                     bind-host + DNS-rebinding-security resolver used by both
                     the standalone server and the embedded /mcp route.
```

## Reconciliation providers

Provider is a runtime choice (`provider` request param or
`SNOOCLE_LLM_PROVIDER`): `anthropic | openai | gemini | agent | mock`.

- `anthropic`/`openai`/`gemini` call the LLM APIs directly with Snoocle-held
  keys.
- **`agent` inverts the direction: Snoocle becomes an MCP *client*.** It calls
  one tool (`SNOOCLE_AGENT_MCP_TOOL`, default `reconcile_song`) on an external
  agent workspace's MCP server (`SNOOCLE_AGENT_MCP_URL`, e.g. a Claude Agent
  SDK environment running specialty agents), passing `{title, artist,
  mediaUrl, chords (MIR-timestamped), mir, candidates, songSchema}` and
  expecting Song JSON back. Snoocle holds no LLM keys in this mode; schema
  validation, the repair loop (`previousOutput`/`validationErrors` are resent),
  and server-side finalization apply to the agent's output exactly as to a
  direct LLM response.
- `mock` is the deterministic offline reconciler used by tests.

## Chord recognition engine

`scripts/setup_chord_model.sh` vendors the real Chord-CNN-LSTM (ISMIR2019)
checkout — pretrained 5-fold checkpoints included in the upstream repo — and
`scripts/snoocle_runner.py` adapts it to the external-runner contract
(`<in.wav> <out.lab>`), shimming the removed numpy aliases and CPU checkpoint
loading so the research code runs unmodified. The Dockerfile bakes it into the
runtime image (CPU torch; `SNOOCLE_CHORD_CNN_LSTM_DIR` preset). Without it,
chordrec falls back to beat-synchronous chroma templates.

## Key decisions & assumptions (made overnight, flag anything wrong)

- **Song schema details** were derived from the brief (iOS repo unreachable
  this session — MCP `list_repos`/`add_repo` permission prompts can't be
  approved unattended). Assumptions: sections use inclusive
  `[startLineIndex, endLineIndex]` ranges + optional MIR `startTime`/`endTime`;
  `syncMap` entries are `{lineIndex, time}` seconds; empty-lyric instrumental
  lines carry ordinal chord slots; ids are `artist--title` slugs.
- **Wolf's repos** (Dr-Lurie-Blog, CMS-Agent, pdf-tool) were likewise
  unreachable, so the primitives the brief named were reimplemented from its
  descriptions: `save(expected_version=...)` CAS (now atomic via a Firestore
  transaction; an in-memory lock for the offline backend), base64 artifact
  fallback on MCP audio tools, local-first routing.
- **Chord rule enforcement is layered:** parser transposes declared capo at
  ingestion → reconciliation prompt states the rule → schema validator
  rejects shapes/tab/N.C. → repair loop feeds violations back to the LLM →
  final spot check in acceptance.
- **Provider parity:** the reconciliation engine hands byte-identical
  (system, turns) to whichever provider is selected; audio snippets are an
  opt-in enhancement only for providers with confirmed audio input
  (openai, gemini — not anthropic, per the brief's capability note).
- **Anthropic default model** is `claude-opus-4-8` (current API docs);
  sampling params intentionally not sent (rejected on Opus 4.7+).
- **Heavy MIR models:** madmom installed from git master and is the live
  beat engine. Chord-CNN-LSTM and SongFormer need multi-GB checkpoints
  (git-lfs) + torch; integrated via a documented external-runner contract
  with honest librosa fallbacks, so the pipeline is always audio-grounded.

## Running

```sh
python3 -m venv .venv && .venv/bin/pip install -e .[mir,dev] anthropic python-multipart
.venv/bin/snoocle-server          # HTTP API on 127.0.0.1:8765
.venv/bin/snoocle-mcp             # MCP server on stdio
.venv/bin/python -m pytest        # 92 tests
.venv/bin/python scripts/acceptance.py --offline   # acceptance report
```
