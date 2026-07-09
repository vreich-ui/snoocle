# Snoocle server — architecture

One Python service (FastAPI + an MCP server sharing the same service layer).
State lives only in the git-backed song store and the audio cache; everything
else is stateless and env-configured (`.env.example`). Deployable to Cloud
Run as two services from one image — see `docs/DEPLOY_CLOUD_RUN.md`.

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
├── reconcile/       step 5: providers.py (anthropic/openai/gemini/mock as a
│                    RUNTIME choice; audio-input capability map), prompt.py
│                    (baseline = ALL candidates + MIR timeline as JSON,
│                    identical across providers), engine.py (validate ->
│                    repair loop -> server-side provenance/guardrails),
│                    mock_reconciler.py (deterministic offline reconciler)
├── store/gitstore.py step 7: dedicated git repo; every save is a commit;
│                    expected_version optimistic locking
│                    (saveRecordIfVersionUnchanged), OS write lock,
│                    append-only provenance enforcement
├── pipeline.py      orchestration, tolerant of partial failure
├── api.py           HTTP surface (one endpoint per step + full pipeline)
└── mcp_server.py    MCP surface (16 step-scoped tools; base64 fallback for
                     binary; save-if-version-unchanged exposed). Defaults to
                     stdio (local subprocess use); SNOOCLE_MCP_TRANSPORT=
                     streamable-http serves it as a long-running HTTP
                     process instead (e.g. its own Cloud Run service).
```

## Key decisions & assumptions (made overnight, flag anything wrong)

- **Song schema details** were derived from the brief (iOS repo unreachable
  this session — MCP `list_repos`/`add_repo` permission prompts can't be
  approved unattended). Assumptions: sections use inclusive
  `[startLineIndex, endLineIndex]` ranges + optional MIR `startTime`/`endTime`;
  `syncMap` entries are `{lineIndex, time}` seconds; empty-lyric instrumental
  lines carry ordinal chord slots; ids are `artist--title` slugs.
- **Wolf's repos** (Dr-Lurie-Blog, CMS-Agent, pdf-tool) were likewise
  unreachable, so the primitives the brief named were reimplemented from its
  descriptions: `save(expected_version=...)` CAS + store-level write lock,
  base64 artifact fallback on MCP audio tools, local-first routing.
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
