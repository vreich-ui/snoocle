# Snoocle Server

An audio-to-song-data foundry — the backend service behind [Snoocle](../Snoocle), a personal iPad-first song-practice app. Given a song, it produces a fully reconciled song JSON (chords, lyrics, structure, timestamps) by combining web-sourced chord/lyric text with independent MIR analysis of the actual recording, reconciled by a configurable LLM.

Built API-first so the iOS app can call it directly, and structured to be wrapped as MCP tools for agent use.

## Status

Early/active development. Architecture defined; implementation driven by long-running Claude sessions against the brief in this repo's setup docs. Not production-hardened — see **Personal-use notice** below before considering any wider deployment.

## Personal-use notice

This service performs server-side YouTube audio acquisition (via `yt-dlp` or equivalent) for personal music-information-retrieval analysis. This is a deliberate, scoped decision for a single-user, non-distributed personal tool — it is **not** intended for public or commercial deployment as-is, and that decision should be revisited before any exposure beyond personal use.

Chords are not copyrightable; lyrics are. This service discovers chord/lyric text via general web search (not a hardcoded scraper against any single named service) and does not redistribute or centrally host copyrighted lyric content beyond what's needed for personal analysis.

## Pipeline

```
song title + artist  — or just a YouTube URL/ID (title+artist derived from it)
        │
        ▼
  text-source discovery  ──►  multiple candidate chord/lyric sources
        │
        ▼
  audio acquisition + MIR analysis
    (madmom → beat/downbeat, Chord-CNN-LSTM → chords,
     SongFormer → structure/sections)
        │
        ▼
  LLM reconciliation (Claude / GPT / Gemini — pluggable)
        │
        ▼
  schema-compliant song JSON
        │
        ├──► returned to caller (iOS app / MCP tool consumer)
        └──► persisted to Firestore as a new immutable version
```

## Output schema

Output must conform to the `Song` schema used by the Snoocle iOS app — `metadata`, `displayPreferences` (capo/tuning), `audio` (youtubeVideoId, syncMap), `sections`, `lines` (`chordPlacements` keyed by `charIndex`), append-only `provenance`.

**Chord normalization rule (non-negotiable):** every stored chord is the actual sounding harmony, never a fretboard shape. Capo/tuning are display-only transforms, never baked into a stored chord's identity.

## Core technologies

- [madmom](https://github.com/CPJKU/madmom) (CPJKU) — beat/downbeat tracking
- [Chord-CNN-LSTM](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition) — large-vocabulary chord recognition
- SongFormer — structural segmentation
- Reconciliation via Anthropic (Claude), OpenAI (GPT), and Google (Gemini) APIs — provider is a runtime/config choice
- Architecture and model integration informed by [ptnghia-j/ChordMiniApp](https://github.com/ptnghia-j/ChordMiniApp) (MIT)

## Setup (high level)

- Python 3.10
- `ffmpeg` for audio format conversion/cropping
- Docker recommended for `madmom` (native build quirks outside containerized environments)
- Git LFS only for SongFormer checkpoints (optional). Chord-CNN-LSTM needs no
  LFS: `scripts/setup_chord_model.sh` clones its ~28 MB checkpoints, then
  `pip install -e '.[chordmodel]' --extra-index-url https://download.pytorch.org/whl/cpu`
  and `export SNOOCLE_CHORD_CNN_LSTM_DIR=$PWD/models/chord-cnn-lstm`
  (the Docker image already bakes all of this in)
- `.env` for API keys: Anthropic, OpenAI, Google (Gemini), and a YouTube-related key/tooling as needed — never commit this file

## Quickstart

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[mir,dev]' anthropic python-multipart
cp .env.example .env                     # fill in API keys

.venv/bin/snoocle-server                 # HTTP API on 127.0.0.1:8765
.venv/bin/snoocle-mcp                    # MCP tool server (stdio)
.venv/bin/python -m pytest               # test suite
.venv/bin/python scripts/acceptance.py --offline   # acceptance report -> docs/ACCEPTANCE.md
```

`ffmpeg` must be on PATH. See `docs/ARCHITECTURE.md` for the module map and
the overnight-build assumptions, and `docs/ACCEPTANCE.md` for the latest
per-step acceptance results.

## Versioned persistence

Songs are stored in **Firestore (Native mode)** and survive Cloud Run instance
restarts. Every analysis run writes a new *immutable version* rather than
overwriting the previous result:

- `songs/{songId}` — the latest Song plus denormalized `{title, artist,
  latestVersion, updatedAt}` (cheap listing/queries).
- `songs/{songId}/versions/{versionSha}` — an immutable snapshot
  `{song, message, timestamp, parent}`; `versionSha` is a content hash (first
  12 hex of sha256 over the song's canonical JSON).

Saves are optimistically locked (`expectedVersion` → a Firestore transaction →
`409` on a stale write), provenance is append-only, and history/diffs are
served by `GET /v1/songs/{id}/versions` and `…/diff?a=&b=`. All access uses
Application Default Credentials (no key files); the project comes from
`GOOGLE_CLOUD_PROJECT`. Storage sits behind a small repository interface, so an
in-memory backend runs the whole path offline for tests, CI, and local dev
(`SNOOCLE_STORE_BACKEND=memory`, the default when no GCP project is set).

## License

Personal project — license TBD.
