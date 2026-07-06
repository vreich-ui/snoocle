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
song title + artist (+ optional YouTube ID)
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
        └──► committed to a git-backed, versioned artifact store
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
- Git LFS for model checkpoints
- `.env` for API keys: Anthropic, OpenAI, Google (Gemini), and a YouTube-related key/tooling as needed — never commit this file

## Versioned artifacts

Every analysis run commits its output JSON to a dedicated git-backed store rather than overwriting the previous result — full history, diffing, and rollback via ordinary git tooling. This store is separate from the code in this repo.

## License

Personal project — license TBD.
