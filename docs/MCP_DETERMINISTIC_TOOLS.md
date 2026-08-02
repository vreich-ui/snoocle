# Deterministic source, candidate, and baseline MCP tools

These six MCP tools expose the model-free source-to-baseline portion of
Snoocle's deterministic core. They are local-only: none uses the network,
cache, a model provider, or persistence.

Every response has the same envelope:

```json
{
  "ok": true,
  "result": {},
  "elapsedMs": 0,
  "cacheStatus": "not_applicable",
  "modelCalls": 0,
  "modelCostUSD": 0,
  "inputSummary": {},
  "outputSummary": {},
  "warnings": []
}
```

Invalid input returns `ok=false` and a structured `error` in the same envelope.
Validation details omit the rejected payload itself.

## Operations

| MCP tool | Inputs | Result | Deterministic service |
|---|---|---|---|
| `parse_candidate_text` | `text`, `source_id`, optional `url`, `title`, `retrieved_at` | `candidate` | `discovery.service.candidate_from_text` |
| `score_candidate_against_mir` | `candidate_json`, `mir_json` | source score, best transposition, match counts, conflicts | `reconcile.match.score_candidate` |
| `rank_candidates_deterministically` | `candidates_json`, optional `mir_json` | ordered `ranked` candidates | `deterministic.rank_candidates_deterministically` |
| `select_candidate_deterministically` | `candidates_json`, explicit `strategy` (`best` or `strict`), optional `mir_json` | selection status, ranking, and compact conflicts | `deterministic.select_candidate_deterministically` |
| `build_song_baseline` | `candidate_json`, `song_id`, `title`, `artist`, optional `youtube_video_id` | schema-valid untimed `song` | `deterministic.build_song_from_candidate` |
| `validate_song_json` | `song_json` | `valid=true` and the normalized `song` | `schema.song.Song.model_validate` |

`*_json` inputs are JSON strings. A candidate input is one `CandidateSource`
object; a ranking or selection input is an array of those objects; MIR is one
`MirAnalysis` object; Song input is one Song object.

## Bounds and invariants

- Each JSON or text payload is at most 5,000,000 UTF-8 bytes.
- Candidate arrays contain at most 20 candidates.
- Candidate text, candidates, and validated Songs contain at most 2,000 lines.
- Candidate parsing uses a fixed epoch retrieval timestamp unless
  `retrieved_at` is supplied, so caller-equivalent parses do not depend on wall
  clock time.
- Baselines copy lyric strings and chord placements in their existing order,
  including each `charIndex`. They clear line and placement timing and set
  display capo to zero.
- The tools never persist implicitly. The complete deterministic orchestrator,
  audio/MIR leaf tools, and production pipeline policy are intentionally not
  exposed by this change.
