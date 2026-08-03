# Deterministic MCP tools

Twenty-four MCP tools expose Snoocle's model-free deterministic core: six
source-to-baseline operations and sixteen MIR, timing, quality, patch, and
evidence leaves, plus two orchestrators. The leaf tools are persistence-free
and cache-free. Only `lookup_lrc` uses the network at the leaf layer, and it is
identified as such in every response.

The full `analyze_and_store_song` entry point is deterministic-first by
default. Its `agent_policy` is `never | unresolved_only | always`; only an
actionable MODEL fault under `unresolved_only` can make one lyric-free bounded
patch call. `list_capabilities` inventories every registered MCP tool and
states execution type, network/cache/persistence behavior, input/output type,
and cost class. `diagnose_mock_songs` is the read-only inventory for legacy
mock/placeholder documents.

All registered tools, including these deterministic tools, publish standard MCP
risk annotations plus the versioned Snoocle GUI contract documented in
[`MCP_TOOL_CONTRACT.md`](MCP_TOOL_CONTRACT.md). Registration fails closed if a
tool has not been classified.

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
  "warnings": [],
  "access": {"network": "none", "cache": "none", "persistence": "none"}
}
```

Invalid input returns `ok=false` and a structured `error` in the same envelope.
Validation details omit the rejected payload itself.

## Source-to-baseline operations

| MCP tool | Inputs | Result | Deterministic service |
|---|---|---|---|
| `parse_candidate_text` | `text`, `source_id`, optional `url`, `title`, `retrieved_at` | `candidate` | `discovery.service.candidate_from_text` |
| `score_candidate_against_mir` | `candidate_json`, `mir_json` | source score, best transposition, match counts, conflicts | `reconcile.match.score_candidate` |
| `rank_candidates_deterministically` | `candidates_json`, optional `mir_json` | ordered `ranked` candidates | `deterministic.rank_candidates_deterministically` |
| `select_candidate_deterministically` | `candidates_json`, explicit `strategy` (`best` or `strict`), optional `mir_json` | selection status, ranking, and compact conflicts | `deterministic.select_candidate_deterministically` |
| `build_song_baseline` | `candidate_json`, `song_id`, `title`, `artist`, optional `youtube_video_id` | schema-valid untimed `song` | `deterministic.build_song_from_candidate` |
| `validate_song_json` | `song_json` | `valid=true` and the normalized `song` | `schema.song.Song.model_validate` |

## MIR, timing, quality, and evidence operations

| MCP tool | Inputs | Result | Deterministic service | Network | Cache | Persistence |
|---|---|---|---|---|---|---|
| `analyze_full_track_mir` | local `audio_path`, `accuracy` | full-track `analysis` | `mir.pipeline.analyze_audio` | none | none | none |
| `analyze_mir_window` | local `audio_path`, `start_seconds`, `end_seconds` | windowed `analysis` | `mir.pipeline.analyze_window` | none | none | none |
| `extend_mir_beat_grid` | `beats_json`, duration and explicit continuation options | bounded beat grid | `mir.beats.extend_beat_grid` | none | none | none |
| `snap_song_to_mir` | `song_json`, optional `mir_json` | timed `song` | `timing.snap.snap_chords` | none | none | none |
| `carry_forward_song_timing` | new/prior Song JSON, optional fallback/version label | `song`, carry statistics | `timing.carry_forward.carry_forward_timing` | none | none | none |
| `lookup_lrc` | title, artist, optional duration | LRCLIB match or explicit miss | `timing.lrc.fetch_lrc_match` | LRCLIB | none | none |
| `match_lrc_to_song` | `lrc_json`, `song_json` | monotonic line matches | `timing.lrc.match_lrc_to_lines` | none | none | none |
| `apply_lrc_to_song` | Song, match, optional MIR JSON | LRC-anchored `song` | `timing.lrc.apply_lrc` | none | none | none |
| `retime_song_sections` | Song JSON, optional duration | `song`, changed count | `timing.realign.retime_sections` | none | none | none |
| `guard_song_timing_collapse` | Song JSON, optional duration | guarded `song`, intervention provenance | `timing.collapse_guard.guard_against_collapsed_timing` | none | none | none |
| `score_song_confidence` | Song, candidate, optional MIR JSON, threshold | scored `song`, scores, review queue | `timing.confidence.score_song` + `build_review_queue` | none | none | none |
| `evaluate_song_quality` | Song/candidate/optional MIR JSON and spent budgets | grade, fault attribution, escalation | `quality.gate.evaluate` | none | none | none |
| `validate_song_theory` | Song JSON, optional key override | theory report | `quality.theory.theory_validity` | none | none | none |
| `calculate_recording_offset` | two local audio paths, maximum offset | offset and confidence | `timing.offset.estimate_offset` | none | none | none |
| `apply_deterministic_song_patch` | Song JSON and closed patch JSON | patched `song`, applied operations | `reconcile.patch_ops.parse_ops_response` + `apply_patch` | none | none | none |
| `build_song_evidence_manifest` | optional candidate/MIR/prior Song JSON and request metadata | evidence manifest | `manifest.build_evidence_manifest` | none | none | none |

The audio-analysis and offset tools read only caller-named server files; they
never acquire recordings implicitly. No leaf accepts a persistence flag,
because none writes state. Consequently expected-version locking is not
applicable at this layer. Persistence and optimistic locking belong to the
explicit orchestrator/store boundary.

## Deterministic orchestrators

| MCP tool | Composition | Inputs and persistence |
|---|---|---|
| `align_song_deterministically` | snap → optional LRC → section timing → collapse guard → confidence/review queue → quality/fault attribution | Caller Song JSON or stored song/version; caller/cached MIR JSON, local audio, or recording ID; optional candidates/LRC. Song writes require `persist=true` and explicit `expected_version` (`""` means create-only). |
| `process_song_deterministically` | identity → acquire/cache → MIR/cache → discovery/cache → ranking/selection → baseline → full deterministic alignment → optional store | Title/artist plus caller audio/recording/MIR/candidates/LRC. Song writes use the same explicit optimistic-lock contract. |

Both tools return `status=needs_review` with a stable `reason` when evidence
cannot safely produce a complete answer. Their results include ordered stage
observations, cache status by subsystem, total elapsed/model usage, compact
candidate or placement conflicts, matched/unmatched chord totals, line and
section timing coverage, interpolation share, collapse interventions, review
queue, quality verdict, and fault attribution. Blocking acquisition,
discovery, and MIR work runs outside the async MCP event loop.

Every orchestration call persists a bounded run trace containing those stage
observations. This operational trace is separate from opt-in Song persistence;
the response advertises `persistence=run_trace_and_optional_song`. Neither
orchestrator imports or invokes reconciliation or any model provider, and all
stage and aggregate usage fields remain zero.

`*_json` inputs are JSON strings. A candidate input is one `CandidateSource`
object; a ranking or selection input is an array of those objects; MIR is one
`MirAnalysis` object; Song input is one Song object.

## Bounds and invariants

- Each JSON or text payload is at most 5,000,000 UTF-8 bytes.
- Candidate arrays contain at most 20 candidates.
- Candidate text, candidates, and validated Songs contain at most 2,000 lines.
- MIR and raw beat-grid inputs contain at most 10,000 beats.
- LRC inputs contain at most 5,000 lines and applied match lists at most 2,000 entries.
- Deterministic patch inputs contain at most 20 operations.
- Candidate parsing uses a fixed epoch retrieval timestamp unless
  `retrieved_at` is supplied, so caller-equivalent parses do not depend on wall
  clock time.
- Baselines copy lyric strings and chord placements in their existing order,
  including each `charIndex`. They clear line and placement timing and set
  display capo to zero.
- Leaf tools never persist implicitly. The two deterministic orchestrators
  always write bounded run traces and write Songs only with explicit
  `persist=true` plus expected-version locking. Production policy lives at the
  full orchestration entry points.
