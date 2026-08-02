# Deterministic Pipeline Completion Prompt Pack

This pack divides the deterministic-first MCP project into reviewable changes.
Run the prompts in order. Each prompt is a separate task, branch, and pull
request. Stop after its acceptance criteria pass; do not begin the next prompt
in the same task.

The first prerequisite is the deterministic core introduced by
`agent/deterministic-pipeline-core`. Later tasks must reuse that service instead
of reimplementing candidate ranking, baseline construction, alignment, conflict
packet construction, or deterministic observability.

## Prompt 1 — Source, candidate, and baseline MCP tools

```text
Work in vreich-ui/snoocle from the latest main after the deterministic core PR
has merged. Review repository instructions and the existing
snoocle_server/deterministic.py service before editing.

Expose only the source/candidate/baseline deterministic operations through MCP:
- parse candidate text supplied by the caller
- score one candidate against MIR across transpositions
- rank candidates
- select a candidate with an explicit strategy
- build a schema-valid baseline Song from one CandidateSource
- validate Song JSON against the existing schema

Requirements:
- Thin wrappers only; reuse existing functions.
- Narrow typed inputs and structured errors.
- Bound JSON size, candidate count, and line count.
- No model calls and no implicit persistence.
- Each response includes elapsedMs, cacheStatus, modelCalls=0, modelCostUSD=0,
  inputSummary, outputSummary, and warnings.
- Preserve exact lyric strings, chord placement order, and charIndex values.
- Baselines contain no speculative timing and capo is zero.

Add focused unit tests at the MCP function boundary. Prove that each wrapper
calls only its intended deterministic service and makes zero model calls. Update
the MCP documentation with these operations only.

Run the focused tests and the existing MCP transport tests. Stop after this
scope passes. Do not add the complete deterministic orchestrator or change the
production pipeline in this task.
```

## Prompt 2 — Audio, MIR, timing, and quality leaf tools

```text
Work from latest main after Prompt 1 has merged. Expose the remaining safe,
operationally useful deterministic leaf services through MCP without changing
their algorithms:
- full-track MIR and windowed MIR
- beat-grid extension
- chord/line snapping
- timing carry-forward
- LRC lookup, matching, and application
- section retiming
- collapse guard
- confidence scoring and review-queue generation
- quality grading, fault attribution, and escalation decision
- theory validation
- recording-offset calculation
- deterministic patch application
- evidence-manifest generation

For every tool specify whether it can use the network, cache, or persistence.
Require explicit persistence inputs and expected-version locking where
applicable. Add payload and list-size bounds. Deterministic tools must always
report zero model usage and cost. Return structured errors rather than raw
tracebacks.

Add focused tests for routing, payload bounds, observability, cache reporting,
and persistence opt-in. Update the internal-function-to-MCP mapping table.

Run focused tests plus MCP transport tests. Stop. Do not add orchestration or
agent policy changes in this task.
```

## Prompt 3 — Direct aligner and complete deterministic orchestrator

```text
Work from latest main after Prompt 2 has merged. Add two orchestration tools by
composing existing deterministic services:

1. align_song_deterministically
   Input: song JSON or stored song/version; MIR JSON, cached MIR, audio path, or
   recording ID; optional LRC; explicit persistence and expected version.
   Order: snap -> optional LRC -> section timing -> collapse guard -> confidence
   -> quality grade and fault attribution.

2. process_song_deterministically
   Order: identity -> acquire/cache -> MIR -> deterministic discovery/parsing ->
   ranking/selection -> baseline -> alignment/LRC/guards/confidence -> quality ->
   optional store.

Neither tool may call reconciliation or any model provider. Insufficient
evidence must return status=needs_review with a stable reason and compact
conflicts. Return per-stage observations and totals, matched/unmatched chords,
line and section timing coverage, interpolation share, collapse interventions,
review queue, quality verdict, fault attribution, cache status, and zero model
cost.

Persist run traces with the stage observations. Keep blocking audio/MIR work off
the async event loop.

Add tests proving identical inputs produce identical aligned Songs, zero model
calls, direct MCP callability, correct stage order, structured early stops, and
explicit optimistic-lock persistence.

Run focused deterministic, timing, quality, store, and MCP transport tests.
Stop. Do not modify the legacy full pipeline or agent policy in this task.
```

## Prompt 4 — Deterministic-first production policy and bounded agent patch

```text
Work from latest main after Prompt 3 has merged. Change the full orchestration
entry points to accept:
agent_policy = never | unresolved_only | always
with unresolved_only as the production default.

Behavior:
- never: run only process_song_deterministically.
- unresolved_only: stop/store a passing deterministic result; SOURCE, AUDIO,
  and UNKNOWN faults never invoke a model; only actionable MODEL conflicts may
  invoke one bounded agent patch.
- always: preserve the historical full-reconciliation experiment.

The agent may receive only the compact conflict packet produced by the
deterministic core. Reject packets containing literal lyrics, full Song JSON,
full beat grids, full provenance, unrelated lines, full schemas, or unnecessary
source URLs. Require a closed patch-operation list, apply it locally, then rerun
deterministic alignment, guards, confidence, grading, and fault attribution.

Do not redesign provider implementations. Preserve targeted correction/scope
behavior unless a test demonstrates an incompatibility.

Add tests proving valid deterministic results use no agent, SOURCE/AUDIO/UNKNOWN
faults use no agent, only MODEL conflicts can invoke an agent, agent input is
compact and lyric-free, patches are local, and always preserves legacy behavior.

Run focused policy tests and the existing API/pipeline reliability suites. Stop.
Do not add mock-production migration or benchmarking in this task.
```

## Prompt 5 — Production mock guard, discovery, and observability audit

```text
Work from latest main after Prompt 4 has merged. Add and validate production
safety and discoverability:

- provider=mock runs only when explicitly requested.
- Mock output is marked testOnly.
- Mock output cannot be stored without allow_test_output=true.
- Production paths fail rather than store placeholder lyric documents.
- Add a read-only diagnostic for existing placeholder/mock songs.
- Add list_capabilities grouped by identity, source retrieval, parsing, audio,
  MIR, baseline, alignment, quality, agent reconciliation, storage, and
  diagnostics.

Every capability entry states deterministic/model-backed, network access,
persistence behavior, input/output types, cache behavior, and cost class.
Verify that list_capabilities covers every registered MCP tool.

Audit all deterministic tool observations and stored run traces for elapsedMs,
cacheStatus, modelCalls, modelCostUSD, compact input/output summaries, and
warnings. Do not silently change legacy tests: tests intentionally storing mock
output must opt in explicitly and assert testOnly metadata.

Run mock-guard, API, pipeline, trace, admission, and MCP suites. Stop after the
canonical tests pass. Do not run the live benchmark in this task.
```

## Prompt 6 — Acceptance benchmark, documentation, and release audit

```text
Work from latest main after Prompt 5 has merged. Do not add new architecture.
Audit the completed deterministic-first implementation against the original
requirements, then close only documented acceptance gaps.

Deliver:
- architecture ordering diagram and migration note
- complete internal-function-to-MCP mapping
- agent policy and mock/test-output documentation
- reproducible Back to Black benchmark using official recording TJAfLE39ZZ8
- benchmark fields: acquisition/cache status, MIR time, baseline time,
  deterministic alignment time, quality score/verdict/fault, model calls, model
  cost, and whether intervention is required
- canonical test-suite result, excluding only clearly documented synchronized
  duplicate files if such duplicates exist locally

Run the benchmark first without agent intervention. If the deterministic result
needs review, report that honestly; do not spend model cost merely to make the
benchmark pass. Verify documentation matches actual tool names and registered
capabilities.

Finish with a concise acceptance matrix: requirement, implementation location,
test evidence, and remaining limitation. Stop after documentation and
verification; do not redesign algorithms or tune the agent prompt.
```
