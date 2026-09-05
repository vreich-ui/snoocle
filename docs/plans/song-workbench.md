# ADR: Snoocle Studio → Song Workbench

Status: accepted (T0). Supersedes the tool-centric studio layout described in
`docs/plans/studio.md` for song work; `agent-and-gui.md` is unaffected.

Scope: `studio/` (React 19 + Vite) plus small additions in `snoocle_server/`.
Repo `vreich-ui/snoocle`; live at `snoocle-99287560712.europe-west1.run.app/studio`.
Land each task with `/ship snoocle <branch>`; `main` is protected; CI gates
`python` and `studio`.

Every file and line reference below was verified against the tree at
`2be3472` on 2026-09-05.

## 1. Why the current studio fails

| Problem | Verified location | Effect |
|---|---|---|
| Tool-centric, not object-centric | `studio/src/ToolStudio.tsx` — a live MCP catalog of **58 tools** (`snoocle_server/tool_contract.py:164`, `TOOL_CONTRACTS`), each a schema form | Work starts from a verb, not from the song |
| 9 top-level sections split one job | `studio/src/navigation.ts:1-11`; only 4 are implemented (`implementedSections`, line 62) | Song context is lost on every tab switch |
| Runs write nothing back, or write blind | `ToolStudio` renders raw JSON; `SongStudio.tsx:211-219` POSTs `/v1/songs/{id}` with `expectedVersion` but shows no diff first | No preview of what a tool changed before it becomes a version |
| Versions are a flat list | `SongVersion` is `{version, timestamp, message}` only (`store/base.py:93-96`), while `parent` **is** persisted (`store/memory.py:45,179`; `store/firestore_store.py:254`) and surfaced nowhere | No lineage, no branching, no way to say which version the app serves |
| No compare surface | `Store.diff` returns unified text only (`store/base.py:332`), exposed at `api.py:1711` and `mcp_server.py:2496`; nothing renders it | Verification still means reading raw JSON over MCP |
| History is a browser-local invocation log | `studio/src/history.ts` (localStorage) | Not tied to a song or a version; useless as provenance |

Keep, do not disturb: content-sha versions; recorded `parent`; optimistic
locking (`Store.save(..., expected_version, enforce_expected)`, `base.py:336`);
append-only provenance; the gold pointer (`api.py:1270/1284`); the tool contract
(`tool_contract.py` — `category`, `browserSafety`, `outputArtifactKinds`,
`modelUse`, `expectedDuration`); `SchemaForm` + `seedFromWorkbench`
(`workbench.ts:211`) auto-fill; `benchMismatches` (`workbench.ts:173`).

## 2. Decision

**Version tree + tool drawer + candidate/commit loop.** One screen per song:
Versions (left) · Song inspector (centre) · Tools (right), plus a candidate bar.

Core loop: **select version → pick tool (form pre-seeded) → Run → candidate →
Diff → Commit or Discard.** Report-only tools attach their report to the
selected version instead of producing a candidate.

Rejected:

| Paradigm | Why not |
|---|---|
| Node graph (ComfyUI, Houdini) | Single object, single-input tools; wiring cost before every run; duplicates the version DAG the store already keeps |
| Modifier stack (Blender, Lightroom) | Right mental model, wrong mechanics — a live re-evaluated stack is invalid for model-backed, non-deterministic, costly tools |
| Notebook / cell log (Jupyter) | Good audit, bad answer to "what is the current song" |

Tree rendering: hand-rolled SVG. Trees are under 100 nodes; `@gitgraph/react` is
unmaintained and `react-git-log` is overkill. No new dependency.

## 3. The tool surface the drawer must present

15 of the 58 contracts declare `song` in `outputArtifactKinds` — these are the
candidate producers. Grouped as the drawer will group them:

**Deterministic, instant, read-only — the workbench's bread and butter**

| Tool | Category | Inputs beyond the song |
|---|---|---|
| `snap_song_to_mir` | alignment | `mir_analysis` |
| `carry_forward_song_timing` | alignment | — |
| `apply_lrc_to_song` | alignment | `lrc_matches`, `mir_analysis` |
| `retime_song_sections` | alignment | — |
| `guard_song_timing_collapse` | alignment | — |
| `score_song_confidence` | alignment | `candidate_source`, `mir_analysis` |
| `apply_deterministic_song_patch` | alignment | `song_patch` |
| `validate_song_json` | parsing | — (also outputs `song_validation`) |
| `build_song_baseline` | baseline | `candidate_source`, `song_identity` |

**Long / server-filesystem-restricted, still deterministic** —
`align_song_deterministically`, `process_song_deterministically`
(`browserSafety: server_filesystem_restricted`, `expectedDuration: minutes`,
also emit `run_trace`, `quality_report`). Run them as jobs, not inline.

**Model-backed** — `reconcile_song` (`modelUse: required`, minutes).

**Self-committing — the one real conflict with the candidate loop**

`analyze_and_store_song`, `realign_song_to_recording` and `save_song` declare
`song_version` in their outputs: they write a version themselves, so there is no
candidate to preview. Decision: the drawer marks these **"writes directly"**,
disables Commit for them, and on completion refreshes the tree so the new node
appears. They are not routed through the candidate bar and never pretend to be.
`get_song` outputs `song` but is a read — it is an inspector action, not a
drawer tool.

## 4. Server changes

| # | Change | Files | Phase |
|---|---|---|---|
| S1 | Add `parent: str \| None` to `SongVersion` and return it from `/versions` | `store/base.py:93`, `store/memory.py:127-137`, `store/firestore_store.py`, `api.py:1703`; `mcp_server.py::list_song_versions` inherits it | M1 |
| S2 | `GET /v1/songs/{id}/diff.json?a=&b=` returning `{lines:[{idx, field, a, b}], chords:[…], sections:[…], summary:{timingsChanged, chordsChanged, linesChanged}}` | new helper beside `Store.diff` (`store/base.py:332`), route in `api.py` next to `:1711`, tests in `tests/` | M1 |
| S3 | Save on a non-head parent + explicit per-song `head` pointer (like gold) | `store/base.py::save` (`:336`), `api.py::post_song`, `SaveSongRequest.parent` | M2 |
| S4 | Per-version reports: `GET/PUT /v1/songs/{id}/versions/{sha}/reports` for grade / confidence / theory output keyed by sha | `api.py`, store backends | M2 |

S1 and S2 are additive and cannot break a client. S3 changes what
`GET /v1/songs/{id}` serves and is deliberately held to M2.

## 5. Studio changes

New default route `/studio/song/:id`. Sections collapse from 9 to 4: **Songs**
(library) · **Song** (workbench) · **Runs** · **Config**. Tool Studio, Repair,
Build, Automatic Pipeline and Evaluation stop being top-level tabs; their tools
move into the drawer and their views become inspector tabs. `Runs` stays — it
monitors async `/analyze`, `/align`, `/stems` jobs, which is not song editing.

New components under `studio/src/workbench/`:

| Component | Responsibility | Reuses |
|---|---|---|
| `SongWorkbench.tsx` | Layout shell, selected-version and candidate state, ⌘K tool search, ⌘↵ run | `useApi`, `mcp.ts` |
| `VersionTree.tsx` | SVG DAG from `parent`; gold + head badges; selection; context actions | S1 |
| `SongInspector.tsx` | Tabs Lines · Chords · Sections · Timing · Provenance · JSON; low-confidence highlighting; timeline strip with section bands and coverage gaps | `Sheet.tsx`, `waveform.ts`, `AudioWorkspace.tsx` |
| `ToolDrawer.tsx` | Grouping per §3; search; selected tool form | `SchemaForm.tsx`, `seedFromWorkbench` (`workbench.ts:211`), `filterTools` (`tooling.ts:158`) |
| `CandidateBar.tsx` | Last-run summary (counts from the S2 diff of selected vs candidate), elapsed, cost; View diff / Discard / Commit | `extractCandidateSong` (`steps.ts:107`), `summariseSongChange` (`steps.ts:130`) |
| `CompareView.tsx` | Two versions side by side, changed-only filter, stat tiles, timeline overlay, raw diff; Keep B / Set gold / Branch from A | S2, `GET /diff` |
| `workbench/state.ts` | Pure reducer `{songId, selectedSha, candidate?, compare?}`, URL-synced `?v=sha&cmp=sha` | `benchMismatches` (`workbench.ts:173`) |

Invariants the UI enforces:

- Nothing writes without an explicit Commit, except the three self-committing
  tools in §3, which are labelled as such before they run.
- Commit sends `expectedVersion` = selected sha; a 409 surfaces as "this song
  moved — reload", never as a silent overwrite.
- Audio-consuming tools stay disabled until audio provenance matches the song
  (`benchMismatches`).
- Model tools show `modelUse` and an estimated cost before Run.
- Timing is never hand-edited in the JSON tab (read-only). Edits go through
  tools — the deterministic-pipeline rule in `CLAUDE.md`.

## 6. Task pipeline

Sequential unless marked ∥. One sub-session and one branch per task. "Effort" is
reasoning effort. No Fable anywhere in this plan.

| # | Task | Branch | Model / effort | Depends | Done when |
|---|---|---|---|---|---|
| T0 | This ADR | `docs/workbench-adr` | Opus / medium | — | Merged |
| T1 | S1 — `parent` in `SongVersion`, `/versions`, both stores, MCP | `feat/version-parent` | Sonnet / medium | T0 | `/versions` returns `parent`; memory + firestore tests pass |
| T2 | S2 — `diff.json` + tests | `feat/diff-json` | Opus / medium | T1 | CCR `ac9cab6e76a1` vs its parent returns non-empty `timingsChanged`; unit tests on synthetic songs |
| T3 ∥ | `workbench/state.ts` reducer + URL sync + tests | `feat/wb-state` | Sonnet / low | T0 | Full branch coverage on the reducer |
| T4 ∥ | `VersionTree.tsx` + 3-branch fixture tests | `feat/wb-tree` | Sonnet / medium | T1 | Renders fixture; click selects; gold/head badges |
| T5 | `SongInspector.tsx` tabs + timeline strip | `feat/wb-inspector` | Sonnet / medium | T3 | Renders the CCR song at any sha; low-confidence rows flagged |
| T6 | `ToolDrawer.tsx` grouping (§3) + form + run → candidate, no write | `feat/wb-tools` | Sonnet / medium | T3 | `snap_song_to_mir` yields a candidate in state; no POST issued; self-committing tools show the "writes directly" badge |
| T7 | `CandidateBar.tsx` + commit + discard | `feat/wb-candidate` | Opus / medium | T2, T6 | Commit creates a version whose `parent` is the selected sha; 409 surfaced cleanly |
| T8 | `CompareView.tsx` | `feat/wb-compare` | Sonnet / medium | T2, T4 | Any two shas render; changed-only filter; tiles match the S2 summary |
| T9 | `SongWorkbench.tsx` shell, routing, nav collapse to 4, delete dead sections | `feat/wb-shell` | Opus / high | T4–T8 | `/studio/` lands on the last-opened song workbench; `studio/e2e/tool-studio.spec.ts` updated or removed with its section |
| T10 | Playwright e2e `studio/e2e/workbench.spec.ts` **and a CI job that runs it** | `test/wb-e2e` | Sonnet / medium | T9 | Green in CI against `SNOOCLE_STORE=memory` |
| T11 | S3 — branch-from-non-head + head pointer, and "Branch from here" | `feat/branching` | Opus / high | T9 | Fork renders as a second column; the app still serves head; tests |
| T12 | S4 — per-version reports + inspector badges | `feat/version-reports` | Sonnet / medium | T9 | Badges without re-running; reports survive reload |
| T13 | Review pass: a11y, perf on 60-line songs, dead-code removal, docs | `chore/wb-polish` | Opus / medium | T10 | Lighthouse a11y ≥ 90; no unused exports |

Milestones: **M1 = T0–T10** (usable workbench, linear history) → deploy.
**M2 = T11–T13** (branching, reports, polish).

Cost guardrails: Sonnet for self-contained components with fixtures (T3–T6, T8,
T10, T12); Opus only where server semantics or integration risk live (T2, T7,
T9, T11).

## 7. CI facts that change how tasks land

- The `studio` job runs `npm ci`, `npm test` (vitest) and `npm run build` in
  `studio/`. **Playwright is not in CI** — `npm run test:browser` exists locally
  and `studio/e2e/tool-studio.spec.ts` is never executed by a gate. T10 must add
  the job, not merely add a spec.
- The workflow's "Committed bundle matches source" step is **vestigial**:
  `snoocle_server/studio/*` is gitignored (`.gitignore:7-8`, only `.gitkeep` is
  tracked), so that check can never fail. The Docker image builds the bundle
  from `studio/` instead. No task should try to commit a bundle; fixing or
  deleting the dead step belongs to T13.
- The `python` job deselects `tests/test_mcp_server.py::test_mcp_tools_over_stdio`.
- `playwright.config.ts` boots uvicorn on 127.0.0.1:4173 with
  `SNOOCLE_STORE=memory` and `SNOOCLE_API_TOKEN=browser-test-token`; the memory
  store is where e2e version trees get built.

## 8. Acceptance for M1

Checked in the browser against the CCR song
(`creedence-clearwater-revival--creedence-clearwater-revival-have-you-ever-seen-the-rain`,
5 stored versions, oldest `cfb45e1f69e6`, newest `ac9cab6e76a1`):

1. The tree shows all 5 versions with parent lines, and the gold badge on
   whichever sha `GET /gold` returns.
2. Selecting an older sha shows that version's lines and chords in the inspector.
3. "Snap timing to audio" pre-fills its form; Run produces a candidate bar
   reading "N timings changed"; no version was written.
4. View diff → side by side; Commit → a new node appears as a child of the
   selected sha with message `snap_song_to_mir`.
5. Compare any two shas; Set gold from there.
6. Tool Studio / Repair / Build / Evaluation are gone, and nothing that was
   possible before is missing — all 15 song-producing tools of §3 are in the
   drawer.

## 9. Defaults if undecided

- Head vs latest: until T11, head = latest by timestamp (current behaviour).
  Do not block M1 on it.
- `Runs` stays a section.
- Tree library: none. SVG.
