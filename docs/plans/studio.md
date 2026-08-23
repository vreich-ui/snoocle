# Snoocle Studio — plan of record

**Version:** 1.0 · 2026-08-22 · amends `SNOOCLE_MASTER_PLAN.md` v1.2 (2026-07-27)

Studio was built on 3 August from a prompt pack that was never committed. Nothing
in this repository said what it was for, which sections it would grow, or how it
related to the vanilla admin at `/ui/`. This document is that missing statement.
It is the plan of record for `/studio/`; where it conflicts with the master plan,
this document wins, and §7 lists exactly what it overrides.

---

## 1. What Studio is

An **operator console** for driving the Snoocle server: run any MCP tool against
any stored song, watch what the pipeline did, and repair what it got wrong.

It is deliberately not two other things:

- **Not the play-along player.** That is `/ui/play/`, it is user-facing, and it
  stays no-build vanilla JS per master plan §6 (Phase C).
- **Not a general admin.** `/ui/` remains the working song admin — the review
  queue, batch queue, agent trace and bracket editor all live there and all
  work. Studio takes those over only as it replaces them section by section,
  and only where being a compiled app buys something real.

The audience is one person who knows the pipeline. Density beats hand-holding;
nothing is hidden because it looks technical.

## 2. Why it is compiled, when `/ui/` is not

Studio talks to `/mcp` with the official MCP TypeScript SDK, generates its forms
from live `inputSchema` JSON, and renders a tool catalog that changes whenever
the server changes. Hand-writing that in dependency-free vanilla JS means
hand-maintaining a tool registry — exactly what the Tool Studio contract forbids
("no proxy and no hand-written tool inventory", PR #74).

That is the whole justification, and it is narrow on purpose: **the SDK and the
schema-driven forms are why Studio compiles.** No other part of Studio may be
used to argue for more build tooling elsewhere.

## 3. Architecture

**A song is a reference, not a document.** As of PR #80 the deterministic MCP
tools accept `song_id` + optional `song_version` instead of a pasted `song_json`
string. Studio must always pass the reference. Shuttling whole Song documents
through form fields is not an acceptable fallback — if a tool cannot take a
reference yet, that is a server gap to fix, not a client workaround.

**REST for reading, MCP for doing.** Library, song detail, runs, queue and
evaluation read `/v1/...` through `client.ts`. Tool invocation goes through
`/mcp`. Two clients is correct here; do not collapse them.

**Auth is a bearer token**, held in `sessionStorage` for the tab only. The server
already runs a full OAuth 2.1 AS for `/mcp`; adopting it for Studio is open
(§6).

**State that belongs to the user's browser** — invocation history, the current
selection — never leaves it, and never carries the token.

## 4. Sections

| Section | State | Plan |
|---|---|---|
| Tool Studio | Built | Live MCP catalog, generated forms, telemetry, local history, audio artifacts |
| Library | Built | Song list, search, needs-identity flags |
| Runs | Built | Queue panel, merged run list, full step trace |
| Repair | Not built | Review queue, confidence heat, identity repair, chord-over-lyric editor (master plan D1/D4/D5) |
| Build | Not built | Stepwise manual build; pasted-sheet import (D6) |
| Automatic Pipeline | Not built | One-shot analyze plus batch queue with retry and cancel (D2) |
| Evaluation | Not built | Scorecard against gold, token and cost rollups (D3) |
| Configuration | Not built | Agent workbench, providers, YouTube cookies, OAuth clients |

Song detail is a route under Library, not a section: it is the gate from a song
onto everything that can be done to it.

Unbuilt sections must say so on screen, name what they will become, and point at
the `/ui/` equivalent that does the job today. A placeholder that pretends to be
a feature is worse than an empty one.

## 5. Sequence

Each step is one PR. Order is by dependency, not by appetite.

1. **This document, and CI.** No workflow exists at all: `pytest`, `vitest`, the
   Vite build and `docker build` are ungated, and nothing checks that the
   committed bundle under `snoocle_server/studio/` matches `studio/src`. That
   check matters — the bundle is committed, so it can silently go stale.
2. **Workbench.** A selection of song + version + audio artifact, held across
   sections, injected into tool forms by field name (`song_id`, `song_version`,
   `audio_ref`, `youtube_url_or_id`, `title`, `artist`). Selecting in Library
   sends to Tool Studio. Depends on PR #80, which is why it comes after it.
3. **MIR by reference.** `mir_json` still has no reference form. A `mir_run_id`
   reading the MIR already stored in a run trace removes the last blob.
4. **Chroma out of MIR.** Already computed inside chord recognition and
   discarded. Return it; it is the cheapest pitch lane.
5. **Timeline.** Transport, ruler in bars|beats and seconds, stacked lanes
   (waveform, sections, chords, lyrics, beats, chroma), playhead, zoom. Drag a
   range and it fills `start_seconds`/`end_seconds` for `analyze_mir_window` —
   selection drives tools.
6. **Melody contour.** `pyin` over the Demucs vocal stem, drawn against the
   timeline. Shows where a transcription drifts from the sung line.
7. **Repair.** The review queue and editor, on top of the timeline.

Automatic Pipeline, Build, Evaluation and Configuration follow, and need the
long-running-request pattern (progress from run traces, cancel, reconnect) that
none of the built sections have yet.

## 6. Open decisions

- **Design tokens.** Master plan §3.5 is binding on "web C/D, iOS F/H" and
  defines the palette Studio ignores (`bg0 #0E1116`, `accent #5B8CFF`,
  `chord #FFB454`, 4-pt spacing, 44 px targets, both themes). Studio ships its
  own navy hex with no custom properties. Either it adopts `ui/tokens.css`, or
  §3.5 is narrowed to exclude the operator console. Adopting is the better
  answer — the chord amber in particular is a semantic, not a taste.
- **Auth.** Retyping the token every tab is friction with no security benefit
  over `localStorage`; using the existing OAuth AS would remove it entirely.
- **`/ui/` end state.** Whether Studio eventually absorbs the admin or the two
  coexist permanently. Not urgent, but it decides how much is worth building
  twice.
- **Two copies of the master plan.** `vreich-ui/snoocle` and
  `vreich-ui/Snoocle-iOS` both carry `SNOOCLE_MASTER_PLAN.md`, nearly identical
  and already drifting. One should be canonical and the other a pointer.

## 7. What this amends

- **D4** ("Web UI stays no-build vanilla JS") and **§0.7** ("No build step for
  the web UI") — narrowed to the player and admin. The operator console at
  `/studio/` is a compiled Vite app, for the reason in §2. `/ui/` and
  `/ui/play/` remain no-build, and the vendored-libraries rule still holds for
  them.
- **§3.5** — unchanged in force, but Studio does not comply today; §6 records
  that as an open decision rather than pretending otherwise.

The master plan is also simply behind: it predates the Mac worker (30 July), the
deterministic pipeline (2–3 August) and Studio itself. Treat its Phase A–H task
lists as still-valid intent and its "current state" section as stale.
