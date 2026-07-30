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
│                    JSON, identical across providers), engine.py (splice ->
│                    validate -> repair loop -> server-side provenance/
│                    guardrails), lyric_refs.py (the model emits lyric
│                    REFERENCES, never lyric text; deterministic code splices
│                    the words in), mock_reconciler.py (deterministic offline
│                    reconciler)
├── store/jobs.py    the job broker: durable queue with time-boxed LEASES, so
│                    a worker that sleeps or dies simply loses its claim and
│                    the job returns to the pool. Reclaimed lazily on read —
│                    no cron, no scheduler, no always-on CPU
├── store/           step 6-7: SongRepository interface (base.py) with
│                    Firestore (firestore_store.py, durable) and in-memory
│                    (memory.py, hermetic) backends; content-hash versions,
│                    expected_version optimistic locking via a Firestore
│                    transaction, append-only provenance, JSON diffs
├── quality/         step 5f: the deterministic grade (grader.py), fault
│                    attribution — MODEL vs AUDIO vs SOURCE (attribution.py),
│                    and the escalation decision with its one-retry ceiling
│                    (escalation.py). theory.py is the optional music21
│                    key-explainability check
├── reconcile/match.py  score_candidate: one candidate sheet against the MIR
│                    chord timeline at all 12 transpositions, so a sheet in
│                    another key reads as good evidence needing a shift
├── timing/root_match.py  the root-pitch-class matcher shared by snap.py and
│                    reconcile/match.py — one implementation, so a grade
│                    measures what the pipeline actually did
├── timing/          Phase A/B deterministic timing: snap.py (MIR chord/line
│                    time assignment), carry_forward.py (the same fields
│                    carried from the PRIOR version when a run doesn't
│                    listen, plus the guard that refuses to store a version
│                    which drops audio-derived data), quantize.py (beat grid
│                    + snapping), confidence.py (per-placement agreement
│                    scoring + the review queue), lrc.py (LRCLIB synced
│                    lyrics), offset.py (cross-video offset by
│                    cross-correlation)
├── audio/stems.py   B4: demucs separation + ffmpeg-rendered practice mixes.
│                    Separation needs torch and minutes of CPU, so it runs on a
│                    worker; READING the cache needs nothing, which is what
│                    lets the API serve what the Mac made. docs/STEMS.md
├── timing/align.py  B2: WhisperX alignment-only — the lyrics are known, so the
│                    model is asked WHEN each word is sung, never what. Prefers
│                    the vocals stem when B4 has produced one, and records
│                    which input it heard
├── batch.py         parses the admin's "add many" textarea into job specs
├── worker.py        the analysis worker (`snoocle-worker`) — claims jobs from
│                    a server and runs them locally. Outbound-only; runs on
│                    Wolf's Mac under launchd. See docs/WORKER.md
├── ui/              the two browser surfaces (vanilla JS, NO build step)
│   ├── tokens.css   the §3.5 design system — every colour/size/space/motion
│   │                token, dark-first + light + stage themes. Both surfaces
│   │                consume these and define none of their own
│   ├── common.js    the only shared code: api/apiJson/el/clear
│   ├── app.js       the admin SPA (edit, agent trace, versions, play)
│   ├── admin-d.js   Phase D admin panes: review queue, batch queue, version
│   │                "what changed" chips
│   ├── vendor/      committed third-party assets (chords-db); see its README
│   │                for what is deliberately NOT vendored, and why
│   └── play/        the play-along player: player.js (library + song page),
│                    timedscroll.js (the iOS scroll model ported as PURE
│                    functions — unit-tested in tests_js/), theory.js
│                    (display-only transpose/capo), chordbox.js (SVG chord
│                    diagrams), diagrams.js, controls.js, sw.js (PWA)
├── pipeline.py      orchestration: per-step timeouts, best-effort
│                    discover/acquire/mir + fatal reconcile/store (502 names
│                    the failed step), truthful per-step status report
├── oauth/           OAuth 2.1 authorization server for /mcp — Claude's
│                    remote-MCP connector cannot use a static bearer token, so
│                    the service issues its own. protocol.py is pure rules
│                    (redirect matching, PKCE, audience) and unit-tested
│                    directly; store.py is durable so a Cloud Run cold start
│                    doesn't silently unauthorize the connector. See
│                    docs/MCP_OAUTH.md
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

## Where analysis actually runs

Two paths, on purpose:

**One song, now** — `POST /v1/songs/analyze` runs in-process on the server and
returns the finished song. It completes inside its own request, so it needs no
worker and no always-on CPU. This is the "Add song" button, and it works from
the iPad whether or not any other machine is awake.

**A queue** — `POST /v1/queue` stores jobs; an external worker claims and runs
them. The server never executes them.

That split exists because background work on Cloud Run is priced out of
proportion to itself: CPU is throttled outside a request under the default
billing mode, so a background task needs `--no-cpu-throttling --min-instances=1`
to survive — about $53/month at 1 vCPU / 2 GiB before analyzing anything. A Mac
that is already on the desk does the same work faster (demucs: ~12s per song on
Apple Silicon vs ~6 minutes on a Cloud Run vCPU) for nothing.

The mechanism that makes it dependable is the **lease**. A claim expires after
five minutes without a heartbeat, so a closed laptop, a dropped connection or a
crashed worker all resolve the same way: the job returns to the queue. Nothing
has to detect the failure, which means nothing can fail to detect it. Leases are
reclaimed lazily when the queue is read, so the property holds with no scheduler
anywhere. `docs/WORKER.md` has the protocol.

## The two browser surfaces

`/ui/` is the **admin**: a developer tool for editing a song, watching the
agent's run trace, comparing versions, and now (Phase D) working the review
queue and the batch queue.

`/ui/play/` is the **player**: the user-facing play-along surface — library
grid, a sheet that scrolls in time with the video, chord-level highlighting,
diagrams on tap, practice controls, installable as a PWA.

They share exactly one file (`common.js`) and one design system
(`tokens.css`). That is deliberate: the admin is allowed to stay dense and
desktop-shaped, and the player is allowed to be touch-first, without either
constraining the other.

**The scroll model is the interesting part.** `play/timedscroll.js` is a port
of the iOS `SyncTimeline` + `TimedScrollModel`, written as pure functions of
(timeline, layout, playhead) so it can be unit-tested with no DOM
(`tests_js/timedscroll.test.mjs`). Four regimes:

1. **Hold at top** until the first timed line is due — a 40-second intro sits
   still rather than drifting a third of the way down the sheet.
2. **Glide** linearly between lines that are close together.
3. **Hold across a break**, then glide in over the final `maxGlide` seconds,
   so the sheet *arrives* on the next line rather than crawling through a
   30-second solo.
4. **Hold** after the last timed line.

`maxGlide = max(8s, medianLineGap * 2.5)` — derived per song, so a dense song
gets a short budget and a ballad a longer one.

Nothing tracks "where we scrolled to" as state: every frame asks the model
where the sheet should be for the current playhead. Seeking, playback-rate
changes and A/B loop wraps are therefore correct without any extra code.

The playhead itself comes from the YouTube IFrame API polled at 4 Hz and
extrapolated per frame from the last (wall clock, media time) anchor. Per-video
offsets (B3) are applied in exactly one place, so every other caller thinks in
song seconds.

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

## Schema v2 (chord/line timing) and the N.C. policy

`schema/song.py` v2 adds OPTIONAL `ChordPlacement.timeSeconds/confidence/beat/
voicingHint`, `Line.timeSeconds/confidence`, and `AudioInfo.analyzedVideoId/
videoOffsets/beats`. All v1 documents keep decoding unchanged. `voicingHint`
is a display hint (e.g. a fret-diagram string) and is deliberately NEVER
chord-parsed — it carries no harmony identity and is exempt from the
sounding-harmony rule.

**N.C. ("no chord") policy:** the server schema rejects `N.C.` (and other
no-chord tokens — see `chords.is_no_chord`) as a stored `chord` value; an
instrumental gap is represented by the ABSENCE of a placement over that span,
not a placement whose chord is "no chord". The iOS app's older model treated
`"N.C."` as a renderable marker chord; that special case is being removed
client-side (see the iOS dev plan, task F1) in favor of rendering a gap
glyph when a timed instrumental hole exceeds the scroll model's
`maxGlideSeconds`. Do not reintroduce N.C. as a storable chord identity.

## The model never writes the lyrics

A valid Song document IS the complete lyrics of a copyrighted song, so every
successful reconciliation used to ask the model for a full verbatim
reproduction — and two live runs were blocked by Anthropic's content filter
with four sources successfully fetched each. That is structural, not a
retrieval failure, and prompt wording cannot fix it.

So a model-backed provider emits a REFERENCE per line instead of text:

    {"lineIndex": 7,
     "lyricRef": {"sourceId": "ultimate-guitar-1087597", "line": 12},
     "chordPlacements": [{"charIndex": 0, "chord": "F"}]}

and `reconcile/lyric_refs.py` splices the real words out of the source that
is already in the run's context, before validation and storage. This is an
internal protocol between the server and the model, **not a schema change**:
`lyricRef` never reaches the store, and `schema/song.py` is untouched. The
agent-facing schema is derived from the real one at request time, so the two
cannot drift.

Four rules make it a guarantee rather than a request:

1. **Retyped lyrics are rejected** and repaired. An instrumental line is the
   one exception and says so with `lyrics: ""` and no ref.
2. **charIndex is validated after splicing**, against the resolved text —
   named per line, repaired, never silently clamped.
3. **Unresolvable refs fail the run.** An unknown sourceId or an
   out-of-range line index is not retried: a retry invites the model to
   supply the line from memory, and a lyric with no valid provenance must
   not reach the store.
4. **Overrides are audited and capped.** `lyricOverride` + a required
   reason covers the genuine cases (merging two partial sources, an obvious
   source typo, a line no source covers); each lands in provenance as
   `action="lyric-override"`, and past ~15% of lines the run fails.

The prior song is registered as a referenceable source under `prior-song`,
which is what lets a notes-only run (which gathers no candidates by design)
return the user's own document without retyping it. Sheets the in-process
agent fetches mid-run are registered the same way, under the `agent-N`
sourceId the tool returns.

`emits_lyric_refs` on the provider decides who is party to the contract: the
four model-backed providers whose prompt this repo writes. The deterministic
mock builds its Song in local code and is never prompted; the external
`agent` MCP workspace is a third-party system that adopts the protocol by
flipping the same flag once it emits refs.

`scripts/measure_lyric_refs.py` measures what this costs the model to emit.
The saving is bounded by the lyric share of the document, which depends
entirely on chord density: ~14% for a dense 4-chords-per-line sheet, ~39%
for a lyric-dense one. A long `sourceId` is repeated on every line and eats
into it — with Ultimate Guitar's ~23-character ids the dense case comes out
slightly LARGER than before. Token reduction is a side effect here, not the
point; the point is that no model-backed path can emit a lyric at all.

## A beat grid that stops before the song does

Beat tracking is onset-strength based (librosa is the engine the production
image actually runs — madmom is excluded from the Docker build), and its
backtrace ends at the last local maximum of the DP's cumulative score. A
fade-out therefore truncates the beat list well before the audio ends:
`bob-marley--three-little-birds` stored 440 beats ending at 177.31s of a
192.18s track — 8% of the song with no beat data, while the structure engine
had section times through the full 192s. Every chord placement past the last
matched MIR segment then piled onto one timestamp, because `snap.py`
interpolates between matched neighbours and there was no later neighbour.
The same loss of lock happens mirrored at the head of a track that fades in
from near silence (measured on synthesized audio, `test_mir_beat_extension`).

`mir/beats.py::extend_beat_grid` continues the grid rather than re-detecting
in the faded region — that region is exactly where onset detection is
unreliable, so tempo continuation is the more accurate answer, not a
compromise. It fires only when the gap at an end exceeds two bars AND at
least 16 beats were detected (enough to trust the tempo), continuing at the
median interval of the nearest 32 beats and cycling the engine's own
beat-in-bar phase. It runs on the full-track path only: a fast-accuracy
window's edges are where the trim fell, not where the music did.

Continued beats are never indistinguishable from measured ones. `Beat.detected`
and `BeatMark.detected` (default `true`, so pre-existing documents decode
unchanged) carry the distinction into the stored song, the run trace payloads
report `beatsDetected`/`beatsInferred`, and `timing-snap` provenance reads
`beats=filled (440 measured, 37 inferred)`. Chord recognition and structure
segmentation are handed the DETECTED beats only — a timeline repair is not
new evidence. The end of the continued grid also becomes the closing anchor
`snap.py` lacked, so trailing placements spread across the fade instead of
collapsing; when nothing was continued, they hold the last matched time
exactly as before.

## A generic guard against collapsed timing, independent of cause

The fade-out above is one specific cause of one specific symptom: several
consecutive lines or in-line chord placements ending up at the exact same
`timeSeconds`. A second, unrelated cause produces the identical symptom: a
1966 live recording too lo-fi for the chord recognizer, whose MIR chord
timeline dies at 86.5s of a 220.6s track — lines 17-20 all end up at
`86.51755102040816`, and nothing flagged it; the scorecard's similarity
metrics scored the document as merely mediocre, and 39% timing coverage
went unreported.

`timing/collapse_guard.py::guard_against_collapsed_timing` is the safety
net for the NEXT cause nobody has found yet: it runs after every
timing-setting pass (snap, carry-forward, LRC — 5c2 in `pipeline.py`,
before the confidence-scoring step so it judges the corrected times) and
looks only at the final song, not at why it got that way. A run of 3+
consecutive lines, or consecutive placements within one line, sharing an
identical time gets spread across the open interval to the next DIFFERENT
time found later in that sequence — onto the beat grid when there is one,
evenly divided otherwise — while the run's own first entry (the genuine
anchor the rest piled onto) is left untouched. When no later time exists to
spread toward — the collapse sits at or past the last thing this song ever
measured, the Paint It Black shape above — nothing is invented; the guard
leaves the entries exactly as found and says so in provenance. This is a
guard, not a fix: it never re-derives WHY the collapse happened, so a rising
intervention rate stays visible rather than quietly disappearing behind it.

The same pass reports `timing_coverage` — the fraction of track duration
spanned by line timings — in its `timing-collapse-guard` provenance entry
on every run that timed anything at all, whether or not a collapse fired,
so "39% coverage" is now a number every run surfaces instead of one this
repo had to notice by hand.

## Grading a document, and escalating only when escalation can help

Every signal above was computed and then dropped on the floor. `timing-snap`
recorded "matched 9/86 chord placement(s)" (0.10) for the 1966 live take and
32/66 (0.48) on the next attempt, both stored without comment. Three Little
Birds was reconciled ten times in sixteen minutes — match ratios 0.41 0.51
0.42 0.53 0.68 0.55 0.48 0.47 0.63 0.49 — no convergence, nothing noticing.

`quality/` closes that loop in three separate pieces, deliberately separate
because they answer different questions:

**`quality/grader.py` — the grade.** Pure, deterministic, no model, no
network. Seven metrics: chord match ratio, timing coverage, interpolation
share, surviving collapse runs, section coverage, theory validity, lyric
completeness. Each one REUSES the pass that already computed it (the match
ratio is `timing.snap`'s own matcher via `timing/root_match.py`; coverage and
collapse runs are `timing.collapse_guard`'s; the interpolation tier is
`snap.INTERPOLATED_CONFIDENCE`) — a grade measuring something subtly different
from what the pipeline did would be worse than no grade. A metric with no
inputs reports `None` and is excluded from the weighted overall: "not
measured" must never read as "measured and bad". Thresholds are configurable
(`SNOOCLE_QUALITY_*`), and the verdict is `pass`/`warn`/`fail`/`unknown` —
`fail` when the overall is below its threshold OR when half of the measurable
metrics fail, since perfect chords and words cannot make an untimed document
playable. The grade lands in provenance (`quality-grade`) and on the run trace
on EVERY run, whatever it says: the grade history is what would have made ten
non-converging Marley runs visible on the second one.

Theory validity is the one new signal: a chord no key explains is usually a
transcription error that survived reconciliation. music21 (extra `theory`)
owns the key -> scale and chord-symbol -> pitch-class mappings; a chord counts
as explained when it is fully diatonic (raised seventh included in minor) or
is a borrowed major/secondary dominant on a diatonic root. Deliberately
permissive — a false "this song is broken" is worse than a missed one — and
absent the extra it reports "not measured", like any other optional engine.

**`quality/attribution.py` — whose fault.** A low grade is not automatically
the model's, and retrying a run whose evidence was bad pays full price for the
same result. Each candidate is scored against the MIR timeline by
`reconcile/match.py::score_candidate`, at all twelve transpositions, so a
sheet in G against a recording in Bb reads as the good source it is (score
1.0 at +3) rather than as garbage. Then: candidates disagreeing with each
other -> **SOURCE**; candidates agreeing with each other while the timeline
contradicts them, or a chord timeline spanning under half the track ->
**AUDIO**; candidates agreeing with each other AND the audio while the
document agrees with neither -> **MODEL**. Anything else is `NONE`/`UNKNOWN`,
and both are non-actionable — guessing "model" is how a retry budget gets
burned on runs that were never going to converge.

**`quality/escalation.py` — what to do.** MODEL fault: one retry, handed the
grade and the specific failures (metric names, offending line and placement
indexes — `build_retry_feedback`), never a vague "try harder". AUDIO fault:
store it and mark the version `timing-unreliable` in provenance; do not
retry. SOURCE fault: one targeted, cache-bypassing search, which earns the
single retry only if the new sheets agree with the audio materially better
than the ones already judged contradictory. **Never more than one retry per
grade**, enforced structurally — `pipeline.py` threads the real spent counts
into `plan_escalation`, so a second escalation cannot be planned. Collapse
runs are deliberately not an escalation path at all: the collapse guard
already spread what could honestly be spread, and a run that reaches the
grader is one where "could not time this region" beats fabricated spacing.

In `pipeline.py` this is step 5f, which is why steps 5-5d live in one
re-runnable unit (`_reconcile_and_time`): a retry that re-reconciled without
re-snapping, re-guarding and re-scoring would be graded on a document no
store would ever receive. Grading is best-effort throughout — a grader
failure records itself in `steps` and never fails a run — and a retry that
dies leaves the first attempt's graded document to be stored, since losing a
storable song to a failed optional retry is strictly worse.

## A permanent id is forever — auditing and repairing a bad one

A song id is minted once (`identity.py` -> `slugify_song_id`) and is then the
content-hash-versioned store key: it is never renamed, because renaming would
orphan the version history the old id carries. Before PR #45/#51 tightened
identity resolution, several documents were minted under an id derived from a
bad guess — a channel name, cover phrasing read as the artist, an unstripped
upload description (`unknown--official-video`,
`wil-per--rolling-stones-paint-it-black-live-1966`, and others). Those ids
cannot be corrected in place.

`snoocle_server/store/identity_audit.py` (CLI: `scripts/fix_song_identity.py`)
is the maintenance path for that, split into three deliberately independent
operations:

- `find_mismatched_identities` — REPORT ONLY, always. Re-resolves every stored
  song's identity from its `audio.youtubeVideoId` through today's
  `identity.py` and lists whichever ids disagree, with the current id, the id
  it would get now, and the confidence. Never writes, never re-analyzes. A
  song with no video id, or whose id already matches, is skipped — it has
  nothing to report. It does not decide which mismatches matter: a wrong
  artist in the id doesn't mean the document's content is wrong (one of the
  eight, the Creedence one, is the current gold version) — the operator picks
  what to act on.
- `supersede_song` — the forward pointer. One new, append-only version of the
  OLD id carrying `Song.supersededBy` and a `"superseded"` provenance entry;
  the content is otherwise untouched and every prior version, including the
  one right before this, stays exactly as it was and fully readable. Never a
  delete — the store is append-only by design and old versions are the only
  record of how the library got here.
- `reanalyze_and_supersede` — DRY RUN BY DEFAULT. Without `--confirm` it only
  previews (a metadata fetch + identity resolution; no pipeline run, no store
  write). With `--confirm` it runs the normal analyze pipeline
  (`pipeline.run_pipeline`, same code path as a live analyze request) under
  whatever id `identity.py` resolves today, then supersedes the old document
  with the id the pipeline actually landed on. It refuses outright — before
  touching the pipeline or the store, in both dry-run and confirmed mode —
  when the old id is already superseded, already matches, has no video id, or
  identity.py still can't resolve it; and when the old id carries a gold-eval
  pointer (`store/evals.py`), since moving that to the new id automatically
  would be deciding on the operator's behalf, not reporting.

## Timing survives a re-analysis that doesn't listen

`snap_chords` is the only pass that fills timing, and it is a documented
no-op when there is no MIR. A re-analysis with `scope.listen=false` has none
by construction, so the reconciler's freshly-emitted document — correctly
timing-less, since a post-pass normally fills it — used to store with
`audio.beats` emptied, `metadata.bpm` nulled, every placement `beat`/
`timeSeconds` gone and every section time gone. It validated and it stored.

Three things now stand between that and the store, in the pipeline rather
than in a prompt (a prompt can only ask):

1. **Precondition.** `listen=false` means "reuse the existing audio
   analysis", so the run must have one. The prior version (request
   `priorSong`, else the stored latest) is resolved before any expensive
   step; a run with neither fails at step `timing` instead of committing a
   timing-less document.
2. **Carry-forward** (`timing/carry_forward.py`) runs after reconcile and in
   place of `snap_chords`. It copies `audio.beats`, `metadata.bpm` and
   `audio.analyzedVideoId`, restores `timeSeconds`/`confidence`/`beat` for
   every line and placement that MATCHES a prior one, carries section times
   by index + line range, and REGENERATES `audio.syncMap` from the resulting
   line times so the two cannot diverge. Matching tolerates the reconciler
   legitimately adding material: placements pair on `(lineIndex, chord,
   charIndex)` with a reading-order fallback within the line, and a
   genuinely new placement keeps empty timing rather than stealing a
   neighbour's. The pass records `action="timing-carry-forward"` provenance
   in `timing-snap`'s notes shape, so the acceptance script and the evidence
   manifest read either.
3. **Guard**, independent of both: no run may store a version whose
   `audio.beats` is empty or `metadata.bpm` null when the prior version had
   them — whatever path produced it. `allowTimingLoss=true` (HTTP) /
   `allow_timing_loss` (MCP) is the explicit opt-out, and also waives the
   precondition in 1.

## A targeted correction is a PATCH, not a rewrite

A one-chord correction used to run the full pipeline: rediscover sources,
re-listen, and ask the model for a complete Song — the same shape as a
first-ever analysis. Reproduced case: "change the C to a B" on Paint It
Black re-discovered three sources and was blocked by the content filter
asking the model to reproduce the whole lyric sheet again, over one chord.
Regenerating the document is also the ONLY way timing loss can happen at
all — carry-forward (above) exists to repair that, after the fact.

**Routing (`correction_routing.py`).** `classify_correction(guidance)` looks
at guidance text and decides two independent things: is this a *targeted*
correction (infer `scope.listen=false, reconcile=false` when the caller left
scope unset), and — separately — does it name lyrics (never patch-eligible,
regardless of how targeted it is). Deterministic regexes run first and are
free: a chord symbol, a line/section reference, or an explicit "X to Y"
replacement is targeted; naming a lyric word sets `targets_lyrics` and
therefore blocks patching even if a chord rule also fired. Only when no
deterministic rule fires does a cheap LLM classification get a vote, and it
can only ever grant `is_targeted_correction` — never `patch_eligible`,
because a generic yes/no isn't trustworthy enough for the one property that
must never be wrong. An explicit caller-supplied `scope` always wins over
inference, and inference never fires with no prior document to correct
against. Whichever rule fired is logged and lands in the `scope` step's
text (`pipeline.py`), e.g. `inferred: targeted (rule=chord-symbol+
line-reference)`.

**The op protocol (`reconcile/patch_ops.py`).** In notes-only scope, when the
provider opts in (`supports_patch_ops`) and the correction is patch-eligible,
the model is asked for a list of operations against the PRIOR document
instead of a Song:

    {"ops": [{"op": "replace_chord", "lineIndex": 12, "charIndex": 21,
               "from": "C", "to": "B", "reason": "..."}]}

The op set is closed and small — chord replace/insert/remove/move, section
rename/bounds change, line split/merge — and an unknown op fails the run
rather than being interpreted. Lyric text can never appear in an op; that is
what makes lyrics categorically unpatchable rather than merely discouraged.
`apply_patch` applies each op to the prior Song and re-validates the result;
an op whose `from` doesn't match, or that targets an out-of-range index, is
NOT fuzzy-matched or silently skipped — it fails the run and names the op.
Over `MAX_OPS` (20) fails and tells the caller to use full re-analysis. Every
applied op lands in provenance individually, `action="patch-applied"`, with
its own reason — a single-op provenance trail rather than one entry for a
whole regenerated document.

**Untouched by construction.** A patch only ever mutates what an op names.
`audio.syncMap` is regenerated only if a structural op (split/merge) ran;
otherwise the applied document is byte-identical to the prior one apart from
the named field. This is stronger than carry-forward: carry-forward repairs
timing loss after a regeneration by matching placements back up; a patch
never regenerates, so there is nothing for carry-forward to repair and
`timing`/`lrc`/`confidence` (the passes that exist to guard a regenerated
document) are skipped — their step text says so (`skipped (patch: ...)`)
rather than silently no-opping.

**Visible fallback, never silent.** The model can still decline to patch —
a correction that looked targeted but, on reading the prior document, needs
different words — by returning `needsFullReconcile` instead of an ops list.
That is a normal, expected outcome, not an error: `reconcile()` falls
through to the ordinary full-reconcile path for that run, and
`patch_ops_applied` stays 0, which is exactly what routes it back through
carry-forward and the guarding passes above. What must never happen is a
silent one — a patch attempt that quietly widens into a rewrite without the
caller being able to tell from the response which path ran.

## Cross-video offset alignment (master plan B3)

`timing/offset.py`'s `estimate_offset(ref, other)` cross-correlates onset-
strength envelopes (librosa, hop 512) over a bounded lag search (default
+/-30s) to find the constant number of seconds to add to a song's stored
(ref-video-based) times so they land correctly on a DIFFERENT video of the
same song — e.g. a re-upload with a longer intro, or a live version.
`POST /v1/songs/{id}/video-offset` acquires both videos' audio (reusing the
existing yt-dlp cache), runs the estimate, and writes
`AudioInfo.videoOffsets[videoId]` plus a `video-offset` provenance entry.

The peak Normalized Cross-Correlation doubles as the confidence. This is a
**documented heuristic, not a statistical guarantee** — empirically, a
genuinely aligned pair scores roughly 0.6-0.97 depending on how rhythmically
distinctive the audio is, while unrelated audio clusters below ~0.3-0.46
(see `tests/test_offset.py`, which pins this calibration). The populations
sit closer together than a naive "just check it's high" read would suggest,
which is exactly why the confidence is always returned to the caller instead
of being collapsed into a silent accept/reject: below
`Settings.offset_min_confidence` (default 0.5) the endpoint refuses with 409
rather than store a guess, and a human can always override by POSTing an
explicit `offsetSeconds` directly (stored at confidence 1.0, gate bypassed).
If real-world usage shows the threshold needs retuning, re-run the
calibration sweep described in `timing/offset.py`'s module docstring before
changing the number.

## Export (master plan B6)

`GET /v1/songs/{id}/export?format=chordpro|txt|json` — deterministic, no LLM.
`export.py`'s `to_chordpro`/`to_txt` share one inline-bracket layout
(`[Chord]` spliced into lyrics at charIndex, `[Section Name]` on its own
line) that the EXISTING generic chord-sheet parser
(`discovery/chordsheet.py`) already understands — so export and re-paste is
a round trip, not a one-way dump (see `tests/test_export.py`). `chordpro`
additionally prefixes ChordPro metadata directives (`{title: ...}` etc.) for
interop with other ChordPro tools; those aren't parsed back by our own
parser (only the sheet body needs to round-trip, not the metadata banner).

## Ultimate Guitar discovery source (master plan B5)

`discovery/sources/ultimate_guitar.py` — OFF by default (`SNOOCLE_SOURCE_UG=1`
to enable). UG has no documented public API; this reads the same JSON its
own React frontend hydrates from, embedded in a `<div class="js-store"
data-content="...">` attribute on both the search-results page and every
tab page. Because that shape is undocumented and known to drift, every field
lookup tries several candidate key paths (see `_JSON_PATHS`-style constants
at the top of the file) rather than betting the source on one exact
contract, and — matching every other discovery source's contract — ANY
failure (network, HTTP status, missing keys, unparseable content) returns
`[]`/`None` and logs, never raises into the pipeline. `discover_sources`
merges its output in additively alongside the generic web search (not as a
fallback tried only when the web search is empty). UG's own `[ch]C[/ch]`
inline-chord markup is converted to the `[C]` bracket convention the
existing generic chord-sheet parser already understands, and `[tab]...[/tab]`
fret-diagram blocks are stripped entirely before parsing. UG's declared capo
(from its separate metadata field, since it isn't always written into the
sheet text itself) is transposed away at ingestion exactly like every other
source's declared capo, and rating/votes feed a confidence prior
(`min(0.95, 0.5 + 0.1*log10(votes+1) + 0.05*rating)`) — real evidence a
well-vetted community tab deserves a higher starting confidence than an
arbitrary, unvoted web hit.

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
  beat engine. Chord-CNN-LSTM ships its ~28 MB checkpoints in the upstream
  repo — a plain `git clone` via `scripts/setup_chord_model.sh` is the
  complete install (no git-lfs), plus CPU torch (`.[chordmodel]`); the
  Docker image bakes it in and presets `SNOOCLE_CHORD_CNN_LSTM_DIR`.
  SongFormer is the one needing multi-GB checkpoints (git-lfs) + torch.
  Both are integrated via a documented external-runner contract with honest
  librosa fallbacks, so the pipeline is always audio-grounded.

## Running

```sh
python3 -m venv .venv && .venv/bin/pip install -e .[mir,dev] anthropic python-multipart
.venv/bin/snoocle-server          # HTTP API on 127.0.0.1:8765
.venv/bin/snoocle-mcp             # MCP server on stdio
.venv/bin/python -m pytest        # 92 tests
.venv/bin/python scripts/acceptance.py --offline   # acceptance report
```
