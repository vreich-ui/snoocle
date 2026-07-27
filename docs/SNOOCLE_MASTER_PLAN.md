# Snoocle Master Plan — Agent-Centric Song Foundry + Play-Along Suite

**Version:** 1.2 · 2026-07-27 · author: Claude (planning session with Wolf)
**v1.1:** added Phase H — singing input + sing/play-along analysis on iOS (voice pitch vs harmony, guitar chord verification, scoring, ambient sync); new building blocks; §12 updated.
**v1.2:** added §3.5 binding design system & UX spec (+ task C0 admin restyle); added §13 Cowork execution guide — a recommended model tier for every task and mixed server+iOS session batches.
**Repos:** server = `github.com/vreich-ui/snoocle` · iOS = local Xcode project `Snoocle`
**Deployed:** Cloud Run `snoocle-99287560712.europe-west1.run.app` (FastAPI + embedded MCP at `/mcp`, Firestore store)

This plan is written to be executed **task by task by a less capable model**. Every task is self-contained: goal, files to touch, exact steps, and acceptance criteria. Read §0 before doing anything.

---

## 0. Executor instructions (read first, follow always)

1. **Do exactly one task at a time**, in the order given inside a phase. Phases can be reordered only where §12 marks them independent.
2. **Never refactor code outside the files a task names.** If a task seems to require it, stop and leave a `TODO(plan-vN)` comment instead.
2b. **UI tasks obey §3.5 (design system) verbatim** — no invented colors/sizes — and **every task runs on the model tier assigned in §13** (escalate per §13.1 after two failed acceptance attempts).
3. **Run the test suite after every task**: `python -m pytest` in the server repo (all green before moving on). For iOS: build the `Snoocle` scheme and run `SnoocleSelfTests`.
4. **Every new module gets tests** in the same style as the existing `tests/` directory. A task is not done until its acceptance criteria pass.
5. **All new Song-schema fields are optional.** Old stored songs must keep loading. Never write a migration that rewrites Firestore documents in bulk.
6. **Chord rule is sacred:** every stored chord is sounding harmony (never a shape/tab/capo'd name). Capo/tuning/transpose are display-only. This is already enforced in `snoocle_server/chords.py` + `schema/song.py` — do not weaken it.
7. **No build step for the web UI.** Vanilla JS, vendored libraries under `snoocle_server/ui/vendor/` (committed to the repo). No npm, no bundler, no CDN at runtime.
8. **Heavy/optional Python deps** (alignment, stems) go in optional extras in `pyproject.toml` and the code must degrade gracefully (clear "engine unavailable" status, never a crash) when they're absent — same pattern the MIR fallbacks already use.
9. **Personal-use project.** Scrapers/downloaders (YouTube, Ultimate Guitar, lyrics) are acceptable; still keep each behind its own module boundary so any source can be disabled with one config flag.
10. When a task says "record in provenance", append a `ProvenanceEntry` — never mutate or delete existing entries.

---

## 1. Current state (verified 2026-07-27)

### Server (`snoocle_server/`)
- **Pipeline:** discovery (web search → site-agnostic chord-sheet parser → `CandidateSource`s) → audio acquire (yt-dlp, video-id cache) → MIR (madmom beats/downbeats, Chord-CNN-LSTM chords w/ start–end timestamps, SongFormer/agglomerative structure, key vote) → LLM reconciliation (anthropic/openai/gemini/**agent**/mock providers; validate→repair loop) → Firestore versioned store (content-hash versions, optimistic locking, append-only provenance).
- **Song schema** (`schema/song.py`, v1): metadata, displayPreferences (capo/tuning), audio {youtubeVideoId, durationSeconds, syncMap[lineIndex→time]}, sections (with optional MIR startTime/endTime), lines (lyrics + chordPlacements keyed by charIndex), provenance.
- **Key gap:** MIR produces a full chord timeline with timestamps, but the reconciled Song only keeps **line-level** times in `syncMap`. Chord-level timing is discarded.
- **Agent hooks that already exist:** `agent` provider = Snoocle acts as MCP *client* calling one tool (`SNOOCLE_AGENT_MCP_TOOL`, default `reconcile_song`) at `SNOOCLE_AGENT_MCP_URL`; re-runs send `prior_song` + human `guidance`; gold versions + scorecard (`/v1/eval/scorecard`); agent workbench (instructions/tools/budgets) in the web UI; full run traces.
- **Web UI** (`ui/`): dependency-free vanilla JS admin — song list, Edit (bracket-text editing, sections), Agent (run trace + MIR timeline viz), Versions (diff, gold), Play (static sheet + plain YouTube embed; `autoScrollTo()` is a stub), Scorecard, Workbench.
- **MCP:** 16 step-scoped tools embedded at `/mcp`.

### iOS app (SwiftUI, iPad-first, works on iPhone)
- Library, Add song (server pipeline trigger), song detail w/ chords-over-lyrics sheet (`WrappedChordLineView`), practice scroll engine: `SyncTimeline` (line start times) + `TimedScrollModel` (hold-at-top through intro ✔, glide between close lines, hold across instrumental gaps, medianGap-derived policy) + `PracticeScrollView` (UIScrollView + CADisplayLink easing) + `PracticeClock` (clock playhead w/ rate 0.6–1.4 when no video). YouTube player via embedded web player (`YouTubePlayerController`), cookies harvesting, background upload/analysis jobs, versions/diff/run-trace views, extensive self-tests.
- **Late-start scrolling already works**: sheet holds at top until the first timed line. What's missing is chord-level highlight, per-video offsets, diagrams, practice tools.
- **Local model drift:** iOS `ChordPlacement.timeSeconds` and `confidence` exist but are always nil (wire never carries them); legacy `SyncPoint {measure, beat}` struct is dead code; `"N.C."` is renderable in iOS but rejected by server schema.

---

## 2. Locked architecture decisions

| # | Decision |
|---|----------|
| D1 | **Agent-centric syncing.** No tap-to-sync UI. Timing corrections come from deterministic engines + the reconciliation agent; the human corrects by *editing data or telling the agent*, not by performing timing rituals. A single per-song "nudge offset" slider is the only manual timing control. |
| D2 | **Schema v2 = v1 + optional fields only** (§4). `SCHEMA_VERSION` becomes 2; v1 docs remain valid. |
| D3 | **Timing is layered, deterministic-first:** LRCLIB synced lyrics → forced alignment (vocals) → MIR beat/chord inference → agent judgment. Each layer records per-line/per-chord confidence + provenance. LLM never invents timestamps that a deterministic layer already supplies. |
| D4 | **Web UI stays no-build vanilla JS**; the play-along player is a new page sharing `app.js` helpers; small MIT libs are vendored (chords-db JSON, chord-diagram renderer). |
| D5 | **Learning lives in CMS Agent for v1** via the existing `agent` provider. `// MARKER(agentic-home):` comments at every integration point so v2 can move to a dedicated "personal projects" agent workspace by changing config only. LibreChat talks to Snoocle through CMS Agent, which registers Snoocle's `/mcp` as a project. No embedded chat in Snoocle. |
| D6 | **Word-level lyric timing lives outside the Song** as a per-song artifact (`alignments/{songId}`), fetched on demand (karaoke mode). The Song schema stays lean: line times + chord times + optional beat refs. |
| D7 | **Stems are cached artifacts**, not schema. Demucs output stored under the song, served by streaming endpoints. |
| D8 | **iOS: one app target, iPhone + iPad layouts via size classes** (NavigationSplitView on regular width). No separate iPad target. |

---

## 3. Free building blocks (verified alive, roles assigned)

| Component | What it is | Used for | Integration |
|---|---|---|---|
| [chords-db](https://github.com/tombatossals/chords-db) (MIT) | JSON DB of guitar + ukulele chord fingerings (positions, barres, fingers, multiple voicings per chord) | Chord diagrams: tap a chord → finger placement, voicing browser, capo advisor scoring | Vendor `lib/guitar.json` (and `ukulele.json`) into `ui/vendor/chords-db/`; also bundle into iOS as a resource for a custom renderer OR use SwiftyChords' own data |
| [vexchords](https://github.com/0xfe/vexchords) (MIT) | JS SVG chord-box renderer | Web chord diagrams | Vendor the single built JS file; feed it positions from chords-db |
| [SwiftyGuitarChords](https://github.com/BeauNouvelle/SwiftyGuitarChords) | Swift package: chord diagram drawing + its own chord JSON | iOS chord diagram popovers | SPM dependency; map sounding chord (+display transpose/capo) → key+suffix lookup |
| [LRCLIB](https://lrclib.net) (free API, no key; [python wrapper](https://github.com/Dr-Blank/lrclibapi)) | Community DB of **synced lyrics** (LRC line timestamps) | Deterministic line-level lyric timing: fetch by (track, artist, duration), match lines to reconciled lyrics | New `timing/lrc.py`; plain `httpx` GET `https://lrclib.net/api/get?artist_name=&track_name=&duration=` — no wrapper dependency needed |
| [WhisperX](https://github.com/m-bain/whisperX) (or [stable-ts](https://github.com/jianfch/stable-ts)) | ASR + **forced alignment** with word-level timestamps | Align *known* lyrics text to the vocal stem → word/line times when LRCLIB misses; karaoke word map | Optional extra `[align]`; run on vocals stem; alignment-only mode (we supply the text, it returns timings) |
| [demucs](https://pypi.org/project/demucs/) (htdemucs) | Stem separation (vocals/drums/bass/other; 6-stem model adds guitar/piano) | Backing tracks (mute guitar / mute vocals), cleaner input for alignment (vocals) and chordrec (accompaniment) | Optional extra `[stems]`; job endpoint + cached artifacts (D7). CPU works, slow — cache aggressively |
| [ChordSheetJS](https://github.com/martijnversluis/ChordSheetJS) (MIT, active) | Parse/format ChordPro + chords-over-lyrics | **Import/export interchange**: export songs as ChordPro; optional import path for pasted sheets | Server-side export is trivial without it; vendor only if pasted-sheet import in web UI is wanted (task D6) |
| [Ultimate Guitar mobile-API scrapers](https://github.com/Pilfer/ultimate-guitar-scraper) ([python variant](https://github.com/joncardasis/ultimate-api)) | UG exposes internal JSON the mobile app uses | High-quality candidate source: chords + declared capo + votes/rating as prior confidence | New discovery source `discovery/sources/ultimate_guitar.py` (personal use, one config flag to disable) |
| [ChordMiniApp](https://github.com/ptnghia-j/ChordMiniApp) (MIT) | Full web app: beat-aligned chord grid, synced YouTube playback, guitar diagrams, lyrics transcription | **Reference implementation** — steal UX patterns (beat-grid chord view, enharmonic correction display), NOT code (it's Next.js/Firebase; conflicts with D4) | Read-only inspiration; its model-serving ideas already informed the MIR layer |
| [tonal](https://github.com/tonaljs/tonal) (MIT) | Music-theory JS (chords, keys, roman numerals, transposition) | Web: client-side transpose display, chord-function coloring (I/IV/V…), capo math | Vendor the single UMD build |
| YouTube IFrame Player API | `getCurrentTime()`, `setPlaybackRate()`, events | Web player playhead + A/B loop + slow-down | Replace static embed with `enablejsapi=1` player object |
| `img.youtube.com/vi/{id}/mqdefault.jpg` | Free thumbnails | Library artwork everywhere | Plain `<img>`/AsyncImage |
| [basic-pitch](https://github.com/spotify/basic-pitch) (Apache-2.0) | Note-level transcription | EXPERIMENTAL (§10): riff/solo → tab suggestions from the guitar stem | Optional extra, phase G only |
| [torchcrepe](https://github.com/maxrmorrison/torchcrepe) (MIT) | CREPE pitch tracker, PyTorch | Server-side: extract the **reference vocal melody** (f0 track) from the demucs vocals stem → target-note track for sing-along scoring | Optional extra `[melody]`; CPU fine at analysis time (H1) |
| [SwiftF0](https://github.com/lars76/swift-f0) (+ [pitch-benchmark](https://github.com/lars76/pitch-benchmark)) | Fast CNN monophonic F0 detector (2025), beats CREPE-tier accuracy at a fraction of the compute; ONNX model | Candidate for **on-device** live voice pitch (Core ML conversion) | Experimental alternative inside H3's `PitchDetecting` protocol |
| [AudioKit](https://github.com/AudioKit/AudioKit) `PitchTap` (MIT) | Battle-tested iOS DSP pitch detection (powers many tuner apps) | **Primary on-device live pitch engine** for singing input | SPM dependency; first `PitchDetecting` implementation (H3) |
| [UltraStar Deluxe](https://github.com/UltraStar-Deluxe/USDX) / [UltraSinger](https://github.com/rakuri255/UltraSinger) (GPL — reference only) | Open-source karaoke game + AI pipeline that auto-builds its note files (whisper + CREPE + demucs — same stack as this plan) | **Algorithm reference** for singing scoring (octave-agnostic semitone match per beat, tolerance bands); do NOT copy code | Scoring spec in H5 is written from first principles |
| [Chord-Detector-and-Chromagram](https://github.com/adamstark/Chord-Detector-and-Chromagram) (GPL — reference only) | Real-time chromagram + chord detection in C++ | **Reference** for H6's on-device chroma math; our implementation is ~200 lines of Accelerate/vDSP, expected-chord verification (easier than open recognition) | Reference only |
| Apple ShazamKit (free, on-device/custom catalogs) | Audio matching with matched-offset timestamps; `SHSignatureGenerator` for custom catalogs | **Ambient sync**: sheet follows the song when it's playing from any speaker in the room, no video needed (H8) | Native framework, no cost |
| Apple AVAudioEngine voice processing (`setVoiceProcessingEnabled`) | Built-in echo cancellation (AEC) | Speaker-mode play-along: subtract the iPad's own playback from the mic. Speech-tuned — treat speaker mode as reduced-quality; **headphones = full quality** | Native; H3 route-change policy |

---

## 3.5 Design system & UX spec — BINDING for every UI task (web C/D, iOS F/H)

> UX review verdict (2026-07-27): the current web UI is a competent *developer admin* — it already has CSS tokens, dark mode, zero dependencies — but it is desktop-only (fixed 260 px sidebar, 100 vh flex), typographically dense (13 px tables, monospace song-ID slugs instead of titles), modal-heavy, and has no touch targets, skeletons, or empty states. That's acceptable for the admin; it is NOT the player. The iOS app is native SwiftUI and structurally sound; it needs consistency polish, not rework. Everything below is the shared design language both players adopt. **Executor rule: when a C/D/F/H task renders UI, the values in this section override any improvisation. Do not invent colors, sizes, or spacing.**

### Tokens (web: `ui/tokens.css` custom properties; iOS: `Theme.swift` constants, same names)
- **Color (dark is the player default):** `bg0 #0E1116` (app background) · `bg1 #151A21` (cards/panels) · `bg2 #1D242E` (raised: popovers, bars) · `stroke #2A3340` · `text #E9EDF2` · `textDim #8B96A5` · `accent #5B8CFF` · `chord #FFB454` (chords are ALWAYS this amber family — never the accent blue; lyrics are `text`) · `chordActive #FFD27D` (+ soft outer glow) · `ok #3FCF8E` · `warn #F6C445` · `danger #FF6B5E`. Confidence tints (D4/H): ok/warn/danger at 18 % alpha backgrounds. Light theme: derive by inverting lightness, keep hues; ship it, but dark-first. **Stage mode:** pure `#000` background option (OLED), text 1.15× size.
- **Type:** UI = system stack (SF Pro on Apple). Sheet = `ui-monospace` (SF Mono) — chords row 15 px/17 pt semibold in `chord`, lyrics 17 px/19 pt regular; sheet line-height 1.6; section headers 13 px uppercase tracking +0.06em `textDim`. Web page titles 20 px/600. Respect Dynamic Type on iOS (sheet scales with the font-size control, UI with the system).
- **Space & shape:** 4-pt spacing scale (4/8/12/16/24/32). Radius: 10 (controls) / 14 (cards) / 20 (sheets & popovers). Hairline strokes, no drop shadows on dark except popovers (y2 blur16 @ 30 %).
- **Motion:** 150–250 ms ease-out for state changes; chord-pill activation ≤ 120 ms; honor `prefers-reduced-motion` / `UIAccessibility.isReduceMotionEnabled` (highlighting still moves — it IS the content — but decorative transitions stop).
- **Touch & a11y:** every tappable ≥ 44×44 (pt/px); AA contrast minimum; visible focus rings on web; keyboard on web: space = play/pause, ←/→ = section jump, L = loop, +/− = transpose.

### Component specs (names used by tasks)
- **SongCard** (C1/F4 library): 16:9 YouTube art, 14-radius, title (1 line, ellipsis), artist `textDim`, badge row: `⏱` timing available · key · difficulty (G3, when present) · `🎚` stems. Skeleton shimmer while loading; `ContentUnavailableView`/empty-state illustration when library is empty ("Add your first song").
- **ChordPill**: monospace chord text in `chord`, 6×2 px padding, 8-radius, transparent at rest; **active** = `chordActive` text + 12 % amber fill + glow; **match/mismatch** (H6) = ok/danger fill at 20 %. Tap target padded to 44 px. Tap → DiagramPopover.
- **DiagramPopover** (C3/F3): `bg2`, 20-radius; header "sounding F#m — shown as Em (capo 2)"; diagram ~180 px wide; voicing pager dots; "approx" badge when simplified.
- **SectionChipsBar**: horizontally scrollable chips (12-radius, `bg1`, active = accent-tinted); current section auto-scrolls into view; long-press (iOS) / right-click (web) = set loop to section.
- **TransportBar** (bottom, `bg2`, safe-area aware): play/pause (56 px), section back/fwd, loop toggle (A/B badges), rate menu, tune (transpose/capo/font) sheet, mic toggle (Phase H). On compact iPhone this is the ONLY chrome once playing.
- **PitchLane** (H4): 72 pt lane above the sheet, `bg1`, lane ticks = current chord tones (amber ticks, labeled), live dot 10 pt with 300 ms trail; melody mode adds note rectangles (accent at 35 % fill, hit portions turn `ok`).
- **ReviewRow** (D1): chord + bar context + confidence meter (3-segment) + reasons as `textDim` chips; actions right-aligned; keyboard j/k/enter.

### Layout specs
- **Player, compact (iPhone portrait / narrow web):** video 16:9 pinned top, collapses to a 64 pt mini-bar (thumbnail + title + playhead) once the user scrolls or playback starts; sheet fills the rest; TransportBar bottom. Nothing else on screen.
- **Player, regular (iPad landscape / desktop):** sheet 62 % left; right rail = video (top), section list, practice controls stack; divider draggable (F4); stage mode = sheet fullscreen, video becomes PiP corner (iOS) / hidden with audio continuing (web can't detach — keep a 200 px mini video).
- **Library:** responsive grid `minmax(168px, 1fr)`, search field pinned top (`.searchable` on iOS), sort menu (Recent/Title/Difficulty), section for "Processing…" queue cards (D2) above the grid.
- **Admin (web `/ui/`):** keep the sidebar layout but restyle with tokens (task C0): song list shows **title — artist** (slug only as tooltip), 15 px minimum text, 44 px rows, skeletons, and the tab bar always visible. No functional changes.

### C0. Tokens + admin restyle (do before C1)
**Touch:** `ui/tokens.css` (new), `ui/style.css`, `ui/app.js` (song-list label only), `ui/common.js` extraction (shared `api/apiJson/el/clear` as specced in Phase C intro)
- Create `tokens.css` from the table above (both themes via `prefers-color-scheme` + a `data-theme` override attr). Rewrite `style.css` to consume tokens only; song list rows render `title — artist` from the extended `GET /v1/songs` (C1 server change; until then, fall back to the id). 44 px targets, visible focus, skeleton class.
**Accept:** admin is visually consistent with the spec, works at 360 px width (sidebar becomes a slide-over), zero JS behavior changes beyond the list label; `tests/test_ui.py` still green.

---

## 4. Phase A — Schema v2 + deterministic timing core (server)

> Outcome: chord-level timestamps and confidences are first-class, populated deterministically from MIR (no LLM), and every downstream surface can rely on them.

### A1. Schema v2 fields
**Touch:** `snoocle_server/schema/song.py`, `tests/test_schema*.py` (new tests file ok)
- `ChordPlacement` += `timeSeconds: Optional[float] (ge=0)`, `confidence: Optional[float] (0..1)`, `beat: Optional[BeatRef]` where `BeatRef = {measure: int ge 1, beat: float ge 1}`, `voicingHint: Optional[str]` (free text like `"x02210"` — ALLOWED here because it is a display hint, not chord identity; document this loudly in the docstring).
- `Line` += `timeSeconds: Optional[float]`, `confidence: Optional[float]`.
- `AudioInfo` += `analyzedVideoId: Optional[str]` (same 11-char validation), `videoOffsets: dict[str, float] = {}` (videoId → seconds to ADD to all song times when playing that video), `beats: Optional[list[BeatMark]]` where `BeatMark = {time: float, measure: int, beatInMeasure: int}` — capped: reject > 10000 entries.
- `SCHEMA_VERSION = 2`. Validators: line `timeSeconds` non-decreasing across lines when present is **NOT** enforced (repeated sections may re-sing earlier lines? no — lines are positional, times are monotonic; DO enforce non-decreasing, matching syncMap's rule); chord `timeSeconds` within a line non-decreasing.
- Keep `syncMap` exactly as-is (back-compat). New invariant: if both `line.timeSeconds` and a syncMap entry exist for a line, they must be equal — enforce by *generating* syncMap from line times in A3, not by hand-editing.
**Accept:** old fixture songs still validate; new fields round-trip; invalid beat/videoOffsets rejected.

### A2. Deterministic MIR-snap post-pass
**Touch:** new `snoocle_server/timing/__init__.py`, `timing/snap.py`, `tests/test_timing_snap.py`
- Pure function `snap_chords(song: Song, mir: MirAnalysis) -> Song`:
  1. Walk the song's chordPlacements in reading order. Walk MIR chord segments in time order.
  2. Greedy alignment: for each placement, find the next MIR segment (at/after the previous match) whose parsed root pitch-class matches the placement's root (use `chords.parse_chord`); tolerance window ±1 segment for passing mismatches. On match: `timeSeconds = segment.start`, `confidence = 0.9` if exact quality-family match else `0.7`.
  3. Unmatched placements: interpolate linearly between neighboring matched placements; `confidence = 0.3`.
  4. Snap each assigned time to the nearest MIR beat if within 0.35 s (`timing/quantize.py`, task A4) — but implement the snap call here behind `if beats:`.
  5. Derive `line.timeSeconds = min(chord times on the line)`; lines with no chords interpolate between neighbor lines; enforce monotonicity by clamping (never reorder).
  6. Regenerate `audio.syncMap` from line times (one entry per line that has a time). Fill `metadata.bpm` and `audio.beats` (downsampled to whole beats) from MIR if absent.
- Wire into `pipeline.py` immediately after successful reconcile, before store. Record provenance `action="timing-snap", actor="snoocle-server/timing"`.
**Accept:** given the existing MIR fixtures, a reconciled fixture song gains chord times covering ≥90% of placements, monotone line times, regenerated syncMap; provenance appended; runs with `mir=None` are a no-op.

### A3. Sounding-harmony guard for `voicingHint` + N.C. policy alignment
**Touch:** `schema/song.py`, `chords.py`, iOS note in §9
- `voicingHint` never parsed as a chord; add schema test proving `chord="x02210"` still rejects while `voicingHint="x02210"` passes.
- Decide N.C.: server keeps rejecting `N.C.` as a *chord*; instrumental gaps are represented by absence. Add to `docs/ARCHITECTURE.md` a note; iOS task F1 removes its `"N.C."` special case (it can render a gap glyph when a timed instrumental hole > maxGlide occurs).
**Accept:** tests as above; doc updated.

### A4. Beat quantize + measure numbering
**Touch:** `timing/quantize.py`, `tests/test_timing_quantize.py`
- `build_beat_grid(mir) -> list[BeatMark]` from madmom beats+downbeats (measure = count of downbeats so far; beatInMeasure resets at downbeat).
- `snap_time(t, grid, tolerance) -> (t', BeatRef|None)`.
- Fill `ChordPlacement.beat` for every placement whose time snapped to the grid. This is the foundation Wolf's future **beat-level chord changes + voicings** builds on.
**Accept:** synthetic grid tests (4/4 at 120 bpm: beat times 0.5 s apart, measures increment every 4).

### A5. Per-chord confidence matrix (agreement scoring)
**Touch:** `timing/confidence.py`, `reconcile/engine.py` (one call site), `store/runs.py` (persist detail), `tests/test_confidence.py`
- After reconcile+snap, compute for each placement: (a) fraction of candidate sources containing the same root at the same lyric position (fuzzy: same line, nearest placement), (b) MIR agreement at `timeSeconds` (root match of the covering MIR segment), (c) combined `confidence = 0.5*sources + 0.5*mir` overriding A2's coarse value when both signals exist.
- Persist a compact `reviewQueue` on the run record: `[{lineIndex, charIndex, chord, confidence, reasons:[str]}]` sorted ascending by confidence, threshold < 0.6.
**Accept:** mock-reconciler pipeline run produces a run record with a reviewQueue; agreeing fixtures score > disagreeing ones.

---

## 5. Phase B — Deterministic enrichment engines (server)

> Outcome: the "agentic sync" promise — lyrics/chords/timestamps aligned with no human tapping — with each engine independent, optional, and honest about availability.

### B1. LRCLIB synced-lyrics engine
**Touch:** `timing/lrc.py`, `config.py` (flag `SNOOCLE_LRCLIB_ENABLED=1`), `pipeline.py`, `tests/test_lrc.py`
- `fetch_lrc(title, artist, duration_s) -> list[{time, text}] | None` via `GET https://lrclib.net/api/get` (params: `track_name`, `artist_name`, `duration`; fall back to `/api/search` best match within ±3 s duration). No API key. Timeout 10 s, failure → None.
- `match_lrc_to_lines(lrc, song) -> dict[lineIndex, (time, similarity)]`: normalize (casefold, strip punctuation), align LRC lines to song lines with difflib `SequenceMatcher` ratio ≥ 0.75, monotonic (never match backwards).
- Merge policy in pipeline: LRC times **win over** MIR-inferred line times when similarity ≥ 0.75 (`line.confidence = similarity`); chord times then re-anchor within each line by distributing between line start and next line start proportionally to charIndex, then re-snap to beat grid. Provenance `action="lrc-align", sources=["lrclib.net"]`.
**Accept:** unit tests with a canned LRC fixture (no network in tests — inject the fetch); pipeline flag off → engine skipped with status note, exactly like MIR fallbacks report.

### B2. Forced alignment engine (vocals → word map)
**Touch:** `timing/align.py`, `pyproject.toml` extra `[align]` (whisperx; pin torch CPU), `api.py` endpoint, `store` artifact
- `align_lyrics(audio_path, lines: list[str]) -> {lines: [{lineIndex, start, end}], words: [{lineIndex, word, start, end}]}`. Use WhisperX **alignment-only** path (we already know the text). If the stems engine (B4) has produced a vocals stem, align against it, else the full mix.
- Store result as artifact `alignments/{songId}/{versionSha}.json` (Firestore doc or GCS/local file — follow existing run-artifact pattern in `store/runs.py`). NOT in the Song (D6).
- Endpoint `POST /v1/songs/{id}/align` (202 + job status via runs) and `GET /v1/songs/{id}/alignment`.
- Merge policy: where LRCLIB was missing/low-similarity, adopt aligned line starts (confidence 0.8); words power web/iOS karaoke highlight later.
**Accept:** module importable without whisperx installed (graceful "engine unavailable"); with extra installed, a 10 s spoken-word fixture aligns 3 known lines in order.

### B3. Cross-correlation video offset (replaces tap-to-sync)
**Touch:** `timing/offset.py`, `api.py`, `tests/test_offset.py`
- `estimate_offset(ref_wav, other_wav) -> {offsetSeconds, confidence}`: librosa onset-strength envelopes (hop 512), full cross-correlation, peak → offset; confidence = peak prominence vs. median.
- Endpoint `POST /v1/songs/{id}/video-offset {videoId}`: acquires the other video's audio (existing acquire step), estimates offset vs. the **analyzed** audio (store `audio.analyzedVideoId` at pipeline time — add to A2's pipeline wiring), writes `audio.videoOffsets[videoId]`, new version, provenance.
- All clients apply `videoOffsets[currentVideoId] ?? 0` to every timestamp at playback.
**Accept:** synthetic test — same signal padded with 3.2 s silence → offset 3.2 ± 0.1 s, high confidence; unrelated noise → low confidence and API refuses to store (409 with reason).

### B4. Stems engine (demucs)
**Touch:** `audio/stems.py`, extra `[stems]`, `api.py`, Dockerfile note (optional layer), `tests/test_stems.py` (contract-level, mock the separator)
- `separate(audio_path, model="htdemucs") -> {vocals, drums, bass, other}` cached under `stems/{songId}/{model}/` (local disk cache dir beside the existing audio cache; add `SNOOCLE_STEMS_DIR`). Derived mixes rendered with ffmpeg (existing `audio/utils.py` ops): `backing_no_vocals`, `backing_no_guitar` (htdemucs 4-stem: no-guitar ≈ vocals+drums+bass, i.e. drop `other`; document the approximation; upgrade path = `htdemucs_6s` guitar stem).
- Endpoints: `POST /v1/songs/{id}/stems` (202, job), `GET /v1/songs/{id}/stems` (list), `GET /v1/songs/{id}/stems/{name}` (audio stream, Range supported).
- Feed-forward: when stems exist, B2 aligns on vocals; optionally re-run chordrec on accompaniment mix and record as an extra MIR opinion (flag `SNOOCLE_CHORDREC_ON_ACCOMPANIMENT=0` default off).
**Accept:** endpoints exist and degrade gracefully without the extra; with it, a 20 s fixture separates and streams; repeat call hits cache (no recompute — assert via mtime).

### B5. Ultimate Guitar discovery source
**Touch:** `discovery/sources/__init__.py` (new subpackage; move nothing), `discovery/sources/ultimate_guitar.py`, `discovery/service.py` (register source behind `SNOOCLE_SOURCE_UG=1`), `tests/test_ug_source.py`
- Fetch UG's internal JSON (the mobile-API approach used by Pilfer's scraper / joncardasis' ultimate-api: search endpoint → tab detail → `wiki_tab` content). Parse with the **existing** generic chord-sheet parser; carry UG metadata: declared capo (transposed away at ingestion as usual), tuning, rating & votes → `CandidateSource.confidence` prior (e.g. `min(0.95, 0.5 + 0.1*log10(votes+1) + 0.05*rating)`).
- Resilience rule: any HTTP/parse failure → source returns `[]` and a status note; never fails the pipeline. HTML layout drift is expected; keep ALL selectors/URLs in one constants block at the top of the file.
**Accept:** unit tests run against saved fixture JSON (no live network); with flag on + network, a search for a common song yields ≥1 CandidateSource with capo already normalized.

### B6. ChordPro export (+ optional import)
**Touch:** `api.py` (`GET /v1/songs/{id}/export?format=chordpro|txt|json`), `tests/test_export.py`
- Deterministic serializer: sections → `{start_of_verse: Verse 1}` comments, placements → inline `[C]` at charIndex, metadata → `{title:} {artist:} {key:} {tempo:}`. Round-trip safe with the bracket-text the web Edit tab already uses.
**Accept:** golden-file test; export of a fixture song re-imports through the existing bracket parser to an equivalent line set.

---

## 6. Phase C — Web play-along player (user-facing)

> Outcome: an iOS-quality play-along experience in any browser: library grid → song page → video-synced scrolling sheet with live chord highlight, diagrams on tap, transpose/capo, section jumps, A/B loop, slow-down, stage mode. PWA-installable.

Design language: match the iOS app (clean sheet, monospace chord row over lyric row, section chips). Keep the existing admin at `/ui/` untouched; the player is `/ui/play/` (new `index.html`, `player.js`, `player.css`) reusing `app.js`'s `api()` helper via a tiny shared `ui/common.js` (extract ONLY `api`, `apiJson`, `el`, `clear` — no other refactors).

### C1. Library grid (player home)
- `GET /v1/songs` (already exists; extend response with `{title, artist, latestVersion, updatedAt, youtubeVideoId, hasTiming: bool}` — server task, same endpoint, additive) → responsive card grid, YouTube `mqdefault.jpg` artwork, search-as-you-type filter (client-side), sort by recent. Tap → `#/song/{id}`. Hash routing, no framework.
**Accept:** loads with 0/1/100 songs; keyboard navigable; no console errors.

### C2. Player core: IFrame API + ported scroll model
- Replace static embed with YT IFrame API player (`enablejsapi`, `playsinline`). Poll `getCurrentTime()` at 4 Hz; between polls extrapolate `anchor + elapsed*rate` (port of `PracticeScrollDrive.timed`).
- **Port `TimedScrollModel` + `SyncTimeline` to JS** (`ui/play/timedscroll.js`, ~120 lines, pure functions, no DOM): same semantics — hold at top until first timed line, glide within `max(8, medianGap*2.5)` s segments, hold across breaks, `leadInset` park position. Unit-test with `node --test tests_js/timedscroll.test.mjs` (add a `make jstest` target; CI optional).
- Apply `videoOffsets[videoId] ?? 0` (B3) to the playhead before mapping. Fallback drive when the song has no timing: constant autoscroll with speed slider (persisted per song in `localStorage` — NOTE: this is a server-rendered page, not a Claude artifact, so localStorage is fine).
- Current line highlighted; **current chord** (first placement with `timeSeconds <= t`, A2 data) gets a pill highlight that steps chord-to-chord.
**Accept:** JS unit tests green; manual: a timed song scrolls in sync, holds through intro/solo; chord pill advances on changes; a song with a late first lyric starts scrolling late (Wolf's core scenario).

### C3. Chord diagrams on tap (chords-db + vexchords)
- Vendor `chords-db` guitar.json (+ ukulele.json) and vexchords. Tap any chord (sheet, pill, or header chord palette) → popover: diagram of the **displayed** chord (after transpose/capo display transform), voicing pager (all chords-db positions), "sounding: F#m — shown as: Em, capo 2" caption line so the sounding-harmony rule stays visible. Unknown chord → nearest simplification (strip extensions via tonal: `Cmaj9 → Cmaj7 → C`) with an "approx" badge.
- Song header shows the song's distinct chord palette in order of first appearance (like UG), each tappable.
**Accept:** every chord in three fixture songs (incl. slash + extended chords) opens a diagram or an approx fallback; capo caption correct.

### C4. Practice controls
- Section chips row (from sections; tap → `seekTo(startTime + offset)`); A/B loop (set A/B at current time, loop via seek when `t > B`); playback-rate menu (0.5–1×, YT supports it) — scroll model needs no change (playhead is the truth); transpose ± semitone + capo display selector (tonal-powered, client-side only, persisted per song via existing save of `displayPreferences`); font size; **stage mode** (fullscreen, dark, oversized fonts, Screen Wake Lock API, tap zones: left = section back, right = section forward).
**Accept:** loop stays within ±0.5 s of A/B; rate change keeps sheet in sync; transpose relabels sheet + diagrams consistently.

### C5. PWA
- `manifest.webmanifest`, icons, service worker caching the app shell + last-opened song JSONs (video excluded). Served by FastAPI static route.
**Accept:** Lighthouse installable; airplane-mode reopen shows library + cached sheets with a "no video offline" banner.

### C6. Karaoke word highlight (needs B2)
- When `GET /v1/songs/{id}/alignment` exists, highlight the current word (background sweep on the lyric line). Toggle in settings; off by default.
**Accept:** fixture alignment drives word sweep in sync; absent alignment → toggle hidden.

---

## 7. Phase D — Admin & agent upgrades (web)

### D1. Review queue UI (needs A5)
- New "Review" pane on the song page (admin side): list from the latest run's `reviewQueue`, each row = chord, line context, confidence, reasons, buttons: **Play this bar** (seek video to `timeSeconds - 1 s`), **Accept**, **Fix** (inline chord input with validation), **Ask agent** (adds to guidance textarea). Accept/Fix writes a new version; fixes are batched into one save.
**Accept:** queue navigable entirely by keyboard (j/k/enter); fixing a chord updates the sheet and clears the row.

### D2. Batch processing queue
- `POST /v1/queue {items: [{title, artist} | {youtubeUrl}], provider?}` → creates queued run records; a background worker (asyncio task in app lifespan; NOTE Cloud Run: document `--min-instances=1` requirement in `docs/DEPLOY_CLOUD_RUN.md`) processes sequentially through the existing pipeline. `GET /v1/queue` = statuses.
- Admin UI: "Add many" textarea (one song per line, `Artist - Title` or URL), queue dashboard card with live per-step status (reuse run trace), retry button.
**Accept:** 3 queued mock-provider songs process sequentially, statuses stream, a failure doesn't stall the queue.

### D3. Scorecard timing metrics (needs A2)
- Extend `eval/metrics.py`: line-time MAE vs gold, % placements with timeSeconds, % lines with confidence ≥ 0.75; scorecard UI columns.
**Accept:** metrics appear for gold-marked fixtures; missing timing → metric reads "—", not 0.

### D4. Confidence heat display
- Sheet views (admin Edit preview + player, behind a toggle) tint chords by confidence (A5): <0.5 red, <0.75 amber. One CSS class per bucket, computed server-side into the song JSON? No — compute client-side from `confidence`.
**Accept:** toggle flips tinting; absence of confidences → toggle hidden.

### D5. Song JSON inspector polish
- Versions tab: add "what changed" chips per version (lines/chords/timing/sections counts changed) computed from the existing diff endpoint.
**Accept:** chips render for fixture history.

### D6 (optional). Pasted-sheet import
- Admin "Paste sheet" box → existing generic parser → CandidateSource → reconcile run seeded with it (covers songs where discovery fails; also the manual UG copy-paste path). Vendor ChordSheetJS ONLY if the existing Python parser proves insufficient for pasted ChordPro; prefer POSTing the paste to the server parser.
**Accept:** pasting the fixture sheet produces a draft song via mock provider.

---

## 8. Phase E — CMS Agent learning loop (v1)

> Outcome: reconciliation runs through Wolf's CMS Agent workspace; every human correction becomes a recorded observation; playbook curation turns observations into durable lessons; LibreChat can drive Snoocle. `// MARKER(agentic-home)` everywhere.

### E1. Point the `agent` provider at CMS Agent
- Config: `SNOOCLE_LLM_PROVIDER=agent`, `SNOOCLE_AGENT_MCP_URL=<CMS Agent MCP endpoint>`, `SNOOCLE_AGENT_MCP_TOOL=reconcile_song`. Nothing to code in Snoocle — verify the existing contract sends `{title, artist, mediaUrl, chords (MIR-timestamped), mir, candidates, songSchema, previousOutput?, validationErrors?, priorSong?, guidance?}` and document it in `docs/AGENT_CONTRACT.md` (new, generated from `reconcile/providers.py` reality — read the code, write the doc).
### E2. CMS-side node (done in CMS Agent, not this repo)
- Create project "Snoocle" in CMS Agent; register Snoocle's `/mcp` (project_call_tool → Snoocle's 16 tools become available to CMS agents and LibreChat). Create workspace node `reconcile_song`: input schema = E1 contract; output schema = Song JSON; prompt = current `reconcile/prompt.py` SYSTEM_PROMPT as the base + playbook injection; skills: none initially. `// MARKER(agentic-home): move to dedicated personal-projects workspace in v2.`
### E3. Correction → observation feedback
**Touch:** `reconcile/engine.py` or `pipeline.py` (one hook), `config.py`
- When a run includes `prior_song`/`guidance` (i.e., a human taught something), after success POST a compact learning record to a configurable webhook `SNOOCLE_LEARNING_WEBHOOK` (CMS Agent's `learning_record_observation` via its MCP/HTTP): `{songId, guidance, diffSummary (changed chords/lines counts), scorecardDelta?}`. Fire-and-forget with 5 s timeout.
**Accept:** mock webhook receives the record in tests; unset → skipped silently.
### E4. Evaluation bridge
- Nightly (or manual) CMS evaluation: rubric = Snoocle scorecard endpoint; optimizer proposals gated on scorecard non-regression (use CMS `evaluation_*`/`optimizer_*`; document the loop in `docs/AGENT_CONTRACT.md`).
### E5. LibreChat cookbook (doc only)
- `docs/LIBRECHAT.md`: worked examples — "add these 5 songs", "why did you pick Bm not D on line 12 of X?" (agent reads run trace via MCP), "requeue X with guidance…".

---

## 9. Phase F — iOS apps (iPhone + iPad, one target)

> Outcome: the existing iPad-first app becomes a polished two-form-factor player with chord-level sync, diagrams, and practice tools. All server data it needs comes from Phases A–B.

### F1. Wire model catch-up (do first)
**Touch:** `Networking/Wire/*`, `SongMapping.swift`, `Models/Song.swift`
- Decode new v2 fields: chord `timeSeconds`/`confidence`/`beat`/`voicingHint`, line `timeSeconds`/`confidence`, `audio.analyzedVideoId`/`videoOffsets`/`beats`. Map chord times into the already-existing `ChordPlacement.timeSeconds` (stop nil-ing it); keep line-time projection as fallback. Delete the dead legacy `SyncPoint {measure,beat}` struct and the `"N.C."` special case (A3). Preserve local-only fields exactly as `SongMapping` does today.
**Accept:** self-tests updated + green; a v1 (old) song still round-trips.

### F2. Current-chord highlight
- `WrappedChordLineView`: highlight the chord pill whose `timeSeconds <= playhead < next`. Reuse the existing playhead plumbing (video 2 Hz / PracticeClock 5 Hz + display-link extrapolation). Also honor `videoOffsets[currentVideoId]`.
**Accept:** visual check on a timed song; no highlight when no chord times exist.

### F3. Chord diagrams on tap
- SPM: SwiftyGuitarChords. Tap a chord pill → popover (iPad) / sheet (iPhone): diagram for the **displayed** chord (apply transpose/capo display transform first), voicing pager, "sounding X shown as Y (capo N)" caption. If `voicingHint` present, preselect the matching voicing.
**Accept:** slash/extended chords fall back gracefully (strip-extension ladder like C3).

### F4. iPhone + iPad layouts
- Regular width (iPad, iPhone landscape?→ no: regular only): `NavigationSplitView` library │ song; song page video-beside-sheet with draggable divider; stage mode = sheet fullscreen, mini video PiP corner.
- Compact width (iPhone): library list → song page video-above-sheet (16:9 collapses to a 56 pt mini-bar on scroll-down); controls in a bottom toolbar.
**Accept:** both size classes navigable in previews/simulator; no layout regressions in existing self-tests.

### F5. Practice controls parity with web (C4)
- Section chips, A/B loop, playback rate (YouTube player `setPlaybackRate` via existing controller; PracticeClock already has rate), transpose display stepper (exists? extend `DisplayPreferences.transpose` UI), font size, stage mode.
**Accept:** loop/rate/transpose function on both a video song and a clock-only song.

### F6. Metronome + count-in (needs `audio.beats`)
- AVAudioEngine click scheduled from the beat grid mapped through the current playhead (+offset); count-in = 1 measure of clicks before `seek(firstLine.time - measureDuration)` on "Practice from here". Clock-mode songs: pure metronome from bpm.
**Accept:** click aligns with beats within perceptual tolerance on a fixture; toggling off tears down the engine.

### F7. Stems player (needs B4)
- Song page "Practice mix" menu: Original video / No-vocals / No-guitar. Stems mode swaps YouTube for AVPlayer streaming the server stem mix (keep video paused-hidden; playhead now comes from AVPlayer). Cache downloaded stems (FileManager, LRU 2 GB).
**Accept:** switching mixes preserves position ±0.5 s; offline replay of a cached stem works.

### F8. Setlists + practice log
- Local-first: `Setlist {name, songIds}` in SongStore; swipe-to-add; setlist play = auto-advance at song end (video ENDED event / clock end). Practice log: per song, seconds practiced + section-loop counts (this powers the future "hardest sections" — record now, visualize later).
**Accept:** persistence across relaunch; log increments during playback only.

---

## 10. Phase G — Imagination tier (each item independent, all optional)

| ID | Idea | Sketch |
|---|---|---|
| G1 | **Capo advisor** (deterministic) | Server endpoint: for each capo 0–7, transpose the song's chord set into shapes, score playability from chords-db (open>barre, fewer fingers, common shapes bonus), return ranked `{capo, shapes, score}`. UI: "Easiest way to play: capo 3 — G C D em". |
| G2 | **Chord-function coloring** | Color chords by scale degree (tonal.js / small Swift port; key from metadata): tonic green, dominant red, etc. Toggle. Teaches *why* progressions repeat. |
| G3 | **Difficulty score** | Deterministic: chord-change rate (changes/min from A2 times), distinct-chord count, barre density (via G1 shapes), bpm. Badge on library cards; sort by it. |
| G4 | **Section practice heatmap** | From F8's loop counts: tint section chips by practice intensity; "you always loop the bridge of Creep". |
| G5 | **Riff transcription (experimental)** | basic-pitch on the guitar/other stem for a selected A/B range → note events → simple monophonic tab suggestion rendered as text under the section. Clearly labeled experimental. |
| G6 | ~~Sing-along scoring~~ | **Promoted to Phase H** (full singing + play-along analysis). |
| G7 | **Smart setlists via agent** | LibreChat: "build me a 20-min easy acoustic set in sharp keys" → agent queries song list + G3 scores through MCP and writes a setlist. Zero new UI. |
| G8 | **Auto re-analysis watchdog** | Nightly job: songs whose latest run predates an engine upgrade (record engine versions in provenance — already there) get requeued at low priority. The library quietly improves as engines do. |
| G9 | **Voicing capture toward beat-level** | When B4's 6-stem guitar track exists: chroma-match each beat's guitar audio against chords-db voicing templates → `voicingHint` per placement. This is the bridge to Wolf's beat-level-voicings end state. Spec (`docs/plans/voicing.md`) first. |

---

## 10-H. Phase H — Singing input & sing/play-along analysis (iOS-first)

> Outcome: the iOS app listens while Wolf sings and/or plays guitar along with a song and gives live, honest feedback — voice pitch against the reference melody AND against the song's harmony, guitar chords verified against the expected chord at each moment, strum timing against the beat grid — plus an end-of-session report the agent can discuss. Everything on-device is real-time DSP or small models; everything server-side reuses the Phase B stems/alignment stack.
>
> **Snoocle's unfair advantage:** after Phase A the app knows *which chord sounds at every second*. So "does the sung note fit the harmony" is a 30-line pitch-class lookup — no ML, no reference melody required. The reference-melody lane (karaoke-style) is a second, optional layer on top.

### Audio-capture ground rules (apply to every H task)
- Mic session: `AVAudioSession` `.playAndRecord`, `.measurement` mode where possible.
- **Headphones = full analysis quality** (mic hears only Wolf). Detect via route observer.
- **Speaker mode = reduced quality**: enable `setVoiceProcessingEnabled(true)` (AEC) on the input node; AEC is speech-tuned and can mangle music — banner "wear headphones for accurate analysis"; never present speaker-mode scores as authoritative (mark session `quality: reduced`).
- All analysis timestamps go through the same playhead used for scrolling (video/clock/AVPlayer + `videoOffsets`), so feedback aligns with the sheet automatically.

### H1. Server: reference vocal melody track
**Touch:** `timing/melody.py`, `pyproject.toml` extra `[melody]` (torchcrepe), `api.py`, `tests/test_melody.py`
- `extract_melody(vocals_wav) -> [{time, f0Hz, midiFloat, confidence}]` (torchcrepe, 10 ms hop, Viterbi decode on).
- `to_note_events(frames) -> [{start, end, midi, cents}]`: confidence hysteresis (in ≥ 0.6, out < 0.4), median filter 5, min note 80 ms, split on ≥ 80-cent moves.
- Merge with B2's word alignment when present: `build_vocal_track(notes, alignment) -> [{start, end, midi, cents, word?, lineIndex?}]`. Artifact `vocaltrack/{songId}/{versionSha}.json`; endpoints `POST /v1/songs/{id}/vocaltrack` (202 job, needs vocals stem from B4 — else 409 "run stems first"), `GET …/vocaltrack`.
**Accept:** synthetic sine-melody fixture → correct MIDI numbers and boundaries ±30 ms; importable without the extra (graceful unavailable status).

### H2. Server+shared: harmony grid logic
**Touch:** `timing/harmony.py`, `tests/test_harmony.py`; mirrored 1:1 in Swift in H4 (`Models/HarmonyGrid.swift`)
- Pure functions over the Song: `chord_at(song, t)` (last placement with `timeSeconds <= t`, honoring lines order), `chord_tones(chord) -> {pitch classes}` (root/3rd/5th/7th/extensions via existing `chords.py` parse), `scale_tones(key) -> {pitch classes}`.
- Verdict for a sung pitch class at time t: `chordTone | scaleTone | outside`.
- `GET /v1/songs/{id}/vocaltrack` response gains per-note `harmony` verdict (server-side precomputed, for display parity checks).
**Accept:** table-driven tests (C major, Am7, D7/F# …); Swift mirror gets the same table in `SnoocleSelfTests` (H4).

### H3. iOS: mic capture + pluggable pitch engine
**Touch:** new `Audio/MicSession.swift`, `Audio/PitchDetecting.swift` (protocol), `Audio/AudioKitPitchEngine.swift`
- Protocol `PitchDetecting`: start/stop, callback `(f0Hz: Double, midiFloat: Double, confidence: Double, rms: Double)` ≥ 20 Hz.
- Implementation 1 (default): AudioKit `PitchTap` (SPM). Leave `// TODO(plan-v2): SwiftF0 Core ML engine` stub — the ONNX→CoreML conversion is an experiment, not on the critical path.
- MicSession owns route policy (headphones/speaker per ground rules), permission prompt copy, and a debug **tuner view** (note name + cents needle) reachable from Settings → Diagnostics.
**Accept:** tuner view tracks humming stably (no octave flapping at steady input — apply 3-frame median); route switch mid-session doesn't crash or leak taps; self-tests cover midiFloat↔note-name math.

### H4. iOS: sing-along HUD (harmony mode first, melody lane second)
**Touch:** `Views/SingAlongHUD.swift`, `Models/HarmonyGrid.swift`, integration in `SongDetailView`/practice screen
- **Harmony mode (works for EVERY song after Phase A, no server extras):** live pitch dot on a horizontal lane above the sheet; dot color = harmony verdict at the playhead (green chordTone / olive scaleTone / red outside); current chord's tone names shown as tick marks (octave-agnostic, mod-12 lane).
- **Melody mode (when `GET …/vocaltrack` exists):** lane becomes a scrolling piano-roll strip (target note rectangles from the vocal track, ±1 octave window, octave-agnostic option); live pitch trail drawn over it; word labels from the track.
- Toggle chip: Off / Harmony / Melody (Melody hidden when no track). HUD is an overlay — the scroll/highlight pipeline is untouched.
**Accept:** harmony mode — singing the root of the current chord shows green within 150 ms; melody mode — fixture track renders and a synthetic pitch stream (injected via `PitchDetecting` fake) rides the rectangles; all with `SnoocleSelfTests` on the pure mapping functions.

### H5. iOS: singing score + report
**Touch:** `Audio/SingScorer.swift`, report UI in H7
- Per note event (melody mode): hit-fraction of frames within ±0.75 semitone, octave-agnostic (mod-12 distance, min over ±1 octave); note score = hit fraction; line score = duration-weighted mean; song score 0–100. Harmony-only sessions score `chordTone%` / `scaleTone%` instead (labeled differently — "harmony fit", not "accuracy").
- Persist per-section results into the practice log (F8 schema += `singing` block).
**Accept:** deterministic unit tests: synthetic pitch streams vs synthetic tracks with known expected scores (perfect=100, half-off=~50, octave-up=100 in octave-agnostic mode).

### H6. iOS: guitar play-along analysis (chroma verification + strum timing)
**Touch:** `Audio/ChromaEngine.swift`, `Audio/GuitarScorer.swift`, HUD extension
- Polyphonic, so no pitch tracking: 4096-pt FFT (Accelerate/vDSP) → HPS-lite folding → 12-bin chroma at ~10 Hz. **Verification, not recognition:** correlate live chroma with the template of the *expected* chord at the playhead (templates root+3rd+5th(+7th) with harmonic bleed weights — same math family as the server's fallback chordrec). Rolling match score with hysteresis → chord pill glows green on match, red flash on sustained mismatch (>0.7 s).
- Strum timing: spectral-flux onset detector (~50 lines); compare onset times to the beat grid (A4 + `videoOffsets`) → per-strum early/late ms, session timing bias meter.
- Guitar mode and singing mode are mutually exclusive per session v1 (`// TODO(plan-v2): simultaneous via stem-aware source separation on-device is out of scope`).
**Accept:** unit tests with additively-synthesized chords (helper in test target): expected-chord match ≥ 90% true-positive / ≤ 10% false-positive across a 12-chord table; synthetic strums time within ±30 ms.

### H7. iOS: session report + agent feedback loop
**Touch:** `Views/PracticeReportView.swift`, `Persistence` practice log (F8), server `POST/GET /v1/songs/{id}/practice-sessions`
- End-of-session card: singing accuracy per section (or harmony-fit %), chord-match rate, strum timing bias, quality badge (headphones/speaker), deltas vs last session. "Share with agent" = POST the compact JSON to the server; LibreChat/CMS agent can then read it via MCP and coach ("bridge accuracy up 20%, the F#m in line 14 is your miss point"). `// MARKER(agentic-home)`.
**Accept:** report renders from a fixture session; endpoint round-trips; list endpoint returns sessions newest-first.

### H8. iOS: ShazamKit ambient sync (delight)
**Touch:** `Audio/AmbientSync.swift`, listen button on song page
- One-time per song: generate + cache a `SHSignature` from the song's cached/streamed audio (`SHSignatureGenerator`). "Listen" mode: `SHSession` matches mic input → `matchedMediaItem` offset → set playhead (+ keep re-anchoring every match) → the sheet scrolls along with the song playing from ANY speaker in the room (vinyl, car, band room). No video, no manual sync.
**Accept:** with a cached signature, playback from a second device syncs the sheet within ~1–2 s and stays locked for 60 s.

---

## 11. Testing & acceptance strategy

- Server: pytest per task (fixtures over network; every engine mockable). Keep `scripts/acceptance.py --offline` green; add sections for timing/lrc/offset/stems as they land.
- JS: `node --test` for pure logic (`timedscroll`, chord display transform). No DOM tests.
- iOS: extend `SnoocleSelfTests` (the project's established pattern) for mapping/model changes; manual checklist per UI task kept in `docs/plans/ios-checklist.md`.
- End-to-end smoke (manual, after each phase): one real song — queue → process → review queue → play in web player → play on iPad → chord diagrams → loop a section.

## 12. Order & dependencies

```
A1 → A2 → A4 → A5          (core, strictly ordered; A3 anytime after A1)
B1, B3, B5, B6              independent after A2
B4 → B2 → C6                stems before alignment-on-vocals (B2 can run on full mix earlier)
C0 → C1 → C2 → C3 → C4 → C5 player, strictly ordered (C0 = tokens, §3.5); C6 after B2
D1 needs A5 · D2 anytime · D3 needs A2 · D4 needs A5 · D5, D6 anytime
E1–E5 anytime after A2 (agent provider benefits from timed chords)
F1 → F2 → F3 → F4 → F5      iOS, strictly ordered; F6 needs A4; F7 needs B4; F8 anytime
G* independent, after their stated deps
H3 → H4(harmony) needs A2+F1 · H4(melody)+H5 need H1 · H1 needs B4 (B2 optional, enriches words)
H2 with A2 · H6 needs H3-infra + A4 · H7 needs H5 or H6 (+F8) · H8 independent (needs cached audio)
```

**Recommended execution order for maximum daily value:** A1–A5 → C1–C4 → F1–F2 → B1 → B3 → D1–D2 → E1–E3 → F3–F5 → **H3 → H4(harmony mode) → H6** → B4 → B2 → **H1 → H4(melody)+H5 → H7** → C5–C6 → F6–F8 → **H8** → D3–D6 → B5–B6 → E4–E5 → G-tier by appetite.

*(Harmony-mode singing feedback and guitar verification land early on purpose: they need nothing from the heavy server stack — only Phase A chord times and a microphone.)*

---

## 13. Cowork execution guide — model per task, session batches

### 13.1 Model tiers
Three tiers, by Claude model family as exposed in Cowork: **Haiku** (cheap, mechanical), **Sonnet** (default workhorse), **Opus** (algorithmic/subtle). Rules of thumb used below: Haiku only where the task is fully specified and failure is obvious (vendoring, docs, manifest, cosmetic); Opus where the task invents an algorithm, fights real-time/DSP subtleties, or spans a large layout overhaul. **Escalation rule: if a session fails a task's acceptance criteria twice, stop, revert to the task's starting commit, and re-run the task one tier up.** Never let a lower model "partially land" a failed task.

| Tier | Tasks |
|---|---|
| **Haiku** | A3 · B6 · C5 · D3 · D4 · D5 · E1 · E5 · G2 · G3 · C0 (Sonnet if the 360 px slide-over proves fiddly) |
| **Sonnet** | A1 · A4 · A5 · B1 · B3 · B4 · B5 · C1 · C3 · C4 · C6 · D1 · D2 · D6 · E2 · E3 · E4 · F1 · F2 · F3 · F5 · F7 · F8 · G1 · G4 · G7 · G8 · H2 · H3 · H5 · H7 · H8 |
| **Opus** | A2 · B2 · C2 · F4 · F6 · H1 · H4 · H6 · G5 · G9 |

Why the Opus ones are Opus: A2 (greedy chord↔MIR alignment with interpolation + monotonicity edge cases), B2 (WhisperX integration + artifact plumbing), C2 (playhead extrapolation + scroll-model port + JS test rig), F4 (dual-form-factor layout overhaul), F6 (audio-clock↔video-clock alignment), H1 (note segmentation hysteresis), H4 (real-time HUD over the scroll pipeline), H6 (on-device DSP from scratch), G5/G9 (experimental ML).

### 13.2 Sessions that mix server + iOS (contract-paired)
One Cowork session CAN touch both sides: the server repo is cloned/edited in the cloud workspace (tests run there), the iOS project is edited through the connected `Snoocle` folder. **Constraint to state in every mixed session prompt: Cowork cannot compile Swift — iOS acceptance = code review + self-test logic written, then Wolf builds/runs `SnoocleSelfTests` in Xcode and reports back before the session marks the task done.** Server-side pytest MUST pass in-session.

| Session | Tasks (order) | Model | Why paired |
|---|---|---|---|
| S1 | A1 → A3 → A4 | Sonnet | Pure schema/timing core, one contract |
| S2 | A2 → A5 | Opus | The two alignment/confidence algorithms |
| S3 | **A1-verify + F1 → F2** | Sonnet | iOS decodes exactly what A1 added — same session keeps the wire contract in one head |
| S4 | C0 → C1 | Haiku→Sonnet (or all Sonnet) | Tokens then library grid |
| S5 | C2 | Opus | Player core, solo on purpose |
| S6 | C3 → C4 | Sonnet | Diagrams + controls on top of a working player |
| S7 | B1 → B3 | Sonnet | Two independent timing engines, small |
| S8 | D1 → D2 | Sonnet | Admin: review queue + batch queue |
| S9 | E1 → E3 (+E2 done interactively in CMS Agent) | Sonnet | Agent wiring end-to-end |
| S10 | **B3-verify + F5** (offset consumption + practice controls) | Sonnet | iOS consumes videoOffsets; loop/rate parity with C4 |
| S11 | **H2 → H3 → H4(harmony)** | Sonnet then **Opus for H4** | Server grid + Swift mirror + mic engine, then HUD |
| S12 | H6 | Opus | Guitar DSP, solo |
| S13 | B4 → B2 | Sonnet then Opus | Stems then alignment-on-vocals |
| S14 | **H1 → H4(melody) → H5** | Opus→Opus→Sonnet | Vocal track, melody lane, scorer |
| S15 | **H7 + F8** (practice log + sessions endpoint) | Sonnet | One data contract, both sides |
| S16 | F3 → F4 | Sonnet → Opus | Diagrams then the layout overhaul |
| S17 | C5 → C6 → D3 → D4 → D5 | Haiku (C6 Sonnet) | Polish sweep |
| S18 | F6 → F7 | Opus → Sonnet | Metronome then stems player |
| S19 | B5 → B6 → D6 | Sonnet/Haiku | Sources & import/export sweep |
| S20 | H8 · G-tier by appetite | Sonnet+ | Delight features |

### 13.3 Session prompt template (paste into Cowork, fill the brackets)
```
Read docs/SNOOCLE_MASTER_PLAN.md in this repo. Execute ONLY tasks [IDs], in order,
under the §0 executor instructions and the §3.5 design system (binding for UI).
Server repo: this repo (run `python -m pytest` after every task — must be green).
iOS project: the connected Snoocle folder — edit files there; you cannot compile
Swift, so finish iOS tasks by (1) updating SnoocleSelfTests and (2) STOPPING with
a checklist for me to build & run in Xcode. Do not touch files outside the tasks'
"Touch" lists. If acceptance fails twice, stop and tell me to escalate the model
per §13.1. Use model: [Haiku|Sonnet|Opus] per the plan.
```

---

*Marker legend: `// MARKER(agentic-home)` = touchpoints to revisit when Snoocle gets its own agent workspace (or a shared personal-projects workspace) instead of CMS Agent.*
