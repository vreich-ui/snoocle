# Build plan: in-process Anthropic reconciler agent + song GUI

This is a complete, self-contained implementation spec. It is written to be executed
in one session by an implementer with no prior context. Follow it in order; every
architectural decision has already been made — do not redesign, do not substitute
technologies, do not skip the tests.

## 0. Context: what Snoocle is and how to work in this repo

Snoocle turns a song (YouTube URL or title+artist) into structured Song JSON
(chords + lyrics + sections) by fusing two evidence sources: chord sheets found on
the web ("candidates") and audio analysis ("MIR": beats, chord timeline, bpm, key).
An LLM "reconciler" merges them. Songs persist in Firestore with immutable versions.

Pipeline: `discover -> acquire (yt-dlp) -> MIR -> reconcile (LLM provider) -> store`
(`snoocle_server/pipeline.py`). Each step also has its own REST endpoint
(`snoocle_server/api.py`) and MCP tool (`snoocle_server/mcp_server.py`).

Key files:

| File | Role |
|---|---|
| `snoocle_server/reconcile/engine.py` | `reconcile()`: builds provider context, validates output against the Song schema, runs repair rounds |
| `snoocle_server/reconcile/providers.py` | LLM providers: `anthropic`, `openai`, `gemini`, `mock`, `agent` (external MCP). Register new providers in `_PROVIDERS` |
| `snoocle_server/schema/song.py` | The Song schema. STRICT: `additionalProperties: false`; validators reject shape/capo-relative chords |
| `snoocle_server/discovery/` | web search backends, page fetch, chord-sheet parser (`candidate_from_text`) |
| `snoocle_server/mir/pipeline.py` | `analyze_audio(path, accuracy)`; `_analyze_windows()` runs windowed analysis with absolute timestamps |
| `snoocle_server/audio/utils.py` | ffmpeg helpers: `probe`, `trim`, `to_analysis_wav` |
| `snoocle_server/config.py` | All settings, env-driven with `SNOOCLE_` prefix |
| `tests/` | pytest; run with `.venv/bin/python -m pytest -q` (create venv: `python -m venv .venv && .venv/bin/pip install -e ".[dev,mir]"`) |

Hard rules (violating any of these is a failed implementation):

1. All existing tests must still pass. Tests that skip without ffmpeg are fine.
2. Never weaken `schema/song.py` validation. Chords are always SOUNDING pitch;
   `displayPreferences.capo` is forced to 0 by the engine.
3. Do not change the default `SNOOCLE_LLM_PROVIDER` or the behavior of existing
   providers.
4. Do not modify `mir/beats.py` fallback logic (a numpy-2 crash was fixed there;
   `float(np.atleast_1d(tempo)[0])` is intentional).
5. No new build toolchains: the GUI is dependency-free static files (no npm, no
   React, no bundler). The server is the only deployable.
6. Follow the existing code style: module docstrings explaining intent, comments
   only for non-obvious constraints, small pure functions, tests colocated in
   `tests/` mirroring existing patterns.

---

## Workstream A: `anthropic-agent` reconciliation provider

An in-process agentic loop using the Anthropic Python SDK: Claude searches the web,
fetches chord sheets, optionally requests extra audio analysis windows, and emits
the final Song JSON. This runs inside the Cloud Run container (no external agent
service, no MCP hop).

### A1. Dependency

Add `anthropic>=0.66` to `[project.dependencies]` in `pyproject.toml`. (The Docker
image already installs `anthropic`; this makes it explicit for dev installs.)
Import the SDK lazily inside methods, like other providers do with their deps.

### A2. Config (`config.py`, in the "LLM reconciliation" section)

```python
# --- in-process Anthropic agent (provider "anthropic-agent") ---
# Agentic reconciliation inside this server: Claude + server-side web search
# + local tools (chord-sheet fetch/parse, windowed MIR). Uses the same
# SNOOCLE_ANTHROPIC_API_KEY as the plain "anthropic" provider.
anthropic_agent_model: str = "claude-opus-4-8"
anthropic_agent_max_turns: int = 20  # hard cap on agent loop iterations
```

Model resolution order inside the provider: explicit `model` arg ->
`settings.llm_model` -> `settings.anthropic_agent_model`. NEVER invent model IDs.
Valid IDs the user may configure: `claude-opus-4-8`, `claude-sonnet-5`,
`claude-haiku-4-5`.

### A3. Provider (`snoocle_server/reconcile/anthropic_agent.py`, new file)

Class `AnthropicAgentProvider(LLMProvider)`:

- `name = "anthropic-agent"`, `default_model = "claude-opus-4-8"`,
  `supports_audio = False`, `wants_context = True`.
- The engine injects `self.context` (dict) before calling `complete()`. Keys:
  `title, artist, song_id, youtube_video_id, media_url, candidates, mir,
  song_schema` — and after change A5 below, `audio_path`.
- `complete(system, turns, model=None, max_tokens=None, audio=None) -> LLMResponse`
  runs the agent loop and returns `LLMResponse(text=<final text>, provider=self.name,
  model=<resolved model>, usage=<accumulated {"input_tokens": .., "output_tokens": ..}>)`.
  The final text MUST be (or contain) the Song JSON — the engine's `extract_json`
  + schema validation + repair loop handle the rest, exactly as for other providers.

**Conversation state across repair rounds.** The engine calls `complete()` once per
attempt on the SAME provider instance, passing `turns` =
`[user, assistant, repair-user, ...]`. Maintain `self._messages` (the real
Anthropic-format conversation including tool_use/tool_result blocks):

- First attempt (`len(turns) == 1`): build fresh messages from context (below).
- Repair attempt (`len(turns) >= 3`): append
  `{"role": "user", "content": turns[-1]["text"]}` to the existing
  `self._messages` (the assistant's previous final answer is already there from
  the prior loop) and continue the loop. This keeps the full tool history and
  benefits from prompt caching.

**System prompt** (a module constant). Content requirements:

- Role: expert music transcriber producing Snoocle Song JSON.
- Evidence rules: the MIR chord timeline (from the actual audio) is primary
  evidence for chord ROOT and major/minor QUALITY; text sources are primary for
  lyrics, sections, and chord EXTENSIONS (7th/9th/sus). Sources declaring a capo
  are written N semitones below sounding pitch — transpose before comparing.
- Music theory: prefer readings diatonic to the established key or classically
  explainable (secondary dominants, borrowed iv/bVII, relative major/minor);
  spell enharmonics per key signature (F#m in A major, never Gbm).
- Hard chord rule: every chord symbol is the SOUNDING harmony; never shape/tab
  chords; `displayPreferences.capo` must be 0.
- Tool guidance: search the web for `"<title>" "<artist>" chords` and lyrics; fetch
  the 2-4 most promising pages with `fetch_chord_sheet`; if sources disagree about
  a specific passage and audio evidence is thin there, call `analyze_audio_window`
  for that time range (absolute seconds in the video). Do not call tools you do
  not need; carrying forward the provided candidates alone can be sufficient.
- Output contract: the FINAL message must be exactly one JSON object — the Song —
  validating against the provided schema. No markdown fences, no commentary. The
  schema is strict: only its keys are allowed. Set id/title/artist/youtubeVideoId
  from the request.

Put `cache_control: {"type": "ephemeral"}` on the system block (it is stable) so
loop turns and repair rounds reuse the cached prefix:
`system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]`.

**First user message** (built from context): a JSON payload with `songId, title,
artist, youtubeVideoId, mediaUrl, mir` (use `mir.to_prompt_payload()` if not None),
`candidates` (each `model_dump(exclude_none=True)`), and `songSchema`.

**Tools.** Two custom tools + two Anthropic server-side tools in one `tools` list:

```python
TOOLS = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 6},
    {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 6},
    {
        "name": "fetch_chord_sheet",
        "description": "Fetch a URL and parse it as a chord sheet. Returns a structured candidate (lines with chord placements at sounding pitch, declared key/capo, confidence) or an error if the page has no usable transcription. Call this for chord/tab pages found via web_search; prefer it over web_fetch for chord sites because it parses and capo-normalizes.",
        "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"], "additionalProperties": False},
    },
    {
        "name": "analyze_audio_window",
        "description": "Run audio chord/beat analysis on a specific window of the actual recording. Returns a chord timeline with ABSOLUTE timestamps (seconds in the video), bpm and beats for that window. Use when text sources disagree about a passage and the provided MIR timeline does not cover it. Windows are capped at 60 seconds.",
        "input_schema": {"type": "object", "properties": {"start_seconds": {"type": "number"}, "end_seconds": {"type": "number"}}, "required": ["start_seconds", "end_seconds"], "additionalProperties": False},
    },
]
```

Custom tool implementations (module-level functions, unit-testable):

- `fetch_chord_sheet(url)`: `discovery.fetch.fetch_page(url)` ->
  `discovery.fetch.extract_sheet_text` -> `discovery.service.candidate_from_text
  (text, source_id=f"agent-{n}", url=url)`. Return `candidate.model_dump
  (exclude_none=True)` as JSON text, or `{"error": "..."}` (not an exception) when
  the page fails or is not a plausible sheet.
- `analyze_audio_window(start_seconds, end_seconds)`: requires
  `self.context["audio_path"]`; if absent return `{"error": "no audio available"}`.
  Clamp the window to <= 60s and within track duration, then call the helper from
  change A4 and return `{"chords": [...], "beats": N, "bpm": ...}` with absolute
  timestamps. Wrap all exceptions into `{"error": str(e)}`.

**The loop** (manual agentic loop — NOT the beta tool runner; server tools can
pause turns and the manual loop handles that explicitly):

```python
client = self._create_client()  # anthropic.Anthropic(api_key=settings.anthropic_api_key)
usage = {}
for _ in range(settings.anthropic_agent_max_turns):
    response = client.messages.create(
        model=resolved_model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM_BLOCKS,
        tools=TOOLS,
        messages=self._messages,
    )
    # accumulate response.usage input/output tokens into `usage`
    if response.stop_reason == "refusal":
        raise ProviderError("anthropic-agent: model refused the request")
    if response.stop_reason == "pause_turn":
        self._messages.append({"role": "assistant", "content": response.content})
        continue  # re-send; the API resumes server-side tool work
    if response.stop_reason == "tool_use":
        self._messages.append({"role": "assistant", "content": response.content})
        results = [self._run_tool(block) for block in response.content if block.type == "tool_use"]
        self._messages.append({"role": "user", "content": results})
        continue
    break  # end_turn / max_tokens: fall through with whatever text we have
else:
    raise ProviderError("anthropic-agent: exceeded max turns without a final answer")
self._messages.append({"role": "assistant", "content": response.content})
final_text = "".join(b.text for b in response.content if b.type == "text")
```

Notes: `thinking` blocks must be passed back unchanged in `self._messages`
(appending `response.content` as-is does this). Each `tool_result` is
`{"type": "tool_result", "tool_use_id": block.id, "content": <json string>}`,
with `"is_error": True` when the tool returned an error object. Do NOT set
`temperature`/`top_p`/`top_k` (rejected on these models). Create the client via a
`_create_client()` method so tests can monkeypatch it. If
`settings.anthropic_api_key` is empty, raise
`ProviderError("anthropic-agent: SNOOCLE_ANTHROPIC_API_KEY is not configured")`
before any network call.

### A4. MIR helper (`mir/pipeline.py`)

Add a small public wrapper (do not change existing functions):

```python
def analyze_window(audio_path: str | Path, start: float, end: float) -> MirAnalysis:
    """Windowed analysis with timestamps in the ORIGINAL track's coordinates."""
    duration = probe(audio_path).duration_seconds
    with tempfile.TemporaryDirectory(prefix="snoocle-mir-") as td:
        return _analyze_windows(Path(audio_path), duration, [(max(start, 0.0), min(end, duration))], td)
```

### A5. Engine change (`reconcile/engine.py`)

In `reconcile()`, where `provider.context` is set for `wants_context` providers,
add `"audio_path": audio_path` to the context dict. Nothing else changes.

### A6. Registration and capabilities (`reconcile/providers.py`)

- Add `"anthropic-agent": AnthropicAgentProvider` to `_PROVIDERS` (import from the
  new module).
- `provider_capabilities()` must report it: configured when
  `settings.anthropic_api_key` is non-empty. Update `config.provider_key()` with
  `"anthropic-agent": self.anthropic_api_key`.
- `api.py` docstrings/comments that enumerate providers: add `anthropic-agent`.

### A7. Tests (`tests/test_anthropic_agent.py`, new file)

Build a fake Anthropic client (no network, no real `anthropic` objects needed —
simple namespace objects with `.type/.text/.id/.name/.input/.stop_reason/.content/
.usage` attributes; `types.SimpleNamespace` works). Monkeypatch
`AnthropicAgentProvider._create_client`. Script these scenarios:

1. **Happy path**: response 1 = `stop_reason "tool_use"` calling
   `fetch_chord_sheet` with a URL; response 2 = `stop_reason "end_turn"` whose text
   is a valid Song JSON (copy the `_SONG` fixture from `tests/fake_agent_mcp.py`).
   Assert: `reconcile(..., provider_name="anthropic-agent", mir=<fixture>)` returns
   a validated song; the fake fetch tool was invoked; the tool_result went back in
   a user message; `result.provider == "anthropic-agent"`.
2. **Repair round**: response 2's text is `{"bad": true}`; response 3 (after the
   engine's repair prompt) is the valid Song. Assert `result.attempts == 2` and the
   provider's message list grew (same conversation continued).
3. **analyze_audio_window tool**: context without `audio_path` -> tool returns an
   error object marked `is_error`, loop still completes.
4. **Unconfigured key** -> `ProviderError` naming `SNOOCLE_ANTHROPIC_API_KEY`.
5. **Max turns exceeded** (fake always returns tool_use) -> `ProviderError`.
6. `provider_capabilities()["anthropic-agent"]["configured"]` flips with the key.

For scenario 1, monkeypatch `snoocle_server.reconcile.anthropic_agent.fetch_page`
(or the module's fetch function) to return a fixture chord sheet
(`tests/fixtures/sheet_over_lyrics.txt`) so no network is touched.

---

## Workstream B: song GUI (add / edit / browse; play-along groundwork)

A dependency-free single-page app served by the existing FastAPI server. Vanilla
HTML/CSS/JS, no build step, no external CDNs (the page must work offline behind
the bearer token).

### B1. Serving (`api.py` + packaging)

- Create `snoocle_server/ui/` with `index.html`, `app.js`, `style.css`.
- Mount AFTER all API routes:
  `app.mount("/ui", StaticFiles(directory=str(Path(__file__).parent / "ui"), html=True), name="ui")`
  and add `@app.get("/")` returning `RedirectResponse("/ui/")`.
- Packaging (REQUIRED — the Docker image runs the installed package, not the
  source tree): in `pyproject.toml` add

  ```toml
  [tool.setuptools.package-data]
  snoocle_server = ["ui/*"]
  ```

- Auth middleware: `_BearerTokenMiddleware` currently exempts only `/healthz`.
  Also exempt `/` and paths starting with `/ui` (static shell only — every API
  call still requires the token). Add a test in `tests/test_auth_token.py`:
  with the token enabled, `GET /ui/` is 200 and `GET /v1/songs` without token
  is still 401.

### B2. App structure and behavior

Single-page app with a left sidebar (song list + "Add song" button) and a main
panel with three tabs per song: **Edit**, **Versions**, **Play**.

**API access**: one `api(path, options)` fetch wrapper. If `localStorage.snoocleToken`
is set, send `Authorization: Bearer <token>`. On any 401, show a token prompt
modal, store the value, retry once. All endpoints are same-origin.

**Song list**: `GET /v1/songs` -> clickable list of song ids. Refresh button.

**Add song** (modal form):
- Fields: YouTube URL or ID (optional), title (optional), artist (optional),
  accuracy select (`fast` / `standard` / `thorough`, default standard), provider
  select (populated from `GET /v1/providers`; default = server default: leave
  unset). Require either URL or title+artist (mirror the API's own validation).
- Submit -> `POST /v1/songs/analyze` with body
  `{youtubeUrlOrId?, title?, artist?, accuracy?, provider?}`. This request can run
  for minutes: show an in-progress panel ("building — this can take a few
  minutes") and set no client timeout. On success show the per-step report
  (`steps` object) and open the song. On failure (HTTP 502) display the `detail`
  string verbatim — it contains the diagnostic `[steps: ...]` breakdown.

**Edit tab**:
- Load `GET /v1/songs/{id}`; keep the loaded version for optimistic locking
  (`GET /v1/songs/{id}/versions`, first entry = current).
- Metadata inputs: title, artist, bpm, key.
- Lines editor: a `<textarea>` in **inline bracket format**, one line per Song
  line: chord placements rendered as `[C]` inserted at `charIndex` into the
  lyric text (e.g. `[C]When I find myself in [G]times of trouble`). Write two
  pure JS functions in `app.js`: `linesToBracketText(lines)` and
  `bracketTextToLines(text)` (charIndex = index in the lyric string AFTER
  removing brackets; chords at a position beyond the lyric length clamp to the
  end; a line with only chords and no lyrics is allowed — empty lyrics string).
  These two functions must round-trip: `bracketTextToLines(linesToBracketText(x))`
  deep-equals `x` for well-formed input.
- Sections editor: simple table of `{name, startLineIndex}` rows (add/remove).
  Preserve any other section fields the loaded song had by merging edits onto the
  loaded objects rather than rebuilding them.
- Save: build the updated Song by merging edits onto the LOADED song object
  (never drop unknown fields — the schema is strict but the loaded object is
  authoritative), then `POST /v1/songs/{id}` with
  `{song, message: "Edited in UI", expectedVersion: <loaded version>}`.
  On 409 show "someone else saved first — reload". On 400 show the validation
  detail. On success reload the song and versions.

**Versions tab**: list versions (`GET /v1/songs/{id}/versions`); selecting two
shows `GET /v1/songs/{id}/diff?a=..&b=..` in a `<pre>`.

**Play tab** (play-along groundwork — keep deliberately minimal):
- If `song.audio.youtubeVideoId` exists, embed
  `https://www.youtube-nocookie.com/embed/<id>` in an `<iframe>` (this is the ONE
  allowed external resource, and only on this tab).
- Below it render the chord sheet read-only: section headings; each line with
  chords positioned above lyrics (monospace font; build the chord line by
  padding spaces to each `charIndex`).
- Add (but do not fully implement) an `autoScrollTo(seconds)` JS stub with a
  comment: future play-along will map playback time -> line via the chord
  timeline in provenance/syncMap. No timer logic now.

**Style**: one small stylesheet, system font stack, dark-friendly via
`prefers-color-scheme`. No frameworks.

### B3. GUI tests (`tests/test_ui.py`, new file)

Server-side only (no browser automation):

1. `GET /` redirects to `/ui/`; `GET /ui/` returns 200 with `text/html` and the
   string `app.js`.
2. Static assets served: `GET /ui/app.js` 200.
3. Auth: with `SNOOCLE_API_TOKEN` set (monkeypatch settings), `/ui/` is 200
   without a token; `/v1/songs` is 401 without and 200 with the token.
4. Round-trip of the bracket format: replicate `linesToBracketText` /
   `bracketTextToLines` in the test by executing the pure JS? NO — instead keep
   the two functions pure and ALSO expose the same logic as two small Python
   helpers is overkill. Test the JS by construction: include in `app.js` a
   deterministic, dependency-free implementation and add representative
   round-trip cases as comments; the enforced Python-side test is: saving a Song
   whose lines came from the documented bracket examples in this spec validates
   against the schema. Concretely: in the test, parse
   `"[C]When I [G]find"` by the SAME rules (a 10-line reference implementation in
   the test) into `{lyrics: "When I find", chordPlacements: [{charIndex: 0,
   chord: "C"}, {charIndex: 7, chord: "G"}]}` and assert
   `Song.model_validate` accepts a song built with it. This pins the format.

---

## Execution order, commits, acceptance

1. Workstream A (provider + tests). Run the full suite; everything green.
   Commit: `Add anthropic-agent provider: in-process agentic reconciliation`.
2. Workstream B (GUI + tests). Full suite green.
   Commit: `Add song GUI: browse, add, edit, versions, play tab`.
3. Do not modify unrelated files. Do not reformat existing code.

Acceptance checklist (all must hold):

- [ ] `.venv/bin/python -m pytest -q` — all tests pass (existing + new).
- [ ] `reconcile(..., provider_name="anthropic-agent")` works end-to-end against
      the fake client in tests; repair rounds continue the same conversation.
- [ ] `provider_capabilities()` includes `anthropic-agent`.
- [ ] `GET /ui/` serves the app; `/` redirects; package-data configured so the
      files ship in the wheel (verify: `python -c "import snoocle_server,
      pathlib; p = pathlib.Path(snoocle_server.__file__).parent/'ui'/'index.html';
      print(p.exists())"` after `pip install .` into a temp venv — or at minimum
      the package-data stanza exists and paths match).
- [ ] With `SNOOCLE_API_TOKEN` set, the UI shell loads without a token but every
      API call requires one.
- [ ] No new runtime dependencies beyond `anthropic`; no JS dependencies at all.

Deploy notes (for the human, not the implementer): rebuild the image; set
`SNOOCLE_LLM_PROVIDER=anthropic-agent` and `SNOOCLE_ANTHROPIC_API_KEY` on Cloud
Run to make the new provider primary (the CMS chain remains available as
`provider="agent"` per request); the GUI is then at `https://<service-url>/ui/`.
