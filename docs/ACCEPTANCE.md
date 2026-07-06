# Acceptance test report

- Run: 2026-07-06T21:42:27+00:00 — mode: offline (fixtures)
- Command: `scripts/acceptance.py --offline`
Summary: 1x BLOCKED, 1x PARTIAL, 5x PASS

## Step 1: title+artist -> video -> audio -> schema-compliant JSON
**PARTIAL**

- live YouTube search+acquire (no URL): HTTP 502 — {'detail': 'YouTube search failed for \'The Beatles Let It Be\': ERROR: query "The Beatles Let It Be" page 1: Unable to download API page: (\'Unable to connect to proxy\', OSError(\'Tunnel connection 
- offline fallback: cache-seeded synthetic recording as video ZZacceptZZ0
- POST /v1/songs/analyze: HTTP 200, steps={'discover': 'ok: 3 candidate source(s)', 'acquire': 'ok: ZZacceptZZ0 (Acceptance Song [ZZacceptZZ0])', 'mir': "ok: engines={'beats': 'madmom', 'chords': 'chroma-template-fallback', 'structure': 'librosa-agglomerative-fallback'}", 'reconcile': 'ok: provider=mock model=mock-reconciler-v1 attempts=1', 'store': 'ok: version 52019163375f'}
- produced JSON contains: {'chords': True, 'lyrics': True, 'sections': True, 'mirTimestamps': True}

## Step 2: reconciliation on >=2 of 3 LLM providers, same input, all sources used
**BLOCKED**

- discovery for shared input: 3 candidates
- provider anthropic: HTTP 502 — anthropic: "Could not resolve authentication method. Expected one of api_key, auth_token, or credentials to be set. Or for one of the `X-Api-Key` or `Authorization` headers to be e
- provider gemini: HTTP 502 — gemini: 403 {
  "error": {
    "code": 403,
    "message": "Method doesn't allow unregistered callers (callers without established identity). Please use API Key or other form of AP
- provider openai: HTTP 502 — openai: connection failed: 403 Forbidden
- mock provider (offline stand-in): HTTP 200, attempts=1
- offline evidence for multi-source use + identical cross-provider input: tests/test_provider_parity.py (3 passed)

## Step 3: re-run creates new committed version; prior preserved and git-diffable
**PASS**

- run A stored f1b59ce42493, run B stored 03f2801fd79e (distinct=True)
- versions endpoint lists 3 versions; both runs present=True
- git diff between runs: 47 lines (rc=0)
- prior version still retrievable: HTTP 200

## Step 4: output validates against Song schema; no capo'd/shape chord stored
**PASS**

- stored JSON validates against Song schema (schemaVersion=1)
- spot check: 32 chord placements, all parse as sounding harmonies, shape-like identities found: 0
- chord vocabulary: ['Am', 'C', 'F', 'G']

## Step 5: whole pipeline callable via curl
**PASS**

- every call in this run was made through `curl` against the live server (http://127.0.0.1:41875), no iOS app involved
- e2e pipeline call: POST /v1/songs/analyze -> HTTP 200 with stored version 03f2801fd79e

## Step 6: MCP wrapper callable from MCP client; distinct per-step tools
**PASS**

- real MCP client over stdio (official `mcp` SDK): 1 passed in 1.52s
- verifies: 16 distinct step-scoped tools listed, server_status + trim_audio (base64 round-trip) + get_song_schema callable

## Step 7: deterministic audio utilities (convert, trim) work on a sample file
**PASS**

- convert wav->mp3 via curl: codec=mp3, duration=6.034286
- trim 1.0-3.5s via curl: duration=2.5 (expected 2.5)
- no AI call anywhere on these paths (ffmpeg only)

## Re-running with live network + API keys

PARTIAL/BLOCKED above are environment constraints (YouTube + general web
egress blocked by network policy; no LLM API keys), not code gaps — every
blocked call fails with the recorded upstream reason. On a machine with
open egress and keys:

```sh
export SNOOCLE_ANTHROPIC_API_KEY=... SNOOCLE_GEMINI_API_KEY=... SNOOCLE_OPENAI_API_KEY=...
.venv/bin/python scripts/acceptance.py --providers anthropic,gemini,openai \
    --title 'Let It Be' --artist 'The Beatles'
```

That exercises: real YouTube search+download (step 1 -> PASS expected),
real web-search discovery, and live reconciliation on all three providers
(step 2 -> PASS with >=2 succeeding).

