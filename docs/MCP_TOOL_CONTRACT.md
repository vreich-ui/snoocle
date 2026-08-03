# MCP tool classification and presentation contract

Snoocle publishes one versioned machine-readable contract for every registered
MCP tool. GUI Studio and other clients can read it in either place on current
MCP 1.x releases:

- `tools/list`: `_meta["snoocle/toolContract"]`
- `list_capabilities`: each tool's `toolContract` field

`list_capabilities` is the compatibility channel on MCP 1.x SDKs whose FastMCP
tool model does not expose `_meta`. MCP 1.10's model predates tool `_meta`, so
the contract attachment does not assume that field exists; standard annotations
remain attached on both tool-model shapes.

The contract is derived by iterating FastMCP's registered tools and joining each
one to the exhaustive classification in `snoocle_server/tool_contract.py`. The
server fails during import, and the test suite fails in CI, if a registered tool
has no classification or if a stale classification names a removed tool.

## Standard MCP annotations

Each tool also sets MCP `title`, `readOnlyHint`, `destructiveHint`,
`idempotentHint`, and `openWorldHint`. Clients should use those standard hints
where they are sufficient. They remain hints as defined by MCP; server-side
authorization and validation continue to enforce actual access.

## Snoocle metadata

The namespaced object adds only fields not represented by standard MCP tool
annotations:

```json
{
  "schemaVersion": 1,
  "category": "alignment",
  "browserSafety": "server_filesystem_restricted",
  "inputArtifactKinds": ["song", "audio_file", "mir_analysis"],
  "outputArtifactKinds": ["song", "run_trace", "quality_report"],
  "access": {
    "mode": "write",
    "readOnly": false,
    "destructive": false,
    "idempotent": false
  },
  "networkAccess": ["external:youtube", "store_backend"],
  "modelUse": "none",
  "persistence": ["run_trace_write", "song_version_optional", "cache"],
  "cacheBehavior": "read_write",
  "expectedDuration": "minutes",
  "specializedRenderer": "song"
}
```

`browserSafety` is one of:

- `safe`: suitable for direct browser invocation under the normal authenticated
  MCP session.
- `confirmation_required`: has persistent effects, model cost, or another
  deliberate side effect a GUI should confirm.
- `server_filesystem_restricted`: accepts a server-local path. A browser must
  not be allowed to supply arbitrary paths. Current audio tools also accept an
  opaque `audio_ref` created by the authenticated artifact API; `audio_path`
  remains only for trusted local MCP callers and backwards compatibility.

`expectedDuration` is a coarse UI scheduling hint (`instant`, `seconds`, or
`minutes`), not a timeout. `specializedRenderer` selects a richer viewer when
available; clients can always fall back to JSON.

The pre-existing `list_capabilities` fields remain present for compatibility.
Its additional flattened fields and nested `toolContract` are sourced from the
same classification, so there is no second behavior table to drift.
