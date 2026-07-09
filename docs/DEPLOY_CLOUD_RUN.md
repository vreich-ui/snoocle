# Deploying Snoocle server to Cloud Run

Two Cloud Run services share one container image, deployed with different
startup commands:

| Service | Runs | Transport | Auth |
|---|---|---|---|
| `snoocle-api`  | `uvicorn snoocle_server.api:app` (image default `CMD`) | HTTP (FastAPI) | Cloud Run IAM |
| `snoocle-mcp`  | `snoocle-mcp` with `SNOOCLE_MCP_TRANSPORT=streamable-http` | MCP streamable-HTTP | Cloud Run IAM |

Both are **private** (`--no-allow-unauthenticated`) — every request must carry
a Google-signed identity token for a principal you've explicitly granted
`roles/run.invoker`. This is deliberate: the service can trigger YouTube
downloads and spend your LLM API budget on request, and the original brief
is explicit that server-side YouTube acquisition is personal-use-only until
reconsidered for wider exposure. IAM auth keeps that posture without any
app-level auth code.

**I don't have `gcloud` or credentials for your GCP project in this
environment** — everything below is a runbook for you (or a CI pipeline with
its own credentials) to execute, not something I ran. Commands assume
`gcloud` is authenticated (`gcloud auth login`) and pointed at your project:

```sh
# PROJECT_ID is the ALPHANUMERIC project id (e.g. "kugelbrands-snoocle"), NOT
# the number. The `99287560712` in your Cloud Run YAML is the project NUMBER;
# it appears in some resource paths but is the wrong value for the
# service-account emails below (`snoocle-run@${PROJECT_ID}.iam...`), which
# require the alphanumeric id. Find it with: `gcloud projects list`.
export PROJECT_ID=<your-alphanumeric-project-id>
export REGION=europe-west1                    # matches your existing service
export REPO=snoocle
gcloud config set project "$PROJECT_ID"
```

## 1. One-time project setup

```sh
# Includes iam (service-account creation, step 1) and cloudbuild (image build,
# step 3) — on a fresh project those commands fail without their APIs enabled.
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
    secretmanager.googleapis.com storage.googleapis.com \
    iam.googleapis.com cloudbuild.googleapis.com

gcloud artifacts repositories create "$REPO" \
    --repository-format=docker --location="$REGION"

# GCS bucket backing the git-versioned song store + audio cache.
# Not multi-region: colocate with the Cloud Run region for latency/cost.
gcloud storage buckets create "gs://${PROJECT_ID}-snoocle-data" \
    --location="$REGION" --uniform-bucket-level-access

# Dedicated service account, least-privilege (narrower than the default
# Compute Engine SA your placeholder service currently runs as).
gcloud iam service-accounts create snoocle-run \
    --display-name="Snoocle Cloud Run runtime"

gcloud storage buckets add-iam-policy-binding "gs://${PROJECT_ID}-snoocle-data" \
    --member="serviceAccount:snoocle-run@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role=roles/storage.objectAdmin
```

## 2. Secrets

```sh
printf '%s' "$ANTHROPIC_API_KEY" | gcloud secrets create snoocle-anthropic-key --data-file=-
printf '%s' "$OPENAI_API_KEY"    | gcloud secrets create snoocle-openai-key    --data-file=-
printf '%s' "$GEMINI_API_KEY"    | gcloud secrets create snoocle-gemini-key    --data-file=-

for s in snoocle-anthropic-key snoocle-openai-key snoocle-gemini-key; do
  gcloud secrets add-iam-policy-binding "$s" \
      --member="serviceAccount:snoocle-run@${PROJECT_ID}.iam.gserviceaccount.com" \
      --role=roles/secretmanager.secretAccessor
done
```

## 3. Build and push the image

Reuses the `Dockerfile` already on `main` — no changes needed for either
service; they differ only by the Cloud Run `--command`/env override in step 4.

```sh
gcloud builds submit --tag "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/snoocle:latest" .
```

## 4. Deploy — `snoocle-api`

```sh
gcloud run deploy snoocle-api \
    --project="$PROJECT_ID" --region="$REGION" \
    --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/snoocle:latest" \
    --service-account="snoocle-run@${PROJECT_ID}.iam.gserviceaccount.com" \
    --no-allow-unauthenticated \
    --execution-environment=gen2 \
    --cpu=1 --memory=2Gi --timeout=900 --concurrency=1 \
    --min-instances=0 --max-instances=1 \
    --add-volume=name=data,type=cloud-storage,bucket="${PROJECT_ID}-snoocle-data",mount-options="uid=10001;gid=10001;file-mode=0640;dir-mode=0750;metadata-cache-ttl-secs=0" \
    --add-volume-mount=volume=data,mount-path=/data \
    --set-env-vars="SNOOCLE_STORE_DIR=/data/songstore,SNOOCLE_AUDIO_CACHE_DIR=/data/audio-cache" \
    --set-secrets="SNOOCLE_ANTHROPIC_API_KEY=snoocle-anthropic-key:latest,SNOOCLE_OPENAI_API_KEY=snoocle-openai-key:latest,SNOOCLE_GEMINI_API_KEY=snoocle-gemini-key:latest"
```

**The `mount-options` uid/gid are required, not optional.** Cloud Run mounts a
GCS FUSE volume **root-owned** by default, but the image runs as non-root
`appuser` (UID/GID 10001, pinned in the Dockerfile). Without
`uid=10001;gid=10001`, the very first write — `GitSongStore._ensure_repo()`
initializing the store, or the first audio-cache write — fails with permission
denied, so the service starts but can't analyze or store anything. (`file-mode`/
`dir-mode` are needed too because GCS FUSE ignores `chmod`, so the app can't fix
perms at runtime.)

**Sharing one GCS-FUSE git store across two services is fragile — read this
before serving real traffic.** Two independent problems, both rooted in the
two-service topology:

1. **Read staleness.** GCS FUSE caches file metadata; Cloud Run's default
   stat-cache TTL is **60s**. `GitSongStore.current_version()`, `get()`, and
   `list_songs()` all read refs/song files through the mount, so *even with
   sequential (non-concurrent) requests*, the MCP service can read seconds
   after an API write and still see stale state — missing the just-committed
   version. The `metadata-cache-ttl-secs=0` in the mount-options above
   disables that cache (each read re-checks object generation), which
   mitigates this. It costs a metadata round-trip per read — acceptable for a
   low-volume personal store.
2. **Write races.** `GitSongStore`'s `fcntl.flock` (`store/gitstore.py`) is
   **not reliably honored over gcsfuse** (POSIX subset, no advisory locking),
   and `metadata-cache-ttl-secs=0` does **not** help here. `--concurrency=1
   --max-instances=1` serializes writes within *one* service, but the two
   services (`snoocle-api` and `snoocle-mcp`) both write the same store
   (`/v1/songs/analyze`, `POST /v1/songs/{id}`; `analyze_and_store_song`,
   `save_song`), so a write in each container can interleave and corrupt the
   git index/refs — the per-service cap does not serialize across services,
   and no mount-option fixes this.

The mount-option above closes (1). For (2), pick one:

- **(recommended) Single writer.** Deploy only ONE of the two services, or
  serve both surfaces from one Cloud Run service (the FastMCP app mounted
  into the FastAPI app so a single container/port answers both REST and MCP —
  then `--concurrency=1` genuinely serializes every write *and* removes the
  cross-service read-staleness entirely, since there's one mount). Small code
  change, not yet implemented — see the follow-up note at the end.
- **Accept the write-race for personal single-user use.** Two simultaneous
  writes to the *same* song id are what corrupt history; a single-user
  workflow rarely fires an API write and an MCP write concurrently. Fine to
  start here; don't rely on it for anything shared.
- **Implement a GCS-native lock.** Replace `flock` with conditional writes
  via object-generation preconditions (the store already has an
  `expected_version` CAS path to build on). The durable fix; not built yet.

## 5. Deploy — `snoocle-mcp`

Same image, overridden startup command; MCP tool calls are typically
longer-running than plain API calls (audio download + MIR + LLM chained
inside one `analyze_and_store_song` tool call), so the timeout is generous:

```sh
gcloud run deploy snoocle-mcp \
    --project="$PROJECT_ID" --region="$REGION" \
    --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/snoocle:latest" \
    --command=snoocle-mcp \
    --service-account="snoocle-run@${PROJECT_ID}.iam.gserviceaccount.com" \
    --no-allow-unauthenticated \
    --execution-environment=gen2 \
    --cpu=1 --memory=2Gi --timeout=900 --concurrency=1 \
    --min-instances=0 --max-instances=1 \
    --add-volume=name=data,type=cloud-storage,bucket="${PROJECT_ID}-snoocle-data",mount-options="uid=10001;gid=10001;file-mode=0640;dir-mode=0750;metadata-cache-ttl-secs=0" \
    --add-volume-mount=volume=data,mount-path=/data \
    --set-env-vars="SNOOCLE_STORE_DIR=/data/songstore,SNOOCLE_AUDIO_CACHE_DIR=/data/audio-cache,SNOOCLE_MCP_TRANSPORT=streamable-http,SNOOCLE_MCP_TRUST_PROXY=true" \
    --set-secrets="SNOOCLE_ANTHROPIC_API_KEY=snoocle-anthropic-key:latest,SNOOCLE_OPENAI_API_KEY=snoocle-openai-key:latest,SNOOCLE_GEMINI_API_KEY=snoocle-gemini-key:latest"
```

`SNOOCLE_MCP_TRUST_PROXY=true` binds `0.0.0.0` (required so Cloud Run can
route traffic to the container) and disables the MCP SDK's DNS-rebinding host
check — correct **only** because Cloud Run IAM (`--no-allow-unauthenticated`)
authenticates every request before it reaches the container. Without a
remote-serving flag the server binds loopback (`127.0.0.1`) with the host
check ON, so a local HTTP run is never exposed on the LAN; set this flag only
when an authenticating proxy sits in front. (Alternatively, once you know the
assigned hostname, set `SNOOCLE_MCP_ALLOWED_HOSTS=snoocle-mcp-….run.app` — it
also binds `0.0.0.0` but keeps the host check on, scoped to that host.)

Both services point at the **same bucket/mount** — they see the same song
store, so a song analyzed via the API is immediately visible to the MCP
`get_song`/`list_songs` tools and vice versa.

## 6. Grant yourself access

```sh
gcloud run services add-iam-policy-binding snoocle-api --region="$REGION" \
    --member="user:vreich@kugelbrands.com" --role=roles/run.invoker
gcloud run services add-iam-policy-binding snoocle-mcp --region="$REGION" \
    --member="user:vreich@kugelbrands.com" --role=roles/run.invoker
```

## 7. Calling the API

```sh
API_URL=$(gcloud run services describe snoocle-api --region="$REGION" --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token --audiences="$API_URL")

curl -H "Authorization: Bearer $TOKEN" "$API_URL/healthz"

curl -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
    -d '{"title":"Let It Be","artist":"The Beatles"}' \
    "$API_URL/v1/songs/analyze"
```

Identity tokens expire (~1 hour) — re-run `print-identity-token` for a fresh
one rather than hardcoding it anywhere.

## 8. Connecting an MCP client

```sh
MCP_URL=$(gcloud run services describe snoocle-mcp --region="$REGION" --format='value(status.url)')
MCP_TOKEN=$(gcloud auth print-identity-token --audiences="$MCP_URL")
```

Any MCP client that supports the streamable-HTTP transport with custom
headers can connect to `"$MCP_URL/mcp"` sending
`Authorization: Bearer $MCP_TOKEN`. With the Python SDK directly:

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client(
    f"{MCP_URL}/mcp", headers={"Authorization": f"Bearer {MCP_TOKEN}"}
) as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        ...
```

For a GUI client (Claude Desktop, etc.) whose remote-MCP config supports a
`headers` field alongside `url`, pass the same Bearer header there. As with
the API, the token expires in ~1 hour — for anything longer-lived, a small
wrapper that refreshes the token per-connection is needed; not built here
since this is a personal-use, manually-invoked tool per the brief, not a
persistent multi-user integration.

**For purely local use** (no Cloud Run involvement, no IAM token dance):
`snoocle-mcp` still defaults to stdio — run it as a subprocess from your own
machine's MCP client config exactly as before; only the deployed variant
needs any of this.

## Known gaps / follow-ups

- **Two services sharing one GCS-FUSE git store** (see step 4). Read
  staleness (60s stat-cache TTL) is mitigated by `metadata-cache-ttl-secs=0`
  in the mount-options. The write race is NOT: per-service `--concurrency=1`
  doesn't serialize across the two services, and `flock` doesn't work over
  GCS FUSE. Real fixes: serve both surfaces from a single Cloud Run service
  (mount the FastMCP ASGI app into the FastAPI app — small code change, not
  yet done), or a GCS-native conditional-write lock in `GitSongStore`. For
  personal single-user use the practical write-race window is small; don't
  rely on the two-service topology for shared use.
- **madmom is not in the runtime image** (the Dockerfile excludes it — heavy
  native build). The deployed beat engine is the librosa fallback, not the
  one used in this session's local acceptance run.
- **No automated re-deploy pipeline** — these are one-shot `gcloud` commands;
  wire up Cloud Build triggers or GitHub Actions if you want push-to-deploy.
