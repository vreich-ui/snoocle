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
export PROJECT_ID=<your-project-id>          # from the YAML: numeric id 99287560712
export REGION=europe-west1                    # matches your existing service
export REPO=snoocle
gcloud config set project "$PROJECT_ID"
```

## 1. One-time project setup

```sh
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
    secretmanager.googleapis.com storage.googleapis.com

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
    --add-volume=name=data,type=cloud-storage,bucket="${PROJECT_ID}-snoocle-data" \
    --add-volume-mount=volume=data,mount-path=/data \
    --set-env-vars="SNOOCLE_STORE_DIR=/data/songstore,SNOOCLE_AUDIO_CACHE_DIR=/data/audio-cache" \
    --set-secrets="SNOOCLE_ANTHROPIC_API_KEY=snoocle-anthropic-key:latest,SNOOCLE_OPENAI_API_KEY=snoocle-openai-key:latest,SNOOCLE_GEMINI_API_KEY=snoocle-gemini-key:latest"
```

**`--concurrency=1` is not a typo.** `GitSongStore`'s write lock
(`fcntl.flock` in `snoocle_server/store/gitstore.py`) protects concurrent
writers on a normal filesystem, but **`flock()` is not reliably supported
over a `gcsfuse` mount** — GCS FUSE implements only a subset of POSIX and
advisory locking is not part of it. With `--concurrency=1`, Cloud Run only
ever hands one request at a time to the container, which sidesteps the
problem entirely rather than depending on a lock that may silently no-op.
If you need to serve concurrent requests, either (a) raise concurrency but
accept that overlapping `analyze_and_store_song` calls could race on the
store, or (b) swap `GitSongStore`'s lock for a GCS-native one (conditional
writes via object generation preconditions) — not implemented, flagged here
rather than shipped untested.

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
    --add-volume=name=data,type=cloud-storage,bucket="${PROJECT_ID}-snoocle-data" \
    --add-volume-mount=volume=data,mount-path=/data \
    --set-env-vars="SNOOCLE_STORE_DIR=/data/songstore,SNOOCLE_AUDIO_CACHE_DIR=/data/audio-cache,SNOOCLE_MCP_TRANSPORT=streamable-http" \
    --set-secrets="SNOOCLE_ANTHROPIC_API_KEY=snoocle-anthropic-key:latest,SNOOCLE_OPENAI_API_KEY=snoocle-openai-key:latest,SNOOCLE_GEMINI_API_KEY=snoocle-gemini-key:latest"
```

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

- **GCS FUSE + `flock` concurrency risk** (see step 4) — mitigated with
  `--concurrency=1` for now, not fixed at the code level.
- **madmom is not in the runtime image** (the Dockerfile excludes it — heavy
  native build). The deployed beat engine is the librosa fallback, not the
  one used in this session's local acceptance run.
- **No automated re-deploy pipeline** — these are one-shot `gcloud` commands;
  wire up Cloud Build triggers or GitHub Actions if you want push-to-deploy.
