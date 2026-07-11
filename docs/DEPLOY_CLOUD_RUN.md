# Deploying Snoocle server to Cloud Run

**One** Cloud Run service runs the combined app: the FastAPI REST API plus the
MCP streamable-HTTP transport, served at `/mcp` on the same service. A single
container/process is therefore the **sole writer** to the git store — which
fully serializes writes (no cross-service race) and removes any cross-mount
read-staleness.

| Path | Surface | Auth |
|---|---|---|
| `/healthz`, `/v1/...` | REST API (FastAPI) | Cloud Run IAM |
| `/mcp` | MCP streamable-HTTP (stateless) | Cloud Run IAM |

The service is **private** (`--no-allow-unauthenticated`) — every request must
carry a Google-signed identity token for a principal you've explicitly granted
`roles/run.invoker`. This is deliberate: the service can trigger YouTube
downloads and spend your LLM API budget on request, and the original brief is
explicit that server-side YouTube acquisition is personal-use-only until
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
# Includes iam (service-account creation, this step) and cloudbuild (image
# build, step 3) — on a fresh project those commands fail without their APIs
# enabled.
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

Reuses the `Dockerfile` already on `main` — its default `CMD`
(`uvicorn snoocle_server.api:app`) is exactly the combined app.

```sh
gcloud builds submit --tag "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/snoocle:latest" .
```

## 4. Deploy the (single) `snoocle` service

```sh
gcloud run deploy snoocle \
    --project="$PROJECT_ID" --region="$REGION" \
    --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/snoocle:latest" \
    --service-account="snoocle-run@${PROJECT_ID}.iam.gserviceaccount.com" \
    --no-allow-unauthenticated \
    --execution-environment=gen2 \
    --cpu=1 --memory=2Gi --timeout=900 --concurrency=1 \
    --min-instances=0 --max-instances=1 \
    --add-volume=name=data,type=cloud-storage,bucket="${PROJECT_ID}-snoocle-data",mount-options="uid=10001;gid=10001;file-mode=0640;dir-mode=0750;metadata-cache-ttl-secs=0" \
    --add-volume-mount=volume=data,mount-path=/data \
    --set-env-vars="SNOOCLE_STORE_DIR=/data/songstore,SNOOCLE_AUDIO_CACHE_DIR=/data/audio-cache,SNOOCLE_MCP_TRUST_PROXY=true" \
    --set-secrets="SNOOCLE_ANTHROPIC_API_KEY=snoocle-anthropic-key:latest,SNOOCLE_OPENAI_API_KEY=snoocle-openai-key:latest,SNOOCLE_GEMINI_API_KEY=snoocle-gemini-key:latest"
```

The default container `CMD` runs `uvicorn snoocle_server.api:app`, which serves
the REST API and mounts the MCP transport at `/mcp` (embedded in the same ASGI
app, one lifespan). `--command`/`--args` are **not** overridden — there is only
one service.

**`--concurrency=1` is genuinely safe here.** Because this one service is the
only writer to the store, the concurrency cap serializes *every* write — API
and MCP alike — with no cross-service interleaving. Combined with the MCP
transport running **stateless** (no persistent GET SSE stream that would
otherwise occupy the single request slot and deadlock tool-call POSTs;
`SNOOCLE_MCP_STATELESS` defaults true), `--concurrency=1` is both correct and
non-deadlocking. `GitSongStore`'s `fcntl.flock` — unreliable over gcsfuse — is
no longer load-bearing, since the concurrency cap does the serialization.

**`SNOOCLE_MCP_TRUST_PROXY=true`** disables the MCP DNS-rebinding host check on
the `/mcp` route — correct **only** because Cloud Run IAM
(`--no-allow-unauthenticated`) authenticates every request before it reaches
the container, and because the co-located REST routes have no such check
either: the whole service's exposure is governed uniformly at the IAM/bind
edge, not per-route. Locally (no flag), the `/mcp` host check stays on with a
localhost allowlist; bind uvicorn to `127.0.0.1` for local runs.

**The `mount-options` uid/gid are required, not optional.** Cloud Run mounts a
GCS FUSE volume **root-owned** by default, but the image runs as non-root
`appuser` (UID/GID 10001, pinned in the Dockerfile). Without
`uid=10001;gid=10001`, the very first write — `GitSongStore._ensure_repo()`
initializing the store, or the first audio-cache write — fails with permission
denied, so the service starts but can't analyze or store anything. (`file-mode`/
`dir-mode` are needed too because GCS FUSE ignores `chmod`, so the app can't fix
perms at runtime.) `metadata-cache-ttl-secs=0` is belt-and-suspenders now that
one process owns the mount (it forces a generation re-check per read); harmless
to keep.

## 5. Grant yourself access

```sh
gcloud run services add-iam-policy-binding snoocle --region="$REGION" \
    --member="user:vreich@kugelbrands.com" --role=roles/run.invoker
```

## 6. Calling the REST API

```sh
URL=$(gcloud run services describe snoocle --region="$REGION" --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token --audiences="$URL")

curl -H "Authorization: Bearer $TOKEN" "$URL/healthz"

curl -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
    -d '{"title":"Let It Be","artist":"The Beatles"}' \
    "$URL/v1/songs/analyze"
```

Identity tokens expire (~1 hour) — re-run `print-identity-token` for a fresh
one rather than hardcoding it anywhere.

## 7. Connecting an MCP client

The MCP endpoint is `/mcp` on the **same** service URL:

```sh
URL=$(gcloud run services describe snoocle --region="$REGION" --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token --audiences="$URL")
```

Any MCP client that supports the streamable-HTTP transport with custom headers
can connect to `"$URL/mcp"` sending `Authorization: Bearer $TOKEN`. With the
Python SDK directly:

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client(
    f"{URL}/mcp", headers={"Authorization": f"Bearer {TOKEN}"}
) as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        ...
```

For a GUI client (Claude Desktop, etc.) whose remote-MCP config supports a
`headers` field alongside `url`, pass the same Bearer header there. As with the
API, the token expires in ~1 hour — for anything longer-lived, a small wrapper
that refreshes the token per-connection is needed; not built here since this is
a personal-use, manually-invoked tool per the brief, not a persistent
multi-user integration.

**For purely local use** (no Cloud Run, no IAM token dance): `snoocle-mcp` still
defaults to **stdio** — run it as a subprocess from your own machine's MCP
client config. Or run the combined app locally with `uvicorn
snoocle_server.api:app` and point an MCP client at `http://127.0.0.1:8000/mcp`.

## Known gaps / follow-ups

- **`flock` over gcsfuse is still unreliable** — but it is no longer relied on:
  the single-service `--concurrency=1` serializes writes. If you ever run
  `--max-instances>1` or `--concurrency>1`, you'd reintroduce a write race and
  should implement a GCS-native conditional-write lock in `GitSongStore` (it
  already has an `expected_version` CAS path to build on) first.
- **madmom is not in the runtime image** (the Dockerfile excludes it — heavy
  native build). The deployed beat engine is the librosa fallback, not the one
  used in this session's local acceptance run.
- **No automated re-deploy pipeline** — these are one-shot `gcloud` commands;
  wire up Cloud Build triggers or GitHub Actions if you want push-to-deploy.
