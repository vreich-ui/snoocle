# Deploying Snoocle server to Cloud Run

**One** Cloud Run service runs the combined app: the FastAPI REST API plus the
MCP streamable-HTTP transport, served at `/mcp` on the same service.

Songs persist in **Firestore (Native mode)** — durable across instance restarts
and horizontally safe (writes use Firestore transactions with optimistic
locking, so concurrent writers can't corrupt version history). The container
filesystem is used only for a disposable audio cache. Auth to Firestore is via
**Application Default Credentials**; the runtime service account needs
`roles/datastore.user`, and the project id comes from `GOOGLE_CLOUD_PROJECT`.

| Path | Surface | Auth |
|---|---|---|
| `/healthz`, `/v1/...` | REST API (FastAPI) | Cloud Run IAM |
| `/mcp` | MCP streamable-HTTP (stateless) | Cloud Run IAM |

By default the service is **private** (`--no-allow-unauthenticated`) — every
request must carry a Google-signed identity token for a principal you've
explicitly granted `roles/run.invoker`. This is deliberate: the service can
trigger YouTube downloads and spend your LLM API budget on request, and the
original brief is explicit that server-side YouTube acquisition is
personal-use-only until reconsidered for wider exposure. IAM auth keeps that
posture without any app-level auth code.

If the rotating-identity-token flow is inconvenient for a client (the iPad app,
a curl script, an MCP client), there is an optional **static bearer token** that
authorizes the REST API and `/mcp` with one fixed value — see
[Optional: a static bearer token](#optional-a-static-bearer-token) below. The
two schemes are mutually exclusive (both live in the `Authorization` header, so
you pick one); the static-token mode trades Google IAM for a shared secret and a
publicly-reachable URL, so use a long random token.

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
# Includes firestore (the song store), iam (service-account creation, this
# step) and cloudbuild (image build, step 3) — on a fresh project those
# commands fail without their APIs enabled.
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
    secretmanager.googleapis.com firestore.googleapis.com \
    iam.googleapis.com cloudbuild.googleapis.com

gcloud artifacts repositories create "$REPO" \
    --repository-format=docker --location="$REGION"

# Firestore in NATIVE mode, colocated with Cloud Run. This is the durable song
# store (collection `songs` + a `versions` subcollection per song). This creates
# the "(default)" database; if it already exists, skip this.
gcloud firestore databases create --location="$REGION"

# To use a NAMED database instead of "(default)" (e.g. snoocle-db), create it
# with --database and point the app at it with the FIRESTORE_DATABASE env var
# (see the deploy step below). The project id (GOOGLE_CLOUD_PROJECT) is separate
# and unchanged — a database id is not a project id.
#   gcloud firestore databases create --database=snoocle-db --location="$REGION" --type=firestore-native

# Dedicated service account, least-privilege (narrower than the default
# Compute Engine SA your placeholder service currently runs as).
gcloud iam service-accounts create snoocle-run \
    --display-name="Snoocle Cloud Run runtime"

# Read/write access to Firestore documents (the song store).
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:snoocle-run@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role=roles/datastore.user
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
    --cpu=1 --memory=2Gi --timeout=3600 --concurrency=4 \
    --min-instances=0 --max-instances=2 \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},SNOOCLE_STORE_BACKEND=firestore,SNOOCLE_MCP_TRUST_PROXY=true" \
    --set-secrets="SNOOCLE_ANTHROPIC_API_KEY=snoocle-anthropic-key:latest,SNOOCLE_OPENAI_API_KEY=snoocle-openai-key:latest,SNOOCLE_GEMINI_API_KEY=snoocle-gemini-key:latest"
```

The default container `CMD` runs `uvicorn snoocle_server.api:app`, which serves
the REST API and mounts the MCP transport at `/mcp` (embedded in the same ASGI
app, one lifespan). `--command`/`--args` are **not** overridden — there is only
one service.

**`--timeout=3600` is required, not optional.** A real analyze
(discover → yt-dlp → MIR chord model → LLM/agent reconcile) takes 2–8 minutes;
Cloud Run's default 300s request timeout would silently kill it mid-flight and
the client would see nothing. Each pipeline step also has its own in-app
timeout (`SNOOCLE_*_TIMEOUT_SECONDS`) so a single stuck step fails loudly (HTTP
502 naming the step) instead of hanging.

**Persistence is Firestore, so writes are safe under concurrency.** Optimistic
locking (`expectedVersion` → a Firestore transaction) means concurrent writers
can't corrupt version history, so `--concurrency` and `--max-instances` no
longer need to be pinned to 1 for correctness (the old git-on-gcsfuse store did
need that). They're kept modest here only because MIR is CPU-bound on
`--cpu=1`; raise them if you add CPU. The MCP transport still runs **stateless**
(`SNOOCLE_MCP_STATELESS` defaults true) so no persistent GET SSE stream occupies
a request slot.

**`SNOOCLE_MCP_TRUST_PROXY=true`** disables the MCP DNS-rebinding host check on
the `/mcp` route — correct **only** because Cloud Run IAM
(`--no-allow-unauthenticated`) authenticates every request before it reaches
the container, and because the co-located REST routes have no such check
either: the whole service's exposure is governed uniformly at the IAM/bind
edge, not per-route. Without it, `/mcp` rejects every request with `Invalid
Host header`. Locally (no flag, non-Docker), the `/mcp` host check stays on
with a localhost allowlist; bind uvicorn to `127.0.0.1` for local runs.

> This value **and `GOOGLE_CLOUD_PROJECT`** are **baked into the image**
> (`Dockerfile` `ENV`), so they survive an automated rebuild/redeploy (e.g. a
> Cloud Build trigger) even if the deploy command doesn't repeat them. A Cloud
> Run service-config env var of the same name still overrides the baked value at
> deploy time, so the baked project id is a durable default, not a lock-in — to
> target a different project, set `GOOGLE_CLOUD_PROJECT` in `--set-env-vars` (or
> edit the `Dockerfile` `ENV`). Passing either in `--set-env-vars` is harmless
> belt-and-suspenders.

## 5. Grant yourself access

```sh
gcloud run services add-iam-policy-binding snoocle --region="$REGION" \
    --member="user:vreich@kugelbrands.com" --role=roles/run.invoker
```

## Optional: a static bearer token

If minting a fresh Google identity token per hour is inconvenient for a client,
set `SNOOCLE_API_TOKEN` and the app enforces a single fixed bearer token on
**both** the REST API and `/mcp` (only `/healthz` stays open, for liveness
probes). This replaces — it does not stack with — Cloud Run IAM: both use the
`Authorization` header, so you deploy the service **public** and let the app
token be the gate.

> **Security note.** This makes the URL reachable by anyone on the internet,
> gated only by the token. The service downloads from YouTube and spends your
> LLM budget, so treat the token like a password: generate a long random one,
> keep it in Secret Manager, and rotate it if it leaks. If you're unsure, keep
> the default IAM posture instead.

```sh
# 1. Generate a strong token and store it in Secret Manager.
python -c "import secrets; print(secrets.token_urlsafe(32))" \
    | gcloud secrets create snoocle-api-token --data-file=-
gcloud secrets add-iam-policy-binding snoocle-api-token \
    --member="serviceAccount:snoocle-run@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role=roles/secretmanager.secretAccessor

# 2. Redeploy PUBLIC, injecting the token (add these flags to the deploy in §4).
gcloud run services update snoocle --region="$REGION" \
    --allow-unauthenticated \
    --update-secrets="SNOOCLE_API_TOKEN=snoocle-api-token:latest"
```

Clients then send the same fixed token to either surface — no `gcloud` needed:

```sh
URL=$(gcloud run services describe snoocle --region="$REGION" --format='value(status.url)')
TOKEN=$(gcloud secrets versions access latest --secret=snoocle-api-token)

curl -H "Authorization: Bearer $TOKEN" "$URL/v1/songs"          # REST
# MCP: pass the same header (see §7) — Authorization: Bearer $TOKEN
```

To go back to IAM-only, `--no-allow-unauthenticated` and drop the secret (unset
`SNOOCLE_API_TOKEN`); the app becomes a pass-through again.

## 6. Calling the REST API

```sh
URL=$(gcloud run services describe snoocle --region="$REGION" --format='value(status.url)')
TOKEN=$(gcloud auth print-identity-token --audiences="$URL")

curl -H "Authorization: Bearer $TOKEN" "$URL/healthz"

curl -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
    -d '{"title":"Let It Be","artist":"The Beatles"}' \
    "$URL/v1/songs/analyze"

# Or give ONLY a URL — title/artist are derived from the video's own metadata:
curl -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
    -d '{"youtubeUrlOrId":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}' \
    "$URL/v1/songs/analyze"
```

`analyze` needs **either** `title`+`artist` **or** `youtubeUrlOrId` (from which it
derives them); neither → `422`.

**iOS read surface** — the app lists songs and downloads each one as Song JSON
(the exact schema `GET /v1/schema/song` returns):

```sh
curl -H "Authorization: Bearer $TOKEN" "$URL/v1/songs"                       # list ids
curl -H "Authorization: Bearer $TOKEN" "$URL/v1/songs/the-beatles--let-it-be" # full Song JSON
curl -H "Authorization: Bearer $TOKEN" "$URL/v1/songs/the-beatles--let-it-be/versions"
```

**Bring your own recording** — upload an audio *or* video file and get MIR pitch
analysis (beats/downbeats, sounding-harmony chord timeline, sections, bpm, key)
with no YouTube step. For a video, the audio track is extracted automatically;
a file with no audio stream returns 422:

```sh
curl -H "Authorization: Bearer $TOKEN" \
    -F "file=@/path/to/song.mp3" \
    "$URL/v1/audio/analyze/upload"

curl -H "Authorization: Bearer $TOKEN" \
    -F "file=@/path/to/clip.mp4" \
    "$URL/v1/audio/analyze/upload"
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
can connect to `"$URL/mcp"` sending `Authorization: Bearer $TOKEN`. (If you
enabled the [static bearer token](#optional-a-static-bearer-token), set
`TOKEN=$(gcloud secrets versions access latest --secret=snoocle-api-token)`
instead — the fixed token doesn't expire, so no per-hour refresh.) With the
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

## YouTube bot-check (server-side acquisition)

YouTube challenges downloads from datacenter IPs (Cloud Run) with *"Sign in to
confirm you're not a bot"*, which surfaces as `502 {"detail":"resolve: …"}` or
`acquire: …`. Two ways to get past it:

1. **Upload the audio instead of pulling from YouTube** — the reliable path, no
   YouTube from the server at all. Download the track on your own machine and
   POST the file:
   ```sh
   curl -H "Authorization: Bearer $TOKEN" -F "file=@paint-it-black.mp3" \
       "$URL/v1/audio/analyze/upload"
   ```
2. **Give the server cookies from a signed-in browser** (what the error asks
   for). Export `cookies.txt` (a browser extension, or `yt-dlp
   --cookies-from-browser`), store it in Secret Manager, and inject it:
   ```sh
   gcloud secrets create snoocle-yt-cookies --data-file=cookies.txt
   gcloud secrets add-iam-policy-binding snoocle-yt-cookies \
       --member="serviceAccount:snoocle-run@${PROJECT_ID}.iam.gserviceaccount.com" \
       --role=roles/secretmanager.secretAccessor
   gcloud run services update snoocle --region="$REGION" \
       --update-secrets="SNOOCLE_YTDLP_COOKIES=snoocle-yt-cookies:latest"
   ```
   (`SNOOCLE_YTDLP_COOKIES` is the cookies.txt *content*; `SNOOCLE_YTDLP_COOKIES_FILE`
   takes a mounted path instead.) Cookies expire — refresh the secret when
   acquisition starts failing again. As a no-cookie long shot, try alternate
   player clients: `--update-env-vars="SNOOCLE_YTDLP_PLAYER_CLIENTS=default,android,ios,tv,web_safari"`.

   **Make cookies last (the hands-off recipe).** YouTube rotates cookies
   aggressively when the *same browser session* keeps using them. So: create a
   throwaway Google account → sign in to youtube.com in a **private/incognito
   window** → export cookies.txt from that window → close the window and never
   sign in with that session again. Cookies exported this way routinely last
   weeks–months for a single-user tool, versus days for cookies from your
   daily-driver browser.

3. **Route yt-dlp through a residential IP** (`SNOOCLE_YTDLP_PROXY`). Only
   acquisition traffic uses the proxy; everything else is direct. Two flavors:
   - **Your own home IP via a Tailscale exit node** — free, personal-use-clean
     (it's your IP and your account). Run Tailscale on an always-on device at
     home (a Mac, Raspberry Pi, or even an Apple TV) as an exit node, add
     `tailscaled` (userspace, SOCKS5 on localhost:1055) to the container, and
     set `SNOOCLE_YTDLP_PROXY=socks5://localhost:1055`.
   - **A commercial residential proxy** — zero infra, paid:
     `SNOOCLE_YTDLP_PROXY=http://user:pass@proxy.vendor.example:8080` (keep the
     credential in Secret Manager).

4. **PO-token provider sidecar** — the yt-dlp-native fix for datacenter IPs:
   run `bgutil-ytdlp-pot-provider` as a Cloud Run sidecar container and install
   its yt-dlp plugin in the image; yt-dlp then attaches Proof-of-Origin tokens
   automatically. Most robust long-term, most moving parts.

Note: the bot-check is per-request IP reputation — it fires **before any bytes
download**, so "only analyze part of the song" does not avoid it. (Partial
analysis exists anyway for speed: set `SNOOCLE_MIR_MAX_ANALYSIS_SECONDS` to clip
what the MIR engines process.)

## Known gaps / follow-ups

- **Firestore document size** — each `versions/{sha}` snapshot stores the full
  Song JSON as a nested map, capped at Firestore's 1 MiB/document limit. That is
  ample for normal songs; a pathologically large chart could exceed it, at which
  point the snapshot would need gzip+bytes or a GCS spill. Not a concern for the
  personal-use corpus.
- **madmom is not in the runtime image** (the Dockerfile excludes it — heavy
  native build). The deployed beat engine is the librosa fallback, not the one
  used in this session's local acceptance run.
- **Chord-CNN-LSTM IS in the runtime image** (cloned + CPU torch in the
  Dockerfile builder stage; `SNOOCLE_CHORD_CNN_LSTM_DIR=/opt/models/
  chord-cnn-lstm` preset). `/healthz` should report `mirEngines.chords:
  "chord-cnn-lstm"`. Expect a few minutes of CPU inference per song — raise
  `--timeout` accordingly if full-pipeline requests start hitting it. For a
  local (non-Docker) install run `scripts/setup_chord_model.sh` and
  `pip install -e .[chordmodel]` (CPU torch: add
  `--index-url https://download.pytorch.org/whl/cpu`).
- **No automated re-deploy pipeline** — these are one-shot `gcloud` commands;
  wire up Cloud Build triggers or GitHub Actions if you want push-to-deploy.
