# Temporary audio artifacts

Studio and remote MCP callers use authenticated opaque audio references instead
of server paths. A reference looks like `aud_<random>` and reveals neither a
local filename nor a GCS object name. Bucket URLs and signed URLs are never
returned: playback is always streamed through Snoocle, so the same REST bearer
token or Cloud Run IAM policy that protects `/v1` also protects the bytes.

## HTTP lifecycle

| Action | Endpoint |
| --- | --- |
| Upload local audio | `POST /v1/audio/artifacts` (`multipart/form-data`, field `file`) |
| Acquire YouTube audio | `POST /v1/audio/artifacts/acquire` |
| Read metadata | `GET /v1/audio/artifacts/{audioRef}` |
| Stream/seek | `GET /v1/audio/artifacts/{audioRef}/content` (`Range: bytes=...`) |
| Delete now | `DELETE /v1/audio/artifacts/{audioRef}` |
| Sweep expired objects | `POST /v1/audio/artifacts/cleanup` |

The older `POST /v1/audio/acquire` route remains as an alias but now returns an
artifact rather than a cache path. `POST /v1/audio/analyze` accepts `audioRef`;
its `audioPath` input remains for a trusted server-side caller but is never
echoed in the response. MCP audio wrappers follow the same rule with
`audio_ref`/`input_ref` plus their existing path inputs.

Every create checks the declared type, extension, probed container, presence of
an audio stream, absence of video, byte size, positive duration, maximum
duration, total bytes, and object count. Upload reads stop as soon as the byte
limit is crossed. References have a TTL, are lazily removed before creates and
reads, can be swept explicitly, and can always be deleted by the caller.

## Local development

No configuration is required. `SNOOCLE_AUDIO_ARTIFACT_BACKEND=auto` selects the
atomic local backend when no bucket is set. Files live under
`SNOOCLE_AUDIO_ARTIFACT_DIR` (default `data/audio-artifacts`) and are suitable
for one local server process. They are temporary and must not be treated as the
song store.

## Required production configuration

Cloud Run instances do not share a filesystem. Configure a pre-existing private
GCS bucket before production use:

```text
SNOOCLE_AUDIO_ARTIFACT_BACKEND=gcs
SNOOCLE_AUDIO_ARTIFACT_GCS_BUCKET=<private bucket name>
SNOOCLE_AUDIO_ARTIFACT_GCS_PREFIX=audio-artifacts
SNOOCLE_AUDIO_ARTIFACT_TTL_SECONDS=86400
SNOOCLE_AUDIO_ARTIFACT_MAX_BYTES=104857600
SNOOCLE_AUDIO_ARTIFACT_MAX_DURATION_SECONDS=1800
SNOOCLE_AUDIO_ARTIFACT_QUOTA_BYTES=1073741824
SNOOCLE_AUDIO_ARTIFACT_QUOTA_COUNT=50
```

The Cloud Run service account needs object create/read/delete/list permission
on that bucket (normally `roles/storage.objectUser` scoped to the bucket). The
bucket must not grant public access and does not need CORS because browsers
talk only to Snoocle. Configure a bucket lifecycle deletion rule slightly
longer than the application TTL (for example two days) as a backstop for an
instance that dies before cleanup. No bucket, IAM grant, lifecycle rule, or
other cloud resource is created by this repository.

The application performs count/byte quota checks by listing live metadata under
the configured prefix and serializes concurrent requests within an instance.
GCS generation preconditions prevent reference overwrite. A provider-side
bucket quota/lifecycle policy remains the hard cross-instance ceiling; two
creates admitted simultaneously on different instances can temporarily exceed
the application aggregate by their combined size, so configure the bucket's
own operational limits accordingly.
