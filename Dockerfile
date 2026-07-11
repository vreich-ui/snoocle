# syntax=docker/dockerfile:1

# =============================================================================
# Snoocle server — production image for Google Cloud Run
#
# Multi-stage build:
#   1. `builder`  compiles/installs all Python deps into a self-contained venv
#                 (build toolchains stay here and never reach the final image).
#   2. `runtime`  a slim image with only the runtime OS packages + the venv,
#                 running as a non-root user.
#
# Cloud Run notes:
#   * The container MUST listen on 0.0.0.0 and honor the injected $PORT env var
#     (default 8080). We run uvicorn directly against the ASGI app so the
#     app's own SNOOCLE_HOST/SNOOCLE_PORT config (which defaults to
#     127.0.0.1:8765) is bypassed.
#   * The filesystem is ephemeral; the git-backed store writes to /data, which
#     does NOT persist across instances/restarts. Wire up a real backing store
#     (e.g. a GCS-mounted volume or external repo) before relying on history.
# =============================================================================

# Pin one base image and reuse it for both stages so the venv's interpreter
# path and ABI match exactly. 3.11-slim has broad wheel coverage for the
# scientific stack (numpy/scipy/librosa/numba).
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build-time system deps. build-essential covers any transitive dep that lacks
# a prebuilt wheel; git is needed by pip for any VCS installs. Stripped from
# the final image because this stage is discarded.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Self-contained virtualenv we can copy wholesale into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv

WORKDIR /app

# Only the package metadata + source are needed to build the wheel.
COPY pyproject.toml ./
COPY snoocle_server ./snoocle_server

# Install the app plus the extras this service needs to run its core pipeline:
#   .[mir]           librosa/soundfile/numpy/scipy fallbacks for MIR analysis
#   anthropic        default SNOOCLE_LLM_PROVIDER (imported lazily at runtime)
#   python-multipart required by FastAPI for the UploadFile audio endpoints
# (dev/pytest and the heavy madmom build are intentionally excluded.)
RUN pip install ".[mir]" anthropic python-multipart


# -----------------------------------------------------------------------------
# Runtime stage
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv

# Runtime-only OS deps discovered in the code:
#   ffmpeg      -> provides ffmpeg + ffprobe (yt-dlp audio extraction, audio utils)
#   git         -> the git-backed versioned song store shells out to `git`
#   libsndfile1 -> libsndfile for soundfile (insurance; wheels usually bundle it)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the fully-built virtualenv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Non-root user for security. Fixed high UID *and GID* keep it clearly
# non-privileged and — critically — let a Cloud Run GCS-FUSE volume be mounted
# with matching uid=10001;gid=10001 mount-options (GCS mounts are root-owned by
# default, so a non-root user can't write without that). Pinning the GID makes
# the value the deploy runbook references reliable. See docs/DEPLOY_CLOUD_RUN.md.
RUN groupadd --gid 10001 appuser \
    && useradd --create-home --uid 10001 --gid 10001 appuser

WORKDIR /app

# Writable, app-owned location for the ephemeral git store + audio cache.
# Cloud Run's filesystem is in-memory/ephemeral — this survives a single
# instance's lifetime only.
RUN mkdir -p /data/songstore /data/audio-cache \
    && chown -R appuser:appuser /data /app

# Point the service's storage at the writable /data dir and make it listen on
# all interfaces. PORT is honored by the CMD below (Cloud Run injects it).
ENV SNOOCLE_DATA_DIR=/data \
    SNOOCLE_STORE_DIR=/data/songstore \
    SNOOCLE_AUDIO_CACHE_DIR=/data/audio-cache \
    SNOOCLE_HOST=0.0.0.0 \
    PORT=8080

USER appuser

# Documentation only — Cloud Run routes to whatever $PORT it sets (default 8080).
EXPOSE 8080

# Local/other-platform liveness check (ignored by Cloud Run, which has its own).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/healthz', timeout=4)" || exit 1

# Run the ASGI app directly so Cloud Run's $PORT is respected. `exec` makes
# uvicorn PID 1's replacement so SIGTERM propagates for graceful shutdown.
CMD exec uvicorn snoocle_server.api:app --host 0.0.0.0 --port ${PORT:-8080}
