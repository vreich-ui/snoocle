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
#   * Songs persist in Firestore (Native mode), NOT on disk — the container
#     filesystem is ephemeral and songs must survive instance restarts. The
#     store uses Application Default Credentials; set GOOGLE_CLOUD_PROJECT at
#     deploy time (the runtime service account needs roles/datastore.user).
#   * /data holds only the audio-cache (a best-effort, disposable cache).
#   * Deploy with --timeout=3600: a real analyze can take 2-8 min and Cloud
#     Run's default 300s request timeout would silently kill it.
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

# Real chord-recognition model: Chord-CNN-LSTM (ISMIR2019, music-x-lab).
# The pretrained 5-fold checkpoints (~28MB) ship in the upstream repo, so a
# shallow clone is a complete install. CPU-only torch keeps the image ~800MB
# smaller than the default CUDA build; inference is a few minutes per song on
# one Cloud Run CPU, well under the 1800s runner timeout. This is the single
# biggest accuracy lever over the chroma-template fallback.
RUN git clone --depth 1 \
        https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition.git \
        /opt/models/chord-cnn-lstm \
    && rm -rf /opt/models/chord-cnn-lstm/.git \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install h5py pretty_midi mir_eval pydub
COPY scripts/snoocle_runner.py /opt/models/chord-cnn-lstm/snoocle_runner.py


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
#   libsndfile1 -> libsndfile for soundfile (insurance; wheels usually bundle it)
# (yt-dlp is a Python dep installed into the venv; the song store is Firestore,
#  so `git` is no longer a runtime dependency.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the fully-built virtualenv and the chord model from the builder stage.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models /opt/models

# Non-root user for security. Fixed high UID *and GID* keep it clearly
# non-privileged and — critically — let a Cloud Run GCS-FUSE volume be mounted
# with matching uid=10001;gid=10001 mount-options (GCS mounts are root-owned by
# default, so a non-root user can't write without that). Pinning the GID makes
# the value the deploy runbook references reliable. See docs/DEPLOY_CLOUD_RUN.md.
RUN groupadd --gid 10001 appuser \
    && useradd --create-home --uid 10001 --gid 10001 appuser

WORKDIR /app

# Writable, app-owned location for the audio cache (disposable — the durable
# store is Firestore). Cloud Run's filesystem is in-memory/ephemeral.
RUN mkdir -p /data/audio-cache \
    && chown -R appuser:appuser /data /app

# Songs persist in Firestore (SNOOCLE_STORE_BACKEND=firestore); GOOGLE_CLOUD_PROJECT
# is injected at deploy time. Listen on all interfaces; PORT is honored by the
# CMD below (Cloud Run injects it).
ENV SNOOCLE_STORE_BACKEND=firestore \
    SNOOCLE_DATA_DIR=/data \
    SNOOCLE_AUDIO_CACHE_DIR=/data/audio-cache \
    SNOOCLE_CHORD_CNN_LSTM_DIR=/opt/models/chord-cnn-lstm \
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
