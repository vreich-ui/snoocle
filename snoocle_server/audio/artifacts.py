"""Opaque, temporary audio artifacts for authenticated HTTP and MCP callers.

The public identifier never contains a filename, bucket name, or filesystem
path. Local development uses atomic files; production can use a private GCS
bucket so a reference created on one Cloud Run instance resolves on every
other instance. Objects are always streamed through Snoocle's authenticated
endpoint rather than exposed with a public or signed bucket URL.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import secrets
import shutil
import tempfile
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Literal

from ..config import settings
from . import utils as audio_utils


_REF_RE = re.compile(r"aud_[A-Za-z0-9_-]{32}")
_SUPPORTED_SUFFIXES = {".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm"}
_MIME_BY_SUFFIX = {
    ".flac": {"audio/flac", "audio/x-flac"},
    ".m4a": {"audio/mp4", "audio/m4a", "audio/x-m4a"},
    ".mp3": {"audio/mpeg", "audio/mp3"},
    ".ogg": {"audio/ogg", "application/ogg"},
    ".opus": {"audio/ogg", "audio/opus"},
    ".wav": {"audio/wav", "audio/x-wav", "audio/wave"},
    ".webm": {"audio/webm"},
}
_FORMAT_BY_SUFFIX = {
    ".flac": {"flac"},
    ".m4a": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"},
    ".mp3": {"mp3"},
    ".ogg": {"ogg"},
    ".opus": {"ogg"},
    ".wav": {"wav"},
    ".webm": {"matroska", "webm"},
}


class AudioArtifactError(RuntimeError):
    pass


class AudioArtifactNotFound(AudioArtifactError):
    pass


class AudioArtifactValidationError(AudioArtifactError):
    pass


class AudioArtifactQuotaError(AudioArtifactError):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def validate_ref(audio_ref: str) -> str:
    if not _REF_RE.fullmatch(audio_ref or ""):
        raise AudioArtifactNotFound("audio artifact not found")
    return audio_ref


def _safe_filename(value: str | None, suffix: str) -> str:
    name = Path(value or f"audio{suffix}").name
    name = "".join(ch for ch in name if ch >= " " and ch not in {"\x7f", "/", "\\"}).strip()
    stem = Path(name).stem if name else "audio"
    stem = stem[: max(1, 240 - len(suffix))]
    return f"{stem or 'audio'}{suffix}"


def _content_type(suffix: str) -> str:
    return {
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
        ".wav": "audio/wav",
        ".webm": "audio/webm",
    }[suffix]


@dataclass(frozen=True)
class AudioArtifact:
    audio_ref: str
    filename: str
    content_type: str
    size_bytes: int
    duration_seconds: float
    source: Literal["upload", "youtube", "mcp"]
    created_at: str
    expires_at: str
    youtube_video_id: str | None = None

    def to_public(self) -> dict:
        return {
            "audioRef": self.audio_ref,
            "filename": self.filename,
            "contentType": self.content_type,
            "sizeBytes": self.size_bytes,
            "durationSeconds": self.duration_seconds,
            "source": self.source,
            "createdAt": self.created_at,
            "expiresAt": self.expires_at,
            "youtubeVideoId": self.youtube_video_id,
            "playbackUrl": f"/v1/audio/artifacts/{self.audio_ref}/content",
        }

    def to_storage(self) -> dict:
        return asdict(self)

    @classmethod
    def from_storage(cls, value: dict) -> "AudioArtifact":
        return cls(**value)

    @property
    def expired(self) -> bool:
        return _parse_time(self.expires_at) <= _utcnow()


def validate_audio_file(
    path: Path,
    *,
    filename: str | None,
    declared_content_type: str | None,
) -> tuple[str, str, audio_utils.AudioProbe]:
    suffix = Path(filename or path.name).suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise AudioArtifactValidationError(
            f"unsupported audio type {suffix or '(missing extension)'}; "
            f"supported: {', '.join(sorted(_SUPPORTED_SUFFIXES))}"
        )
    if declared_content_type:
        claimed = declared_content_type.split(";", 1)[0].strip().lower()
        if claimed not in _MIME_BY_SUFFIX[suffix]:
            raise AudioArtifactValidationError(
                f"content type {claimed or '(missing)'} does not match {suffix} audio"
            )
    size = path.stat().st_size
    if size <= 0:
        raise AudioArtifactValidationError("audio file is empty")
    if size > settings.audio_artifact_max_bytes:
        raise AudioArtifactValidationError(
            f"audio file is {size} bytes; limit is {settings.audio_artifact_max_bytes}"
        )
    try:
        probe = audio_utils.probe(path)
    except audio_utils.AudioToolError as error:
        raise AudioArtifactValidationError(str(error)) from error
    actual_formats = {part.strip().lower() for part in probe.format_name.split(",")}
    if not (actual_formats & _FORMAT_BY_SUFFIX[suffix]):
        raise AudioArtifactValidationError(
            f"file contents ({probe.format_name or 'unknown'}) do not match {suffix}"
        )
    if probe.video_streams:
        raise AudioArtifactValidationError("audio artifacts must not contain a video stream")
    if probe.duration_seconds <= 0:
        raise AudioArtifactValidationError("audio duration must be positive")
    if probe.duration_seconds > settings.audio_artifact_max_duration_seconds:
        raise AudioArtifactValidationError(
            f"audio duration is {probe.duration_seconds:.1f}s; limit is "
            f"{settings.audio_artifact_max_duration_seconds:.1f}s"
        )
    return suffix, _content_type(suffix), probe


class AudioArtifactStore(ABC):
    backend = "unknown"

    def __init__(self) -> None:
        self._lock = threading.RLock()

    @abstractmethod
    def _put(self, artifact: AudioArtifact, source: Path) -> None: ...

    @abstractmethod
    def _load(self, audio_ref: str) -> AudioArtifact: ...

    @abstractmethod
    def _list(self) -> list[AudioArtifact]: ...

    @abstractmethod
    def _delete(self, audio_ref: str) -> bool: ...

    @abstractmethod
    def iter_content(self, audio_ref: str, start: int, end: int) -> Iterator[bytes]: ...

    @abstractmethod
    @contextmanager
    def materialize(self, audio_ref: str) -> Iterator[Path]: ...

    def create(
        self,
        source_path: str | Path,
        *,
        filename: str | None,
        declared_content_type: str | None = None,
        source: Literal["upload", "youtube", "mcp"] = "upload",
        youtube_video_id: str | None = None,
    ) -> AudioArtifact:
        path = Path(source_path)
        suffix, content_type, probe = validate_audio_file(
            path, filename=filename, declared_content_type=declared_content_type
        )
        now = _utcnow()
        with self._lock:
            self.cleanup_expired()
            current = self._list()
            total = sum(item.size_bytes for item in current)
            if len(current) >= settings.audio_artifact_quota_count:
                raise AudioArtifactQuotaError(
                    f"audio artifact quota allows {settings.audio_artifact_quota_count} objects"
                )
            size = path.stat().st_size
            if total + size > settings.audio_artifact_quota_bytes:
                raise AudioArtifactQuotaError(
                    f"audio artifact quota allows {settings.audio_artifact_quota_bytes} total bytes"
                )
            for _ in range(8):
                audio_ref = "aud_" + secrets.token_urlsafe(24)
                try:
                    self._load(audio_ref)
                except AudioArtifactNotFound:
                    break
            else:  # pragma: no cover - cryptographic collision defense
                raise AudioArtifactError("could not allocate an audio reference")
            artifact = AudioArtifact(
                audio_ref=audio_ref,
                filename=_safe_filename(filename, suffix),
                content_type=content_type,
                size_bytes=size,
                duration_seconds=probe.duration_seconds,
                source=source,
                created_at=_iso(now),
                expires_at=_iso(now + timedelta(seconds=settings.audio_artifact_ttl_seconds)),
                youtube_video_id=youtube_video_id,
            )
            try:
                self._put(artifact, path)
            except AudioArtifactError:
                raise
            except Exception as error:
                raise AudioArtifactError("audio artifact storage write failed") from error
            return artifact

    def get(self, audio_ref: str) -> AudioArtifact:
        validate_ref(audio_ref)
        with self._lock:
            artifact = self._load(audio_ref)
            try:
                expired = artifact.expired
            except (TypeError, ValueError) as error:
                raise AudioArtifactError("audio artifact metadata is corrupt") from error
            if expired:
                self._delete(audio_ref)
                raise AudioArtifactNotFound("audio artifact not found")
            return artifact

    def delete(self, audio_ref: str) -> bool:
        validate_ref(audio_ref)
        with self._lock:
            return self._delete(audio_ref)

    def cleanup_expired(self) -> int:
        with self._lock:
            expired = []
            for item in self._list():
                try:
                    if item.expired:
                        expired.append(item.audio_ref)
                except (TypeError, ValueError) as error:
                    raise AudioArtifactError("audio artifact metadata is corrupt") from error
            return sum(1 for audio_ref in expired if self._delete(audio_ref))


class LocalAudioArtifactStore(AudioArtifactStore):
    backend = "local"

    def __init__(self, root: str | Path) -> None:
        super().__init__()
        self.root = Path(root)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise AudioArtifactError("local audio artifact directory is unavailable") from error

    def _metadata_path(self, audio_ref: str) -> Path:
        validate_ref(audio_ref)
        return self.root / f"{audio_ref}.json"

    def _content_path(self, audio_ref: str) -> Path:
        validate_ref(audio_ref)
        return self.root / f"{audio_ref}.audio"

    def _put(self, artifact: AudioArtifact, source: Path) -> None:
        content = self._content_path(artifact.audio_ref)
        metadata = self._metadata_path(artifact.audio_ref)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{artifact.audio_ref}-", dir=self.root)
        os.close(fd)
        temporary = Path(temporary_name)
        meta_temporary = temporary.with_suffix(".json.tmp")
        try:
            shutil.copyfile(source, temporary)
            meta_temporary.write_text(json.dumps(artifact.to_storage(), sort_keys=True))
            os.replace(temporary, content)
            os.replace(meta_temporary, metadata)
        except Exception:
            temporary.unlink(missing_ok=True)
            meta_temporary.unlink(missing_ok=True)
            content.unlink(missing_ok=True)
            metadata.unlink(missing_ok=True)
            raise

    def _load(self, audio_ref: str) -> AudioArtifact:
        metadata = self._metadata_path(audio_ref)
        content = self._content_path(audio_ref)
        if not metadata.is_file() or not content.is_file():
            raise AudioArtifactNotFound("audio artifact not found")
        try:
            return AudioArtifact.from_storage(json.loads(metadata.read_text()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise AudioArtifactError("audio artifact metadata is corrupt") from error

    def _list(self) -> list[AudioArtifact]:
        found = []
        for metadata in self.root.glob("aud_*.json"):
            try:
                found.append(AudioArtifact.from_storage(json.loads(metadata.read_text())))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return found

    def _delete(self, audio_ref: str) -> bool:
        metadata = self._metadata_path(audio_ref)
        content = self._content_path(audio_ref)
        existed = metadata.exists() or content.exists()
        metadata.unlink(missing_ok=True)
        content.unlink(missing_ok=True)
        return existed

    def iter_content(self, audio_ref: str, start: int, end: int) -> Iterator[bytes]:
        artifact = self.get(audio_ref)
        if start < 0 or end < start or end >= artifact.size_bytes:
            raise AudioArtifactValidationError("invalid byte range")

        def chunks() -> Iterator[bytes]:
            with self._content_path(audio_ref).open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return chunks()

    @contextmanager
    def materialize(self, audio_ref: str) -> Iterator[Path]:
        artifact = self.get(audio_ref)
        suffix = Path(artifact.filename).suffix or ".audio"
        directory = Path(tempfile.mkdtemp(prefix="snoocle-artifact-"))
        destination = directory / f"input{suffix}"
        try:
            try:
                os.link(self._content_path(audio_ref), destination)
            except OSError:
                try:
                    shutil.copyfile(self._content_path(audio_ref), destination)
                except OSError as error:
                    raise AudioArtifactError("audio artifact content is unavailable") from error
            yield destination
        finally:
            shutil.rmtree(directory, ignore_errors=True)


class GCSAudioArtifactStore(AudioArtifactStore):
    backend = "gcs"

    def __init__(self, bucket_name: str, prefix: str, client=None) -> None:
        super().__init__()
        if not bucket_name:
            raise ValueError("SNOOCLE_AUDIO_ARTIFACT_GCS_BUCKET is required for the GCS backend")
        if client is None:
            try:
                from google.cloud import storage

                client = storage.Client(project=settings.google_cloud_project or None)
            except Exception as error:
                raise AudioArtifactError("GCS audio artifact client is unavailable") from error
        self.bucket = client.bucket(bucket_name)
        self.prefix = prefix.strip("/") or "audio-artifacts"

    def _name(self, audio_ref: str, leaf: str) -> str:
        validate_ref(audio_ref)
        return f"{self.prefix}/{audio_ref}/{leaf}"

    def _put(self, artifact: AudioArtifact, source: Path) -> None:
        content = self.bucket.blob(self._name(artifact.audio_ref, "content"))
        metadata = self.bucket.blob(self._name(artifact.audio_ref, "metadata.json"))
        content_created = False
        try:
            content.upload_from_filename(
                str(source), content_type=artifact.content_type, if_generation_match=0
            )
            content_created = True
            metadata.upload_from_string(
                json.dumps(artifact.to_storage(), sort_keys=True),
                content_type="application/json",
                if_generation_match=0,
            )
        except Exception:
            if content_created:
                try:
                    content.delete()
                except Exception:  # noqa: BLE001 - cleanup after a partial write
                    pass
            raise

    def _load(self, audio_ref: str) -> AudioArtifact:
        blob = self.bucket.blob(self._name(audio_ref, "metadata.json"))
        try:
            raw = blob.download_as_bytes()
        except Exception as error:  # google exceptions are optional at import time
            if getattr(error, "code", None) == 404 or error.__class__.__name__ == "NotFound":
                raise AudioArtifactNotFound("audio artifact not found") from error
            raise AudioArtifactError("audio artifact metadata is unavailable") from error
        try:
            return AudioArtifact.from_storage(json.loads(raw))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise AudioArtifactError("audio artifact metadata is corrupt") from error

    def _list(self) -> list[AudioArtifact]:
        found = []
        try:
            blobs = self.bucket.list_blobs(prefix=f"{self.prefix}/")
            for blob in blobs:
                if not blob.name.endswith("/metadata.json"):
                    continue
                try:
                    found.append(AudioArtifact.from_storage(json.loads(blob.download_as_bytes())))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
        except Exception as error:
            raise AudioArtifactError("audio artifact quota inventory is unavailable") from error
        return found

    def _delete(self, audio_ref: str) -> bool:
        existed = False
        for leaf in ("metadata.json", "content"):
            blob = self.bucket.blob(self._name(audio_ref, leaf))
            try:
                blob.delete()
                existed = True
            except Exception as error:
                if getattr(error, "code", None) != 404 and error.__class__.__name__ != "NotFound":
                    raise AudioArtifactError("audio artifact delete failed") from error
        return existed

    def iter_content(self, audio_ref: str, start: int, end: int) -> Iterator[bytes]:
        artifact = self.get(audio_ref)
        if start < 0 or end < start or end >= artifact.size_bytes:
            raise AudioArtifactValidationError("invalid byte range")
        blob = self.bucket.blob(self._name(audio_ref, "content"))

        def chunks() -> Iterator[bytes]:
            position = start
            while position <= end:
                stop = min(position + 1024 * 1024 - 1, end)
                try:
                    chunk = blob.download_as_bytes(start=position, end=stop)
                except Exception as error:
                    raise AudioArtifactError("audio artifact content is unavailable") from error
                if not chunk:
                    break
                position += len(chunk)
                yield chunk

        return chunks()

    @contextmanager
    def materialize(self, audio_ref: str) -> Iterator[Path]:
        artifact = self.get(audio_ref)
        suffix = Path(artifact.filename).suffix or mimetypes.guess_extension(artifact.content_type) or ".audio"
        directory = Path(tempfile.mkdtemp(prefix="snoocle-artifact-"))
        destination = directory / f"input{suffix}"
        try:
            blob = self.bucket.blob(self._name(audio_ref, "content"))
            try:
                blob.download_to_filename(str(destination))
            except Exception as error:
                raise AudioArtifactError("audio artifact content is unavailable") from error
            yield destination
        finally:
            shutil.rmtree(directory, ignore_errors=True)


_store: AudioArtifactStore | None = None
_store_signature: tuple | None = None
_store_lock = threading.Lock()


def artifact_backend_label() -> str:
    return get_audio_artifact_store().backend


def get_audio_artifact_store() -> AudioArtifactStore:
    global _store, _store_signature
    signature = (
        settings.audio_artifact_backend,
        str(settings.audio_artifact_dir),
        settings.audio_artifact_gcs_bucket,
        settings.audio_artifact_gcs_prefix,
        settings.google_cloud_project,
    )
    with _store_lock:
        if _store is not None and signature == _store_signature:
            return _store
        backend = settings.audio_artifact_backend.strip().lower()
        if backend == "auto":
            backend = "gcs" if settings.audio_artifact_gcs_bucket else "local"
        if backend == "local":
            _store = LocalAudioArtifactStore(settings.audio_artifact_dir)
        elif backend == "gcs":
            _store = GCSAudioArtifactStore(
                settings.audio_artifact_gcs_bucket, settings.audio_artifact_gcs_prefix
            )
        else:
            raise ValueError(
                f"unsupported audio artifact backend {settings.audio_artifact_backend!r} "
                "(expected auto, local, or gcs)"
            )
        _store_signature = signature
        return _store


def reset_audio_artifact_store() -> None:
    global _store, _store_signature
    with _store_lock:
        _store = None
        _store_signature = None
