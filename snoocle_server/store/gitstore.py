"""Git-backed, versioned song-JSON store.

Every analysis run is a real commit in a dedicated repository — never an
overwrite. History/diff/rollback are ordinary git operations; no custom
versioning system.

Concurrency-safety primitives (the saveRecordIfVersionUnchanged /
expected-record-version shape from Dr-Lurie-Blog/CMS-Agent, reimplemented
here since those repos weren't reachable this session):

- save(..., expected_version=...) rejects the write with VersionConflictError
  when the song's current version (last commit touching its file) differs
  from what the caller last read — optimistic locking.
- an OS-level file lock around read-check-write-commit serializes concurrent
  writers on the same store, so two processes can't interleave between the
  version check and the commit.
- provenance is append-only: a save whose provenance does not extend the
  stored song's provenance is rejected.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..schema import Song

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "snoocle-server",
    "GIT_AUTHOR_EMAIL": "snoocle-server@localhost",
    "GIT_COMMITTER_NAME": "snoocle-server",
    "GIT_COMMITTER_EMAIL": "snoocle-server@localhost",
}


class StoreError(RuntimeError):
    pass


class VersionConflictError(StoreError):
    """The record changed since the caller last read it (optimistic-lock miss)."""


@dataclass
class SongVersion:
    version: str  # commit sha
    timestamp: str  # ISO-8601
    message: str


@dataclass
class SaveResult:
    song_id: str
    version: str  # commit sha of the new version
    path: str


class GitSongStore:
    def __init__(self, store_dir: str | Path):
        self.dir = Path(store_dir)
        self._ensure_repo()

    # --- internals -------------------------------------------------------

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            ["git", "-C", str(self.dir), *args],
            capture_output=True,
            text=True,
            env={**os.environ, **_GIT_ENV},
        )
        if check and proc.returncode != 0:
            raise StoreError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc

    def _ensure_repo(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if not (self.dir / ".git").exists():
            self._git("init", "-q", "-b", "main")
            (self.dir / "songs").mkdir(exist_ok=True)
            readme = self.dir / "README.md"
            readme.write_text(
                "# Snoocle song store\n\n"
                "Git-backed artifact store: every analysis run is a commit. "
                "Managed by snoocle-server; safe to inspect with ordinary git tooling.\n"
            )
            self._git("add", "-A")
            self._git("commit", "-q", "-m", "Initialize song store")

    @contextmanager
    def _write_lock(self):
        lock_path = self.dir / ".snoocle-store.lock"
        with open(lock_path, "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _song_path(self, song_id: str) -> Path:
        if "/" in song_id or song_id.startswith("."):
            raise StoreError(f"invalid song id {song_id!r}")
        return self.dir / "songs" / f"{song_id}.json"

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self.dir))

    # --- public API --------------------------------------------------------

    def current_version(self, song_id: str) -> str | None:
        """Last commit sha that touched this song, or None if it doesn't exist."""
        path = self._song_path(song_id)
        if not path.exists():
            return None
        proc = self._git("log", "-1", "--format=%H", "--", self._rel(path))
        sha = proc.stdout.strip()
        return sha or None

    def save(
        self,
        song: Song,
        message: str,
        expected_version: str | None = None,
        enforce_expected: bool = False,
    ) -> SaveResult:
        """Commit a new version of the song.

        expected_version: pass the version you read to get optimistic locking.
        enforce_expected: when True, a None expected_version is only valid for
        a song that does not exist yet (strict create-or-CAS semantics).
        """
        path = self._song_path(song.id)
        with self._write_lock():
            current = self.current_version(song.id)
            if enforce_expected or expected_version is not None:
                if expected_version != current:
                    raise VersionConflictError(
                        f"song {song.id!r}: expected version "
                        f"{expected_version!r} but store has {current!r}"
                    )
            if path.exists():
                old = json.loads(path.read_text())
                old_prov = old.get("provenance", [])
                new_prov = [p.model_dump() for p in song.provenance]
                if new_prov[: len(old_prov)] != old_prov:
                    raise StoreError(
                        f"song {song.id!r}: provenance is append-only; the new "
                        "version must extend the stored provenance history"
                    )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(song.model_dump_json(indent=2) + "\n")
            self._git("add", self._rel(path))
            staged = self._git("diff", "--cached", "--quiet", check=False)
            if staged.returncode == 0:  # empty diff -> no-op save is an error
                raise StoreError(f"song {song.id!r}: save produced no changes")
            self._git("commit", "-q", "-m", message)
            new_version = self._git("rev-parse", "HEAD").stdout.strip()
        return SaveResult(song_id=song.id, version=new_version, path=str(path))

    def get(self, song_id: str, version: str | None = None) -> Song:
        path = self._song_path(song_id)
        if version is None:
            if not path.exists():
                raise StoreError(f"song {song_id!r} not found")
            return Song.model_validate_json(path.read_text())
        proc = self._git("show", f"{version}:{self._rel(path)}", check=False)
        if proc.returncode != 0:
            raise StoreError(f"song {song_id!r} at version {version!r} not found")
        return Song.model_validate_json(proc.stdout)

    def versions(self, song_id: str) -> list[SongVersion]:
        path = self._song_path(song_id)
        proc = self._git(
            "log", "--format=%H%x09%cI%x09%s", "--follow", "--", self._rel(path), check=False
        )
        out = []
        for line in proc.stdout.splitlines():
            sha, ts, msg = line.split("\t", 2)
            out.append(SongVersion(version=sha, timestamp=ts, message=msg))
        return out

    def diff(self, song_id: str, version_a: str, version_b: str) -> str:
        path = self._rel(self._song_path(song_id))
        proc = self._git("diff", version_a, version_b, "--", path)
        return proc.stdout

    def list_songs(self) -> list[str]:
        songs_dir = self.dir / "songs"
        if not songs_dir.exists():
            return []
        return sorted(p.stem for p in songs_dir.glob("*.json"))
