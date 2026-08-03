"""The analysis worker — the half that runs on Wolf's Mac.

    while awake:
        claim a job  ->  run the pipeline  ->  report the result

That is the whole program. Everything else here is about the three ways it can
go wrong in real life: the laptop sleeps, the network drops, or a song is
simply broken.

**Why it exists.** The pipeline is CPU-bound (MIR) and, once stems land, GPU-
shaped. Cloud Run charges for an always-on instance to do background work
(~$53/month before any analysis happens) and runs it on a slower machine than
the laptop already on the desk. So the server brokers and the Mac works. See
``store/jobs.py`` for the protocol and ``docs/WORKER.md`` for setup.

**It is outbound-only.** Every exchange is an HTTPS request the worker
initiates. Nothing listens, no port is opened, no tunnel is needed, and the Mac
stays invisible to the internet. The same bearer token the admin UI uses is the
only credential.

**Sleep is not an error.** Closing the lid mid-song stops the heartbeat; the
lease expires server-side and the job returns to the queue. On wake the worker
finds its lease gone (409 on the next heartbeat), abandons the work in progress
rather than racing whoever holds it now, and asks for something new.

Run it directly for a one-shot drain (`snoocle-worker --once`), or let launchd
keep it alive (`scripts/install_worker.command`).
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import signal
import socket
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Optional

import httpx

log = logging.getLogger("snoocle.worker")

# How often to tell the server we're still alive. Comfortably inside the
# server's 300s lease so a single dropped request is survivable.
HEARTBEAT_SECONDS = 30

# Poll spacing when the queue is empty. Long enough that a laptop sitting idle
# all day makes a few thousand trivial requests, not a few hundred thousand.
IDLE_POLL_SECONDS = 15

# Backoff bounds for a server that is down or unreachable.
MIN_BACKOFF = 5
MAX_BACKOFF = 300


def default_worker_name() -> str:
    """Something a human can recognise in the queue dashboard."""
    host = socket.gethostname().split(".")[0]
    return f"{host} ({platform.system()} {platform.machine()})"


# Typographic characters macOS puts in machine names ("Wolf's Laptop" ships
# with U+2019, the default for every "<Name>'s MacBook"), mapped to their
# ASCII lookalikes. Anything else non-ASCII is transliterated or dropped.
_ASCII_SUBSTITUTIONS = str.maketrans({
    "‘": "'", "’": "'",   # curly single quotes
    "“": '"', "”": '"',   # curly double quotes
    "–": "-", "—": "-",   # en/em dash
    " ": " ",                  # non-breaking space
})


def _header_safe(name: str) -> str:
    """The worker name, made safe for an HTTP header value.

    Header values must encode as latin-1 and in practice should be ASCII;
    httpx enforces this at client construction, so a curly apostrophe in the
    machine name used to crash the worker before its first request — and
    launchd would restart it into the same crash, forever. The JSON payloads
    keep the original name (UTF-8 is fine there); only the header is coerced.
    """
    text = name.translate(_ASCII_SUBSTITUTIONS)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return text.strip() or "unnamed-worker"


def detect_capabilities() -> list[str]:
    """What this machine can actually do, declared honestly.

    The broker only hands a worker jobs whose `wants` it satisfies, so
    over-claiming here means jobs get claimed and then fail. Each capability is
    proven by an import, not assumed from configuration.
    """
    caps = ["analyze"]  # the reconcile pipeline — the reason the worker exists

    def importable(module: str) -> bool:
        try:
            __import__(module)
            return True
        except Exception:  # noqa: BLE001
            return False

    if importable("demucs"):
        caps.append("stems")
    if importable("whisperx"):
        caps.append("align")
    if importable("torchcrepe"):
        caps.append("melody")
    return caps


@dataclass
class WorkerConfig:
    base_url: str
    token: Optional[str] = None
    name: str = ""
    capabilities: Optional[list[str]] = None
    lease_seconds: int = 300
    idle_poll_seconds: int = IDLE_POLL_SECONDS
    once: bool = False
    timeout: float = 30.0

    @classmethod
    def from_env(cls, **overrides: Any) -> "WorkerConfig":
        cfg = cls(
            base_url=os.environ.get("SNOOCLE_SERVER_URL", "").rstrip("/"),
            token=os.environ.get("SNOOCLE_API_TOKEN") or None,
            name=os.environ.get("SNOOCLE_WORKER_NAME") or default_worker_name(),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        if cfg.capabilities is None:
            cfg.capabilities = detect_capabilities()
        cfg.base_url = cfg.base_url.rstrip("/")
        return cfg


class LeaseLost(RuntimeError):
    """The server no longer thinks we hold this job — stop, don't race."""


class Heartbeat:
    """Keeps a lease alive on a background thread while a job runs.

    The pipeline is a long synchronous call, so the heartbeat cannot share its
    thread. When the server says the lease is gone (409), `lost` is set and the
    pipeline's result is discarded rather than written over whatever the new
    holder produces.
    """

    def __init__(self, client: httpx.Client, job_id: str, worker: str,
                 lease_seconds: int, interval: int = HEARTBEAT_SECONDS) -> None:
        self._client = client
        self._job_id = job_id
        self._worker = worker
        self._lease_seconds = lease_seconds
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.lost = False

    def __enter__(self) -> "Heartbeat":
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"heartbeat-{self._job_id}")
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                res = self._client.post(
                    f"/v1/queue/{self._job_id}/heartbeat",
                    json={"worker": self._worker, "leaseSeconds": self._lease_seconds},
                )
                if res.status_code in (404, 409):
                    log.warning("lease lost on job %s: %s", self._job_id, res.text[:200])
                    self.lost = True
                    return
            except httpx.HTTPError as e:
                # A transient network blip is not a lost lease — the lease has
                # 300s of slack and this retries in 30. Only the server saying
                # so counts.
                log.debug("heartbeat failed (will retry): %s", e)


class Worker:
    def __init__(self, config: WorkerConfig,
                 run_job: Optional[Callable[[dict], dict]] = None) -> None:
        if not config.base_url:
            raise ValueError(
                "no server URL — set SNOOCLE_SERVER_URL (e.g. "
                "https://snoocle-....run.app)"
            )
        self.config = config
        self._run_job = run_job or run_pipeline_job
        headers = {"User-Agent": f"snoocle-worker/{_header_safe(config.name)}"}
        if config.token:
            try:
                config.token.encode("ascii")
            except UnicodeEncodeError as exc:
                # Don't coerce a credential — a mangled token would just 401
                # mysteriously. Name the actual problem instead.
                raise ValueError(
                    "SNOOCLE_API_TOKEN contains a non-ASCII character "
                    "(often a smart quote or invisible space from copy-paste); "
                    "re-copy the token as plain text"
                ) from exc
            headers["Authorization"] = f"Bearer {config.token}"
        self.client = httpx.Client(
            base_url=config.base_url, headers=headers, timeout=config.timeout,
        )
        self._stopping = threading.Event()

    # --- lifecycle -------------------------------------------------------

    def stop(self) -> None:
        """Finish the job in flight, then exit. A SIGTERM mid-analysis should
        not throw away ten minutes of work."""
        self._stopping.set()

    def close(self) -> None:
        self.client.close()

    def run_forever(self) -> int:
        """Returns the number of jobs completed (useful for --once and tests)."""
        completed = 0
        backoff = MIN_BACKOFF
        while not self._stopping.is_set():
            try:
                job = self.claim()
                backoff = MIN_BACKOFF
            except httpx.HTTPError as e:
                log.warning("cannot reach the server (%s); retrying in %ss", e, backoff)
                if self._stopping.wait(backoff) or self.config.once:
                    break
                backoff = min(backoff * 2, MAX_BACKOFF)
                continue

            if job is None:
                if self.config.once:
                    break
                if self._stopping.wait(self.config.idle_poll_seconds):
                    break
                continue

            self.process(job)
            completed += 1
            if self.config.once:
                break
        return completed

    # --- protocol --------------------------------------------------------

    def claim(self) -> Optional[dict]:
        res = self.client.post("/v1/queue/claim", json={
            "worker": self.config.name,
            "capabilities": self.config.capabilities or [],
            "leaseSeconds": self.config.lease_seconds,
        })
        if res.status_code == 204:
            return None
        res.raise_for_status()
        return res.json()

    def process(self, job: dict) -> None:
        job_id = job["id"]
        log.info("claimed %s: %s", job_id, job.get("title") or job.get("youtubeUrlOrId"))
        started = time.monotonic()
        try:
            with Heartbeat(self.client, job_id, self.config.name,
                           self.config.lease_seconds) as beat:
                result = self._run_job(job)
                if beat.lost:
                    raise LeaseLost(job_id)
        except LeaseLost:
            log.warning("dropped %s — the lease expired while it ran; the server "
                        "has requeued it", job_id)
            return
        except Exception as e:  # noqa: BLE001
            # One bad song must never stop the worker. Report it and move on;
            # the broker decides whether it is worth another attempt.
            log.exception("job %s failed", job_id)
            self._report_failure(job_id, e)
            return

        elapsed = time.monotonic() - started
        log.info("finished %s in %.0fs -> %s", job_id, elapsed, result.get("songId"))
        self._report_success(job_id, result)

    def _report_success(self, job_id: str, result: dict) -> None:
        try:
            res = self.client.post(f"/v1/queue/{job_id}/complete", json={
                "worker": self.config.name,
                "songId": result["songId"],
                "runId": result.get("runId"),
                "storedVersion": result.get("storedVersion"),
            })
            if res.status_code == 409:
                # The song is saved either way — the pipeline writes it — so a
                # lost lease here costs bookkeeping, not work.
                log.warning("completed %s but the lease was gone; the song is "
                            "still saved as %s", job_id, result.get("songId"))
                return
            res.raise_for_status()
        except httpx.HTTPError as e:
            log.error("could not report completion of %s: %s", job_id, e)

    def _report_failure(self, job_id: str, error: BaseException) -> None:
        try:
            self.client.post(f"/v1/queue/{job_id}/fail", json={
                "worker": self.config.name,
                "error": f"{type(error).__name__}: {error}"[:2000],
                "retry": _is_worth_retrying(error),
            })
        except httpx.HTTPError as e:
            # Nothing more to do: the lease will expire and the broker will
            # requeue it on its own. That fallback is why leases exist.
            log.error("could not report failure of %s: %s", job_id, e)


def _is_worth_retrying(error: BaseException) -> bool:
    """A missing video or an unparseable song will fail identically next time;
    a network hiccup or a rate limit will not. Guessing wrong is cheap in one
    direction (a wasted retry) and annoying in the other (a good song parked as
    broken), so the default leans toward retrying."""
    text = f"{type(error).__name__}: {error}".lower()
    permanent = (
        "not found", "unavailable", "private video", "removed",
        "no candidate sources", "validationerror", "does not match",
    )
    return not any(marker in text for marker in permanent)


def run_stems_job(job: dict) -> dict:
    """Separate an existing song's audio into stems.

    Note what this does NOT do: re-run the analysis. The song already exists
    and its times were measured against a specific upload; producing a new
    version as a side effect of asking for a backing track would be a nasty
    surprise. This only writes files beside the song.

    Imported lazily, like everything else that needs the heavy extras.
    """
    from .audio import stems
    from .audio.acquire import acquire

    song_id = job.get("targetSongId")
    if not song_id:
        raise ValueError("a stems job needs targetSongId")

    audio = acquire(job.get("youtubeUrlOrId"))
    result = stems.separate(audio.path, song_id, model=job.get("model") or stems.DEFAULT_MODEL)
    return {
        "songId": song_id,
        "stems": sorted(result.stems),
        "mixes": sorted(result.mixes),
    }


def run_align_job(job: dict) -> dict:
    """Time an existing song's lyrics against its audio and store the result.

    Unlike stems, the output here is small structured data, so it goes where
    the rest of the song's timing lives — on the song, as a new version with a
    provenance entry naming the engine and which audio it heard. That is what
    makes it reviewable later instead of a number nobody can account for.

    Only lines that have no time yet are filled. LRCLIB is the more reliable
    layer where it has coverage, and a human correction in the review queue is
    more reliable than either; neither should be quietly overwritten by a
    later alignment run.
    """
    from .audio.acquire import acquire
    from .pipeline import get_store
    from .schema import ProvenanceEntry
    from .timing.align import align_lyrics

    song_id = job.get("targetSongId")
    if not song_id:
        raise ValueError("an align job needs targetSongId")

    store = get_store()
    song = store.get(song_id)
    audio = acquire(job.get("youtubeUrlOrId"))

    result = align_lyrics(
        audio.path,
        [line.lyrics or "" for line in song.lines],
        song_id=song_id,
    )

    filled = 0
    for timed in result.lines:
        if timed.start is None or timed.lineIndex >= len(song.lines):
            continue
        line = song.lines[timed.lineIndex]
        if line.timeSeconds is None:
            line.timeSeconds = round(timed.start, 3)
            line.confidence = round(timed.coverage, 3)
            filled += 1

    song.provenance.append(ProvenanceEntry(
        step="align",
        detail=(
            f"whisperx forced alignment on the {result.source}; "
            f"coverage {result.coverage:.2f}; filled {filled} previously untimed lines"
        ),
    ))
    version = store.save(song, message=f"align: {filled} line times from audio")
    return {
        "songId": song_id,
        "storedVersion": getattr(version, "version", None) or str(version),
        "filledLines": filled,
        "coverage": round(result.coverage, 3),
        "source": result.source,
    }


def run_pipeline_job(job: dict) -> dict:
    """Execute one job, dispatching on its kind.

    Imported lazily so the worker module stays importable (and testable) on a
    machine without the MIR/ML extras installed.
    """
    import asyncio

    kind = job.get("kind")
    if kind == "stems":
        return run_stems_job(job)
    if kind == "align":
        return run_align_job(job)

    from .pipeline import run_pipeline_async

    report = asyncio.run(run_pipeline_async(
        job.get("title"),
        job.get("artist"),
        youtube_url_or_id=job.get("youtubeUrlOrId"),
        provider=job.get("provider"),
        analysis_depth=job.get("analysisDepth"),
        batch_id=job.get("batchId"),
        agent_policy=job.get("agentPolicy"),
        allow_test_output=bool(job.get("allowTestOutput", False)),
    ))
    return {
        "songId": report.song_id,
        "runId": report.run_id,
        "storedVersion": report.stored_version,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="snoocle-worker",
        description="Claim and run Snoocle analysis jobs from a Snoocle server.",
    )
    parser.add_argument("--server", dest="base_url",
                        help="server base URL (default: $SNOOCLE_SERVER_URL)")
    parser.add_argument("--name", help="worker name shown in the queue dashboard")
    parser.add_argument("--once", action="store_true",
                        help="run at most one job, then exit")
    parser.add_argument("--capabilities", help="comma-separated override of the "
                                               "auto-detected capabilities")
    parser.add_argument("--lease-seconds", type=int, default=None)
    parser.add_argument("--poll-seconds", type=int, default=None,
                        dest="idle_poll_seconds")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    config = WorkerConfig.from_env(
        base_url=args.base_url,
        name=args.name,
        once=args.once or None,
        lease_seconds=args.lease_seconds,
        idle_poll_seconds=args.idle_poll_seconds,
        capabilities=(args.capabilities.split(",") if args.capabilities else None),
    )

    try:
        worker = Worker(config)
    except ValueError as e:
        print(f"snoocle-worker: {e}", file=sys.stderr)
        return 2

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: worker.stop())

    log.info("worker %r -> %s (capabilities: %s)",
             config.name, config.base_url, ",".join(config.capabilities or []))
    try:
        worker.run_forever()
    finally:
        worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
