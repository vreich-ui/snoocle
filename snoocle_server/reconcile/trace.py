"""Agent run trace — the process record that reconciliation used to discard.

A reconciliation run is a sequence of steps (read inputs, model turns, tool
calls, repair rounds, final validation). Historically only the final Song
survived; the *logic* — what the agent searched, fetched, and decided — was
thrown away. :class:`TraceRecorder` captures each step as it happens so the GUI
can replay a run in human-readable form, and a run in progress can be polled
live (see :data:`LIVE_RUNS`).

Everything here is plain, JSON-serializable data: a trace is persisted to the
run store and returned over the REST API verbatim.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    return uuid.uuid4().hex[:16]


def _truncate(value, limit: int = 2000):
    """Bound the size of a detail payload so a trace can't blow up storage."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 1] + "…"
    if isinstance(value, list):
        return [_truncate(v, limit) for v in value[:50]]
    if isinstance(value, dict):
        return {k: _truncate(v, limit) for k, v in list(value.items())[:50]}
    return value


@dataclass
class TraceStep:
    index: int
    kind: str  # inputs | model | tool | repair | final | error
    label: str  # short machine-ish label, e.g. "tool:fetch_chord_sheet"
    summary: str  # one human-readable line
    detail: dict = field(default_factory=dict)  # expandable raw payload
    timestamp: str = field(default_factory=_now_iso)
    duration_seconds: float | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "label": self.label,
            "summary": self.summary,
            "detail": self.detail,
            "timestamp": self.timestamp,
            "durationSeconds": (
                round(self.duration_seconds, 2) if self.duration_seconds is not None else None
            ),
        }


@dataclass
class RunTrace:
    run_id: str
    song_id: str
    provider: str
    model: str = ""
    depth: str = "standard"
    status: str = "running"  # running | ok | error
    started_at: str = field(default_factory=_now_iso)
    finished_at: str | None = None
    error: str | None = None
    config_version: str | None = None  # agent-config fingerprint (attribution)
    steps: list[TraceStep] = field(default_factory=list)
    # Full MIR snapshot for this run (MirAnalysis.to_run_payload()) — stored
    # UN-truncated (the payload bounds itself), unlike step details.
    mir: dict | None = None
    # Every analyze_audio_window probe the agent made: {window, chords, bpm, beats}.
    mir_windows: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "runId": self.run_id,
            "songId": self.song_id,
            "provider": self.provider,
            "model": self.model,
            "depth": self.depth,
            "status": self.status,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "error": self.error,
            "configVersion": self.config_version,
            "stepCount": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
            "mir": self.mir,
            "mirWindows": self.mir_windows,
        }


class TraceRecorder:
    """Thread-safe accumulator for a single run's steps.

    The engine owns one recorder per reconciliation and hands it to the
    provider (via ``provider.trace``) so model turns and tool calls land in the
    same timeline as the engine's own inputs/repair/final steps. Every append
    also refreshes the live registry so a concurrent poll sees progress.
    """

    def __init__(self, trace: RunTrace):
        self.trace = trace
        self._lock = threading.Lock()

    def step(self, kind: str, label: str, summary: str, detail: dict | None = None,
             duration_seconds: float | None = None) -> TraceStep:
        with self._lock:
            step = TraceStep(
                index=len(self.trace.steps),
                kind=kind,
                label=label,
                summary=summary,
                detail=_truncate(detail or {}),
                duration_seconds=duration_seconds,
            )
            self.trace.steps.append(step)
        _publish(self.trace)
        return step

    def attach_mir(self, payload: dict) -> None:
        """Attach the run's full MIR snapshot (already bounded — no truncation)."""
        with self._lock:
            self.trace.mir = payload
        _publish(self.trace)

    def add_mir_window(self, window: dict) -> None:
        """Record one agent analyze_audio_window probe (window + its results)."""
        with self._lock:
            self.trace.mir_windows.append(window)
        _publish(self.trace)

    def finish(self, status: str, model: str = "", error: str | None = None) -> None:
        with self._lock:
            self.trace.status = status
            self.trace.finished_at = _now_iso()
            if model:
                self.trace.model = model
            if error is not None:
                self.trace.error = error
        _publish(self.trace)


# --- live registry: last-N runs kept in-process for near-live polling --------
# Cloud Run may run multiple instances; a poll only sees a run in progress when
# it lands on the instance running it. That is acceptable for the "watch it
# happen" view — the completed trace is always durably readable from the run
# store regardless of instance.

_LIVE_MAX = 64
LIVE_RUNS: "OrderedDict[str, RunTrace]" = OrderedDict()
_live_lock = threading.Lock()


def _publish(trace: RunTrace) -> None:
    with _live_lock:
        LIVE_RUNS[trace.run_id] = trace
        LIVE_RUNS.move_to_end(trace.run_id)
        while len(LIVE_RUNS) > _LIVE_MAX:
            LIVE_RUNS.popitem(last=False)


def get_live_run(run_id: str) -> RunTrace | None:
    with _live_lock:
        return LIVE_RUNS.get(run_id)


def start_run(song_id: str, provider: str, depth: str, model: str = "") -> TraceRecorder:
    """Create a recorder + live-registered RunTrace for a new reconciliation."""
    trace = RunTrace(
        run_id=new_run_id(),
        song_id=song_id,
        provider=provider,
        model=model,
        depth=depth,
    )
    _publish(trace)
    return TraceRecorder(trace)


# Elapsed-time helper so callers don't each import time.monotonic.
def clock() -> float:
    return time.monotonic()
