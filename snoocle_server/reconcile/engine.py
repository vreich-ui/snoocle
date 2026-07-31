"""Reconciliation engine: prompt -> LLM -> splice -> validate -> repair -> provenance.

Schema compliance and the chord-normalization rule are ENFORCED here, not
hoped for: the LLM's JSON must validate against the Song schema (whose
validators reject shape/tab chords outright). Validation errors are fed back
to the model for up to SNOOCLE_LLM_REPAIR_ATTEMPTS repair rounds.

Lyrics are the same story. A model-backed provider does not emit them at
all: it emits REFERENCES into the sources already in this run's context, and
``lyric_refs.splice_lyrics`` substitutes the real text here, before
validation. See that module for why (a Song document IS the complete lyrics
of a copyrighted song) and for the four rules that make it a guarantee
rather than a request.

Timing is enforced here too, in the same spirit and for the same reason: the
model is never the authority on it. Every deterministic timing pass guards its
writes with "is this field empty?", and model-supplied timing is never empty —
so a re-emitted ``timeSeconds`` silently outranked the measured value the pass
would have written, and nothing downstream could see it happen. ``_finalize``
strips model-supplied timing before those passes run, field by field, keeping
only what nothing else on that run can supply. See ``_strip_model_timing``.

WHICH pass will time the document is the CALLER's fact, not something this
module may infer: two public entry points (``POST /v1/reconcile`` and the MCP
``reconcile_song``) hand over a MIR and then run no timing pass at all. So the
caller declares it — :class:`TimingAuthority`, defaulting to ``NONE`` (strip
nothing) — and a caller that declares a pass and then doesn't complete it can
put back exactly what was taken (:meth:`TimingStrip.restore`).

A targeted correction in notes-only scope is a THIRD path, not a smaller
version of the first two: the model doesn't reconcile or reference anything,
it names a short list of OPERATIONS against the prior document
(``patch_ops.py``), and this module applies them in local code. Unlike a
schema-validation slip or an unresolvable lyric ref, a patch failure is
never retried — see ``_attempt_patch``'s docstring for why.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import ValidationError

from .. import __version__
from ..audio.utils import probe, trim
from ..config import settings
from ..discovery.models import CandidateSource
from ..identity import require_resolved_song_id
from ..mir.base import MirAnalysis
from ..schema import Song, song_json_schema
from ..schema.song import ProvenanceEntry, slugify_song_id
from ..scope import AnalysisScope
from .agent_config import AgentConfig, config_version
from .delta import (
    AppliedDelta,
    ReconcileDeltaError,
    apply_reconcile_delta,
    reconcile_delta_json_schema,
    strip_postpass_schema,
)
from .depth import DepthProfile, resolve_depth
from .lyric_refs import (
    LyricOverride,
    LyricSpliceError,
    UnresolvableLyricRefError,
    agent_song_json_schema,
    build_ref_index,
    splice_lyrics,
)
from .patch_ops import AppliedOp, PatchError, apply_patch, parse_ops_response
from .prompt import (
    build_patch_system_prompt,
    build_patch_user_prompt,
    build_delta_repair_prompt,
    build_repair_prompt,
    build_system_prompt,
    build_user_prompt,
)
from .providers import AudioAttachment, LLMProvider, get_provider
from .trace import RunTrace, TraceRecorder, clock

log = logging.getLogger(__name__)


def _load_agent_config() -> AgentConfig:
    """The stored operator config, or built-in defaults if unset/unreadable."""
    try:
        from ..store.agent_config import get_agent_config_store

        doc = get_agent_config_store().get()
        return AgentConfig.model_validate(doc) if doc else AgentConfig()
    except Exception as e:  # noqa: BLE001 — never fail a run over config
        log.warning("agent config unavailable, using defaults: %s", e)
        return AgentConfig()

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class ReconcileError(RuntimeError):
    # Optional machine-readable classification the pipeline surfaces to clients.
    error_code: str | None = None


class LyricProvenanceError(ReconcileError):
    """A lyric reached the document with no valid provenance: a ref this run
    cannot resolve, or so many overrides that reference resolution is plainly
    broken. Deliberately NOT repairable — see lyric_refs.py, rules 3 and 4."""

    error_code = "lyric_provenance"


class PatchApplicationError(ReconcileError):
    """A patch op didn't apply cleanly, or the response wasn't a valid patch
    at all (unknown op, over the cap, malformed). Deliberately NOT repaired
    — see patch_ops.py's module docstring and ``_attempt_patch`` below: a
    wrong op is not the kind of mistake a retry fixes, and "try again, maybe
    you'll guess differently" is exactly the fuzziness this path exists to
    refuse. Distinct from the model explicitly declining the patch path
    (``needsFullReconcile``), which is not an error at all."""

    error_code = "patch_failed"


@dataclass
class ReconcileResult:
    song: Song
    provider: str
    model: str
    attempts: int
    audio_attached: bool
    usage: dict = field(default_factory=dict)
    trace: RunTrace | None = None  # the run's step-by-step logic (if recorded)
    # >0 only when the patch path (patch_ops.py) produced this song: the
    # pipeline reads this to skip timing carry-forward/snap/LRC entirely —
    # nothing was regenerated, so there is nothing for those passes to do
    # except risk disturbing what a patch deliberately left alone.
    patch_ops_applied: int = 0
    # What the model-timing strip removed (None when nothing was), so the caller
    # that DECLARED the authority can put the recording-level fields back if the
    # pass it named never completed. See :class:`TimingStrip`.
    timing_strip: TimingStrip | None = None
    output_format: str = "full"
    patch_size_vs_full: dict | None = None


def extract_json(text: str) -> str:
    """Pull the JSON document out of an LLM response (tolerate fences/preamble)."""
    text = _FENCE_RE.sub("", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return text[start : end + 1]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


#: Provenance ``action`` for the normalisation below. Greppable on purpose: a
#: reader of a song's history must be able to find every run whose model output
#: claimed timing, and see exactly what was dropped.
TIMING_STRIP_ACTION = "model-timing-stripped"

#: Provenance ``action`` for the undo (:meth:`TimingStrip.restore`). Equally
#: greppable: a strip entry whose authority never delivered must be readable as
#: such off the song's own history, not inferred from a missing field.
TIMING_RESTORE_ACTION = "model-timing-restored"


class TimingAuthority(str, Enum):
    """WHICH deterministic pass will time this document once ``reconcile``
    returns — declared by the CALLER, never inferred here.

    The engine used to read this off ``mir is not None``, which is a property of
    the INPUTS, not of what the caller is going to do with the output. Two
    public entry points (``POST /v1/reconcile``, MCP ``reconcile_song``) pass a
    MIR and then hand the document straight back with no timing pass at all: on
    those, inferring an authority stripped ``metadata.bpm`` and ``audio.syncMap``
    for a pass that never ran, and named it in provenance as if it had.

    ``NONE`` is the default precisely because it is the safe answer: a caller
    that says nothing gets nothing stripped, and only a caller that OWNS a
    timing pass can ask for the fields that pass writes to be cleared.
    """

    #: No pass will time this document — strip nothing, the model's timing is
    #: the only timing there is (``POST /v1/reconcile``, MCP ``reconcile_song``,
    #: and any pipeline run that ends up with no MIR and no prior to carry).
    NONE = "none"
    #: ``timing.snap`` (``timing/snap.py``) will run against THIS run's MIR.
    SNAP = "snap"
    #: ``timing.carry_forward`` (``timing/carry_forward.py``) will copy the prior
    #: version's timing onto this document.
    CARRY_FORWARD = "carry_forward"


#: How each authority is named in provenance and on the run trace. Read by a
#: human six months later, so it says both the pass and where its numbers come
#: from.
_AUTHORITY_NAMES = {
    TimingAuthority.SNAP: "timing.snap (this run's MIR)",
    TimingAuthority.CARRY_FORWARD: "timing.carry_forward (the prior version)",
}


@dataclass(frozen=True)
class TimingStrip:
    """What :func:`_strip_model_timing` removed, and who owed it back.

    Rides on :class:`ReconcileResult` so the caller that DECLARED the authority
    can undo the strip when that authority did not actually complete — see
    :meth:`restore`. Only the RECORDING-level values are recorded: the
    per-element and per-section strip happens exclusively on the
    ``CARRY_FORWARD`` branch, whose only caller treats a carry-forward failure
    as fatal (``pipeline.py``'s ``timing_carry_forward_failed``), so a run that
    loses those never reaches a store to lose them in.
    """

    authority: TimingAuthority
    #: The field groups dropped, in the wording the provenance entry uses.
    fields: tuple[str, ...]
    #: The removed recording-level values, keyed by their path on the Song.
    #: Only keys actually stripped are present, so a restore puts back exactly
    #: what was taken and never invents a value the model never sent.
    recording: dict = field(default_factory=dict)

    @property
    def authority_name(self) -> str:
        return _AUTHORITY_NAMES.get(self.authority, self.authority.value)

    def restore(self, song: Song, *, reason: str) -> tuple[Song, list[str]]:
        """Put back every recording-level field this strip took that is STILL
        empty, with a provenance entry saying why. Returns ``(song, restored)``;
        ``restored`` is empty when the authority delivered after all.

        The strip's whole justification is "a pass is about to write this". When
        that pass skips or raises, the justification is void, and the honest
        repair is to undo the removal rather than substitute a third source:
        after the undo the document is byte-for-byte what it would have been
        with no strip at all, so nothing downstream — the pre-store audio-data
        guard included (``timing/carry_forward.audio_data_lost``) — can behave
        differently than it did before the strip existed.
        """
        restored: list[str] = []
        audio_update: dict = {}
        beats = self.recording.get("audio.beats")
        if beats and not song.audio.beats:
            audio_update["beats"] = list(beats)
            restored.append(f"audio.beats ({len(beats)} entries)")
        sync_map = self.recording.get("audio.syncMap")
        # Every restored value is one this document already validated with, so
        # the only cross-field rule the undo could break is the syncMap's
        # "lineIndex within range" (schema/song.py:309) — and only if something
        # dropped lines between the strip and here. Skip that one field rather
        # than let the undo be the thing that raises: the fields the pre-store
        # guard reads must come back regardless.
        if (
            sync_map
            and not song.audio.syncMap
            and all(p.lineIndex < len(song.lines) for p in sync_map)
        ):
            audio_update["syncMap"] = list(sync_map)
            restored.append(f"audio.syncMap ({len(sync_map)} points)")
        updates: dict = {}
        if audio_update:
            updates["audio"] = song.audio.model_copy(update=audio_update)
        bpm = self.recording.get("metadata.bpm")
        if bpm is not None and song.metadata.bpm is None:
            updates["metadata"] = song.metadata.model_copy(update={"bpm": bpm})
            restored.append(f"metadata.bpm ({bpm})")
        if not restored:
            return song, []
        entry = ProvenanceEntry(
            timestamp=_now(),
            actor=f"snoocle-server/{__version__}",
            action=TIMING_RESTORE_ACTION,
            confidence=None,
            notes=(
                f"{self.authority_name} did not time this document ({reason}), so "
                "the fields stripped for it were put back exactly as the model "
                "supplied them: " + ", ".join(restored)
                + " — nothing measured them on this run"
            ),
        )
        # Same rule as the strip itself: never hand a caller a Song this pass
        # hasn't re-validated (model_copy skips validators in pydantic v2).
        return (
            Song.model_validate(
                song.model_copy(
                    update={**updates, "provenance": list(song.provenance) + [entry]}
                ).model_dump()
            ),
            restored,
        )


def _strip_model_timing(
    song: Song, *, authority: TimingAuthority, mir: MirAnalysis | None
) -> tuple[Song, TimingStrip | None]:
    """Drop the timing the MODEL supplied, so the deterministic pass the caller
    named can actually write it. Returns ``(song, strip)`` — ``strip`` is
    ``None`` when nothing was stripped (or nothing may be).

    Every timing pass guards its writes with "is this field empty?"
    (``timing/snap.py:303,308``, ``timing/carry_forward.py:272,313,324``), and
    model-supplied timing is never empty — so the model's own numbers silently
    outrank the measured or carried-forward ones. That is the whole bug. The
    fix is not another guard downstream; it is that a model claim never reaches
    a pass as if it were evidence.

    WHICH fields go depends on what the named pass will actually refill.
    Stripping a field nothing else writes just makes coverage worse, so:

    * **``SNAP`` — ``timing.snap`` will run against this run's MIR.** Only the
      RECORDING-level fields go: ``audio.beats`` (blocked by ``if not
      song.audio.beats``), ``metadata.bpm`` (blocked by ``if bpm is None``) and
      ``audio.syncMap`` (which ``snap_chords`` regenerates unconditionally).
      ``audio.beats``/``metadata.bpm`` are stripped only when THIS run's MIR
      actually carries them, so the strip can never empty a field the pass would
      then leave empty.

      Per-element timing (placement/line ``timeSeconds``) is deliberately NOT
      stripped: ``snap_chords`` writes it with no presence guard
      (``timing/snap.py:275,293``) and so already overrules the model whenever
      it has anything to write — but it writes NOTHING when no chord root
      matched the MIR timeline at all, and in that case the model's times are
      the only ones there are. Section spans are NOT stripped either: on this
      path nothing else can write them (``timing/realign.retime_sections`` is
      only reachable from ``realign.py``'s Mode B), so stripping them would
      guarantee ``sectionCoverage=0.00``.

      A ``SNAP`` declaration with no MIR strips nothing: ``snap_chords`` is a
      documented no-op in that case, so there is no authority to strip for.

    * **``CARRY_FORWARD`` — ``timing.carry_forward`` will copy the prior
      version's timing on.** Everything goes, including section spans:
      ``_carry_sections`` (``timing/carry_forward.py:202-206``) requires BOTH
      section times to be ``None`` before it will carry the prior's, so
      stripping is what ENABLES section carry-forward here. The recording-level
      fields come from the prior version or the caller's audio fallback
      (``carry_forward.py:309-325``), which are the same documents the pre-store
      guard compares against, so emptying them here can never outrun a refill.

    * **``NONE``** — nothing is stripped. Either no pass will read this document
      at all (``POST /v1/reconcile``, MCP ``reconcile_song``), or the run reached
      the snap branch with no MIR (a failed analysis, ``listen=off`` with no
      prior to carry): the model is genuinely the only writer and dropping its
      timing would only destroy data.
    """
    if authority is TimingAuthority.NONE:
        return song, None
    if authority is TimingAuthority.SNAP and mir is None:
        # Declared, but `snap_chords` returns the document untouched without a
        # MIR (timing/snap.py:250) — no pass, so no strip.
        return song, None
    recording_fields_only = authority is TimingAuthority.SNAP

    stripped: list[str] = []
    removed: dict = {}
    updates: dict = {}

    if not recording_fields_only:
        new_lines = []
        placements_cleared = 0
        lines_cleared = 0
        for line in song.lines:
            placements = []
            for placement in line.chordPlacements:
                if (
                    placement.timeSeconds is not None
                    or placement.confidence is not None
                    or placement.beat is not None
                ):
                    placements_cleared += 1
                    placement = placement.model_copy(
                        update={"timeSeconds": None, "confidence": None, "beat": None}
                    )
                placements.append(placement)
            line_update: dict = {"chordPlacements": placements}
            if line.timeSeconds is not None or line.confidence is not None:
                lines_cleared += 1
                line_update["timeSeconds"] = None
                line_update["confidence"] = None
            new_lines.append(line.model_copy(update=line_update))
        if placements_cleared or lines_cleared:
            updates["lines"] = new_lines
        if placements_cleared:
            stripped.append(
                f"chordPlacement.timeSeconds/confidence/beat ({placements_cleared})"
            )
        if lines_cleared:
            stripped.append(f"line.timeSeconds/confidence ({lines_cleared})")

        sections_cleared = sum(
            1 for s in song.sections if s.startTime is not None or s.endTime is not None
        )
        if sections_cleared:
            updates["sections"] = [
                s.model_copy(update={"startTime": None, "endTime": None})
                for s in song.sections
            ]
            stripped.append(f"section.startTime/endTime ({sections_cleared})")

    # The recording-level fields, and who can put each one back.
    if authority is TimingAuthority.CARRY_FORWARD:
        # The prior version, or the caller's audio fallback. USUALLY those are
        # the same documents the pre-store guard measures against, so emptying
        # these cannot outrun a refill. Not always: with a request-supplied
        # priorSong and NO stored version, carry-forward's only audio source is
        # that prior, and if it carries `beats: []`/`bpm: null` nothing refills
        # them. That is the right outcome on a listen=off run — nothing measured
        # them — and the guard stays quiet because its baseline is the same
        # bpm-less document. Do not read this as "a refill is guaranteed".
        beats_refillable = bpm_refillable = True
    else:
        # Only THIS run's analysis can, so a field its MIR hasn't got is left
        # alone rather than emptied for nobody.
        beats_refillable = bool(mir and mir.beats)
        bpm_refillable = bool(mir and mir.bpm)

    audio_update: dict = {}
    if song.audio.beats and beats_refillable:
        audio_update["beats"] = []
        removed["audio.beats"] = list(song.audio.beats)
        stripped.append(f"audio.beats ({len(song.audio.beats)} entries)")
    if song.audio.syncMap:
        # Both passes REGENERATE the syncMap from the line times they wrote
        # (snap.py:302, carry_forward.py:310), so this one is never a loss.
        audio_update["syncMap"] = []
        removed["audio.syncMap"] = list(song.audio.syncMap)
        stripped.append(f"audio.syncMap ({len(song.audio.syncMap)} points)")
    if audio_update:
        updates["audio"] = song.audio.model_copy(update=audio_update)
    if song.metadata.bpm is not None and bpm_refillable:
        removed["metadata.bpm"] = song.metadata.bpm
        updates["metadata"] = song.metadata.model_copy(update={"bpm": None})
        stripped.append(f"metadata.bpm ({song.metadata.bpm})")

    if not stripped:
        return song, None
    # model_copy(update=...) does not re-run validators (pydantic v2). Same
    # rule as timing.snap and timing.carry_forward: never hand the pipeline a
    # Song this pass hasn't re-validated.
    return (
        Song.model_validate(song.model_copy(update=updates).model_dump()),
        TimingStrip(
            authority=authority, fields=tuple(stripped), recording=removed
        ),
    )


def _make_snippet(audio_path: str) -> AudioAttachment | None:
    """A short mid-song excerpt (30s from 25% in) as mp3 for providers that
    accept audio input. Failure here never fails the reconciliation."""
    try:
        info = probe(audio_path)
        start = max(info.duration_seconds * 0.25, 0.0)
        end = min(start + 30.0, info.duration_seconds)
        out = Path(tempfile.mkdtemp(prefix="snoocle-snippet-")) / "snippet.mp3"
        trim(audio_path, out, start, end)
        return AudioAttachment(path=str(out))
    except Exception as e:  # noqa: BLE001
        log.warning("audio snippet preparation failed (continuing without): %s", e)
        return None


def _finalize(
    song: Song,
    *,
    song_id: str,
    title: str,
    artist: str,
    youtube_video_id: str | None,
    candidates: list[CandidateSource],
    mir: MirAnalysis | None,
    provider: LLMProvider,
    model: str,
    attempts: int,
    guidance: str | None = None,
    guidance_origin: str | None = None,
    guidance_withheld: str | None = None,
    scope: AnalysisScope | None = None,
    mir_cache=None,
    lyric_refs_resolved: int = 0,
    lyric_overrides: tuple[LyricOverride, ...] = (),
    patch_ops_applied: tuple[AppliedOp, ...] = (),
    quality_feedback: str | None = None,
    structure_feedback: str | None = None,
    timing_authority: TimingAuthority = TimingAuthority.NONE,
    trace: TraceRecorder | None = None,
) -> tuple[Song, TimingStrip | None]:
    """Server-side guardrails + append provenance (never trusted to the LLM).

    ``timing_authority`` enforces the one rule the eleven timing paths had no
    single owner for: the model is never the authority on timing (see
    :func:`_strip_model_timing`). It defaults to ``NONE`` — strip nothing —
    because the safe answer must be the one a caller gets by saying nothing, and
    because the PATCH path must never strip: a patch copies the prior document's
    timing through verbatim and the pipeline skips every timing pass for it, so
    there would be nothing to refill what a strip emptied.

    Returns the song and the :class:`TimingStrip` describing what was removed
    (``None`` when nothing was), which the caller needs to undo the strip if the
    authority it declared does not complete.
    """
    updates: dict = {"id": song_id, "provenance": []}
    if song.displayPreferences.capo != 0:
        log.warning("reconciler set capo=%d; forcing display capo to 0", song.displayPreferences.capo)
        updates["displayPreferences"] = song.displayPreferences.model_copy(update={"capo": 0})
    md_updates = {}
    if song.metadata.title != title:
        md_updates["title"] = title
    if song.metadata.artist != artist:
        md_updates["artist"] = artist
    if md_updates:
        updates["metadata"] = song.metadata.model_copy(update=md_updates)
    if youtube_video_id and song.audio.youtubeVideoId != youtube_video_id:
        updates["audio"] = song.audio.model_copy(update={"youtubeVideoId": youtube_video_id})
    song = song.model_copy(update=updates)

    # The model is not the authority on timing, and this is the last place the
    # rule can be enforced before the deterministic passes read the document.
    song, timing_strip = _strip_model_timing(song, authority=timing_authority, mir=mir)

    prov: list[ProvenanceEntry] = []
    if candidates:
        prov.append(
            ProvenanceEntry(
                timestamp=_now(),
                actor=f"snoocle-server/{__version__}",
                action="discovered-sources",
                sources=[c.url or c.sourceId for c in candidates],
                confidence=round(max(c.confidence for c in candidates), 3),
                notes=f"{len(candidates)} candidate text source(s) gathered via general web search",
            )
        )
    if mir is not None:
        # `timestamp` is when THIS version was produced; `analyzedAt` in the
        # notes is when the audio was actually listened to. On a cache hit
        # those differ, and collapsing them would make a song's history claim
        # a re-listen that never happened. Reading a song must still tell you
        # when its audio was really analyzed.
        reuse_note = ""
        if mir_cache is not None and getattr(mir_cache, "from_cache", False):
            reuse_note = f"; reused from cache (analyzedAt={mir_cache.analyzed_at})"
        prov.append(
            ProvenanceEntry(
                timestamp=_now(),
                actor=f"snoocle-server/{__version__}",
                action="mir-analysis",
                sources=[f"{slot}:{impl}" for slot, impl in mir.engines.items()],
                notes=f"audio-grounded analysis; bpm={mir.bpm}, key={mir.key}{reuse_note}",
            )
        )
    # more independent sources -> higher reconciliation confidence
    conf = min(0.45 + 0.1 * min(len(candidates), 3) + (0.15 if mir else 0.0), 0.9)
    if provider.name == "mock":
        conf = min(conf, 0.5)
    # Guidance is never applied silently: a reader of the song's history must be
    # able to see that a human instruction shaped this run, and whether it came
    # from the request or replayed from the song's stored notes.
    guidance_note = ""
    if guidance:
        guidance_note = f"; guidance applied (from {guidance_origin or 'this request'})"
    # And never WITHHELD silently either — the same claim about the same run.
    # A notes-only run is shown its correction alone (pipeline.py's
    # `model_guidance`), so a standing preference was in force for this version
    # and deliberately not acted on. Without this the only record of that lived
    # in the HTTP response's `steps`, which nothing persists, and a reader six
    # months later would see a version built as if no preference existed.
    if guidance_withheld:
        guidance_note += "; standing preference held back (notes-only run)"
    # Same reasoning for scope: a version produced from the prior song + notes
    # alone is a very different artifact from a full re-analysis, and six
    # months later the history is the only place that difference survives.
    scope_note = f"; scope: {scope.describe()}" if scope is not None else ""
    # How the words got here. A reader of the history should be able to tell
    # "every line was spliced from a source the run actually had" from "some
    # lines were written by the model", without diffing anything.
    lyric_note = ""
    overrides = lyric_overrides
    if lyric_refs_resolved or overrides:
        lyric_note = (
            f"; lyrics spliced from {lyric_refs_resolved} source reference(s)"
            + (f" with {len(overrides)} override(s)" if overrides else "")
        )
    patch_note = (
        f"; patch: {len(patch_ops_applied)} op(s) applied" if patch_ops_applied else ""
    )
    # A retry driven by the quality grader is a different artifact from a first
    # attempt, and the history is the only place that difference survives. It is
    # NOT reported as guidance: no human asked for it.
    quality_note = (
        "; quality retry: the previous attempt's grade and its specific failures "
        "were handed back to the model"
        if quality_feedback
        else ""
    )
    # Mode B consulted the model, which it only does when the deterministic
    # structural comparison found something the stored document cannot explain.
    # Worth its own clause: "this version's repeats were decided by a model"
    # is exactly the kind of thing a reader of the history needs to know.
    structure_note = (
        "; structural re-align: a measured difference between the source document "
        "and the new recording's arrangement was handed to the model"
        if structure_feedback
        else ""
    )
    prov.append(
        ProvenanceEntry(
            timestamp=_now(),
            actor=f"reconcile:{provider.name}/{model}",
            action="reconciled",
            sources=[c.sourceId for c in candidates],
            confidence=round(conf, 3),
            notes=(
                f"attempt(s)={attempts}; chord rule enforced by schema validation"
                + guidance_note
                + scope_note
                + lyric_note
                + patch_note
                + quality_note
                + structure_note
            ),
        )
    )
    # One entry per override, so the escape hatch is auditable line by line
    # rather than a count buried in a summary.
    for override in overrides:
        prov.append(
            ProvenanceEntry(
                timestamp=_now(),
                actor=f"reconcile:{provider.name}/{model}",
                action="lyric-override",
                confidence=None,
                notes=(
                    f"line {override.line_index}: written by the model instead "
                    f"of referenced — {override.reason}"
                ),
            )
        )
    # Same reasoning, one entry per op: a correction history has to be
    # readable op by op, not collapsed into a count.
    for applied in patch_ops_applied:
        prov.append(
            ProvenanceEntry(
                timestamp=_now(),
                actor=f"reconcile:{provider.name}/{model}",
                action="patch-applied",
                confidence=None,
                notes=(
                    applied.description
                    + (f"; reason: {applied.reason}" if applied.reason else "")
                ),
            )
        )
    # A silent strip would trade one invisible write for another. The operator
    # must be able to read, off the song's own history, that the model claimed
    # timing on this run and exactly which claims were dropped — the actor is
    # the SERVER, not the model: this is a server-side normalisation, not
    # something the model was asked to do.
    if timing_strip is not None:
        prov.append(
            ProvenanceEntry(
                timestamp=_now(),
                actor=f"snoocle-server/{__version__}",
                action=TIMING_STRIP_ACTION,
                confidence=None,
                notes=(
                    "the model is not the authority on timing: dropped "
                    + ", ".join(timing_strip.fields)
                    + " from the model's document before the timing pass ran; "
                    + f"{timing_strip.authority_name} decides these fields on this run"
                ),
            )
        )
        if trace is not None:
            trace.step(
                "timing", TIMING_STRIP_ACTION,
                f"dropped model-supplied timing ({len(timing_strip.fields)} field "
                f"group(s)); {timing_strip.authority_name} is the authority",
                detail={
                    "fields": list(timing_strip.fields),
                    "authority": timing_strip.authority_name,
                },
            )
    return song.model_copy(update={"provenance": prov}), timing_strip


def _attempt_patch(
    provider: LLMProvider,
    model: str | None,
    title: str,
    artist: str,
    song_id: str,
    prior_song: dict,
    prior_song_obj: Song,
    guidance: str,
    trace: TraceRecorder | None,
    run_cap: float,
) -> tuple[Song, list[AppliedOp], str, dict] | None:
    """One model call, patch-shaped. Returns ``(song, applied_ops,
    resolved_model, usage)`` on success.

    Returns ``None`` when the model explicitly declined the patch path
    (``needsFullReconcile`` — the correction genuinely needs different words,
    or is otherwise bigger than a targeted fix): the caller falls through to
    the normal reconcile prompt, in the SAME run, rather than failing it.
    That is the visible fallback the brief asks for — not a silent one, and
    not a failure.

    Raises :class:`PatchApplicationError` for anything that actually went
    wrong (malformed JSON, an op that doesn't apply, an unknown op, over the
    cap). Never retried: seeing a WRONG op and being asked to guess again is
    the fuzziness this whole path exists to refuse. One call, one verdict.
    """
    system_prompt = build_patch_system_prompt()
    user_prompt = build_patch_user_prompt(title, artist, song_id, prior_song, guidance)
    started = clock()
    from ..usage import BudgetExceededError, persisted_usage
    if trace is not None and trace.trace.cost_usd >= run_cap:
        raise BudgetExceededError(
            "run", trace.trace.cost_usd, run_cap, refused="patch model call"
        )
    response = provider.complete(system_prompt, [{"role": "user", "text": user_prompt}], model=model)
    usage: dict = {}
    for k, v in (response.usage or {}).items():
        if isinstance(v, (int, float)):
            usage[k] = usage.get(k, 0) + v
    resolved_model = response.model
    if trace is not None:
        step_cost = trace.record_model_usage(resolved_model, response.usage or {})
        trace.step(
            "model", "patch-attempt", "patch model response received",
            detail={
                "usage": persisted_usage(response.usage or {}),
                "costUSD": step_cost,
            },
            duration_seconds=clock() - started,
        )
        if trace.trace.cost_usd > run_cap:
            raise BudgetExceededError(
                "run", trace.trace.cost_usd, run_cap,
                refused="continuation after patch model call",
            )

    try:
        document = json.loads(extract_json(response.text))
    except (ValueError, json.JSONDecodeError) as e:
        if trace is not None:
            trace.step(
                "patch", "patch-parse-failed", f"patch response was not valid JSON: {e}",
                detail={"errors": str(e)[:2000]}, duration_seconds=clock() - started,
            )
        raise PatchApplicationError(f"patch response was not valid JSON: {e}") from e

    if isinstance(document, dict) and document.get("needsFullReconcile"):
        reason = document.get("reason") or "the model reported it needs a full reconcile"
        if trace is not None:
            trace.step(
                "patch", "needs-full-reconcile",
                f"falling back to full reconcile: {reason}",
                detail={"reason": reason}, duration_seconds=clock() - started,
            )
        return None

    try:
        ops = parse_ops_response(document)
        patched_song, applied = apply_patch(prior_song_obj, ops)
    except PatchError as e:
        if trace is not None:
            trace.step(
                "patch", "patch-failed", str(e),
                detail={"errors": str(e)[:2000]}, duration_seconds=clock() - started,
            )
        raise PatchApplicationError(str(e)) from e

    if trace is not None:
        patch_bytes = len(
            json.dumps(
                document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        )
        full_bytes = len(
            json.dumps(
                patched_song.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        )
        trace.set_output_format(
            "patch", patch_bytes=patch_bytes, full_bytes=full_bytes
        )
        trace.step(
            "patch", "patch-applied", f"{len(applied)} op(s) applied",
            detail={"ops": [a.description for a in applied]},
            duration_seconds=clock() - started,
        )
    return patched_song, applied, resolved_model, usage


def reconcile(
    title: str,
    artist: str,
    candidates: list[CandidateSource],
    mir: MirAnalysis | None,
    provider_name: str | None = None,
    model: str | None = None,
    audio_path: str | None = None,
    attach_audio: bool | None = None,
    youtube_video_id: str | None = None,
    song_id: str | None = None,
    media_url: str | None = None,
    trace: TraceRecorder | None = None,
    guidance: str | None = None,
    guidance_origin: str | None = None,
    #: A standing preference that WAS in force for this run and was deliberately
    #: not handed over (a notes-only run sees its correction alone — see
    #: pipeline.py's `model_guidance`). Recorded in provenance, never in a
    #: prompt: the point is that the model does not see it.
    guidance_withheld: str | None = None,
    prior_song: dict | None = None,
    depth: DepthProfile | None = None,
    scope: AnalysisScope | None = None,
    evidence_manifest: dict | None = None,
    mir_cache=None,
    patch_ops_eligible: bool = False,
    quality_feedback: str | None = None,
    structure_feedback: str | None = None,
    timing_authority: TimingAuthority = TimingAuthority.NONE,
) -> ReconcileResult:
    """Reconcile the run's evidence into a validated Song.

    ``timing_authority`` is the caller's declaration of WHICH deterministic
    timing pass will run against the returned document (:class:`TimingAuthority`).
    It defaults to ``NONE`` — nothing is stripped — because a caller that runs no
    timing pass must get the model's timing back untouched: an entry point that
    returns the document to a client (``POST /v1/reconcile``, MCP
    ``reconcile_song``) has no pass to hand authority to, and stripping there
    dropped ``metadata.bpm``/``audio.syncMap`` for nobody. Only a caller that
    OWNS a pass may declare it, and if that pass then skips or raises the caller
    puts the fields back from ``result.timing_strip``.
    """
    song_id = song_id or slugify_song_id(artist, title)
    require_resolved_song_id(song_id)
    provider = get_provider(provider_name)
    depth = depth or resolve_depth(None)

    # Operator agent config (runtime-editable instructions/tooling). A store
    # failure degrades to built-in defaults — observability config must never
    # fail a reconciliation.
    agent_config = _load_agent_config()
    if trace is not None:
        trace.trace.config_version = config_version(agent_config)

    # The mock provider is a deterministic offline reconciler: it can synthesize
    # a small Song from title/artist alone, so it never requires inputs. Every
    # other provider needs something concrete to reconcile — and a PRIOR SONG
    # counts: the notes-only scope (listen=off, reconcile=off) deliberately
    # gathers nothing and asks the model to correct what the user already has.
    # Without a prior song that scope would have the model invent a song from
    # the title alone, which is exactly the hallucination this guard exists to
    # prevent, so it still fails here.
    if not candidates and mir is None and prior_song is None and provider.name != "mock":
        raise ReconcileError(
            "nothing to reconcile: no candidate sources and no MIR analysis"
            + (" and no prior song to correct" if scope is not None else "")
        )

    if media_url is None and youtube_video_id:
        media_url = f"https://www.youtube.com/watch?v={youtube_video_id}"

    # The patch protocol (patch_ops.py): a targeted correction in notes-only
    # scope doesn't reconcile anything — it names exact changes, and this
    # replaces the WHOLE reconcile prompt with a much smaller, closed-set
    # ask. `supports_patch_ops` mirrors `emits_lyric_refs`: the contract is
    # between this server's prompt and a model actually told it, and
    # providers with their own prompt pipeline (the in-process agent, which
    # ignores the `system` argument entirely and builds its own) or none at
    # all (mock) are not party to it.
    prior_song_obj: Song | None = None
    if prior_song:
        try:
            prior_song_obj = Song.model_validate(prior_song)
        except Exception:  # noqa: BLE001 — a malformed prior can't be patched;
            prior_song_obj = None  # demote to the full-reconcile path below.
    uses_patch_ops = bool(
        scope is not None and scope.notes_only
        and patch_ops_eligible
        and prior_song_obj is not None
        and getattr(provider, "supports_patch_ops", False)
    )

    if uses_patch_ops:
        run_cap = (
            agent_config.run_cost_cap_usd
            if agent_config.run_cost_cap_usd is not None
            else settings.run_cost_cap_usd
        )
        if trace is not None:
            trace.step(
                "inputs", "read-inputs",
                f"patch mode: correcting {song_id!r} in place; human corrections attached",
                detail={"provider": provider.name, "mode": "patch", "guidance": guidance},
            )
        attempt = _attempt_patch(
            provider, model, title, artist, song_id, prior_song, prior_song_obj,
            guidance or "", trace, run_cap,
        )
        if attempt is not None:
            patched_song, applied_ops, resolved_model, usage = attempt
            # No `timing_authority` whatever the caller declared: a patch
            # regenerated nothing, and the pipeline skips every timing pass for
            # it (pipeline.py's `patched -> skip entirely`), so there would be
            # nothing to refill what a strip emptied.
            song, _ = _finalize(
                patched_song,
                song_id=song_id,
                title=title,
                artist=artist,
                youtube_video_id=youtube_video_id,
                candidates=[],
                mir=None,
                provider=provider,
                model=resolved_model,
                attempts=1,
                guidance=guidance,
                guidance_origin=guidance_origin,
                guidance_withheld=guidance_withheld,
                scope=scope,
                patch_ops_applied=tuple(applied_ops),
            )
            if trace is not None:
                trace.step(
                    "final", "reconciled",
                    f"produced Song via patch: {len(applied_ops)} op(s) applied",
                    detail={"opsApplied": len(applied_ops), "usage": usage},
                )
            return ReconcileResult(
                song=song,
                provider=provider.name,
                model=resolved_model,
                attempts=1,
                audio_attached=False,
                usage=usage,
                trace=trace.trace if trace is not None else None,
                patch_ops_applied=len(applied_ops),
                output_format="patch",
            )
        # else: the model explicitly asked for a full reconcile. Falls
        # through to EXACTLY today's notes-only path below (full-Song
        # regeneration via the lyric-reference protocol) — visibly (the
        # trace step _attempt_patch already recorded), not silently.

    # The lyric-reference protocol (lyric_refs.py). `emits_lyric_refs` is a
    # PROVIDER capability, not a global switch: the contract is between this
    # server's prompt and the model that reads it, so a provider opts in only
    # once it is actually told the contract. The deterministic mock builds its
    # Song in local code and is never prompted at all.
    uses_lyric_refs = bool(getattr(provider, "emits_lyric_refs", False))
    ref_index = build_ref_index(candidates, prior_song) if uses_lyric_refs else {}
    base_schema = song_json_schema()
    uses_reconcile_delta = bool(prior_song_obj is not None and uses_lyric_refs)
    output_format = "patch" if uses_reconcile_delta else "full"
    if uses_reconcile_delta:
        schema_for_model = reconcile_delta_json_schema()
    else:
        schema_for_model = strip_postpass_schema(base_schema, mir_present=mir is not None)
        if uses_lyric_refs:
            schema_for_model = agent_song_json_schema(schema_for_model)
    if trace is not None:
        trace.set_output_format(output_format)

    if trace is not None:
        # The FULL MIR snapshot rides on the run itself (un-truncated; the GUI
        # timeline renders from it). The inputs step keeps only a compact
        # summary — putting the whole timeline in a step detail would silently
        # hit the 50-item truncation cap.
        if mir is not None:
            trace.attach_mir(mir.to_run_payload())
        trace.step(
            "inputs", "read-inputs",
            f"{len(candidates)} text source(s), "
            + (f"MIR key={mir.key} bpm={mir.bpm}" if mir else "no MIR")
            + (f"; human corrections attached" if guidance else "")
            + ("; quality retry" if quality_feedback else "")
            + ("; structural re-align" if structure_feedback else ""),
            detail={
                "provider": provider.name,
                "depth": depth.name,
                "qualityRetry": bool(quality_feedback),
                "structuralRealign": bool(structure_feedback),
                "candidateSources": [c.url or c.sourceId for c in candidates],
                "mir": (
                    {
                        "engines": mir.engines,
                        "estimatedKey": mir.key,
                        "bpm": mir.bpm,
                        "durationSeconds": mir.duration_seconds,
                        "chordSegments": len(mir.chords),
                        "beatCount": len(mir.beats),
                        "analyzedWindows": [
                            {"start": w.start, "end": w.end} for w in mir.analyzed_windows
                        ],
                    }
                    if mir is not None
                    else None
                ),
                "guidance": guidance,
                "scope": scope.describe() if scope is not None else None,
            },
        )

    # Context-driven providers (mock, agent) consume the structured inputs
    # directly instead of the rendered prompt text. The trace + depth overrides
    # ride along so an agentic provider records its own turns/tool calls into
    # this run's timeline and honors the requested effort/budget.
    if getattr(provider, "wants_context", False):
        provider.context = {
            "title": title,
            "artist": artist,
            "song_id": song_id,
            "youtube_video_id": youtube_video_id,
            "media_url": media_url,
            "candidates": candidates,
            "mir": mir,
            "song_schema": schema_for_model,
            "ref_index": ref_index,
            "audio_path": audio_path,
            "guidance": guidance,
            "prior_song": prior_song,
            "depth": depth,
            "scope": scope,
            "agent_config": agent_config if not agent_config.is_default() else None,
            "evidence_manifest": evidence_manifest,
            "recording_variant": (
                ((evidence_manifest or {}).get("request") or {}).get("recordingVariant")
                or "studio"
            ),
            "quality_feedback": quality_feedback,
            "structure_feedback": structure_feedback,
            "output_format": output_format,
        }
    if hasattr(provider, "trace"):
        provider.trace = trace

    audio: AudioAttachment | None = None
    attach = settings.llm_audio_snippet if attach_audio is None else attach_audio
    if attach and audio_path and provider.supports_audio:
        audio = _make_snippet(audio_path)

    user_prompt = build_user_prompt(
        title, artist, candidates, mir, schema_for_model, song_id, youtube_video_id,
        guidance=guidance, prior_song=prior_song, time_align=depth.time_align,
        scope=scope, evidence_manifest=evidence_manifest,
        ref_index=ref_index if uses_lyric_refs else None,
        quality_feedback=quality_feedback,
        structure_feedback=structure_feedback,
        output_format=output_format,
    )
    system_prompt = build_system_prompt(
        lyric_refs=uses_lyric_refs, output_format=output_format
    )
    turns: list[dict] = [{"role": "user", "text": user_prompt}]

    usage: dict = {}
    resolved_model = model or settings.llm_model or provider.default_model
    last_errors = ""
    max_attempts = 2 if uses_reconcile_delta else settings.llm_repair_attempts + 1
    for attempt in range(1, max_attempts + 1):
        from ..usage import BudgetExceededError
        run_cap = (
            agent_config.run_cost_cap_usd
            if agent_config.run_cost_cap_usd is not None
            else settings.run_cost_cap_usd
        )
        if trace is not None and trace.trace.cost_usd >= run_cap:
            raise BudgetExceededError(
                "run", trace.trace.cost_usd, run_cap, refused=f"model attempt {attempt}"
            )
        started = clock()
        response = provider.complete(
            system_prompt, turns, model=model, audio=audio if attempt == 1 else None
        )
        if trace is not None and not getattr(provider, "records_usage_in_trace", False):
            step_cost = trace.record_model_usage(response.model, response.usage or {})
            from ..usage import persisted_usage
            trace.step(
                "model", f"attempt-{attempt}", "model response received",
                detail={
                    "attempt": attempt,
                    "usage": persisted_usage(response.usage or {}),
                    "costUSD": step_cost,
                },
                duration_seconds=clock() - started,
            )
            if trace.trace.cost_usd > run_cap:
                raise BudgetExceededError(
                    "run", trace.trace.cost_usd, run_cap,
                    refused=f"continuation after model attempt {attempt}",
                )
        for k, v in (response.usage or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
        resolved_model = response.model
        overrides: tuple[LyricOverride, ...] = ()
        refs_resolved = 0
        applied_delta: AppliedDelta | None = None
        try:
            document = json.loads(extract_json(response.text))
            if uses_reconcile_delta:
                applied_delta = apply_reconcile_delta(
                    prior_song_obj, document, ref_index
                )
                song = applied_delta.song
                overrides = applied_delta.lyric_overrides
                refs_resolved = applied_delta.lyric_refs_resolved
            elif uses_lyric_refs:
                # Splice BEFORE schema validation: the Song schema knows
                # nothing about refs, and its charIndex rule only means
                # something once the real text is in place.
                spliced = splice_lyrics(document, ref_index)
                document = spliced.document
                overrides = spliced.overrides
                refs_resolved = spliced.refs_resolved
                song = Song.model_validate(document)
            else:
                song = Song.model_validate(document)
        except UnresolvableLyricRefError as e:
            # NOT repaired. A lyric with no valid provenance must not reach
            # the store, and a retry is an invitation to supply one from
            # memory instead — see lyric_refs.py rules 3 and 4.
            if trace is not None:
                trace.step(
                    "repair", f"lyric-refs-attempt-{attempt}",
                    "unresolvable lyric reference — failing the run",
                    detail={"errors": str(e)[:2000]},
                    duration_seconds=clock() - started,
                )
            raise LyricProvenanceError(str(e)) from e
        except (
            ValidationError,
            ValueError,
            json.JSONDecodeError,
            LyricSpliceError,
            ReconcileDeltaError,
        ) as e:
            last_errors = str(e)[:4000]
            log.info("reconcile attempt %d failed validation: %s", attempt, last_errors[:300])
            if trace is not None:
                trace.step(
                    "repair", f"validate-attempt-{attempt}",
                    f"attempt {attempt} failed schema validation — repairing",
                    detail={"errors": last_errors[:2000]},
                    duration_seconds=clock() - started,
                )
            turns.append({"role": "assistant", "text": response.text})
            turns.append({
                "role": "user",
                "text": (
                    build_delta_repair_prompt(last_errors)
                    if uses_reconcile_delta
                    else build_repair_prompt(last_errors, uses_lyric_refs)
                ),
            })
            continue
        song, timing_strip = _finalize(
            song,
            song_id=song_id,
            title=title,
            artist=artist,
            youtube_video_id=youtube_video_id,
            candidates=candidates,
            mir=mir,
            provider=provider,
            model=resolved_model,
            attempts=attempt,
            guidance=guidance,
            guidance_origin=guidance_origin,
            guidance_withheld=guidance_withheld,
            scope=scope,
            mir_cache=mir_cache,
            lyric_refs_resolved=refs_resolved,
            lyric_overrides=overrides,
            quality_feedback=quality_feedback,
            structure_feedback=structure_feedback,
            # Whatever the CALLER said will time this document. Never inferred
            # from `mir is not None`: that is a fact about this run's inputs,
            # and two public entry points hand a MIR in and then run no timing
            # pass at all. A caller that declares nothing strips nothing.
            timing_authority=timing_authority,
            trace=trace,
        )
        if trace is not None:
            if applied_delta is not None:
                trace.set_output_format(
                    "patch",
                    patch_bytes=applied_delta.patch_bytes,
                    full_bytes=applied_delta.full_bytes,
                )
            else:
                full_bytes = len(
                    json.dumps(
                        song.model_dump(mode="json"),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode()
                )
                trace.set_output_format("full", full_bytes=full_bytes)
            trace.step(
                "final", "reconciled",
                f"produced Song: {len(song.lines)} lines, "
                f"{sum(len(l.chordPlacements) for l in song.lines)} chords, "
                f"{len(song.sections)} sections (attempt {attempt})",
                detail={
                    "attempts": attempt,
                    "lineCount": len(song.lines),
                    "chordCount": sum(len(l.chordPlacements) for l in song.lines),
                    "syncMapPoints": len(song.audio.syncMap),
                    "lyricRefsResolved": refs_resolved if uses_lyric_refs else None,
                    "lyricOverrides": [
                        {"lineIndex": o.line_index, "reason": o.reason} for o in overrides
                    ],
                    "usage": usage,
                    "outputFormat": output_format,
                    "patchSizeVsFull": (
                        {
                            "patchBytes": applied_delta.patch_bytes,
                            "fullBytes": applied_delta.full_bytes,
                            "ratio": round(applied_delta.ratio, 4),
                        }
                        if applied_delta is not None else None
                    ),
                },
                duration_seconds=clock() - started,
            )
        return ReconcileResult(
            song=song,
            provider=provider.name,
            model=resolved_model,
            attempts=attempt,
            audio_attached=audio is not None,
            usage=usage,
            trace=trace.trace if trace is not None else None,
            timing_strip=timing_strip,
            output_format=output_format,
            patch_size_vs_full=(
                {
                    "patchBytes": applied_delta.patch_bytes,
                    "fullBytes": applied_delta.full_bytes,
                    "ratio": round(applied_delta.ratio, 4),
                }
                if applied_delta is not None else None
            ),
        )

    raise ReconcileError(
        f"reconciliation failed schema validation after "
        f"{max_attempts} attempts; last errors: {last_errors[:1000]}"
    )
