"""Classify reconciliation guidance to decide HOW a run should proceed.

A targeted correction ("change the C to a B in line 12") and an open-ended
re-analysis request ("re-check this song") are different tasks, and treating
them the same is expensive and dangerous: the targeted one needs no new
sources — everything it refers to is already in the stored document — yet
without this module the caller had to know that and set scope explicitly.
Left to infer nothing, a request naming one chord still re-discovers sources,
hands the model the whole song, and asks it to reproduce every lyric
verbatim — which is what a real run did today, and what got blocked by a
content filter over a single chord.

Two independent questions, both answered here:

1. **Is this a targeted correction at all** (:attr:`is_targeted_correction`)?
   If so, the pipeline should infer ``scope`` = notes-only (listen=off,
   reconcile=off) when the caller didn't specify one — see
   ``pipeline.py``'s scope-inference step. An EXPLICIT caller scope always
   wins over anything this module decides; it is only ever consulted when
   ``scope is None``.
2. **Does it need different WORDS** (:attr:`targets_lyrics`)? A patch op
   never carries lyric text (see ``reconcile/patch_ops.py``), so a
   correction that targets lyrics cannot go through the patch path even
   inside a notes-only run — it must fall back to the existing full-Song
   notes-only reconcile (the lyric-reference protocol), visibly, not by
   quietly attempting and failing a patch.

Deterministic rules run first and are cheap (no network, no model): a
mention of a chord symbol, a line reference, a section reference, or lyric
vocabulary each independently signals "this names something specific to
change". Only when NONE of them fire does a single cheap LLM call decide
whether the guidance is a targeted correction at all — and even then, the
LLM tier can only ever grant :attr:`is_targeted_correction`, never
:attr:`patch_eligible`: telling a patch apart from a full correction from a
generic yes/no isn't reliable enough for a path whose whole point is "lyrics
never leak into an op".
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, Optional

from .chords import ChordParseError, parse_chord
from .config import settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorrectionClassification:
    is_targeted_correction: bool
    targets_lyrics: bool
    matched_rules: tuple[str, ...] = ()
    via_llm: bool = False

    @property
    def patch_eligible(self) -> bool:
        """Chords/lines/sections were named, lyrics were not, and a
        deterministic rule is what decided it — never the LLM tier (see the
        module docstring)."""
        return self.is_targeted_correction and not self.targets_lyrics and not self.via_llm

    def describe(self) -> str:
        rule = "+".join(self.matched_rules) if self.matched_rules else "none"
        return f"{'targeted' if self.is_targeted_correction else 'open-ended'} (rule={rule})"


# --- deterministic rules ------------------------------------------------------

_LINE_RE = re.compile(r"\bline\s*#?\s*\d+\b", re.IGNORECASE)

# Natural-language section vocabulary. Deliberately a fixed word list rather
# than the schema's SectionKind values verbatim: guidance is prose ("the
# bridge", "pre-chorus"), not a schema enum.
_SECTION_WORD_RE = re.compile(
    r"\b(?:section\s*#?\s*\d+|section|intro(?:duction)?|verse|pre-?chorus|"
    r"chorus|post-?chorus|bridge|solo|instrumental|interlude|breakdown|outro)\b",
    re.IGNORECASE,
)

_LYRIC_WORD_RE = re.compile(r"\b(?:lyrics?|words?)\b", re.IGNORECASE)

# A candidate chord token: a root letter, optional accidental, optional
# quality/extension/alteration text, optional slash bass. Deliberately loose
# — `parse_chord` is the real filter; this regex just finds candidates
# cheaply. Case-SENSITIVE on purpose: a chord symbol's root is always
# capitalized by convention (chords.py's own root regex requires it too), so
# a lowercase "do"/"go" mid-sentence never even becomes a candidate — only a
# sentence-initial "Do"/"Go" can, which is the one case the stoplist below
# still has to cover.
_CHORD_TOKEN_RE = re.compile(r"\b[A-G](?:#|b)?(?:[A-Za-z0-9()+°Δø]*)?(?:/[A-G](?:#|b)?)?\b")

# "C to B", "C to a B", "from C to Bm7" — an explicit replacement shape.
# Reinforces a stoplisted word used the same way a real replacement would be.
_CHORD_REPLACEMENT_RE = re.compile(
    r"\b([A-G](?:#|b)?[A-Za-z0-9()+°Δø]*)\b\s+to\s+(?:an?\s+)?\b([A-G](?:#|b)?[A-Za-z0-9()+°Δø]*)\b",
)

# Ordinary English words that ALSO happen to parse as a valid chord symbol
# once capitalized at a sentence's start: "Do" -> D° (the jazz "o" = "dim"
# alias), "Go" -> G°, "Am" -> A minor, "No"/"So"/"Be" similarly collide with
# their own quality/root readings. A guidance note that opens "Do the chords
# in verse 2 sound right?" or "Am I reading this correctly?" must not count
# as naming a chord on the strength of that one word alone.
_AMBIGUOUS_CHORD_WORDS = {"do", "go", "am", "no", "so", "be"}


def _is_chord_token(token: str) -> bool:
    try:
        parse_chord(token)
    except ChordParseError:
        return False
    return True


def _mentions_chord(text: str) -> bool:
    """A chord mention counts when a non-stoplisted token parses as a chord,
    OR the word "chord"/"chords" appears anywhere, OR a stoplisted word
    appears in an explicit "X to Y" replacement shape (which is real support,
    not just the word alone)."""
    has_chord_word = bool(re.search(r"\bchords?\b", text, re.IGNORECASE))
    for m in _CHORD_TOKEN_RE.finditer(text):
        token = m.group(0)
        if token.lower() in _AMBIGUOUS_CHORD_WORDS:
            continue
        if _is_chord_token(token):
            return True
    if has_chord_word:
        return True
    for m in _CHORD_REPLACEMENT_RE.finditer(text):
        if _is_chord_token(m.group(1)) and _is_chord_token(m.group(2)):
            return True
    return False


def _deterministic_signals(guidance: str) -> tuple[bool, bool, list[str]]:
    """(targets_structure, targets_lyrics, matched_rule_names)."""
    rules: list[str] = []
    targets_structure = False
    targets_lyrics = False

    if _mentions_chord(guidance):
        targets_structure = True
        rules.append("chord-symbol")
    if _LINE_RE.search(guidance):
        targets_structure = True
        rules.append("line-reference")
    if _SECTION_WORD_RE.search(guidance):
        targets_structure = True
        rules.append("section-reference")
    if _LYRIC_WORD_RE.search(guidance):
        targets_lyrics = True
        rules.append("lyric-word")

    return targets_structure, targets_lyrics, rules


# --- LLM tier (only when no deterministic rule fires) -------------------------

_LLM_SYSTEM = """\
You classify a correction note attached to a re-analysis request for a \
chord/lyric song document.

Return ONLY a JSON object, no prose and no code fences:
{"isTargetedCorrection": bool, "confidence": float}

"isTargetedCorrection" is true when the note asks to fix or change specific, \
already-present content in the existing document (a wrong chord, a mislabeled \
section, an incorrect line) WITHOUT needing new research. It is false when \
the note asks for a broader re-analysis, re-verification against sources, or \
says the existing document is unreliable in a way that isn't a targeted fix.

"confidence" is your genuine 0-1 certainty in that classification."""


def _llm_classify(guidance: str) -> Optional[dict]:
    """One cheap call to the configured Haiku. None on any failure — an
    inconclusive classification must default to NOT inferring scope, never
    to guessing."""
    from .reconcile.providers import ProviderError, get_provider

    if not settings.provider_key("anthropic"):
        log.info("correction_routing: no anthropic key configured; skipping LLM tier")
        return None
    try:
        provider = get_provider("anthropic")
        response = provider.complete(
            _LLM_SYSTEM,
            [{"role": "user", "text": f"Correction note: {guidance!r}"}],
            model=settings.identity_model,
            max_tokens=settings.identity_max_tokens,
        )
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()
        return json.loads(text)
    except (ProviderError, json.JSONDecodeError, ValueError, KeyError) as e:
        log.warning("correction_routing: LLM classification failed: %s", e)
        return None
    except Exception as e:  # noqa: BLE001 — classification must never hard-fail
        log.warning("correction_routing: LLM classification errored: %s", e)
        return None


def classify_correction(
    guidance: str,
    llm: Optional[Callable[[str], Optional[dict]]] = None,
) -> CorrectionClassification:
    """Classify a guidance string. Never raises — an unclassifiable note is
    just :attr:`CorrectionClassification.is_targeted_correction` = False,
    which leaves scope exactly as the caller (or its absence) implies.

    ``llm`` is injectable so tests can drive the fallback tier without a
    network call; it defaults to :func:`_llm_classify`.
    """
    text = (guidance or "").strip()
    if not text:
        return CorrectionClassification(is_targeted_correction=False, targets_lyrics=False)

    targets_structure, targets_lyrics, rules = _deterministic_signals(text)
    if targets_structure or targets_lyrics:
        return CorrectionClassification(
            is_targeted_correction=True,
            targets_lyrics=targets_lyrics,
            matched_rules=tuple(rules),
        )

    payload = (llm or _llm_classify)(text)
    if payload is not None:
        try:
            is_targeted = bool(payload.get("isTargetedCorrection"))
            confidence = float(payload.get("confidence"))
        except (TypeError, ValueError):
            is_targeted = False
            confidence = 0.0
        if is_targeted and confidence >= settings.identity_min_confidence:
            return CorrectionClassification(
                is_targeted_correction=True,
                targets_lyrics=False,  # unknown from this tier — see patch_eligible
                matched_rules=("llm",),
                via_llm=True,
            )

    return CorrectionClassification(is_targeted_correction=False, targets_lyrics=False)
