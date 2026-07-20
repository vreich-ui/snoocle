"""Content metrics comparing a candidate Song to a gold (ground-truth) Song.

Everything here is pure and deterministic — no model, no network — so the eval
harness is fast and unit-testable. Songs may be passed as pydantic ``Song``
models or plain dicts; both are normalized to dicts first.

Metrics (all similarities are 0..1, higher = better):
- chordSimilarity      full chord-symbol sequence agreement (root + quality)
- chordRootSimilarity  same, roots only (forgives extension disagreements)
- lyricSimilarity      1 - word error rate over the concatenated lyrics
- sectionSimilarity    ordered section-kind/name sequence agreement
- timingMAE            mean abs error of syncMap line times (seconds) or None
- overall              weighted mean of the available similarities
"""

from __future__ import annotations

import difflib
import re

_ROOT_RE = re.compile(r"^([A-G][#b]?)")


def _as_dict(song) -> dict:
    if hasattr(song, "model_dump"):
        return song.model_dump(mode="json")
    return dict(song)


def _chord_sequence(song: dict) -> list[str]:
    """Every chord symbol in reading order (line, then ascending charIndex)."""
    out: list[str] = []
    for line in song.get("lines") or []:
        placements = sorted(
            line.get("chordPlacements") or [], key=lambda p: p.get("charIndex", 0)
        )
        out.extend(str(p.get("chord", "")) for p in placements)
    return out


def _root(chord: str) -> str:
    m = _ROOT_RE.match(chord or "")
    return m.group(1) if m else chord


def _words(song: dict) -> list[str]:
    text = " ".join((line.get("lyrics") or "") for line in song.get("lines") or [])
    return re.findall(r"[^\s]+", text.lower())


def _section_kinds(song: dict) -> list[str]:
    out = []
    for s in song.get("sections") or []:
        out.append(str(s.get("kind") or s.get("name") or s.get("label") or "").lower())
    return out


def _seq_ratio(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _word_error_rate(pred: list[str], gold: list[str]) -> float:
    """Levenshtein distance over word lists / gold length (classic WER)."""
    if not gold:
        return 0.0 if not pred else 1.0
    # DP edit distance
    prev = list(range(len(pred) + 1))
    for gi, gw in enumerate(gold, start=1):
        curr = [gi]
        for pi, pw in enumerate(pred, start=1):
            cost = 0 if pw == gw else 1
            curr.append(min(prev[pi] + 1, curr[pi - 1] + 1, prev[pi - 1] + cost))
        prev = curr
    return prev[-1] / len(gold)


def _sync_map(song: dict) -> dict[int, float]:
    return {
        p["lineIndex"]: p["time"]
        for p in ((song.get("audio") or {}).get("syncMap") or [])
        if "lineIndex" in p and "time" in p
    }


def _timing_mae(pred: dict, gold: dict) -> float | None:
    a, b = _sync_map(pred), _sync_map(gold)
    common = set(a) & set(b)
    if not common:
        return None
    return sum(abs(a[i] - b[i]) for i in common) / len(common)


def score_song(candidate, gold) -> dict:
    """Compare a candidate Song to a gold Song; return a metrics dict."""
    cand, gold = _as_dict(candidate), _as_dict(gold)

    c_chords, g_chords = _chord_sequence(cand), _chord_sequence(gold)
    chord_sim = _seq_ratio(c_chords, g_chords)
    root_sim = _seq_ratio([_root(c) for c in c_chords], [_root(c) for c in g_chords])

    wer = _word_error_rate(_words(cand), _words(gold))
    lyric_sim = max(0.0, 1.0 - wer)

    section_sim = _seq_ratio(_section_kinds(cand), _section_kinds(gold))
    timing_mae = _timing_mae(cand, gold)

    # Weighted overall over the similarities that apply. Timing only counts when
    # both songs carry a syncMap (else its weight is redistributed).
    parts = [(chord_sim, 0.4), (lyric_sim, 0.3), (section_sim, 0.2)]
    if timing_mae is not None:
        # map MAE (seconds) to a 0..1 score: 0s -> 1.0, >=8s -> 0.0
        parts.append((max(0.0, 1.0 - timing_mae / 8.0), 0.1))
    total_w = sum(w for _, w in parts)
    overall = sum(v * w for v, w in parts) / total_w if total_w else 0.0

    return {
        "chordSimilarity": round(chord_sim, 4),
        "chordRootSimilarity": round(root_sim, 4),
        "lyricSimilarity": round(lyric_sim, 4),
        "lyricWER": round(wer, 4),
        "sectionSimilarity": round(section_sim, 4),
        "timingMAE": round(timing_mae, 3) if timing_mae is not None else None,
        "overall": round(overall, 4),
        "counts": {
            "candLines": len(cand.get("lines") or []),
            "goldLines": len(gold.get("lines") or []),
            "candChords": len(c_chords),
            "goldChords": len(g_chords),
        },
    }
