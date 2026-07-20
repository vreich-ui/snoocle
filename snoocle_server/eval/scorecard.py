"""Scorecard assembly — score every gold-marked song against its gold.

Extracted so the REST route (`GET /v1/eval/scorecard`) and the MCP tool
(`get_scorecard`) build the identical report from one place.
"""

from __future__ import annotations

from .metrics import score_song


def _aggregate_scores(metrics: list[dict]) -> dict:
    if not metrics:
        return {}
    keys = ["chordSimilarity", "chordRootSimilarity", "lyricSimilarity",
            "sectionSimilarity", "overall"]
    out = {k: round(sum(m[k] for m in metrics) / len(metrics), 4) for k in keys}
    timings = [m["timingMAE"] for m in metrics if m.get("timingMAE") is not None]
    out["timingMAE"] = round(sum(timings) / len(timings), 3) if timings else None
    return out


def build_scorecard(store, process_metrics=None) -> dict:
    """Score each gold-marked song's current version vs its gold, newest-worst
    first, with an aggregate. ``store`` is the song repository (passed in so the
    caller controls which one); ``process_metrics(song_id) -> dict`` optionally
    attaches per-song run cost/effort — pass None to omit it."""
    from ..store.base import StoreError
    from ..store.evals import get_eval_store

    rows = []
    for rec in get_eval_store().list_gold():
        song_id = rec["songId"]
        gold_version = rec.get("goldVersion")
        try:
            gold = store.get(song_id, version=gold_version)
            cand = store.get(song_id)  # current
        except StoreError:
            continue
        row = {
            "songId": song_id,
            "goldVersion": gold_version,
            "currentVersion": store.current_version(song_id),
            "metrics": score_song(cand, gold),
        }
        if process_metrics is not None:
            row["process"] = process_metrics(song_id)
        rows.append(row)
    rows.sort(key=lambda r: r["metrics"]["overall"])
    return {
        "count": len(rows),
        "aggregate": _aggregate_scores([r["metrics"] for r in rows]),
        "songs": rows,
    }
