#!/usr/bin/env python3
"""Run the fixed Back to Black deterministic-first acceptance benchmark."""

from __future__ import annotations

import argparse
import json
import time

from snoocle_server.deterministic_process import process_song_deterministically_service


DEFAULT_RECORDING_ID = "TJAfLE39ZZ8"


def _elapsed(stages: list[dict], *names: str) -> float:
    wanted = set(names)
    return round(sum(float(stage["elapsedMs"]) for stage in stages if stage["name"] in wanted), 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-id", default=DEFAULT_RECORDING_ID)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        result = process_song_deterministically_service(
            title="Back to Black",
            artist="Amy Winehouse",
            recording_id=args.recording_id,
            use_lrc=True,
            selection_strategy="strict",
            mir_accuracy="standard",
            refresh_mir_cache=args.refresh_cache,
            refresh_discovery_cache=args.refresh_cache,
        )
        payload = result.to_dict()
        stages = payload["stages"]
        quality = payload.get("quality") or {}
        grade = quality.get("grade") or {}
        attribution = quality.get("attribution") or {}
        record = {
            "benchmark": "back-to-black-deterministic-first",
            "recordingId": args.recording_id,
            "agentPolicy": "never",
            "status": result.status,
            "reason": result.reason,
            "acquisitionCacheStatus": payload["cache"].get("audio", "not_applicable"),
            "mirCacheStatus": payload["cache"].get("mir", "not_applicable"),
            "discoveryCacheStatus": payload["cache"].get("discovery", "not_applicable"),
            "acquisitionMs": _elapsed(stages, "acquire_audio"),
            "mirMs": _elapsed(stages, "mir"),
            "baselineMs": _elapsed(stages, "candidate_selection", "baseline"),
            "deterministicAlignmentMs": _elapsed(
                stages, "snap_chords", "lrc_alignment", "section_timing",
                "collapse_guard", "confidence_scoring", "quality_grading",
            ),
            "qualityScore": grade.get("overall"),
            "qualityVerdict": grade.get("verdict"),
            "faultAttribution": attribution.get("fault"),
            "modelCalls": 0,
            "modelCostUSD": 0,
            "interventionRequired": result.status != "completed",
            "totalElapsedMs": round((time.perf_counter() - started) * 1000, 3),
        }
    except Exception as error:  # noqa: BLE001 - failure is benchmark evidence
        record = {
            "benchmark": "back-to-black-deterministic-first",
            "recordingId": args.recording_id,
            "agentPolicy": "never",
            "status": "failed",
            "reason": type(error).__name__,
            "error": str(error),
            "acquisitionCacheStatus": "unknown",
            "mirCacheStatus": "not_run",
            "discoveryCacheStatus": "not_run",
            "acquisitionMs": None,
            "mirMs": None,
            "baselineMs": None,
            "deterministicAlignmentMs": None,
            "qualityScore": None,
            "qualityVerdict": None,
            "faultAttribution": None,
            "modelCalls": 0,
            "modelCostUSD": 0,
            "interventionRequired": True,
            "totalElapsedMs": round((time.perf_counter() - started) * 1000, 3),
        }
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] in {"completed", "needs_review"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
