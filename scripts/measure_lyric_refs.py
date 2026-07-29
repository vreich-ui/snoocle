#!/usr/bin/env python3
"""Measure what the lyric-reference protocol costs the model to emit.

The claim behind reconcile/lyric_refs.py is that lyrics are most of the
reconciliation's OUTPUT payload, so replacing them with references should cut
output tokens — and with them the wall-clock that pushes long songs into the
900s reconcile timeout. This script measures the first half of that claim
exactly, on a real song document, with no network and no model:

    .venv/bin/python scripts/measure_lyric_refs.py
    .venv/bin/python scripts/measure_lyric_refs.py --song-id ccr--have-you-ever-seen-the-rain
    .venv/bin/python scripts/measure_lyric_refs.py --song-file some-song.json

With no arguments it builds a song from the repo's own chord-sheet fixture and
scales it to a realistic length (--lines, default 48 — about one pop song).

What is measured: the exact JSON the model must produce, in both formats. The
token figures are ESTIMATES (chars/4, the usual English-text rule of thumb) —
this runs offline, so there is no tokenizer call. The character counts and the
ratio between them are exact, and that ratio is the number that matters.

What is NOT measured here: wall-clock and input tokens. Wall-clock needs a
live model — see the note printed at the end. Input is unchanged by design:
the candidates still carry their lyrics INTO the request; only the output
side changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from snoocle_server.discovery.service import candidate_from_text  # noqa: E402

#: Rough English-text tokens-per-character, used only for the estimate column.
_CHARS_PER_TOKEN = 4.0


def _fixture_song(line_count: int) -> dict:
    """A realistic reconciled document, built from the repo's own sheet fixture
    and repeated to `line_count` lines (real songs repeat too)."""
    sheet = (REPO / "tests" / "fixtures" / "sheet_over_lyrics.txt").read_text()
    candidate = candidate_from_text(sheet, "web-1", url="https://example/sheet")
    if candidate is None:  # pragma: no cover — the fixture is a valid sheet
        raise SystemExit("fixture sheet did not parse as a chord sheet")

    source_lines = candidate.lines
    lines = []
    for i in range(line_count):
        src = source_lines[i % len(source_lines)]
        lines.append(
            {
                "lineIndex": i,
                "lyrics": src.lyrics,
                "chordPlacements": [
                    {"charIndex": p.charIndex, "chord": p.chord} for p in src.chordPlacements
                ],
                "_sourceLine": i % len(source_lines),  # consumed below, never emitted
            }
        )
    return {
        "id": "the-beatles--let-it-be",
        "metadata": {"title": "Let It Be", "artist": "The Beatles", "key": "C major", "bpm": 73.5},
        "displayPreferences": {"capo": 0, "tuning": "standard"},
        "audio": {"youtubeVideoId": "QDYfEBY9NM4", "syncMap": []},
        "sections": [
            {"sectionIndex": 0, "name": "Verse 1", "kind": "verse",
             "startLineIndex": 0, "endLineIndex": max(0, line_count - 1)}
        ],
        "lines": lines,
        "provenance": [],
    }


def _load_song(song_id: str | None, song_file: str | None, line_count: int) -> tuple[dict, str]:
    if song_file:
        return json.loads(Path(song_file).read_text()), f"file {song_file}"
    if song_id:
        from snoocle_server.store import get_repository

        song = get_repository().get(song_id)
        return song.model_dump(mode="json"), f"stored song {song_id}"
    return _fixture_song(line_count), f"fixture sheet, scaled to {line_count} lines"


def as_emitted_today(song: dict) -> dict:
    """What the model has to write under the old contract: the whole document,
    lyrics and all."""
    out = json.loads(json.dumps(song))
    for line in out["lines"]:
        line.pop("_sourceLine", None)
    return out


def as_emitted_with_refs(song: dict, source_id: str = "web-1") -> dict:
    """The same document under the reference protocol: a pointer per line."""
    out = json.loads(json.dumps(song))
    for position, line in enumerate(out["lines"]):
        source_line = line.pop("_sourceLine", position)
        if not line.get("lyrics"):
            line["lyrics"] = ""  # instrumental: deliberate, and cheap
            continue
        line.pop("lyrics")
        line["lyricRef"] = {"sourceId": source_id, "line": source_line}
    return out


def _measure(document: dict) -> tuple[int, int]:
    text = json.dumps(document, separators=(",", ":"))
    return len(text), round(len(text) / _CHARS_PER_TOKEN)


def _lyric_share(song: dict) -> float:
    """What fraction of today's emitted document is lyric TEXT. This is the
    ceiling on any saving — and the number that decides whether "lyrics are
    most of the payload" is true for a given song's chord density."""
    emitted = as_emitted_today(song)
    total = len(json.dumps(emitted, separators=(",", ":")))
    lyric_chars = sum(len(line.get("lyrics") or "") for line in emitted["lines"])
    return lyric_chars / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--song-id", help="measure a song from the configured store")
    parser.add_argument("--song-file", help="measure a Song JSON file")
    parser.add_argument("--lines", type=int, default=48,
                        help="line count for the built-in fixture song (default 48)")
    args = parser.parse_args()

    song, described = _load_song(args.song_id, args.song_file, args.lines)
    lines = song["lines"]
    chords = sum(len(line.get("chordPlacements") or []) for line in lines)
    before_chars, before_tokens = _measure(as_emitted_today(song))

    print(f"Source: {described}")
    print(f"Lines:  {len(lines)}  ({chords / max(len(lines), 1):.1f} chord placements per line, "
          f"{sum(len(l.get('lyrics') or '') for l in lines) / max(len(lines), 1):.0f} lyric chars per line)")
    print(f"Lyric text is {_lyric_share(song):.0%} of the document emitted today "
          f"— that is the ceiling on any saving.")
    print()
    print(f"{'':30} {'characters':>12} {'~tokens':>10} {'vs today':>10}")
    print(f"{'model output today':30} {before_chars:>12,} {before_tokens:>10,} {'—':>10}")
    # sourceId length is repeated on EVERY line, so it is not a detail: a long
    # one (Ultimate Guitar's ids are ~23 chars) can cost more than the lyric it
    # replaces. Both ends are shown rather than the flattering one.
    for source_id in ("web-1", "ultimate-guitar-1087597"):
        after_chars, after_tokens = _measure(as_emitted_with_refs(song, source_id))
        delta = (before_chars - after_chars) / before_chars
        label = f"refs, sourceId {source_id!r}"
        print(f"{label[:30]:30} {after_chars:>12,} {after_tokens:>10,} "
              f"{delta:>+9.0%}")
    print()
    print("~tokens is an estimate (chars/4); the character counts and the ratios")
    print("are exact. Input tokens are unchanged by design — the candidates still")
    print("carry their lyrics into the request; only the output side changes.")
    print("Wall-clock needs a live model: compare the `reconcile` step duration on")
    print("a run trace (GET /v1/runs/{runId}) before and after.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
