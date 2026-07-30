#!/usr/bin/env python3
"""Audit and repair stored song ids from failed identity resolution.

Song ids are permanent, content-hash-versioned store keys (see
snoocle_server/identity.py) — before PR #45/#51 tightened resolution, several
documents were minted under an id derived from a bad guess: a channel name, a
raw upload description, cover phrasing read as the artist. Those ids cannot be
corrected in place. This script:

  1. lists stored songs whose id does not match what identity.py would
     resolve TODAY from their stored audio.youtubeVideoId, and
  2. can re-analyze one named song under the correct id, marking the old
     document superseded (a forward pointer, never a delete) rather than
     rewriting it.

DRY RUN BY DEFAULT for everything. `list` never writes. `reanalyze` only
previews (a metadata fetch + identity resolution, no pipeline run, no store
write) unless given --confirm.

    .venv/bin/python scripts/fix_song_identity.py list
    .venv/bin/python scripts/fix_song_identity.py list --song-id wil-per--rolling-stones-paint-it-black-live-1966
    .venv/bin/python scripts/fix_song_identity.py reanalyze wil-per--rolling-stones-paint-it-black-live-1966
    .venv/bin/python scripts/fix_song_identity.py reanalyze wil-per--rolling-stones-paint-it-black-live-1966 --confirm

`reanalyze` without --confirm only previews; the operator decides which
mismatches are worth acting on — a mismatched artist doesn't necessarily mean
the document's content is wrong (see the module docstring in
snoocle_server/store/identity_audit.py). If the song is gold-marked, this
refuses outright rather than moving the gold pointer for you.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from snoocle_server.store import get_repository  # noqa: E402
from snoocle_server.store.identity_audit import (  # noqa: E402
    find_mismatched_identities,
    reanalyze_and_supersede,
)


def _cmd_list(args: argparse.Namespace) -> int:
    store = get_repository()
    song_ids = [args.song_id] if args.song_id else None
    checks = find_mismatched_identities(store, song_ids=song_ids)
    if not checks:
        print("No mismatches found." if not args.song_id else f"{args.song_id}: matches, or nothing to check.")
        return 0
    print(f"{len(checks)} song(s) whose stored id does not match identity.py's current answer:\n")
    for c in checks:
        print(f"  {c.describe()}")
    print(
        "\nThis is a report only — nothing was written. Re-run with "
        "'reanalyze <song-id>' to preview a fix, and '--confirm' to act on it."
    )
    return 0


def _cmd_reanalyze(args: argparse.Namespace) -> int:
    store = get_repository()
    report = reanalyze_and_supersede(
        store, args.song_id, confirm=args.confirm, provider=args.provider, model=args.model,
    )
    print(report.describe())
    if report.steps:
        print("\nPipeline steps:")
        for step, outcome in report.steps.items():
            print(f"  {step}: {outcome}")
    return 1 if report.refused else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="report stored songs whose id no longer matches identity.py")
    p_list.add_argument("--song-id", help="check only this song (default: every stored song)")
    p_list.set_defaults(func=_cmd_list)

    p_re = sub.add_parser("reanalyze", help="re-analyze one song under the correct id and supersede it")
    p_re.add_argument("song_id")
    p_re.add_argument("--confirm", action="store_true", help="actually run the pipeline and write (default: dry run)")
    p_re.add_argument("--provider", help="LLM provider override for the re-analysis")
    p_re.add_argument("--model", help="model override for the re-analysis")
    p_re.set_defaults(func=_cmd_reanalyze)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
