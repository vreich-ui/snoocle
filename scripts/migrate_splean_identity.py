#!/usr/bin/env python3
"""Print the dry-run repair plan for the Splean identity collision.

This command never writes. It reads the legacy ``unknown--unknown`` and
``splean--unknown`` documents, groups their versions/runs by stored source URL
or audio content hash, and prints the planned destinations:

    splean--romans
    splean--vyhoda-net
    splean--bog-ustal-nas-lyubit

Use the same project/database environment as Cloud Run, or pass ``--project``
when running locally with Application Default Credentials.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", help="Firestore project (defaults to configured repository)")
    parser.add_argument("--database", default="(default)", help="Firestore database name")
    parser.add_argument(
        "--use-gcloud-token", action="store_true",
        help="use the active gcloud user's short-lived token when ADC is unavailable (read-only dry run)",
    )
    args = parser.parse_args()

    if args.project:
        from snoocle_server.store.firestore_store import FirestoreSongRepository
        from snoocle_server.store.runs import FirestoreRunRepository

        credentials = None
        if args.use_gcloud_token:
            from google.oauth2.credentials import Credentials

            token = subprocess.run(
                ["gcloud", "auth", "print-access-token"], check=True, capture_output=True, text=True
            ).stdout.strip()
            credentials = Credentials(token=token)
        songs = FirestoreSongRepository(
            project=args.project, database=args.database, credentials=credentials
        )
        runs = FirestoreRunRepository(
            project=args.project, database=args.database, credentials=credentials
        )
    else:
        from snoocle_server.store import get_repository
        from snoocle_server.store.runs import get_run_store

        songs, runs = get_repository(), get_run_store()

    from snoocle_server.store.splean_migration import plan_splean_identity_migration

    print(plan_splean_identity_migration(songs, runs).describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
