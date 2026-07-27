"""Parsing the admin's "add many" textarea into queue job specs.

This module used to hold an in-process asyncio worker as well. That worker is
gone: jobs now live in the store (``store/jobs.py``) and are executed by an
external worker that claims them, which is what lets the Cloud Run service stay
on request-based billing at ``--min-instances=0``. What's left here is the one
piece that genuinely belongs to the *submission* side — turning a human's
pasted list into structured jobs.
"""

from __future__ import annotations

from typing import Optional

# Refuse absurd submissions outright rather than accepting them and then
# spending an hour discovering the caller pasted their clipboard.
MAX_ITEMS_PER_SUBMIT = 100

_DASHES = (" — ", " – ", " - ")


def parse_batch_line(line: str) -> Optional[dict]:
    """One line of the "add many" textarea -> a job spec.

    Accepted, in this order:
      - a YouTube URL or a bare 11-character video id
      - ``Artist - Title`` (also with an en- or em-dash)
      - anything else is treated as a title with no artist, which the pipeline
        can still work with because discovery searches by text

    Returns None for a blank or comment line, so a pasted list can carry
    whitespace and notes without special handling upstream.
    """
    line = (line or "").strip()
    if not line or line.startswith("#"):
        return None

    lowered = line.lower()
    if lowered.startswith(("http://", "https://")) or "youtu" in lowered:
        return {"youtubeUrlOrId": line}
    # A bare video id: exactly 11 characters of YouTube's alphabet.
    if len(line) == 11 and all(c.isalnum() or c in "-_" for c in line):
        return {"youtubeUrlOrId": line}

    for dash in _DASHES:
        if dash in line:
            artist, title = line.split(dash, 1)
            artist, title = artist.strip(), title.strip()
            if artist and title:
                return {"title": title, "artist": artist}
    return {"title": line}


__all__ = ["parse_batch_line", "MAX_ITEMS_PER_SUBMIT"]
