"""Agent evaluation: score a produced Song against a human-approved gold Song.

The reconciler's quality is now observable per run (traces); this package makes
it *measurable* — content metrics (chords, lyrics, sections, timing) comparing a
candidate Song to a gold Song, so prompt/rule changes can be scored instead of
eyeballed.
"""

from .metrics import score_song  # noqa: F401
