"""Production guardrails for mock/test-only Song documents."""

from __future__ import annotations

from .schema import Song


class TestOutputRejectedError(ValueError):
    __test__ = False
    error_code = "test_output_not_allowed"


def require_test_output_opt_in(song: Song, allow_test_output: bool) -> None:
    if song.testOnly and not allow_test_output:
        raise TestOutputRejectedError(
            "refusing to store testOnly output; pass allow_test_output=true "
            "only in an explicit test workflow"
        )


def mock_output_reasons(song: Song) -> list[str]:
    reasons: list[str] = []
    if song.testOnly:
        reasons.append("testOnly")
    if any(entry.actor.startswith("reconcile:mock/") for entry in song.provenance):
        reasons.append("mock-provenance")
    if any("mock reconciliation" in line.lyrics.casefold() for line in song.lines):
        reasons.append("placeholder-lyrics")
    return reasons
