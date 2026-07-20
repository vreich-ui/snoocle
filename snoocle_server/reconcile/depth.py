"""Analysis-depth presets.

The iOS app (and the GUI) send a single ``analysisDepth`` — fast | standard |
thorough — instead of tuning MIR accuracy, agent effort, and the tool budget
separately. This module is the ONE place that expands that preset into concrete
knobs, so the app never has to know the internal levers and the mapping can
evolve server-side without a client release.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DepthProfile:
    name: str
    accuracy: str          # MIR accuracy mode (fast | standard)
    effort: str            # anthropic-agent reasoning effort
    max_web_search: int    # server-tool budget for the agent loop
    max_fetch: int
    max_windows: int
    time_align: bool       # ask the agent to populate audio.syncMap from MIR


_PROFILES = {
    # Quick pass: windowed MIR, cheap effort, minimal web work. For a fast draft.
    "fast": DepthProfile("fast", "fast", "low", 1, 2, 1, False),
    # The default: full-track MIR, consolidated tool use.
    "standard": DepthProfile("standard", "standard", "medium", 2, 3, 2, False),
    # Deep pass: full-track MIR (never capped), high effort, a larger budget,
    # AND time alignment (fills syncMap so the app can scroll/​highlight in time).
    "thorough": DepthProfile("thorough", "thorough", "high", 3, 4, 3, True),
}

DEFAULT_DEPTH = "standard"


def resolve_depth(depth: str | None) -> DepthProfile:
    """Return the profile for a depth name, falling back to standard."""
    return _PROFILES.get((depth or DEFAULT_DEPTH).lower(), _PROFILES[DEFAULT_DEPTH])


def depth_names() -> list[str]:
    return list(_PROFILES.keys())
