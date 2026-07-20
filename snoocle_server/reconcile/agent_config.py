"""Runtime-editable configuration for the in-process reconciliation agent.

Lets an operator *program* the agent — its instructions, tool budget, effort,
model — without a redeploy, from the GUI Workbench or over MCP. Stored via the
same `snoocle_config` pattern as YouTube cookies (see ``store/agent_config.py``).

What is editable: extra instructions (appended), the theory-rules and
retrieval-recipe sections (swapped), the tool roster + budgets, max turns,
effort, and model. What is NOT, by design: the **output contract** — the strict
Song schema, sounding-pitch chords, and capo=0. That contract is always
appended to the system prompt and enforced by ``engine.py``'s schema validation
+ repair loop, which this config never touches. A bad instruction can degrade
quality but cannot make the server emit an invalid Song.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field, field_validator

# The four tools a config may switch off (server-side web tools + local tools).
KNOWN_TOOLS = frozenset({"web_search", "web_fetch", "fetch_chord_sheet", "analyze_audio_window"})
VALID_EFFORTS = frozenset({"low", "medium", "high"})


class AgentConfig(BaseModel):
    """Operator overrides for the anthropic-agent loop. All fields optional;
    an empty config reproduces the built-in defaults exactly."""

    model_config = {"extra": "forbid"}

    # Prompt shaping (the output contract is never here — see module docstring).
    instructions_extra: str = ""      # appended verbatim after the base prompt
    theory_rules: str = ""            # replaces the built-in music-theory section
    retrieval_recipe: str = ""        # replaces the built-in retrieval recipe
    instructions_override: str = ""   # DANGEROUS: replaces the whole base prompt

    # Loop / tooling knobs (None -> fall through to depth profile / settings).
    max_turns: int | None = Field(default=None, ge=1, le=30)
    effort: str | None = None
    max_web_search: int | None = Field(default=None, ge=0, le=10)
    max_fetch: int | None = Field(default=None, ge=0, le=10)
    max_windows: int | None = Field(default=None, ge=0, le=10)
    disabled_tools: list[str] = Field(default_factory=list)
    model: str | None = None

    # Provenance (server-set on save; ignored/overwritten on input).
    updated_at: str = ""
    source: str = ""

    @field_validator("effort")
    @classmethod
    def _effort_valid(cls, v: str | None) -> str | None:
        if v not in (None, "") and v not in VALID_EFFORTS:
            raise ValueError(f"effort must be one of {sorted(VALID_EFFORTS)}")
        return v or None

    @field_validator("disabled_tools")
    @classmethod
    def _tools_known(cls, v: list[str]) -> list[str]:
        bad = [t for t in v if t not in KNOWN_TOOLS]
        if bad:
            raise ValueError(f"unknown tool(s) {bad}; valid: {sorted(KNOWN_TOOLS)}")
        return v

    def is_default(self) -> bool:
        """True when no override is set (the built-in agent runs unchanged)."""
        return not any([
            self.instructions_extra, self.theory_rules, self.retrieval_recipe,
            self.instructions_override, self.max_turns, self.effort,
            self.max_web_search is not None, self.max_fetch is not None,
            self.max_windows is not None, self.disabled_tools, self.model,
        ])


def config_version(cfg: AgentConfig) -> str:
    """Stable 12-hex fingerprint of the behavior-affecting fields — stamped on
    each run so scorecard comparisons are attributable to a config."""
    payload = cfg.model_dump(exclude={"updated_at", "source"})
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]
