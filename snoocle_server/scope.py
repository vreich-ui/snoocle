"""Re-analysis SCOPE: which evidence-gathering stages a run is allowed to do.

Re-analysis has two INDEPENDENT evidence-gathering stages, and the owner asked
to be able to turn each off:

    listen   -> acquire the audio + run MIR on it
    reconcile-> discover chord/lyric sources on the web

The LLM reconciliation itself ALWAYS runs — it is what produces the Song and
what applies the user's notes. The scope only decides what evidence it is
handed:

    listen  reconcile  behaviour
    off     off        apply the user's notes to the prior song, gather nothing
    off     on         re-search the web, reuse the existing audio analysis
    on      off        re-listen to the audio, no new web sources
    on      on         full re-analysis (the historical behaviour)

Why a dedicated module rather than a field on PipelineRequest: both the
pipeline AND the reconciler need it (the reconciler has to say plainly in its
prompt that nothing was gathered, and record the scope in provenance), and
``reconcile`` must not import ``pipeline`` — that would be a cycle.

**Absence is not a scope.** A request that says nothing about scope gets the
full pipeline, byte-for-byte as before this feature existed; every caller and
test that predates it is unaffected. Only an explicitly supplied scope
constrains anything.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisScope:
    """One run's evidence budget. Both stages default to ON, so constructing
    a bare ``AnalysisScope()`` is the full pipeline."""

    listen: bool = True
    reconcile: bool = True

    @property
    def notes_only(self) -> bool:
        """Nothing new is gathered: the reconciler gets the prior song and the
        user's notes and nothing else. "Just do what I said"."""
        return not self.listen and not self.reconcile

    def describe(self) -> str:
        """Stable, greppable text for step reports and provenance notes."""
        return (
            f"listen={'on' if self.listen else 'off'}, "
            f"reconcile={'on' if self.reconcile else 'off'}"
        )

    @classmethod
    def parse(cls, value: object) -> "AnalysisScope | None":
        """Build a scope from a wire ``{"listen": bool, "reconcile": bool}``.

        ``None`` in, ``None`` out — the caller must be able to tell "no scope
        was given" (full pipeline) from "a scope was given" (constrained).
        Missing keys default to ON so a partial object can only ever widen,
        never silently disable a stage the client never mentioned.
        """
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError('scope must be an object like {"listen": bool, "reconcile": bool}')
        return cls(
            listen=bool(value.get("listen", True)),
            reconcile=bool(value.get("reconcile", True)),
        )


#: The scope every pre-scope caller effectively runs at.
FULL_SCOPE = AnalysisScope()
