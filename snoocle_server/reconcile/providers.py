"""Pluggable LLM providers: Anthropic (Claude), OpenAI (GPT), Google (Gemini), Mock.

Provider is a RUNTIME choice (per-request parameter or SNOOCLE_LLM_PROVIDER)
so output quality can be compared across all three on identical input.

Audio capability map: raw-audio input is NOT assumed to be equivalent across
providers. The baseline request is text/JSON-only everywhere; an audio
snippet is attached only when (a) the caller opted in, (b) the provider is
known to support it. As of build time: OpenAI (gpt-4o-audio family) and
Gemini (inline_data) support audio input; Claude's audio support is
unclear/inconsistent, so `anthropic` reports supports_audio=False and never
receives audio.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ..config import settings

log = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    pass


@dataclass
class AudioAttachment:
    path: str  # wav or mp3 on disk

    def b64(self) -> str:
        return base64.standard_b64encode(Path(self.path).read_bytes()).decode()

    @property
    def media_format(self) -> str:
        return Path(self.path).suffix.lstrip(".").lower()


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: dict = field(default_factory=dict)


class LLMProvider:
    name: str = "base"
    default_model: str = ""
    supports_audio: bool = False

    def complete(
        self,
        system: str,
        turns: list[dict],  # [{"role": "user"|"assistant", "text": str}], oldest first
        model: str | None = None,
        max_tokens: int | None = None,
        audio: AudioAttachment | None = None,
    ) -> LLMResponse:
        raise NotImplementedError

    def _model(self, model: str | None) -> str:
        return model or settings.llm_model or self.default_model


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    default_model = "claude-opus-4-8"
    supports_audio = False  # unclear/inconsistent support — baseline is structured-only

    def complete(self, system, turns, model=None, max_tokens=None, audio=None):
        import anthropic

        if audio is not None:
            log.info("anthropic provider ignores audio attachment (unsupported)")
        try:
            client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key or None,
                base_url=settings.anthropic_base_url,
            )
        except (anthropic.AnthropicError, TypeError) as e:  # TypeError: SDK's no-credentials error
            raise ProviderError(f"anthropic: {e}") from e
        model = self._model(model)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens or settings.llm_max_tokens,
                system=system,
                # no temperature: sampling params are rejected on Opus 4.7+
                messages=[
                    {"role": t["role"], "content": t["text"]} for t in turns
                ],
            )
        except anthropic.APIStatusError as e:
            raise ProviderError(f"anthropic: {e.status_code} {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise ProviderError(f"anthropic: connection failed: {e}") from e
        except (anthropic.AnthropicError, TypeError) as e:
            # TypeError is the SDK's "could not resolve authentication" error
            raise ProviderError(f"anthropic: {e}") from e
        if response.stop_reason == "refusal":
            raise ProviderError("anthropic: request refused by safety classifiers")
        text = "".join(b.text for b in response.content if b.type == "text")
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return LLMResponse(text=text, provider=self.name, model=response.model, usage=usage)


class OpenAIProvider(LLMProvider):
    name = "openai"
    default_model = "gpt-4o"
    audio_model = "gpt-4o-audio-preview"
    supports_audio = True

    def complete(self, system, turns, model=None, max_tokens=None, audio=None):
        model = self._model(model)
        messages: list[dict] = [{"role": "system", "content": system}]
        for i, t in enumerate(turns):
            content: str | list = t["text"]
            is_last_user = t["role"] == "user" and i == len(turns) - 1
            if audio is not None and is_last_user:
                model = self.audio_model if model == self.default_model else model
                content = [
                    {"type": "text", "text": t["text"]},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio.b64(), "format": audio.media_format},
                    },
                ]
            messages.append({"role": t["role"], "content": content})
        try:
            r = httpx.post(
                f"{settings.openai_base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={
                    "model": model,
                    "max_completion_tokens": max_tokens or settings.llm_max_tokens,
                    "temperature": settings.llm_temperature,
                    "messages": messages,
                },
                timeout=300.0,
            )
        except httpx.HTTPError as e:
            raise ProviderError(f"openai: connection failed: {e}") from e
        if r.status_code != 200:
            raise ProviderError(f"openai: {r.status_code} {r.text[:300]}")
        data = r.json()
        return LLMResponse(
            text=data["choices"][0]["message"]["content"] or "",
            provider=self.name,
            model=data.get("model", model),
            usage=data.get("usage", {}),
        )


class GeminiProvider(LLMProvider):
    name = "gemini"
    default_model = "gemini-2.5-pro"
    supports_audio = True

    def complete(self, system, turns, model=None, max_tokens=None, audio=None):
        model = self._model(model)
        contents = []
        for i, t in enumerate(turns):
            role = "user" if t["role"] == "user" else "model"
            parts: list[dict] = [{"text": t["text"]}]
            if audio is not None and role == "user" and i == len(turns) - 1:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": f"audio/{audio.media_format}",
                            "data": audio.b64(),
                        }
                    }
                )
            contents.append({"role": role, "parts": parts})
        try:
            r = httpx.post(
                f"{settings.gemini_base_url}/v1beta/models/{model}:generateContent",
                params={"key": settings.gemini_api_key},
                json={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": max_tokens or settings.llm_max_tokens,
                        "temperature": settings.llm_temperature,
                        "responseMimeType": "application/json",
                    },
                },
                timeout=300.0,
            )
        except httpx.HTTPError as e:
            raise ProviderError(f"gemini: connection failed: {e}") from e
        if r.status_code != 200:
            raise ProviderError(f"gemini: {r.status_code} {r.text[:300]}")
        data = r.json()
        try:
            text = "".join(
                p.get("text", "") for p in data["candidates"][0]["content"]["parts"]
            )
        except (KeyError, IndexError) as e:
            raise ProviderError(f"gemini: unexpected response shape: {str(data)[:300]}") from e
        return LLMResponse(
            text=text,
            provider=self.name,
            model=model,
            usage=data.get("usageMetadata", {}),
        )


class MockProvider(LLMProvider):
    """Deterministic offline reconciler.

    Exists so the full pipeline is testable with zero network/keys, and as
    executable documentation of what a reconciliation must produce. It merges
    the highest-confidence candidate's lines with MIR-derived metadata,
    sections and syncMap — a real but LLM-free reconciliation.
    """

    name = "mock"
    default_model = "mock-reconciler-v1"
    supports_audio = False
    wants_context = True

    # engine.py injects the structured inputs here before calling complete()
    context: dict | None = None

    def complete(self, system, turns, model=None, max_tokens=None, audio=None):
        if not self.context:
            raise ProviderError("mock provider requires engine-injected context")
        from .mock_reconciler import reconcile_deterministically

        ctx = self.context
        song = reconcile_deterministically(
            title=ctx["title"],
            artist=ctx["artist"],
            song_id=ctx["song_id"],
            youtube_video_id=ctx["youtube_video_id"],
            candidates=ctx["candidates"],
            mir=ctx["mir"],
        )
        return LLMResponse(
            text=song.model_dump_json(), provider=self.name, model=self.default_model
        )


class AgentMcpProvider(LLMProvider):
    """Delegates reconciliation to an EXTERNAL AGENT over MCP.

    In this mode Snoocle holds no LLM keys and runs no AI itself: it is the
    MCP *client*. The configured agent workspace (e.g. a Claude Agent SDK
    environment with specialty agents) exposes an MCP server; Snoocle calls
    one tool there (SNOOCLE_AGENT_MCP_TOOL, default "reconcile_song") with a
    structured JSON request:

        {"request": {
            "songId": ..., "title": ..., "artist": ...,
            "mediaUrl": <YouTube watch URL or other media URL>,
            "youtubeVideoId": ...,
            "chords": [{"start": s, "end": s, "chord": "Am7"}, ...],  # MIR-timestamped
            "mir": {bpm, key, beats, sections, ...},
            "candidates": [...web text sources...],
            "songSchema": {...}
        }}

    and on repair rounds adds {"previousOutput": ..., "validationErrors": ...}.
    The tool's text result must be (or contain) the reconciled Song JSON; the
    engine's schema validation and repair loop apply to it exactly as they do
    to a direct LLM response.
    """

    name = "agent"
    default_model = "agent-mcp"
    supports_audio = False  # media is referenced by URL, not attached
    wants_context = True

    context: dict | None = None

    def complete(self, system, turns, model=None, max_tokens=None, audio=None):
        import asyncio

        if not settings.agent_mcp_url:
            raise ProviderError(
                "agent: SNOOCLE_AGENT_MCP_URL is not configured — point it at the "
                "agent workspace's MCP endpoint (streamable HTTP)"
            )
        if not self.context:
            raise ProviderError("agent provider requires engine-injected context")

        ctx = self.context
        mir = ctx.get("mir")
        request: dict = {
            "songId": ctx["song_id"],
            "title": ctx["title"],
            "artist": ctx["artist"],
            "mediaUrl": ctx.get("media_url"),
            "youtubeVideoId": ctx.get("youtube_video_id"),
            # the timestamped chord changes, first-class per the integration contract
            "chords": [c.model_dump() for c in mir.chords] if mir is not None else [],
            "mir": mir.to_prompt_payload() if mir is not None else None,
            "candidates": [c.model_dump(exclude_none=True) for c in ctx.get("candidates") or []],
            "songSchema": ctx["song_schema"],
        }
        args: dict = {"request": request}
        # Repair round: turns are [user, assistant, repair-user, ...] — hand the
        # agent its previous output and the validation errors verbatim.
        if len(turns) >= 3:
            args["previousOutput"] = turns[-2]["text"]
            args["validationErrors"] = turns[-1]["text"]

        try:
            text = asyncio.run(self._call_tool(args))
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001 — SDK raises ExceptionGroups/transport errors
            raise ProviderError(f"agent: MCP call failed: {e}") from e
        return LLMResponse(
            text=text,
            provider=self.name,
            model=f"mcp:{settings.agent_mcp_tool}",
        )

    async def _call_tool(self, args: dict) -> str:
        from datetime import timedelta

        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        headers = {}
        if settings.agent_mcp_auth_token:
            headers["Authorization"] = f"Bearer {settings.agent_mcp_auth_token}"
        timeout = timedelta(seconds=settings.agent_mcp_timeout_seconds)
        async with streamablehttp_client(
            settings.agent_mcp_url, headers=headers or None, timeout=timeout, sse_read_timeout=timeout
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    settings.agent_mcp_tool, args, read_timeout_seconds=timeout
                )
        texts = [b.text for b in result.content if getattr(b, "type", "") == "text"]
        if result.isError:
            raise ProviderError(
                f"agent: tool {settings.agent_mcp_tool!r} returned an error: "
                + (" ".join(texts)[:500] or "(no message)")
            )
        if not texts:
            raise ProviderError(
                f"agent: tool {settings.agent_mcp_tool!r} returned no text content"
            )
        return "\n".join(texts)


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "agent": AgentMcpProvider,
    "mock": MockProvider,
}


def get_provider(name: str | None = None) -> LLMProvider:
    name = (name or settings.llm_provider).lower()
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ProviderError(f"unknown LLM provider {name!r} (have: {sorted(_PROVIDERS)})")
    return cls()


def provider_capabilities() -> dict[str, dict]:
    out = {}
    for name, cls in _PROVIDERS.items():
        out[name] = {
            "defaultModel": cls.default_model,
            "supportsAudioInput": cls.supports_audio,
            "configured": bool(settings.provider_key(name)) or name == "mock",
        }
    return out
