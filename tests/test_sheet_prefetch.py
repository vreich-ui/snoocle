"""Deterministic chord-sheet ranking, parsing, caching, and agent boundary."""

from __future__ import annotations

import json
import pathlib
import types

from snoocle_server.config import settings
from snoocle_server.discovery.prefetch import gather_chord_sheets, rank_search_hits
from snoocle_server.discovery.resource import (
    FetchedResource,
    _google_export_url,
    extract_resource_text,
)
from snoocle_server.discovery.search import SearchHit
from snoocle_server.discovery.service import candidate_from_text
from snoocle_server.discovery.sources.ultimate_guitar import candidate_from_ug_tab
from snoocle_server.manifest import build_evidence_manifest
from snoocle_server.reconcile import anthropic_agent as agent_mod
from snoocle_server.reconcile.agent_config import AgentConfig
from snoocle_server.reconcile.anthropic_agent import AnthropicAgentProvider, _build_tools
from snoocle_server.reconcile.trace import start_run
from snoocle_server.discovery.cache import DiscoveryCacheInfo
from snoocle_server.pipeline import _record_prefetch_trace
from snoocle_server.store.blob_cache import InMemoryBlobCache


FIXTURES = pathlib.Path(__file__).parent / "fixtures"
SHEET_TEXT = (FIXTURES / "sheet_over_lyrics.txt").read_text()


def _pdf_fixture(text: str) -> bytes:
    """Small, valid one-page PDF fixture with extractable Helvetica text."""
    lines = []
    for raw in text.splitlines():
        escaped = raw.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        lines.extend([f"({escaped}) Tj", "T*"])
    stream = ("BT\n/F1 10 Tf\n45 750 Td\n12 TL\n" + "\n".join(lines) + "\nET").encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(out)


def test_ranking_prefers_official_ug_and_uses_regular_ug_when_absent():
    generic = SearchHit(url="https://example.net/song", title="Song chords")
    ug_chords = SearchHit(
        url="https://tabs.ultimate-guitar.com/tab/a/song-chords-100", title="Song Chords"
    )
    chordify = SearchHit(url="https://chordify.net/chords/song", title="Song")
    ug_official = SearchHit(
        url="https://tabs.ultimate-guitar.com/tab/a/song-official-200", title="Song Official"
    )

    ranked = rank_search_hits(
        [generic, ug_chords, chordify, ug_official], locale="global"
    )
    assert ranked[0].hit.url == ug_official.url
    assert [item.hit.url for item in ranked[1:3]] == [ug_chords.url, chordify.url]

    without_official = rank_search_hits([generic, chordify, ug_chords], locale="global")
    assert without_official[0].hit.url == ug_chords.url
    assert all(not item.official for item in without_official)


def test_parser_supports_html_pdf_and_ultimate_guitar_fixtures():
    html_resource = FetchedResource(
        url="https://example.test/song",
        content=(
            "<html><nav>advertising and links</nav><main>"
            + "".join(f"<p>{line}</p>" for line in SHEET_TEXT.splitlines())
            + "</main></html>"
        ).encode(),
        content_type="text/html",
    )
    pdf_resource = FetchedResource(
        url="https://example.test/song.pdf",
        content=_pdf_fixture(SHEET_TEXT),
        content_type="application/pdf",
    )
    html_candidate = candidate_from_text(
        extract_resource_text(html_resource), "html-fixture"
    )
    pdf_candidate = candidate_from_text(
        extract_resource_text(pdf_resource), "pdf-fixture"
    )

    hit = {
        "id": 2296,
        "song_name": "Let It Be",
        "artist_name": "The Beatles",
        "tab_url": "https://tabs.ultimate-guitar.com/tab/a/let-it-be-chords-2296",
        "type": "Official",
        "rating": 4.9,
        "votes": 500,
    }
    tab = {
        "content": (
            "[Verse 1]\n"
            "[ch]C[/ch]When I find my[ch]G[/ch]self in times of [ch]Am[/ch]trouble\n"
            "[ch]F[/ch]Mother Mary comes to me\n"
            "[ch]C[/ch]Speaking words of [ch]G[/ch]wisdom\n"
            "[ch]Am[/ch]Let it [ch]F[/ch]be"
        ),
        "capo": 0,
        "rating": 4.9,
        "votes": 500,
    }
    ug_candidate = candidate_from_ug_tab(
        hit, tab, title="Let It Be", artist="The Beatles", source_id="ug-fixture"
    )

    for candidate in (html_candidate, pdf_candidate, ug_candidate):
        assert candidate is not None
        assert candidate.parseConfidence and candidate.parseConfidence > 0
        assert candidate.lines
        assert candidate.chord_vocabulary()


def test_google_docs_pages_use_the_clean_text_export():
    assert _google_export_url(
        "https://docs.google.com/document/d/abc123/edit?usp=sharing"
    ) == "https://docs.google.com/document/d/abc123/export?format=txt"


def test_regular_web_ug_hit_uses_dedicated_parser(monkeypatch):
    monkeypatch.setattr(settings, "source_ug_enabled", False)
    url = "https://tabs.ultimate-guitar.com/tab/a/let-it-be-official-2296"
    tab = {
        "content": (
            "[Verse 1]\n"
            "[ch]C[/ch]When I find my[ch]G[/ch]self in times of [ch]Am[/ch]trouble\n"
            "[ch]F[/ch]Mother Mary comes to me\n"
            "[ch]C[/ch]Speaking words of [ch]G[/ch]wisdom\n"
            "[ch]Am[/ch]Let it [ch]F[/ch]be"
        ),
        "capo": 0,
    }

    result = gather_chord_sheets(
        "the-beatles--let-it-be",
        "The Beatles",
        "Let It Be",
        max_sheets=1,
        search_fn=lambda query, limit: [SearchHit(url=url, title="Official chords")],
        fetch_fn=lambda _: (_ for _ in ()).throw(AssertionError("generic HTML fetch used")),
        ug_fetch_fn=lambda _: tab,
        cache_store=InMemoryBlobCache(ttl_seconds=86400),
    )

    assert result.candidates[0].sourceId == "ultimate-guitar-2296"
    assert result.candidates[0].contentType == "application/x-ultimate-guitar"


def test_second_gather_uses_parsed_url_content_cache(monkeypatch):
    monkeypatch.setattr(settings, "source_ug_enabled", False)
    store = InMemoryBlobCache(ttl_seconds=86400)
    calls = {"fetch": 0}

    def search(query: str, limit: int):
        return [SearchHit(url="https://example.test/song", title="Song chords")]

    def fetch(url: str):
        calls["fetch"] += 1
        return FetchedResource(url=url, content=SHEET_TEXT.encode(), content_type="text/plain")

    first = gather_chord_sheets(
        "artist--song", "Artist", "Song", search_fn=search, fetch_fn=fetch,
        cache_store=store, max_sheets=1,
    )
    second = gather_chord_sheets(
        "artist--song", "Artist", "Song", search_fn=search, fetch_fn=fetch,
        cache_store=store, max_sheets=1,
    )

    assert calls["fetch"] == 1
    assert first.report["outcomes"][0]["status"] == "parsed"
    assert second.report["outcomes"][0]["status"] == "cache-hit"
    assert first.candidates[0].contentHash == second.candidates[0].contentHash


def test_manifest_marks_prefetched_sheet_parsed_and_capo_normalized():
    capo_sheet = (FIXTURES / "sheet_capo.txt").read_text()
    candidate = candidate_from_text(capo_sheet, "capo-sheet")
    assert candidate is not None and candidate.declaredCapo == 2
    candidate = candidate.model_copy(update={"contentHash": "abc123", "cacheStatus": "miss"})

    sheet = build_evidence_manifest(candidates=[candidate])["sources"]["sheets"][0]
    assert sheet["parseStatus"] == "parsed"
    assert sheet["parseConfidence"] == candidate.parseConfidence
    assert sheet["declaredCapo"] == 2
    assert sheet["normalizedCapo"] == 0
    assert sheet["contentHash"] == "abc123"


def test_agent_only_sees_parsed_search_tool_and_enforces_one_call(monkeypatch):
    names = {tool["name"] for tool in _build_tools(10, 10)}
    assert names == {"search_and_fetch_sheet", "analyze_audio_window"}
    assert {"web_search", "web_fetch", "fetch_chord_sheet"}.isdisjoint(names)
    assert "search_and_fetch_sheet" not in {tool["name"] for tool in _build_tools(10, 0)}

    calls = {"count": 0}

    def fake_search(context: dict, query: str, reason: str):
        calls["count"] += 1
        return {"count": 0, "sheets": []}, {
            "query": query,
            "reason": reason,
            "rankedResults": [],
            "chosenUrls": [],
            "outcomes": [],
        }

    monkeypatch.setattr(agent_mod, "search_and_fetch_sheet", fake_search)
    provider = AnthropicAgentProvider()
    provider.context = {"song_id": "artist--song", "artist": "Artist", "title": "Song"}
    provider._sheet_search_budget = 1
    provider.trace = start_run("artist--song", "anthropic-agent", "standard")
    block = lambda ident: types.SimpleNamespace(
        id=ident,
        name="search_and_fetch_sheet",
        input={"query": "Song Artist chords", "reason": "missing chorus"},
    )

    first = provider._run_tool(block("one"))
    refused = provider._run_tool(block("two"))
    assert calls["count"] == 1
    assert "tool_budget_exceeded" in refused["content"]
    assert refused["is_error"] is True
    assert "is_error" not in first

    steps = provider.trace.trace.steps
    assert steps[-2].detail["retrieval"]["query"] == "Song Artist chords"
    assert steps[-1].detail["retrieval"]["outcomes"][0]["status"] == "refused"


def test_agent_initial_context_hides_chord_source_urls():
    candidate = candidate_from_text(
        SHEET_TEXT,
        "sheet-1",
        url="https://secret-source.test/song",
    )
    assert candidate is not None
    provider = AnthropicAgentProvider()
    provider.context = {
        "song_id": "artist--song",
        "artist": "Artist",
        "title": "Song",
        "candidates": [candidate],
        "song_schema": {},
        "evidence_manifest": {
            "sources": {
                "ids": ["sheet-1"],
                "sheets": [{
                    "sourceId": "sheet-1",
                    "url": "https://secret-source.test/song",
                    "parseConfidence": 0.9,
                }],
            }
        },
    }

    payload = json.loads(provider._build_first_user_message()["content"])
    assert payload["candidates"][0]["sourceId"] == "sheet-1"
    assert "url" not in payload["candidates"][0]
    assert "url" not in payload["evidenceManifest"]["sources"]["sheets"][0]
    assert "secret-source.test" not in json.dumps(payload)


def test_prefetch_rank_and_every_parse_outcome_are_run_steps():
    recorder = start_run("artist--song", "anthropic-agent", "standard")
    report = {
        "query": '"Song" "Artist" chords',
        "recordingVariant": "studio",
        "rankedResults": [
            {"rank": 1, "url": "https://one.test", "site": "one.test"},
            {"rank": 2, "url": "https://two.test", "site": "two.test"},
        ],
        "chosenUrls": ["https://one.test", "https://two.test"],
        "outcomes": [
            {"rank": 1, "url": "https://one.test", "status": "parsed"},
            {"rank": 2, "url": "https://two.test", "status": "failed"},
        ],
    }
    _record_prefetch_trace(
        recorder,
        DiscoveryCacheInfo(status="miss", gathered_at="2026-07-31T00:00:00Z", report=report),
    )

    assert [step.label for step in recorder.trace.steps] == [
        "prefetch:rank", "prefetch:source-1", "prefetch:source-2"
    ]
    assert recorder.trace.steps[0].detail["rankedResults"] == report["rankedResults"]
    assert recorder.trace.steps[2].detail["status"] == "failed"


def test_default_sheet_budget_is_one_configurable_and_zero_with_two_full_sheets(monkeypatch):
    captured: list[dict] = []

    class FakeClient:
        def __init__(self):
            self.messages = self

        def create(self, **kwargs):
            captured.append(kwargs)
            return types.SimpleNamespace(
                stop_reason="end_turn",
                content=[types.SimpleNamespace(type="text", text=json.dumps({}))],
                usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
                container=None,
            )

    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(AnthropicAgentProvider, "_create_client", lambda self: FakeClient())
    base_context = {
        "song_id": "artist--song",
        "artist": "Artist",
        "title": "Song",
        "song_schema": {},
        "candidates": [],
    }

    provider = AnthropicAgentProvider()
    provider.context = dict(base_context)
    provider.complete("", [{"role": "user", "text": "reconcile"}])
    assert provider._sheet_search_budget == 1
    assert "search_and_fetch_sheet" in {tool["name"] for tool in captured[-1]["tools"]}

    configured = AnthropicAgentProvider()
    configured.context = {**base_context, "agent_config": AgentConfig(max_fetch=3)}
    configured.complete("", [{"role": "user", "text": "reconcile"}])
    assert configured._sheet_search_budget == 3

    provider = AnthropicAgentProvider()
    provider.context = {
        **base_context,
        "evidence_manifest": {"sources": {"fullSongSheets": 2}},
    }
    provider.complete("", [{"role": "user", "text": "reconcile"}])
    assert {tool["name"] for tool in captured[-1]["tools"]} == {"analyze_audio_window"}
    assert provider._sheet_search_budget == 0
