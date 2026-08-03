import type { CallToolResult, Tool } from "@modelcontextprotocol/sdk/types.js";
import { describe, expect, it } from "vitest";
import { contract, tool } from "./test-fixtures";
import {
  TOOL_CONTRACT_META_KEY,
  capabilityContracts,
  classifyTools,
  filterTools,
  invocationView,
} from "./tooling";

describe("dynamic tool classification", () => {
  it("uses the MCP 1.x list_capabilities fallback without dropping any discovered tool", () => {
    const first = tool("first") as Tool;
    const second = tool("second") as Tool;
    const capabilities = {
      content: [{ type: "text" as const, text: JSON.stringify({
        tools: [
          { name: "first", title: "First", toolContract: contract({ title: undefined }) },
          { name: "second", title: "Second", toolContract: contract({ title: undefined, category: "storage" }) },
        ],
      }) }],
    } as CallToolResult;
    const classified = classifyTools([first, second], capabilityContracts(capabilities));
    expect(classified.map((item) => item.name)).toEqual(["first", "second"]);
    expect(classified.map((item) => item.classificationSource)).toEqual(["capabilities", "capabilities"]);
    expect(classified[1].contract?.category).toBe("storage");
    expect(classified[1].contract?.title).toBe("Second");
  });

  it("prefers namespaced tools/list metadata over compatibility metadata", () => {
    const metaContract = contract({ category: "alignment" });
    const discovered = tool("aligned", {
      _meta: { [TOOL_CONTRACT_META_KEY]: metaContract },
    }) as Tool;
    const classified = classifyTools(discovered ? [discovered] : [], new Map([
      ["aligned", contract({ category: "fallback" })],
    ]));
    expect(classified[0].contract?.category).toBe("alignment");
    expect(classified[0].classificationSource).toBe("tool-meta");
  });

  it("searches metadata and filters category and browser safety", () => {
    const tools = [
      tool("safe_echo"),
      tool("write_song", { contract: contract({ category: "storage", browserSafety: "confirmation_required" }) }),
      tool("read_path", { contract: contract({ category: "audio", browserSafety: "server_filesystem_restricted" }) }),
    ];
    expect(filterTools(tools, "safe_echo", "all", "all").map((item) => item.name)).toEqual(["safe_echo"]);
    expect(filterTools(tools, "", "storage", "all").map((item) => item.name)).toEqual(["write_song"]);
    expect(filterTools(tools, "", "all", "browser-runnable").map((item) => item.name)).toEqual([
      "safe_echo", "write_song",
    ]);
  });
});

describe("tool results", () => {
  it("shows structured and raw results with server observability fields", () => {
    const raw = {
      content: [{ type: "text" as const, text: "fallback" }],
      structuredContent: {
        ok: true,
        elapsedMs: 17,
        cacheStatus: "hit",
        model: "test-model",
        modelCostUSD: 0.0123,
        usage: { inputTokens: 10 },
      },
    } as CallToolResult;
    const view = invocationView(raw, 99);
    expect(view.structured).toEqual(raw.structuredContent);
    expect(view.raw).toBe(raw);
    expect(view.telemetry).toEqual({
      elapsedMs: 17,
      cache: "hit",
      model: "test-model",
      costUSD: 0.0123,
      usage: { inputTokens: 10 },
    });
  });

  it("recognizes structured application errors even without MCP isError", () => {
    const raw = {
      content: [{ type: "text" as const, text: JSON.stringify({ ok: false, error: { message: "invalid song" } }) }],
    } as CallToolResult;
    const view = invocationView(raw, 4.6);
    expect(view.failed).toBe(true);
    expect(view.errorMessage).toBe("invalid song");
    expect(view.telemetry.elapsedMs).toBe(5);
  });
});
