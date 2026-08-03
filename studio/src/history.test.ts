import { beforeEach, describe, expect, it } from "vitest";
import { clearHistory, loadHistory, saveHistory, type InvocationHistoryEntry } from "./history";

function entry(index: number): InvocationHistoryEntry {
  return {
    id: String(index),
    invokedAt: new Date(2026, 7, 3, 12, 0, index).toISOString(),
    toolName: `tool_${index}`,
    arguments: { index },
    status: "success",
    telemetry: { elapsedMs: index, cache: "miss", model: "none", costUSD: 0, usage: null },
  };
}

describe("local invocation history", () => {
  beforeEach(() => window.localStorage.clear());

  it("persists arguments and telemetry locally, newest first, capped at 50", () => {
    for (let index = 0; index < 55; index += 1) saveHistory(entry(index));
    const loaded = loadHistory();
    expect(loaded).toHaveLength(50);
    expect(loaded[0]).toMatchObject({ toolName: "tool_54", arguments: { index: 54 } });
    expect(loaded.at(-1)?.toolName).toBe("tool_5");
  });

  it("clears history without touching session-only authentication", () => {
    saveHistory(entry(1));
    window.sessionStorage.setItem("snoocle.studio.bearer-token", "keep-me");
    clearHistory();
    expect(loadHistory()).toEqual([]);
    expect(window.sessionStorage.getItem("snoocle.studio.bearer-token")).toBe("keep-me");
  });
});
