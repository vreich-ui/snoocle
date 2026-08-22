import { describe, expect, it } from "vitest";
import { formatCost, formatDateTime, orDash, pairOrDash } from "./format";

describe("formatters", () => {
  it("renders a dash rather than throwing on a field a historical record never had", () => {
    expect(formatCost(undefined)).toBe("—");
    expect(formatCost(null)).toBe("—");
    expect(formatCost(0)).toBe("$0.0000");
    expect(formatCost(0.0123)).toBe("$0.0123");
  });

  it("keeps an unparseable timestamp visible instead of showing Invalid Date", () => {
    expect(formatDateTime(undefined)).toBe("—");
    expect(formatDateTime("")).toBe("—");
    expect(formatDateTime("not a date")).toBe("not a date");
    expect(formatDateTime("2026-08-01T00:00:00Z")).not.toBe("—");
  });

  it("dashes empty scalars and pairs", () => {
    expect(orDash(undefined)).toBe("—");
    expect(orDash("")).toBe("—");
    expect(orDash(0)).toBe("0");
    expect(pairOrDash(undefined, undefined)).toBe("—");
    expect(pairOrDash("anthropic-agent", undefined)).toBe("anthropic-agent / —");
  });
});
