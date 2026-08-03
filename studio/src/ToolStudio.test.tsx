import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ToolStudioClient } from "./mcp";
import { contract, tool } from "./test-fixtures";
import { ToolStudio } from "./ToolStudio";
import type { StudioTool } from "./tooling";

function tools(): StudioTool[] {
  return [
    tool(),
    tool("save_dynamic", {
      title: "Save Dynamic",
      contract: contract({ title: "Save Dynamic", category: "storage", browserSafety: "confirmation_required" }),
    }),
    tool("read_server_path", {
      title: "Read Server Path",
      contract: contract({ title: "Read Server Path", category: "audio", browserSafety: "server_filesystem_restricted" }),
      inputSchema: { type: "object", properties: { audio_path: { type: "string" } }, required: ["audio_path"] },
    }),
  ];
}

function mockClient(callTool?: ToolStudioClient["callTool"]): ToolStudioClient {
  return {
    connectAndDiscover: vi.fn().mockResolvedValue(tools()),
    callTool: callTool ?? vi.fn().mockResolvedValue({
      content: [{ type: "text", text: "ok" }],
      structuredContent: { ok: true, echoed: "hello", elapsedMs: 12, cacheStatus: "hit", modelCalls: 0, modelCostUSD: 0 },
    } satisfies CallToolResult),
    close: vi.fn().mockResolvedValue(undefined),
  };
}

describe("ToolStudio", () => {
  beforeEach(() => window.localStorage.clear());
  afterEach(cleanup);

  it("renders every dynamically discovered tool and searches/filters the catalog", async () => {
    const client = mockClient();
    render(<ToolStudio token="" clientFactory={() => client} />);
    expect(await screen.findByText(/Connected · 3 tools · 2 browser-runnable/)).toBeVisible();
    expect(screen.getByRole("button", { name: /dynamic_echo/ })).toBeVisible();
    expect(screen.getByRole("button", { name: /save_dynamic/ })).toBeVisible();
    expect(screen.getByRole("button", { name: /read_server_path/ })).toBeVisible();

    await userEvent.type(screen.getByLabelText("Search tools"), "server path");
    expect(screen.getByText("Showing 1 of 3")).toBeVisible();
    expect(screen.getByRole("button", { name: /read_server_path/ })).toBeVisible();
    await userEvent.clear(screen.getByLabelText("Search tools"));
    await userEvent.selectOptions(screen.getByLabelText("Browser safety"), "browser-runnable");
    expect(screen.getByText("Showing 2 of 3")).toBeVisible();
    expect(screen.queryByRole("button", { name: /read_server_path/ })).not.toBeInTheDocument();
  });

  it("uses the discovered input schema and renders telemetry plus structured/raw results", async () => {
    const client = mockClient();
    render(<ToolStudio token="tab-token" clientFactory={() => client} />);
    await screen.findByText(/Connected · 3 tools/);
    await userEvent.type(screen.getByLabelText(/Message/), "hello");
    await userEvent.click(screen.getByRole("button", { name: "Invoke tool" }));
    const resultHeading = await screen.findByRole("heading", { name: "Tool result" });
    expect(resultHeading).toBeVisible();
    const panel = resultHeading.closest("section")!;
    expect(screen.getByTestId("tool-result")).toHaveTextContent('"echoed": "hello"');
    expect(within(panel).getByText("12 ms")).toBeVisible();
    expect(within(panel).getByText("hit")).toBeVisible();
    expect(within(panel).getByText("$0.000000")).toBeVisible();
    expect(client.callTool).toHaveBeenCalledWith(expect.objectContaining({ name: "dynamic_echo" }), { message: "hello" }, expect.any(AbortSignal));

    await userEvent.click(screen.getByRole("button", { name: "Raw MCP result" }));
    expect(screen.getByTestId("tool-result")).toHaveTextContent('"content"');
    expect(screen.getByRole("button", { name: /dynamic_echo.*success/ })).toBeVisible();
  });

  it("requires confirmation and blocks server-filesystem tools", async () => {
    const client = mockClient();
    render(<ToolStudio token="" clientFactory={() => client} />);
    await screen.findByText(/Connected · 3 tools/);
    await userEvent.click(screen.getByRole("button", { name: /save_dynamic/ }));
    const confirmedInvoke = screen.getByRole("button", { name: "Confirm and invoke" });
    expect(confirmedInvoke).toBeDisabled();
    await userEvent.click(screen.getByRole("checkbox"));
    await userEvent.type(screen.getByLabelText(/Message/), "save");
    await userEvent.click(confirmedInvoke);
    await waitFor(() => expect(client.callTool).toHaveBeenCalledTimes(1));

    await userEvent.click(screen.getByRole("button", { name: /read_server_path/ }));
    expect(screen.getByRole("alert")).toHaveTextContent("server-local file path");
    expect(screen.getByRole("button", { name: "Invoke tool" })).toBeDisabled();
  });

  it("cancels an in-flight request through AbortSignal and records cancellation locally", async () => {
    const callTool = vi.fn((_: StudioTool, __: Record<string, unknown>, signal: AbortSignal) => new Promise<CallToolResult>((_, reject) => {
      signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
    }));
    const client = mockClient(callTool);
    render(<ToolStudio token="" clientFactory={() => client} />);
    await screen.findByText(/Connected · 3 tools/);
    await userEvent.type(screen.getByLabelText(/Message/), "wait");
    await userEvent.click(screen.getByRole("button", { name: "Invoke tool" }));
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Invocation cancelled");
    expect(screen.getByRole("button", { name: /dynamic_echo.*cancelled/ })).toBeVisible();
  });

  it("surfaces transport errors and can restore historical arguments", async () => {
    const client = mockClient(vi.fn().mockRejectedValueOnce(new Error("MCP connection lost")).mockResolvedValueOnce({
      content: [{ type: "text", text: "ok" }], structuredContent: { ok: true },
    } satisfies CallToolResult));
    render(<ToolStudio token="" clientFactory={() => client} />);
    await screen.findByText(/Connected · 3 tools/);
    await userEvent.type(screen.getByLabelText(/Message/), "restore me");
    await userEvent.click(screen.getByRole("button", { name: "Invoke tool" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("MCP connection lost");
    await userEvent.click(screen.getByRole("button", { name: /dynamic_echo.*error/ }));
    expect(screen.getByLabelText(/Message/)).toHaveValue("restore me");
  });
});
