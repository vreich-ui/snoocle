import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { looksUnauthorized } from "./api";
import type { ToolStudioClient } from "./mcp";
import { contract, tool } from "./test-fixtures";
import { ToolStudio } from "./ToolStudio";
import type { StudioTool } from "./tooling";
import { EMPTY_WORKBENCH, type Workbench } from "./workbench";

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

/** A tool shaped like PR #80's deterministic tools: song_id + song_version, plus a plain field. */
function benchTools(): StudioTool[] {
  return [
    tool("process_song_deterministically", {
      title: "Process Song",
      contract: contract({ title: "Process Song", category: "pipeline", browserSafety: "safe" }),
      inputSchema: {
        type: "object",
        properties: {
          song_id: { type: "string" },
          song_version: { type: "string" },
          message: { type: "string" },
        },
        required: ["song_id"],
      },
    }),
  ];
}

/** A tool shaped like acquire_audio: identity strings alongside a recording source that overrides them. */
function acquireTools(): StudioTool[] {
  return [
    tool("acquire_audio", {
      title: "Acquire Audio",
      contract: contract({ title: "Acquire Audio", category: "audio", browserSafety: "safe" }),
      inputSchema: {
        type: "object",
        properties: {
          title: { type: "string" },
          artist: { type: "string" },
          youtube_url_or_id: { type: "string" },
        },
      },
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
    render(<ToolStudio token="tab-token" bench={EMPTY_WORKBENCH} onBenchChange={vi.fn()} clientFactory={() => client} />);
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
    render(<ToolStudio token="tab-token" bench={EMPTY_WORKBENCH} onBenchChange={vi.fn()} clientFactory={() => client} />);
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
    render(<ToolStudio token="tab-token" bench={EMPTY_WORKBENCH} onBenchChange={vi.fn()} clientFactory={() => client} />);
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
    render(<ToolStudio token="tab-token" bench={EMPTY_WORKBENCH} onBenchChange={vi.fn()} clientFactory={() => client} />);
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
    render(<ToolStudio token="tab-token" bench={EMPTY_WORKBENCH} onBenchChange={vi.fn()} clientFactory={() => client} />);
    await screen.findByText(/Connected · 3 tools/);
    await userEvent.type(screen.getByLabelText(/Message/), "restore me");
    await userEvent.click(screen.getByRole("button", { name: "Invoke tool" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("MCP connection lost");
    await userEvent.click(screen.getByRole("button", { name: /dynamic_echo.*error/ }));
    expect(screen.getByLabelText(/Message/)).toHaveValue("restore me");
  });

  it("never connects with an empty token and shows the unauthenticated panel", async () => {
    const client = mockClient();
    render(<ToolStudio token="" bench={EMPTY_WORKBENCH} onBenchChange={vi.fn()} clientFactory={() => client} />);
    expect(await screen.findByText("Enter a bearer token to connect.")).toBeVisible();
    expect(screen.getByText(/Paste the server's SNOOCLE_API_TOKEN/)).toBeVisible();
    expect(screen.getByText("No token")).toBeVisible();
    expect(client.connectAndDiscover).not.toHaveBeenCalled();
  });

  it("shows a rejected-token panel, not the raw transport error, when the server answers 401", async () => {
    const client = mockClient();
    client.connectAndDiscover = vi.fn().mockRejectedValue(
      new Error('Streamable HTTP error: Error POSTing to endpoint: {"error":"invalid_token","error_description":"authorization required; see the WWW-Authenticate header"}'),
    );
    render(<ToolStudio token="bad-token" bench={EMPTY_WORKBENCH} onBenchChange={vi.fn()} clientFactory={() => client} />);
    expect(await screen.findByText("The bearer token was rejected.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Retry connection" })).toBeVisible();
    expect(screen.queryByText(/Streamable HTTP error/)).not.toBeInTheDocument();
  });

  it("seeds a form from the workbench and shows the From workbench line", async () => {
    const client = mockClient();
    client.connectAndDiscover = vi.fn().mockResolvedValue(benchTools());
    const bench: Workbench = { song: { id: "artist--song", title: "Song", artist: "Artist", version: "v2" } };
    render(<ToolStudio token="tab-token" bench={bench} onBenchChange={vi.fn()} clientFactory={() => client} />);

    await screen.findByText(/Connected · 1 tools/);
    expect(screen.getByText("From workbench: song_id, song_version")).toBeVisible();
    expect(screen.getByLabelText(/Song Id/i)).toHaveValue("artist--song");
    expect(screen.getByLabelText(/Song Version/i)).toHaveValue("v2");
    // Seeded fields stay fully editable, never disabled.
    expect(screen.getByLabelText(/Song Id/i)).toBeEnabled();
  });

  it("keeps a restored history value winning over a seeded workbench value for the same key", async () => {
    const callTool = vi.fn().mockResolvedValue({
      content: [{ type: "text", text: "ok" }],
      structuredContent: { ok: true },
    } satisfies CallToolResult);
    const client = mockClient(callTool);
    client.connectAndDiscover = vi.fn().mockResolvedValue(benchTools());
    const bench: Workbench = { song: { id: "bench-seed-song", title: "Song", artist: "Artist" } };
    render(<ToolStudio token="tab-token" bench={bench} onBenchChange={vi.fn()} clientFactory={() => client} />);
    await screen.findByText(/Connected · 1 tools/);

    const songIdField = screen.getByLabelText(/Song Id/i);
    expect(songIdField).toHaveValue("bench-seed-song");
    await userEvent.clear(songIdField);
    await userEvent.type(songIdField, "explicit-song");
    await userEvent.click(screen.getByRole("button", { name: "Invoke tool" }));
    await waitFor(() => expect(callTool).toHaveBeenCalledTimes(1));

    // Reselecting the same tool (by its catalog title) clears any restored args and re-seeds from the workbench.
    await userEvent.click(screen.getByRole("button", { name: /Process Song/ }));
    expect(await screen.findByLabelText(/Song Id/i)).toHaveValue("bench-seed-song");

    // Restoring the history entry must win over the workbench seed for song_id.
    await userEvent.click(screen.getByRole("button", { name: /process_song_deterministically.*success/ }));
    expect(await screen.findByLabelText(/Song Id/i)).toHaveValue("explicit-song");
  });
});

describe("looksUnauthorized", () => {
  it("matches a 401 status, invalid_token, and not a plain network error", () => {
    expect(looksUnauthorized("Request failed with status code 401")).toBe(true);
    expect(looksUnauthorized('{"error":"invalid_token","error_description":"..."}')).toBe(true);
    expect(looksUnauthorized("MCP connection lost")).toBe(false);
  });

  // The reported bug: the form said "From workbench: title, artist" for
  // "Back to Black — Amy Winehouse" while a pasted URL fetched a Nirvana
  // cover. The URL wins server-side, so the identity was a claim the call
  // could not keep.
  it("does not seed title or artist into a tool that acquires a recording", async () => {
    const client = mockClient();
    client.connectAndDiscover = vi.fn().mockResolvedValue(acquireTools());
    const bench: Workbench = { song: { id: "amy--back-to-black", title: "Back to Black", artist: "Amy Winehouse" } };
    render(<ToolStudio token="tab-token" bench={bench} onBenchChange={vi.fn()} clientFactory={() => client} />);

    expect(await screen.findByLabelText(/Youtube Url Or Id/i)).toHaveValue("");
    expect(screen.getByLabelText(/^Title/i)).toHaveValue("");
    expect(screen.getByLabelText(/^Artist/i)).toHaveValue("");
    expect(screen.queryByText(/From workbench/)).toBeNull();
    expect(screen.getByText(/A URL, id or reference here decides the/)).toBeVisible();
  });
});
