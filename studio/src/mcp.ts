import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import {
  capabilityContracts,
  classifyTools,
  type ExpectedDuration,
  type StudioTool,
} from "./tooling";

export interface ToolStudioClient {
  connectAndDiscover(): Promise<StudioTool[]>;
  callTool(tool: StudioTool, args: Record<string, unknown>, signal: AbortSignal): Promise<CallToolResult>;
  close(): Promise<void>;
}

function timeoutFor(duration: ExpectedDuration | undefined): number {
  if (duration === "minutes") return 15 * 60_000;
  if (duration === "seconds") return 2 * 60_000;
  return 60_000;
}

export class SnoocleMcpClient implements ToolStudioClient {
  private readonly client = new Client(
    { name: "snoocle-studio", version: "0.1.0" },
    { capabilities: {} },
  );
  private readonly transport: StreamableHTTPClientTransport;

  constructor(token: string, endpoint = "/mcp") {
    const headers = new Headers();
    if (token) headers.set("Authorization", `Bearer ${token}`);
    this.transport = new StreamableHTTPClientTransport(
      new URL(endpoint, window.location.origin),
      { requestInit: { headers } },
    );
  }

  async connectAndDiscover(): Promise<StudioTool[]> {
    await this.client.connect(this.transport);
    const listed = await this.client.listTools();
    const hasCompleteMeta = listed.tools.every((tool) => tool._meta?.["snoocle/toolContract"] !== undefined);
    let fallback = new Map();
    if (!hasCompleteMeta && listed.tools.some((tool) => tool.name === "list_capabilities")) {
      const result = await this.client.callTool({ name: "list_capabilities", arguments: {} });
      if ("content" in result) fallback = capabilityContracts(result as CallToolResult);
    }
    return classifyTools(listed.tools, fallback);
  }

  async callTool(
    tool: StudioTool,
    args: Record<string, unknown>,
    signal: AbortSignal,
  ): Promise<CallToolResult> {
    const timeout = timeoutFor(tool.contract?.expectedDuration);
    const result = await this.client.callTool(
      { name: tool.name, arguments: args },
      undefined,
      { signal, timeout, maxTotalTimeout: timeout, resetTimeoutOnProgress: true },
    );
    if (!("content" in result)) throw new Error("The server returned an unsupported task result");
    return result as CallToolResult;
  }

  async close(): Promise<void> {
    await this.client.close();
  }
}

export function createToolStudioClient(token: string): ToolStudioClient {
  return new SnoocleMcpClient(token);
}
