import type { CallToolResult, Tool } from "@modelcontextprotocol/sdk/types.js";

export const TOOL_CONTRACT_META_KEY = "snoocle/toolContract";

export type BrowserSafety = "safe" | "confirmation_required" | "server_filesystem_restricted";
export type ExpectedDuration = "instant" | "seconds" | "minutes";
export type ModelUse = "none" | "conditional" | "required";

export interface ToolContract {
  schemaVersion: number;
  title?: string;
  category: string;
  browserSafety: BrowserSafety;
  inputArtifactKinds: string[];
  outputArtifactKinds: string[];
  access: {
    mode: "read" | "write";
    readOnly: boolean;
    destructive: boolean;
    idempotent: boolean;
  };
  networkAccess: string[];
  modelUse: ModelUse;
  persistence: string[];
  cacheBehavior: "none" | "read" | "read_write";
  expectedDuration: ExpectedDuration;
  specializedRenderer: string;
}

export type ClassificationSource = "tool-meta" | "capabilities" | "missing";

export interface StudioTool extends Tool {
  contract?: ToolContract;
  classificationSource: ClassificationSource;
}

export interface ResultTelemetry {
  elapsedMs: number;
  cache: string;
  model: string;
  costUSD: number | null;
  usage: unknown;
}

export interface InvocationView {
  raw: CallToolResult;
  structured: unknown;
  failed: boolean;
  errorMessage: string;
  telemetry: ResultTelemetry;
}

type JsonRecord = Record<string, unknown>;

export function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function normalizeAccess(value: unknown): ToolContract["access"] | undefined {
  if (!isRecord(value)) return undefined;
  const mode = value.mode;
  if (mode !== "read" && mode !== "write") return undefined;
  if (
    typeof value.readOnly !== "boolean" ||
    typeof value.destructive !== "boolean" ||
    typeof value.idempotent !== "boolean"
  ) return undefined;
  return {
    mode,
    readOnly: value.readOnly,
    destructive: value.destructive,
    idempotent: value.idempotent,
  };
}

export function normalizeToolContract(value: unknown, title?: unknown): ToolContract | undefined {
  if (!isRecord(value)) return undefined;
  const access = normalizeAccess(value.access);
  const browserSafety = value.browserSafety;
  const modelUse = value.modelUse;
  const cacheBehavior = value.cacheBehavior;
  const expectedDuration = value.expectedDuration;
  if (
    typeof value.schemaVersion !== "number" ||
    typeof value.category !== "string" ||
    !["safe", "confirmation_required", "server_filesystem_restricted"].includes(String(browserSafety)) ||
    !isStringArray(value.inputArtifactKinds) ||
    !isStringArray(value.outputArtifactKinds) ||
    !access ||
    !isStringArray(value.networkAccess) ||
    !["none", "conditional", "required"].includes(String(modelUse)) ||
    !isStringArray(value.persistence) ||
    !["none", "read", "read_write"].includes(String(cacheBehavior)) ||
    !["instant", "seconds", "minutes"].includes(String(expectedDuration)) ||
    typeof value.specializedRenderer !== "string"
  ) return undefined;
  return {
    schemaVersion: value.schemaVersion,
    title: typeof title === "string" ? title : undefined,
    category: value.category,
    browserSafety: browserSafety as BrowserSafety,
    inputArtifactKinds: value.inputArtifactKinds,
    outputArtifactKinds: value.outputArtifactKinds,
    access,
    networkAccess: value.networkAccess,
    modelUse: modelUse as ModelUse,
    persistence: value.persistence,
    cacheBehavior: cacheBehavior as ToolContract["cacheBehavior"],
    expectedDuration: expectedDuration as ExpectedDuration,
    specializedRenderer: value.specializedRenderer,
  };
}

export function structuredPayload(result: CallToolResult): unknown {
  if (result.structuredContent !== undefined) return result.structuredContent;
  const text = result.content.find((item) => item.type === "text");
  if (!text || text.type !== "text") return result.content;
  try {
    return JSON.parse(text.text) as unknown;
  } catch {
    return text.text;
  }
}

export function capabilityContracts(result: CallToolResult): Map<string, ToolContract> {
  const payload = structuredPayload(result);
  if (!isRecord(payload) || !Array.isArray(payload.tools)) return new Map();
  const contracts = new Map<string, ToolContract>();
  for (const entry of payload.tools) {
    if (!isRecord(entry) || typeof entry.name !== "string") continue;
    const contract = normalizeToolContract(entry.toolContract, entry.title);
    if (contract) contracts.set(entry.name, contract);
  }
  return contracts;
}

export function classifyTools(tools: Tool[], fallback: Map<string, ToolContract>): StudioTool[] {
  return tools.map((tool) => {
    const metaValue = isRecord(tool._meta) ? tool._meta[TOOL_CONTRACT_META_KEY] : undefined;
    const fromMeta = normalizeToolContract(metaValue, tool.title);
    const fromCapabilities = fallback.get(tool.name);
    return {
      ...tool,
      contract: fromMeta ?? fromCapabilities,
      classificationSource: fromMeta ? "tool-meta" : fromCapabilities ? "capabilities" : "missing",
    };
  });
}

export function isBrowserRunnable(tool: StudioTool): boolean {
  return tool.contract?.browserSafety === "safe" || tool.contract?.browserSafety === "confirmation_required";
}

export function filterTools(
  tools: StudioTool[],
  search: string,
  category: string,
  safety: string,
): StudioTool[] {
  const query = search.trim().toLocaleLowerCase();
  return tools.filter((tool) => {
    const contract = tool.contract;
    const haystack = [
      tool.name,
      tool.title,
      tool.description,
      contract?.title,
      contract?.category,
      ...(contract?.inputArtifactKinds ?? []),
      ...(contract?.outputArtifactKinds ?? []),
    ].filter(Boolean).join(" ").toLocaleLowerCase();
    const matchesSearch = !query || haystack.includes(query);
    const matchesCategory = category === "all" || contract?.category === category;
    const matchesSafety = safety === "all" ||
      (safety === "browser-runnable" ? isBrowserRunnable(tool) : contract?.browserSafety === safety);
    return matchesSearch && matchesCategory && matchesSafety;
  });
}

function findFirstKey(root: unknown, keys: Set<string>): unknown {
  const queue: Array<{ value: unknown; depth: number }> = [{ value: root, depth: 0 }];
  const seen = new Set<object>();
  let visited = 0;
  while (queue.length && visited < 2_000) {
    const { value, depth } = queue.shift()!;
    visited += 1;
    if (!isRecord(value) || seen.has(value) || depth > 6) continue;
    seen.add(value);
    for (const [key, child] of Object.entries(value)) {
      if (keys.has(key) && child !== undefined && child !== null && child !== "") return child;
    }
    for (const child of Object.values(value)) {
      if (isRecord(child)) queue.push({ value: child, depth: depth + 1 });
      else if (Array.isArray(child)) {
        for (const item of child) if (isRecord(item)) queue.push({ value: item, depth: depth + 1 });
      }
    }
  }
  return undefined;
}

function displayValue(value: unknown): string {
  if (value === undefined || value === null || value === "") return "not reported";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return "reported";
  }
}

function payloadError(payload: unknown): string {
  if (!isRecord(payload)) return "";
  if (isRecord(payload.error) && typeof payload.error.message === "string") return payload.error.message;
  if (typeof payload.error === "string") return payload.error;
  if (typeof payload.message === "string") return payload.message;
  return "";
}

export function invocationView(result: CallToolResult, clientElapsedMs: number): InvocationView {
  const structured = structuredPayload(result);
  const serverElapsed = findFirstKey(structured, new Set(["elapsedMs"]));
  const cache = findFirstKey(structured, new Set(["cacheStatus", "cacheBehavior", "cache"]));
  const model = findFirstKey(structured, new Set(["model", "modelName", "modelUse", "modelCalls"]));
  const cost = findFirstKey(structured, new Set(["modelCostUSD", "costUSD", "costUsd"]));
  const usage = findFirstKey(structured, new Set(["usage", "tokenUsage"]));
  const applicationError = isRecord(structured) && structured.ok === false;
  const failed = result.isError === true || applicationError;
  const textError = result.content.find((item) => item.type === "text");
  const errorMessage = failed
    ? payloadError(structured) || (textError?.type === "text" ? textError.text : "Tool invocation failed")
    : "";
  return {
    raw: result,
    structured,
    failed,
    errorMessage,
    telemetry: {
      elapsedMs: typeof serverElapsed === "number" ? serverElapsed : Math.max(0, Math.round(clientElapsedMs)),
      cache: displayValue(cache),
      model: displayValue(model),
      costUSD: typeof cost === "number" ? cost : null,
      usage: usage ?? null,
    },
  };
}

export function formatJson(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
