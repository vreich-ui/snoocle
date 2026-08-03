import type { ResultTelemetry } from "./tooling";

const HISTORY_KEY = "snoocle.studio.tool-history.v1";
const HISTORY_LIMIT = 50;

export type InvocationStatus = "success" | "error" | "cancelled";

export interface InvocationHistoryEntry {
  id: string;
  invokedAt: string;
  toolName: string;
  arguments: Record<string, unknown>;
  status: InvocationStatus;
  telemetry: ResultTelemetry;
  errorMessage?: string;
}

function isEntry(value: unknown): value is InvocationHistoryEntry {
  if (typeof value !== "object" || value === null) return false;
  const entry = value as Partial<InvocationHistoryEntry>;
  return typeof entry.id === "string" && typeof entry.invokedAt === "string" &&
    typeof entry.toolName === "string" && typeof entry.arguments === "object" &&
    ["success", "error", "cancelled"].includes(String(entry.status)) &&
    typeof entry.telemetry === "object" && entry.telemetry !== null;
}

export function loadHistory(storage: Storage = window.localStorage): InvocationHistoryEntry[] {
  try {
    const raw = storage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? parsed.filter(isEntry).slice(0, HISTORY_LIMIT) : [];
  } catch {
    return [];
  }
}

export function saveHistory(
  entry: InvocationHistoryEntry,
  storage: Storage = window.localStorage,
): InvocationHistoryEntry[] {
  const next = [entry, ...loadHistory(storage).filter((item) => item.id !== entry.id)].slice(0, HISTORY_LIMIT);
  try {
    storage.setItem(HISTORY_KEY, JSON.stringify(next));
  } catch {
    // History is a convenience. Invocation must still work when storage is unavailable.
  }
  return next;
}

export function clearHistory(storage: Storage = window.localStorage): void {
  try {
    storage.removeItem(HISTORY_KEY);
  } catch {
    // See saveHistory: storage failure must never block Studio.
  }
}

export function newHistoryId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
