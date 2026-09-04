import { useEffect, useMemo, useRef, useState } from "react";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { looksUnauthorized } from "./api";
import { SchemaForm, type JsonSchema } from "./SchemaForm";
import { clearHistory, loadHistory, newHistoryId, saveHistory, type InvocationHistoryEntry } from "./history";
import { createToolStudioClient, type ToolStudioClient } from "./mcp";
import {
  filterTools,
  formatJson,
  invocationView,
  isBrowserRunnable,
  isRecord,
  type InvocationView,
  type ResultTelemetry,
  type StudioTool,
} from "./tooling";
import { benchMismatches, derivesIdentityFromRecording, seedFromWorkbench, type Workbench } from "./workbench";

type ClientFactory = (token: string) => ToolStudioClient;

interface ToolStudioProps {
  token: string;
  bench: Workbench;
  onBenchChange(next: Workbench): void;
  clientFactory?: ClientFactory;
}

/**
 * MIR-shaped output is detected from the tool's own contract first — any
 * outputArtifactKind mentioning "mir" — and only falls back to guessing from
 * the payload's shape (a chords or beats key) when the contract doesn't say.
 * This stays intentionally narrow to MIR; other artifact kinds are not worth
 * generalising to in this pass.
 */
function looksLikeMir(tool: StudioTool | undefined, structured: unknown): boolean {
  const declaresMir = tool?.contract?.outputArtifactKinds.some((kind) => kind.toLowerCase().includes("mir")) ?? false;
  if (declaresMir) return true;
  return isRecord(structured) && ("chords" in structured || "beats" in structured);
}

type ConnectionState = "unauthenticated" | "connecting" | "connected" | "error" | "rejected";

const emptyTelemetry: ResultTelemetry = {
  elapsedMs: 0,
  cache: "not reported",
  model: "not reported",
  costUSD: null,
  usage: null,
};

function displayTitle(tool: StudioTool): string {
  return tool.contract?.title || tool.title || tool.name.replaceAll("_", " ");
}

function safetyLabel(tool: StudioTool): string {
  return tool.contract?.browserSafety.replaceAll("_", " ") ?? "unclassified";
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError" ||
    error instanceof Error && error.name === "AbortError";
}

export function ToolStudio({ token, bench, onBenchChange, clientFactory = createToolStudioClient }: ToolStudioProps) {
  const clientRef = useRef<ToolStudioClient | undefined>(undefined);
  const abortRef = useRef<AbortController | undefined>(undefined);
  const [connection, setConnection] = useState<ConnectionState>(() => (token ? "connecting" : "unauthenticated"));
  const [connectionError, setConnectionError] = useState("");
  const [tools, setTools] = useState<StudioTool[]>([]);
  const [selectedName, setSelectedName] = useState("");
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("all");
  const [safety, setSafety] = useState("all");
  const [retry, setRetry] = useState(0);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<InvocationView>();
  const [invocationError, setInvocationError] = useState("");
  const [resultTab, setResultTab] = useState<"structured" | "raw">("structured");
  const [history, setHistory] = useState<InvocationHistoryEntry[]>(loadHistory);
  const [restoredArgs, setRestoredArgs] = useState<Record<string, unknown>>();
  const [formVersion, setFormVersion] = useState(0);

  useEffect(() => {
    let active = true;
    setConnectionError("");
    setTools([]);
    setSelectedName("");
    if (!token) {
      setConnection("unauthenticated");
      return () => {
        active = false;
      };
    }
    const client = clientFactory(token);
    clientRef.current = client;
    setConnection("connecting");
    client.connectAndDiscover().then((discovered) => {
      if (!active) return;
      setTools(discovered);
      setSelectedName(discovered[0]?.name ?? "");
      setConnection("connected");
    }).catch((error: unknown) => {
      if (!active) return;
      const message = error instanceof Error ? error.message : String(error);
      setConnection(looksUnauthorized(message) ? "rejected" : "error");
      setConnectionError(message);
    });
    return () => {
      active = false;
      abortRef.current?.abort();
      void client.close().catch(() => undefined);
      if (clientRef.current === client) clientRef.current = undefined;
    };
  }, [clientFactory, retry, token]);

  // Switching the workbench selection (a different song, fresh audio, newly
  // captured MIR) should re-seed the open form, so remount it the same way
  // selecting a tool or restoring history already does.
  useEffect(() => {
    setFormVersion((value) => value + 1);
  }, [bench]);

  const categories = useMemo(
    () => [...new Set(tools.map((tool) => tool.contract?.category).filter((item): item is string => Boolean(item)))].sort(),
    [tools],
  );
  const visibleTools = useMemo(() => filterTools(tools, search, category, safety), [tools, search, category, safety]);
  const selected = tools.find((tool) => tool.name === selectedName);
  const runnableCount = tools.filter(isBrowserRunnable).length;
  const seed = useMemo(
    () => (selected ? seedFromWorkbench(selected.inputSchema, bench) : {}),
    [selected, bench],
  );
  const takesRecording = Boolean(selected && derivesIdentityFromRecording(selected.inputSchema));
  const mismatches = useMemo(() => benchMismatches(bench), [bench]);

  const recordHistory = (
    tool: StudioTool,
    args: Record<string, unknown>,
    status: InvocationHistoryEntry["status"],
    telemetry: ResultTelemetry,
    errorMessage = "",
  ) => {
    setHistory(saveHistory({
      id: newHistoryId(),
      invokedAt: new Date().toISOString(),
      toolName: tool.name,
      arguments: args,
      status,
      telemetry,
      errorMessage: errorMessage || undefined,
    }));
  };

  const invoke = async (args: Record<string, unknown>) => {
    const client = clientRef.current;
    if (!selected || !client || !isBrowserRunnable(selected)) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    setResult(undefined);
    setInvocationError("");
    const started = performance.now();
    try {
      const raw = await client.callTool(selected, args, controller.signal) as CallToolResult;
      const view = invocationView(raw, performance.now() - started);
      setResult(view);
      setResultTab("structured");
      if (view.failed) setInvocationError(view.errorMessage);
      recordHistory(selected, args, view.failed ? "error" : "success", view.telemetry, view.errorMessage);
    } catch (error) {
      const elapsedMs = Math.max(0, Math.round(performance.now() - started));
      const cancelled = isAbort(error);
      const message = cancelled ? "Invocation cancelled." : error instanceof Error ? error.message : String(error);
      setInvocationError(message);
      recordHistory(selected, args, cancelled ? "cancelled" : "error", { ...emptyTelemetry, elapsedMs }, message);
    } finally {
      if (abortRef.current === controller) abortRef.current = undefined;
      setBusy(false);
    }
  };

  const restore = (entry: InvocationHistoryEntry) => {
    if (!tools.some((tool) => tool.name === entry.toolName)) return;
    setSelectedName(entry.toolName);
    setRestoredArgs(entry.arguments);
    setFormVersion((value) => value + 1);
  };

  const blockedReason = !selected?.contract
    ? "This tool has no validated Snoocle classification and cannot be invoked from the browser."
    : selected.contract.browserSafety === "server_filesystem_restricted"
      ? "This tool accepts a server-local file path. Browser invocation is disabled; use an upload/base64-capable tool or a trusted server client."
      : undefined;

  return (
    <div className="tool-studio">
      <section className="tool-toolbar" aria-labelledby="tool-studio-heading">
        <div>
          <p className="eyebrow">Live MCP registry</p>
          <h2 id="tool-studio-heading">Tool Studio</h2>
          <p className="muted">Official MCP client → same-origin <code>/mcp</code>. No proxy and no hand-written tool inventory.</p>
        </div>
        <div className={`connection ${connection}`} role="status">
          {connection === "unauthenticated"
            ? "No token"
            : connection === "rejected"
              ? "Token rejected"
              : connection === "connecting"
                ? "Connecting…"
                : connection === "connected"
                  ? `Connected · ${tools.length} tools · ${runnableCount} browser-runnable`
                  : "Connection failed"}
        </div>
      </section>

      {connection === "unauthenticated" && (
        <section className="connection-error notice" role="status">
          <strong>Enter a bearer token to connect.</strong>
          <p>Paste the server's SNOOCLE_API_TOKEN into the sidebar. It stays in this browser tab only and is never stored or sent anywhere but this server.</p>
        </section>
      )}

      {connection === "rejected" && (
        <section className="connection-error" role="alert">
          <strong>The bearer token was rejected.</strong>
          <p>/mcp answered 401. Check that the value matches SNOOCLE_API_TOKEN on the server, then retry.</p>
          <button type="button" onClick={() => setRetry((value) => value + 1)}>Retry connection</button>
        </section>
      )}

      {connection === "error" && (
        <section className="connection-error" role="alert">
          <strong>Could not discover MCP tools.</strong>
          <p>{connectionError}</p>
          <button type="button" onClick={() => setRetry((value) => value + 1)}>Retry connection</button>
        </section>
      )}

      <div className="tool-layout" aria-busy={connection === "connecting"}>
        <aside className="catalog" aria-label="Tool catalog">
          <label>
            <span>Search tools</span>
            <input type="search" value={search} onChange={(event) => setSearch(event.target.value)} />
          </label>
          <div className="catalog-filters">
            <label>
              <span>Category</span>
              <select value={category} onChange={(event) => setCategory(event.target.value)}>
                <option value="all">All categories</option>
                {categories.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </label>
            <label>
              <span>Browser safety</span>
              <select value={safety} onChange={(event) => setSafety(event.target.value)}>
                <option value="all">All tools</option>
                <option value="browser-runnable">Browser-runnable</option>
                <option value="safe">Safe</option>
                <option value="confirmation_required">Confirmation required</option>
                <option value="server_filesystem_restricted">Server filesystem restricted</option>
              </select>
            </label>
          </div>
          <p className="catalog-count">Showing {visibleTools.length} of {tools.length}</p>
          <div className="tool-list">
            {visibleTools.map((tool) => (
              <button
                className={selectedName === tool.name ? "tool-card selected" : "tool-card"}
                key={tool.name}
                type="button"
                aria-pressed={selectedName === tool.name}
                onClick={() => {
                  setSelectedName(tool.name);
                  setRestoredArgs(undefined);
                  setFormVersion((value) => value + 1);
                  setResult(undefined);
                  setInvocationError("");
                }}
              >
                <strong>{displayTitle(tool)}</strong>
                <code>{tool.name}</code>
                <span>{tool.contract?.category ?? "unclassified"} · {safetyLabel(tool)}</span>
              </button>
            ))}
            {connection === "connected" && !visibleTools.length && <p className="muted">No tools match these filters.</p>}
          </div>
        </aside>

        <section className="tool-detail" aria-live="polite">
          {selected ? (
            <>
              <div className="detail-heading">
                <div>
                  <p className="eyebrow">{selected.contract?.category ?? "Unclassified"}</p>
                  <h3>{displayTitle(selected)}</h3>
                  <code>{selected.name}</code>
                </div>
                <span className={`safety-badge ${selected.contract?.browserSafety ?? "missing"}`}>{safetyLabel(selected)}</span>
              </div>
              {selected.description && <p>{selected.description}</p>}
              {selected.contract && (
                <dl className="classification-grid">
                  <div><dt>Access</dt><dd>{selected.contract.access.mode}{selected.contract.access.destructive ? " · destructive" : ""}</dd></div>
                  <div><dt>Duration</dt><dd>{selected.contract.expectedDuration}</dd></div>
                  <div><dt>Model use</dt><dd>{selected.contract.modelUse}</dd></div>
                  <div><dt>Cache</dt><dd>{selected.contract.cacheBehavior.replace("_", "/")}</dd></div>
                  <div><dt>Inputs</dt><dd>{selected.contract.inputArtifactKinds.join(", ")}</dd></div>
                  <div><dt>Outputs</dt><dd>{selected.contract.outputArtifactKinds.join(", ")}</dd></div>
                  <div><dt>Network</dt><dd>{selected.contract.networkAccess.join(", ") || "none"}</dd></div>
                  <div><dt>Persistence</dt><dd>{selected.contract.persistence.join(", ") || "none"}</dd></div>
                </dl>
              )}
              {selected.contract?.browserSafety === "confirmation_required" && (
                <p className="warning" role="note">Review the arguments carefully. This tool requires explicit confirmation before each run.</p>
              )}
              {selected.contract?.modelUse !== "none" && (
                <p className="warning" role="note">This tool may use a model and incur cost according to server policy.</p>
              )}
              {Object.keys(seed).length > 0 && (
                <p className="muted">From workbench: {Object.keys(seed).join(", ")}</p>
              )}
              {takesRecording && bench.song && (
                <p className="warning" role="note">
                  This tool works on whichever recording you give it. A URL, id or reference here decides the
                  song — the workbench selection ({bench.song.title} — {bench.song.artist}) does not, and its
                  title and artist are deliberately not filled in.
                </p>
              )}
              {mismatches.length > 0 && (
                <div className="warning" role="alert">
                  {mismatches.map((problem) => <p key={problem}>{problem}</p>)}
                </div>
              )}
              <SchemaForm
                key={`${selected.name}-${formVersion}`}
                schema={selected.inputSchema as JsonSchema}
                initialValue={{ ...seed, ...restoredArgs }}
                busy={busy}
                blockedReason={blockedReason}
                requiresConfirmation={selected.contract?.browserSafety === "confirmation_required"}
                onSubmit={invoke}
                onCancel={() => abortRef.current?.abort()}
              />
              {invocationError && <p className="error invocation-error" role="alert">{invocationError}</p>}
              {result && (
                <section className={result.failed ? "result-panel failed" : "result-panel"} aria-labelledby="result-heading">
                  <h4 id="result-heading">{result.failed ? "Tool error" : "Tool result"}</h4>
                  <dl className="telemetry-grid">
                    <div><dt>Elapsed</dt><dd>{result.telemetry.elapsedMs} ms</dd></div>
                    <div><dt>Cache</dt><dd>{result.telemetry.cache}</dd></div>
                    <div><dt>Model</dt><dd>{result.telemetry.model}</dd></div>
                    <div><dt>Cost</dt><dd>{result.telemetry.costUSD === null ? "not reported" : `$${result.telemetry.costUSD.toFixed(6)}`}</dd></div>
                  </dl>
                  {result.telemetry.usage !== null && <details><summary>Usage</summary><pre>{formatJson(result.telemetry.usage)}</pre></details>}
                  {!result.failed && looksLikeMir(selected, result.structured) && (
                    <div className="form-actions">
                      <button
                        type="button"
                        onClick={() => onBenchChange({
                          ...bench,
                          mir: {
                            json: JSON.stringify(result.structured),
                            songId: bench.song?.id,
                            audioRef: bench.audio?.audioRef,
                          },
                        })}
                      >Use as MIR</button>
                    </div>
                  )}
                  <div className="mode-switch" role="group" aria-label="Result view">
                    <button className={resultTab === "structured" ? "selected" : ""} type="button" onClick={() => setResultTab("structured")}>Structured</button>
                    <button className={resultTab === "raw" ? "selected" : ""} type="button" onClick={() => setResultTab("raw")}>Raw MCP result</button>
                  </div>
                  <pre data-testid="tool-result">{formatJson(resultTab === "structured" ? result.structured : result.raw)}</pre>
                </section>
              )}
            </>
          ) : <p className="muted">{connection === "connecting" ? "Discovering tools…" : connection === "unauthenticated" ? "Connect with a bearer token to load the live tool catalog." : "Select a tool from the live catalog."}</p>}
        </section>

        <aside className="history" aria-label="Local invocation history">
          <div className="history-heading">
            <div><p className="eyebrow">Browser local</p><h3>History</h3></div>
            <button
              type="button"
              disabled={!history.length}
              onClick={() => {
                clearHistory();
                setHistory([]);
              }}
            >Clear</button>
          </div>
          <p className="muted">Arguments and status stay in this browser; bearer tokens and full results are never stored.</p>
          <div className="history-list">
            {history.map((entry) => (
              <button key={entry.id} type="button" onClick={() => restore(entry)} disabled={!tools.some((tool) => tool.name === entry.toolName)}>
                <strong>{entry.toolName}</strong>
                <span className={`history-status ${entry.status}`}>{entry.status}</span>
                <time dateTime={entry.invokedAt}>{new Date(entry.invokedAt).toLocaleString()}</time>
                <span>{entry.telemetry.elapsedMs} ms</span>
              </button>
            ))}
            {!history.length && <p className="muted">No local invocations yet.</p>}
          </div>
        </aside>
      </div>
    </div>
  );
}
