import { useEffect, useRef, useState } from "react";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { apiJson, ApiError, type Song, type SongsResponse, type SongSummary, type VersionsResponse } from "./client";
import { looksUnauthorized } from "./api";
import { createToolStudioClient, type ToolStudioClient } from "./mcp";
import { sectionPath, songStudioPath } from "./navigation";
import { Sheet } from "./Sheet";
import { argsForStep, extractCandidateSong, SONG_STEPS, summariseSongChange, type SongStep } from "./steps";
import { formatDateTime } from "./format";
import { formatJson, invocationView, isRecord, type StudioTool } from "./tooling";
import { useApi } from "./useApi";
import { type Workbench } from "./workbench";

type ClientFactory = (token: string) => ToolStudioClient;

interface SongStudioProps {
  songId?: string;
  token: string;
  bench: Workbench;
  onBenchChange(next: Workbench): void;
  onNavigate(path: string): void;
  clientFactory?: ClientFactory;
}

interface StepLogEntry {
  id: string;
  stepId: string;
  stepLabel: string;
  invokedAt: string;
  ok: boolean;
  elapsedMs: number;
  outputSummary?: Record<string, unknown>;
  result: unknown;
  errorCode?: string;
  errorMessage?: string;
  candidate?: { song: Song; diff: string[] };
}

type ConnectionState = "unauthenticated" | "connecting" | "connected" | "error" | "rejected";

function matchesQuery(item: SongSummary, query: string): boolean {
  return `${item.title} ${item.artist} ${item.id}`.toLocaleLowerCase().includes(query);
}

function newLogId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "unknown";
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

function summaryValueText(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return String(value);
  return formatJson(value);
}

export function SongStudio({ songId, token, bench, onBenchChange, onNavigate, clientFactory = createToolStudioClient }: SongStudioProps) {
  const [search, setSearch] = useState("");
  const songsApi = useApi<SongsResponse>(songId ? null : "/v1/songs", token);

  const [pinnedVersion, setPinnedVersion] = useState("");
  useEffect(() => {
    setPinnedVersion("");
  }, [songId]);

  const versionsApi = useApi<VersionsResponse>(songId ? `/v1/songs/${encodeURIComponent(songId)}/versions` : null, token);
  const songPath = songId
    ? `/v1/songs/${encodeURIComponent(songId)}${pinnedVersion ? `?version=${encodeURIComponent(pinnedVersion)}` : ""}`
    : null;
  const songApi = useApi<Song>(songPath, token);

  const clientRef = useRef<ToolStudioClient | undefined>(undefined);
  const [connection, setConnection] = useState<ConnectionState>(() => (token ? "connecting" : "unauthenticated"));
  const [connectionError, setConnectionError] = useState("");
  const [discovered, setDiscovered] = useState<StudioTool[]>([]);
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    let active = true;
    setConnectionError("");
    if (!token || !songId) {
      setConnection(token ? "connecting" : "unauthenticated");
      return () => {
        active = false;
      };
    }
    const client = clientFactory(token);
    clientRef.current = client;
    setConnection("connecting");
    client.connectAndDiscover().then((tools) => {
      if (!active) return;
      setDiscovered(tools);
      setConnection("connected");
    }).catch((error: unknown) => {
      if (!active) return;
      const message = error instanceof Error ? error.message : String(error);
      setConnection(looksUnauthorized(message) ? "rejected" : "error");
      setConnectionError(message);
    });
    return () => {
      active = false;
      void client.close().catch(() => undefined);
      if (clientRef.current === client) clientRef.current = undefined;
    };
  }, [clientFactory, token, songId, retry]);

  const [log, setLog] = useState<StepLogEntry[]>([]);
  const [busyStepId, setBusyStepId] = useState<string | undefined>(undefined);
  const [savingId, setSavingId] = useState<string | undefined>(undefined);
  const [saveErrors, setSaveErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    setLog([]);
    setSaveErrors({});
  }, [songId]);

  const toolFor = (name: string): StudioTool => discovered.find((tool) => tool.name === name) ?? ({ name } as StudioTool);

  const runStep = async (step: SongStep) => {
    const client = clientRef.current;
    if (!client || !songId) return;
    const args = argsForStep(step, songId, pinnedVersion || undefined, bench.audio?.audioRef);
    setBusyStepId(step.id);
    const started = performance.now();
    try {
      const raw = await client.callTool(toolFor(step.tool), args, new AbortController().signal) as CallToolResult;
      const view = invocationView(raw, performance.now() - started);
      const structured = view.structured;
      const envelopeResult = isRecord(structured) ? structured.result : undefined;
      const outputSummary = isRecord(structured) && isRecord(structured.outputSummary) ? structured.outputSummary : undefined;
      const errorRecord = isRecord(structured) ? structured.error : undefined;
      const errorCode = isRecord(errorRecord) && typeof errorRecord.code === "string" ? errorRecord.code : undefined;

      let candidate: StepLogEntry["candidate"];
      if (!view.failed && step.produces === "song" && songApi.data) {
        const candidateSong = extractCandidateSong(envelopeResult);
        if (candidateSong) candidate = { song: candidateSong, diff: summariseSongChange(songApi.data, candidateSong) };
      }

      setLog((current) => [{
        id: newLogId(),
        stepId: step.id,
        stepLabel: step.label,
        invokedAt: new Date().toISOString(),
        ok: !view.failed,
        elapsedMs: view.telemetry.elapsedMs,
        outputSummary,
        result: envelopeResult ?? structured,
        errorCode,
        errorMessage: view.failed ? view.errorMessage : undefined,
        candidate,
      }, ...current]);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setLog((current) => [{
        id: newLogId(),
        stepId: step.id,
        stepLabel: step.label,
        invokedAt: new Date().toISOString(),
        ok: false,
        elapsedMs: Math.max(0, Math.round(performance.now() - started)),
        result: undefined,
        errorMessage: message,
      }, ...current]);
    } finally {
      setBusyStepId(undefined);
    }
  };

  const noVersions = versionsApi.state === "error" && versionsApi.error?.status === 404;
  const versionList = noVersions ? [] : versionsApi.data?.versions ?? [];

  const saveCandidate = async (entry: StepLogEntry) => {
    if (!entry.candidate || !songId) return;
    setSavingId(entry.id);
    setSaveErrors((current) => ({ ...current, [entry.id]: "" }));
    const expectedVersion = pinnedVersion || versionList[0]?.version;
    try {
      await apiJson(`/v1/songs/${encodeURIComponent(songId)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          song: entry.candidate.song,
          message: `Song Studio: ${entry.stepLabel}`,
          expectedVersion,
        }),
      });
      setLog((current) => current.map((item) => (item.id === entry.id ? { ...item, candidate: undefined } : item)));
      songApi.reload();
      versionsApi.reload();
    } catch (error) {
      const message = error instanceof ApiError && error.status === 409
        ? "Someone saved first — reload."
        : error instanceof Error ? error.message : String(error);
      setSaveErrors((current) => ({ ...current, [entry.id]: message }));
    } finally {
      setSavingId(undefined);
    }
  };

  const discardCandidate = (entry: StepLogEntry) => {
    setLog((current) => current.map((item) => (item.id === entry.id ? { ...item, candidate: undefined } : item)));
  };

  if (!songId) {
    const items = songsApi.data?.items ?? [];
    const query = search.trim().toLocaleLowerCase();
    const visible = query ? items.filter((item) => matchesQuery(item, query)) : items;
    return (
      <section className="workspace song-studio" aria-labelledby="song-studio-heading">
        <p className="eyebrow">Work on a song</p>
        <h2 id="song-studio-heading">Song Studio</h2>

        {songsApi.state === "unauthenticated" && (
          <p className="muted">Paste the server's SNOOCLE_API_TOKEN into the sidebar to load songs.</p>
        )}
        {songsApi.state === "loading" && <p className="muted" role="status">Loading songs…</p>}
        {songsApi.state === "error" && (
          <div className="error" role="alert">
            <p>{songsApi.error?.detail}</p>
            <button type="button" onClick={songsApi.reload}>Retry</button>
          </div>
        )}
        {songsApi.state === "ready" && (
          items.length === 0 ? (
            <p className="muted">No songs in the store yet.</p>
          ) : (
            <>
              <label className="library-search">
                <span>Search</span>
                <input type="search" value={search} onChange={(event) => setSearch(event.target.value)} />
              </label>
              <p className="catalog-count">Showing {visible.length} of {items.length}</p>
              <div className="song-list">
                {visible.map((item) => (
                  <button
                    className="song-row"
                    key={item.id}
                    type="button"
                    onClick={() => {
                      onBenchChange({ ...bench, song: { id: item.id, title: item.title, artist: item.artist } });
                      onNavigate(songStudioPath(item.id));
                    }}
                  >
                    <strong>{item.title} — {item.artist}</strong>
                    <code>{item.id}</code>
                  </button>
                ))}
                {!visible.length && <p className="muted">No songs match.</p>}
              </div>
            </>
          )
        )}
      </section>
    );
  }

  if (songApi.state === "unauthenticated") {
    return (
      <section className="workspace song-studio" aria-labelledby="song-studio-heading">
        <h2 id="song-studio-heading">Song Studio</h2>
        <p className="muted">Paste the server's SNOOCLE_API_TOKEN into the sidebar to load this song.</p>
      </section>
    );
  }

  if (songApi.state === "loading" || songApi.state === "idle") {
    return (
      <section className="workspace song-studio" aria-labelledby="song-studio-heading">
        <h2 id="song-studio-heading">Song Studio</h2>
        <p className="muted" role="status">Loading song…</p>
      </section>
    );
  }

  if (songApi.state === "error" || !songApi.data) {
    return (
      <section className="workspace song-studio" aria-labelledby="song-studio-heading">
        <h2 id="song-studio-heading">Song Studio</h2>
        <div className="error" role="alert">
          <p>{songApi.error?.detail}</p>
          <button type="button" onClick={songApi.reload}>Retry</button>
        </div>
      </section>
    );
  }

  const data = songApi.data;

  return (
    <section className="workspace song-studio" aria-labelledby="song-studio-heading">
      <p className="eyebrow">Song Studio</p>
      <h2 id="song-studio-heading">Song Studio</h2>

      <section className="song-studio-subject" aria-labelledby="subject-heading">
        <div className="song-studio-subject-head">
          <div>
            <h3 id="subject-heading">{data.metadata.title} — {data.metadata.artist}</h3>
            <code>{data.id}</code>
          </div>
          <button type="button" onClick={() => onNavigate(sectionPath("Song Studio"))}>Change song</button>
        </div>
        <label>
          <span>Version</span>
          <select value={pinnedVersion} onChange={(event) => setPinnedVersion(event.target.value)}>
            <option value="">Latest</option>
            {versionList.map((version) => (
              <option key={version.version} value={version.version}>
                {version.version.slice(0, 10)} · {formatDateTime(version.timestamp)}
              </option>
            ))}
          </select>
        </label>
        <dl className="classification-grid song-meta">
          <div><dt>Key</dt><dd>{data.metadata.key ?? "unknown"}</dd></div>
          <div><dt>BPM</dt><dd>{data.metadata.bpm ?? "unknown"}</dd></div>
          <div><dt>Time signature</dt><dd>{data.metadata.timeSignature ?? "unknown"}</dd></div>
          <div><dt>Duration</dt><dd>{formatDuration(data.audio.durationSeconds)}</dd></div>
        </dl>
        <Sheet song={data} />
      </section>

      {connection === "rejected" && (
        <div className="connection-error" role="alert">
          <strong>The bearer token was rejected.</strong>
          <p>Steps cannot run until a valid token connects to /mcp.</p>
          <button type="button" onClick={() => setRetry((value) => value + 1)}>Retry connection</button>
        </div>
      )}
      {connection === "error" && (
        <div className="connection-error" role="alert">
          <strong>Could not connect to run steps.</strong>
          <p>{connectionError}</p>
          <button type="button" onClick={() => setRetry((value) => value + 1)}>Retry connection</button>
        </div>
      )}

      <div className="song-studio-body">
        <section className="song-studio-steps" aria-label="Pipeline steps">
          <p className="eyebrow">Steps</p>
          <div className="step-list">
            {SONG_STEPS.map((step) => {
              const audioBlocked = Boolean(step.needsAudio) && !bench.audio;
              const running = busyStepId === step.id;
              return (
                <div className="tool-card step-card" key={step.id}>
                  <strong>{step.label}</strong>
                  <p className="muted">{step.description}</p>
                  <span className="status-pill">{step.produces === "song" ? "Changes the song" : "Report only"}</span>
                  {audioBlocked && <p className="field-help">Needs audio in the workbench.</p>}
                  <div className="form-actions">
                    <button
                      type="button"
                      disabled={audioBlocked || Boolean(busyStepId) || connection !== "connected"}
                      onClick={() => runStep(step)}
                    >{running ? "Running…" : "Run"}</button>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="song-studio-log" aria-label="Step log">
          <p className="eyebrow">Step log</p>
          {!log.length && <p className="muted">No steps run yet.</p>}
          <div className="step-log-list">
            {log.map((entry) => (
              <article className={entry.ok ? "result-panel step-log-entry" : "result-panel step-log-entry failed"} key={entry.id}>
                <div className="step-log-heading">
                  <strong>{entry.stepLabel}</strong>
                  <span className={entry.ok ? "history-status success" : "history-status error"}>{entry.ok ? "ok" : "failed"}</span>
                  <span className="muted">{entry.elapsedMs} ms</span>
                </div>
                {entry.ok ? (
                  entry.outputSummary && Object.keys(entry.outputSummary).length > 0 && (
                    <dl className="telemetry-grid step-log-summary">
                      {Object.entries(entry.outputSummary).map(([key, value]) => (
                        <div key={key}><dt>{key}</dt><dd>{summaryValueText(value)}</dd></div>
                      ))}
                    </dl>
                  )
                ) : (
                  <p className="error" role="alert">{entry.errorCode ? `${entry.errorCode}: ` : ""}{entry.errorMessage}</p>
                )}
                <details><summary>Raw result</summary><pre>{formatJson(entry.result)}</pre></details>
                {entry.candidate && (
                  <div className="candidate-panel">
                    <h4>Candidate</h4>
                    <ul className="candidate-diff">
                      {entry.candidate.diff.map((line) => <li key={line}>{line}</li>)}
                    </ul>
                    {saveErrors[entry.id] && <p className="error" role="alert">{saveErrors[entry.id]}</p>}
                    <div className="form-actions">
                      <button type="button" disabled={savingId === entry.id} onClick={() => saveCandidate(entry)}>
                        {savingId === entry.id ? "Saving…" : "Save as new version"}
                      </button>
                      <button type="button" disabled={savingId === entry.id} onClick={() => discardCandidate(entry)}>Discard</button>
                    </div>
                  </div>
                )}
              </article>
            ))}
          </div>
        </section>
      </div>
    </section>
  );
}
