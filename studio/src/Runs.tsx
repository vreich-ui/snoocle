import { useEffect, useState } from "react";
import { fetchRecentRuns, type QueueResponse, type RunDetail, type RunSummary } from "./client";
import { runDetailPath } from "./navigation";
import { useApi } from "./useApi";

interface RunsProps {
  token: string;
  detailId?: string;
  onNavigate(path: string): void;
}

const RECENT_RUNS_LIMIT = 25;

type RecentRunsState = "idle" | "loading" | "ready" | "error";

/** No endpoint lists runs across every song, so this drives fetchRecentRuns (client.ts) the same way useApi drives a single GET. */
function useRecentRuns(token: string) {
  const [state, setState] = useState<RecentRunsState>(token ? "loading" : "idle");
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    if (!token) {
      setState("idle");
      setRuns([]);
      return () => {
        active = false;
      };
    }
    setState("loading");
    fetchRecentRuns(RECENT_RUNS_LIMIT).then((result) => {
      if (!active) return;
      setRuns(result);
      setState("ready");
    }).catch(() => {
      if (!active) return;
      setState("error");
    });
    return () => {
      active = false;
    };
  }, [token, attempt]);

  return { state, runs, reload: () => setAttempt((value) => value + 1) };
}

interface RunDetailViewProps {
  runId: string;
  token: string;
  onNavigate(path: string): void;
}

function RunDetailView({ runId, token, onNavigate }: RunDetailViewProps) {
  const run = useApi<RunDetail>(`/v1/runs/${encodeURIComponent(runId)}`, token);

  if (run.state === "unauthenticated") {
    return (
      <section className="workspace" aria-labelledby="run-detail-heading">
        <h2 id="run-detail-heading">Run</h2>
        <p className="muted">Paste the server's SNOOCLE_API_TOKEN into the sidebar to load this run.</p>
      </section>
    );
  }

  if (run.state === "loading" || run.state === "idle") {
    return (
      <section className="workspace" aria-labelledby="run-detail-heading">
        <h2 id="run-detail-heading">Run</h2>
        <p className="muted" role="status">Loading run…</p>
      </section>
    );
  }

  if (run.state === "error" || !run.data) {
    return (
      <section className="workspace" aria-labelledby="run-detail-heading">
        <h2 id="run-detail-heading">Run</h2>
        <div className="error" role="alert">
          <p>{run.error?.detail}</p>
          <button type="button" onClick={run.reload}>Retry</button>
        </div>
      </section>
    );
  }

  const data = run.data;
  return (
    <section className="workspace run-detail" aria-labelledby="run-detail-heading">
      <button type="button" className="back-link" onClick={() => onNavigate("/studio/runs")}>← Back to Runs</button>
      <p className="eyebrow">Run</p>
      <h2 id="run-detail-heading"><code>{data.runId}</code></h2>
      {data.status === "error" && data.error && <p className="error" role="alert">{data.error}</p>}
      <dl className="classification-grid">
        <div><dt>Status</dt><dd>{data.status}</dd></div>
        <div><dt>Provider / model</dt><dd>{data.provider} / {data.model}</dd></div>
        <div><dt>Depth</dt><dd>{data.depth}</dd></div>
        <div><dt>Effort</dt><dd>{data.effortLevel}</dd></div>
        <div><dt>Started</dt><dd>{new Date(data.startedAt).toLocaleString()}</dd></div>
        <div><dt>Finished</dt><dd>{data.finishedAt ? new Date(data.finishedAt).toLocaleString() : "—"}</dd></div>
        <div><dt>Cost</dt><dd>${data.costUSD.toFixed(4)}</dd></div>
      </dl>
      <div className="run-steps">
        {data.steps.map((step) => (
          <details key={step.index}>
            <summary>{step.label} — {step.summary}</summary>
            <pre>{JSON.stringify(step.detail, null, 2)}</pre>
          </details>
        ))}
      </div>
    </section>
  );
}

export function Runs({ token, detailId, onNavigate }: RunsProps) {
  const queue = useApi<QueueResponse>("/v1/queue", token);
  const recent = useRecentRuns(token);

  if (detailId) return <RunDetailView runId={detailId} token={token} onNavigate={onNavigate} />;

  return (
    <section className="workspace runs" aria-labelledby="runs-heading">
      <p className="eyebrow">Run traces</p>
      <h2 id="runs-heading">Runs</h2>

      <section aria-labelledby="queue-heading">
        <h3 id="queue-heading">Queue</h3>
        {queue.state === "unauthenticated" && (
          <p className="muted">Paste the server's SNOOCLE_API_TOKEN into the sidebar to load the queue.</p>
        )}
        {queue.state === "loading" && <p className="muted" role="status">Loading queue…</p>}
        {queue.state === "error" && (
          <div className="error" role="alert">
            <p>{queue.error?.detail}</p>
            <button type="button" onClick={queue.reload}>Retry</button>
          </div>
        )}
        {queue.state === "ready" && queue.data && (
          <>
            <dl className="stat-row">
              {Object.entries(queue.data.counts).map(([status, count]) => (
                <div key={status}><dt>{status}</dt><dd>{count}</dd></div>
              ))}
            </dl>
            <p className="muted">
              {queue.data.workerSeenRecently ? "Worker seen recently" : "No recent worker heartbeat"}
              {queue.data.lastWorker ? ` · ${queue.data.lastWorker}` : ""}
              {queue.data.lastHeartbeatAt ? ` · ${new Date(queue.data.lastHeartbeatAt).toLocaleString()}` : ""}
            </p>
            {queue.data.jobs.length === 0 ? (
              <p className="muted">Queue is empty.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Label</th><th>Kind</th><th>Status</th><th>Queued</th><th>Worker</th><th>Attempts</th></tr></thead>
                  <tbody>
                    {queue.data.jobs.map((job) => (
                      <tr key={job.id}>
                        <td>{job.label}</td>
                        <td>{job.kind}</td>
                        <td>{job.status}</td>
                        <td>{new Date(job.queuedAt).toLocaleString()}</td>
                        <td>{job.worker ?? "—"}</td>
                        <td>{job.attempts}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </section>

      <section aria-labelledby="recent-runs-heading">
        <h3 id="recent-runs-heading">Recent runs</h3>
        <p className="muted">Aggregated in the browser from every song's run list, capped at {RECENT_RUNS_LIMIT}.</p>
        {recent.state === "loading" && <p className="muted" role="status">Loading recent runs…</p>}
        {recent.state === "error" && (
          <div className="error" role="alert">
            <p>Could not load recent runs.</p>
            <button type="button" onClick={recent.reload}>Retry</button>
          </div>
        )}
        {recent.state === "ready" && (
          recent.runs.length === 0 ? <p className="muted">No runs yet.</p> : (
            <div className="table-wrap">
              <table>
                <thead><tr><th>Run</th><th>Song</th><th>Status</th><th>Provider / model</th><th>Started</th><th>Cost</th></tr></thead>
                <tbody>
                  {recent.runs.map((run) => (
                    <tr key={run.runId} className="row-button" onClick={() => onNavigate(runDetailPath(run.runId))}>
                      <td><code>{run.runId.slice(0, 10)}</code></td>
                      <td><code>{run.songId}</code></td>
                      <td>{run.status}</td>
                      <td>{run.provider} / {run.model}</td>
                      <td>{new Date(run.startedAt).toLocaleString()}</td>
                      <td>${run.costUSD.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        )}
      </section>
    </section>
  );
}
