import { useEffect, useState } from "react";
import { ApiError, fetchRecentRuns, runFailed, runStepView, type QueueResponse, type RunDetail, type RunSummary } from "./client";
import { runDetailPath } from "./navigation";
import { formatCost, formatDateTime, orDash, pairOrDash } from "./format";
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
  const [error, setError] = useState<ApiError | Error>();
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
    setError(undefined);
    fetchRecentRuns(RECENT_RUNS_LIMIT).then((result) => {
      if (!active) return;
      setRuns(result);
      setState("ready");
    }).catch((caught: unknown) => {
      if (!active) return;
      // Discarding this left "Could not load recent runs." as the whole
      // message, with a Retry that could only fail identically — while the
      // Queue panel beside it correctly said the token was rejected.
      setError(caught instanceof Error ? caught : new Error(String(caught)));
      setState("error");
    });
    return () => {
      active = false;
    };
  }, [token, attempt]);

  return { state, runs, error, reload: () => setAttempt((value) => value + 1) };
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
      {runFailed(data.status) && (data.error || data.reason) && (
        <p className="error" role="alert">{data.error ?? data.reason}</p>
      )}
      <dl className="classification-grid">
        <div><dt>Status</dt><dd>{data.status}</dd></div>
        <div><dt>Provider / model</dt><dd>{pairOrDash(data.provider, data.model)}</dd></div>
        <div><dt>Depth</dt><dd>{orDash(data.depth)}</dd></div>
        <div><dt>Effort</dt><dd>{orDash(data.effortLevel)}</dd></div>
        <div><dt>Started</dt><dd>{formatDateTime(data.startedAt)}</dd></div>
        <div><dt>Finished</dt><dd>{formatDateTime(data.finishedAt)}</dd></div>
        <div><dt>Cost</dt><dd>{formatCost(data.costUSD)}</dd></div>
      </dl>
      <div className="run-steps">
        {data.steps.map((step, index) => {
          const view = runStepView(step, index);
          return (
            <details key={view.key}>
              <summary>{view.summary ? `${view.label} — ${view.summary}` : view.label}</summary>
              {view.warnings.length > 0 && (
                <ul className="run-step-warnings">
                  {view.warnings.map((warning) => <li key={warning}>{warning}</li>)}
                </ul>
              )}
              <pre>{JSON.stringify(view.detail, null, 2)}</pre>
            </details>
          );
        })}
        {!data.steps.length && <p className="muted">This run recorded no steps.</p>}
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
              {queue.data.lastHeartbeatAt ? ` · ${formatDateTime(queue.data.lastHeartbeatAt)}` : ""}
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
                        <td>{formatDateTime(job.queuedAt)}</td>
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
        {recent.state === "idle" && (
          <p className="muted">Paste the server's SNOOCLE_API_TOKEN into the sidebar to load recent runs.</p>
        )}
        {recent.state === "loading" && <p className="muted" role="status">Loading recent runs…</p>}
        {recent.state === "error" && (
          <div className="error" role="alert">
            <p>
              {recent.error instanceof ApiError && recent.error.unauthorized
                ? "The bearer token was rejected. Check it matches SNOOCLE_API_TOKEN on the server."
                : recent.error?.message ?? "Could not load recent runs."}
            </p>
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
                    <tr key={run.runId}>
                      <td>
                        <button
                          type="button"
                          className="row-open"
                          onClick={() => onNavigate(runDetailPath(run.runId))}
                        >
                          <code>{run.runId.slice(0, 10)}</code>
                        </button>
                      </td>
                      <td><code>{run.songId}</code></td>
                      <td>{run.status}</td>
                      <td>{pairOrDash(run.provider, run.model)}</td>
                      <td>{formatDateTime(run.startedAt)}</td>
                      <td>{formatCost(run.costUSD)}</td>
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
