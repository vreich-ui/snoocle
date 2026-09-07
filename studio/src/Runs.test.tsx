import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Runs } from "./Runs";

function jsonResponse(status: number, body: unknown) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "Error",
    json: async () => body,
  };
}

const queueBody = {
  jobs: [],
  counts: { queued: 2, running: 1, done: 5 },
  lastHeartbeatAt: "2026-08-22T09:00:00Z",
  lastWorker: "worker-1",
  workerSeenRecently: true,
  leaseSeconds: 60,
  maxAttempts: 3,
  maxPerSubmit: 20,
};

const emptySongs = { songs: [], items: [] };

describe("Runs", () => {
  const fetchMock = vi.fn();
  const onNavigate = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    onNavigate.mockReset();
    window.sessionStorage.clear();
    vi.stubGlobal("fetch", fetchMock);
    fetchMock.mockImplementation(async (path: string) => {
      if (path === "/v1/queue") return jsonResponse(200, queueBody);
      if (path === "/v1/songs") return jsonResponse(200, emptySongs);
      return jsonResponse(404, { detail: `unhandled ${path}` });
    });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders queue counts and the worker line", async () => {
    render(<Runs token="tab-token" onNavigate={onNavigate} />);

    expect(await screen.findByText(/Worker seen recently/)).toBeVisible();
    expect(screen.getByText(/worker-1/)).toBeVisible();
    expect(screen.getByText("queued")).toBeVisible();
    expect(screen.getByText("2")).toBeVisible();
    expect(screen.getByText("Queue is empty.")).toBeVisible();
    expect(await screen.findByText("No runs yet.")).toBeVisible();
  });
  it("renders a run written before costUSD/effortLevel existed", async () => {
    const legacyRun = {
      runId: "c5825eaccaca4ba8",
      songId: "unknown--unknown",
      provider: "anthropic-agent",
      model: "claude-opus-4-8",
      depth: "standard",
      status: "ok",
      startedAt: "2026-07-31T07:03:13+00:00",
      finishedAt: "2026-07-31T07:07:56+00:00",
      error: null,
      stepCount: 7,
    };
    fetchMock.mockImplementation(async (path: string) => {
      if (path === "/v1/queue") return jsonResponse(200, queueBody);
      if (path === "/v1/songs") return jsonResponse(200, { songs: ["unknown--unknown"], items: [{ id: "unknown--unknown", title: "Unknown", artist: "Unknown", latestVersion: "abc", updatedAt: "2026-07-31T07:07:56Z", youtubeVideoId: null, hasTiming: true }] });
      if (path === "/v1/songs/unknown--unknown/runs") return jsonResponse(200, { songId: "unknown--unknown", runs: [legacyRun] });
      return jsonResponse(404, { detail: `unhandled ${path}` });
    });

    render(<Runs token="tab-token" onNavigate={onNavigate} />);

    expect(await screen.findByText(/c5825ea/)).toBeVisible();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  // Every Song Studio step and every *_deterministically tool writes stage
  // records, which share no field name with the agent trace shape the reader
  // knew about — so each one rendered as a blank row keyed on undefined.
  it("renders a deterministic run's stages instead of blank rows", async () => {
    fetchMock.mockImplementation(async (path: string) => {
      if (path === "/v1/runs/run-det") {
        return jsonResponse(200, {
          runId: "run-det",
          songId: "artist--song",
          status: "failed",
          reason: "timing_collapse",
          runType: "deterministic-align",
          steps: [
            {
              name: "snap_to_mir",
              elapsedMs: 240,
              cacheStatus: "hit",
              modelCalls: 0,
              modelCostUSD: 0,
              inputSummary: { lines: 42 },
              outputSummary: { timedLines: 42 },
              warnings: ["two lines share a timestamp"],
            },
            { name: "guard_collapse", elapsedMs: 12, cacheStatus: "not_applicable", warnings: [] },
          ],
        });
      }
      throw new Error(`unexpected path ${path}`);
    });
    render(<Runs token="tab-token" detailId="run-det" onNavigate={vi.fn()} />);

    expect(await screen.findByText(/snap_to_mir/)).toBeVisible();
    expect(screen.getByText(/guard_collapse/)).toBeVisible();
    // The count is on the summary line; the text itself is inside the closed
    // disclosure, which is where a stage's detail belongs.
    expect(screen.getByText(/240 ms · cache hit · 1 warning/)).toBeVisible();
    expect(screen.getByText("two lines share a timestamp")).toBeInTheDocument();
    // A deterministic failure reports `reason`, not `error`, and its status is
    // "failed" — so the old `status === "error"` guard hid it entirely.
    expect(screen.getByRole("alert")).toHaveTextContent("timing_collapse");
  });

  it("still renders an agent trace's steps", async () => {
    fetchMock.mockImplementation(async (path: string) => {
      if (path === "/v1/runs/run-agent") {
        return jsonResponse(200, {
          runId: "run-agent",
          songId: "artist--song",
          status: "ok",
          steps: [{
            index: 0,
            kind: "tool",
            label: "tool:fetch_chord_sheet",
            summary: "fetched 3 candidates",
            detail: { count: 3 },
            timestamp: "2026-08-01T00:00:00Z",
            durationSeconds: 1.2,
          }],
        });
      }
      throw new Error(`unexpected path ${path}`);
    });
    render(<Runs token="tab-token" detailId="run-agent" onNavigate={vi.fn()} />);

    expect(await screen.findByText(/tool:fetch_chord_sheet — fetched 3 candidates/)).toBeVisible();
  });
});
