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
});
