import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { StudioApp } from "./App";
import { routeFromPath, runDetailPath, sectionPlan, songDetailPath } from "./navigation";

function jsonResponse(status: number, body: unknown) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "Error",
    json: async () => body,
  };
}

describe("StudioApp", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  beforeEach(() => {
    window.history.replaceState({}, "", "/studio/repair");
    window.sessionStorage.clear();
  });

  it("provides every Studio section", () => {
    render(<StudioApp />);
    for (const label of ["Repair", "Build", "Automatic Pipeline", "Tool Studio", "Library", "Runs", "Evaluation", "Configuration"]) {
      expect(screen.getByRole("button", { name: label })).toBeVisible();
    }
  });

  it("supports keyboard navigation with native buttons", async () => {
    render(<StudioApp />);
    const runs = screen.getByRole("button", { name: "Runs" });
    runs.focus();
    await userEvent.keyboard("{Enter}");
    expect(window.location.pathname).toBe("/studio/runs");
    expect(await screen.findByRole("heading", { name: "Runs" })).toBeVisible();
    expect(runs).toHaveAttribute("aria-current", "page");
  });

  it("keeps a entered bearer token in session storage only", () => {
    render(<StudioApp />);
    fireEvent.change(screen.getByLabelText("Bearer token"), { target: { value: "temporary-token" } });
    expect(window.sessionStorage.getItem("snoocle.studio.bearer-token")).toBe("temporary-token");
    expect(window.localStorage.getItem("snoocle.studio.bearer-token")).toBeNull();
  });

  it("lands an unknown path and a bare /studio/ on Tool Studio", async () => {
    window.history.replaceState({}, "", "/studio/nope");
    render(<StudioApp />);
    expect(await screen.findByRole("heading", { name: "Tool Studio" })).toBeVisible();
    cleanup();

    window.history.replaceState({}, "", "/studio/");
    render(<StudioApp />);
    expect(await screen.findByRole("heading", { name: "Tool Studio" })).toBeVisible();
  });

  it("shows the Not built yet pill, its plan sentence and a link to /ui/ for a placeholder section", () => {
    render(<StudioApp />);
    expect(screen.getByRole("heading", { name: "Repair" })).toBeVisible();
    expect(screen.getByText("Not built yet")).toBeVisible();
    expect(screen.getByText(sectionPlan.Repair)).toBeVisible();
    const link = screen.getByRole("link", { name: "/ui/" });
    expect(link).toHaveAttribute("href", "/ui/");
  });

  it("parses both detail routes, round-trips an id with a slash and spaces, and falls back to Tool Studio", () => {
    expect(routeFromPath("/studio/library/radiohead--karma-police")).toEqual({
      section: "Library",
      detailId: "radiohead--karma-police",
    });
    expect(routeFromPath("/studio/runs/run-123")).toEqual({ section: "Runs", detailId: "run-123" });
    expect(routeFromPath("/studio/nope")).toEqual({ section: "Tool Studio" });
    expect(routeFromPath("/studio/")).toEqual({ section: "Tool Studio" });

    const trickyId = "weird/id with spaces";
    expect(routeFromPath(songDetailPath(trickyId))).toEqual({ section: "Library", detailId: trickyId });
    expect(routeFromPath(runDetailPath(trickyId))).toEqual({ section: "Runs", detailId: trickyId });
  });

  it("navigates to Library and renders a song row fetched from the REST API", async () => {
    window.sessionStorage.setItem("snoocle.studio.bearer-token", "tab-token");
    vi.stubGlobal("fetch", vi.fn(async (path: string) => {
      if (path === "/v1/songs") {
        return jsonResponse(200, {
          songs: ["artist--song"],
          items: [{
            id: "artist--song", title: "Song", artist: "Artist", latestVersion: "v1",
            updatedAt: "2026-01-01T00:00:00Z", youtubeVideoId: null, hasTiming: false,
          }],
        });
      }
      if (path === "/v1/songs/needs-identity") return jsonResponse(200, { songs: [] });
      return jsonResponse(404, { detail: `unhandled ${path}` });
    }));
    render(<StudioApp />);
    await userEvent.setup().click(screen.getByRole("button", { name: "Library" }));
    expect(await screen.findByRole("heading", { name: "Library" })).toBeVisible();
    expect(await screen.findByText("Song — Artist")).toBeVisible();
    expect(window.location.pathname).toBe("/studio/library");
  });

  it("navigates to Runs and renders the queue's worker line", async () => {
    window.sessionStorage.setItem("snoocle.studio.bearer-token", "tab-token");
    vi.stubGlobal("fetch", vi.fn(async (path: string) => {
      if (path === "/v1/queue") {
        return jsonResponse(200, {
          jobs: [], counts: {}, lastHeartbeatAt: null, lastWorker: null, workerSeenRecently: false,
          leaseSeconds: 60, maxAttempts: 3, maxPerSubmit: 20,
        });
      }
      if (path === "/v1/songs") return jsonResponse(200, { songs: [], items: [] });
      return jsonResponse(404, { detail: `unhandled ${path}` });
    }));
    render(<StudioApp />);
    await userEvent.setup().click(screen.getByRole("button", { name: "Runs" }));
    expect(await screen.findByRole("heading", { name: "Runs" })).toBeVisible();
    expect(await screen.findByText("No recent worker heartbeat")).toBeVisible();
  });
});
