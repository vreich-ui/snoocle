import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SongDetail } from "./SongDetail";
import type { Song } from "./client";

function jsonResponse(status: number, body: unknown) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "Error",
    json: async () => body,
    text: async () => JSON.stringify(body),
  };
}

const song: Song = {
  schemaVersion: 2,
  id: "radiohead--karma-police",
  metadata: { title: "Karma Police", artist: "Radiohead", album: null, year: 1997, key: "Am", bpm: 74, timeSignature: "4/4" },
  displayPreferences: { capo: 0, tuning: "standard" },
  audio: { youtubeVideoId: "1uYWYWPc9HU", durationSeconds: 261, analyzedVideoId: "1uYWYWPc9HU" },
  sections: [{ sectionIndex: 0, name: "Verse 1", kind: "verse", startLineIndex: 0, endLineIndex: 0 }],
  lines: [{ lineIndex: 0, lyrics: "This is what you'll get", chordPlacements: [{ charIndex: 0, chord: "Am" }] }],
  testOnly: false,
};

const runsBody = {
  songId: song.id,
  runs: [{
    runId: "run-1", songId: song.id, provider: "mock", model: "mock-1", depth: "standard", status: "ok",
    startedAt: "2026-08-01T00:00:00Z", finishedAt: "2026-08-01T00:01:00Z", error: null,
    stepCount: 3, costUSD: 0.01, effortLevel: "standard", batchId: null,
  }],
};

/** A real July run: written before costUSD/effortLevel/batchId existed. */
const legacyRunsBody = {
  songId: song.id,
  runs: [{
    runId: "17f2eb8ef20a4525", songId: song.id, provider: "anthropic-agent", model: "claude-opus-4-8",
    depth: "standard", status: "ok", startedAt: "2026-07-31T07:03:13+00:00",
    finishedAt: "2026-07-31T07:07:01+00:00", error: null, stepCount: 9,
  }],
};

const goldBody = { songId: song.id, goldVersion: null };
const notesBody = { songId: song.id, notes: "", updatedAt: null, preference: null, correction: null };

function route(path: string, extra: Record<string, unknown> = {}) {
  if (path === `/v1/songs/${song.id}`) return jsonResponse(200, song);
  if (path === `/v1/songs/${song.id}/runs`) return jsonResponse(200, runsBody);
  if (path === `/v1/songs/${song.id}/gold`) return jsonResponse(200, goldBody);
  if (path === `/v1/songs/${song.id}/notes`) return jsonResponse(200, notesBody);
  if (path === `/v1/songs/${song.id}/versions`) return jsonResponse(404, { detail: "no versions" });
  return jsonResponse(404, { detail: `unhandled ${path}`, ...extra });
}

describe("SongDetail", () => {
  const fetchMock = vi.fn();
  const onNavigate = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    onNavigate.mockReset();
    window.sessionStorage.clear();
    vi.stubGlobal("fetch", fetchMock);
    fetchMock.mockImplementation(async (path: string) => route(path));
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders metadata, runs and the sheet, and shows 'no versions' rather than an error on a 404", async () => {
    render(<SongDetail songId={song.id} token="tab-token" onNavigate={onNavigate} />);

    expect(await screen.findByText("Karma Police — Radiohead")).toBeVisible();
    expect(screen.getByText("74")).toBeVisible();
    expect(screen.getByText("No versions yet.")).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByText(/mock \/ mock-1/)).toBeVisible();
    expect(screen.getByTestId("sheet").textContent).toContain("Am");
    expect(screen.getByTestId("sheet").textContent).toContain("This is what you'll get");
  });

  it("POSTs {artist,title} to the identity endpoint and navigates to the returned new id", async () => {
    fetchMock.mockImplementation(async (path: string, init?: RequestInit) => {
      if (path === `/v1/songs/${song.id}/identity`) {
        expect(JSON.parse(String(init?.body))).toEqual({ artist: "Radiohead", title: "Karma Police (Live)" });
        return jsonResponse(200, {
          oldSongId: song.id,
          songId: "radiohead--karma-police-live",
          versionMap: { v1: "v1-new" },
          migratedRunIds: ["run-1"],
        });
      }
      return route(path);
    });
    const user = userEvent.setup();
    render(<SongDetail songId={song.id} token="tab-token" onNavigate={onNavigate} />);
    await screen.findByText("Karma Police — Radiohead");

    const titleInput = screen.getByLabelText("Title");
    await user.clear(titleInput);
    await user.type(titleInput, "Karma Police (Live)");
    await user.click(screen.getByRole("button", { name: "Rename song" }));

    await waitFor(() => expect(onNavigate).toHaveBeenCalledWith("/studio/library/radiohead--karma-police-live"));
  });

  it("lists the missing fields from a 422 identity_unresolved response", async () => {
    fetchMock.mockImplementation(async (path: string) => {
      if (path === `/v1/songs/${song.id}/identity`) {
        return jsonResponse(422, {
          detail: "identity unresolved: missing artist",
          errorCode: "identity_unresolved",
          missing: ["artist"],
          evidenceTried: [],
          needsIdentity: true,
        });
      }
      return route(path);
    });
    const user = userEvent.setup();
    render(<SongDetail songId={song.id} token="tab-token" onNavigate={onNavigate} />);
    await screen.findByText("Karma Police — Radiohead");

    const artistInput = screen.getByLabelText("Artist");
    await user.clear(artistInput);
    await user.click(screen.getByRole("button", { name: "Rename song" }));

    expect(await screen.findByText("Missing: artist")).toBeVisible();
    expect(onNavigate).not.toHaveBeenCalled();
  });
  it("renders a song whose runs predate costUSD, instead of blanking the page", async () => {
    fetchMock.mockImplementation(async (path: string) => {
      if (path === `/v1/songs/${song.id}/runs`) return jsonResponse(200, legacyRunsBody);
      return route(path);
    });

    render(<SongDetail songId={song.id} token="tab-token" onNavigate={onNavigate} />);

    expect(await screen.findByText(/17f2eb8/)).toBeVisible();
    expect(screen.getByText("Karma Police — Radiohead")).toBeVisible();
  });
});
