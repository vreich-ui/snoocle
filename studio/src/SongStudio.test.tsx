import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Song } from "./client";
import type { ToolStudioClient } from "./mcp";
import { SONG_STEPS } from "./steps";
import { contract, tool } from "./test-fixtures";
import { SongStudio } from "./SongStudio";
import type { StudioTool } from "./tooling";
import { EMPTY_WORKBENCH } from "./workbench";

function jsonResponse(status: number, body: unknown) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "Error",
    json: async () => body,
  };
}

function songFixture(overrides: Partial<Song> = {}): Song {
  return {
    schemaVersion: 2,
    id: "artist--song",
    metadata: { title: "Song", artist: "Artist", bpm: 120 },
    displayPreferences: { capo: 0, tuning: "standard" },
    audio: {},
    sections: [{ sectionIndex: 0, name: "Verse 1", kind: "verse", startLineIndex: 0, endLineIndex: 1 }],
    lines: [
      { lineIndex: 0, lyrics: "hello", chordPlacements: [{ charIndex: 0, chord: "C" }], timeSeconds: 1 },
      { lineIndex: 1, lyrics: "world", chordPlacements: [], timeSeconds: 2 },
    ],
    testOnly: false,
    ...overrides,
  };
}

const songsBody = {
  songs: ["artist--song"],
  items: [{
    id: "artist--song", title: "Song", artist: "Artist",
    latestVersion: "v1", updatedAt: "2026-08-01T00:00:00Z", youtubeVideoId: null, hasTiming: true,
  }],
};

const versionsBody = { songId: "artist--song", versions: [{ version: "v1", timestamp: "2026-08-01T00:00:00Z", message: "Manual save" }] };

function discoveredTools(): StudioTool[] {
  return SONG_STEPS.map((step) => tool(step.tool, {
    title: step.label,
    contract: contract({ title: step.label, category: "pipeline", browserSafety: "safe" }),
  }));
}

function mockClient(callTool?: ToolStudioClient["callTool"]): ToolStudioClient {
  return {
    connectAndDiscover: vi.fn().mockResolvedValue(discoveredTools()),
    callTool: callTool ?? vi.fn().mockResolvedValue({
      content: [{ type: "text", text: "ok" }],
      structuredContent: { ok: true, result: {}, elapsedMs: 1, outputSummary: {} },
    } satisfies CallToolResult),
    close: vi.fn().mockResolvedValue(undefined),
  };
}

function stepCard(label: string): HTMLElement {
  return screen.getByText(label).closest(".step-card") as HTMLElement;
}

describe("SongStudio", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    window.sessionStorage.clear();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  function setupFetch(options: { postStatus?: number; postBody?: unknown } = {}) {
    fetchMock.mockImplementation(async (path: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      if (path === "/v1/songs" && method === "GET") return jsonResponse(200, songsBody);
      if (path === "/v1/songs/artist--song/versions" && method === "GET") return jsonResponse(200, versionsBody);
      if (path === "/v1/songs/artist--song" && method === "GET") return jsonResponse(200, songFixture());
      if (path === "/v1/songs/artist--song" && method === "POST") {
        return jsonResponse(
          options.postStatus ?? 200,
          options.postBody ?? { version: "v2", timestamp: "2026-08-02T00:00:00Z", message: "saved" },
        );
      }
      throw new Error(`unexpected request ${method} ${path}`);
    });
  }

  it("lists songs when no songId is given, and selecting one navigates and sets the workbench song", async () => {
    setupFetch();
    const onNavigate = vi.fn();
    const onBenchChange = vi.fn();
    render(
      <SongStudio
        token="tab-token"
        bench={EMPTY_WORKBENCH}
        onBenchChange={onBenchChange}
        onNavigate={onNavigate}
        clientFactory={() => mockClient()}
      />,
    );

    expect(await screen.findByText("Song — Artist")).toBeVisible();
    await userEvent.click(screen.getByText("Song — Artist"));
    expect(onNavigate).toHaveBeenCalledWith("/studio/song-studio/artist--song");
    expect(onBenchChange).toHaveBeenCalledWith(
      expect.objectContaining({ song: { id: "artist--song", title: "Song", artist: "Artist" } }),
    );
  });

  it("shows the subject header and the step list once a song is loaded", async () => {
    setupFetch();
    render(
      <SongStudio
        songId="artist--song"
        token="tab-token"
        bench={EMPTY_WORKBENCH}
        onBenchChange={vi.fn()}
        onNavigate={vi.fn()}
        clientFactory={() => mockClient()}
      />,
    );

    expect(await screen.findByRole("heading", { name: "Song — Artist" })).toBeVisible();
    expect(screen.getByText("artist--song")).toBeVisible();
    for (const step of SONG_STEPS) {
      expect(screen.getByText(step.label)).toBeVisible();
    }
  });

  it("runs a report step and shows its outputSummary as a definition list, not a raw dump", async () => {
    setupFetch();
    const callTool: ToolStudioClient["callTool"] = vi.fn(async (calledTool) => {
      if (calledTool.name !== "validate_song_json") throw new Error(`unexpected tool ${calledTool.name}`);
      return {
        content: [{ type: "text", text: "ok" }],
        structuredContent: { ok: true, result: { valid: true }, elapsedMs: 12, outputSummary: { valid: true, lineCount: 2 } },
      } satisfies CallToolResult;
    });
    render(
      <SongStudio
        songId="artist--song"
        token="tab-token"
        bench={EMPTY_WORKBENCH}
        onBenchChange={vi.fn()}
        onNavigate={vi.fn()}
        clientFactory={() => mockClient(callTool)}
      />,
    );
    await screen.findByRole("heading", { name: "Song — Artist" });

    const runButton = within(stepCard("Validate against the schema")).getByRole("button", { name: "Run" });
    await waitFor(() => expect(runButton).toBeEnabled());
    await userEvent.click(runButton);

    expect(await screen.findByText("valid")).toBeVisible();
    expect(screen.getByText("lineCount")).toBeVisible();
    expect(screen.getByText("true")).toBeVisible();
    expect(screen.getByText("2")).toBeVisible();
    expect(screen.getByText("Raw result")).toBeVisible();
    expect(callTool).toHaveBeenCalledWith(
      expect.objectContaining({ name: "validate_song_json" }),
      { song_id: "artist--song" },
      expect.any(AbortSignal),
    );
  });

  it("runs a song-producing step, shows the Candidate diff, and Save posts with the loaded expectedVersion", async () => {
    setupFetch();
    const alteredSong = songFixture({
      lines: [
        { lineIndex: 0, lyrics: "hello", chordPlacements: [{ charIndex: 0, chord: "C" }], timeSeconds: 9 },
        { lineIndex: 1, lyrics: "world", chordPlacements: [], timeSeconds: 2 },
      ],
    });
    const callTool: ToolStudioClient["callTool"] = vi.fn(async (calledTool) => {
      if (calledTool.name !== "snap_song_to_mir") throw new Error(`unexpected tool ${calledTool.name}`);
      return {
        content: [{ type: "text", text: "ok" }],
        structuredContent: {
          ok: true,
          result: { song: alteredSong, songSource: "song_id", songVersion: "v1" },
          elapsedMs: 80,
          outputSummary: { timedLines: 2 },
        },
      } satisfies CallToolResult;
    });
    render(
      <SongStudio
        songId="artist--song"
        token="tab-token"
        bench={EMPTY_WORKBENCH}
        onBenchChange={vi.fn()}
        onNavigate={vi.fn()}
        clientFactory={() => mockClient(callTool)}
      />,
    );
    await screen.findByRole("heading", { name: "Song — Artist" });

    const runButton = within(stepCard("Snap timing to the audio")).getByRole("button", { name: "Run" });
    await waitFor(() => expect(runButton).toBeEnabled());
    await userEvent.click(runButton);

    expect(await screen.findByText("Candidate")).toBeVisible();
    expect(screen.getByText("1 of 2 lines changed timing")).toBeVisible();

    await userEvent.click(screen.getByRole("button", { name: "Save as new version" }));

    const findPostCall = () => fetchMock.mock.calls.find(
      (call) => call[0] === "/v1/songs/artist--song" && (call[1] as RequestInit | undefined)?.method === "POST",
    );
    await waitFor(() => expect(findPostCall()).toBeDefined());
    const postInit = findPostCall()![1] as RequestInit;
    const body = JSON.parse(postInit.body as string);
    expect(body.expectedVersion).toBe("v1");
    expect(body.message).toBe("Song Studio: Snap timing to the audio");
    expect(body.song.lines[0].timeSeconds).toBe(9);

    await waitFor(() => expect(screen.queryByText("Candidate")).not.toBeInTheDocument());
  });

  it("renders 'someone saved first' on a 409 from save, not a generic error", async () => {
    setupFetch({ postStatus: 409, postBody: { detail: "version conflict" } });
    const alteredSong = songFixture({
      lines: [
        { lineIndex: 0, lyrics: "hello", chordPlacements: [{ charIndex: 0, chord: "C" }], timeSeconds: 9 },
        { lineIndex: 1, lyrics: "world", chordPlacements: [], timeSeconds: 2 },
      ],
    });
    const callTool: ToolStudioClient["callTool"] = vi.fn().mockResolvedValue({
      content: [{ type: "text", text: "ok" }],
      structuredContent: { ok: true, result: { song: alteredSong }, elapsedMs: 80, outputSummary: {} },
    } satisfies CallToolResult);
    render(
      <SongStudio
        songId="artist--song"
        token="tab-token"
        bench={EMPTY_WORKBENCH}
        onBenchChange={vi.fn()}
        onNavigate={vi.fn()}
        clientFactory={() => mockClient(callTool)}
      />,
    );
    await screen.findByRole("heading", { name: "Song — Artist" });

    const runButton = within(stepCard("Snap timing to the audio")).getByRole("button", { name: "Run" });
    await waitFor(() => expect(runButton).toBeEnabled());
    await userEvent.click(runButton);
    await screen.findByText("Candidate");

    await userEvent.click(screen.getByRole("button", { name: "Save as new version" }));
    expect(await screen.findByText(/someone saved first/i)).toBeVisible();
  });

  it("disables a step that needs audio when the workbench has none", async () => {
    setupFetch();
    render(
      <SongStudio
        songId="artist--song"
        token="tab-token"
        bench={EMPTY_WORKBENCH}
        onBenchChange={vi.fn()}
        onNavigate={vi.fn()}
        clientFactory={() => mockClient()}
      />,
    );
    await screen.findByRole("heading", { name: "Song — Artist" });

    const validateButton = within(stepCard("Validate against the schema")).getByRole("button", { name: "Run" });
    await waitFor(() => expect(validateButton).toBeEnabled());

    const alignCard = stepCard("Align deterministically (whole pass)");
    expect(within(alignCard).getByRole("button", { name: "Run" })).toBeDisabled();
    expect(within(alignCard).getByText("Needs audio in the workbench.")).toBeVisible();
  });

  // Arriving by route (bookmark, back button, a Library link) changes the song
  // on screen. Leaving the workbench on the previous song is how one song's
  // audio ends up aligned against another's transcription.
  it("adopts the routed song into the workbench when they disagree", async () => {
    setupFetch();
    const onBenchChange = vi.fn();
    render(
      <SongStudio
        songId="artist--song"
        token="tab-token"
        bench={{ song: { id: "someone-else--other", title: "Other", artist: "Someone Else" } }}
        onBenchChange={onBenchChange}
        onNavigate={vi.fn()}
        clientFactory={() => mockClient()}
      />,
    );

    await waitFor(() => expect(onBenchChange).toHaveBeenCalled());
    expect(onBenchChange.mock.calls[0][0].song).toEqual({
      id: "artist--song", title: "Song", artist: "Artist",
    });
  });

  it("leaves the workbench alone when it already holds the routed song", async () => {
    setupFetch();
    const onBenchChange = vi.fn();
    render(
      <SongStudio
        songId="artist--song"
        token="tab-token"
        bench={{ song: { id: "artist--song", title: "Song", artist: "Artist" } }}
        onBenchChange={onBenchChange}
        onNavigate={vi.fn()}
        clientFactory={() => mockClient()}
      />,
    );

    await screen.findByRole("heading", { name: "Song — Artist" });
    expect(onBenchChange).not.toHaveBeenCalled();
  });

  it("refuses to run audio steps against a recording acquired for another song", async () => {
    setupFetch();
    render(
      <SongStudio
        songId="artist--song"
        token="tab-token"
        bench={{
          song: { id: "artist--song", title: "Song", artist: "Artist" },
          audio: {
            audioRef: "aud_123",
            filename: "other.webm",
            songId: "nirvana--smells-like-teen-spirit",
            videoTitle: "Nirvana - Smells Like Teen Spirit",
          },
        }}
        onBenchChange={vi.fn()}
        onNavigate={vi.fn()}
        clientFactory={() => mockClient()}
      />,
    );

    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText(/acquired for a different song/)).toBeVisible();

    const align = SONG_STEPS.find((step) => step.needsAudio)!;
    await waitFor(() => expect(within(stepCard(align.label)).getByRole("button")).toBeDisabled());
  });

  it("names the recording an audio step will use", async () => {
    setupFetch();
    render(
      <SongStudio
        songId="artist--song"
        token="tab-token"
        bench={{
          song: { id: "artist--song", title: "Song", artist: "Artist" },
          audio: {
            audioRef: "aud_123",
            filename: "song.webm",
            songId: "artist--song",
            videoTitle: "Artist - Song",
            youtubeVideoId: "vid123",
          },
        }}
        onBenchChange={vi.fn()}
        onNavigate={vi.fn()}
        clientFactory={() => mockClient()}
      />,
    );

    const align = SONG_STEPS.find((step) => step.needsAudio)!;
    const card = await waitFor(() => stepCard(align.label));
    expect(within(card).getByText("Uses Artist - Song (YouTube vid123).")).toBeVisible();
  });
});
