import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WorkbenchBar } from "./Workbench";
import type { Workbench } from "./workbench";

function jsonResponse(status: number, body: unknown) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "Error",
    json: async () => body,
  };
}

const songsBody = {
  songs: ["radiohead--karma-police", "unknown--mystery-track"],
  items: [
    {
      id: "radiohead--karma-police", title: "Karma Police", artist: "Radiohead",
      latestVersion: "v1", updatedAt: "2026-08-01T12:00:00Z", youtubeVideoId: "abc123", hasTiming: true,
    },
    {
      id: "unknown--mystery-track", title: "Mystery Track", artist: "Unknown",
      latestVersion: "v1", updatedAt: "2026-08-02T12:00:00Z", youtubeVideoId: null, hasTiming: false,
    },
  ],
};

describe("WorkbenchBar", () => {
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

  it("renders title — artist for the song slot and Clear empties it", async () => {
    const onChange = vi.fn();
    const bench: Workbench = { song: { id: "artist--song", title: "Song Title", artist: "The Artist" } };
    fetchMock.mockImplementation(async (path: string) => {
      if (path === "/v1/songs/artist--song/versions") return jsonResponse(404, { detail: "no versions" });
      throw new Error(`unexpected path ${path}`);
    });
    const user = userEvent.setup();
    render(<WorkbenchBar bench={bench} token="tab-token" onChange={onChange} />);

    expect(await screen.findByText("Song Title — The Artist")).toBeVisible();
    expect(screen.getByText("artist--song")).toBeVisible();

    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange.mock.calls[0][0].song).toBeUndefined();
  });

  it("shows 'Pick song' when the slot is empty, and 'none' for audio and MIR", () => {
    render(<WorkbenchBar bench={{}} token="tab-token" onChange={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Pick song" })).toBeVisible();
    expect(screen.getAllByText("none")).toHaveLength(2);
  });

  it("filters the picker by search and selecting a song sets the slot", async () => {
    const onChange = vi.fn();
    fetchMock.mockImplementation(async (path: string) => {
      if (path === "/v1/songs") return jsonResponse(200, songsBody);
      throw new Error(`unexpected path ${path}`);
    });
    const user = userEvent.setup();
    render(<WorkbenchBar bench={{}} token="tab-token" onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: "Pick song" }));
    expect(await screen.findByText("Karma Police — Radiohead")).toBeVisible();
    expect(screen.getByText("Mystery Track — Unknown")).toBeVisible();

    await user.type(screen.getByLabelText("Search songs"), "karma");
    expect(screen.queryByText("Mystery Track — Unknown")).not.toBeInTheDocument();

    await user.click(screen.getByText("Karma Police — Radiohead"));
    expect(onChange).toHaveBeenCalledWith({
      song: { id: "radiohead--karma-police", title: "Karma Police", artist: "Radiohead" },
    });
    // Selecting a song closes the inline panel.
    expect(screen.queryByLabelText("Search songs")).not.toBeInTheDocument();
  });

  it("defaults the version select to Latest and choosing a version sets song.version", async () => {
    const onChange = vi.fn();
    const bench: Workbench = { song: { id: "artist--song", title: "Song Title", artist: "The Artist" } };
    fetchMock.mockImplementation(async (path: string) => {
      if (path === "/v1/songs/artist--song/versions") {
        return jsonResponse(200, {
          songId: "artist--song",
          versions: [
            { version: "v2sha1234567890", timestamp: "2026-08-01T00:00:00Z", message: "second" },
            { version: "v1sha0987654321", timestamp: "2026-07-01T00:00:00Z", message: "first" },
          ],
        });
      }
      throw new Error(`unexpected path ${path}`);
    });
    const user = userEvent.setup();
    render(<WorkbenchBar bench={bench} token="tab-token" onChange={onChange} />);

    const select = await screen.findByLabelText("Song version");
    expect(select).toHaveValue("");
    // Wait for the fetched versions to land before interacting with the select.
    await waitFor(() => expect(within(select).getAllByRole("option")).toHaveLength(3));

    await user.selectOptions(select, "v2sha1234567890");
    expect(onChange).toHaveBeenCalledWith({
      ...bench,
      song: { ...bench.song, version: "v2sha1234567890" },
    });
  });

  it("warns when the audio slot was acquired for a different song, and clears it on request", async () => {
    const onChange = vi.fn();
    const bench: Workbench = {
      song: { id: "artist--song", title: "Song Title", artist: "The Artist" },
      audio: {
        audioRef: "aud_123",
        filename: "other.webm",
        songId: "nirvana--smells-like-teen-spirit",
        videoTitle: "Nirvana - Smells Like Teen Spirit",
        youtubeVideoId: "RNCH0xA-hNY",
      },
    };
    fetchMock.mockImplementation(async (path: string) => {
      if (path === "/v1/songs/artist--song/versions") return jsonResponse(404, { detail: "no versions" });
      throw new Error(`unexpected path ${path}`);
    });
    const user = userEvent.setup();
    render(<WorkbenchBar bench={bench} token="tab-token" onChange={onChange} />);

    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText(/Nirvana - Smells Like Teen Spirit/)).toBeVisible();

    await user.click(within(alert).getByRole("button", { name: "Clear the mismatched slots" }));
    expect(onChange.mock.calls[0][0].audio).toBeUndefined();
    expect(onChange.mock.calls[0][0].song).toEqual(bench.song);
  });

  it("shows what the audio slot actually is, and stays quiet when it matches", async () => {
    const bench: Workbench = {
      song: { id: "artist--song", title: "Song Title", artist: "The Artist" },
      audio: {
        audioRef: "aud_123",
        filename: "song.webm",
        durationSeconds: 185,
        songId: "artist--song",
        videoTitle: "The Artist - Song Title",
        youtubeVideoId: "vid123",
      },
    };
    fetchMock.mockImplementation(async (path: string) => {
      if (path === "/v1/songs/artist--song/versions") return jsonResponse(404, { detail: "no versions" });
      throw new Error(`unexpected path ${path}`);
    });
    render(<WorkbenchBar bench={bench} token="tab-token" onChange={vi.fn()} />);

    expect(await screen.findByText("The Artist - Song Title")).toBeVisible();
    expect(screen.getByText(/YouTube vid123/)).toBeVisible();
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
  });
});
