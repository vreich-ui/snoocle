import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LoadSongAudio } from "./LoadSongAudio";
import { saveBearerToken } from "./api";
import type { Workbench } from "./workbench";

const song = {
  id: "amy--back-to-black",
  title: "Back to Black",
  artist: "Amy Winehouse",
  youtubeVideoId: "abcdefghijk",
};

const artifact = {
  audioRef: "aud_abcdefghijklmnopqrstuvwxyzABCDEF",
  filename: "Back to Black [abcdefghijk].webm",
  contentType: "audio/webm",
  durationSeconds: 241,
  sizeBytes: 4_000_000,
  expiresAt: "2026-09-05T00:00:00Z",
  playbackUrl: "/v1/audio/artifacts/aud_x/content",
};

describe("LoadSongAudio", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    window.sessionStorage.clear();
    saveBearerToken("tab-token");
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("acquires the song's own recording and files it under that song", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({
      artifact,
      youtubeVideoId: "abcdefghijk",
      videoTitle: "Amy Winehouse - Back to Black",
    }), { status: 201, headers: { "Content-Type": "application/json" } }));
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<LoadSongAudio bench={{ song }} token="tab-token" onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: "Load this song's audio" }));

    await waitFor(() => expect(onChange).toHaveBeenCalled());
    const [path, init] = fetchMock.mock.calls[0];
    expect(path).toBe("/v1/audio/artifacts/acquire");
    expect(JSON.parse(init.body as string)).toEqual({ youtubeUrlOrId: "abcdefghijk" });
    expect(onChange.mock.calls[0][0].audio).toEqual({
      audioRef: artifact.audioRef,
      filename: artifact.filename,
      durationSeconds: 241,
      songId: song.id,
      youtubeVideoId: "abcdefghijk",
      videoTitle: "Amy Winehouse - Back to Black",
    });
  });

  // Nothing honest to offer: the store does not know which recording this song
  // came from, so there is no button to press.
  it("renders nothing when the song has no recording on file", () => {
    const { container } = render(
      <LoadSongAudio bench={{ song: { ...song, youtubeVideoId: undefined } }} token="tab-token" onChange={vi.fn()} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing with no song selected", () => {
    const { container } = render(<LoadSongAudio bench={{}} token="tab-token" onChange={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("says the audio is already loaded rather than fetching it twice", () => {
    const bench: Workbench = {
      song,
      audio: { audioRef: "aud_1", filename: "a.webm", songId: song.id, youtubeVideoId: "abcdefghijk" },
    };
    render(<LoadSongAudio bench={bench} token="tab-token" onChange={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Audio loaded" })).toBeDisabled();
  });

  it("surfaces an acquisition failure instead of leaving the button silent", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ detail: "video unavailable" }), {
      status: 502, headers: { "Content-Type": "application/json" },
    }));
    const user = userEvent.setup();
    render(<LoadSongAudio bench={{ song }} token="tab-token" onChange={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Load this song's audio" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("video unavailable");
  });

  // The failure that blocked the pipeline in production: YouTube bot-checks the
  // datacenter address, and the raw yt-dlp text said nothing about the fix.
  it("offers to reconnect YouTube when the download is bot-checked", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({
      detail: "yt-dlp failed for abcdefghijk: ERROR: [youtube] abcdefghijk: Sign in to confirm you're not a bot.",
      errorCode: "youtube_auth_required",
      reason: "YouTube connection expired or was blocked. Reconnect YouTube (sign in again in the app) and retry.",
    }), { status: 502, headers: { "Content-Type": "application/json" } }));
    const user = userEvent.setup();
    render(<LoadSongAudio bench={{ song }} token="tab-token" onChange={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Load this song's audio" }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("YouTube refused this download.");
    expect(screen.getByRole("button", { name: "Reconnect YouTube" })).toBeVisible();
  });

  it("still shows an ordinary failure as an ordinary failure", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ detail: "Video unavailable" }), {
      status: 502, headers: { "Content-Type": "application/json" },
    }));
    const user = userEvent.setup();
    render(<LoadSongAudio bench={{ song }} token="tab-token" onChange={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Load this song's audio" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Video unavailable");
    expect(screen.queryByRole("button", { name: "Reconnect YouTube" })).toBeNull();
  });
});
