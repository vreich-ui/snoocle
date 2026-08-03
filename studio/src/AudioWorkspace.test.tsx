import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AudioWorkspace } from "./AudioWorkspace";
import { saveBearerToken } from "./api";
import { createWaveform } from "./waveform";

vi.mock("./waveform", () => ({ createWaveform: vi.fn() }));

const artifact = {
  audioRef: "aud_abcdefghijklmnopqrstuvwxyzABCDEF",
  filename: "tone.wav",
  contentType: "audio/wav",
  durationSeconds: 1.25,
  sizeBytes: 25_000,
  expiresAt: "2026-08-04T00:00:00Z",
  playbackUrl: "/v1/audio/artifacts/aud_abcdefghijklmnopqrstuvwxyzABCDEF/content",
};

describe("AudioWorkspace", () => {
  const destroy = vi.fn();
  const playPause = vi.fn();
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    destroy.mockReset();
    playPause.mockReset();
    vi.mocked(createWaveform).mockReset();
    window.sessionStorage.clear();
    saveBearerToken("tab-token");
    vi.stubGlobal("fetch", fetchMock);
    vi.mocked(createWaveform).mockReturnValue({ destroy, playPause });
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ artifact }), {
      status: 201,
      headers: { "Content-Type": "application/json" },
    }));
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("uploads audio and previews it with an authenticated WaveSurfer request", async () => {
    const user = userEvent.setup();
    render(<AudioWorkspace />);
    await user.upload(
      screen.getByLabelText("Local audio"),
      new File([new Uint8Array([1, 2, 3])], "tone.wav", { type: "audio/wav" }),
    );
    await user.click(screen.getByRole("button", { name: "Upload and preview" }));

    await waitFor(() => expect(createWaveform).toHaveBeenCalled());
    const [path, init] = fetchMock.mock.calls[0];
    expect(path).toBe("/v1/audio/artifacts");
    expect(init.headers.get("Authorization")).toBe("Bearer tab-token");
    expect(createWaveform).toHaveBeenCalledWith(
      expect.any(HTMLElement), artifact.playbackUrl, "tab-token",
    );
    expect(artifact.playbackUrl).not.toContain("tab-token");

    await user.click(screen.getByRole("button", { name: "Play or pause" }));
    expect(playPause).toHaveBeenCalledOnce();
  });

  it("acquires YouTube audio and explicitly deletes the temporary artifact", async () => {
    const user = userEvent.setup();
    render(<AudioWorkspace />);
    await user.type(screen.getByLabelText("YouTube URL or video ID"), "dQw4w9WgXcQ");
    await user.click(screen.getByRole("button", { name: "Acquire and preview" }));
    await waitFor(() => expect(screen.getByText("tone.wav")).toBeVisible());

    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ deleted: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }));
    await user.click(screen.getByRole("button", { name: "Delete temporary audio" }));

    const acquireInit = fetchMock.mock.calls[0][1];
    expect(JSON.parse(acquireInit.body)).toEqual({ youtubeUrlOrId: "dQw4w9WgXcQ" });
    expect(fetchMock.mock.calls[1][0]).toBe(`/v1/audio/artifacts/${artifact.audioRef}`);
    expect(fetchMock.mock.calls[1][1].method).toBe("DELETE");
    expect(screen.getByRole("status")).toHaveTextContent("Temporary audio deleted.");
    expect(destroy).toHaveBeenCalled();
  });
});
