import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Library } from "./Library";

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

const needsIdentityBody = {
  songs: [{ songId: "unknown--mystery-track", artist: "Unknown", title: "Mystery Track", needsIdentity: true }],
};

describe("Library", () => {
  const fetchMock = vi.fn();
  const onNavigate = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    onNavigate.mockReset();
    window.sessionStorage.clear();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders title — artist rows, filters by search, and flags a needs-identity song", async () => {
    fetchMock.mockImplementation(async (path: string) => {
      if (path === "/v1/songs") return jsonResponse(200, songsBody);
      if (path === "/v1/songs/needs-identity") return jsonResponse(200, needsIdentityBody);
      throw new Error(`unexpected path ${path}`);
    });
    const user = userEvent.setup();
    render(<Library token="tab-token" onNavigate={onNavigate} />);

    expect(await screen.findByText("Karma Police — Radiohead")).toBeVisible();
    expect(screen.getByText("Mystery Track — Unknown")).toBeVisible();
    expect(screen.getByText("Needs identity")).toBeVisible();
    expect(screen.getByText("Showing 2 of 2")).toBeVisible();

    await user.type(screen.getByLabelText("Search"), "karma");
    expect(screen.getByText("Showing 1 of 2")).toBeVisible();
    expect(screen.queryByText("Mystery Track — Unknown")).not.toBeInTheDocument();

    await user.click(screen.getByText("Karma Police — Radiohead"));
    expect(onNavigate).toHaveBeenCalledWith("/studio/library/radiohead--karma-police");
  });

  it("shows the unauthenticated state and never calls fetch when the token is empty", () => {
    render(<Library token="" onNavigate={onNavigate} />);
    expect(screen.getByText(/paste the server's/i)).toBeVisible();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("surfaces the detail from a 500 and retries via the button", async () => {
    fetchMock.mockImplementation(async (path: string) => {
      if (path === "/v1/songs") return jsonResponse(500, { detail: "store unavailable" });
      return jsonResponse(200, needsIdentityBody);
    });
    render(<Library token="tab-token" onNavigate={onNavigate} />);

    expect(await screen.findByText("store unavailable")).toBeVisible();
    const callsBefore = fetchMock.mock.calls.length;

    fetchMock.mockImplementation(async (path: string) => {
      if (path === "/v1/songs") return jsonResponse(200, songsBody);
      return jsonResponse(200, needsIdentityBody);
    });
    await userEvent.setup().click(screen.getByRole("button", { name: "Retry" }));

    expect(await screen.findByText("Karma Police — Radiohead")).toBeVisible();
    expect(fetchMock.mock.calls.length).toBeGreaterThan(callsBefore);
  });
});
