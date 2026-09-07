import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { YouTubeSession } from "./YouTubeSession";
import { saveBearerToken } from "./api";

function jsonResponse(status: number, body: unknown) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "Error",
    json: async () => body,
  };
}

const cookiesTxt = [
  "# Netscape HTTP Cookie File",
  ".youtube.com\tTRUE\t/\tTRUE\t1790000000\tSID\tvalue",
].join("\n");

describe("YouTubeSession", () => {
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

  it("reports a stored session without ever showing the cookies back", async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, {
      configured: true, updatedAt: "2026-09-01T10:00:00Z", source: "studio", lineCount: 14,
    }));
    render(<YouTubeSession token="tab-token" />);

    expect(await screen.findByText("Connected")).toBeVisible();
    expect(screen.getByText("14")).toBeVisible();
    expect(screen.getByLabelText("cookies.txt")).toHaveValue("");
  });

  it("stores a pasted file, clears the box and re-reads the status", async () => {
    fetchMock.mockImplementation(async (_path: string, init?: RequestInit) => {
      if ((init?.method ?? "GET") === "POST") return jsonResponse(200, { status: "stored" });
      return jsonResponse(200, { configured: true, updatedAt: "2026-09-01T10:00:00Z", source: "studio", lineCount: 1 });
    });
    const onReconnected = vi.fn();
    const user = userEvent.setup();
    render(<YouTubeSession token="tab-token" onReconnected={onReconnected} />);

    await screen.findByLabelText("cookies.txt");
    await user.click(screen.getByLabelText("cookies.txt"));
    await user.paste(cookiesTxt);
    expect(screen.getByText("1 cookie line detected.")).toBeVisible();

    await user.click(screen.getByRole("button", { name: "Store session" }));

    await waitFor(() => expect(onReconnected).toHaveBeenCalled());
    const post = fetchMock.mock.calls.find(([, init]) => init?.method === "POST")!;
    expect(post[0]).toBe("/v1/config/youtube-cookies");
    expect(JSON.parse(post[1].body as string)).toEqual({ cookiesTxt, source: "studio" });
    expect(screen.getByLabelText("cookies.txt")).toHaveValue("");
  });

  it("will not send an empty or comment-only paste", async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, { configured: false }));
    const user = userEvent.setup();
    render(<YouTubeSession token="tab-token" />);

    const box = await screen.findByLabelText("cookies.txt");
    expect(screen.getByRole("button", { name: "Store session" })).toBeDisabled();
    await user.click(box);
    await user.paste("# Netscape HTTP Cookie File");
    expect(screen.getByRole("button", { name: "Store session" })).toBeDisabled();
  });

  // A 409 here is the server refusing to hold session credentials while it is
  // itself unauthenticated — a deployment fact that needs saying, not a crash.
  it("explains a 409 rather than showing it as a failure", async () => {
    fetchMock.mockResolvedValue(jsonResponse(409, {
      detail: "refusing to manage YouTube session cookies on an unauthenticated service; set SNOOCLE_API_TOKEN",
    }));
    render(<YouTubeSession token="tab-token" />);

    expect(await screen.findByText(/will not hold session cookies/)).toBeVisible();
    expect(screen.getByText(/set SNOOCLE_API_TOKEN/)).toBeVisible();
  });

  it("does not call the server without a token", () => {
    render(<YouTubeSession token="" />);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(screen.getByText(/Paste the server's SNOOCLE_API_TOKEN/)).toBeVisible();
  });

  it("offers to clear a stored session and re-reads afterwards", async () => {
    fetchMock.mockImplementation(async (_path: string, init?: RequestInit) => {
      if ((init?.method ?? "GET") === "DELETE") return jsonResponse(200, { status: "cleared" });
      return jsonResponse(200, { configured: true, updatedAt: "2026-09-01T10:00:00Z", source: "studio", lineCount: 3 });
    });
    const user = userEvent.setup();
    render(<YouTubeSession token="tab-token" />);

    await user.click(await screen.findByRole("button", { name: "Clear stored session" }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([, init]) => init?.method === "DELETE")).toBe(true));
    expect(await screen.findByText("Stored YouTube session cleared.")).toBeVisible();
  });
});
