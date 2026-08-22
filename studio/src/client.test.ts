import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { saveBearerToken } from "./api";
import { apiBlob, apiJson, ApiError, fetchRecentRuns, type SongRunsResponse, type SongsResponse } from "./client";

function response(status: number, body: unknown, ok = status >= 200 && status < 300) {
  return {
    ok,
    status,
    statusText: "Error",
    json: async () => body,
    text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
    blob: async () => new Blob([JSON.stringify(body)]),
  };
}

describe("client", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    window.sessionStorage.clear();
    saveBearerToken("tab-token");
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => vi.restoreAllMocks());

  it("apiJson returns the parsed body on success", async () => {
    fetchMock.mockResolvedValue(response(200, { hello: "world" }));
    await expect(apiJson("/v1/songs")).resolves.toEqual({ hello: "world" });
  });

  it("apiJson throws ApiError with the body's detail on a non-OK response", async () => {
    fetchMock.mockResolvedValue(response(401, { detail: "missing or invalid bearer token" }));
    const error = await apiJson("/v1/songs").catch((err: unknown) => err);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).detail).toBe("missing or invalid bearer token");
    expect((error as ApiError).status).toBe(401);
    expect((error as ApiError).unauthorized).toBe(true);
  });

  it("apiJson falls back to status text when the body isn't JSON", async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      json: async () => {
        throw new Error("not json");
      },
    });
    const error = await apiJson("/v1/songs").catch((err: unknown) => err);
    expect((error as ApiError).detail).toBe("Internal Server Error");
    expect((error as ApiError).unauthorized).toBe(false);
  });

  it("apiBlob returns a Blob on success and throws ApiError otherwise", async () => {
    fetchMock.mockResolvedValue(response(200, { a: 1 }));
    await expect(apiBlob("/v1/songs/x/export?format=json")).resolves.toBeInstanceOf(Blob);

    fetchMock.mockResolvedValue(response(404, { detail: "song not found" }));
    const error = await apiBlob("/v1/songs/x/export?format=json").catch((err: unknown) => err);
    expect((error as ApiError).detail).toBe("song not found");
  });

  it("fetchRecentRuns merges every song's runs, sorts newest first, caps at the limit, and skips a failing song", async () => {
    const songs: SongsResponse = {
      songs: ["a", "b", "c"],
      items: [
        { id: "a", title: "A", artist: "Artist", latestVersion: "v1", updatedAt: "2026-01-01", youtubeVideoId: null, hasTiming: false },
        { id: "b", title: "B", artist: "Artist", latestVersion: "v1", updatedAt: "2026-01-01", youtubeVideoId: null, hasTiming: false },
        { id: "c", title: "C", artist: "Artist", latestVersion: "v1", updatedAt: "2026-01-01", youtubeVideoId: null, hasTiming: false },
      ],
    };
    const runFixture = (runId: string, songId: string, startedAt: string) => ({
      runId, songId, provider: "mock", model: "mock", depth: "standard", status: "ok",
      startedAt, finishedAt: null, error: null, stepCount: 1, costUSD: 0, effortLevel: "standard", batchId: null,
    });
    const runsA: SongRunsResponse = { songId: "a", runs: [runFixture("r1", "a", "2026-01-01T10:00:00Z")] };
    const runsC: SongRunsResponse = { songId: "c", runs: [runFixture("r2", "c", "2026-01-02T10:00:00Z")] };

    fetchMock.mockImplementation(async (path: string) => {
      if (path === "/v1/songs") return response(200, songs);
      if (path === "/v1/songs/a/runs") return response(200, runsA);
      if (path === "/v1/songs/b/runs") return response(500, { detail: "boom" });
      if (path === "/v1/songs/c/runs") return response(200, runsC);
      throw new Error(`unexpected path ${path}`);
    });

    const runs = await fetchRecentRuns(1);
    expect(runs).toHaveLength(1);
    expect(runs[0].runId).toBe("r2");
  });
});
