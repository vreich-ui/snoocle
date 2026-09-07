import { describe, expect, it } from "vitest";
import { ApiError } from "./client";
import { countCookieLines, isYouTubeAuthFailure, YOUTUBE_AUTH_CODE } from "./youtube";

describe("isYouTubeAuthFailure", () => {
  it("trusts the server's classification over any text matching", () => {
    expect(isYouTubeAuthFailure(new ApiError(502, "yt-dlp failed", YOUTUBE_AUTH_CODE))).toBe(true);
    expect(isYouTubeAuthFailure(new ApiError(502, "yt-dlp failed", "identity_unresolved"))).toBe(false);
  });

  // The MCP path cannot carry a code: acquire_audio invoked from Tool Studio
  // arrives as the raw yt-dlp text, which is exactly the failure the operator
  // hit in production.
  it("recognises the raw yt-dlp text an MCP tool error arrives as", () => {
    const raw = "Error executing tool acquire_audio: yt-dlp failed for bO28lB1uwp4: ERROR: "
      + "[youtube] bO28lB1uwp4: Sign in to confirm you're not a bot. Use --cookies-from-browser "
      + "or --cookies for the authentication.";
    expect(isYouTubeAuthFailure(new Error(raw))).toBe(true);
    expect(isYouTubeAuthFailure(raw)).toBe(true);
  });

  it("recognises the other session markers the server matches", () => {
    for (const text of [
      "ERROR: Sign in to confirm your age",
      "The provided YouTube account cookies are no longer valid",
      "ERROR: Please sign in",
      "login required",
    ]) {
      expect(isYouTubeAuthFailure(new Error(text))).toBe(true);
    }
  });

  it("does not claim unrelated failures", () => {
    expect(isYouTubeAuthFailure(new Error("Video unavailable"))).toBe(false);
    expect(isYouTubeAuthFailure(new Error("network timeout"))).toBe(false);
    expect(isYouTubeAuthFailure(undefined)).toBe(false);
    expect(isYouTubeAuthFailure("")).toBe(false);
  });
});

describe("countCookieLines", () => {
  it("counts only real entries, matching the server's own count", () => {
    const file = [
      "# Netscape HTTP Cookie File",
      "# This is a generated file!  Do not edit.",
      "",
      ".youtube.com\tTRUE\t/\tTRUE\t1790000000\tSID\tvalue",
      "   ",
      ".youtube.com\tTRUE\t/\tTRUE\t1790000000\tHSID\tvalue",
    ].join("\n");
    expect(countCookieLines(file)).toBe(2);
    expect(countCookieLines("")).toBe(0);
    expect(countCookieLines("# only a comment")).toBe(0);
  });
});
