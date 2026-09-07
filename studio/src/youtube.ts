import { apiJson, ApiError } from "./client";

export interface YouTubeSessionStatus {
  configured: boolean;
  updatedAt?: string;
  source?: string;
  lineCount?: number;
}

export const YOUTUBE_AUTH_CODE = "youtube_auth_required";

/**
 * The same yt-dlp fragments the server matches in
 * `snoocle_server/audio/acquire.py`, for the one path that cannot carry a
 * classified code: an MCP tool result is a text error, so `acquire_audio`
 * invoked from Tool Studio arrives as a raw string. REST callers should read
 * `ApiError.errorCode` and never need this.
 */
const AUTH_MARKERS = [
  "sign in to confirm you're not a bot",
  "sign in to confirm your age",
  "confirm you're not a robot",
  "use --cookies",
  "cookies are no longer valid",
  "please sign in",
  "login required",
  "not a bot",
];

/** True when a failure means the YouTube session is the problem, whatever shape the failure arrived in. */
export function isYouTubeAuthFailure(error: unknown): boolean {
  if (error instanceof ApiError && error.errorCode === YOUTUBE_AUTH_CODE) return true;
  const text = (error instanceof Error ? error.message : typeof error === "string" ? error : "").toLowerCase();
  if (!text) return false;
  if (text.includes(YOUTUBE_AUTH_CODE)) return true;
  return AUTH_MARKERS.some((marker) => text.includes(marker));
}

export function fetchYouTubeSession(): Promise<YouTubeSessionStatus> {
  return apiJson<YouTubeSessionStatus>("/v1/config/youtube-cookies");
}

export function storeYouTubeCookies(cookiesTxt: string, source = "studio"): Promise<unknown> {
  return apiJson("/v1/config/youtube-cookies", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cookiesTxt, source }),
  });
}

export function clearYouTubeCookies(): Promise<unknown> {
  return apiJson("/v1/config/youtube-cookies", { method: "DELETE" });
}

/**
 * Non-comment, non-blank lines — the same count the server reports back, so
 * an empty or wrong-format paste is caught before it is sent.
 */
export function countCookieLines(cookiesTxt: string): number {
  return cookiesTxt.split("\n").filter((line) => line.trim() && !line.trimStart().startsWith("#")).length;
}
