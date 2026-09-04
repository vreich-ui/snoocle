import { apiFetch } from "./api";

export type AudioArtifact = {
  audioRef: string;
  filename: string;
  contentType: string;
  durationSeconds: number;
  sizeBytes: number;
  expiresAt: string;
  playbackUrl: string;
  /** Present for acquisitions: what the server actually fetched, as opposed to what was asked for. */
  youtubeVideoId?: string;
  videoTitle?: string;
  fromCache?: boolean;
};

/**
 * The acquire response reports what was fetched as siblings of `artifact`
 * (`youtubeVideoId`, `videoTitle`, `fromCache`). Dropping them left the UI
 * with only a filename to go on, so a recording that did not match the song
 * on screen looked no different from one that did. They are folded onto the
 * artifact here and travel with it into the workbench.
 */
export async function artifactResponse(response: Response): Promise<AudioArtifact> {
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail ?? body.reason ?? `Request failed (${response.status})`);
  }
  const artifact = body.artifact as AudioArtifact;
  return {
    ...artifact,
    youtubeVideoId: typeof body.youtubeVideoId === "string" ? body.youtubeVideoId : undefined,
    videoTitle: typeof body.videoTitle === "string" ? body.videoTitle : undefined,
    fromCache: typeof body.fromCache === "boolean" ? body.fromCache : undefined,
  };
}

/** Fetches a recording server-side and retains it under an opaque, expiring reference. */
export async function acquireArtifact(youtubeUrlOrId: string): Promise<AudioArtifact> {
  return artifactResponse(await apiFetch("/v1/audio/artifacts/acquire", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ youtubeUrlOrId }),
  }));
}
