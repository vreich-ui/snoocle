import { apiFetch } from "./api";

/** A REST call that reached the server but was answered with a non-OK status. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  readonly unauthorized: boolean;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.unauthorized = status === 401;
  }
}

async function errorFromResponse(response: Response): Promise<ApiError> {
  let detail = response.statusText || `Request failed (${response.status})`;
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && typeof (body as { detail?: unknown }).detail === "string") {
      detail = (body as { detail: string }).detail;
    }
  } catch {
    // Body wasn't JSON (or was empty) — fall back to status text.
  }
  return new ApiError(response.status, detail);
}

export async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await apiFetch(path, init);
  if (!response.ok) throw await errorFromResponse(response);
  return response.json() as Promise<T>;
}

export async function apiBlob(path: string): Promise<Blob> {
  const response = await apiFetch(path);
  if (!response.ok) throw await errorFromResponse(response);
  return response.blob();
}

// --- /v1/songs ---------------------------------------------------------

export interface SongSummary {
  id: string;
  title: string;
  artist: string;
  latestVersion: string;
  updatedAt: string;
  youtubeVideoId: string | null;
  hasTiming: boolean;
}

export interface SongsResponse {
  songs: string[];
  items: SongSummary[];
}

export interface NeedsIdentitySong {
  songId: string;
  artist: string;
  title: string;
  needsIdentity: true;
}

export interface NeedsIdentityResponse {
  songs: NeedsIdentitySong[];
}

// --- /v1/songs/{id} (the Song document itself) --------------------------

export interface SongMetadata {
  title: string;
  artist: string;
  album?: string | null;
  year?: number | null;
  key?: string | null;
  bpm?: number | null;
  timeSignature?: string | null;
}

export interface DisplayPreferences {
  capo: number;
  tuning: string;
}

export interface SongAudio {
  youtubeVideoId?: string | null;
  durationSeconds?: number | null;
  analyzedVideoId?: string | null;
}

export interface ChordPlacement {
  charIndex: number;
  chord: string;
}

export interface SongLine {
  lineIndex: number;
  lyrics: string;
  chordPlacements: ChordPlacement[];
}

export interface SongSection {
  sectionIndex: number;
  name: string;
  kind: string;
  startLineIndex: number;
  endLineIndex: number;
}

export interface Song {
  schemaVersion: number;
  id: string;
  metadata: SongMetadata;
  displayPreferences: DisplayPreferences;
  audio: SongAudio;
  sections: SongSection[];
  lines: SongLine[];
  testOnly: boolean;
}

// --- /v1/songs/{id}/versions, /diff --------------------------------------

export interface SongVersion {
  version: string;
  timestamp: string;
  message: string;
}

export interface VersionsResponse {
  songId: string;
  versions: SongVersion[];
}

// --- /v1/songs/{id}/identity ----------------------------------------------

export interface IdentityRenameResponse {
  oldSongId: string;
  songId: string;
  versionMap: Record<string, string>;
  migratedRunIds: string[];
}

/** The 422 body set_song_identity answers with when artist/title still leave the id unresolved. */
export interface IdentityUnresolvedBody {
  detail: string;
  errorCode: "identity_unresolved";
  missing: string[];
  evidenceTried: string[];
  needsIdentity: true;
  reason?: string;
}

/** The 409 body when the rename target collides with an existing song. */
export interface IdentityRenameFailedBody {
  detail: string;
  errorCode: "identity_rename_failed";
}

// --- /v1/songs/{id}/gold, /notes -------------------------------------------

export interface GoldResponse {
  songId: string;
  goldVersion: string | null;
}

export interface NotesComponent {
  notes: string;
  updatedAt: string | null;
}

export interface NotesResponse {
  songId: string;
  notes: string;
  updatedAt: string | null;
  preference: NotesComponent | null;
  correction: NotesComponent | null;
}

// --- runs -------------------------------------------------------------

/**
 * Everything but the identifiers is optional on purpose: these are historical
 * records, and a run written before a field existed simply has no such key.
 * costUSD/effortLevel/batchId are absent on every run older than they are.
 */
export interface RunSummary {
  runId: string;
  songId: string;
  status: string;
  provider?: string;
  model?: string;
  depth?: string;
  startedAt?: string;
  finishedAt?: string | null;
  error?: string | null;
  stepCount?: number;
  costUSD?: number;
  effortLevel?: string;
  batchId?: string | null;
}

export interface SongRunsResponse {
  songId: string;
  runs: RunSummary[];
}

export interface RunStep {
  index: number;
  kind: string;
  label: string;
  summary: string;
  detail: Record<string, unknown>;
  timestamp: string;
  durationSeconds: number | null;
}

export interface RunDetail extends RunSummary {
  steps: RunStep[];
}

// --- queue --------------------------------------------------------------

export interface QueueJob {
  id: string;
  label: string;
  kind: string;
  status: string;
  queuedAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  worker: string | null;
  attempts: number;
  error: string | null;
  songId: string | null;
  runId: string | null;
}

export interface QueueResponse {
  jobs: QueueJob[];
  counts: Record<string, number>;
  lastHeartbeatAt: string | null;
  lastWorker: string | null;
  workerSeenRecently: boolean;
  leaseSeconds: number;
  maxAttempts: number;
  maxPerSubmit: number;
}

/**
 * There is no server endpoint that lists runs across every song, so this
 * aggregates client-side: list the library, then fetch each song's runs in
 * parallel. A song whose runs request fails is skipped rather than failing
 * the whole list — one bad song should not blank the Runs page.
 */
export async function fetchRecentRuns(limit: number): Promise<RunSummary[]> {
  const { items } = await apiJson<SongsResponse>("/v1/songs");
  const settled = await Promise.allSettled(
    items.map((item) => apiJson<SongRunsResponse>(`/v1/songs/${encodeURIComponent(item.id)}/runs`)),
  );
  const runs = settled.flatMap((result) => (result.status === "fulfilled" ? result.value.runs : []));
  return runs.sort((a, b) => (b.startedAt ?? "").localeCompare(a.startedAt ?? "")).slice(0, limit);
}
