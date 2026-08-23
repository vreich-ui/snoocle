import { isRecord } from "./tooling";

export interface WorkbenchSong {
  id: string;
  title: string;
  artist: string;
  version?: string;
}

export interface WorkbenchAudio {
  audioRef: string;
  filename: string;
  durationSeconds?: number;
}

export interface Workbench {
  song?: WorkbenchSong;
  audio?: WorkbenchAudio;
  mirJson?: string;
}

export const EMPTY_WORKBENCH: Workbench = {};

const WORKBENCH_KEY = "snoocle.studio.workbench";

function isWorkbenchSong(value: unknown): value is WorkbenchSong {
  return isRecord(value) &&
    typeof value.id === "string" &&
    typeof value.title === "string" &&
    typeof value.artist === "string" &&
    (value.version === undefined || typeof value.version === "string");
}

function isWorkbenchAudio(value: unknown): value is WorkbenchAudio {
  return isRecord(value) &&
    typeof value.audioRef === "string" &&
    typeof value.filename === "string" &&
    (value.durationSeconds === undefined || typeof value.durationSeconds === "number");
}

function isWorkbench(value: unknown): value is Workbench {
  if (!isRecord(value)) return false;
  if (value.song !== undefined && !isWorkbenchSong(value.song)) return false;
  if (value.audio !== undefined && !isWorkbenchAudio(value.audio)) return false;
  if (value.mirJson !== undefined && typeof value.mirJson !== "string") return false;
  return true;
}

/** Tab-scoped like the bearer token: the workbench selection should not leak across tabs or outlive the session. */
export function loadWorkbench(): Workbench {
  try {
    const raw = window.sessionStorage.getItem(WORKBENCH_KEY);
    if (!raw) return EMPTY_WORKBENCH;
    const parsed = JSON.parse(raw) as unknown;
    return isWorkbench(parsed) ? parsed : EMPTY_WORKBENCH;
  } catch {
    return EMPTY_WORKBENCH;
  }
}

export function saveWorkbench(next: Workbench): void {
  try {
    window.sessionStorage.setItem(WORKBENCH_KEY, JSON.stringify(next));
  } catch {
    // The workbench is a convenience; a full or blocked session storage must not break Studio.
  }
}

/**
 * Seeds a tool's form from the workbench, but only for property names the
 * tool's own JSON Schema declares — an unrelated tool never receives fields
 * it never asked for.
 *
 * song_json/prior_song_json and every *_path property are deliberately never
 * seeded here, even though a tool may declare them: song_json is mutually
 * exclusive with song_id server-side (invalid_song_source), and *_path fields
 * are the server-filesystem inputs Studio already blocks from browser
 * invocation. Both stay copy-paste-only.
 */
export function seedFromWorkbench(schema: unknown, bench: Workbench): Record<string, unknown> {
  const properties = isRecord(schema) && isRecord(schema.properties) ? schema.properties : undefined;
  if (!properties) return {};
  const declares = (name: string) => Object.prototype.hasOwnProperty.call(properties, name);
  const seed: Record<string, unknown> = {};

  if (declares("song_id") && bench.song?.id) {
    seed.song_id = bench.song.id;
    if (declares("song_version") && bench.song.version) seed.song_version = bench.song.version;
  }
  if (declares("audio_ref") && bench.audio?.audioRef) seed.audio_ref = bench.audio.audioRef;
  if (declares("input_ref") && bench.audio?.audioRef) seed.input_ref = bench.audio.audioRef;
  if (declares("mir_json") && bench.mirJson) seed.mir_json = bench.mirJson;
  if (declares("cached_mir_json") && bench.mirJson) seed.cached_mir_json = bench.mirJson;
  if (declares("title") && bench.song?.title) seed.title = bench.song.title;
  if (declares("artist") && bench.song?.artist) seed.artist = bench.song.artist;

  return seed;
}
