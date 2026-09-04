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
  /**
   * Provenance. `songId` is the workbench song that was selected when this
   * recording entered the slot; the YouTube fields are what the server said it
   * actually fetched. Without these a recording is an anonymous reference and
   * song A's audio can be fed to a step for song B in silence.
   */
  songId?: string;
  youtubeVideoId?: string;
  videoTitle?: string;
}

export interface WorkbenchMir {
  json: string;
  /** The workbench song selected when this MIR was captured, and the recording it was computed from. */
  songId?: string;
  audioRef?: string;
}

export interface Workbench {
  song?: WorkbenchSong;
  audio?: WorkbenchAudio;
  mir?: WorkbenchMir;
}

export const EMPTY_WORKBENCH: Workbench = {};

const WORKBENCH_KEY = "snoocle.studio.workbench";

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" && value !== "" ? value : undefined;
}

/** Assigns only the keys that actually have a value, so a parsed slot deep-equals the object that was saved. */
function withDefined<T extends object>(base: T, extras: Record<string, unknown>): T {
  for (const [key, value] of Object.entries(extras)) {
    if (value !== undefined) (base as Record<string, unknown>)[key] = value;
  }
  return base;
}

function parseSong(value: unknown): WorkbenchSong | undefined {
  if (!isRecord(value)) return undefined;
  if (typeof value.id !== "string" || typeof value.title !== "string" || typeof value.artist !== "string") return undefined;
  return withDefined<WorkbenchSong>(
    { id: value.id, title: value.title, artist: value.artist },
    { version: optionalString(value.version) },
  );
}

function parseAudio(value: unknown): WorkbenchAudio | undefined {
  if (!isRecord(value)) return undefined;
  if (typeof value.audioRef !== "string" || typeof value.filename !== "string") return undefined;
  return withDefined<WorkbenchAudio>({ audioRef: value.audioRef, filename: value.filename }, {
    durationSeconds: typeof value.durationSeconds === "number" ? value.durationSeconds : undefined,
    songId: optionalString(value.songId),
    youtubeVideoId: optionalString(value.youtubeVideoId),
    videoTitle: optionalString(value.videoTitle),
  });
}

function parseMir(value: unknown): WorkbenchMir | undefined {
  if (!isRecord(value) || typeof value.json !== "string" || value.json === "") return undefined;
  return withDefined<WorkbenchMir>({ json: value.json }, {
    songId: optionalString(value.songId),
    audioRef: optionalString(value.audioRef),
  });
}

/**
 * Tab-scoped like the bearer token: the workbench selection should not leak
 * across tabs or outlive the session.
 *
 * Parsing is slot-by-slot on purpose. A slot written by an older build (or by
 * hand) is dropped on its own rather than discarding the whole workbench, so a
 * shape change never silently clears the song you were working on.
 */
export function loadWorkbench(): Workbench {
  try {
    const raw = window.sessionStorage.getItem(WORKBENCH_KEY);
    if (!raw) return EMPTY_WORKBENCH;
    const parsed = JSON.parse(raw) as unknown;
    if (!isRecord(parsed)) return EMPTY_WORKBENCH;
    const bench: Workbench = {};
    const song = parseSong(parsed.song);
    if (song) bench.song = song;
    const audio = parseAudio(parsed.audio);
    if (audio) bench.audio = audio;
    const mir = parseMir(parsed.mir);
    if (mir) bench.mir = mir;
    return bench;
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
 * Property names that mean "here is the recording to work on". When a tool
 * declares one of these, the recording — not the workbench — decides which
 * song the call is about.
 */
const RECORDING_SOURCE_FIELDS = ["youtube_url_or_id", "audio_ref", "audio_path", "input_ref", "input_base64"];

/**
 * Slots whose provenance disagrees with the selected song, in words fit for
 * the screen. An empty list means every filled slot belongs to this song (or
 * predates provenance and cannot be judged, which is treated as fine).
 */
export function benchMismatches(bench: Workbench): string[] {
  const songId = bench.song?.id;
  if (!songId) return [];
  const problems: string[] = [];
  if (bench.audio?.songId && bench.audio.songId !== songId) {
    problems.push(`The audio (${bench.audio.videoTitle ?? bench.audio.filename}) was acquired for a different song.`);
  }
  if (bench.mir?.songId && bench.mir.songId !== songId) {
    problems.push("The captured MIR was computed for a different song.");
  }
  if (bench.mir?.audioRef && bench.audio && bench.mir.audioRef !== bench.audio.audioRef) {
    problems.push("The captured MIR was computed from a different recording than the one in the audio slot.");
  }
  return problems;
}

/**
 * Seeds a tool's form from the workbench, but only for property names the
 * tool's own JSON Schema declares — an unrelated tool never receives fields
 * it never asked for.
 *
 * Only *references* are seeded freely: song_id, song_version, audio_ref and
 * the captured MIR each name one exact object.
 *
 * title/artist are different. They are descriptive strings, and for the
 * acquisition tools (acquire_audio, analyze_audio, analyze_and_store_song)
 * they are merely search terms that a supplied youtube_url_or_id overrides
 * server-side. Seeding them there produced a form that claimed one song while
 * fetching another, so they are seeded only for tools that take no recording
 * at all — discover_song and friends, where title/artist really are the
 * subject.
 *
 * song_json/prior_song_json and every *_path property are deliberately never
 * seeded, even though a tool may declare them: song_json is mutually exclusive
 * with song_id server-side (invalid_song_source), and *_path fields are the
 * server-filesystem inputs Studio already blocks from browser invocation. Both
 * stay copy-paste-only.
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
  if (declares("mir_json") && bench.mir) seed.mir_json = bench.mir.json;
  if (declares("cached_mir_json") && bench.mir) seed.cached_mir_json = bench.mir.json;

  const takesRecording = RECORDING_SOURCE_FIELDS.some(declares);
  if (!takesRecording) {
    if (declares("title") && bench.song?.title) seed.title = bench.song.title;
    if (declares("artist") && bench.song?.artist) seed.artist = bench.song.artist;
  }

  return seed;
}

/** True when the tool takes a recording, so its subject is the recording rather than the workbench song. */
export function derivesIdentityFromRecording(schema: unknown): boolean {
  const properties = isRecord(schema) && isRecord(schema.properties) ? schema.properties : undefined;
  if (!properties) return false;
  return RECORDING_SOURCE_FIELDS.some((name) => Object.prototype.hasOwnProperty.call(properties, name));
}
