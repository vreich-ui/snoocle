import { isRecord } from "./tooling";

export interface WorkbenchSong {
  id: string;
  title: string;
  artist: string;
  version?: string;
  /** The recording this song was built from, when the store knows one. It is what makes the song's audio reachable without retyping a URL. */
  youtubeVideoId?: string;
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
    { version: optionalString(value.version), youtubeVideoId: optionalString(value.youtubeVideoId) },
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
 * The one property name whose presence means title/artist are merely
 * acquisition *search terms*: a tool that declares `youtube_url_or_id` will
 * prefer the URL and ignore the names. Exactly three server tools do —
 * acquire_audio, analyze_audio, analyze_and_store_song.
 *
 * This is deliberately narrower than "takes a recording". reconcile_song and
 * build_song_baseline both accept an audio_ref while *requiring* title and
 * artist as the song's real identity; withholding those left required fields
 * blank on the tools that most need them.
 */
const ACQUISITION_SEARCH_FIELD = "youtube_url_or_id";

/** Everything that names a recording, for provenance and conflict checks. */
const RECORDING_ID_FIELDS = ["youtube_url_or_id", "youtube_video_id"];

/**
 * The eleven-character YouTube video id inside a URL, or the id itself.
 * Returns undefined for anything that is not recognisably one, so a conflict
 * check never fires on a string it does not understand.
 */
export function parseVideoId(urlOrId: string): string | undefined {
  const trimmed = urlOrId.trim();
  if (!trimmed) return undefined;
  if (/^[\w-]{11}$/.test(trimmed)) return trimmed;
  const match = trimmed.match(/(?:v=|\/shorts\/|\/embed\/|youtu\.be\/)([\w-]{11})/);
  return match?.[1];
}

/** The recording the workbench knows about: what is loaded, else what the song was built from. */
export function knownVideoId(bench: Workbench): string | undefined {
  return bench.audio?.youtubeVideoId ?? bench.song?.youtubeVideoId;
}

/**
 * A best-effort "Artist - Title" split of a YouTube video title, used only
 * when no stored song is selected — a starting point for discover_song or
 * reconcile_song that the operator can correct, never a claim of verified
 * identity. Only the first separator counts, so "A - B - C" reads as
 * artist "A", title "B - C".
 */
export function splitVideoTitle(videoTitle: string): { artist: string; title: string } | undefined {
  const match = videoTitle.match(/^(.{1,80}?)\s+[-–—|]\s+(.+)$/);
  if (!match) return undefined;
  const artist = match[1].trim();
  const title = match[2].trim();
  return artist && title ? { artist, title } : undefined;
}

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
 * Seeds a tool's form from the workbench, for every property name the tool's
 * own JSON Schema declares — an unrelated tool never receives fields it never
 * asked for.
 *
 * The aim is that selecting a song leaves nothing to retype: the references
 * (song_id, song_version, audio_ref, the captured MIR), the recording the
 * song was built from (youtube_url_or_id, youtube_video_id) and its identity
 * (title, artist) are all filled in.
 *
 * The one case that stays empty is identity for an acquisition tool whose
 * recording we cannot supply: with no URL to agree with, a seeded title and
 * artist are a claim about whatever the operator pastes next. Whenever a
 * recording *is* seeded the names go in beside it, and `identityConflict`
 * watches for the operator replacing one without the other.
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

  const videoId = knownVideoId(bench);
  if (videoId) {
    for (const name of RECORDING_ID_FIELDS) if (declares(name)) seed[name] = videoId;
  }

  // With a stored song, its identity is the answer — except on an acquisition
  // tool we cannot also hand a recording, where the names would be a claim
  // about whatever URL the operator pastes next. Without a stored song, an
  // acquired video title is the only identity there is, and a split of it is
  // offered as a starting point since nothing can contradict it.
  const identity = bench.song
    ? { title: bench.song.title, artist: bench.song.artist }
    : bench.audio?.videoTitle
      ? splitVideoTitle(bench.audio.videoTitle)
      : undefined;
  const withheld = Boolean(bench.song) && declares(ACQUISITION_SEARCH_FIELD) && !videoId;
  if (identity && !withheld) {
    if (declares("title")) seed.title = identity.title;
    if (declares("artist")) seed.artist = identity.artist;
  }

  return seed;
}

/** True when the tool treats title/artist as search terms a URL overrides. */
export function derivesIdentityFromRecording(schema: unknown): boolean {
  const properties = isRecord(schema) && isRecord(schema.properties) ? schema.properties : undefined;
  if (!properties) return false;
  return Object.prototype.hasOwnProperty.call(properties, ACQUISITION_SEARCH_FIELD);
}

/**
 * The live check that replaces withholding identity: once a form is filled in,
 * watch for the operator pointing an acquisition tool at one recording while
 * the title and artist beside it still name another. The server resolves the
 * URL and ignores the names, so this is the moment the two can silently part
 * company — the failure that made a Nirvana cover arrive under Amy
 * Winehouse's name.
 */
export function identityConflict(
  schema: unknown,
  values: Record<string, unknown>,
  bench: Workbench,
): string | undefined {
  if (!derivesIdentityFromRecording(schema)) return undefined;
  const song = bench.song;
  if (!song) return undefined;
  const typed = typeof values[ACQUISITION_SEARCH_FIELD] === "string"
    ? values[ACQUISITION_SEARCH_FIELD] as string
    : "";
  const typedId = parseVideoId(typed);
  if (!typedId) return undefined;
  const known = knownVideoId(bench);
  if (known && typedId === known) return undefined;
  const claimsSong = values.title === song.title || values.artist === song.artist;
  if (!claimsSong) return undefined;
  return `This will fetch ${typedId}, which is not the recording on file for ${song.title} — ${song.artist}. `
    + "The recording wins: the title and artist below are ignored, and the result will be whatever that video is.";
}
