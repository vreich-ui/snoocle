import type { ChordPlacement, Song } from "./client";
import { isRecord } from "./tooling";

export interface SongStep {
  /** Stable, e.g. "score-confidence". Never shown to a user; used as a React key and a log correlator. */
  id: string;
  label: string;
  description: string;
  tool: string;
  /** Does a successful run yield a candidate Song, or just a report to read? */
  produces: "song" | "report";
  /** Requires an audio artifact in the workbench (align_song_deterministically's audio_ref is the only one of these tools that takes one). */
  needsAudio?: boolean;
}

export const SONG_STEPS: SongStep[] = [
  {
    id: "validate",
    label: "Validate against the schema",
    description: "Checks the song document against the schema and flags anything structurally invalid before you trust the data.",
    tool: "validate_song_json",
    produces: "report",
  },
  {
    id: "score-confidence",
    label: "Score confidence",
    description: "Scores how confidently each line and chord placement matches the evidence, and lists the lowest-confidence spots for review.",
    tool: "score_song_confidence",
    produces: "report",
  },
  {
    id: "evaluate-quality",
    label: "Grade quality",
    description: "Grades the overall transcription quality and attributes any faults to the pipeline stage that likely caused them.",
    tool: "evaluate_song_quality",
    produces: "report",
  },
  {
    id: "validate-theory",
    label: "Check the chords against the key",
    description: "Flags chords that don't fit the song's key, which usually means a wrong transcription or a wrong key.",
    tool: "validate_song_theory",
    produces: "report",
  },
  {
    id: "evidence",
    label: "Build the evidence manifest",
    description: "Assembles the sources and signals behind the transcription into one manifest, useful for auditing where a decision came from.",
    tool: "build_song_evidence_manifest",
    produces: "report",
  },
  {
    id: "snap-mir",
    label: "Snap timing to the audio",
    description: "Aligns each line and chord to the audio's beat and onset evidence, replacing rough or missing timestamps.",
    tool: "snap_song_to_mir",
    produces: "song",
  },
  {
    id: "retime-sections",
    label: "Re-time the sections",
    description: "Recomputes section boundaries (verse, chorus, bridge, ...) from the current line timing so section markers track the song.",
    tool: "retime_song_sections",
    produces: "song",
  },
  {
    id: "guard-collapse",
    label: "Repair collapsed timing",
    description: "Detects and repairs stretches where timing has collapsed onto a single instant, a common failure after a bad snap.",
    tool: "guard_song_timing_collapse",
    produces: "song",
  },
  {
    id: "align",
    label: "Align deterministically (whole pass)",
    description: "Runs the full snap, section-retime, collapse-guard, and confidence/quality pass in one step, without saving the result.",
    tool: "align_song_deterministically",
    produces: "song",
    needsAudio: true,
  },
];

/**
 * PR #80 made every one of these tools accept a stored song id in place of
 * pasted-in JSON, so this always references the song by id rather than
 * inlining it. song_json is mutually exclusive with song_id server-side
 * (invalid_song_source), and every *_path field is a server-local-file input
 * Studio never sends from the browser — so neither is ever produced here.
 */
export function argsForStep(
  step: SongStep,
  songId: string,
  version: string | undefined,
  audioRef: string | undefined,
): Record<string, unknown> {
  const args: Record<string, unknown> = { song_id: songId };
  if (version) args.song_version = version;
  if (step.needsAudio && audioRef) args.audio_ref = audioRef;
  return args;
}

function looksLikeSong(value: unknown): value is Song {
  return isRecord(value) && isRecord(value.metadata) && Array.isArray(value.lines);
}

/** Finds the candidate Song in a tool's `result` field — either nested at `.song`, or the result itself when it is shaped like a Song. */
export function extractCandidateSong(result: unknown): Song | undefined {
  if (isRecord(result) && looksLikeSong(result.song)) return result.song;
  if (looksLikeSong(result)) return result;
  return undefined;
}

function countPlacementChanges(before: ChordPlacement[], after: ChordPlacement[]): number {
  const byIndex = (list: ChordPlacement[]) => new Map(list.map((placement) => [placement.charIndex, placement.chord]));
  const beforeByIndex = byIndex(before);
  const afterByIndex = byIndex(after);
  const charIndices = new Set([...beforeByIndex.keys(), ...afterByIndex.keys()]);
  let changed = 0;
  for (const charIndex of charIndices) {
    if (beforeByIndex.get(charIndex) !== afterByIndex.get(charIndex)) changed += 1;
  }
  return changed;
}

/**
 * A compact, human-readable diff between two versions of a Song. Deliberately
 * limited to counts and before→after scalars — not a full structural diff —
 * so it stays readable as a candidate-review summary rather than a patch.
 */
export function summariseSongChange(before: Song, after: Song): string[] {
  const facts: string[] = [];

  const beforeLines = new Map(before.lines.map((line) => [line.lineIndex, line]));
  const afterLines = new Map(after.lines.map((line) => [line.lineIndex, line]));
  const sharedIndices = [...afterLines.keys()].filter((index) => beforeLines.has(index));

  let timingChanged = 0;
  let placementsChanged = 0;
  for (const index of sharedIndices) {
    const beforeLine = beforeLines.get(index)!;
    const afterLine = afterLines.get(index)!;
    if ((beforeLine.timeSeconds ?? null) !== (afterLine.timeSeconds ?? null)) timingChanged += 1;
    placementsChanged += countPlacementChanges(beforeLine.chordPlacements, afterLine.chordPlacements);
  }

  if (sharedIndices.length) facts.push(`${timingChanged} of ${sharedIndices.length} lines changed timing`);
  if (placementsChanged > 0) facts.push(`${placementsChanged} chord placement${placementsChanged === 1 ? "" : "s"} changed`);
  if (before.lines.length !== after.lines.length) facts.push(`lines: ${before.lines.length} → ${after.lines.length}`);
  if (before.sections.length !== after.sections.length) facts.push(`sections: ${before.sections.length} → ${after.sections.length}`);
  facts.push(
    before.metadata.bpm === after.metadata.bpm
      ? "bpm unchanged"
      : `bpm: ${before.metadata.bpm ?? "unknown"} → ${after.metadata.bpm ?? "unknown"}`,
  );

  return facts;
}
