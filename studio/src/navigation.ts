export const studioSections = [
  "Repair",
  "Build",
  "Automatic Pipeline",
  "Tool Studio",
  "Library",
  "Runs",
  "Evaluation",
  "Configuration",
] as const;

export type StudioSection = (typeof studioSections)[number];

/** A bare /studio/ or an unknown path lands on the one section that is actually built. */
export const defaultSection: StudioSection = "Tool Studio";

export function sectionPath(section: StudioSection): string {
  return `/studio/${section.toLowerCase().replaceAll(" ", "-")}`;
}

export interface StudioRoute {
  section: StudioSection;
  detailId?: string;
}

const LIBRARY_DETAIL_PREFIX = "/studio/library/";
const RUNS_DETAIL_PREFIX = "/studio/runs/";

/** A detail path (song or run) is not itself a section path, so it is matched separately, after the exact section match fails. */
export function routeFromPath(pathname: string): StudioRoute {
  const section = studioSections.find((item) => sectionPath(item) === pathname);
  if (section) return { section };
  if (pathname.startsWith(LIBRARY_DETAIL_PREFIX) && pathname.length > LIBRARY_DETAIL_PREFIX.length) {
    return { section: "Library", detailId: decodeURIComponent(pathname.slice(LIBRARY_DETAIL_PREFIX.length)) };
  }
  if (pathname.startsWith(RUNS_DETAIL_PREFIX) && pathname.length > RUNS_DETAIL_PREFIX.length) {
    return { section: "Runs", detailId: decodeURIComponent(pathname.slice(RUNS_DETAIL_PREFIX.length)) };
  }
  return { section: defaultSection };
}

export function sectionFromPath(pathname: string): StudioSection {
  return routeFromPath(pathname).section;
}

export function songDetailPath(songId: string): string {
  return `/studio/library/${encodeURIComponent(songId)}`;
}

export function runDetailPath(runId: string): string {
  return `/studio/runs/${encodeURIComponent(runId)}`;
}

export const implementedSections = ["Tool Studio", "Library", "Runs"] as const satisfies readonly StudioSection[];

export function isImplemented(section: StudioSection): boolean {
  return (implementedSections as readonly StudioSection[]).includes(section);
}

export const sectionPlan: Record<StudioSection, string> = {
  Repair: "Low-confidence review queue, confidence heat, identity repair and the chord-over-lyric editor.",
  Build: "Stepwise manual build — candidates, MIR, baseline, alignment, save — plus pasted-sheet import.",
  "Automatic Pipeline": "One-shot analyze runs plus the batch job queue with retry and cancel.",
  "Tool Studio": "Live MCP tool catalog with generated forms, telemetry and local history.",
  Library: "Song browser with versions, diff, export and gold marking.",
  Runs: "Run traces with per-step timing and the MIR chord timeline.",
  Evaluation: "Scorecard against gold versions, plus token and cost rollups.",
  Configuration: "Agent workbench, provider status, YouTube session cookies and OAuth clients.",
};
