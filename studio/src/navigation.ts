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

export function sectionFromPath(pathname: string): StudioSection {
  const match = studioSections.find((section) => sectionPath(section) === pathname);
  return match ?? defaultSection;
}

export const implementedSections = ["Tool Studio"] as const satisfies readonly StudioSection[];

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
