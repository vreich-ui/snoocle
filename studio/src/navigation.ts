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

export function sectionPath(section: StudioSection): string {
  return `/studio/${section.toLowerCase().replaceAll(" ", "-")}`;
}

export function sectionFromPath(pathname: string): StudioSection {
  const match = studioSections.find((section) => sectionPath(section) === pathname);
  return match ?? "Repair";
}
