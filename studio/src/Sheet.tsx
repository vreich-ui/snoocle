import type { Song, SongLine } from "./client";

/**
 * Builds one chord line by placing each chord at its charIndex, padding with
 * spaces. Placements are sorted by charIndex first; when the next chord's
 * column would fall inside the one just written (an overlap, not just
 * touching), it is pushed one column past the previous chord's end so no
 * character is overwritten.
 */
export function chordRow(line: Pick<SongLine, "lyrics" | "chordPlacements">): string {
  const sorted = [...line.chordPlacements].sort((a, b) => a.charIndex - b.charIndex);
  let row = "";
  let nextFree = 0;
  for (const placement of sorted) {
    const start = placement.charIndex < nextFree ? nextFree + 1 : placement.charIndex;
    row = row.padEnd(start, " ") + placement.chord;
    nextFree = start + placement.chord.length;
  }
  return row;
}

function sheetText(song: Song): string {
  const headingByLine = new Map(song.sections.map((section) => [section.startLineIndex, section.name]));
  const rows: string[] = [];
  for (const line of song.lines) {
    const heading = headingByLine.get(line.lineIndex);
    if (heading) rows.push(`[${heading}]`);
    if (line.chordPlacements.length) rows.push(chordRow(line));
    rows.push(line.lyrics);
  }
  return rows.join("\n");
}

export function Sheet({ song }: { song: Song }) {
  return <pre className="sheet" data-testid="sheet">{sheetText(song)}</pre>;
}
