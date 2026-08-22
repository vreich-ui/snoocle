import { describe, expect, it } from "vitest";
import { chordRow } from "./Sheet";

describe("chordRow", () => {
  it("places each chord at its charIndex column", () => {
    const row = chordRow({
      lyrics: "Hello world",
      chordPlacements: [{ charIndex: 0, chord: "C" }, { charIndex: 6, chord: "G" }],
    });
    expect(row).toBe("C     G");
    expect(row.indexOf("C")).toBe(0);
    expect(row.indexOf("G")).toBe(6);
  });

  it("pushes a later chord one column past the previous chord's end when they would overlap", () => {
    const row = chordRow({
      lyrics: "Something else",
      chordPlacements: [{ charIndex: 0, chord: "Dsus4" }, { charIndex: 2, chord: "G" }],
    });
    // "Dsus4" occupies columns 0-4; charIndex 2 for "G" would overwrite it, so
    // "G" is pushed to column 6 (one column past "Dsus4"'s end at column 5).
    expect(row).toBe("Dsus4 G");
    expect(row.indexOf("G")).toBe(6);
  });

  it("sorts out-of-order placements before laying them out", () => {
    const row = chordRow({
      lyrics: "abcdef",
      chordPlacements: [{ charIndex: 4, chord: "F" }, { charIndex: 0, chord: "C" }],
    });
    expect(row).toBe("C   F");
  });

  it("returns an empty string for a line with no placements", () => {
    expect(chordRow({ lyrics: "no chords here", chordPlacements: [] })).toBe("");
  });
});
