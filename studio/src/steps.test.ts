import { describe, expect, it } from "vitest";
import type { Song } from "./client";
import { argsForStep, extractCandidateSong, SONG_STEPS, summariseSongChange } from "./steps";

function songFixture(overrides: Partial<Song> = {}): Song {
  return {
    schemaVersion: 2,
    id: "artist--song",
    metadata: { title: "Song", artist: "Artist", bpm: 120 },
    displayPreferences: { capo: 0, tuning: "standard" },
    audio: {},
    sections: [{ sectionIndex: 0, name: "Verse 1", kind: "verse", startLineIndex: 0, endLineIndex: 1 }],
    lines: [
      { lineIndex: 0, lyrics: "hello", chordPlacements: [{ charIndex: 0, chord: "C" }], timeSeconds: 1 },
      { lineIndex: 1, lyrics: "world", chordPlacements: [], timeSeconds: 2 },
    ],
    testOnly: false,
    ...overrides,
  };
}

describe("SONG_STEPS", () => {
  it("has a non-empty label, description, and a valid produces for every entry", () => {
    expect(SONG_STEPS.length).toBeGreaterThan(0);
    for (const step of SONG_STEPS) {
      expect(step.id.length).toBeGreaterThan(0);
      expect(step.label.length).toBeGreaterThan(0);
      expect(step.description.length).toBeGreaterThan(0);
      expect(["song", "report"]).toContain(step.produces);
      expect(step.tool.length).toBeGreaterThan(0);
    }
  });

  it("has unique ids", () => {
    const ids = SONG_STEPS.map((step) => step.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});

describe("argsForStep", () => {
  it("always includes song_id, and never song_json or a *_path key", () => {
    for (const step of SONG_STEPS) {
      const args = argsForStep(step, "artist--song", undefined, undefined);
      expect(args.song_id).toBe("artist--song");
      expect(args).not.toHaveProperty("song_json");
      for (const key of Object.keys(args)) {
        expect(key.endsWith("_path")).toBe(false);
      }
    }
  });

  it("includes song_version only when a version is pinned", () => {
    const step = SONG_STEPS[0];
    expect(argsForStep(step, "artist--song", undefined, undefined)).not.toHaveProperty("song_version");
    expect(argsForStep(step, "artist--song", "v2", undefined)).toMatchObject({ song_version: "v2" });
  });

  it("includes audio_ref only for a needsAudio step", () => {
    const audioStep = SONG_STEPS.find((step) => step.needsAudio);
    expect(audioStep).toBeDefined();
    expect(argsForStep(audioStep!, "artist--song", undefined, "audio-ref-1")).toMatchObject({ audio_ref: "audio-ref-1" });

    for (const step of SONG_STEPS.filter((item) => !item.needsAudio)) {
      const args = argsForStep(step, "artist--song", undefined, "audio-ref-1");
      expect(args).not.toHaveProperty("audio_ref");
    }
  });
});

describe("extractCandidateSong", () => {
  it("finds a song nested under result.song", () => {
    const song = songFixture();
    expect(extractCandidateSong({ song, songSource: "song_id" })).toEqual(song);
  });

  it("finds a song when the result itself looks like a Song", () => {
    const song = songFixture();
    expect(extractCandidateSong(song)).toEqual(song);
  });

  it("returns undefined for a result with no song shape", () => {
    expect(extractCandidateSong({ valid: true })).toBeUndefined();
    expect(extractCandidateSong(undefined)).toBeUndefined();
  });
});

describe("summariseSongChange", () => {
  it("reports a line-timing change", () => {
    const before = songFixture();
    const after = songFixture({
      lines: [
        { lineIndex: 0, lyrics: "hello", chordPlacements: [{ charIndex: 0, chord: "C" }], timeSeconds: 5 },
        { lineIndex: 1, lyrics: "world", chordPlacements: [], timeSeconds: 2 },
      ],
    });
    const facts = summariseSongChange(before, after);
    expect(facts).toContain("1 of 2 lines changed timing");
  });

  it("reports a section count change", () => {
    const before = songFixture();
    const after = songFixture({
      sections: [
        { sectionIndex: 0, name: "Verse 1", kind: "verse", startLineIndex: 0, endLineIndex: 0 },
        { sectionIndex: 1, name: "Chorus", kind: "chorus", startLineIndex: 1, endLineIndex: 1 },
      ],
    });
    const facts = summariseSongChange(before, after);
    expect(facts).toContain("sections: 1 → 2");
  });

  it("reports a chord placement change", () => {
    const before = songFixture();
    const after = songFixture({
      lines: [
        { lineIndex: 0, lyrics: "hello", chordPlacements: [{ charIndex: 0, chord: "G" }], timeSeconds: 1 },
        { lineIndex: 1, lyrics: "world", chordPlacements: [], timeSeconds: 2 },
      ],
    });
    const facts = summariseSongChange(before, after);
    expect(facts).toContain("1 chord placement changed");
  });

  it("says nothing misleading when the two songs are identical", () => {
    const song = songFixture();
    const facts = summariseSongChange(song, songFixture());
    expect(facts).toContain("0 of 2 lines changed timing");
    expect(facts).toContain("bpm unchanged");
    expect(facts.some((fact) => /^\d+ chord placement/.test(fact))).toBe(false);
    expect(facts.some((fact) => fact.startsWith("sections:"))).toBe(false);
    expect(facts.some((fact) => fact.startsWith("lines:"))).toBe(false);
  });
});
