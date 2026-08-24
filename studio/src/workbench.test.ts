import { beforeEach, describe, expect, it } from "vitest";
import { EMPTY_WORKBENCH, loadWorkbench, saveWorkbench, seedFromWorkbench, type Workbench } from "./workbench";

const fullBench: Workbench = {
  song: { id: "artist--song", title: "Song Title", artist: "The Artist", version: "abc123" },
  audio: { audioRef: "aud_123", filename: "tone.wav", durationSeconds: 12.5 },
  mirJson: '{"chords":[]}',
};

describe("seedFromWorkbench", () => {
  it("returns song_id/song_version for a schema declaring them", () => {
    const schema = { type: "object", properties: { song_id: { type: "string" }, song_version: { type: "string" } } };
    expect(seedFromWorkbench(schema, fullBench)).toEqual({ song_id: "artist--song", song_version: "abc123" });
  });

  it("returns {} for a schema declaring neither song_id nor song_version", () => {
    const schema = { type: "object", properties: { unrelated: { type: "string" } } };
    expect(seedFromWorkbench(schema, fullBench)).toEqual({});
  });

  it("omits song_version when the schema has no song_id", () => {
    const schema = { type: "object", properties: { song_version: { type: "string" } } };
    expect(seedFromWorkbench(schema, fullBench)).toEqual({});
  });

  it("never seeds song_json, audio_path or input_path even when the schema declares them and the workbench is full", () => {
    const schema = {
      type: "object",
      properties: {
        song_id: { type: "string" },
        song_json: { type: "object" },
        prior_song_json: { type: "object" },
        audio_path: { type: "string" },
        input_path: { type: "string" },
      },
    };
    const seed = seedFromWorkbench(schema, fullBench);
    expect(seed).toEqual({ song_id: "artist--song" });
    expect(seed).not.toHaveProperty("song_json");
    expect(seed).not.toHaveProperty("prior_song_json");
    expect(seed).not.toHaveProperty("audio_path");
    expect(seed).not.toHaveProperty("input_path");
  });

  it("seeds audio_ref, input_ref, mir_json, cached_mir_json, title and artist when declared", () => {
    const schema = {
      type: "object",
      properties: {
        audio_ref: { type: "string" },
        input_ref: { type: "string" },
        mir_json: { type: "string" },
        cached_mir_json: { type: "string" },
        title: { type: "string" },
        artist: { type: "string" },
      },
    };
    expect(seedFromWorkbench(schema, fullBench)).toEqual({
      audio_ref: "aud_123",
      input_ref: "aud_123",
      mir_json: '{"chords":[]}',
      cached_mir_json: '{"chords":[]}',
      title: "Song Title",
      artist: "The Artist",
    });
  });

  it("returns {} for a schema with no properties, and for an empty workbench", () => {
    expect(seedFromWorkbench({ type: "object" }, fullBench)).toEqual({});
    const schema = { type: "object", properties: { song_id: { type: "string" } } };
    expect(seedFromWorkbench(schema, EMPTY_WORKBENCH)).toEqual({});
  });
});

describe("loadWorkbench / saveWorkbench", () => {
  beforeEach(() => window.sessionStorage.clear());

  it("round-trips a full workbench through session storage", () => {
    saveWorkbench(fullBench);
    expect(loadWorkbench()).toEqual(fullBench);
  });

  it("loads an absent value as empty", () => {
    expect(loadWorkbench()).toEqual(EMPTY_WORKBENCH);
  });

  it("loads a corrupt stored value as empty rather than throwing", () => {
    window.sessionStorage.setItem("snoocle.studio.workbench", "{not json");
    expect(loadWorkbench()).toEqual(EMPTY_WORKBENCH);
  });

  it("loads a well-formed but wrongly-shaped stored value as empty", () => {
    window.sessionStorage.setItem("snoocle.studio.workbench", JSON.stringify({ song: { id: 42 } }));
    expect(loadWorkbench()).toEqual(EMPTY_WORKBENCH);
  });
});
