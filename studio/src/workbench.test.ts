import { beforeEach, describe, expect, it } from "vitest";
import {
  benchMismatches,
  derivesIdentityFromRecording,
  EMPTY_WORKBENCH,
  loadWorkbench,
  saveWorkbench,
  seedFromWorkbench,
  type Workbench,
} from "./workbench";

const fullBench: Workbench = {
  song: { id: "artist--song", title: "Song Title", artist: "The Artist", version: "abc123" },
  audio: {
    audioRef: "aud_123",
    filename: "tone.wav",
    durationSeconds: 12.5,
    songId: "artist--song",
    youtubeVideoId: "vid123",
    videoTitle: "The Artist — Song Title",
  },
  mir: { json: '{"chords":[]}', songId: "artist--song", audioRef: "aud_123" },
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

  it("seeds the reference fields when declared", () => {
    const schema = {
      type: "object",
      properties: {
        audio_ref: { type: "string" },
        input_ref: { type: "string" },
        mir_json: { type: "string" },
        cached_mir_json: { type: "string" },
      },
    };
    expect(seedFromWorkbench(schema, fullBench)).toEqual({
      audio_ref: "aud_123",
      input_ref: "aud_123",
      mir_json: '{"chords":[]}',
      cached_mir_json: '{"chords":[]}',
    });
  });

  it("seeds title and artist for a tool that takes no recording", () => {
    const schema = { type: "object", properties: { title: { type: "string" }, artist: { type: "string" } } };
    expect(seedFromWorkbench(schema, fullBench)).toEqual({ title: "Song Title", artist: "The Artist" });
  });

  // The reported bug: acquire_audio declares title, artist and youtube_url_or_id,
  // and the URL wins server-side. Seeding identity there produced a form that
  // claimed one song while fetching another.
  it("does not seed title or artist for a tool that takes a recording", () => {
    const schema = {
      type: "object",
      properties: {
        title: { type: "string" },
        artist: { type: "string" },
        youtube_url_or_id: { type: "string" },
      },
    };
    expect(seedFromWorkbench(schema, fullBench)).toEqual({});
  });

  it("does not seed title or artist alongside audio_ref either", () => {
    const schema = {
      type: "object",
      properties: {
        title: { type: "string" },
        artist: { type: "string" },
        audio_ref: { type: "string" },
      },
    };
    expect(seedFromWorkbench(schema, fullBench)).toEqual({ audio_ref: "aud_123" });
  });

  it("returns {} for a schema with no properties, and for an empty workbench", () => {
    expect(seedFromWorkbench({ type: "object" }, fullBench)).toEqual({});
    const schema = { type: "object", properties: { song_id: { type: "string" } } };
    expect(seedFromWorkbench(schema, EMPTY_WORKBENCH)).toEqual({});
  });
});

describe("derivesIdentityFromRecording", () => {
  it("is true for any recording-source field and false otherwise", () => {
    for (const name of ["youtube_url_or_id", "audio_ref", "audio_path", "input_ref", "input_base64"]) {
      expect(derivesIdentityFromRecording({ type: "object", properties: { [name]: {} } })).toBe(true);
    }
    expect(derivesIdentityFromRecording({ type: "object", properties: { title: {}, artist: {} } })).toBe(false);
    expect(derivesIdentityFromRecording({ type: "object" })).toBe(false);
    expect(derivesIdentityFromRecording(null)).toBe(false);
  });
});

describe("benchMismatches", () => {
  it("is silent when every slot belongs to the selected song", () => {
    expect(benchMismatches(fullBench)).toEqual([]);
  });

  it("is silent with no song selected", () => {
    expect(benchMismatches({ ...fullBench, song: undefined })).toEqual([]);
  });

  it("names audio acquired for another song, by the title that was actually fetched", () => {
    const bench: Workbench = {
      ...fullBench,
      audio: { ...fullBench.audio!, songId: "someone-else--other", videoTitle: "Something Else" },
    };
    expect(benchMismatches(bench)).toContain(
      "The audio (Something Else) was acquired for a different song.",
    );
  });

  it("names MIR captured for another song", () => {
    const bench: Workbench = { ...fullBench, mir: { json: "{}", songId: "someone-else--other" } };
    expect(benchMismatches(bench)).toContain("The captured MIR was computed for a different song.");
  });

  it("names MIR computed from a recording other than the one in the audio slot", () => {
    const bench: Workbench = {
      ...fullBench,
      mir: { json: "{}", songId: "artist--song", audioRef: "aud_other" },
    };
    expect(benchMismatches(bench)).toContain(
      "The captured MIR was computed from a different recording than the one in the audio slot.",
    );
  });

  // Slots written before provenance existed carry no songId; they cannot be
  // judged, and treating them as wrong would cry wolf on every stored session.
  it("says nothing about slots that carry no provenance", () => {
    const bench: Workbench = {
      song: fullBench.song,
      audio: { audioRef: "aud_123", filename: "tone.wav" },
      mir: { json: "{}" },
    };
    expect(benchMismatches(bench)).toEqual([]);
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

  it("drops only the malformed slot, keeping the rest of the workbench", () => {
    window.sessionStorage.setItem("snoocle.studio.workbench", JSON.stringify({
      song: { id: "artist--song", title: "Song Title", artist: "The Artist" },
      audio: { audioRef: 42 },
      // The pre-provenance shape: a bare mirJson string, which this build no
      // longer understands. It must not take the song selection down with it.
      mirJson: '{"chords":[]}',
    }));
    expect(loadWorkbench()).toEqual({
      song: { id: "artist--song", title: "Song Title", artist: "The Artist" },
    });
  });

  it("loads a wrongly-shaped song as empty", () => {
    window.sessionStorage.setItem("snoocle.studio.workbench", JSON.stringify({ song: { id: 42 } }));
    expect(loadWorkbench()).toEqual(EMPTY_WORKBENCH);
  });
});
