import { beforeEach, describe, expect, it } from "vitest";
import {
  benchMismatches,
  derivesIdentityFromRecording,
  EMPTY_WORKBENCH,
  identityConflict,
  knownVideoId,
  loadWorkbench,
  parseVideoId,
  saveWorkbench,
  seedFromWorkbench,
  splitVideoTitle,
  type Workbench,
} from "./workbench";

const fullBench: Workbench = {
  song: { id: "artist--song", title: "Song Title", artist: "The Artist", version: "abc123", youtubeVideoId: "stored99" },
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
  // An acquisition tool gets the recording as well as the names, so the two
  // agree from the start and nothing has to be retyped.
  it("seeds the recording and the identity together for an acquisition tool", () => {
    const schema = {
      type: "object",
      properties: {
        title: { type: "string" },
        artist: { type: "string" },
        youtube_url_or_id: { type: "string" },
      },
    };
    expect(seedFromWorkbench(schema, fullBench)).toEqual({
      youtube_url_or_id: "vid123",
      title: "Song Title",
      artist: "The Artist",
    });
  });

  // The reported bug, in the one case that still admits it: with no recording
  // to hand over, a seeded title and artist are a claim about whatever URL is
  // pasted next.
  it("withholds identity from an acquisition tool when it cannot supply the recording", () => {
    const schema = {
      type: "object",
      properties: {
        title: { type: "string" },
        artist: { type: "string" },
        youtube_url_or_id: { type: "string" },
      },
    };
    const bench: Workbench = { song: { id: "artist--song", title: "Song Title", artist: "The Artist" } };
    expect(seedFromWorkbench(schema, bench)).toEqual({});
  });

  // reconcile_song and build_song_baseline both require title and artist while
  // accepting an audio_ref. Treating "takes a recording" as the test left those
  // required fields blank on the tools that most need them.
  it("seeds identity alongside audio_ref, which is evidence rather than a search term", () => {
    const schema = {
      type: "object",
      properties: {
        title: { type: "string" },
        artist: { type: "string" },
        audio_ref: { type: "string" },
      },
    };
    expect(seedFromWorkbench(schema, fullBench)).toEqual({
      audio_ref: "aud_123",
      title: "Song Title",
      artist: "The Artist",
    });
  });

  it("seeds the song's own recording when nothing is loaded in the audio slot", () => {
    const schema = { type: "object", properties: { youtube_url_or_id: {}, youtube_video_id: {} } };
    const bench: Workbench = {
      song: { id: "artist--song", title: "Song Title", artist: "The Artist", youtubeVideoId: "stored99" },
    };
    expect(seedFromWorkbench(schema, bench)).toEqual({
      youtube_url_or_id: "stored99",
      youtube_video_id: "stored99",
    });
  });

  it("prefers the loaded recording over the song's stored one", () => {
    const schema = { type: "object", properties: { youtube_url_or_id: {} } };
    const bench: Workbench = {
      song: { id: "artist--song", title: "Song Title", artist: "The Artist", youtubeVideoId: "stored99" },
      audio: { audioRef: "aud_1", filename: "a.webm", songId: "artist--song", youtubeVideoId: "loaded11" },
    };
    expect(seedFromWorkbench(schema, bench)).toEqual({ youtube_url_or_id: "loaded11" });
  });

  it("offers an identity read off the video title when no song is selected", () => {
    const schema = { type: "object", properties: { title: {}, artist: {} } };
    const bench: Workbench = {
      audio: { audioRef: "aud_1", filename: "a.webm", videoTitle: "Nirvana - Smells Like Teen Spirit" },
    };
    expect(seedFromWorkbench(schema, bench)).toEqual({
      artist: "Nirvana",
      title: "Smells Like Teen Spirit",
    });
  });

  it("returns {} for a schema with no properties, and for an empty workbench", () => {
    expect(seedFromWorkbench({ type: "object" }, fullBench)).toEqual({});
    const schema = { type: "object", properties: { song_id: { type: "string" } } };
    expect(seedFromWorkbench(schema, EMPTY_WORKBENCH)).toEqual({});
  });
});

describe("derivesIdentityFromRecording", () => {
  // Only youtube_url_or_id makes title/artist into search terms the server
  // will ignore. audio_ref does not: reconcile_song requires the names.
  it("is true only for the acquisition search field", () => {
    expect(derivesIdentityFromRecording({ type: "object", properties: { youtube_url_or_id: {} } })).toBe(true);
    for (const name of ["audio_ref", "audio_path", "input_ref", "input_base64", "title"]) {
      expect(derivesIdentityFromRecording({ type: "object", properties: { [name]: {} } })).toBe(false);
    }
    expect(derivesIdentityFromRecording({ type: "object" })).toBe(false);
    expect(derivesIdentityFromRecording(null)).toBe(false);
  });
});

describe("knownVideoId", () => {
  it("prefers the loaded recording, falls back to the song's, else nothing", () => {
    expect(knownVideoId(fullBench)).toBe("vid123");
    expect(knownVideoId({ song: fullBench.song })).toBe("stored99");
    expect(knownVideoId({})).toBeUndefined();
  });
});

describe("parseVideoId", () => {
  it("reads an id out of the URL shapes YouTube actually hands out", () => {
    expect(parseVideoId("RNCH0xA-hNY")).toBe("RNCH0xA-hNY");
    expect(parseVideoId("https://www.youtube.com/watch?v=RNCH0xA-hNY&list=RDRNCH0xA-hNY&start_radio=1"))
      .toBe("RNCH0xA-hNY");
    expect(parseVideoId("https://youtu.be/RNCH0xA-hNY?t=30")).toBe("RNCH0xA-hNY");
    expect(parseVideoId("https://www.youtube.com/shorts/RNCH0xA-hNY")).toBe("RNCH0xA-hNY");
  });

  it("returns undefined rather than guessing at anything it does not recognise", () => {
    expect(parseVideoId("")).toBeUndefined();
    expect(parseVideoId("   ")).toBeUndefined();
    expect(parseVideoId("Smells Like Teen Spirit")).toBeUndefined();
    expect(parseVideoId("https://example.com/song")).toBeUndefined();
  });
});

describe("splitVideoTitle", () => {
  it("splits on the first separator only", () => {
    expect(splitVideoTitle("Nirvana - Smells Like Teen Spirit"))
      .toEqual({ artist: "Nirvana", title: "Smells Like Teen Spirit" });
    expect(splitVideoTitle("Amy Winehouse – Back to Black"))
      .toEqual({ artist: "Amy Winehouse", title: "Back to Black" });
    expect(splitVideoTitle("A - B - C")).toEqual({ artist: "A", title: "B - C" });
  });

  it("gives up on a title with no separator", () => {
    expect(splitVideoTitle("SmellsLikeTeenSpirit")).toBeUndefined();
    expect(splitVideoTitle("")).toBeUndefined();
  });
});

describe("identityConflict", () => {
  const acquireSchema = {
    type: "object",
    properties: { title: {}, artist: {}, youtube_url_or_id: {} },
  };

  it("names the video that will actually be fetched when it is not the song's own", () => {
    const message = identityConflict(acquireSchema, {
      title: "Back to Black",
      artist: "Amy Winehouse",
      youtube_url_or_id: "https://www.youtube.com/watch?v=RNCH0xA-hNY&list=RDRNCH0xA-hNY",
    }, { song: { id: "amy--back-to-black", title: "Back to Black", artist: "Amy Winehouse", youtubeVideoId: "abcdefghijk" } });
    expect(message).toContain("RNCH0xA-hNY");
    expect(message).toContain("Back to Black");
  });

  it("stays quiet when the URL is the song's own recording", () => {
    expect(identityConflict(acquireSchema, {
      title: "Back to Black",
      artist: "Amy Winehouse",
      youtube_url_or_id: "https://www.youtube.com/watch?v=abcdefghijk",
    }, { song: { id: "amy--back-to-black", title: "Back to Black", artist: "Amy Winehouse", youtubeVideoId: "abcdefghijk" } }))
      .toBeUndefined();
  });

  it("stays quiet once the identity fields no longer claim the workbench song", () => {
    expect(identityConflict(acquireSchema, {
      title: "Something Else",
      artist: "Someone Else",
      youtube_url_or_id: "RNCH0xA-hNY",
    }, { song: { id: "amy--back-to-black", title: "Back to Black", artist: "Amy Winehouse" } }))
      .toBeUndefined();
  });

  it("stays quiet for a tool that does not acquire, an empty field, and no selected song", () => {
    const values = { title: "Back to Black", artist: "Amy Winehouse", youtube_url_or_id: "RNCH0xA-hNY" };
    const song = { id: "amy--back-to-black", title: "Back to Black", artist: "Amy Winehouse" };
    expect(identityConflict({ type: "object", properties: { title: {}, artist: {} } }, values, { song })).toBeUndefined();
    expect(identityConflict(acquireSchema, { ...values, youtube_url_or_id: "" }, { song })).toBeUndefined();
    expect(identityConflict(acquireSchema, values, {})).toBeUndefined();
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
