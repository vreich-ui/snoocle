/* Unit tests for the ported scroll model (master plan C2).
 *
 * Run with:  node --test tests_js/
 *
 * These cover the behaviours that are easy to break and impossible to eyeball:
 * the intro hold, the glide, the hold-across-a-break, and the offset applied
 * when a different upload of the same song is playing.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const ts = require("../snoocle_server/ui/play/timedscroll.js");

// A 5-line song: lines at 10s, 12s, 14s, then a 40s instrumental break, then
// a line at 54s. Every line is 20px tall in the fake layout.
const SONG = {
  lines: [
    { lineIndex: 0, lyrics: "one", timeSeconds: 10, chordPlacements: [
      { charIndex: 0, chord: "C", timeSeconds: 10, confidence: 0.9 }] },
    { lineIndex: 1, lyrics: "two", timeSeconds: 12, chordPlacements: [
      { charIndex: 0, chord: "G", timeSeconds: 12.5 }] },
    { lineIndex: 2, lyrics: "three", timeSeconds: 14, chordPlacements: [] },
    { lineIndex: 3, lyrics: "four", timeSeconds: 54, chordPlacements: [
      { charIndex: 0, chord: "Am", timeSeconds: 54 }] },
  ],
  audio: { youtubeVideoId: "dQw4w9WgXcQ", analyzedVideoId: "dQw4w9WgXcQ" },
};

const LAYOUT = {
  lineTops: [0, 20, 40, 60],
  contentHeight: 400,
  viewportHeight: 100,
  leadInset: 0,
};

// --- timeline ---------------------------------------------------------------

test("timeline collects timed lines in order and derives a glide budget", () => {
  const tl = ts.buildTimeline(SONG);
  assert.deepEqual(tl.entries.map((e) => e.lineIndex), [0, 1, 2, 3]);
  assert.deepEqual(tl.entries.map((e) => e.time), [10, 12, 14, 54]);
  // gaps 2, 2, 40 -> median 2 -> 2 * 2.5 = 5, floored at MIN_GLIDE (8).
  assert.equal(tl.medianGap, 2);
  assert.equal(tl.maxGlide, 8);
});

test("timeline falls back to a v1 syncMap when lines carry no times", () => {
  const v1 = {
    lines: [
      { lineIndex: 0, lyrics: "a", chordPlacements: [] },
      { lineIndex: 1, lyrics: "b", chordPlacements: [] },
    ],
    audio: {
      youtubeVideoId: "dQw4w9WgXcQ",
      syncMap: [
        { lineIndex: 0, time: 3 },
        { lineIndex: 1, time: 9 },
      ],
    },
  };
  const tl = ts.buildTimeline(v1);
  assert.deepEqual(tl.entries, [
    { lineIndex: 0, time: 3 },
    { lineIndex: 1, time: 9 },
  ]);
});

test("schema-v2 line times win over a stale syncMap entry", () => {
  const both = {
    lines: [{ lineIndex: 0, lyrics: "a", timeSeconds: 7, chordPlacements: [] }],
    audio: { syncMap: [{ lineIndex: 0, time: 99 }] },
  };
  assert.equal(ts.buildTimeline(both).entries[0].time, 7);
});

test("a song with no timing at all yields an empty timeline, not a crash", () => {
  const tl = ts.buildTimeline({ lines: [{ lineIndex: 0, lyrics: "a", chordPlacements: [] }] });
  assert.deepEqual(tl.entries, []);
  assert.equal(ts.offsetAt(tl, LAYOUT, 12), 0);
  assert.equal(ts.activeLineAt(tl, 12), -1);
});

// --- the intro hold (Wolf's core scenario) ----------------------------------

test("holds at the top through the intro and does not creep", () => {
  const tl = ts.buildTimeline(SONG);
  for (const t of [0, 2, 5, 9, 9.99]) {
    assert.equal(ts.offsetAt(tl, LAYOUT, t), 0, `t=${t}`);
  }
  assert.equal(ts.activeLineAt(tl, 9.99), -1);
});

test("starts moving exactly when the first line lands", () => {
  const tl = ts.buildTimeline(SONG);
  assert.equal(ts.offsetAt(tl, LAYOUT, 10), 0); // line 0's anchor IS the top
  assert.equal(ts.activeLineAt(tl, 10), 0);
  assert.ok(ts.offsetAt(tl, LAYOUT, 11) > 0);
});

// --- gliding ----------------------------------------------------------------

test("glides linearly between two close lines", () => {
  const tl = ts.buildTimeline(SONG);
  // line 0 (top 0) -> line 1 (top 20) across 10s..12s
  assert.equal(ts.offsetAt(tl, LAYOUT, 11), 10);
  assert.equal(ts.offsetAt(tl, LAYOUT, 12), 20);
});

test("a lead inset lifts every anchor by the same amount", () => {
  const tl = ts.buildTimeline(SONG);
  const inset = Object.assign({}, LAYOUT, { leadInset: 8 });
  // line 1's anchor is 20 - 8 = 12
  assert.equal(ts.offsetAt(tl, inset, 12), 12);
});

// --- the break --------------------------------------------------------------

test("holds through an instrumental break, then glides in over maxGlide", () => {
  const tl = ts.buildTimeline(SONG); // maxGlide = 8
  // line 2 at 14s, line 3 at 54s: hold on line 2 (top 40) until 54-8 = 46.
  assert.equal(ts.offsetAt(tl, LAYOUT, 14), 40);
  assert.equal(ts.offsetAt(tl, LAYOUT, 30), 40, "middle of the solo: still parked");
  assert.equal(ts.offsetAt(tl, LAYOUT, 46), 40, "glide has not started yet");
  assert.equal(ts.offsetAt(tl, LAYOUT, 50), 50, "halfway through the glide");
  assert.equal(ts.offsetAt(tl, LAYOUT, 54), 60, "arrives exactly on the line");
});

test("holds on the last line after the song's final timed line", () => {
  const tl = ts.buildTimeline(SONG);
  assert.equal(ts.offsetAt(tl, LAYOUT, 54), 60);
  assert.equal(ts.offsetAt(tl, LAYOUT, 300), 60);
});

test("offsets never exceed the scrollable range", () => {
  const tl = ts.buildTimeline(SONG);
  const shallow = Object.assign({}, LAYOUT, { contentHeight: 65, viewportHeight: 50 });
  // max scroll is 15; line 3's anchor (60) must clamp to it.
  assert.equal(ts.offsetAt(tl, shallow, 54), 15);
});

// --- chord-level highlight --------------------------------------------------

test("chord timeline is flat, ascending, and skips untimed placements", () => {
  const ct = ts.buildChordTimeline(SONG);
  assert.deepEqual(ct.map((c) => c.chord), ["C", "G", "Am"]);
  assert.deepEqual(ct.map((c) => c.time), [10, 12.5, 54]);
  assert.equal(ct[0].confidence, 0.9);
  assert.equal(ct[1].confidence, null);
});

test("the sounding chord steps forward and holds", () => {
  const ct = ts.buildChordTimeline(SONG);
  assert.equal(ts.activeChordIndex(ct, 0), -1);
  assert.equal(ts.activeChordIndex(ct, 10), 0);
  assert.equal(ts.activeChordIndex(ct, 12.4), 0);
  assert.equal(ts.activeChordIndex(ct, 12.5), 1);
  assert.equal(ts.activeChordIndex(ct, 53.9), 1);
  assert.equal(ts.activeChordIndex(ct, 999), 2);
});

// --- playhead extrapolation -------------------------------------------------

test("extrapolates a playing head from its last poll anchor", () => {
  const anchor = { mediaTime: 30, wallTime: 1000, rate: 1, playing: true };
  assert.equal(ts.extrapolate(anchor, 1000), 30);
  assert.equal(ts.extrapolate(anchor, 1250), 30.25);
});

test("respects the playback rate", () => {
  const anchor = { mediaTime: 30, wallTime: 1000, rate: 0.5, playing: true };
  assert.equal(ts.extrapolate(anchor, 2000), 30.5);
});

test("a paused head freezes at its anchor", () => {
  const anchor = { mediaTime: 30, wallTime: 1000, rate: 1, playing: false };
  assert.equal(ts.extrapolate(anchor, 9999), 30);
});

test("a clock that ran backwards never rewinds the playhead", () => {
  const anchor = { mediaTime: 30, wallTime: 1000, rate: 1, playing: true };
  assert.equal(ts.extrapolate(anchor, 900), 30);
});

// --- per-video offsets (B3) -------------------------------------------------

test("the analyzed upload needs no correction", () => {
  assert.equal(ts.offsetForVideo(SONG, "dQw4w9WgXcQ"), 0);
});

test("a different upload applies its stored offset", () => {
  const song = {
    audio: { analyzedVideoId: "dQw4w9WgXcQ", videoOffsets: { otherVideoX: 3.2 } },
  };
  assert.equal(ts.offsetForVideo(song, "otherVideoX"), 3.2);
  assert.equal(ts.offsetForVideo(song, "unknownVideo"), 0);
  assert.equal(ts.offsetForVideo(song, null), 0);
});
