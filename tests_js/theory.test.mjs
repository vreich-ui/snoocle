/* Display-only chord transforms (master plan C3/C4).
 *
 * The rule under test is the project's non-negotiable one: transpose and capo
 * change the PRINTED name and never the sounding identity. These tests pin
 * the direction of the capo transform (a capo makes chords print LOWER, not
 * higher), because getting that backwards is the single most likely bug here
 * and it would be silently wrong rather than visibly broken.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const T = require("../snoocle_server/ui/play/theory.js");

// --- parsing ----------------------------------------------------------------

test("splits root, quality and slash bass", () => {
  assert.deepEqual(T.split("F#m7/A#"), {
    pc: 6, quality: "m7", bass: "A#", accidental: "#",
  });
  assert.equal(T.split("Bb").pc, 10);
  assert.equal(T.split("Cb").pc, 11);
  assert.equal(T.split("nonsense"), null);
});

// --- transpose --------------------------------------------------------------

test("transposes root and bass together, keeping the quality", () => {
  assert.equal(T.transpose("Am7", 2), "Bm7");
  assert.equal(T.transpose("C/E", 5), "F/A");
  assert.equal(T.transpose("G", -2), "F");
});

test("wraps around the octave in both directions", () => {
  assert.equal(T.transpose("B", 1), "C");
  assert.equal(T.transpose("C", -1), "B");
});

test("enharmonic spelling follows the SOURCE symbol", () => {
  // A song written in flats stays in flats: Eb + 2 is F, not E#; and a
  // flat-spelled chord landing on a black key prints Db, not C#.
  assert.equal(T.transpose("Eb", 2), "F");
  assert.equal(T.transpose("Bb", 3), "Db");
  assert.equal(T.transpose("A", 4), "C#"); // sharp-spelled source -> sharps
});

test("a zero transpose is the identity, string for string", () => {
  assert.equal(T.transpose("F#m7b5/C", 0), "F#m7b5/C");
});

// --- the capo direction (the bug that would be silently wrong) --------------

test("a capo prints chords LOWER while the sound stays put", () => {
  // Capo 2 + play Em shapes = F#m sounds. So a stored (sounding) F#m must
  // PRINT as Em.
  assert.equal(T.displayChord("F#m", 0, 2), "Em");
  assert.equal(T.displayChord("A", 0, 2), "G");
});

test("transpose and capo compose", () => {
  // +2 transpose then capo 2 cancels out.
  assert.equal(T.displayChord("C", 2, 2), "C");
  assert.equal(T.displayChord("C", 2, 0), "D");
});

test("the caption always names the sounding chord", () => {
  assert.equal(
    T.caption("F#m", 0, 2),
    "sounding F#m — shown as Em (capo 2)"
  );
  assert.equal(T.caption("C", 0, 0), "sounding C");
});

// --- simplification ladder --------------------------------------------------

test("simplification walks extensions down to a triad", () => {
  const ladder = T.simplifications("Cmaj9");
  assert.equal(ladder[0], "Cmaj9");
  assert.ok(ladder.includes("Cmaj7"), ladder.join(","));
  assert.equal(ladder[ladder.length - 1], "C");
});

test("the seventh-preserving rung is tried BEFORE the bare triad", () => {
  // The ordering bug worth a test of its own: falling straight from Cmaj9 to
  // C throws away the major seventh when Cmaj7 was available.
  const ladder = T.simplifications("Cmaj9");
  assert.ok(ladder.indexOf("Cmaj7") < ladder.indexOf("C"), ladder.join(","));

  const minor = T.simplifications("Am11");
  assert.ok(minor.indexOf("Am7") < minor.indexOf("Am"), minor.join(","));

  const dom = T.simplifications("G13");
  assert.ok(dom.indexOf("G7") < dom.indexOf("G"), dom.join(","));
});

test("a diminished chord never simplifies into a major triad first", () => {
  const ladder = T.simplifications("Bdim7");
  assert.ok(ladder.indexOf("Bdim") < ladder.indexOf("B"), ladder.join(","));
});

test("minor extensions land on m7 then m", () => {
  const ladder = T.simplifications("Am11");
  assert.equal(ladder[0], "Am11");
  assert.ok(ladder.includes("Am7"));
  assert.ok(ladder.includes("Am"));
});

test("a slash chord simplifies to its triad, dropping the bass", () => {
  const ladder = T.simplifications("D/F#");
  assert.equal(ladder[0], "D");
});

test("a plain triad has nothing to simplify", () => {
  assert.deepEqual(T.simplifications("G"), ["G"]);
});

// --- chords-db lookup keys --------------------------------------------------

test("maps chord symbols onto chords-db key/suffix pairs", () => {
  assert.deepEqual(T.dbKeySuffix("C"), { key: "C", suffix: "major" });
  assert.deepEqual(T.dbKeySuffix("Am"), { key: "A", suffix: "minor" });
  assert.deepEqual(T.dbKeySuffix("F#m7"), { key: "F#", suffix: "m7" });
  assert.deepEqual(T.dbKeySuffix("Bbmaj7"), { key: "Bb", suffix: "maj7" });
  // chords-db spells this pitch class Ab, never G#.
  assert.equal(T.dbKeySuffix("G#m").key, "Ab");
});
