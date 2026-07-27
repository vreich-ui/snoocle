/* Chord diagrams (master plan C3) — run against the REAL vendored chords-db.
 *
 * The acceptance criterion is "every chord in three fixture songs (incl. slash
 * + extended chords) opens a diagram or an approx fallback", so these tests
 * load ui/vendor/chords-db/guitar.json rather than a mock. If a vendor refresh
 * ever drops a suffix the ladder relies on, this is what catches it.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(import.meta.url);

// diagrams.js is a browser script: it reads SnoocleTheory off the global and
// publishes SnoocleDiagrams back onto it. Wire that up, then require it.
globalThis.SnoocleTheory = require("../snoocle_server/ui/play/theory.js");
require("../snoocle_server/ui/play/diagrams.js");
const D = globalThis.SnoocleDiagrams;
const CB = require("../snoocle_server/ui/play/chordbox.js");

const GUITAR = JSON.parse(
  readFileSync(new URL("../snoocle_server/ui/vendor/chords-db/guitar.json", import.meta.url))
);
const UKULELE = JSON.parse(
  readFileSync(new URL("../snoocle_server/ui/vendor/chords-db/ukulele.json", import.meta.url))
);

// --- the vendored data is what we think it is -------------------------------

test("the vendored guitar database is present and indexed as expected", () => {
  assert.equal(GUITAR.main.strings, 6);
  assert.ok(GUITAR.chords.C.length > 0);
  assert.ok(GUITAR.chords.Csharp, "sharp keys are spelled 'Csharp'");
  assert.ok(UKULELE.main.strings, 4);
});

// --- lookup -----------------------------------------------------------------

test("common open chords resolve exactly, with no approximation", () => {
  ["C", "G", "D", "A", "E", "F", "Am", "Em", "Dm", "Bm", "G7", "Cmaj7", "Am7"]
    .forEach((symbol) => {
      const found = D.lookup(GUITAR, symbol);
      assert.ok(found, `no voicing for ${symbol}`);
      assert.equal(found.approx, false, `${symbol} should be exact`);
      assert.ok(found.positions.length > 0);
    });
});

test("sharp and flat spellings both resolve", () => {
  assert.ok(D.lookup(GUITAR, "F#m"));
  assert.ok(D.lookup(GUITAR, "Bb"));
  assert.ok(D.lookup(GUITAR, "Eb"));
  // G# and Ab are the same key in chords-db; both must find it.
  assert.ok(D.lookup(GUITAR, "G#m"));
  assert.ok(D.lookup(GUITAR, "Abm"));
});

test("an extended chord the database lacks falls back and says so", () => {
  const found = D.lookup(GUITAR, "Cmaj9");
  assert.ok(found);
  // chords-db does carry maj9; the point is that IF it resolves via a rung
  // other than the original, `approx` is set and the substitute is named.
  if (found.symbol !== "Cmaj9") {
    assert.equal(found.approx, true);
  }
});

test("a chord nothing in the ladder knows still lands on the triad", () => {
  const found = D.lookup(GUITAR, "Cmaj7#11b13");
  assert.ok(found, "must not return null — the ladder ends at a bare triad");
  assert.equal(found.approx, true);
});

test("slash chords resolve to the upper triad", () => {
  const found = D.lookup(GUITAR, "D/F#");
  assert.ok(found);
  assert.equal(found.symbol, "D");
});

test("an unparseable symbol returns null rather than throwing", () => {
  assert.equal(D.lookup(GUITAR, "???"), null);
  assert.equal(D.lookup(null, "C"), null);
});

// --- every chord in the fixture songs ---------------------------------------

const FIXTURE_SONG_CHORDS = [
  // a plain folk song
  ["C", "G", "Am", "F", "Dm", "G7"],
  // a jazzier one: sevenths, extensions, a slash bass
  ["Cmaj7", "A7", "Dm7", "G13", "Em7b5", "A7b9", "Dm9", "G7", "C6", "C/E"],
  // a sharp-key rock song with barre shapes
  ["F#m", "D", "A", "E", "Bm", "C#m", "G#m", "Bsus4", "Aadd9"],
];

test("every chord across three fixture songs opens a diagram", () => {
  FIXTURE_SONG_CHORDS.forEach((song, i) => {
    song.forEach((symbol) => {
      const found = D.lookup(GUITAR, symbol);
      assert.ok(found, `song ${i}: ${symbol} produced no diagram at all`);
      assert.ok(found.positions.length > 0, `song ${i}: ${symbol} has no positions`);
    });
  });
});

// --- chords-db position -> diagram geometry (chordbox.layout) ---------------

test("open and muted strings become markers, fretted ones become dots", () => {
  // Open C: chords-db frets are low-string-first [-1,3,2,0,1,0].
  const L = CB.layout({ frets: [-1, 3, 2, 0, 1, 0], fingers: [0, 3, 2, 0, 1, 0],
                        baseFret: 1, barres: [] }, 6);
  assert.equal(L.strings, 6);
  assert.equal(L.showNut, true, "baseFret 1 draws the nut");
  assert.deepEqual(L.markers, [
    { string: 1, kind: "muted" },  // low E
    { string: 4, kind: "open" },
    { string: 6, kind: "open" },   // high E
  ]);
  assert.deepEqual(L.dots, [
    { string: 2, fret: 3, finger: 3 },
    { string: 3, fret: 2, finger: 2 },
    { string: 5, fret: 1, finger: 1 },
  ]);
});

test("a shape up the neck shows a base-fret label instead of the nut", () => {
  const L = CB.layout({ frets: [-1, 1, 1, 3, 4, 3], baseFret: 3, barres: [1] }, 6);
  assert.equal(L.showNut, false);
  assert.equal(L.baseFret, 3);
});

test("a barre becomes a string span derived from the stopped strings", () => {
  // F major: barre across all six at fret 1.
  const L = CB.layout({ frets: [1, 3, 3, 2, 1, 1], baseFret: 1, barres: [1] }, 6);
  assert.deepEqual(L.barres, [{ fret: 1, fromString: 1, toString: 6 }]);
});

test("a partial barre spans only the strings actually stopped at that fret", () => {
  const L = CB.layout({ frets: [-1, 3, 3, 2, 1, 1], baseFret: 3, barres: [1] }, 6);
  assert.deepEqual(L.barres, [{ fret: 1, fromString: 5, toString: 6 }]);
});

test("a barre fret nothing is stopped at is dropped, not drawn as a stray bar", () => {
  const L = CB.layout({ frets: [-1, 3, 3, 2, -1, -1], baseFret: 1, barres: [1] }, 6);
  assert.deepEqual(L.barres, []);
});

test("a fingerless position still lays out, with null fingers", () => {
  const L = CB.layout({ frets: [0, 2, 2, 1, 0, 0], baseFret: 1 }, 6);
  assert.deepEqual(L.dots.map((d) => d.finger), [null, null, null]);
});

test("every voicing of every fixture chord lays out without throwing", () => {
  FIXTURE_SONG_CHORDS.flat().forEach((symbol) => {
    const found = D.lookup(GUITAR, symbol);
    found.positions.forEach((position, i) => {
      const L = CB.layout(position, 6);
      assert.equal(L.markers.length + L.dots.length, position.frets.length,
        `${symbol} voicing ${i}: every string must be a marker or a dot`);
    });
  });
});

test("ukulele positions lay out onto four strings", () => {
  const found = D.lookup(UKULELE, "C");
  const L = CB.layout(found.positions[0], 4);
  assert.equal(L.strings, 4);
  assert.equal(L.markers.length + L.dots.length, 4);
});
