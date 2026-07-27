"use strict";
/* Display-only chord transforms for the web player (master plan C3/C4).
 *
 * THE RULE THIS FILE EXISTS TO PROTECT (§0.6, and the README's non-negotiable):
 * a stored chord is the SOUNDING harmony. Transpose and capo are DISPLAY
 * transforms and nothing here ever writes back into the Song. Every function
 * takes a sounding chord and returns a string to draw; the caller keeps the
 * sounding symbol for diagrams, harmony logic and saving.
 *
 * Deliberately hand-rolled rather than delegating to tonal: enharmonic
 * spelling has to FOLLOW THE SOURCE SYMBOL (a song written in Eb transposed
 * +2 must render F, never E#), and the server already guarantees every stored
 * chord is a clean sounding-harmony symbol. See ui/vendor/README.md.
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.SnoocleTheory = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  var SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  var FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];
  var PC = {
    C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11,
  };

  /** Split "F#m7/A#" into {root, quality, bass}. Returns null if it doesn't
   *  look like a chord symbol at all (then callers pass the text through). */
  function split(symbol) {
    var m = /^([A-G])([#b]*)(.*)$/.exec((symbol || "").trim());
    if (!m) return null;
    var pc = PC[m[1]];
    for (var i = 0; i < m[2].length; i++) pc += m[2][i] === "#" ? 1 : -1;
    pc = ((pc % 12) + 12) % 12;

    var rest = m[3] || "";
    var bass = null;
    var slash = rest.indexOf("/");
    if (slash !== -1) {
      bass = rest.slice(slash + 1);
      rest = rest.slice(0, slash);
    }
    return { pc: pc, quality: rest, bass: bass, accidental: m[2] };
  }

  function pcName(pc, preferFlats) {
    pc = ((pc % 12) + 12) % 12;
    return (preferFlats ? FLAT : SHARP)[pc];
  }

  /**
   * Transpose a chord symbol by `semitones`, keeping the quality and the
   * slash bass. Enharmonic spelling follows the ORIGINAL symbol: a chord
   * written with a flat stays flat-spelled, so a song in Eb doesn't
   * suddenly render in D#.
   */
  function transpose(symbol, semitones) {
    if (!semitones) return symbol;
    var parts = split(symbol);
    if (!parts) return symbol;
    var preferFlats = parts.accidental.indexOf("b") !== -1;
    var out = pcName(parts.pc + semitones, preferFlats) + parts.quality;
    if (parts.bass) {
      var b = split(parts.bass);
      out += "/" + (b ? pcName(b.pc + semitones, preferFlats) + b.quality : parts.bass);
    }
    return out;
  }

  /**
   * What to PRINT for a sounding chord, given the user's display settings.
   *
   * `transposeSemitones` shifts the sounding pitch (the listener hears the
   * change — used when playing along in a different key). `capo` does NOT
   * shift the sounding pitch: it changes which SHAPE produces it, so the
   * printed name goes DOWN by the capo fret while the sound stays put.
   */
  function displayChord(sounding, transposeSemitones, capo) {
    var shift = (transposeSemitones || 0) - (capo || 0);
    return transpose(sounding, shift);
  }

  /** The human-readable caption a diagram shows, e.g.
   *  "sounding F#m — shown as Em (capo 2)". */
  function caption(sounding, transposeSemitones, capo) {
    var shown = displayChord(sounding, transposeSemitones, capo);
    if (shown === sounding) return "sounding " + sounding;
    var suffix = capo ? " (capo " + capo + ")" : "";
    return "sounding " + sounding + " — shown as " + shown + suffix;
  }

  /**
   * Simplification ladder for chords no diagram database knows:
   * Cmaj9 -> Cmaj7 -> C. Each step is a chord a guitarist can substitute
   * without lying about the harmony's function. Returns the candidates in
   * order, most faithful first, INCLUDING the original.
   */
  function simplifications(symbol) {
    var parts = split(symbol);
    if (!parts) return [symbol];
    var out = [];
    var seen = {};
    function push(q) {
      var s = pcName(parts.pc, parts.accidental.indexOf("b") !== -1) + q;
      if (!seen[s]) { seen[s] = 1; out.push(s); }
    }
    var q = parts.quality;
    push(q);                                    // as written, minus any slash

    // ORDER MATTERS. Each rung must be a closer substitute than the next, so
    // the seventh-preserving reductions come BEFORE any blind token-stripping
    // — otherwise Cmaj9 would fall all the way to a C triad while Cmaj7, a
    // far better substitute, sat one rung further down and never got tried.
    if (/^maj(9|11|13)/.test(q)) push("maj7");
    else if (/^mmaj/.test(q)) push("mmaj7");
    else if (/^m(9|11|13)/.test(q)) push("m7");
    else if (/^(9|11|13)/.test(q)) push("7");
    else if (/^m.*7/.test(q)) push("m7");
    else if (/7/.test(q) && !/maj/.test(q)) push("7");
    else if (/maj7/.test(q)) push("maj7");

    // Then drop decorations that only colour a triad.
    ["#11", "b9", "#9", "b5", "#5", "add9", "13", "11", "9", "6"].forEach(function (ext) {
      if (q.indexOf(ext) !== -1) push(q.split(ext).join(""));
    });

    // Finally the plain triad: minor keeps its third, everything else is major.
    if (/^m(?!aj)/.test(q)) push("m");
    else if (/^(dim|o)/.test(q)) push("dim");
    push("");                                   // bare major triad, last resort
    return out;
  }

  /** Split a symbol into the {key, suffix} pair chords-db indexes by. */
  function dbKeySuffix(symbol) {
    var parts = split(symbol);
    if (!parts) return null;
    // chords-db keys are sharp-spelled except Eb/Ab/Bb/Db — its own key list.
    var DB_KEYS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
    var key = DB_KEYS[parts.pc];
    var q = parts.quality;
    var suffix =
      q === "" ? "major" :
      q === "m" || q === "min" || q === "-" ? "minor" :
      q === "M" || q === "maj" ? "major" :
      q === "dim" || q === "o" ? "dim" :
      q === "+" || q === "aug" ? "aug" :
      q.replace(/^min/, "m").replace(/^maj/, "maj");
    return { key: key, suffix: suffix };
  }

  return {
    split: split,
    transpose: transpose,
    displayChord: displayChord,
    caption: caption,
    simplifications: simplifications,
    dbKeySuffix: dbKeySuffix,
    pcName: pcName,
  };
});
