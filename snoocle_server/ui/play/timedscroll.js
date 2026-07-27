/* Snoocle timed-scroll model — the JS port of the iOS practice scroll engine
 * (`SyncTimeline` + `TimedScrollModel`), master plan task C2.
 *
 * PURE FUNCTIONS ONLY. No DOM, no timers, no fetch. Everything here is a
 * function of (timeline, layout, playhead) -> a number. That is deliberate:
 * it is the part of the player that is easy to get subtly wrong, so it is the
 * part that gets unit tests (`node --test tests_js/`).
 *
 * The semantics are the iOS ones, restated:
 *
 *   1. HOLD AT TOP through the intro. Until the first timed line is due the
 *      sheet parks at `leadInset` and does not creep. This is the scenario
 *      that matters most in practice: a song with 40 seconds of intro should
 *      sit still for 40 seconds, not drift a third of the way down.
 *   2. GLIDE between lines that are close together — offset moves linearly
 *      from one line's anchor to the next across the gap between them.
 *   3. HOLD ACROSS BREAKS. When the gap to the next line exceeds the glide
 *      budget (a solo, an instrumental section), the sheet stays put on the
 *      current line and only starts moving `maxGlide` seconds before the next
 *      line lands — so the scroll always *arrives* on time rather than
 *      crawling through the whole break.
 *   4. HOLD AT THE END after the last timed line.
 *
 * The glide budget is derived from the song, not hardcoded per song:
 * `maxGlide = max(MIN_GLIDE, medianGap * GLIDE_FACTOR)`. A song with dense
 * lines gets a short budget, a slow ballad a longer one.
 *
 * Loaded as a classic script in the browser (`window.SnoocleTimedScroll`) and
 * as a CommonJS module under `node --test`.
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.SnoocleTimedScroll = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  // Glide policy constants — the iOS model's, kept in one place.
  var MIN_GLIDE = 8;      // seconds; never glide over a shorter window
  var GLIDE_FACTOR = 2.5; // multiples of the median line gap

  // ---------------------------------------------------------------------
  // SyncTimeline
  // ---------------------------------------------------------------------

  /**
   * Build the ordered list of timed lines from a Song.
   *
   * Prefers schema-v2 `line.timeSeconds` and falls back to the v1
   * `audio.syncMap` so songs analyzed before Phase A still scroll. Entries
   * are sorted by time and de-duplicated per line; a line without a time
   * simply isn't in the timeline (the model interpolates around it).
   *
   * @returns {{entries: Array<{lineIndex:number,time:number}>, medianGap:number, maxGlide:number}}
   */
  function buildTimeline(song) {
    song = song || {};
    var byLine = new Map();

    (song.lines || []).forEach(function (line) {
      if (line && typeof line.timeSeconds === "number" && isFinite(line.timeSeconds)) {
        byLine.set(line.lineIndex, line.timeSeconds);
      }
    });
    var syncMap = (song.audio && song.audio.syncMap) || [];
    syncMap.forEach(function (p) {
      if (!p || typeof p.time !== "number" || !isFinite(p.time)) return;
      if (!byLine.has(p.lineIndex)) byLine.set(p.lineIndex, p.time);
    });

    var entries = Array.from(byLine.entries())
      .map(function (kv) { return { lineIndex: kv[0], time: kv[1] }; })
      .sort(function (a, b) { return a.time - b.time || a.lineIndex - b.lineIndex; });

    var medianGap = medianOfGaps(entries);
    return {
      entries: entries,
      medianGap: medianGap,
      maxGlide: Math.max(MIN_GLIDE, medianGap * GLIDE_FACTOR),
    };
  }

  function medianOfGaps(entries) {
    if (entries.length < 2) return 0;
    var gaps = [];
    for (var i = 1; i < entries.length; i++) {
      gaps.push(entries[i].time - entries[i - 1].time);
    }
    gaps.sort(function (a, b) { return a - b; });
    var mid = Math.floor(gaps.length / 2);
    return gaps.length % 2 ? gaps[mid] : (gaps[mid - 1] + gaps[mid]) / 2;
  }

  /**
   * Index into `timeline.entries` of the last entry due at or before `t`,
   * or -1 when the song hasn't reached its first timed line yet.
   * Binary search: the player calls this on every frame.
   */
  function indexAt(timeline, t) {
    var e = timeline.entries;
    var lo = 0;
    var hi = e.length - 1;
    var found = -1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (e[mid].time <= t) { found = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    return found;
  }

  // ---------------------------------------------------------------------
  // TimedScrollModel
  // ---------------------------------------------------------------------

  /**
   * The scroll offset (pixels from the top of the sheet) for a playhead.
   *
   * @param timeline  from buildTimeline()
   * @param layout    {lineTops: number[], contentHeight: number,
   *                   viewportHeight: number, leadInset: number}
   *                  `lineTops[i]` is line i's y offset inside the sheet;
   *                  `leadInset` is how far below the viewport top the
   *                  current line should sit (the "park position").
   * @param t         playhead seconds (already offset-corrected)
   */
  function offsetAt(timeline, layout, t) {
    var entries = timeline.entries;
    if (!entries.length) return 0;

    var i = indexAt(timeline, t);

    // (1) Before the first timed line: hold at the top. No creep — the sheet
    // is parked at offset 0 until the first line is actually due, which is
    // what makes a long intro sit still instead of drifting.
    if (i < 0) return 0;

    var current = anchorFor(layout, entries[i].lineIndex);

    // (4) After the last timed line: hold.
    if (i >= entries.length - 1) return clampOffset(current, layout);

    var next = anchorFor(layout, entries[i + 1].lineIndex);
    var from = entries[i].time;
    var to = entries[i + 1].time;
    var span = to - from;
    if (span <= 0) return clampOffset(next, layout);

    // (3) Hold across a break, then glide in over the last maxGlide seconds.
    var glideStart = span > timeline.maxGlide ? to - timeline.maxGlide : from;
    if (t <= glideStart) return clampOffset(current, layout);

    // (2) Glide.
    var progress = (t - glideStart) / (to - glideStart);
    return clampOffset(current + (next - current) * clamp01(progress), layout);
  }

  /** Where line `lineIndex` parks: its top, pulled up by the lead inset. */
  function anchorFor(layout, lineIndex) {
    var tops = layout.lineTops || [];
    var top = tops[lineIndex];
    if (typeof top !== "number") return 0;
    return top - (layout.leadInset || 0);
  }

  function clampOffset(offset, layout) {
    var max = Math.max(0, (layout.contentHeight || 0) - (layout.viewportHeight || 0));
    return Math.min(Math.max(offset, 0), max);
  }

  function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

  /** The line that should be highlighted at `t` (-1 before the first). */
  function activeLineAt(timeline, t) {
    var i = indexAt(timeline, t);
    return i < 0 ? -1 : timeline.entries[i].lineIndex;
  }

  // ---------------------------------------------------------------------
  // Chord timeline (schema v2 chord-level times)
  // ---------------------------------------------------------------------

  /**
   * Flatten every timed chord placement into one ascending list so the
   * player can step a highlight chord-to-chord rather than line-to-line.
   * Placements without a time are skipped — they simply never highlight.
   */
  function buildChordTimeline(song) {
    var out = [];
    ((song || {}).lines || []).forEach(function (line) {
      (line.chordPlacements || []).forEach(function (p, i) {
        if (typeof p.timeSeconds !== "number" || !isFinite(p.timeSeconds)) return;
        out.push({
          lineIndex: line.lineIndex,
          placementIndex: i,
          charIndex: p.charIndex,
          chord: p.chord,
          time: p.timeSeconds,
          confidence: typeof p.confidence === "number" ? p.confidence : null,
        });
      });
    });
    out.sort(function (a, b) { return a.time - b.time; });
    return out;
  }

  /** Index of the sounding chord at `t` (the last one started), or -1. */
  function activeChordIndex(chordTimeline, t) {
    var lo = 0;
    var hi = chordTimeline.length - 1;
    var found = -1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (chordTimeline[mid].time <= t) { found = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    return found;
  }

  // ---------------------------------------------------------------------
  // Playhead extrapolation (port of PracticeScrollDrive.timed)
  // ---------------------------------------------------------------------

  /**
   * The YouTube IFrame API only gives `getCurrentTime()` on demand and its
   * value is coarse, so the player polls at ~4 Hz and extrapolates between
   * polls from the last (wall clock, media time) anchor at the current
   * playback rate. Pausing freezes the estimate at the anchor.
   *
   * @param anchor {mediaTime, wallTime, rate, playing}
   * @param nowMs  current wall clock in ms
   */
  function extrapolate(anchor, nowMs) {
    if (!anchor) return 0;
    if (!anchor.playing) return anchor.mediaTime;
    var elapsed = (nowMs - anchor.wallTime) / 1000;
    if (elapsed < 0) elapsed = 0;
    return anchor.mediaTime + elapsed * (anchor.rate || 1);
  }

  /**
   * Seconds to ADD to every stored time when `videoId` is the upload
   * playing — the song's times were measured against `analyzedVideoId`
   * (master plan B3). Unknown video: no correction, which is right, since
   * the analyzed upload is the common case.
   */
  function offsetForVideo(song, videoId) {
    var audio = (song || {}).audio || {};
    var offsets = audio.videoOffsets || {};
    if (!videoId) return 0;
    var v = offsets[videoId];
    return typeof v === "number" && isFinite(v) ? v : 0;
  }

  return {
    MIN_GLIDE: MIN_GLIDE,
    GLIDE_FACTOR: GLIDE_FACTOR,
    buildTimeline: buildTimeline,
    indexAt: indexAt,
    offsetAt: offsetAt,
    activeLineAt: activeLineAt,
    buildChordTimeline: buildChordTimeline,
    activeChordIndex: activeChordIndex,
    extrapolate: extrapolate,
    offsetForVideo: offsetForVideo,
  };
});
