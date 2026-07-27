"use strict";
/* Snoocle play-along player — library grid (C1) + player core (C2).
 *
 * Vanilla, no framework, hash routing. Shares `api`/`apiJson`/`el`/`clear`
 * with the admin via ../common.js; the scroll semantics live in the pure,
 * unit-tested timedscroll.js; chord diagrams in diagrams.js; practice
 * controls in controls.js.
 *
 * The playhead is the single source of truth. Nothing in here tracks "which
 * line we scrolled to" as state — every frame asks the model where the sheet
 * should be for the current playhead and puts it there. That is what makes
 * seeking, rate changes and A/B loops correct for free.
 */

var TS = window.SnoocleTimedScroll;

var P = {
  songs: [],          // library rows (GET /v1/songs -> items)
  song: null,         // the open Song
  songId: null,
  timeline: null,     // TS.buildTimeline(song)
  chords: [],         // TS.buildChordTimeline(song)
  layout: null,       // {lineTops, contentHeight, viewportHeight, leadInset}
  yt: null,           // YT.Player
  ytReady: false,
  videoId: null,
  offset: 0,          // seconds to ADD to stored times for THIS video (B3)
  anchor: null,       // {mediaTime, wallTime, rate, playing}
  raf: null,
  poll: null,
  activeLine: -1,
  activeChord: -1,
  activeSection: -1,
  userScrolledAt: 0,  // wall ms of the last manual scroll (pauses auto-scroll)
  charWidth: 0,       // measured px width of one lyric-font character
  prefs: null,        // per-song persisted display prefs
};

var AUTOSCROLL_GRACE_MS = 4000; // manual scroll wins for this long

// ---------------------------------------------------------------------------
// Persistence of per-song display preferences
//
// This is a server-rendered page, not a Claude artifact, so localStorage is
// the right tool (master plan C2 says so explicitly).
// ---------------------------------------------------------------------------

function prefsKey(songId) { return "snoocle.play." + songId; }

function loadPrefs(songId) {
  var base = {
    transpose: 0, capo: 0, fontScale: 1, rate: 1,
    autoscrollSpeed: 30, heat: false, loopA: null, loopB: null,
  };
  try {
    var raw = localStorage.getItem(prefsKey(songId));
    return raw ? Object.assign(base, JSON.parse(raw)) : base;
  } catch (e) { return base; }
}

function savePrefs() {
  if (!P.songId) return;
  try { localStorage.setItem(prefsKey(P.songId), JSON.stringify(P.prefs)); } catch (e) {}
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

function route() {
  var hash = location.hash || "#/";
  var m = hash.match(/^#\/song\/(.+)$/);
  if (m) { openSong(decodeURIComponent(m[1])); return; }
  showLibrary();
}

function show(viewId) {
  ["view-library", "view-song"].forEach(function (id) {
    document.getElementById(id).hidden = id !== viewId;
  });
}

// ---------------------------------------------------------------------------
// Library (C1)
// ---------------------------------------------------------------------------

function thumbUrl(videoId) {
  return videoId ? "https://img.youtube.com/vi/" + videoId + "/mqdefault.jpg" : null;
}

async function showLibrary() {
  teardownSong();
  show("view-library");
  document.title = "Snoocle";

  var grid = document.getElementById("library-grid");
  if (!P.songs.length) {
    clear(grid);
    for (var i = 0; i < 8; i++) grid.appendChild(el("div", { class: "song-card skeleton" }, []));
    var r = await apiJson("/v1/songs");
    P.songs = (r.ok && (r.body.items || (r.body.songs || []).map(function (id) {
      return { id: id, title: "", artist: "" };
    }))) || [];
  }
  renderLibrary();
}

function renderLibrary() {
  var grid = document.getElementById("library-grid");
  var empty = document.getElementById("library-empty");
  var q = (document.getElementById("library-search").value || "").trim().toLowerCase();
  var sort = document.getElementById("library-sort").value;

  var rows = P.songs.filter(function (s) {
    if (!q) return true;
    return ((s.title || "") + " " + (s.artist || "") + " " + s.id).toLowerCase().indexOf(q) !== -1;
  });

  rows.sort(function (a, b) {
    if (sort === "title") return (a.title || a.id).localeCompare(b.title || b.id);
    if (sort === "artist") return (a.artist || "").localeCompare(b.artist || "");
    return (b.updatedAt || "").localeCompare(a.updatedAt || ""); // recent
  });

  clear(grid);
  empty.hidden = P.songs.length !== 0;
  grid.hidden = P.songs.length === 0;

  rows.forEach(function (s) {
    var art = thumbUrl(s.youtubeVideoId);
    var badges = [];
    if (s.hasTiming) badges.push(el("span", { class: "badge timing", title: "Timed — scrolls in sync" }, ["⏱ timing"]));
    if (s.key) badges.push(el("span", { class: "badge" }, [s.key]));

    var card = el("button", { class: "song-card", type: "button" }, [
      art
        ? el("img", { class: "art", src: art, alt: "", loading: "lazy" })
        : el("div", { class: "art" }, []),
      el("div", { class: "card-title" }, [s.title || s.id]),
      el("div", { class: "card-artist" }, [s.artist || " "]),
      badges.length ? el("div", { class: "badge-row" }, badges) : null,
    ]);
    card.addEventListener("click", function () {
      location.hash = "#/song/" + encodeURIComponent(s.id);
    });
    grid.appendChild(card);
  });
}

// ---------------------------------------------------------------------------
// Song page
// ---------------------------------------------------------------------------

async function openSong(songId) {
  show("view-song");
  P.songId = songId;
  P.prefs = loadPrefs(songId);

  var r = await apiJson("/v1/songs/" + encodeURIComponent(songId));
  if (!r.ok) {
    document.getElementById("song-head").textContent = "Song not found.";
    return;
  }
  P.song = r.body;
  document.title = (P.song.metadata && P.song.metadata.title) || songId;

  P.timeline = TS.buildTimeline(P.song);
  P.chords = TS.buildChordTimeline(P.song);
  P.activeLine = -1;
  P.activeChord = -1;
  P.activeSection = -1;
  P.userScrolledAt = 0;
  P.lastScrollTick = 0;
  P.anchor = null;
  document.getElementById("sheet-scroll").scrollTop = 0;
  P.videoId = (P.song.audio && P.song.audio.youtubeVideoId) || null;
  P.offset = TS.offsetForVideo(P.song, P.videoId);

  renderSongHead();
  renderSheet();
  SnoocleControls.renderSectionChips(P);
  SnoocleControls.renderTransport(P);
  applyFontScale();

  if (P.videoId) mountYouTube(P.videoId);
  startLoop();
}

function teardownSong() {
  stopLoop();
  if (P.yt && P.yt.destroy) { try { P.yt.destroy(); } catch (e) {} }
  P.yt = null;
  P.song = null;
  P.songId = null;
  P.anchor = null;
  document.body.classList.remove("stage");
  document.documentElement.removeAttribute("data-theme");
}

function renderSongHead() {
  var head = document.getElementById("song-head");
  clear(head);
  var md = P.song.metadata || {};
  head.appendChild(el("h2", {}, [md.title || P.songId]));

  var sub = [md.artist, md.key, md.bpm ? Math.round(md.bpm) + " bpm" : null]
    .filter(Boolean).join("  ·  ");
  head.appendChild(el("div", { class: "muted" }, [sub]));

  // Chord palette: distinct chords in order of first appearance (like UG),
  // each one tappable straight to its diagram.
  var seen = [];
  (P.song.lines || []).forEach(function (line) {
    (line.chordPlacements || []).forEach(function (p) {
      if (seen.indexOf(p.chord) === -1) seen.push(p.chord);
    });
  });
  if (seen.length) {
    var palette = el("div", { class: "palette" }, []);
    seen.forEach(function (chord) {
      var b = el("button", { class: "btn secondary small palette-chord", type: "button" }, [
        SnoocleTheory.displayChord(chord, P.prefs.transpose, P.prefs.capo),
      ]);
      b.style.color = "var(--chord)";
      b.addEventListener("click", function () { SnoocleDiagrams.open(chord, P.prefs); });
      palette.appendChild(b);
    });
    head.appendChild(palette);
  }
}

// --- the sheet --------------------------------------------------------------

/** Measure one character of the LYRIC font so chord pills can be positioned
 *  at exact character columns even though the chord font-size differs. */
function measureCharWidth(sample) {
  var probe = el("span", { class: "lyric-row" }, ["0".repeat(40)]);
  probe.style.position = "absolute";
  probe.style.visibility = "hidden";
  probe.style.whiteSpace = "pre";
  sample.appendChild(probe);
  var w = probe.getBoundingClientRect().width / 40;
  probe.remove();
  return w || 8;
}

function sectionForLine(lineIndex) {
  var sections = P.song.sections || [];
  for (var i = 0; i < sections.length; i++) {
    var s = sections[i];
    if (lineIndex >= s.startLineIndex && lineIndex <= s.endLineIndex) return i;
  }
  return -1;
}

function renderSheet() {
  var sheet = document.getElementById("sheet");
  clear(sheet);
  sheet.classList.toggle("untimed", !P.timeline.entries.length);
  sheet.classList.toggle("heat", !!P.prefs.heat);

  P.charWidth = measureCharWidth(sheet);

  var lines = P.song.lines || [];
  var sections = P.song.sections || [];
  var sectionStart = {};
  sections.forEach(function (s, i) { sectionStart[s.startLineIndex] = { s: s, i: i }; });

  P.lineNodes = [];
  lines.forEach(function (line) {
    var start = sectionStart[line.lineIndex];
    if (start) {
      sheet.appendChild(el("div", {
        class: "sheet-section", "data-section": String(start.i),
      }, [start.s.name || start.s.kind || "Section"]));
    }

    var chordRow = el("div", { class: "chord-row" }, []);
    (line.chordPlacements || []).forEach(function (p, pi) {
      var pill = el("button", {
        class: "chord-pill",
        type: "button",
        "data-line": String(line.lineIndex),
        "data-placement": String(pi),
      }, [SnoocleTheory.displayChord(p.chord, P.prefs.transpose, P.prefs.capo)]);
      pill.style.left = (p.charIndex * P.charWidth) + "px";
      if (typeof p.confidence === "number") {
        if (p.confidence < 0.5) pill.classList.add("conf-low");
        else if (p.confidence < 0.75) pill.classList.add("conf-mid");
      }
      pill.addEventListener("click", function (e) {
        e.stopPropagation();
        SnoocleDiagrams.open(p.chord, P.prefs);
      });
      chordRow.appendChild(pill);
    });

    var node = el("div", { class: "sheet-line", "data-line": String(line.lineIndex) }, [
      chordRow,
      el("div", { class: "lyric-row" }, [line.lyrics || " "]),
    ]);
    sheet.appendChild(node);
    P.lineNodes[line.lineIndex] = node;
  });

  measureLayout();
}

/** Cache each line's y offset inside the scroller — the model needs pixels.
 *  Rect-relative rather than offsetTop, so it is correct regardless of which
 *  ancestor happens to be positioned. */
function measureLayout() {
  var scroller = document.getElementById("sheet-scroll");
  var scrollerTop = scroller.getBoundingClientRect().top;
  var tops = [];
  (P.lineNodes || []).forEach(function (node, i) {
    if (!node) return;
    tops[i] = node.getBoundingClientRect().top - scrollerTop + scroller.scrollTop;
  });
  P.layout = {
    lineTops: tops,
    contentHeight: scroller.scrollHeight,
    viewportHeight: scroller.clientHeight,
    // Park the current line ~28% down the viewport: far enough from the top
    // that the next couple of lines are already readable.
    leadInset: Math.round(scroller.clientHeight * 0.28),
  };
}

function applyFontScale() {
  document.documentElement.style.setProperty("--sheet-scale", String(P.prefs.fontScale));
  // Character width depends on font size, so both the pills and the layout
  // must be re-derived.
  requestAnimationFrame(function () {
    var sheet = document.getElementById("sheet");
    if (!sheet || !P.song) return;
    P.charWidth = measureCharWidth(sheet);
    Array.prototype.forEach.call(sheet.querySelectorAll(".chord-pill"), function (pill) {
      var line = P.song.lines[Number(pill.getAttribute("data-line"))];
      var p = line && line.chordPlacements[Number(pill.getAttribute("data-placement"))];
      if (p) pill.style.left = (p.charIndex * P.charWidth) + "px";
    });
    measureLayout();
  });
}

/** Re-label every chord after a transpose/capo change (display only — the
 *  stored chord is the sounding harmony and is never rewritten). */
function relabelChords() {
  var sheet = document.getElementById("sheet");
  Array.prototype.forEach.call(sheet.querySelectorAll(".chord-pill"), function (pill) {
    var line = P.song.lines[Number(pill.getAttribute("data-line"))];
    var p = line && line.chordPlacements[Number(pill.getAttribute("data-placement"))];
    if (p) pill.textContent = SnoocleTheory.displayChord(p.chord, P.prefs.transpose, P.prefs.capo);
  });
  renderSongHead();
}

// ---------------------------------------------------------------------------
// YouTube IFrame API (C2)
// ---------------------------------------------------------------------------

var ytApiPromise = null;

function loadYouTubeApi() {
  if (ytApiPromise) return ytApiPromise;
  ytApiPromise = new Promise(function (resolve) {
    if (window.YT && window.YT.Player) return resolve(window.YT);
    window.onYouTubeIframeAPIReady = function () { resolve(window.YT); };
    var s = document.createElement("script");
    s.src = "https://www.youtube.com/iframe_api";
    document.head.appendChild(s);
  });
  return ytApiPromise;
}

async function mountYouTube(videoId) {
  var YT = await loadYouTubeApi();
  var host = document.getElementById("yt-player");
  clear(host);
  var mount = el("div", { id: "yt-mount" }, []);
  host.appendChild(mount);

  P.yt = new YT.Player(mount, {
    videoId: videoId,
    playerVars: {
      enablejsapi: 1, playsinline: 1, rel: 0, modestbranding: 1, origin: location.origin,
    },
    events: {
      onReady: function () {
        P.ytReady = true;
        if (P.prefs.rate !== 1) trySetRate(P.prefs.rate);
        SnoocleControls.syncTransport(P);
      },
      onStateChange: function (e) {
        pollPlayhead();
        SnoocleControls.syncTransport(P);
        if (e.data === YT.PlayerState.ENDED) stopLoop();
      },
      onPlaybackRateChange: function () { pollPlayhead(); },
    },
  });
}

function trySetRate(rate) {
  if (P.yt && P.yt.setPlaybackRate) { try { P.yt.setPlaybackRate(rate); } catch (e) {} }
}

// ---------------------------------------------------------------------------
// Playhead: poll at 4 Hz, extrapolate every frame
// ---------------------------------------------------------------------------

function pollPlayhead() {
  if (!P.yt || !P.ytReady || !P.yt.getCurrentTime) return;
  var state = P.yt.getPlayerState ? P.yt.getPlayerState() : -1;
  var playing = state === 1; // YT.PlayerState.PLAYING
  var rate = P.yt.getPlaybackRate ? P.yt.getPlaybackRate() : 1;
  P.anchor = {
    mediaTime: P.yt.getCurrentTime() || 0,
    wallTime: Date.now(),
    rate: rate,
    playing: playing,
  };
}

/** The playhead in SONG time: media time plus this upload's stored offset. */
function playhead() {
  if (!P.anchor) return 0;
  return TS.extrapolate(P.anchor, Date.now()) + P.offset;
}

function isPlaying() { return !!(P.anchor && P.anchor.playing); }

function startLoop() {
  stopLoop();
  P.poll = setInterval(function () {
    pollPlayhead();
    SnoocleControls.tickTransport(P, playhead());
  }, 250);
  var frame = function () {
    P.raf = requestAnimationFrame(frame);
    tick();
  };
  P.raf = requestAnimationFrame(frame);
}

function stopLoop() {
  if (P.poll) { clearInterval(P.poll); P.poll = null; }
  if (P.raf) { cancelAnimationFrame(P.raf); P.raf = null; }
}

function tick() {
  if (!P.song) return;
  var t = playhead();

  SnoocleControls.enforceLoop(P, t);
  highlight(t);
  autoScroll(t);
}

function highlight(t) {
  var line = TS.activeLineAt(P.timeline, t);
  if (line !== P.activeLine) {
    if (P.lineNodes && P.lineNodes[P.activeLine]) {
      P.lineNodes[P.activeLine].classList.remove("active");
    }
    if (P.lineNodes && P.lineNodes[line]) P.lineNodes[line].classList.add("active");
    P.activeLine = line;

    var section = line >= 0 ? sectionForLine(line) : -1;
    if (section !== P.activeSection) {
      P.activeSection = section;
      SnoocleControls.markActiveSection(section);
    }
  }

  var ci = TS.activeChordIndex(P.chords, t);
  if (ci !== P.activeChord) {
    setChordActive(P.activeChord, false);
    setChordActive(ci, true);
    P.activeChord = ci;
  }
}

function setChordActive(index, on) {
  if (index < 0 || index >= P.chords.length) return;
  var c = P.chords[index];
  var node = document.querySelector(
    '.chord-pill[data-line="' + c.lineIndex + '"][data-placement="' + c.placementIndex + '"]'
  );
  if (node) node.classList.toggle("active", on);
}

function autoScroll(t) {
  var scroller = document.getElementById("sheet-scroll");
  if (!scroller || !P.layout) return;
  if (Date.now() - P.userScrolledAt < AUTOSCROLL_GRACE_MS) return;

  var now = Date.now();
  var dt = P.lastScrollTick ? Math.min((now - P.lastScrollTick) / 1000, 0.1) : 0;
  P.lastScrollTick = now;

  if (!P.timeline.entries.length) {
    // No timing at all: constant autoscroll at the song's persisted speed
    // (px/s), and only while the video is actually playing.
    if (!isPlaying() || !dt) return;
    scroller.scrollTop += P.prefs.autoscrollSpeed * dt;
    return;
  }

  // Ease toward the model's target rather than snapping — the model already
  // decides WHERE; this only smooths frame-to-frame jitter from the 4 Hz poll.
  var target = TS.offsetAt(P.timeline, P.layout, t);
  var current = scroller.scrollTop;
  var delta = target - current;
  if (Math.abs(delta) < 0.5) return;
  // Frame-rate independent exponential approach (~0.18 per 60Hz frame).
  scroller.scrollTop = current + delta * (1 - Math.pow(1 - 0.18, dt * 60));
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

function init() {
  document.getElementById("library-search").addEventListener("input", renderLibrary);
  document.getElementById("library-sort").addEventListener("change", renderLibrary);
  document.getElementById("back-btn").addEventListener("click", function () {
    location.hash = "#/";
  });

  var scroller = document.getElementById("sheet-scroll");
  ["wheel", "touchstart", "pointerdown"].forEach(function (evt) {
    scroller.addEventListener(evt, function () { P.userScrolledAt = Date.now(); }, { passive: true });
  });

  window.addEventListener("hashchange", route);
  window.addEventListener("resize", function () { if (P.song) { applyFontScale(); } });
  window.addEventListener("online", updateOfflineBanner);
  window.addEventListener("offline", updateOfflineBanner);
  updateOfflineBanner();

  document.addEventListener("keydown", function (e) {
    if (!P.song) return;
    if (/^(INPUT|SELECT|TEXTAREA)$/.test((e.target || {}).tagName || "")) return;
    SnoocleControls.handleKey(P, e);
  });

  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("sw.js").catch(function () {});
  }

  route();
}

function updateOfflineBanner() {
  document.getElementById("offline-banner").hidden = navigator.onLine !== false;
}

// Exposed for controls.js (same-origin, no modules — one shared namespace).
window.SnoocleP = {
  state: P,
  playhead: playhead,
  isPlaying: isPlaying,
  pollPlayhead: pollPlayhead,
  trySetRate: trySetRate,
  relabelChords: relabelChords,
  applyFontScale: applyFontScale,
  renderSheet: renderSheet,
  measureLayout: measureLayout,
  savePrefs: savePrefs,
  sectionForLine: sectionForLine,
};

document.addEventListener("DOMContentLoaded", init);
