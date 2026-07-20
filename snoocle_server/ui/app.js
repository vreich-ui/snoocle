"use strict";
/* Snoocle single-page GUI — vanilla JS, no build step, no dependencies.
 *
 * Everything is same-origin against this server's REST API. The only external
 * resource anywhere in the app is the YouTube embed on the Play tab.
 */

// ---------------------------------------------------------------------------
// Bracket line format (pure, round-trippable).
//
// A Song line is edited as one text line in "inline bracket" form: each chord
// placement is written as `[Chord]` inserted at its charIndex into the lyric
// text, e.g. `[C]When I find myself in [G]times of trouble`.
//
// Invariant: bracketTextToLines(linesToBracketText(x)) deep-equals x for
// well-formed input (placements ascending, charIndex within lyric length,
// non-empty chord tokens). These two functions are the authority for the
// format; the Python test suite pins the same rules.
// ---------------------------------------------------------------------------

function linesToBracketText(lines) {
  return lines
    .map(function (line) {
      var lyrics = line.lyrics || "";
      var placements = (line.chordPlacements || [])
        .slice()
        .sort(function (a, b) { return a.charIndex - b.charIndex; });
      var out = "";
      var cursor = 0;
      placements.forEach(function (p) {
        // chords beyond the lyric length clamp to the end
        var idx = Math.max(0, Math.min(p.charIndex, lyrics.length));
        if (idx > cursor) out += lyrics.slice(cursor, idx);
        out += "[" + p.chord + "]";
        cursor = Math.max(cursor, idx);
      });
      out += lyrics.slice(cursor);
      return out;
    })
    .join("\n");
}

function bracketTextToLines(text) {
  var rawLines = text.split("\n");
  return rawLines.map(function (raw, i) {
    var lyrics = "";
    var chordPlacements = [];
    var re = /\[([^\]]*)\]/g;
    var lastIndex = 0;
    var m;
    while ((m = re.exec(raw)) !== null) {
      lyrics += raw.slice(lastIndex, m.index);
      // charIndex is the position in the de-bracketed lyric string
      chordPlacements.push({ charIndex: lyrics.length, chord: m[1] });
      lastIndex = m.index + m[0].length;
    }
    lyrics += raw.slice(lastIndex);
    return { lineIndex: i, lyrics: lyrics, chordPlacements: chordPlacements };
  });
}

// ---------------------------------------------------------------------------
// API access
// ---------------------------------------------------------------------------

var state = {
  songId: null,
  loadedSong: null,
  loadedVersion: null,
  providers: null,
  activeTab: "edit",
  runs: [],          // run summaries for the open song (newest first)
  activeRunId: null, // run being viewed in the Agent tab
  runPoll: null,     // interval handle while a run is in progress
};

function tokenModal() {
  // Resolves to the entered token string, or null if cancelled.
  return new Promise(function (resolve) {
    var backdrop = el("div", { class: "modal-backdrop" });
    var input = el("input", { type: "password", placeholder: "Bearer token" });
    var modal = el("div", { class: "modal" }, [
      el("h2", {}, ["Authorization required"]),
      el("p", { class: "muted" }, [
        "This server requires a bearer token (SNOOCLE_API_TOKEN). It is stored " +
          "only in this browser.",
      ]),
      input,
      el("div", { class: "actions" }, [
        button("Cancel", "secondary", function () { close(null); }),
        button("Save", "", function () { close(input.value.trim() || null); }),
      ]),
    ]);
    function close(v) { backdrop.remove(); resolve(v); }
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);
    input.focus();
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") close(input.value.trim() || null);
    });
  });
}

async function api(path, options) {
  options = options || {};
  var opts = Object.assign({}, options);
  opts.headers = Object.assign({}, options.headers || {});
  var token = localStorage.getItem("snoocleToken");
  if (token) opts.headers["Authorization"] = "Bearer " + token;

  var res = await fetch(path, opts);
  if (res.status === 401) {
    var entered = await tokenModal();
    if (entered) {
      localStorage.setItem("snoocleToken", entered);
      opts.headers["Authorization"] = "Bearer " + entered;
      res = await fetch(path, opts); // retry once
    }
  }
  return res;
}

async function apiJson(path, options) {
  var res = await api(path, options);
  var body = null;
  try { body = await res.json(); } catch (e) { body = null; }
  return { ok: res.ok, status: res.status, body: body };
}

// ---------------------------------------------------------------------------
// Tiny DOM helpers
// ---------------------------------------------------------------------------

function el(tag, attrs, children) {
  var node = document.createElement(tag);
  attrs = attrs || {};
  Object.keys(attrs).forEach(function (k) {
    if (k === "class") node.className = attrs[k];
    else if (k === "html") node.innerHTML = attrs[k];
    else node.setAttribute(k, attrs[k]);
  });
  (children || []).forEach(function (c) {
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  });
  return node;
}

function button(label, cls, onClick) {
  var b = el("button", { class: "btn " + (cls || "") }, [label]);
  b.addEventListener("click", onClick);
  return b;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

// ---------------------------------------------------------------------------
// Song list
// ---------------------------------------------------------------------------

async function loadSongList() {
  var r = await apiJson("/v1/songs");
  var list = document.getElementById("song-list");
  clear(list);
  if (!r.ok) {
    list.appendChild(el("li", { class: "muted" }, ["(failed to load songs)"]));
    return;
  }
  (r.body.songs || []).forEach(function (id) {
    var li = el("li", { "data-id": id }, [id]);
    if (id === state.songId) li.className = "active";
    li.addEventListener("click", function () { openSong(id); });
    list.appendChild(li);
  });
  if (!(r.body.songs || []).length) {
    list.appendChild(el("li", { class: "muted" }, ["(no songs yet)"]));
  }
}

// ---------------------------------------------------------------------------
// Add song
// ---------------------------------------------------------------------------

async function ensureProviders() {
  if (state.providers) return state.providers;
  var r = await apiJson("/v1/providers");
  state.providers = r.ok ? r.body : {};
  return state.providers;
}

async function addSongModal() {
  var providers = await ensureProviders();
  var backdrop = el("div", { class: "modal-backdrop" });

  var urlInput = el("input", { type: "text", placeholder: "https://youtu.be/... or video id" });
  var titleInput = el("input", { type: "text", placeholder: "Song title" });
  var artistInput = el("input", { type: "text", placeholder: "Artist" });

  var depth = el("select", {}, [
    optionEl("fast", "fast"),
    optionEl("standard", "standard", true),
    optionEl("thorough", "thorough"),
  ]);

  var providerSel = el("select", {}, [optionEl("(server default)", "")]);
  Object.keys(providers).forEach(function (name) {
    providerSel.appendChild(optionEl(name, name));
  });

  var status = el("div", { class: "muted" }, []);
  var building = el("div", { class: "building hidden" }, ["Building — this can take a few minutes…"]);

  var submitBtn = button("Analyze", "", onSubmit);

  var modal = el("div", { class: "modal" }, [
    el("h2", {}, ["Add song"]),
    el("label", {}, ["YouTube URL or ID (optional)"]), urlInput,
    el("div", { class: "row" }, [
      el("div", {}, [el("label", {}, ["Title (optional)"]), titleInput]),
      el("div", {}, [el("label", {}, ["Artist (optional)"]), artistInput]),
    ]),
    el("div", { class: "row" }, [
      el("div", {}, [el("label", {}, ["Depth"]), depth]),
      el("div", {}, [el("label", {}, ["Provider"]), providerSel]),
    ]),
    status,
    building,
    el("div", { class: "actions" }, [
      button("Cancel", "secondary", function () { backdrop.remove(); }),
      submitBtn,
    ]),
  ]);

  async function onSubmit() {
    var url = urlInput.value.trim();
    var title = titleInput.value.trim();
    var artist = artistInput.value.trim();
    // Mirror the API's own validation: need a URL/ID, or both title and artist.
    if (!url && !(title && artist)) {
      status.className = "err";
      status.textContent = "Provide a YouTube URL/ID, or both title and artist.";
      return;
    }
    var body = { analysisDepth: depth.value };
    if (url) body.youtubeUrlOrId = url;
    if (title) body.title = title;
    if (artist) body.artist = artist;
    if (providerSel.value) body.provider = providerSel.value;

    status.className = "muted";
    status.textContent = "";
    building.classList.remove("hidden");
    submitBtn.disabled = true;

    // This request can run for minutes — no client timeout is set.
    var r = await apiJson("/v1/songs/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    building.classList.add("hidden");
    submitBtn.disabled = false;

    if (!r.ok) {
      status.className = "err";
      // On HTTP 502 the detail string carries the "[steps: ...]" diagnostic.
      status.textContent = (r.body && r.body.detail) ? r.body.detail : ("failed (HTTP " + r.status + ")");
      return;
    }
    backdrop.remove();
    await loadSongList();
    if (r.body.songId) {
      await openSong(r.body.songId);
      watchRun(r.body.runId); // jump to the Agent tab and replay this run
    }
  }

  backdrop.appendChild(modal);
  document.body.appendChild(backdrop);
  urlInput.focus();
}

function optionEl(label, value, selected) {
  var o = el("option", { value: value }, [label]);
  if (selected) o.setAttribute("selected", "selected");
  return o;
}

// ---------------------------------------------------------------------------
// Open a song + tabs
// ---------------------------------------------------------------------------

async function openSong(id) {
  if (id !== state.songId) { stopRunPoll(); state.activeRunId = null; state.runs = []; }
  state.playMir = null; // always refetch — a re-run may have produced fresher MIR
  state.songId = id;
  document.getElementById("empty").classList.add("hidden");
  document.getElementById("tabs").style.display = "flex";
  // reflect selection in the list
  Array.prototype.forEach.call(document.querySelectorAll("#song-list li"), function (li) {
    li.classList.toggle("active", li.getAttribute("data-id") === id);
  });

  var song = await apiJson("/v1/songs/" + encodeURIComponent(id));
  var versions = await apiJson("/v1/songs/" + encodeURIComponent(id) + "/versions");
  if (!song.ok) {
    document.getElementById("tab-edit").textContent = "Failed to load song.";
    return;
  }
  state.loadedSong = song.body;
  // versions ordered newest-first; the first entry is the current version.
  state.loadedVersion =
    versions.ok && versions.body.versions && versions.body.versions.length
      ? versions.body.versions[0].version
      : null;

  renderEditTab();
  renderVersionsTab(versions.ok ? versions.body.versions || [] : []);
  renderPlayTab();
  loadRuns();
  selectTab(state.activeTab);
}

function selectTab(name) {
  state.activeTab = name;
  Array.prototype.forEach.call(document.querySelectorAll("#tabs button"), function (b) {
    b.classList.toggle("active", b.getAttribute("data-tab") === name);
  });
  ["edit", "agent", "versions", "play"].forEach(function (t) {
    document.getElementById("tab-" + t).classList.toggle("active", t === name);
  });
}

// ---------------------------------------------------------------------------
// Edit tab
// ---------------------------------------------------------------------------

var editRefs = {};

function renderEditTab() {
  var song = state.loadedSong;
  var panel = document.getElementById("tab-edit");
  clear(panel);
  editRefs = {};

  var md = song.metadata || {};
  editRefs.title = el("input", { type: "text", value: md.title || "" });
  editRefs.artist = el("input", { type: "text", value: md.artist || "" });
  editRefs.bpm = el("input", { type: "number", step: "0.1", value: md.bpm != null ? md.bpm : "" });
  editRefs.key = el("input", { type: "text", value: md.key || "" });

  editRefs.lines = el("textarea", {}, [linesToBracketText(song.lines || [])]);

  // sections editor
  editRefs.sectionRows = [];
  var tbody = el("tbody", {}, []);
  (song.sections || []).forEach(function (s) { addSectionRow(tbody, s); });

  var status = el("div", { class: "muted" }, []);
  editRefs.status = status;

  panel.appendChild(el("div", { class: "row" }, [
    el("div", {}, [el("label", {}, ["Title"]), editRefs.title]),
    el("div", {}, [el("label", {}, ["Artist"]), editRefs.artist]),
  ]));
  panel.appendChild(el("div", { class: "row" }, [
    el("div", {}, [el("label", {}, ["BPM"]), editRefs.bpm]),
    el("div", {}, [el("label", {}, ["Key"]), editRefs.key]),
  ]));

  panel.appendChild(el("label", {}, ["Lines (inline bracket format: [C]lyrics)"]));
  panel.appendChild(editRefs.lines);

  // Live alignment preview: shows chords sitting over their syllables exactly
  // as the Play tab will render them, updating as you move brackets. This is
  // the visual aid for adjusting chord placement by hand.
  panel.appendChild(el("label", {}, ["Alignment preview"]));
  var preview = el("div", { class: "sheet-scroll" }, []);
  panel.appendChild(preview);
  function refreshPreview() {
    clear(preview);
    bracketTextToLines(editRefs.lines.value).forEach(function (line) {
      var chordLine = buildChordLine(line);
      preview.appendChild(el("pre", { class: "sheet" }, [
        (chordLine ? chordLine + "\n" : "") + (line.lyrics || ""),
      ]));
    });
  }
  editRefs.lines.addEventListener("input", refreshPreview);
  refreshPreview();

  panel.appendChild(el("label", {}, ["Sections"]));
  panel.appendChild(el("table", {}, [
    el("thead", {}, [el("tr", {}, [
      el("th", {}, ["Name"]), el("th", {}, ["Start line"]), el("th", {}, [""]),
    ])]),
    tbody,
  ]));
  panel.appendChild(button("+ Add section", "secondary", function () { addSectionRow(tbody, null); }));

  panel.appendChild(el("div", { class: "row", style: "margin-top:16px" }, [
    el("div", {}, [button("Save", "", saveSong)]),
    el("div", {}, []),
  ]));
  panel.appendChild(status);

  // --- help the agent adjust it: re-run reconciliation with your corrections
  editRefs.guidance = el("textarea", {
    placeholder: "e.g. the bridge is Bm not D; keep my chorus lyrics; capo the audio is a half-step sharp",
    style: "min-height:70px",
  }, []);
  editRefs.rerunDepth = el("select", {}, [
    optionEl("fast", "fast"),
    optionEl("standard", "standard", true),
    optionEl("thorough (fills time alignment)", "thorough"),
  ]);
  editRefs.rerunStatus = el("div", { class: "muted" }, []);
  panel.appendChild(el("hr", { style: "margin:22px 0 6px; border:none; border-top:1px solid var(--border)" }, []));
  panel.appendChild(el("div", { class: "section-head" }, ["Help the agent adjust it"]));
  panel.appendChild(el("p", { class: "muted" }, [
    "Re-run reconciliation with your current edits as the starting point plus " +
      "notes below. The agent honors your fixes and fills in the rest; watch it " +
      "work on the Agent tab.",
  ]));
  panel.appendChild(el("label", {}, ["Correction notes (optional)"]));
  panel.appendChild(editRefs.guidance);
  panel.appendChild(el("div", { class: "row", style: "margin-top:10px; align-items:flex-end" }, [
    el("div", {}, [el("label", {}, ["Depth"]), editRefs.rerunDepth]),
    el("div", {}, [button("Re-run agent with my fixes", "", rerunWithFixes)]),
  ]));
  panel.appendChild(editRefs.rerunStatus);
}

async function rerunWithFixes() {
  // Build the current edited Song (same shape saveSong sends) and hand it to
  // the reconciler as prior human-edited evidence + free-text guidance.
  var song = Object.assign({}, state.loadedSong);
  song.metadata = Object.assign({}, state.loadedSong.metadata || {}, {
    title: editRefs.title.value.trim(),
    artist: editRefs.artist.value.trim(),
    bpm: editRefs.bpm.value !== "" ? parseFloat(editRefs.bpm.value) : null,
    key: editRefs.key.value.trim() || null,
  });
  song.lines = bracketTextToLines(editRefs.lines.value);
  song.sections = buildSectionsFromRows();

  var md = song.metadata;
  if (!(md.title && md.artist)) {
    editRefs.rerunStatus.className = "err";
    editRefs.rerunStatus.textContent = "Title and artist are required to re-run.";
    return;
  }
  var body = {
    title: md.title,
    artist: md.artist,
    analysisDepth: editRefs.rerunDepth.value,
    priorSong: song,
    guidance: editRefs.guidance.value.trim() || null,
    expectedVersion: state.loadedVersion,
  };
  var vid = state.loadedSong.audio && state.loadedSong.audio.youtubeVideoId;
  if (vid) body.youtubeUrlOrId = vid;

  editRefs.rerunStatus.className = "muted";
  editRefs.rerunStatus.textContent = "Re-running — watch the Agent tab…";
  var r = await apiJson("/v1/songs/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    editRefs.rerunStatus.className = "err";
    editRefs.rerunStatus.textContent = (r.body && r.body.detail) ? r.body.detail : ("failed (HTTP " + r.status + ")");
    return;
  }
  await openSong(state.songId);
  watchRun(r.body.runId);
}

function addSectionRow(tbody, section) {
  var nameInput = el("input", { type: "text", value: section ? (section.name || "") : "" });
  var startInput = el("input", { type: "number", min: "0", value: section ? (section.startLineIndex != null ? section.startLineIndex : 0) : 0 });
  var ref = { name: nameInput, start: startInput, loaded: section || null, removed: false };
  var tr = el("tr", {}, [
    el("td", {}, [nameInput]),
    el("td", {}, [startInput]),
    el("td", {}, [button("✕", "secondary", function () { ref.removed = true; tr.remove(); })]),
  ]);
  editRefs.sectionRows.push(ref);
  tbody.appendChild(tr);
}

function buildSectionsFromRows() {
  var out = [];
  var i = 0;
  editRefs.sectionRows.forEach(function (ref) {
    if (ref.removed) return;
    var name = ref.name.value.trim();
    var start = parseInt(ref.start.value, 10);
    if (isNaN(start)) start = 0;
    // Merge edits onto the loaded section object so unknown fields (kind,
    // times, endLineIndex) survive; rebuild only for genuinely new rows.
    var base = ref.loaded
      ? Object.assign({}, ref.loaded)
      : { kind: "other", endLineIndex: start };
    base.name = name;
    base.startLineIndex = start;
    base.sectionIndex = i;
    out.push(base);
    i += 1;
  });
  return out;
}

async function saveSong() {
  var song = Object.assign({}, state.loadedSong); // authoritative loaded object
  song.metadata = Object.assign({}, state.loadedSong.metadata || {}, {
    title: editRefs.title.value.trim(),
    artist: editRefs.artist.value.trim(),
    bpm: editRefs.bpm.value !== "" ? parseFloat(editRefs.bpm.value) : null,
    key: editRefs.key.value.trim() || null,
  });
  song.lines = bracketTextToLines(editRefs.lines.value);
  song.sections = buildSectionsFromRows();

  var status = editRefs.status;
  status.className = "muted";
  status.textContent = "Saving…";

  var r = await apiJson("/v1/songs/" + encodeURIComponent(state.songId), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ song: song, message: "Edited in UI", expectedVersion: state.loadedVersion }),
  });

  if (r.ok) {
    status.className = "ok";
    status.textContent = "Saved (version " + (r.body.version || "?") + ").";
    openSong(state.songId); // reload song + versions
    loadSongList();
    return;
  }
  status.className = "err";
  if (r.status === 409) status.textContent = "Someone else saved first — reload.";
  else if (r.status === 400) status.textContent = "Validation error: " + ((r.body && r.body.detail) || "");
  else status.textContent = (r.body && r.body.detail) ? r.body.detail : ("Save failed (HTTP " + r.status + ")");
}

// ---------------------------------------------------------------------------
// Versions tab
// ---------------------------------------------------------------------------

async function renderVersionsTab(versions) {
  var panel = document.getElementById("tab-versions");
  clear(panel);
  var selected = [];

  var diffPre = el("pre", { class: "diff" }, ["Select two versions to diff."]);

  // current gold pointer for this song (drives the ★ marker + score readout)
  var goldRes = await apiJson("/v1/songs/" + encodeURIComponent(state.songId) + "/gold");
  var goldVersion = (goldRes.ok && goldRes.body.goldVersion) || null;
  var scoreLine = el("div", { class: "muted", style: "margin:8px 0" }, []);

  var tbody = el("tbody", {}, []);
  versions.forEach(function (v) {
    var cb = el("input", { type: "checkbox" });
    cb.addEventListener("change", function () {
      if (cb.checked) selected.push(v.version);
      else selected = selected.filter(function (x) { return x !== v.version; });
      if (selected.length === 2) showDiff(selected[0], selected[1], diffPre);
    });
    var isGold = v.version === goldVersion;
    var goldBtn = button(isGold ? "★ gold" : "set gold", "secondary", function () {
      setGold(v.version);
    });
    tbody.appendChild(el("tr", {}, [
      el("td", {}, [cb]),
      el("td", { class: "mono" }, [(isGold ? "★ " : "") + v.version]),
      el("td", {}, [v.timestamp || ""]),
      el("td", {}, [v.message || ""]),
      el("td", {}, [goldBtn]),
    ]));
  });

  panel.appendChild(el("p", { class: "muted" }, [
    "Tick two versions to diff. Mark a human-approved version as ★ gold to " +
      "score the agent against it.",
  ]));
  panel.appendChild(scoreLine);
  panel.appendChild(el("table", {}, [
    el("thead", {}, [el("tr", {}, [
      el("th", {}, [""]), el("th", {}, ["Version"]), el("th", {}, ["When"]),
      el("th", {}, ["Message"]), el("th", {}, ["Gold"]),
    ])]),
    tbody,
  ]));
  panel.appendChild(diffPre);

  // if gold is set, show how the current version scores against it
  if (goldVersion) {
    var sc = await apiJson("/v1/songs/" + encodeURIComponent(state.songId) + "/score");
    if (sc.ok) {
      var m = sc.body.metrics;
      scoreLine.textContent =
        "Current vs gold — overall " + m.overall +
        " · chords " + m.chordSimilarity +
        " · lyrics " + m.lyricSimilarity +
        " · sections " + m.sectionSimilarity +
        (m.timingMAE != null ? " · timing ±" + m.timingMAE + "s" : "");
    }
  }
}

async function setGold(version) {
  var r = await apiJson("/v1/songs/" + encodeURIComponent(state.songId) + "/gold", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ version: version }),
  });
  if (r.ok) openSong(state.songId); // re-render versions with the new gold marker
}

async function showDiff(a, b, pre) {
  pre.textContent = "Loading diff…";
  var res = await api("/v1/songs/" + encodeURIComponent(state.songId) + "/diff?a=" + encodeURIComponent(a) + "&b=" + encodeURIComponent(b));
  var text = await res.text();
  pre.textContent = res.ok ? (text || "(identical)") : ("diff failed: " + text);
}

// ---------------------------------------------------------------------------
// Play tab (play-along groundwork — deliberately minimal)
// ---------------------------------------------------------------------------

function renderPlayTab() {
  var song = state.loadedSong;
  var panel = document.getElementById("tab-play");
  clear(panel);

  var videoId = song.audio && song.audio.youtubeVideoId;
  if (videoId) {
    // The ONE external resource in the whole app, and only on this tab.
    panel.appendChild(el("iframe", {
      class: "yt",
      src: "https://www.youtube-nocookie.com/embed/" + encodeURIComponent(videoId),
      allow: "encrypted-media; picture-in-picture",
      allowfullscreen: "true",
    }, []));
  }

  // MIR chord timeline from the latest successful analysis run — shows what
  // the audio said and which spans were analyzed, alongside the video.
  var mirSlot = el("div", {}, []);
  panel.appendChild(mirSlot);
  attachPlayMirTimeline(mirSlot);

  var sectionForLine = {};
  (song.sections || []).forEach(function (s) {
    if (s.startLineIndex != null) sectionForLine[s.startLineIndex] = s.name || s.kind || "";
  });

  var lines = song.lines || [];
  // ONE scroll container for the whole song: a long line scrolls the sheet, not
  // an invisible per-line box, and chord columns stay aligned across lines.
  var scroll = el("div", { class: "sheet-scroll" }, []);
  lines.forEach(function (line) {
    if (sectionForLine[line.lineIndex] !== undefined) {
      scroll.appendChild(el("div", { class: "section-head" }, [sectionForLine[line.lineIndex]]));
    }
    var chordLine = buildChordLine(line);
    scroll.appendChild(el("pre", { class: "sheet", "data-line": line.lineIndex }, [
      (chordLine ? chordLine + "\n" : "") + (line.lyrics || ""),
    ]));
  });
  if (lines.length) panel.appendChild(scroll);
  else panel.appendChild(el("p", { class: "muted" }, ["No lines yet."]));
}

async function attachPlayMirTimeline(slot) {
  var songId = state.songId;
  // cache per song so tab switches don't refetch
  if (state.playMir && state.playMir.songId === songId) {
    renderPlayMir(slot, state.playMir.run);
    return;
  }
  var list = await apiJson("/v1/songs/" + encodeURIComponent(songId) + "/runs");
  if (!list.ok || state.songId !== songId) return;
  var ok = (list.body.runs || []).filter(function (r) { return r.status === "ok"; });
  if (!ok.length) return;
  var run = await apiJson("/v1/runs/" + encodeURIComponent(ok[0].runId));
  if (!run.ok || state.songId !== songId) return;
  state.playMir = { songId: songId, run: run.body };
  renderPlayMir(slot, run.body);
}

function renderPlayMir(slot, run) {
  if (!run || !run.mir) return;
  var when = (run.startedAt || "").slice(0, 16).replace("T", " ");
  var tl = renderMirTimeline(run.mir, run.mirWindows, "From analysis run " + when);
  if (tl) { clear(slot); slot.appendChild(tl); }
}

function buildChordLine(line) {
  // Position each chord above its charIndex by padding with spaces (monospace).
  var chordLine = "";
  (line.chordPlacements || [])
    .slice()
    .sort(function (a, b) { return a.charIndex - b.charIndex; })
    .forEach(function (p) {
      if (chordLine.length < p.charIndex) {
        chordLine += " ".repeat(p.charIndex - chordLine.length);
      }
      chordLine += p.chord + " ";
    });
  return chordLine.replace(/\s+$/, "");
}

// Future play-along: map playback time -> line via the chord timeline in
// provenance/syncMap and scroll that line into view. No timer logic yet.
function autoScrollTo(seconds) {
  void seconds;
}

// ---------------------------------------------------------------------------
// MIR timeline — what the audio said, and which spans were examined
// ---------------------------------------------------------------------------

// Deterministic chord colors: root pitch class -> hue; minor darker; N gray.
var _PC = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

function chordHue(chord) {
  if (!chord || chord === "N") return "hsl(0, 0%, 55%)";
  var m = /^([A-G])([#b]?)/.exec(chord);
  if (!m) return "hsl(0, 0%, 55%)";
  var pc = _PC[m[1]] + (m[2] === "#" ? 1 : m[2] === "b" ? -1 : 0);
  pc = ((pc % 12) + 12) % 12;
  var minor = /^[A-G][#b]?m(?![a-z])/.test(chord);
  return "hsl(" + pc * 30 + ", " + (minor ? 45 : 65) + "%, " + (minor ? 34 : 44) + "%)";
}

function _fmtTime(s) {
  var m = Math.floor(s / 60);
  var sec = Math.round(s % 60);
  return m + ":" + (sec < 10 ? "0" : "") + sec;
}

// Renders chords + coverage between t0..t1 (seconds). `mirWindows` are the
// agent's analyze_audio_window probes; `analyzedWindows` is MIR-run coverage.
function _timelineTrack(chords, analyzedWindows, mirWindows, t0, t1) {
  var span = Math.max(t1 - t0, 0.001);
  var track = el("div", { class: "mir-timeline" }, []);
  (analyzedWindows || []).forEach(function (w) {
    var left = Math.max(w.start, t0), right = Math.min(w.end, t1);
    if (right <= left) return;
    track.appendChild(el("div", {
      class: "mir-cover",
      style: "left:" + ((left - t0) / span) * 100 + "%;width:" + ((right - left) / span) * 100 + "%",
      title: "analyzed " + _fmtTime(left) + "–" + _fmtTime(right),
    }, []));
  });
  (chords || []).forEach(function (c) {
    var left = Math.max(c.start, t0), right = Math.min(c.end, t1);
    if (right <= left) return;
    var wpct = ((right - left) / span) * 100;
    var seg = el("div", {
      class: "mir-chord",
      style: "left:" + ((left - t0) / span) * 100 + "%;width:" + wpct + "%;background:" + chordHue(c.chord),
      title: c.chord + "  " + _fmtTime(c.start) + "–" + _fmtTime(c.end),
    }, [wpct >= 3.5 ? c.chord : ""]);
    track.appendChild(seg);
  });
  (mirWindows || []).forEach(function (w) {
    var win = w.window || {};
    var left = Math.max(win.start || 0, t0), right = Math.min(win.end || 0, t1);
    if (right <= left) return;
    track.appendChild(el("div", {
      class: "mir-probe",
      style: "left:" + ((left - t0) / span) * 100 + "%;width:" + ((right - left) / span) * 100 + "%",
      title: "agent probed " + _fmtTime(left) + "–" + _fmtTime(right) +
        (w.bpm ? " (bpm " + w.bpm + ")" : ""),
    }, []));
  });
  return track;
}

function _timelineRuler(t0, t1) {
  var span = t1 - t0;
  var step = span > 360 ? 60 : 30;
  var ruler = el("div", { class: "mir-ruler" }, []);
  for (var t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
    ruler.appendChild(el("span", {
      class: "mir-tick",
      style: "left:" + ((t - t0) / span) * 100 + "%",
    }, [_fmtTime(t)]));
  }
  return ruler;
}

// The full-run MIR view: colored chord timeline + analyzed-coverage shading +
// agent-probe overlays + ruler + readout (key/bpm/engines).
function renderMirTimeline(mir, mirWindows, label) {
  var dur = mir.durationSeconds || 0;
  if (!dur || !(mir.chordTimeline || []).length) return null;
  var wrap = el("div", { class: "mir-box" }, []);
  if (label) wrap.appendChild(el("div", { class: "muted", style: "margin-bottom:4px" }, [label]));
  wrap.appendChild(_timelineTrack(mir.chordTimeline, mir.analyzedWindows, mirWindows, 0, dur));
  wrap.appendChild(_timelineRuler(0, dur));
  var readout = "key " + (mir.estimatedKey || "?") + " · bpm " + (mir.bpm || "?") +
    (mir.timeSignature ? " · " + mir.timeSignature : "") +
    " · chords: " + ((mir.engines || {}).chords || "?") +
    " · beats: " + ((mir.engines || {}).beats || "?") +
    " · " + _fmtTime(dur);
  if (mir.truncated) readout += " · (timeline sampled)";
  wrap.appendChild(el("div", { class: "mir-readout" }, [readout]));
  return wrap;
}

// A tool-call-scoped view: just the probed window and the chords it returned.
function renderWindowMiniTimeline(detail) {
  var result = (detail || {}).result || {};
  var win = result.window ||
    { start: (detail.input || {}).start_seconds, end: (detail.input || {}).end_seconds };
  if (win.start == null || win.end == null || !(result.chords || []).length) return null;
  var wrap = el("div", { class: "mir-box mir-mini" }, []);
  wrap.appendChild(_timelineTrack(result.chords, [], [], win.start, win.end));
  wrap.appendChild(_timelineRuler(win.start, win.end));
  wrap.appendChild(el("div", { class: "mir-readout" }, [
    "window " + _fmtTime(win.start) + "–" + _fmtTime(win.end) +
      " · " + result.chords.length + " segment(s)" +
      (result.bpm ? " · bpm " + result.bpm : ""),
  ]));
  return wrap;
}

// ---------------------------------------------------------------------------
// Agent tab — replay the reconciler's step-by-step logic
// ---------------------------------------------------------------------------

function stopRunPoll() {
  if (state.runPoll) { clearInterval(state.runPoll); state.runPoll = null; }
}

async function loadRuns() {
  var r = await apiJson("/v1/songs/" + encodeURIComponent(state.songId) + "/runs");
  state.runs = (r.ok && r.body.runs) ? r.body.runs : [];
  // Default selection: the run we're already watching, else the newest.
  if (!state.activeRunId && state.runs.length) state.activeRunId = state.runs[0].runId;
  renderAgentTab();
}

function renderAgentTab() {
  var panel = document.getElementById("tab-agent");
  clear(panel);

  panel.appendChild(el("p", { class: "muted" }, [
    "Every reconciliation run's logic — what the agent read, searched, fetched, " +
      "and decided. Pick a run; open a step to see its raw input and result.",
  ]));

  if (!state.runs.length) {
    panel.appendChild(el("p", { class: "muted" }, ["No runs recorded for this song yet."]));
    return;
  }

  var chips = el("div", { class: "run-list" }, []);
  state.runs.forEach(function (run) {
    var when = (run.startedAt || "").replace("T", " ").replace("Z", "");
    var chip = el("div", {
      class: "run-chip" + (run.runId === state.activeRunId ? " active" : ""),
      "data-run": run.runId,
    }, [ (run.depth || "?") + " · " + when ]);
    chip.addEventListener("click", function () { openRun(run.runId); });
    chips.appendChild(chip);
  });
  panel.appendChild(chips);

  var body = el("div", { id: "run-body" }, [el("p", { class: "muted" }, ["Loading…"])]);
  panel.appendChild(body);
  if (state.activeRunId) openRun(state.activeRunId);
}

async function openRun(runId) {
  state.activeRunId = runId;
  // reflect selection in the chips without a full re-render
  Array.prototype.forEach.call(document.querySelectorAll(".run-chip"), function (c) {
    c.classList.toggle("active", c.getAttribute("data-run") === runId);
  });
  var r = await apiJson("/v1/runs/" + encodeURIComponent(runId));
  if (!r.ok) return;
  renderRunTrace(r.body);
  // Poll while the run is still in progress (near-live view).
  stopRunPoll();
  if (r.body.status === "running") {
    state.runPoll = setInterval(async function () {
      var rr = await apiJson("/v1/runs/" + encodeURIComponent(runId));
      if (rr.ok) {
        renderRunTrace(rr.body);
        if (rr.body.status !== "running") { stopRunPoll(); loadRuns(); }
      }
    }, 2500);
  }
}

function renderRunTrace(run) {
  var body = document.getElementById("run-body");
  if (!body) return;
  clear(body);

  var badge = el("span", { class: "badge " + (run.status || "") }, [run.status || "?"]);
  var meta = el("div", { class: "run-meta" }, [
    badge, " ",
    document.createTextNode(
      "provider " + (run.provider || "?") + " · model " + (run.model || "?") +
      " · depth " + (run.depth || "?") + " · " + (run.stepCount || 0) + " steps"),
  ]);
  body.appendChild(meta);
  if (run.error) body.appendChild(el("p", { class: "err" }, [run.error]));

  // The audio ground truth for this run: full chord timeline, what was
  // analyzed (shaded), and every window the agent probed (accented).
  if (run.mir) {
    var tl = renderMirTimeline(run.mir, run.mirWindows, "MIR — what the audio said");
    if (tl) body.appendChild(tl);
  }

  (run.steps || []).forEach(function (s) {
    var summary = el("summary", {}, [
      el("span", { class: "kind" }, [s.kind || ""]),
      el("span", { class: "sum" }, [s.summary || ""]),
      el("span", { class: "dur" }, [s.durationSeconds != null ? s.durationSeconds + "s" : ""]),
    ]);
    var children = [summary];
    // analyze_audio_window steps get their window drawn, not just JSON
    if (s.kind === "tool" && (s.detail || {}).tool === "analyze_audio_window") {
      var mini = renderWindowMiniTimeline(s.detail);
      if (mini) children.push(mini);
    }
    children.push(el("pre", { class: "detail" }, [JSON.stringify(s.detail || {}, null, 2)]));
    var det = el("details", { class: "step", "data-kind": s.kind || "" }, children);
    body.appendChild(det);
  });
}

// Called after an analyze kicks off: jump to the Agent tab and watch the run.
function watchRun(runId) {
  if (!runId) return;
  state.activeRunId = runId;
  selectTab("agent");
  loadRuns();
}

// ---------------------------------------------------------------------------
// Scorecard — how the agent scores across every gold-marked song
// ---------------------------------------------------------------------------

async function scorecardModal() {
  var backdrop = el("div", { class: "modal-backdrop" });
  var bodyEl = el("div", {}, [el("p", { class: "muted" }, ["Scoring…"])]);
  var modal = el("div", { class: "modal", style: "width:720px" }, [
    el("h2", {}, ["Agent scorecard"]),
    el("p", { class: "muted" }, [
      "Every song with a ★ gold version, scored: the current version against " +
        "its gold. Lower overall = more room to teach the agent.",
    ]),
    bodyEl,
    el("div", { class: "actions" }, [button("Close", "secondary", function () { backdrop.remove(); })]),
  ]);
  backdrop.appendChild(modal);
  document.body.appendChild(backdrop);

  var r = await apiJson("/v1/eval/scorecard");
  clear(bodyEl);
  if (!r.ok || !r.body.count) {
    bodyEl.appendChild(el("p", { class: "muted" }, [
      "No gold versions marked yet. Open a song → Versions → 'set gold' on a " +
        "human-approved version.",
    ]));
    return;
  }
  var agg = r.body.aggregate || {};
  bodyEl.appendChild(el("p", {}, [
    el("strong", {}, ["Aggregate (" + r.body.count + " songs): "]),
    "overall " + agg.overall + " · chords " + agg.chordSimilarity +
      " · lyrics " + agg.lyricSimilarity + " · sections " + agg.sectionSimilarity +
      (agg.timingMAE != null ? " · timing ±" + agg.timingMAE + "s" : ""),
  ]));
  var tbody = el("tbody", {}, []);
  r.body.songs.forEach(function (row) {
    var m = row.metrics, p = row.process || {};
    var open = button("open", "secondary", function () { backdrop.remove(); openSong(row.songId); });
    tbody.appendChild(el("tr", {}, [
      el("td", { class: "mono" }, [row.songId]),
      el("td", {}, [String(m.overall)]),
      el("td", {}, [String(m.chordSimilarity)]),
      el("td", {}, [String(m.lyricSimilarity)]),
      el("td", {}, [String(m.sectionSimilarity)]),
      el("td", {}, [p.firstPassValid == null ? "" : (p.firstPassValid ? "1st-pass" : (p.attempts + " tries"))]),
      el("td", {}, [p.toolCalls != null ? (p.toolCalls + " tools") : ""]),
      el("td", {}, [open]),
    ]));
  });
  bodyEl.appendChild(el("table", {}, [
    el("thead", {}, [el("tr", {}, [
      el("th", {}, ["Song"]), el("th", {}, ["Overall"]), el("th", {}, ["Chords"]),
      el("th", {}, ["Lyrics"]), el("th", {}, ["Sections"]), el("th", {}, ["Validity"]),
      el("th", {}, ["Effort"]), el("th", {}, [""]),
    ])]),
    tbody,
  ]));
}

// ---------------------------------------------------------------------------
// Wire up
// ---------------------------------------------------------------------------

function init() {
  document.getElementById("add-song-btn").addEventListener("click", addSongModal);
  document.getElementById("refresh-btn").addEventListener("click", loadSongList);
  document.getElementById("scorecard-btn").addEventListener("click", scorecardModal);
  Array.prototype.forEach.call(document.querySelectorAll("#tabs button"), function (b) {
    b.addEventListener("click", function () { selectTab(b.getAttribute("data-tab")); });
  });
  loadSongList();
}

document.addEventListener("DOMContentLoaded", init);
