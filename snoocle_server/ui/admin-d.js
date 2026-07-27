"use strict";
/* Admin upgrades — the Phase D surfaces (master plan D1, D2, D5).
 *
 * Kept in its own file so app.js stays the song editor it already is. Loaded
 * after app.js and uses its globals (`state`, `openSong`, `apiJson`, `el`,
 * `button`, `clear`) directly — one shared script scope, no modules, no build.
 *
 * D1 Review queue — the low-confidence chords from the latest run, with
 *    enough context to judge each one and three ways to resolve it.
 * D2 Queue dashboard — submit many songs, watch them drain.
 * D5 Version "what changed" chips.
 */

// ===========================================================================
// D1 — Review queue
// ===========================================================================

var reviewState = {
  entries: [],      // {lineIndex, charIndex, chord, confidence, reasons}
  cursor: 0,
  edits: {},        // "line:char" -> new chord (batched into ONE save)
  accepted: {},     // "line:char" -> true
  runId: null,
};

function reviewKey(entry) { return entry.lineIndex + ":" + entry.charIndex; }

/** The latest run that actually produced a review queue. Runs are newest
 *  first; a re-run that errored before scoring has none, so fall through. */
function latestReviewRun() {
  for (var i = 0; i < (state.runs || []).length; i++) {
    var run = state.runs[i];
    if (run.reviewQueue && run.reviewQueue.length) return run;
  }
  return null;
}

async function renderReviewTab() {
  var panel = document.getElementById("tab-review");
  clear(panel);

  var run = latestReviewRun();
  if (!run) {
    panel.appendChild(el("p", { class: "muted" }, [
      "No review queue. It is written by a reconciliation run once per-chord " +
        "confidences are scored — re-run this song from the Agent tab to " +
        "produce one.",
    ]));
    return;
  }

  if (reviewState.runId !== run.runId) {
    reviewState = {
      entries: run.reviewQueue.slice(),
      cursor: 0, edits: {}, accepted: {}, runId: run.runId,
    };
  }

  var status = el("div", { class: "muted" }, []);
  var saveBtn = button("Save fixes", "", function () { saveReviewFixes(status); });
  saveBtn.disabled = true;
  reviewState.saveBtn = saveBtn;

  panel.appendChild(el("p", { class: "muted" }, [
    reviewState.entries.length + " placement(s) below the confidence " +
      "threshold, least confident first. j/k to move, Enter to accept. " +
      "Fixes batch into one save.",
  ]));

  var list = el("div", { id: "review-list" }, []);
  panel.appendChild(list);
  panel.appendChild(el("div", { class: "row", style: "margin-top:16px" }, [
    saveBtn,
    button("Ask agent about these", "secondary", askAgentAboutReview),
  ]));
  panel.appendChild(status);

  renderReviewRows(list);
}

function renderReviewRows(list) {
  clear(list);
  reviewState.entries.forEach(function (entry, i) {
    list.appendChild(reviewRow(entry, i, list));
  });
}

function reviewRow(entry, index, list) {
  var key = reviewKey(entry);
  var resolved = !!reviewState.accepted[key] || key in reviewState.edits;
  var current = reviewState.edits[key] || entry.chord;

  var row = el("div", {
    class: "review-row" + (index === reviewState.cursor ? " selected" : "") +
      (resolved ? " resolved" : ""),
    "data-index": String(index),
  }, []);

  row.appendChild(el("div", { class: "review-chord" }, [current]));
  row.appendChild(el("div", { class: "review-context" }, [lineContext(entry)]));
  row.appendChild(confidenceMeter(entry.confidence));
  row.appendChild(el("div", { class: "review-reasons" },
    (entry.reasons || []).map(function (r) { return el("span", { class: "chip" }, [r]); })));

  var actions = el("div", { class: "review-actions" }, []);
  actions.appendChild(smallBtn("▶ bar", "Play this bar", function () {
    playBar(entry);
  }));
  actions.appendChild(smallBtn("Accept", "Keep this chord as-is", function () {
    reviewState.accepted[key] = true;
    delete reviewState.edits[key];
    refreshReview(list);
  }));
  actions.appendChild(smallBtn("Fix", "Replace this chord", function () {
    openFixInput(row, entry, list);
  }));
  row.appendChild(actions);

  row.addEventListener("click", function () {
    reviewState.cursor = index;
    refreshReview(list);
  });
  return row;
}

function smallBtn(label, title, onClick) {
  var b = button(label, "secondary small", onClick);
  b.setAttribute("title", title);
  return b;
}

/** The lyric line with the chord's position marked, so a reviewer can judge
 *  the chord against the words rather than in the abstract. */
function lineContext(entry) {
  var line = ((state.loadedSong || {}).lines || [])[entry.lineIndex];
  if (!line) return "(line " + entry.lineIndex + ")";
  var lyrics = line.lyrics || "";
  var at = Math.max(0, Math.min(entry.charIndex, lyrics.length));
  return (lyrics.slice(0, at) + "‸" + lyrics.slice(at)) || "(instrumental line)";
}

function confidenceMeter(confidence) {
  var bucket = confidence < 0.4 ? "low" : confidence < 0.6 ? "mid" : "high";
  var lit = confidence < 0.4 ? 1 : confidence < 0.6 ? 2 : 3;
  var segments = [0, 1, 2].map(function (i) {
    return el("span", { class: i < lit ? "on" : "" }, []);
  });
  var meter = el("div", { class: "meter " + bucket }, segments);
  meter.setAttribute("title", "confidence " + confidence);
  return meter;
}

/** Seek the Play tab's video to one second before the chord sounds. */
function playBar(entry) {
  var song = state.loadedSong || {};
  var placement = ((song.lines || [])[entry.lineIndex] || {}).chordPlacements || [];
  var match = placement.filter(function (p) { return p.charIndex === entry.charIndex; })[0];
  var t = match && typeof match.timeSeconds === "number" ? match.timeSeconds : null;
  if (t === null) {
    var sync = ((song.audio || {}).syncMap || []).filter(function (p) {
      return p.lineIndex === entry.lineIndex;
    })[0];
    t = sync ? sync.time : null;
  }
  if (t === null) {
    alert("This placement has no timestamp yet — re-run the song so the " +
      "timing pass can attach one.");
    return;
  }
  selectTab("play");
  var frame = document.querySelector("#tab-play iframe.yt");
  if (frame) {
    var vid = (song.audio || {}).youtubeVideoId;
    var start = Math.max(0, Math.floor(t - 1));
    frame.src = "https://www.youtube.com/embed/" + vid + "?start=" + start + "&autoplay=1";
  }
}

function openFixInput(row, entry, list) {
  var key = reviewKey(entry);
  var input = el("input", {
    type: "text", value: reviewState.edits[key] || entry.chord,
    style: "max-width:140px",
  });
  var err = el("span", { class: "err" }, []);

  function commit() {
    var value = input.value.trim();
    if (!value) { err.textContent = "empty"; return; }
    // Client-side guard mirroring the server's rule: a fretboard shape is not
    // a chord. The server validates authoritatively on save.
    if (/^[xX0-9]{4,6}$/.test(value)) {
      err.textContent = "that's a shape, not a chord";
      return;
    }
    if (!/^[A-G][#b]?/.test(value)) {
      err.textContent = "must start with a note name";
      return;
    }
    reviewState.edits[key] = value;
    delete reviewState.accepted[key];
    refreshReview(list);
  }

  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") { e.preventDefault(); commit(); }
    if (e.key === "Escape") refreshReview(list);
  });
  var wrap = el("div", { class: "review-actions" }, [
    input, button("OK", "small", commit), err,
  ]);
  clear(row.querySelector(".review-actions"));
  row.querySelector(".review-actions").appendChild(wrap);
  input.focus();
  input.select();
}

function refreshReview(list) {
  list = list || document.getElementById("review-list");
  if (!list) return;
  renderReviewRows(list);
  var pending = Object.keys(reviewState.edits).length;
  if (reviewState.saveBtn) {
    reviewState.saveBtn.disabled = pending === 0;
    reviewState.saveBtn.textContent = pending
      ? "Save " + pending + " fix" + (pending === 1 ? "" : "es")
      : "Save fixes";
  }
  var selected = list.querySelector(".review-row.selected");
  if (selected) selected.scrollIntoView({ block: "nearest" });
}

/** One save for every fix — not one save per chord. Each save is a new
 *  immutable version, and twelve versions for twelve chord fixes would make
 *  the history unreadable. */
async function saveReviewFixes(status) {
  var song = JSON.parse(JSON.stringify(state.loadedSong));
  var applied = 0;

  Object.keys(reviewState.edits).forEach(function (key) {
    var parts = key.split(":");
    var lineIndex = Number(parts[0]);
    var charIndex = Number(parts[1]);
    var line = (song.lines || [])[lineIndex];
    if (!line) return;
    (line.chordPlacements || []).forEach(function (p) {
      if (p.charIndex === charIndex) {
        p.chord = reviewState.edits[key];
        // A human just overruled the engine: the placement is now certain,
        // and leaving the old low confidence would put it straight back in
        // the next review queue.
        p.confidence = 1.0;
        applied++;
      }
    });
  });

  if (!applied) { status.textContent = "Nothing to save."; return; }

  song.provenance = (song.provenance || []).concat([{
    timestamp: new Date().toISOString(),
    actor: "snoocle-ui/review",
    action: "review-queue-fix",
    sources: [],
    notes: applied + " chord(s) corrected from the review queue",
  }]);

  status.className = "muted";
  status.textContent = "Saving…";
  var r = await apiJson("/v1/songs/" + encodeURIComponent(state.songId), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      song: song,
      message: "Review queue: " + applied + " chord fix(es)",
      expectedVersion: state.loadedVersion,
    }),
  });
  if (!r.ok) {
    status.className = "err";
    status.textContent = "Save failed: " + ((r.body && r.body.detail) || r.status);
    return;
  }
  status.className = "ok";
  status.textContent = "Saved " + applied + " fix(es).";
  reviewState.runId = null; // force a rebuild against the new song
  openSong(state.songId);
}

/** "Ask agent" doesn't guess a prompt — it drops the specific placements into
 *  the guidance box so the human says what they actually mean. */
function askAgentAboutReview() {
  var lines = reviewState.entries.slice(0, 12).map(function (e) {
    return "- line " + e.lineIndex + " char " + e.charIndex + ": " + e.chord +
      " (confidence " + e.confidence + (e.reasons && e.reasons.length
        ? "; " + e.reasons.join(", ") : "") + ")";
  });
  var text = "Please re-check these low-confidence chords:\n" + lines.join("\n");
  selectTab("agent");
  var box = document.querySelector("#tab-agent textarea");
  if (box) {
    box.value = box.value ? box.value + "\n\n" + text : text;
    box.focus();
  }
}

function handleReviewKey(e) {
  if (state.activeTab !== "review" || !reviewState.entries.length) return;
  if (/^(INPUT|TEXTAREA|SELECT)$/.test((e.target || {}).tagName || "")) return;
  var list = document.getElementById("review-list");
  if (e.key === "j") {
    e.preventDefault();
    reviewState.cursor = Math.min(reviewState.cursor + 1, reviewState.entries.length - 1);
    refreshReview(list);
  } else if (e.key === "k") {
    e.preventDefault();
    reviewState.cursor = Math.max(reviewState.cursor - 1, 0);
    refreshReview(list);
  } else if (e.key === "Enter") {
    e.preventDefault();
    var entry = reviewState.entries[reviewState.cursor];
    if (!entry) return;
    reviewState.accepted[reviewKey(entry)] = true;
    reviewState.cursor = Math.min(reviewState.cursor + 1, reviewState.entries.length - 1);
    refreshReview(list);
  }
}

// ===========================================================================
// D2 — Batch queue dashboard
// ===========================================================================

var queuePoll = null;

async function queueModal() {
  var backdrop = el("div", { class: "modal-backdrop" });
  var textarea = el("textarea", {
    placeholder: "One song per line:\n\nThe Beatles - Let It Be\nhttps://youtu.be/dQw4w9WgXcQ\ndQw4w9WgXcQ",
    style: "min-height:140px",
  }, []);

  var depth = el("select", {}, [
    optionEl("fast", "fast"),
    optionEl("standard", "standard", true),
    optionEl("thorough", "thorough"),
  ]);

  var providers = await ensureProviders();
  var providerSel = el("select", {}, [optionEl("(server default)", "")]);
  Object.keys(providers).forEach(function (name) {
    providerSel.appendChild(optionEl(name, name));
  });

  var status = el("div", { class: "muted" }, []);
  var dash = el("div", { id: "queue-dash" }, []);

  var modal = el("div", { class: "modal", style: "width:620px" }, [
    el("h2", {}, ["Analysis queue"]),
    el("p", { class: "muted" }, [
      "One worker drains these in order. A failure marks that song and moves " +
        "on — it never stalls the rest.",
    ]),
    el("label", {}, ["Add many"]), textarea,
    el("div", { class: "row" }, [
      el("div", {}, [el("label", {}, ["Depth"]), depth]),
      el("div", {}, [el("label", {}, ["Provider"]), providerSel]),
    ]),
    el("div", { class: "row", style: "margin-top:12px" }, [
      button("Queue them", "", submit),
      button("Clear finished", "secondary", clearFinished),
    ]),
    status,
    el("div", { class: "section-head" }, ["Queue"]),
    dash,
    el("div", { class: "actions" }, [
      button("Close", "secondary", close),
    ]),
  ]);

  async function submit() {
    var text = textarea.value.trim();
    if (!text) { status.className = "err"; status.textContent = "Nothing to queue."; return; }
    status.className = "muted";
    status.textContent = "Queueing…";
    var r = await apiJson("/v1/queue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        analysisDepth: depth.value,
        provider: providerSel.value || undefined,
      }),
    });
    if (!r.ok) {
      status.className = "err";
      status.textContent = (r.body && r.body.detail) || ("HTTP " + r.status);
      return;
    }
    status.className = "ok";
    status.textContent = "Queued " + r.body.queued + ".";
    textarea.value = "";
    refresh();
  }

  async function clearFinished() {
    await apiJson("/v1/queue", { method: "DELETE" });
    refresh();
  }

  async function refresh() {
    var r = await apiJson("/v1/queue");
    clear(dash);
    if (!r.ok) {
      dash.appendChild(el("p", { class: "err" }, ["Could not read the queue."]));
      return;
    }
    if (!r.body.workerAlive) {
      dash.appendChild(el("p", { class: "err" }, [
        "The queue worker is not running. On Cloud Run this means the instance " +
          "scaled to zero — deploy with --min-instances=1.",
      ]));
    }
    var jobs = r.body.jobs || [];
    if (!jobs.length) {
      dash.appendChild(el("p", { class: "muted" }, ["Queue is empty."]));
      return;
    }
    jobs.slice().reverse().forEach(function (job) { dash.appendChild(queueCard(job, refresh)); });
  }

  function close() {
    if (queuePoll) { clearInterval(queuePoll); queuePoll = null; }
    backdrop.remove();
    loadSongList();
  }

  backdrop.appendChild(modal);
  document.body.appendChild(backdrop);
  refresh();
  if (queuePoll) clearInterval(queuePoll);
  queuePoll = setInterval(refresh, 3000);
}

function queueCard(job, refresh) {
  var badgeClass =
    job.status === "done" ? "ok" :
    job.status === "error" ? "error" :
    job.status === "running" ? "running" : "warn";

  var actions = [];
  if (job.status === "done" && job.songId) {
    actions.push(button("Open", "secondary small", function () {
      openSong(job.songId);
    }));
  }
  if (job.status === "error" || job.status === "cancelled") {
    actions.push(button("Retry", "secondary small", async function () {
      await apiJson("/v1/queue/" + encodeURIComponent(job.id) + "/retry", { method: "POST" });
      refresh();
    }));
  }
  if (job.status === "queued") {
    actions.push(button("Cancel", "secondary small", async function () {
      await apiJson("/v1/queue/" + encodeURIComponent(job.id) + "/cancel", { method: "POST" });
      refresh();
    }));
  }

  return el("div", { class: "queue-card" }, [
    el("span", { class: "badge " + badgeClass }, [job.status]),
    el("div", { class: "grow" }, [
      el("div", {}, [job.label]),
      job.error ? el("div", { class: "err", style: "font-size:12px" }, [job.error]) : null,
      job.attempts > 1
        ? el("div", { class: "muted" }, ["attempt " + job.attempts])
        : null,
    ]),
    el("div", { class: "review-actions" }, actions),
  ]);
}

// ===========================================================================
// D5 — "What changed" chips on the Versions tab
// ===========================================================================

/** Derive counts from the unified diff the server already serves, rather than
 *  fetching two whole songs per row. Cheap and good enough for a chip. */
function changeChips(diffText) {
  var counts = { lines: 0, chords: 0, timing: 0, sections: 0 };
  (diffText || "").split("\n").forEach(function (line) {
    if (line[0] !== "+" && line[0] !== "-") return;
    if (line.startsWith("+++") || line.startsWith("---")) return;
    if (/"chord"\s*:/.test(line)) counts.chords++;
    else if (/"lyrics"\s*:/.test(line)) counts.lines++;
    else if (/"(timeSeconds|time|syncMap|beats)"\s*:/.test(line)) counts.timing++;
    else if (/"(sectionIndex|startLineIndex|endLineIndex|kind)"\s*:/.test(line)) counts.sections++;
  });
  // Each JSON change shows as a - and a + line; halve to count edits, and
  // round up so a pure addition still reads as one change rather than zero.
  Object.keys(counts).forEach(function (k) { counts[k] = Math.ceil(counts[k] / 2); });
  return counts;
}

function renderChangeChips(container, counts) {
  clear(container);
  var any = false;
  [["chords", "chord"], ["lines", "lyric"], ["timing", "timing"], ["sections", "section"]]
    .forEach(function (pair) {
      var n = counts[pair[0]];
      if (!n) return;
      any = true;
      container.appendChild(el("span", { class: "chip" }, [
        n + " " + pair[1] + (n === 1 ? "" : "s"),
      ]));
    });
  if (!any) container.appendChild(el("span", { class: "chip" }, ["metadata only"]));
}

/** Fill in the chips for each consecutive version pair, newest first. */
async function loadVersionChips(versions) {
  for (var i = 0; i < versions.length - 1; i++) {
    var newer = versions[i];
    var older = versions[i + 1];
    var cell = document.querySelector('[data-changes="' + newer.version + '"]');
    if (!cell) continue;
    var r = await api(
      "/v1/songs/" + encodeURIComponent(state.songId) +
      "/diff?a=" + encodeURIComponent(older.version) + "&b=" + encodeURIComponent(newer.version)
    );
    if (!r.ok) { clear(cell); continue; }
    renderChangeChips(cell, changeChips(await r.text()));
  }
  var first = versions.length
    ? document.querySelector('[data-changes="' + versions[versions.length - 1].version + '"]')
    : null;
  if (first) { clear(first); first.appendChild(el("span", { class: "chip" }, ["first version"])); }
}

// ===========================================================================
// Wiring
// ===========================================================================

document.addEventListener("keydown", handleReviewKey);

// Exposed to app.js (single shared script scope).
window.SnoocleAdminD = {
  renderReviewTab: renderReviewTab,
  queueModal: queueModal,
  changeChips: changeChips,
  renderChangeChips: renderChangeChips,
  loadVersionChips: loadVersionChips,
};
