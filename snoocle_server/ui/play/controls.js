"use strict";
/* Practice controls (master plan C4) — section chips, A/B loop, playback
 * rate, transpose/capo display, font size, stage mode.
 *
 * Every control here changes either the PLAYHEAD or the DISPLAY. None of them
 * touch the scroll position directly: the scroll model reads the playhead
 * every frame, so a seek, a rate change or a loop wrap all resynchronise the
 * sheet without a line of extra code.
 *
 * Transpose and capo are display-only (§0.6). They are persisted per song in
 * localStorage, never written back into the Song, and the sheet's stored
 * chords stay the sounding harmony.
 */
(function (root) {
  var RATES = [0.5, 0.75, 0.9, 1];

  var ui = {}; // transport element refs

  // -------------------------------------------------------------------
  // Section chips
  // -------------------------------------------------------------------

  function renderSectionChips(P) {
    var bar = document.getElementById("section-chips");
    clear(bar);
    var sections = (P.song && P.song.sections) || [];
    if (!sections.length) { bar.hidden = true; return; }
    bar.hidden = false;

    sections.forEach(function (s, i) {
      var chip = el("button", {
        class: "section-chip", type: "button", "data-section": String(i),
      }, [s.name || s.kind || "Section " + (i + 1)]);
      chip.addEventListener("click", function () { seekToSection(P, i); });
      // Long-press (touch) / right-click (mouse) sets the loop to this section.
      chip.addEventListener("contextmenu", function (e) {
        e.preventDefault();
        loopSection(P, i);
      });
      var timer = null;
      chip.addEventListener("touchstart", function () {
        timer = setTimeout(function () { loopSection(P, i); }, 550);
      }, { passive: true });
      ["touchend", "touchmove", "touchcancel"].forEach(function (evt) {
        chip.addEventListener(evt, function () { clearTimeout(timer); }, { passive: true });
      });
      bar.appendChild(chip);
    });
  }

  function markActiveSection(index) {
    var bar = document.getElementById("section-chips");
    Array.prototype.forEach.call(bar.querySelectorAll(".section-chip"), function (chip, i) {
      var on = i === index;
      chip.classList.toggle("active", on);
      if (on) chip.scrollIntoView({ block: "nearest", inline: "center" });
    });
  }

  /** A section's start in SONG time: its MIR start, else the first timed line
   *  inside it, else nothing (the chip then just highlights). */
  function sectionStartTime(P, index) {
    var s = (P.song.sections || [])[index];
    if (!s) return null;
    if (typeof s.startTime === "number") return s.startTime;
    var entry = P.timeline.entries.find(function (e) {
      return e.lineIndex >= s.startLineIndex && e.lineIndex <= s.endLineIndex;
    });
    return entry ? entry.time : null;
  }

  function seekToSection(P, index) {
    var t = sectionStartTime(P, index);
    if (t === null) return;
    seek(P, t);
  }

  function loopSection(P, index) {
    var s = (P.song.sections || [])[index];
    var start = sectionStartTime(P, index);
    if (start === null) return;
    var end = typeof s.endTime === "number" ? s.endTime : sectionStartTime(P, index + 1);
    P.prefs.loopA = start;
    P.prefs.loopB = end !== null && end !== undefined ? end : null;
    root.SnoocleP.savePrefs();
    syncTransport(P);
  }

  // -------------------------------------------------------------------
  // Seeking + loop
  // -------------------------------------------------------------------

  /** Seek in SONG time — the per-video offset is subtracted here, in exactly
   *  one place, so every caller can think in song seconds. */
  function seek(P, songTime) {
    if (!P.yt || !P.yt.seekTo) return;
    P.yt.seekTo(Math.max(0, songTime - P.offset), true);
    root.SnoocleP.pollPlayhead();
    P.userScrolledAt = 0; // a deliberate seek re-arms auto-scroll immediately
  }

  function enforceLoop(P, t) {
    var a = P.prefs.loopA;
    var b = P.prefs.loopB;
    if (a === null || b === null || b <= a) return;
    if (t > b) seek(P, a);
  }

  // -------------------------------------------------------------------
  // Transport bar
  // -------------------------------------------------------------------

  function renderTransport(P) {
    var bar = document.getElementById("transport");
    clear(bar);
    ui = {};

    ui.play = el("button", { class: "btn play-btn", type: "button", "aria-label": "Play" }, ["▶"]);
    ui.play.addEventListener("click", function () { togglePlay(P); });

    ui.prev = iconBtn("⏮", "Previous section", function () { stepSection(P, -1); });
    ui.next = iconBtn("⏭", "Next section", function () { stepSection(P, 1); });

    ui.a = el("button", { class: "btn secondary", type: "button", title: "Set loop start" }, ["A"]);
    ui.a.addEventListener("click", function () {
      P.prefs.loopA = root.SnoocleP.playhead();
      if (P.prefs.loopB !== null && P.prefs.loopB <= P.prefs.loopA) P.prefs.loopB = null;
      root.SnoocleP.savePrefs(); syncTransport(P);
    });
    ui.b = el("button", { class: "btn secondary", type: "button", title: "Set loop end" }, ["B"]);
    ui.b.addEventListener("click", function () {
      P.prefs.loopB = root.SnoocleP.playhead();
      root.SnoocleP.savePrefs(); syncTransport(P);
    });
    ui.loopClear = iconBtn("↺", "Clear loop", function () {
      P.prefs.loopA = null; P.prefs.loopB = null;
      root.SnoocleP.savePrefs(); syncTransport(P);
    });

    ui.rate = el("select", { "aria-label": "Playback rate" }, RATES.map(function (r) {
      var o = el("option", { value: String(r) }, [r === 1 ? "1×" : r + "×"]);
      if (r === P.prefs.rate) o.setAttribute("selected", "selected");
      return o;
    }));
    ui.rate.addEventListener("change", function () {
      P.prefs.rate = parseFloat(ui.rate.value);
      root.SnoocleP.trySetRate(P.prefs.rate);
      root.SnoocleP.savePrefs();
    });

    ui.tune = el("button", { class: "btn secondary", type: "button", title: "Tune & display" }, ["Tune"]);
    ui.tune.addEventListener("click", function () { openTuneSheet(P); });

    ui.stage = el("button", { class: "btn secondary", type: "button", title: "Stage mode" }, ["Stage"]);
    ui.stage.addEventListener("click", function () { toggleStage(P); });

    ui.time = el("span", { class: "muted" }, ["0:00"]);

    bar.appendChild(ui.prev);
    bar.appendChild(ui.play);
    bar.appendChild(ui.next);
    bar.appendChild(ui.time);
    bar.appendChild(el("span", { class: "spacer" }, []));
    bar.appendChild(ui.a);
    bar.appendChild(ui.b);
    bar.appendChild(ui.loopClear);
    bar.appendChild(ui.rate);
    bar.appendChild(ui.tune);
    bar.appendChild(ui.stage);

    syncTransport(P);
  }

  function iconBtn(glyph, label, onClick) {
    var b = el("button", { class: "btn secondary icon-btn", type: "button", "aria-label": label, title: label }, [glyph]);
    b.addEventListener("click", onClick);
    return b;
  }

  function togglePlay(P) {
    if (!P.yt) return;
    if (root.SnoocleP.isPlaying()) P.yt.pauseVideo(); else P.yt.playVideo();
    setTimeout(root.SnoocleP.pollPlayhead, 60);
  }

  function stepSection(P, direction) {
    var sections = (P.song.sections || []);
    if (!sections.length) return;
    var current = root.SnoocleP.state.activeSection;
    if (current < 0) current = 0;
    var target = Math.min(Math.max(current + direction, 0), sections.length - 1);
    // Stepping back within the first 2s of a section goes to the previous one;
    // otherwise "back" restarts the current section, which is what a player
    // practising a passage actually wants.
    if (direction < 0) {
      var start = sectionStartTime(P, current);
      if (start !== null && root.SnoocleP.playhead() - start > 2) target = current;
    }
    seekToSection(P, target);
  }

  function syncTransport(P) {
    if (!ui.play) return;
    var playing = root.SnoocleP.isPlaying();
    ui.play.textContent = playing ? "❚❚" : "▶";
    ui.play.setAttribute("aria-label", playing ? "Pause" : "Play");
    ui.a.classList.toggle("on", P.prefs.loopA !== null);
    ui.b.classList.toggle("on", P.prefs.loopB !== null);
    var looping = P.prefs.loopA !== null && P.prefs.loopB !== null;
    ui.loopClear.disabled = !looping && P.prefs.loopA === null;
    if (!P.yt) ui.play.disabled = true;
  }

  function tickTransport(P, t) {
    if (ui.time) ui.time.textContent = formatTime(t);
    var miniTime = document.getElementById("mini-time");
    if (miniTime && !document.getElementById("mini-bar").hidden) {
      miniTime.textContent = formatTime(t);
    }
  }

  function formatTime(seconds) {
    if (!isFinite(seconds) || seconds < 0) seconds = 0;
    var m = Math.floor(seconds / 60);
    var s = Math.floor(seconds % 60);
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  // -------------------------------------------------------------------
  // Tune sheet: transpose, capo, font size, confidence heat
  // -------------------------------------------------------------------

  function openTuneSheet(P) {
    var rows = [];

    rows.push(el("h3", {}, ["Tune & display"]));
    rows.push(el("p", { class: "caption" }, [
      "Transpose and capo change what is PRINTED. The stored chords stay the " +
        "sounding harmony — always.",
    ]));

    rows.push(stepperRow("Transpose", function () { return semitoneLabel(P.prefs.transpose); },
      function (delta, refresh) {
        P.prefs.transpose = Math.max(-11, Math.min(11, P.prefs.transpose + delta));
        root.SnoocleP.relabelChords(); root.SnoocleP.savePrefs(); refresh();
      }));

    rows.push(stepperRow("Capo", function () {
      return P.prefs.capo ? "fret " + P.prefs.capo : "none";
    }, function (delta, refresh) {
      P.prefs.capo = Math.max(0, Math.min(11, P.prefs.capo + delta));
      root.SnoocleP.relabelChords(); root.SnoocleP.savePrefs(); refresh();
    }));

    rows.push(stepperRow("Font size", function () {
      return Math.round(P.prefs.fontScale * 100) + "%";
    }, function (delta, refresh) {
      P.prefs.fontScale = Math.max(0.8, Math.min(1.8,
        Math.round((P.prefs.fontScale + delta * 0.1) * 10) / 10));
      root.SnoocleP.applyFontScale(); root.SnoocleP.savePrefs(); refresh();
    }));

    // Confidence heat (D4): hidden entirely when the song has no confidences.
    var hasConfidence = (P.song.lines || []).some(function (line) {
      return (line.chordPlacements || []).some(function (p) {
        return typeof p.confidence === "number";
      });
    });
    if (hasConfidence) {
      var heat = el("button", { class: "btn secondary", type: "button" }, [
        P.prefs.heat ? "On" : "Off",
      ]);
      heat.addEventListener("click", function () {
        P.prefs.heat = !P.prefs.heat;
        heat.textContent = P.prefs.heat ? "On" : "Off";
        document.getElementById("sheet").classList.toggle("heat", P.prefs.heat);
        root.SnoocleP.savePrefs();
      });
      rows.push(el("div", { class: "control-row" }, [
        el("label", {}, ["Confidence heat"]), heat,
      ]));
    }

    if (!P.timeline.entries.length) {
      rows.push(stepperRow("Autoscroll speed", function () {
        return P.prefs.autoscrollSpeed + " px/s";
      }, function (delta, refresh) {
        P.prefs.autoscrollSpeed = Math.max(5, Math.min(200, P.prefs.autoscrollSpeed + delta * 5));
        root.SnoocleP.savePrefs(); refresh();
      }));
    }

    root.SnoocleDiagrams.buildPopover(rows);
  }

  function semitoneLabel(n) {
    if (!n) return "none";
    return (n > 0 ? "+" : "") + n + " semitone" + (Math.abs(n) === 1 ? "" : "s");
  }

  function stepperRow(label, read, apply) {
    var value = el("span", {}, [read()]);
    function refresh() { value.textContent = read(); }
    var minus = el("button", { class: "btn secondary icon-btn", type: "button",
      "aria-label": label + " down" }, ["−"]);
    var plus = el("button", { class: "btn secondary icon-btn", type: "button",
      "aria-label": label + " up" }, ["+"]);
    minus.addEventListener("click", function () { apply(-1, refresh); });
    plus.addEventListener("click", function () { apply(1, refresh); });
    return el("div", { class: "control-row" }, [
      el("label", {}, [label]), minus, value, plus,
    ]);
  }

  // -------------------------------------------------------------------
  // Stage mode
  // -------------------------------------------------------------------

  var wakeLock = null;

  async function toggleStage(P) {
    var on = !document.body.classList.contains("stage");
    document.body.classList.toggle("stage", on);
    if (on) {
      document.documentElement.setAttribute("data-theme", "stage");
      addStageZones(P);
      try {
        if (navigator.wakeLock) wakeLock = await navigator.wakeLock.request("screen");
      } catch (e) { /* wake lock is a nicety, never a blocker */ }
    } else {
      document.documentElement.removeAttribute("data-theme");
      removeStageZones();
      if (wakeLock) { try { wakeLock.release(); } catch (e) {} wakeLock = null; }
    }
    root.SnoocleP.applyFontScale();
  }

  function addStageZones(P) {
    removeStageZones();
    [["left", -1], ["right", 1]].forEach(function (pair) {
      var zone = el("button", {
        class: "stage-zone " + pair[0], type: "button",
        "aria-label": pair[1] < 0 ? "Previous section" : "Next section",
      }, []);
      zone.addEventListener("click", function () { stepSection(P, pair[1]); });
      document.body.appendChild(zone);
    });
  }

  function removeStageZones() {
    Array.prototype.forEach.call(document.querySelectorAll(".stage-zone"), function (z) {
      z.remove();
    });
  }

  // -------------------------------------------------------------------
  // Keyboard (§3.5: space play/pause, arrows section jump, L loop, +/- transpose)
  // -------------------------------------------------------------------

  function handleKey(P, e) {
    switch (e.key) {
      case " ":
        e.preventDefault(); togglePlay(P); break;
      case "ArrowLeft":
        e.preventDefault(); stepSection(P, -1); break;
      case "ArrowRight":
        e.preventDefault(); stepSection(P, 1); break;
      case "l": case "L":
        e.preventDefault();
        if (P.prefs.loopA === null) P.prefs.loopA = root.SnoocleP.playhead();
        else if (P.prefs.loopB === null) P.prefs.loopB = root.SnoocleP.playhead();
        else { P.prefs.loopA = null; P.prefs.loopB = null; }
        root.SnoocleP.savePrefs(); syncTransport(P);
        break;
      case "+": case "=":
        e.preventDefault();
        P.prefs.transpose = Math.min(11, P.prefs.transpose + 1);
        root.SnoocleP.relabelChords(); root.SnoocleP.savePrefs();
        break;
      case "-": case "_":
        e.preventDefault();
        P.prefs.transpose = Math.max(-11, P.prefs.transpose - 1);
        root.SnoocleP.relabelChords(); root.SnoocleP.savePrefs();
        break;
      case "Escape":
        if (document.body.classList.contains("stage")) toggleStage(P);
        break;
      default: break;
    }
  }

  root.SnoocleControls = {
    renderSectionChips: renderSectionChips,
    markActiveSection: markActiveSection,
    renderTransport: renderTransport,
    syncTransport: syncTransport,
    tickTransport: tickTransport,
    enforceLoop: enforceLoop,
    handleKey: handleKey,
    seek: seek,
    sectionStartTime: sectionStartTime,
    formatTime: formatTime,
  };
})(typeof globalThis !== "undefined" ? globalThis : this);
