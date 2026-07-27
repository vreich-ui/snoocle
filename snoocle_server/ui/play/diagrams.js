"use strict";
/* Chord diagrams on tap (master plan C3).
 *
 * Data: vendored chords-db (MIT) — a JSON database of real guitar and ukulele
 * fingerings, several voicings per chord, committed under ui/vendor/. Nothing
 * is fetched from a CDN. Rendering is chordbox.js, ours — see the note at the
 * top of that file for why not vexchords.
 *
 * The diagram always shows the chord as it is PRINTED (after the user's
 * transpose/capo display transform), while the caption keeps naming the
 * SOUNDING harmony — so the rule that capo never changes a chord's identity
 * stays visible at the exact moment it would otherwise get confusing.
 *
 * A chord the database doesn't have falls back down the simplification ladder
 * (Cmaj9 -> Cmaj7 -> C) and the popover says "approx" so nobody mistakes the
 * substitute for the real voicing.
 */
(function (root) {
  var DB = { guitar: null, ukulele: null };
  var loading = {};

  // chords-db spells sharp keys "Csharp"; its own key list uses "C#".
  function dbKeyToken(key) { return key.replace("#", "sharp"); }

  function loadDb(instrument) {
    if (DB[instrument]) return Promise.resolve(DB[instrument]);
    if (loading[instrument]) return loading[instrument];
    loading[instrument] = fetch("../vendor/chords-db/" + instrument + ".json")
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (json) { DB[instrument] = json; return json; })
      .catch(function () { return null; });
    return loading[instrument];
  }

  /**
   * Find voicings for a printed chord symbol.
   * @returns {{positions: Array, symbol: string, approx: boolean}|null}
   */
  function lookup(db, symbol) {
    if (!db) return null;
    var candidates = root.SnoocleTheory.simplifications(symbol);
    for (var i = 0; i < candidates.length; i++) {
      var ks = root.SnoocleTheory.dbKeySuffix(candidates[i]);
      if (!ks) continue;
      var entries = db.chords[dbKeyToken(ks.key)];
      if (!entries) continue;
      for (var j = 0; j < entries.length; j++) {
        if (entries[j].suffix === ks.suffix && entries[j].positions.length) {
          return {
            positions: entries[j].positions,
            symbol: candidates[i],
            approx: i > 0,
          };
        }
      }
    }
    return null;
  }

  function drawInto(host, position, numStrings) {
    host.innerHTML = "";
    host.appendChild(root.SnoocleChordBox.render(position, numStrings));
  }

  /**
   * Open the diagram popover for a SOUNDING chord.
   * @param sounding the stored chord symbol (never mutated)
   * @param prefs    {transpose, capo, instrument?}
   */
  async function open(sounding, prefs) {
    prefs = prefs || {};
    var instrument = prefs.instrument === "ukulele" ? "ukulele" : "guitar";
    var shown = root.SnoocleTheory.displayChord(sounding, prefs.transpose, prefs.capo);

    var stage = document.createElement("div");
    stage.className = "diagram-stage";
    var pager = document.createElement("div");
    pager.className = "voicing-pager";
    var dots = document.createElement("div");
    dots.className = "voicing-dots";
    var badge = document.createElement("span");
    badge.className = "badge approx";
    badge.textContent = "approx";
    badge.hidden = true;

    var title = document.createElement("h3");
    title.textContent = shown;
    var caption = document.createElement("div");
    caption.className = "caption";
    caption.textContent = root.SnoocleTheory.caption(sounding, prefs.transpose, prefs.capo);

    var pop = buildPopover([title, caption, badge, stage, pager, dots]);

    stage.textContent = "…";
    var db = await loadDb(instrument);
    var found = lookup(db, shown);
    if (!found) {
      stage.textContent = "No diagram for " + shown + ".";
      return pop;
    }
    badge.hidden = !found.approx;
    if (found.approx) {
      caption.textContent += " · diagram shown for " + found.symbol;
    }

    var numStrings = (db.main && db.main.strings) || 6;
    var index = 0;
    function render() {
      drawInto(stage, found.positions[index], numStrings);
      dots.innerHTML = "";
      found.positions.forEach(function (_, i) {
        var dot = document.createElement("button");
        dot.type = "button";
        dot.className = "voicing-dot" + (i === index ? " active" : "");
        dot.setAttribute("aria-label", "Voicing " + (i + 1));
        dot.addEventListener("click", function () { index = i; render(); });
        dots.appendChild(dot);
      });
      pager.innerHTML = "";
      pager.appendChild(pagerBtn("‹", function () {
        index = (index - 1 + found.positions.length) % found.positions.length;
        render();
      }));
      var label = document.createElement("span");
      label.className = "muted";
      label.textContent = (index + 1) + " / " + found.positions.length;
      pager.appendChild(label);
      pager.appendChild(pagerBtn("›", function () {
        index = (index + 1) % found.positions.length;
        render();
      }));
    }
    render();
    return pop;
  }

  function pagerBtn(label, onClick) {
    var b = document.createElement("button");
    b.type = "button";
    b.className = "btn secondary icon-btn";
    b.textContent = label;
    b.addEventListener("click", onClick);
    return b;
  }

  function buildPopover(children) {
    var backdrop = document.createElement("div");
    backdrop.className = "popover-backdrop";
    var pop = document.createElement("div");
    pop.className = "popover";
    pop.setAttribute("role", "dialog");
    pop.setAttribute("aria-modal", "true");
    children.forEach(function (c) { pop.appendChild(c); });

    var actions = document.createElement("div");
    actions.className = "actions";
    var close = document.createElement("button");
    close.type = "button";
    close.className = "btn secondary";
    close.textContent = "Close";
    close.addEventListener("click", function () { backdrop.remove(); });
    actions.appendChild(close);
    pop.appendChild(actions);

    backdrop.appendChild(pop);
    backdrop.addEventListener("click", function (e) {
      if (e.target === backdrop) backdrop.remove();
    });
    document.addEventListener("keydown", function esc(e) {
      if (e.key === "Escape") { backdrop.remove(); document.removeEventListener("keydown", esc); }
    });
    document.getElementById("popover-root").appendChild(backdrop);
    close.focus();
    return backdrop;
  }

  root.SnoocleDiagrams = {
    open: open,
    lookup: lookup,
    buildPopover: buildPopover,
  };
})(typeof globalThis !== "undefined" ? globalThis : this);
