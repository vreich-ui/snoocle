"use strict";
/* Chord-diagram renderer — plain SVG, ~150 lines, no dependency.
 *
 * WHY THIS EXISTS INSTEAD OF vexchords (which the master plan named):
 * vexchords 1.2.0's published minified bundle throws
 * `TypeError: A[e] is not a constructor` on first draw in current Chromium —
 * its bundled svg.js resolves element classes through a name registry that the
 * minifier mangles. Its unminified bundle works, but costs 152 KB gzipped, for
 * drawing six lines and some dots, in a PWA meant to run on a phone. Neither
 * option is acceptable, so the diagram is drawn here.
 *
 * The payoff beyond size: the boxes are themed from tokens.css like everything
 * else, and `layout()` is a pure function, so the geometry is unit-tested in
 * node with no DOM at all.
 *
 * Input is a chords-db position verbatim:
 *   {frets: [-1,3,2,0,1,0], fingers: [0,3,2,0,1,0], baseFret: 1, barres: [1]}
 * `frets` is indexed low-string-first; -1 = muted, 0 = open, N = N frets above
 * baseFret. This module owns that convention so nothing above it has to.
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.SnoocleChordBox = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  var SVG_NS = "http://www.w3.org/2000/svg";

  var GEOM = {
    width: 180,
    height: 216,
    padX: 22,
    padTop: 40,     // room for the o/x markers and the base-fret label
    padBottom: 22,  // room for finger numbers under the box
    frets: 5,
    dotR: 7,
    nutHeight: 5,
  };

  /**
   * Pure geometry: a chords-db position -> drawable primitives.
   * No DOM, no strings-as-markup — just numbers and labels, so the layout can
   * be asserted directly in tests.
   *
   * @returns {{
   *   strings: number, showNut: boolean, baseFret: number,
   *   markers: Array<{string:number, kind:"open"|"muted"}>,
   *   dots: Array<{string:number, fret:number, finger:number|null}>,
   *   barres: Array<{fret:number, fromString:number, toString:number}>
   * }}
   * `string` is 1-based counting from the LOWEST-pitched string, matching the
   * chords-db array order, and is what x() converts to a coordinate.
   */
  function layout(position, numStrings) {
    var frets = (position && position.frets) || [];
    var fingers = (position && position.fingers) || [];
    var n = numStrings || frets.length || 6;
    var baseFret = position && position.baseFret ? position.baseFret : 1;

    var markers = [];
    var dots = [];
    frets.forEach(function (fret, i) {
      var stringNo = i + 1;
      if (fret === -1) { markers.push({ string: stringNo, kind: "muted" }); return; }
      if (fret === 0) { markers.push({ string: stringNo, kind: "open" }); return; }
      dots.push({
        string: stringNo,
        fret: fret,
        finger: fingers[i] ? fingers[i] : null,
      });
    });

    // chords-db records a barre as a bare fret number. The span is whichever
    // strings are actually stopped at that fret — a barre nothing sits on is
    // dropped rather than drawn as a stray bar.
    var barres = [];
    (position && position.barres ? position.barres : []).forEach(function (fret) {
      var stopped = frets
        .map(function (f, i) { return f === fret ? i + 1 : null; })
        .filter(function (v) { return v !== null; });
      if (!stopped.length) return;
      barres.push({
        fret: fret,
        fromString: Math.min.apply(null, stopped),
        toString: Math.max.apply(null, stopped),
      });
    });

    return {
      strings: n,
      baseFret: baseFret,
      showNut: baseFret === 1,
      markers: markers,
      dots: dots,
      barres: barres,
    };
  }

  // --- drawing ------------------------------------------------------------

  function svgEl(name, attrs) {
    var node = document.createElementNS(SVG_NS, name);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    return node;
  }

  function render(position, numStrings) {
    var L = layout(position, numStrings);
    var g = GEOM;
    var boxW = g.width - g.padX * 2;
    var boxH = g.height - g.padTop - g.padBottom;
    var stringGap = boxW / (L.strings - 1);
    var fretGap = boxH / g.frets;

    // Strings run left-to-right HIGHEST-pitched last, the way a right-handed
    // player sees the neck: string 1 (low) on the left.
    function x(stringNo) { return g.padX + (stringNo - 1) * stringGap; }
    function y(fret) { return g.padTop + (fret - 0.5) * fretGap; }

    var svg = svgEl("svg", {
      viewBox: "0 0 " + g.width + " " + g.height,
      width: g.width, height: g.height,
      role: "img", "aria-label": "chord diagram",
    });

    // Frets
    for (var f = 0; f <= g.frets; f++) {
      svg.appendChild(svgEl("line", {
        x1: g.padX, x2: g.padX + boxW,
        y1: g.padTop + f * fretGap, y2: g.padTop + f * fretGap,
        stroke: "var(--text-dim)",
        "stroke-width": f === 0 && L.showNut ? g.nutHeight : 1,
        "stroke-linecap": "round",
      }));
    }
    // Strings
    for (var s = 1; s <= L.strings; s++) {
      svg.appendChild(svgEl("line", {
        x1: x(s), x2: x(s), y1: g.padTop, y2: g.padTop + boxH,
        stroke: "var(--text-dim)", "stroke-width": 1,
      }));
    }

    // Base-fret label when the shape sits up the neck.
    if (!L.showNut) {
      var label = svgEl("text", {
        x: g.padX - 8, y: g.padTop + fretGap * 0.65,
        "text-anchor": "end", fill: "var(--text-dim)",
        "font-size": "12", "font-family": "var(--font-mono)",
      });
      label.textContent = String(L.baseFret) + "fr";
      svg.appendChild(label);
    }

    // Open / muted markers above the nut.
    L.markers.forEach(function (m) {
      if (m.kind === "open") {
        svg.appendChild(svgEl("circle", {
          cx: x(m.string), cy: g.padTop - 12, r: 4.5,
          fill: "none", stroke: "var(--text)", "stroke-width": 1.5,
        }));
      } else {
        ["m -4,-4 l 8,8", "m 4,-4 l -8,8"].forEach(function (d) {
          svg.appendChild(svgEl("path", {
            d: "M " + x(m.string) + "," + (g.padTop - 12) + " " + d,
            stroke: "var(--text-dim)", "stroke-width": 1.5, "stroke-linecap": "round",
          }));
        });
      }
    });

    // Barres first, so finger dots sit on top of them.
    L.barres.forEach(function (b) {
      svg.appendChild(svgEl("rect", {
        x: x(b.fromString) - GEOM.dotR, y: y(b.fret) - GEOM.dotR,
        width: x(b.toString) - x(b.fromString) + GEOM.dotR * 2,
        height: GEOM.dotR * 2, rx: GEOM.dotR,
        fill: "var(--chord)",
      }));
    });

    L.dots.forEach(function (dot) {
      svg.appendChild(svgEl("circle", {
        cx: x(dot.string), cy: y(dot.fret), r: GEOM.dotR,
        fill: "var(--chord)",
      }));
      if (dot.finger) {
        var t = svgEl("text", {
          x: x(dot.string), y: y(dot.fret) + 4,
          "text-anchor": "middle", fill: "var(--bg0)",
          "font-size": "10", "font-weight": "600",
          "font-family": "var(--font-ui)",
        });
        t.textContent = String(dot.finger);
        svg.appendChild(t);
      }
    });

    return svg;
  }

  return { layout: layout, render: render, GEOM: GEOM };
});
