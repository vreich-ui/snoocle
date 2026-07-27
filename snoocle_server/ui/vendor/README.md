# Vendored browser libraries

Committed to the repo on purpose. Master plan §0.7: **no build step for the web
UI** — no npm at deploy time, no bundler, no CDN at runtime. What ships is what
is in this directory.

| Directory | Package | Version | Licence | Used by |
|---|---|---|---|---|
| `chords-db/` | [`@tombatossals/chords-db`](https://github.com/tombatossals/chords-db) | 0.5.1 | MIT | `ui/play/diagrams.js` — guitar + ukulele fingerings (`guitar.json`, `ukulele.json`) |

## Not vendored, and why

**vexchords**, which §3 names as the chord-box renderer. Its published
*minified* bundle (`dist/vexchords.js`, 25 KB gzipped) throws
`TypeError: A[e] is not a constructor` on first draw in current Chromium: the
svg.js it bundles resolves element classes through a name registry that the
minifier mangles. The *unminified* bundle works — and costs 152 KB gzipped, to
draw six lines and some dots, in a PWA meant to run on a phone. Neither is
acceptable, so `ui/play/chordbox.js` draws the diagram directly: ~150 lines of
plain SVG, themed from `tokens.css` like everything else, with a pure
`layout()` function that is unit-tested in node with no DOM. Verified against
every voicing of every chord in the fixture songs
(`tests_js/diagrams.test.mjs`).

**tonal.** §3 lists it for client-side transpose and capo maths. It is not here
because nothing needs it: the chord vocabulary the server stores is already
normalized (`schema/song.py` rejects anything that isn't a sounding-harmony
symbol), so `ui/play/theory.js` does the transform in ~40 lines and — the part
that actually matters — keeps enharmonic spelling under our control. A song
written in Eb transposed by +2 renders F, not E#, because `theory.js` follows
the source symbol's accidental. A general-purpose library respells by its own
rules. If a later task needs real key/Roman-numeral analysis (G2's chord-
function colouring is the obvious one), vendor `tonal`'s `browser/tonal.min.js`
here then — it is a single 43 KB UMD file with no dependencies.

**The YouTube IFrame Player API** is loaded from `youtube.com` at runtime. It is
not a library we could vendor (it must talk to YouTube's own player), and C2
names it explicitly. It is the only external runtime resource in the player.

## Refreshing a library

```sh
npm pack @tombatossals/chords-db
tar xzf <tarball>
cp package/lib/*.json snoocle_server/ui/vendor/chords-db/
```

Then re-run `python -m pytest tests/test_player_ui.py`, which asserts these
files are served and that `chords-db` still contains the chords the diagram
fallback ladder relies on.
