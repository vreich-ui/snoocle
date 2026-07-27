/* Snoocle player service worker (master plan C5).
 *
 * Two caches, two policies:
 *
 *   SHELL  — the app's own files (HTML/CSS/JS/vendor JSON). Cache-first after
 *            the install precache, so opening the player offline shows the
 *            library instead of the browser's dinosaur.
 *   DATA   — song JSON from /v1/songs*. Network-first with a cache fallback,
 *            so an online reload always gets the current version and an
 *            offline reload still shows the sheets you last opened.
 *
 * Video is deliberately NOT cached — it can't be (YouTube), and pretending
 * otherwise would be worse than the honest "no video offline" banner the page
 * shows when `navigator.onLine` is false.
 *
 * Bump CACHE_VERSION when shell assets change; the activate handler drops
 * every older cache in one pass.
 */

var CACHE_VERSION = "v1";
var SHELL_CACHE = "snoocle-shell-" + CACHE_VERSION;
var DATA_CACHE = "snoocle-data-" + CACHE_VERSION;

var SHELL_ASSETS = [
  "./",
  "./index.html",
  "./player.css",
  "./player.js",
  "./controls.js",
  "./diagrams.js",
  "./theory.js",
  "./timedscroll.js",
  "./chordbox.js",
  "./manifest.webmanifest",
  "./icons/icon.svg",
  "../tokens.css",
  "../common.js",
  "../vendor/chords-db/guitar.json",
  "../vendor/chords-db/ukulele.json",
];

self.addEventListener("install", function (event) {
  event.waitUntil(
    caches.open(SHELL_CACHE).then(function (cache) {
      // addAll is atomic — one 404 would abort the whole install, so each
      // asset is added individually and a miss is logged, not fatal.
      return Promise.all(SHELL_ASSETS.map(function (url) {
        return cache.add(url).catch(function () { /* optional asset */ });
      }));
    }).then(function () { return self.skipWaiting(); })
  );
});

self.addEventListener("activate", function (event) {
  event.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.map(function (key) {
        if (key !== SHELL_CACHE && key !== DATA_CACHE) return caches.delete(key);
        return null;
      }));
    }).then(function () { return self.clients.claim(); })
  );
});

self.addEventListener("fetch", function (event) {
  var req = event.request;
  if (req.method !== "GET") return;

  var url = new URL(req.url);
  if (url.origin !== self.location.origin) return; // YouTube, thumbnails: pass through

  if (url.pathname.indexOf("/v1/songs") === 0) {
    event.respondWith(networkFirst(req));
    return;
  }
  if (url.pathname.indexOf("/ui/") === 0) {
    event.respondWith(cacheFirst(req));
  }
});

function cacheFirst(req) {
  return caches.match(req).then(function (hit) {
    if (hit) return hit;
    return fetch(req).then(function (res) {
      if (res && res.ok) {
        var copy = res.clone();
        caches.open(SHELL_CACHE).then(function (c) { c.put(req, copy); });
      }
      return res;
    });
  });
}

function networkFirst(req) {
  return fetch(req).then(function (res) {
    if (res && res.ok) {
      var copy = res.clone();
      caches.open(DATA_CACHE).then(function (c) { c.put(req, copy); });
    }
    return res;
  }).catch(function () {
    return caches.match(req).then(function (hit) {
      return hit || new Response(
        JSON.stringify({ detail: "offline and not cached" }),
        { status: 503, headers: { "Content-Type": "application/json" } }
      );
    });
  });
}
