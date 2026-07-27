"use strict";
/* Snoocle shared browser helpers — the ONLY code the admin (/ui/) and the
 * play-along player (/ui/play/) have in common.
 *
 * Deliberately tiny. Per the master plan's Phase C intro this file holds
 * `api`, `apiJson`, `el` and `clear` and nothing else — plus the two private
 * helpers `api` cannot work without (the 401 token prompt and the button it
 * renders). Everything else stays where it lives; this is an extraction, not
 * a refactor.
 *
 * Loaded as a classic script (no modules, no build step), so it publishes onto
 * `window` by plain top-level function declarations.
 */

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
    if (c === null || c === undefined) return;
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
// API access
//
// Same-origin fetch with the bearer token this browser has stored. A 401
// prompts once for the token (SNOOCLE_API_TOKEN), stores it locally, and
// retries the request exactly once.
// ---------------------------------------------------------------------------

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
