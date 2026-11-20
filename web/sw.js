// Offline support: visit the page once, use it forever without a server.
//
// This is what makes "I want it locally" work without a local server.
// file:// cannot be made to work - the browser blocks both workers and
// fetch from a null origin, which is a security rule and not something a
// page can opt out of. But a service worker caches the whole thing on
// first visit, so afterwards the hosted URL opens with no network at all:
// on a plane, in a meeting room with no wifi, wherever.
//
// Cache-first for everything, because every byte here is pinned to a
// version: Pyodide 314.0.8, numpy 2.4.6, Mol* 5.13.1 and a core zip built
// from this repo. Nothing is fetched that is expected to change under us,
// so there is no staleness to reason about - a new deploy bumps CACHE and
// the old one is dropped whole.

const CACHE = "hitpicking-v1";

// The big, pinned runtime. Pre-cached on install so the first offline use
// cannot find a hole; everything else is cached as it is asked for.
const CORE = [
  "./",
  "./index.html",
  "./js/viewer.js",
  "./js/worker.js",
  "./js/engine.py",
  "./ui/bridge.js",
  "./ui/panel.html",
  "./vendor/hitpicking.zip",
  "./vendor/molstar/molstar.js",
  "./vendor/molstar/molstar.css",
  "./vendor/pyodide/pyodide.mjs",
  "./vendor/pyodide/pyodide.asm.mjs",
  "./vendor/pyodide/pyodide.asm.wasm",
  "./vendor/pyodide/python_stdlib.zip",
  "./vendor/pyodide/pyodide-lock.json",
  "./vendor/pyodide/numpy-2.4.6-cp314-cp314-pyemscripten_2026_0_wasm32.whl"
];

self.addEventListener("install", (e) => {
  e.waitUntil((async () => {
    const c = await caches.open(CACHE);
    // addAll fails the whole install if one entry 404s, which would leave
    // no worker at all. Each is added on its own so a renamed vendor file
    // costs that one file, not offline support.
    await Promise.all(CORE.map((u) =>
      c.add(new Request(u, {cache: "reload"})).catch(() => null)));
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names.filter((n) => n !== CACHE)
                           .map((n) => caches.delete(n)));
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== self.location.origin) return;
  e.respondWith((async () => {
    const hit = await caches.match(e.request, {ignoreSearch: false});
    if (hit) return hit;
    try {
      const res = await fetch(e.request);
      if (res && res.ok && res.type === "basic") {
        const c = await caches.open(CACHE);
        c.put(e.request, res.clone());
      }
      return res;
    } catch (err) {
      // Offline and not cached: let the page report it rather than
      // showing the browser's own error, which says nothing useful.
      const fallback = await caches.match("./index.html");
      if (fallback && e.request.mode === "navigate") return fallback;
      throw err;
    }
  })());
});
