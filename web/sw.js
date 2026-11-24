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

// The cache name carries the build hash, written by
// tools/build_web_core.py. This is not tidiness: a service worker that
// keeps its name across a deploy goes on serving the previous build from
// an origin that looks identical, and the symptom is a method that is
// plainly in the file coming back as "no such method". It cost an hour
// once; it is not going to be a thing anyone has to remember.
try { self.importScripts("./build.js"); } catch (e) { /* dev tree */ }
const CACHE = "hitpicking-" + (self.__BUILD__ || "dev");

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
  "./build.js",
  "./logo.png",
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

// Two strategies, and the split matters more than it looks.
//
// vendor/ is pinned: Pyodide 314.0.8, numpy 2.4.6, Mol* 5.13.1. Those
// bytes never change under a given URL, they are 22 MB, and they are the
// whole reason for caching. Cache-first.
//
// Everything else is OUR code, and it changes whenever the tool does.
// Cache-first on that was a real defect, not a tuning choice: it served
// an old index.html against a newer worker.js, so the page threw on an
// element the new script expected and the old markup did not have. The
// cache name could not save it either - it was derived from the core
// zip, which does not change when index.html does. Network-first here,
// falling back to the cache, so being online always means being current
// and being offline still works.
function isPinned(url) {
  return url.pathname.includes("/vendor/");
}

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== self.location.origin) return;

  if (isPinned(url)) {
    e.respondWith((async () => {
      const hit = await caches.match(e.request);
      if (hit) return hit;
      const res = await fetch(e.request);
      if (res && res.ok && res.type === "basic") {
        (await caches.open(CACHE)).put(e.request, res.clone());
      }
      return res;
    })());
    return;
  }

  e.respondWith((async () => {
    try {
      const res = await fetch(e.request, {cache: "no-store"});
      if (res && res.ok && res.type === "basic") {
        (await caches.open(CACHE)).put(e.request, res.clone());
      }
      return res;
    } catch (err) {
      const hit = await caches.match(e.request);
      if (hit) return hit;
      if (e.request.mode === "navigate") {
        const page = await caches.match("./index.html");
        if (page) return page;
      }
      throw err;
    }
  })());
});
