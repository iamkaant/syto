// The shared core, running in a Web Worker so the page never blocks.
//
// Pyodide and numpy are vendored under web/vendor, and the hitpicking
// package travels as a zip built by tools/build_web_core.py, so the page
// needs no network at all. The worker speaks one message shape:
//
//     {id, method, args}  ->  {id, ok, result}  |  {id, ok:false, error}
//
// which is what ui/bridge.js expects from either host. Nothing here
// decides anything about chemistry; it loads a file and calls engine.py.

// A MODULE worker: Pyodide 314 removed classic-worker support, so
// importScripts is gone and this must be loaded with {type: "module"}.
import { loadPyodide } from "../vendor/pyodide/pyodide.mjs";

let pyodide = null;
let engine = null;

const VENDOR = "../vendor";

async function boot() {
  if (pyodide) return pyodide;
  pyodide = await loadPyodide({ indexURL: VENDOR + "/pyodide/" });

  // numpy from the vendored wheel, not from the network
  await pyodide.loadPackage(new URL(
    VENDOR + "/pyodide/numpy-2.4.6-cp314-cp314-pyemscripten_2026_0_wasm32.whl",
    import.meta.url).href);

  const zipUrl = new URL(VENDOR + "/hitpicking.zip", import.meta.url);
  const zip = await (await fetch(zipUrl)).arrayBuffer();
  pyodide.unpackArchive(zip, "zip", { extractDir: "/lib" });
  pyodide.runPython("import sys; sys.path.insert(0, '/lib')");

  const src = await (await fetch(new URL("engine.py", import.meta.url))).text();
  pyodide.FS.writeFile("/lib/hp_engine.py", src);
  engine = pyodide.pyimport("hp_engine");
  return pyodide;
}

function toJs(value) {
  // engine.py returns JSON strings on purpose: one crossing, no proxies to
  // leak, and the exact same bytes a parity check can diff. Depending on
  // the call, Pyodide hands that back either as a JS string or as a
  // PyProxy wrapping one, so both are unwrapped here - and the proxy is
  // destroyed, because a leaked one keeps the Python object alive for the
  // life of the page.
  let text = value;
  let proxy = null;
  if (value && typeof value === "object" && typeof value.toString === "function"
      && typeof value.destroy === "function") {
    proxy = value;
    text = value.toString();
  }
  try {
    if (typeof text === "string") {
      const parsed = JSON.parse(text);
      return parsed;
    }
    return text;
  } catch (e) {
    // Never swallow this: a result that silently stays a string looks like
    // a harness bug for an hour before anyone suspects the bridge.
    throw new Error("the engine returned something that is not JSON: "
                    + String(text).slice(0, 200));
  } finally {
    if (proxy) proxy.destroy();
  }
}

const METHODS = {
  async version() { return toJs(engine.version()); },
  async setup(a) { return toJs(engine.setup(a.name, a.text)); },
  async loadPoses(a) {
    return toJs(engine.load_poses(a.name, a.text, a.limit || null));
  },
  async analyse(a) {
    return toJs(engine.analyse(a.index || 0,
                               a.enabled ? pyodide.toPy(a.enabled) : null,
                               a.strain !== false));
  },
  async explain(a) { return toJs(engine.explain(a.index || 0, a.atom)); },
  async scene() { return toJs(engine.scene()); },
  async featurise(a) {
    return toJs(engine.featurise(a.start || 0, a.count || 1,
                                 a.strain !== false));
  },
  async alStatus() { return toJs(engine.al_status()); },
  async grade(a) { return toJs(engine.grade(a.index, a.value || null)); },
  async train(a) {
    return toJs(engine.train(a.epochs || 100, a.batch || null,
                             a.patience || 20));
  },
  async predictions(a) {
    return toJs(engine.predictions(a.strategy || "uncertainty",
                                   a.count || 25));
  },
  async saveParty() { return toJs(engine.save_party()); },
  async loadParty(a) { return toJs(engine.load_party(a.text)); },
  async explainAt(a) {
    return toJs(engine.explain_at(a.x, a.y, a.z, a.tolerance || 0.05));
  },
};

self.onmessage = async (ev) => {
  const { id, method, args } = ev.data || {};
  try {
    await boot();
    const fn = METHODS[method];
    if (!fn) throw new Error("no such method: " + method);
    self.postMessage({ id, ok: true, result: await fn(args || {}) });
  } catch (err) {
    self.postMessage({
      id, ok: false,
      error: (err && err.message) ? err.message : String(err),
    });
  }
};
