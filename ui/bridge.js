// One interface, two hosts.
//
// The panels under ui/ never learn whether they are inside ChimeraX or a
// browser tab. ChimeraX answers over its custom URL scheme; the web page
// answers from a Pyodide worker. Both satisfy this:
//
//     HPBridge.call(method, args) -> Promise
//     HPBridge.on(event, cb)
//
// Keeping the shape identical is what lets a panel be written once, and
// what makes a disagreement between the two a bug rather than a feature.

(function (global) {
  function WorkerBridge(url) {
    // Pyodide 314 only runs in a module worker; a classic one refuses
    // to start with "Classic web workers are not supported".
    this.worker = new Worker(url, { type: "module" });
    this.pending = new Map();
    this.listeners = new Map();
    this.nextId = 1;
    this.worker.onmessage = (ev) => {
      const m = ev.data || {};
      if (m.event) {
        (this.listeners.get(m.event) || []).forEach((cb) => cb(m.data));
        return;
      }
      const p = this.pending.get(m.id);
      if (!p) return;
      this.pending.delete(m.id);
      m.ok ? p.resolve(m.result) : p.reject(new Error(m.error));
    };
    this.worker.onerror = (e) => {
      // a worker that dies takes every outstanding call with it; failing
      // them loudly beats a page that just stops responding
      const err = new Error(e.message || "the analysis worker stopped");
      this.pending.forEach((p) => p.reject(err));
      this.pending.clear();
    };
  }

  WorkerBridge.prototype.call = function (method, args) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, method, args: args || {} });
    });
  };

  WorkerBridge.prototype.on = function (event, cb) {
    if (!this.listeners.has(event)) this.listeners.set(event, []);
    this.listeners.get(event).push(cb);
  };

  global.HPBridge = { WorkerBridge: WorkerBridge };
})(typeof self !== "undefined" ? self : this);
