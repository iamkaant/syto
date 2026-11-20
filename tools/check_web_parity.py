#!/usr/bin/env python
"""Does the browser give the same answer as ChimeraX?

That is the whole premise of this project: one core, two front ends, no
reimplementation. If it ever stops being true, everything downstream -
the handbook, the scores, the model's features - quietly forks.

So this runs the SAME receptor and the SAME pose through both, and diffs
the interaction lists term by term. It needs headless Chrome and a local
HTTP server (workers and wasm will not load from file://), and skips
rather than fails when either is missing.

Usage:  python tools/check_web_parity.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
AUTOPARTY = "/Users/ak87/ownCloud/owncloud/Calculations/TMP/AutoParty"
REC = os.path.join(AUTOPARTY, "autoparty/inputs/rec.crg.pdb")
LIG = os.path.join(AUTOPARTY,
                   "autoparty/inputs/more_opcore_enum_asp_h2o_hbond.mol2")
CHROME = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
          "/Applications/Chromium.app/Contents/MacOS/Chromium",
          "/usr/bin/google-chrome", "/usr/bin/chromium")
PORT = 8731


def find_chrome():
    for c in CHROME:
        if os.path.exists(c):
            return c
    return None


def serve(directory, port, inbox):
    """Static files, plus a POST box the page reports its result into.

    Chrome's --virtual-time-budget fast-forwards timers but does not wait
    for wasm to compile, so --dump-dom came back with an empty page every
    time. Letting the page tell us when it is done removes the guesswork.
    """
    import http.server
    import functools

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n).decode("utf-8", "replace")
            inbox.append(body)
            self.send_response(204)
            self.end_headers()

    httpd = http.server.ThreadingHTTPServer(
        ("127.0.0.1", port),
        functools.partial(Handler, directory=directory))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


# ------------------------------------------------------------- the two runs

def native_explain(res, limit=6):
    """The handbook answer for a few ligand atoms, ChimeraX-side.

    Interactions alone would not catch a drift in the handbook, the
    moiety perception or the rule that decides when pKBH+ joins pKBHX -
    all of which the page shows and all of which come from the core.
    """
    from hitpicking.handbook import Handbook
    hb = Handbook.load_or_none()
    if hb is None:
        return []
    ms = res.combined
    out = []
    for i in range(ms.n_atoms):
        if not ms.is_ligand[i] or str(ms.element[i]) in ("C", "H"):
            continue
        hit = hb.for_atom(ms, i)
        if not hit.moiety_ids:
            continue
        wanted = {"pKBHX"} | {a["quantity"] for a in hit.annotations
                              if a.get("changes_the_answer")}
        scales = [q for q in ("pKBHX", "pKBH+") if q in wanted
                  and hb.scale(q, hit.moiety_ids, same_element_only=False)]
        out.append({"atom": i, "title": hit.title,
                    "moieties": list(hit.moiety_ids), "scales": scales})
        if len(out) >= limit:
            break
    return out


def run_native():
    sys.path.insert(0, os.path.join(ROOT, "core"))
    from hitpicking.analysis import ReceptorContext, Options
    from hitpicking.io.pdb import read_pdb
    from hitpicking.io.mol2 import read_mol2_iter
    ctx = ReceptorContext(read_pdb(REC), Options())
    lig = next(iter(read_mol2_iter(LIG)))
    res = ctx.analyze(lig, options=Options())
    return normalise([i.to_dict() for i in res.interactions]), res


PAGE = """<!doctype html><meta charset=utf-8>
<pre id="log"></pre>
<script src="ui/bridge.js"></script>
<script>
const L = document.getElementById("log");
function say(s) { L.textContent += s + "\\n"; }
function report(o) {
  o.log = L.textContent;
  navigator.sendBeacon("/__result", JSON.stringify(o));
}
window.onerror = (m, u, l) => report({ok: false, error: m + " @" + l});
const b = new HPBridge.WorkerBridge("js/worker.js");
b.on("progress", (d) => say("  " + d));
(async function () {
  try {
    say("asking for the version");
    const v = await b.call("version");
    say("engine up: " + (v.handbook || "no handbook"));
    const rec = await (await fetch("__rec")).text();
    const lig = await (await fetch("__lig")).text();
    say("files fetched");
    await b.call("setup", {name: "rec.crg.pdb", text: rec});
    say("receptor perceived");
    await b.call("loadPoses", {name: "poses.mol2", text: lig, limit: 1});
    say("poses loaded");
    const r = await b.call("analyse", {index: 0});
    say("analysed");
    const explained = [];
    for (const idx of __ATOMS__) {
      const e = await b.call("explain", {index: 0, atom: idx});
      explained.push({atom: idx, title: e.title,
                      moieties: Array.from(e.moiety_ids || []),
                      scales: (e.scales || []).map((s) => s.quantity)});
    }
    say("explained " + explained.length + " atom(s)");
    report({ok: true, result: r, version: v, explained: explained});
  } catch (e) {
    report({ok: false, error: (e && e.message) || String(e)});
  }
})();
</script>"""


def run_browser(chrome, port, inbox, timeout=420):
    proc = subprocess.Popen(
        [chrome, "--headless", "--disable-gpu", "--no-sandbox",
         "--user-data-dir=" + os.path.join(ROOT, ".chrome-parity"),
         "http://127.0.0.1:%d/__parity.html" % port],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if inbox:
                return json.loads(inbox[0])
            time.sleep(0.25)
        raise AssertionError(
            "the page never reported within %ds; Pyodide may not have "
            "loaded" % timeout)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


def normalise(rows):
    """Compare what a user would see, not float noise."""
    out = []
    for r in rows:
        out.append({
            "kind": r.get("kind"),
            "quality": round(float(r.get("quality", 0)), 4),
            "distance": round(float(r.get("distance", 0)), 4),
            "lig_atoms": sorted(int(a) for a in (r.get("lig_atoms") or [])),
            "rec_atoms": sorted(int(a) for a in (r.get("rec_atoms") or [])),
            "label": r.get("label"),
            "flags": sorted(r.get("flags") or []),
        })
    out.sort(key=lambda r: (r["kind"], r["lig_atoms"], r["rec_atoms"],
                            r["distance"]))
    return out


def main():
    for p in (REC, LIG):
        if not os.path.exists(p):
            print("sample data unavailable; skipping the web parity check")
            return 0
    chrome = find_chrome()
    if chrome is None:
        print("no headless Chrome found; skipping the web parity check")
        return 0

    native, res = run_native()
    atoms = native_explain(res)
    print("ChimeraX-side core: %d interaction(s), %d atom(s) explained"
          % (len(native), len(atoms)))

    page = os.path.join(WEB, "__parity.html")
    with open(page, "w", encoding="utf-8") as fh:
        fh.write(PAGE.replace("__ATOMS__",
                              json.dumps([a["atom"] for a in atoms]))
                 .replace("</script>", """
  const el = document.createElement("pre");
  el.id = "__result";
  const wait = setInterval(function () {
    if (!window.__done) return;
    clearInterval(wait);
    el.textContent = JSON.stringify(window.__done);
    document.body.appendChild(el);
  }, 100);
</script>""", 1))
    # the fetched file paths the page asks for
    for name, src in (("__rec", REC), ("__lig", LIG)):
        with open(os.path.join(WEB, name), "w", encoding="utf-8") as fh:
            fh.write(open(src, encoding="utf-8", errors="replace").read())

    inbox = []
    httpd = serve(WEB, PORT, inbox)
    try:
        t0 = time.time()
        got = run_browser(chrome, PORT, inbox)
        secs = time.time() - t0
    finally:
        httpd.shutdown()
        for n in ("__parity.html", "__rec", "__lig"):
            try:
                os.remove(os.path.join(WEB, n))
            except OSError:
                pass

    if got.get("log"):
        for line in got["log"].strip().splitlines():
            print("   | " + line)
    if not got.get("ok"):
        print("  FAIL browser run: %s" % got.get("error"))
        return 1
    web = normalise(got["result"]["interactions"])
    print("browser-side core:   %d interaction(s)  (%.0f s incl. boot)"
          % (len(web), secs))

    ok = True
    if web == native:
        print("  OK   every interaction matches, term for term")
    else:
        ok = False

    got_atoms = got.get("explained") or []
    if atoms:
        mism = []
        for n, w in zip(atoms, got_atoms):
            if (n["title"] != w.get("title")
                    or n["scales"] != list(w.get("scales") or [])
                    or n["moieties"] != list(w.get("moieties") or [])):
                mism.append((n, w))
        if len(got_atoms) != len(atoms):
            print("  FAIL the browser explained %d atom(s), not %d"
                  % (len(got_atoms), len(atoms)))
            ok = False
        elif mism:
            ok = False
            print("  FAIL the handbook answers differ")
            for n, w in mism[:3]:
                print("    atom %d\n      ChimeraX: %s %s\n      browser:  %s %s"
                      % (n["atom"], n["title"][:46], n["scales"],
                         str(w.get("title"))[:46], w.get("scales")))
        else:
            print("  OK   the handbook agrees on %d atom(s), scales included"
                  % len(atoms))

    if ok:
        print("WEB PARITY OK")
        return 0

    print("  FAIL the two hosts disagree")
    only_n = [r for r in native if r not in web]
    only_w = [r for r in web if r not in native]
    for tag, rows in (("only in ChimeraX's core", only_n),
                      ("only in the browser", only_w)):
        for r in rows[:5]:
            print("   %s: %s %s d=%.3f q=%.3f"
                  % (tag, r["kind"], r["lig_atoms"], r["distance"],
                     r["quality"]))
    print("   (%d vs %d rows)" % (len(native), len(web)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
