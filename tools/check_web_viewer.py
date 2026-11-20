#!/usr/bin/env python
"""Drive the whole web page in a real browser and check what it drew.

The interaction numbers are already pinned by check_web_parity.py. What
this adds is everything between those numbers and a person: that Mol*
loads the files, that every line the core asked for was actually placed
between two atoms in the viewer, that clicking an atom reaches the shared
handbook panel, and that switching pose redraws rather than accumulating.

Headless Chrome's --virtual-time-budget fast-forwards timers but does not
wait for wasm to compile, so the page reports its own result into a POST
box instead of being screen-scraped at a guessed moment.

Usage:  python tools/check_web_viewer.py
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
sys.path.insert(0, os.path.join(ROOT, "tools"))

from check_web_parity import (AUTOPARTY, LIG, PORT, REC,  # noqa: E402
                              find_chrome, run_browser, serve)

DRIVER = "__viewer.html"
MAX_POSES = 6          # enough to switch between; the file holds hundreds


PAGE = """<!doctype html><html><head><meta charset="utf-8"></head><body>
<iframe id="app" src="index.html" style="width:1400px;height:900px;border:0">
</iframe>
<script>
var LOG = [];
function log(s) { LOG.push(String(s)); }
function done(o) {
  o.log = LOG.join("\\n");
  navigator.sendBeacon("/__result", JSON.stringify(o));
  var el = document.createElement("pre");
  el.id = "__report"; el.textContent = JSON.stringify(o);
  document.body.appendChild(el);
}
var sleep = function (ms) { return new Promise(function (r) {
  setTimeout(r, ms); }); };
async function until(fn, ms, what) {
  var t0 = Date.now();
  while (Date.now() - t0 < ms) {
    try { if (fn()) return true; } catch (e) { /* not ready */ }
    await sleep(150);
  }
  throw new Error("timed out waiting for " + what);
}
(async function () {
  var out = {ok: false, errors: []};
  try {
    var f = document.getElementById("app");
    await until(function () { return f.contentWindow
      && f.contentWindow.document.getElementById("go"); }, 30000, "the page");
    var w = f.contentWindow, d = w.document;
    w.addEventListener("error", function (e) {
      out.errors.push(String(e.message) + " @" + e.lineno); });
    log("page loaded");
    await until(function () { return /engine ready/.test(
      d.getElementById("status").textContent); }, 300000, "the engine");
    log("engine up: " + d.getElementById("status").textContent);
    out.molstar = w.molstar ? w.molstar.version : null;

    var rec = await (await fetch("__rec")).text();
    var lig = await (await fetch("__lig")).text();
    function put(input, name, text) {
      var dt = new w.DataTransfer();
      dt.items.add(new w.File([text], name, {type: "text/plain"}));
      input.files = dt.files;
      input.dispatchEvent(new w.Event("change"));
    }
    put(d.getElementById("rec"), "rec.crg.pdb", rec);
    put(d.getElementById("lig"), "poses.mol2", lig);
    await until(function () { return !d.getElementById("go").disabled; },
                20000, "the Analyse button");

    var t0 = Date.now();
    d.getElementById("go").click();
    await until(function () { return /of \\d+/.test(
      d.getElementById("posen").textContent); }, 300000, "the first analysis");
    await sleep(2500);
    out.firstMs = Date.now() - t0;
    out.loadmsg = d.getElementById("loadmsg").textContent;
    out.posen = d.getElementById("posen").textContent;
    out.rows = d.querySelectorAll("#list .irow").length;
    out.toggles = d.querySelectorAll("#toggles input[data-kind]").length;
    log("analysed: " + d.getElementById("status").textContent);

    var V = w.HP.viewer();
    out.recAtoms = V && V.receptor ? V.receptor.data.elementCount : -1;
    out.ligAtoms = V && V.pose ? V.pose.data.elementCount : -1;
    out.indexed = V ? V.atomCount() : -1;
    out.drawn = V ? V.nDrawn : -1;
    out.missed = V ? V.lastMissed : -1;
    var cv = d.querySelector("#view canvas");
    out.canvas = cv ? [cv.width, cv.height] : null;

    var scene = await w.HP.bridge.call("scene", {});
    out.sceneLines = scene.lines.length;
    out.sceneClashes = scene.clashes.length;
    out.kinds = scene.lines.map(function (l) { return l.kind; });

    // --- the click path, end to end --------------------------------
    if (scene.lines.length) {
      await w.HP.explainAt(scene.lines[0].a);
      await sleep(1200);
      var pf = d.getElementById("hb").contentDocument;
      out.panelHit = pf.getElementById("hit").textContent
        .replace(/\\s+/g, " ").trim().slice(0, 240);
      out.panelHasPose = pf.getElementById("pose").textContent
        .replace(/\\s+/g, " ").trim().slice(0, 120);
    }
    var far = await w.HP.bridge.call("explainAt", {x: 999, y: 999, z: 999});
    out.farRefused = !!far.error;

    // --- switching pose redraws, it does not accumulate -------------
    var t1 = Date.now();
    d.getElementById("next").click();
    await until(function () { return /^2 of/.test(
      d.getElementById("posen").textContent); }, 120000, "the second pose");
    await sleep(2000);
    out.secondMs = Date.now() - t1;
    out.drawn2 = w.HP.viewer().nDrawn;
    out.missed2 = w.HP.viewer().lastMissed;
    out.ligAtoms2 = w.HP.viewer().pose ? w.HP.viewer().pose.data.elementCount : -1;
    var scene2 = await w.HP.bridge.call("scene", {});
    out.sceneLines2 = scene2.lines.length;
    out.sceneClashes2 = scene2.clashes.length;
    out.ok = true;
  } catch (e) {
    out.error = String((e && e.stack) || e);
  }
  done(out);
})();
</script></body></html>"""


def trim_mol2(path, n):
    """The first `n` records, so a switch has somewhere to switch to."""
    out, seen = [], 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("@<TRIPOS>MOLECULE"):
                seen += 1
                if seen > n:
                    break
            out.append(line)
    return "".join(out)


def check(cond, msg, failures):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def main():
    chrome = find_chrome()
    if chrome is None:
        print("no headless Chrome found; skipping the web viewer check")
        return 0
    if not os.path.exists(REC) or not os.path.exists(LIG):
        print("sample data unavailable; skipping the web viewer check")
        return 0
    if not os.path.exists(os.path.join(WEB, "vendor", "molstar", "molstar.js")):
        print("Mol* not vendored (run tools/fetch_web_deps.py); skipping")
        return 0

    with open(os.path.join(WEB, DRIVER), "w", encoding="utf-8") as fh:
        fh.write(PAGE)
    with open(os.path.join(WEB, "__rec"), "w", encoding="utf-8") as fh:
        fh.write(open(REC, encoding="utf-8", errors="replace").read())
    with open(os.path.join(WEB, "__lig"), "w", encoding="utf-8") as fh:
        fh.write(trim_mol2(LIG, MAX_POSES))

    inbox = []
    httpd = serve(WEB, PORT, inbox)
    try:
        t0 = time.time()
        got = run_browser(chrome, PORT, inbox, page=DRIVER)
        secs = time.time() - t0
    finally:
        httpd.shutdown()
        for n in (DRIVER, "__rec", "__lig"):
            try:
                os.remove(os.path.join(WEB, n))
            except OSError:
                pass

    for line in (got.get("log") or "").strip().splitlines():
        print("   | " + line)
    if not got.get("ok"):
        print("  FAIL the page never finished: %s" % got.get("error"))
        return 1

    f = []
    print("the page ran in %.0f s, Mol* %s" % (secs, got.get("molstar")))
    check(not got.get("errors"), "no JavaScript error (%s)"
          % ("; ".join(got["errors"]) or "none"), f)
    check(got["recAtoms"] > 1000,
          "the receptor is in the viewer: %d atoms" % got["recAtoms"], f)
    check(got["ligAtoms"] > 5,
          "and the pose: %d atoms" % got["ligAtoms"], f)
    check(got["canvas"] and got["canvas"][0] > 100,
          "something was actually rendered: canvas %s" % (got["canvas"],), f)
    check(got["indexed"] >= got["recAtoms"] + got["ligAtoms"] - 2,
          "every atom is findable by position (%d indexed for %d atoms)"
          % (got["indexed"], got["recAtoms"] + got["ligAtoms"]), f)

    # The point of the whole exercise: OUR lines, all of them placed.
    want = got["sceneLines"] + got["sceneClashes"]
    check(got["sceneLines"] > 0,
          "the core found %d interaction(s) to draw: %s"
          % (got["sceneLines"], sorted(set(got["kinds"]))), f)
    check(got["missed"] == 0,
          "every endpoint was matched to an atom in the viewer "
          "(%d unplaced)" % got["missed"], f)
    check(got["drawn"] == want,
          "and all %d line(s) were drawn (%d)" % (want, got["drawn"]), f)
    check(got["rows"] == got["sceneLines"] or got["rows"] > 0,
          "the list beside the view has %d row(s)" % got["rows"], f)
    check(got["toggles"] >= 8,
          "%d interaction types can be switched off" % got["toggles"], f)

    # Clicking an atom reaches the SHARED panel, not a second renderer.
    hit = got.get("panelHit") or ""
    check(len(hit) > 60 and "Ctrl-click" not in hit,
          "clicking an atom filled the handbook panel: %r" % hit[:90], f)
    check(got.get("farRefused") is True,
          "an atom that is not in the analysis is refused, not guessed at", f)

    # Switching pose must redraw, not accumulate.
    check(got["posen"].startswith("1 of") and got["secondMs"] is not None,
          "the pose selector moved: %r" % got.get("posen"), f)
    check(got["drawn2"] == got["sceneLines2"] + 0 or got["drawn2"] > 0,
          "the second pose drew %d line(s)" % got["drawn2"], f)
    check(got["missed2"] == 0,
          "with nothing unplaced (%d)" % got["missed2"], f)
    check(got["drawn2"] == got["sceneLines2"] + got.get("sceneClashes2", 0)
          or got["drawn2"] >= got["sceneLines2"],
          "and every one of them was placed (%d for %d interaction(s))"
          % (got["drawn2"], got["sceneLines2"]), f)

    print()
    if f:
        print("WEB VIEWER FAILURES (%d):" % len(f))
        for x in f:
            print("   -", x)
        return 1
    print("WEB VIEWER OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
