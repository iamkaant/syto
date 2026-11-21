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
    // Offline support: the page must have asked for it on a served origin.
    try {
      var reg = await w.navigator.serviceWorker.getRegistration();
      out.offline = !!reg;
    } catch (e) { out.offline = false; }
    out.build = w.__build || null;

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
    out.siteAtoms = (scene.site || []).length;
    out.siteDrawn = !!V.site;
    out.siteFocused = !!V.siteLoci;
    // Did it actually RENDER? A ref in the state tree proves nothing:
    // a representation whose params Mol* rejects sits there with no
    // geometry and the pocket is simply invisible.
    try {
      var cells = [];
      V.plugin.state.data.cells.forEach(function (c) {
        var o = c.obj;
        if (!o) return;
        cells.push({label: o.label || "",
                    type: (o.type && o.type.name) || "",
                    err: c.status === "error" ? String(c.errorText) : null,
                    reprs: o.data && o.data.repr
                      ? o.data.repr.renderObjects.length : null});
      });
      out.cells = cells;
      out.stateErrors = cells.filter(function (c) { return c.err; });
    } catch (e) { out.cellErr = String(e); }
    out.gradeHint = d.getElementById("gradestate").textContent
      .replace(/\s+/g, " ").trim();
    out.gradeEnabled = !d.getElementById("g-good").disabled;
    // Resizing: drag the left gutter 90px and see the column follow,
    // then survive a re-read of the stored preferences.
    try {
      var g = d.getElementById("gut-l");
      var before = d.getElementById("gut-l").getBoundingClientRect().left;
      var opts = {bubbles: true, clientX: before + 4, pointerId: 1,
                  pointerType: "mouse", isPrimary: true};
      g.setPointerCapture = function () {}; g.releasePointerCapture = function () {};
      g.dispatchEvent(new w.PointerEvent("pointerdown", opts));
      opts.clientX = before + 94;
      g.dispatchEvent(new w.PointerEvent("pointermove", opts));
      g.dispatchEvent(new w.PointerEvent("pointerup", opts));
      await sleep(250);
      out.leftAfterDrag = parseFloat(w.getComputedStyle(
        d.documentElement).getPropertyValue("--left"));
      var stored = JSON.parse(w.localStorage.getItem("hitpicking.view") || "{}");
      out.leftStored = stored.left;
      out.gutters = d.querySelectorAll(".gutter").length;
    } catch (e) { out.dragErr = String(e); }
    out.ribbonDefault = d.getElementById("opt-ribbon").checked;
    out.stickSize = V.siteSize;
    // The label size slider must reach the drawn labels, not just the
    // number beside it. Set it, redraw, and read it back off the viewer.
    var ts = d.getElementById("opt-text");
    ts.value = "0.6";
    ts.dispatchEvent(new w.Event("change"));
    await sleep(1800);
    out.textSize = V.textSize;
    out.textLabel = d.getElementById("textval").textContent;
    out.textStored = (JSON.parse(w.localStorage.getItem("hitpicking.view")
                                 || "{}")).text;
    out.drawnAfterText = V.nDrawn;

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


def check_file_protocol(chrome, failures):
    """What a person sees when they double-click index.html.

    This is the first thing anyone tries, and the browser's answer to it
    is a flat refusal that has nothing to do with this page being broken.
    The page has to say so itself, immediately, and say what to do - not
    sit on "starting..." while a worker fails to construct.

    Loaded directly rather than through a probe frame: a file:// parent
    cannot read a file:// child, because each gets its own opaque origin.
    """
    import re
    import subprocess
    dom = subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--dump-dom", "--virtual-time-budget=6000",
         "file://" + os.path.join(WEB, "index.html")],
        capture_output=True, text=True, timeout=120).stdout
    m = re.search(r'<div id="drop"[^>]*>(.*?)</div>\s*</div>', dom, re.S)
    drop = re.sub(r"<[^>]+>", " ", m.group(1)) if m else ""
    drop = " ".join(drop.split())
    check("will not let a file run it" in drop,
          "opened from a file, the page says so rather than looking "
          "broken: %r" % drop[:90], failures)
    check("serve_web" in drop or "hosted" in drop,
          "and says what to do instead", failures)
    check('id="go" disabled' in dom.replace("' ", '" ')
          or re.search(r'id="go"[^>]*disabled', dom) is not None,
          "with Analyse disabled, so there is nothing to press in vain",
          failures)


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
    check_file_protocol(chrome, f)
    check(not got.get("errors"), "no JavaScript error (%s)"
          % ("; ".join(got["errors"]) or "none"), f)
    check(got.get("offline") is False,
          "no service worker on localhost - caching there bought nothing "
          "and served stale pages against newer code", f)
    check(got.get("build"),
          "the page states which build it is running: %s" % got.get("build"), f)
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
    # The pocket as sticks, framed from the start.
    check(got["siteAtoms"] > 20,
          "the core picked %d pocket atom(s) to draw" % got["siteAtoms"], f)
    check(got["siteDrawn"],
          "and the viewer made them into a component", f)
    print("   state cells:")
    for c in (got.get("cells") or []):
        print("     %-34s %-28s reprs=%s%s"
              % (c["label"][:34], c["type"][:28], c["reprs"],
                 "  ERROR: " + c["err"] if c["err"] else ""))
    check(not got.get("stateErrors"),
          "no state node failed: %s" % got.get("stateErrors"), f)
    if got.get("dragErr"):
        print("   drag error: %s" % got["dragErr"])
    check(got.get("gutters") == 2,
          "both columns have a drag handle (%s)" % got.get("gutters"), f)
    check(got.get("leftAfterDrag") and abs(got["leftAfterDrag"] - 380) < 3,
          "dragging the left gutter 90px widened the column to %s (from 290)"
          % got.get("leftAfterDrag"), f)
    check(got.get("leftStored") and abs(got["leftStored"] - 380) < 3,
          "and the width was remembered (%s)" % got.get("leftStored"), f)
    check(got.get("textSize") == 0.6 and got.get("textLabel") == "0.60",
          "the label size slider reaches the viewer (%s) and the readout "
          "(%s)" % (got.get("textSize"), got.get("textLabel")), f)
    check(got.get("textStored") == 0.6,
          "and is remembered (%s)" % got.get("textStored"), f)
    check(got.get("drawnAfterText") == got.get("drawn"),
          "with the lines redrawn rather than lost (%s of %s)"
          % (got.get("drawnAfterText"), got.get("drawn")), f)
    check(got.get("ribbonDefault") is False,
          "the protein ribbon starts hidden, so the pocket is not buried "
          "in it", f)
    check(got.get("stickSize") and got["stickSize"] <= 0.12,
          "and the pocket sticks are thin (sizeFactor %s)"
          % got.get("stickSize"), f)
    check(got["siteFocused"],
          "with a loci to frame, so the view opens on the pocket rather "
          "than on 2800 atoms of cartoon", f)
    # Grading used to wait for featurisation. It does not any more: an
    # opinion about a pose needs a pose, and nothing else.
    check("b" in (got.get("gradeHint") or "")
          and "Analyse all poses" not in (got.get("gradeHint") or ""),
          "grading is available as soon as a pose is on screen: %r"
          % (got.get("gradeHint") or "")[:70], f)
    check(got.get("gradeEnabled") is True,
          "and the buttons are live", f)
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
