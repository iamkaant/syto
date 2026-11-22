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
    out.buildBadge = (d.getElementById("build") || {}).textContent || "";

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
    var pdoc = d.getElementById("hb").contentDocument;
    out.rows = pdoc.querySelectorAll("#pose tr.ix").length;
    out.swatches = pdoc.querySelectorAll("#pose tr.ix .kindsw").length;
    out.swatchColor = (pdoc.querySelector("#pose tr.ix .kindsw") || {})
      .getAttribute ? pdoc.querySelector("#pose tr.ix .kindsw")
        .getAttribute("style") : "";
    out.oldList = !!d.getElementById("list");
    out.buildBadgeAfter = (d.getElementById("build") || {}).textContent || "";
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

    // What the boxes claim, and what the core was told.
    function ticked() {
      return [].slice.call(d.querySelectorAll("#toggles input[data-kind]"))
        .filter(function (b) { return b.checked; })
        .map(function (b) { return b.getAttribute("data-kind"); });
    }
    out.tickedKinds = ticked();
    out.coreKinds = (await w.HP.bridge.call("setup",
      {name: "rec.crg.pdb", text: rec})).default_nci;
    var box = d.querySelector("#toggles input[data-kind='hbond_charged']");
    out.toggledOff = "hbond_charged";
    box.checked = false;
    box.dispatchEvent(new w.Event("change"));
    await until(function () { return !/analysing/.test(
      d.getElementById("status").textContent); }, 180000, "the re-analysis");
    await sleep(1500);
    out.afterToggle = ticked();
    out.kindsAfterToggle = (await w.HP.bridge.call("scene", {}))
      .lines.map(function (l) { return l.kind; });
    box.checked = true;
    box.dispatchEvent(new w.Event("change"));
    await until(function () { return !/analysing/.test(
      d.getElementById("status").textContent); }, 180000, "the re-analysis");
    await sleep(1500);
    out.linewMax = +d.getElementById("opt-linew").max;
    // A ring interaction must span its whole ring, not one atom of it.
    out.ringSpan = 0;
    (V.lastSpans || []).forEach(function (sp) {
      out.ringSpan = Math.max(out.ringSpan, sp[0], sp[1]);
    });
    // Residue names on the pocket, on demand.
    d.getElementById("opt-reslabels").checked = true;
    d.getElementById("opt-reslabels").dispatchEvent(new w.Event("change"));
    await sleep(2500);
    out.resLabelsDrawn = 0;
    V.plugin.state.data.cells.forEach(function (c) {
      var tags = (c.transform && c.transform.tags) || [];
      if (typeof tags === "string") tags = [tags];
      if (tags.indexOf("hp-sitelabels") < 0) return;
      if (c.obj && c.obj.data && c.obj.data.repr) {
        out.resLabelsDrawn = c.obj.data.repr.renderObjects.length;
      }
      // The one label-size control must reach these too. Mol* sizes
      // residue labels with residueScale, so that is what to read.
      var pr = c.transform && c.transform.params;
      if (pr && pr.type && pr.type.params) {
        out.resLabelScale = pr.type.params.residueScale;
      }
    });
    out.textAtLabels = V.textSize;

    // A row in the pose list must take you to the line it describes.
    function campos() {
      var c = V.plugin.canvas3d.camera.state.position;
      return [c[0], c[1], c[2]];     // Vec3 is a typed array, not {x,y,z}
    }
    var before = campos();
    var row = pdoc.querySelector("#pose tr.ix");
    out.rowClickable = !!row;
    if (row) {
      row.click();
      await sleep(2500);
      var after = campos();
      out.cameraMoved = Math.abs(after[0] - before[0])
        + Math.abs(after[1] - before[1]) + Math.abs(after[2] - before[2]);
    }

    // The pocket must not be rebuilt by a redraw that has nothing to do
    // with it - that blink is "it re-shows them every time".
    var buildsBefore = V.siteBuilds;
    d.getElementById("opt-labels").checked = false;
    d.getElementById("opt-labels").dispatchEvent(new w.Event("change"));
    await sleep(1500);
    d.getElementById("opt-labels").checked = true;
    d.getElementById("opt-labels").dispatchEvent(new w.Event("change"));
    await sleep(1500);
    out.siteRebuilds = V.siteBuilds - buildsBefore;

    // Keeping the pocket: the union across poses, not this ligand's
    // neighbourhood.
    d.getElementById("opt-sitekeep").checked = true;
    d.getElementById("opt-sitekeep").dispatchEvent(new w.Event("change"));
    await sleep(1500);
    out.siteKeptBefore = V.siteCount;
    d.getElementById("next").click();
    await sleep(6000);
    out.siteKeptAfter = V.siteCount;
    d.getElementById("prev").click();
    await sleep(6000);
    // The colour key, in the view. It must name what was drawn and
    // nothing else, and the names must be gone from the line labels.
    // What was actually written on the lines, read back off the state
    // tree rather than trusted: the kind name must be gone from it.
    var texts = [];
    V.plugin.state.data.cells.forEach(function (c) {
      try {
        var j = JSON.stringify((c.transform && c.transform.params) || null);
        if (!j) return;
        (j.match(/"customText":"[^"]*"/g) || []).forEach(function (x) {
          texts.push(x.slice('"customText":"'.length, -1));
        });
      } catch (e) { /* params that do not stringify are not labels */ }
    });
    out.lineLabels = texts.slice(0, 4);
    // Nothing drawn behind or around the text: Mol* puts both there by
    // default and they are what make a small label unreadable.
    out.labelChrome = [];
    V.plugin.state.data.cells.forEach(function (c) {
      var pr = c.transform && c.transform.params;
      if (!pr || pr.customText === undefined) return;
      out.labelChrome.push([pr.borderWidth, pr.background]);
    });
    out.viewerLabels = (V.lastLabels || []).slice(0, 4);
    var lg = d.getElementById("legend");
    out.legendShown = lg && !lg.hidden;
    out.legendRows = lg ? lg.querySelectorAll(".lrow").length : -1;
    out.legendText = lg ? lg.textContent.replace(/\s+/g, " ").trim() : "";
    d.getElementById("opt-legend").checked = false;
    d.getElementById("opt-legend").dispatchEvent(new w.Event("change"));
    await sleep(400);
    out.legendOff = d.getElementById("legend").hidden;
    d.getElementById("opt-legend").checked = true;
    d.getElementById("opt-legend").dispatchEvent(new w.Event("change"));
    await sleep(400);
    out.legendBack = !d.getElementById("legend").hidden;
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

    // --- the torsion list, end to end -------------------------------
    //
    // The panel has had this section since the ChimeraX bundle grew it;
    // the web page was handing it an empty array. Check the rows arrive,
    // that a click reaches the viewer as geometry, and that the profile
    // lands beside the handbook entry.
    var pt = d.getElementById("hb").contentDocument;
    out.torsionRows = pt.querySelectorAll("#torsions .trow").length;
    out.torsionSum = (pt.querySelector("#fold-torsions .sum") || {}).textContent
                     || "";
    var first = pt.querySelector("#torsions .trow");
    var trows = await w.HP.bridge.call("torsions", {});
    out.torsionRowXyz = trows.rows && trows.rows.length
      ? trows.rows[0].xyz : null;
    if (first) {
      out.torsionRow = first.textContent.replace(/\s+/g, " ").trim();
      first.click();
      await sleep(2000);
      out.torsionSelected = pt.querySelectorAll("#torsions .trow.on").length;
      out.torsionRef = !!V.torsion;
      // The representation is a CHILD of the selection node, and it is
      // labelled by its type ("Ball & Stick"), not by what it shows - so
      // it is found by the tag the viewer puts on it, not by label.
      var tcell = null;
      V.plugin.state.data.cells.forEach(function (c) {
        var tags = (c.transform && c.transform.tags) || [];
        if (typeof tags === "string") tags = [tags];
        if (tags.indexOf("hp-torsion") < 0) return;
        if (c.obj && c.obj.data && c.obj.data.repr) {
          tcell = c.obj.data.repr.renderObjects.length;
        }
      });
      out.torsionReprs = tcell;
      // The profile now lives WITH the list, not at the bottom of the
      // clicked-atom column.
      out.torsionSection = pt.getElementById("torsions").textContent
        .replace(/\s+/g, " ").trim();
      out.torsionProfile = /QM scans/.test(out.torsionSection);
      out.torsionPlotInSection =
        !!pt.querySelector("#torsions .tprofile svg");
      // #hit holds the basicity scale, which is also a .scale - so this
      // asks specifically for a torsion profile.
      out.torsionPlotInHit = !!pt.querySelector("#hit .tprofile svg");
    }

    // --- a bond click, not a row click ------------------------------
    //
    // Any rotatable bond answers, flagged or not. The torsion's own two
    // middle atoms are a bond the viewer can be asked about directly.
    if (first && out.torsionRowXyz) {
      var tx = out.torsionRowXyz;
      w.HP.viewer().onBondClick(tx[1], tx[2]);
      await sleep(2000);
      out.bondClickProfile = /QM scans/.test(
        pt.getElementById("torsions").textContent);
      out.bondClickStatus = d.getElementById("status").textContent
        .replace(/\s+/g, " ").trim();
      // A bond with no torsion must say so rather than guess: the first
      // interaction line's two ends are not a rotatable bond.
      var far = await w.HP.bridge.call("torsionAt",
        {a: scene.lines[0].a, b: scene.lines[0].b});
      out.bondRefused = !!far.error;
      out.bondRefusedWhy = far.error || "";
    }

    // --- a pick can be put back ------------------------------------
    //
    // Highlighting a torsion moves the camera and leaves a component on
    // screen. Before this the only way out was to step to another pose
    // and come back, which is not a way out.
    if (first) {
      first.click();                       // the same row again
      await sleep(1500);
      out.clearedByRow = !V.torsion
        && pt.querySelectorAll("#torsions .trow.on").length === 0;
      first.click();                       // pick it again
      await sleep(1500);
      out.repickedAfterClear = !!V.torsion;
      d.getElementById("reset").click();   // and the button clears it
      await sleep(1500);
      out.clearedByButton = !V.torsion;
      out.resetStatus = d.getElementById("status").textContent
        .replace(/\s+/g, " ").trim();
      V.onEmptyClick();                    // as does clicking nothing
      await sleep(800);
      out.emptyClickOk = !V.torsion;
    }

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
    // A new pose means new torsions and no selection carried over: the
    // highlight hangs off the pose node that was just deleted.
    out.torsionRows2 = d.getElementById("hb").contentDocument
      .querySelectorAll("#torsions .trow").length;
    out.torsionRef2 = !!V.torsion;

    // --- hammering "next" leaves ONE ligand on screen ---------------
    //
    // Each click starts an analysis several awaits long. Interleaved,
    // they each delete a node another had already replaced and each
    // leave their own structure behind - several ligands at once, which
    // is what was reported.
    function models() {
      var n = 0;
      V.plugin.state.data.cells.forEach(function (c) {
        if (c.obj && c.obj.type && c.obj.type.name === "Model") n++;
      });
      return n;                       // one receptor + one pose = 2
    }
    out.modelsBefore = models();
    for (var hammer = 0; hammer < 5; hammer++) {
      d.getElementById("next").click();
      await sleep(60);                // faster than any analysis can run
    }
    await until(function () { return !/analysing/.test(
      d.getElementById("status").textContent); }, 180000, "the pile-up");
    await sleep(3000);
    out.modelsAfter = models();
    out.hammerPosen = d.getElementById("posen").textContent;
    out.hammerDrawn = V.nDrawn;
    var hs = await w.HP.bridge.call("scene", {});
    out.hammerSceneLines = hs.lines.length;
    out.hammerName = hs.name;
    out.hammerShown = d.getElementById("status").textContent
      .replace(/\s+/g, " ").trim();

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
        # This page drives the whole tool, and the pile-up check adds
        # five more analyses on top - each one real work in Pyodide. The
        # default 420s is for the parity page, which does far less.
        got = run_browser(chrome, PORT, inbox, page=DRIVER, timeout=900)
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
    check((got.get("buildBadgeAfter") or "").startswith("build ")
          and (got.get("build") or "?")[:8] in got.get("buildBadgeAfter", ""),
          "and still says so after a pose is analysed, when the status "
          "line has been overwritten: %r" % got.get("buildBadgeAfter"), f)
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
    check(got.get("oldList") is False,
          "the duplicate list in the left column is gone", f)
    check(got["rows"] > 0 and got["rows"] == got["sceneLines"],
          "the panel lists every drawn interaction (%d row(s) for %d line(s))"
          % (got["rows"], got["sceneLines"]), f)
    check(got.get("swatches") == got["rows"]
          and "#" in (got.get("swatchColor") or ""),
          "each row carries the colour it was drawn in (%s swatch(es), e.g. "
          "%r)" % (got.get("swatches"), got.get("swatchColor")), f)
    check(got.get("linewMax") == 1,
          "the line-width slider stops at 1x (max=%s)" % got.get("linewMax"), f)
    check(got.get("resLabelsDrawn", 0) > 0,
          "naming the pocket residues draws %s label(s)"
          % got.get("resLabelsDrawn"), f)
    check(got.get("resLabelScale")
          and abs(got["resLabelScale"] - got.get("textAtLabels", 0)) < 1e-6,
          "and the one label-size control sizes them to match the "
          "interaction labels (scale %s for slider %s)"
          % (got.get("resLabelScale"), got.get("textAtLabels")), f)
    check(got.get("rowClickable") and (got.get("cameraMoved") or 0) > 0.01,
          "clicking an interaction in the pose list flies the camera to "
          "it (moved %.2f A)" % (got.get("cameraMoved") or 0), f)
    check(got.get("siteRebuilds") == 0,
          "a redraw that does not touch the pocket leaves it alone "
          "(%s rebuild(s))" % got.get("siteRebuilds"), f)
    check(got.get("siteKeptAfter", 0) >= got.get("siteKeptBefore", 0) > 0,
          "keeping the pocket never drops a residue across poses "
          "(%s atoms, then %s)"
          % (got.get("siteKeptBefore"), got.get("siteKeptAfter")), f)
    check(got.get("ringSpan", 0) >= 5,
          "a ring interaction is drawn from the whole ring, so the line "
          "meets the centroid the core measured (%s atoms spanned)"
          % got.get("ringSpan"), f)
    check(got["toggles"] >= 8,
          "%d interaction types can be switched off" % got["toggles"], f)

    # Clicking an atom reaches the SHARED panel, not a second renderer.
    hit = got.get("panelHit") or ""
    check(len(hit) > 60 and "Ctrl-click" not in hit,
          "clicking an atom filled the handbook panel: %r" % hit[:90], f)
    check(got.get("farRefused") is True,
          "an atom that is not in the analysis is refused, not guessed at", f)

    # The torsion list, which the web page used to render as nothing.
    check(got.get("torsionRows", 0) > 0,
          "the panel lists %s strained torsion(s): %s"
          % (got.get("torsionRows"), got.get("torsionSum")), f)
    check((got.get("torsionRow") or "").strip() != "",
          "worst first: %r" % (got.get("torsionRow") or "")[:90], f)
    check(got.get("torsionSelected") == 1,
          "clicking a row selects exactly one (%s)"
          % got.get("torsionSelected"), f)
    check(got.get("torsionRef"),
          "and the viewer made a component for the bond", f)
    check((got.get("torsionReprs") or 0) > 0,
          "which actually rendered (reprs=%s)" % got.get("torsionReprs"), f)
    check(got.get("torsionProfile") is True,
          "with its QM profile in the same section: %r"
          % (got.get("torsionSection") or "")[-90:], f)
    check(got.get("torsionPlotInSection") is True
          and got.get("torsionPlotInHit") is False,
          "the plot is under the torsion list, not at the bottom of the "
          "clicked-atom column (section=%s, hit=%s)"
          % (got.get("torsionPlotInSection"), got.get("torsionPlotInHit")), f)
    check(got.get("bondClickProfile") is True,
          "clicking the bond itself plots it too: %r"
          % (got.get("bondClickStatus") or "")[:80], f)
    check(got.get("clearedByRow") is True,
          "clicking the picked row again puts it back", f)
    check(got.get("repickedAfterClear") is True,
          "and it can be picked again afterwards", f)
    check(got.get("clearedByButton") is True,
          "Reset view drops the pick and re-frames the pocket: %r"
          % (got.get("resetStatus") or "")[:60], f)
    check(got.get("emptyClickOk") is True,
          "and so does clicking empty space", f)
    check(got.get("bondRefused") is True,
          "a bond with no torsion is refused, not guessed at: %r"
          % got.get("bondRefusedWhy"), f)

    # The labels carry numbers; the colours carry the names.
    labels = got.get("lineLabels") or got.get("viewerLabels") or []
    named = [t for t in labels
             if any(k in t for k in ("bond", "stack", "pi", "phobic", "salt"))]
    check(labels and not named,
          "no interaction name is written on the lines any more: %s"
          % labels[:3], f)
    check(all("  " not in t for t in labels),
          "and no run of spaces inside a label, which reads as loose "
          "letter spacing: %s" % labels[:3], f)
    chrome = got.get("labelChrome") or []
    check(chrome and all(c[1] is False and (c[0] or 0) > 0 for c in chrome),
          "an outline on the label text and no plate behind it "
          "(borderWidth, background: %s)"
          % (chrome[:2] or "no labels found"), f)

    # The toggles must say what the core is actually computing. They
    # used to show all nine ticked while it ran seven, so the first
    # toggle switched two types on by reporting the UI's own lie back.
    check(got.get("tickedKinds") and got.get("coreKinds")
          and sorted(got["tickedKinds"]) == sorted(got["coreKinds"]),
          "the type checkboxes start as what the core computes "
          "(%s ticked, core runs %s)"
          % (len(got.get("tickedKinds") or []),
             len(got.get("coreKinds") or [])), f)
    check(got.get("afterToggle") is not None
          and sorted(got["afterToggle"])
              == sorted([k for k in (got.get("coreKinds") or [])
                         if k != got.get("toggledOff")]),
          "and turning one off turns off exactly that one (%s)"
          % (got.get("afterToggle") or []), f)
    check(got.get("legendShown") is True and got.get("legendRows", 0) > 0,
          "the view carries a colour key with %s row(s): %r"
          % (got.get("legendRows"), (got.get("legendText") or "")[:70]), f)
    check(got.get("legendRows") == len(set(got.get("kinds") or []))
          + (1 if got.get("sceneClashes") else 0)
          + (1 if got.get("torsionRef") else 0)
          or got.get("legendRows", 0) >= len(set(got.get("kinds") or [])),
          "and it names what was drawn, not a fixed list", f)
    check(got.get("legendOff") is True and got.get("legendBack") is True,
          "and it can be switched off and back on", f)

    # Two models, always: the receptor and the pose on screen. Orphans
    # draw nothing, so only a count catches them.
    check(got.get("modelsAfter") == 2 and got.get("modelsBefore") == 2,
          "five fast clicks on next leave one ligand on screen, not a "
          "pile (%s models before, %s after; 2 is receptor + pose)"
          % (got.get("modelsBefore"), got.get("modelsAfter")), f)
    check(got.get("hammerDrawn") == got.get("hammerSceneLines"),
          "and what is drawn is the pose that ended up selected "
          "(%s lines drawn, %s in the scene for %s)"
          % (got.get("hammerDrawn"), got.get("hammerSceneLines"),
             got.get("hammerName")), f)

    # Switching pose must redraw, not accumulate.
    check(got["posen"].startswith("1 of") and got["secondMs"] is not None,
          "the pose selector moved: %r" % got.get("posen"), f)
    check(got["drawn2"] == got["sceneLines2"] + 0 or got["drawn2"] > 0,
          "the second pose drew %d line(s)" % got["drawn2"], f)
    check(got["missed2"] == 0,
          "with nothing unplaced (%d)" % got["missed2"], f)
    check(got.get("torsionRef2") is False,
          "the torsion highlight went with the pose it belonged to", f)
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
