#!/usr/bin/env python
"""The grading loop, driven in a browser.

The party is the point of the tool: analyse the poses, grade a few, train,
be handed the ones worth looking at next. All of it runs in the tab - the
same hitpicking.al the ChimeraX bundle uses, so a party started here can
be finished there.

What this pins is the loop itself, because every part of it is easy to
break in a way that still looks like it works: a grade that reaches the UI
but not the learner, a model that trains on nothing, suggestions that
never exclude what you already graded, a saved party that restores zero
grades in silence.

Usage:  python tools/check_web_al.py
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
sys.path.insert(0, os.path.join(ROOT, "tools"))

from check_web_parity import (LIG, PORT, REC, find_chrome,  # noqa: E402
                              run_browser, serve)
from check_web_viewer import check, trim_mol2  # noqa: E402

DRIVER = "__al.html"
POSES = 14


PAGE = """<!doctype html><html><head><meta charset="utf-8"></head><body>
<iframe id="app" src="index.html" style="width:1400px;height:900px;border:0">
</iframe>
<script>
var LOG = [];
function log(s) { LOG.push(String(s)); }
function done(o) {
  o.log = LOG.join("\\n");
  navigator.sendBeacon("/__result", JSON.stringify(o));
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
    await until(function () { return /engine ready/.test(
      d.getElementById("status").textContent); }, 300000, "the engine");
    log("engine up");

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
                20000, "Analyse");
    d.getElementById("go").click();
    await until(function () { return /of \\d+/.test(
      d.getElementById("posen").textContent); }, 300000, "the first pose");
    await sleep(1500);
    out.poses = w.HP.state().poses.length;
    log("loaded " + out.poses + " poses");

    // --- a party with no model at all --------------------------------
    // Grading must not require featurising: most of a hit-picking party
    // is walking a list and writing down what you think.
    d.getElementById("opt-advance").checked = false;
    d.getElementById("pose").value = "0";
    d.getElementById("pose").dispatchEvent(new w.Event("change"));
    await sleep(600);
    await w.HP.grade("good");
    await sleep(400);
    out.gradedBeforeFeaturise = w.HP.state().al.graded || 0;
    out.gradeHintEarly = d.getElementById("gradestate").textContent
      .replace(/\s+/g, " ").trim();
    // and it must be exportable and saveable with no model in sight
    var earlyCsv = await w.HP.bridge.call("exportRows", {});
    out.earlyRows = earlyCsv.rows.length;
    out.earlyGradedRow = earlyCsv.rows[0].your_grade;
    var earlyParty = await w.HP.bridge.call("saveParty", {});
    out.earlySavedGrades = Object.keys(earlyParty.human_grades || {}).length;
    out.earlySavedModel = !!earlyParty.committee;

    // --- analyse them all ---------------------------------------------
    var t0 = Date.now();
    await w.HP.featuriseAll();
    out.featuriseMs = Date.now() - t0;
    out.featurised = w.HP.state().al.poses;
    log("featurised " + out.featurised + " in " + out.featuriseMs + " ms");

    // --- grade a spread -----------------------------------------------
    var plan = ["bad", "good", "maybe", "good", "bad", "good",
                "maybe", "bad", "good", "maybe"];
    for (var i = 0; i < plan.length; i++) {
      d.getElementById("pose").value = String(i);
      d.getElementById("pose").dispatchEvent(new w.Event("change"));
      await until(function (n) { return function () {
        return w.HP.state().current === n; }; }(i), 60000, "pose " + i);
      await sleep(250);
      await w.HP.grade(plan[i]);
      await sleep(250);
    }
    out.graded = w.HP.state().al.graded;
    out.gradedFeaturised = w.HP.state().al.graded_featurised;
    out.gradeCounts = w.HP.state().al.grades;
    out.canTrain = w.HP.state().al.can_train;
    log("graded " + out.graded);

    // the hotkey path is how a party actually grades
    d.getElementById("pose").value = "11";
    d.getElementById("pose").dispatchEvent(new w.Event("change"));
    await sleep(600);
    d.dispatchEvent(new w.KeyboardEvent("keydown", {key: "g", bubbles: true}));
    await sleep(800);
    out.afterHotkey = w.HP.state().al.graded;
    out.hotkeyGrade = w.HP.state().grades["11"] || null;

    // --- train ---------------------------------------------------------
    t0 = Date.now();
    d.getElementById("train").click();
    await until(function () { return w.HP.state().al.trained; },
                180000, "training");
    out.trainMs = Date.now() - t0;
    out.trained = w.HP.state().al.trained;
    log("trained in " + out.trainMs + " ms");

    await sleep(1200);
    var st = w.HP.state();
    out.suggestions = (st.suggestions || []).length;
    out.suggestedIndices = (st.suggestions || []).map(function (s) {
      return s.index; });
    out.gradedIndices = Object.keys(st.grades).map(Number);
    out.suggestBoxShown = d.getElementById("suggestbox").style.display !== "none";
    out.poseOptionHasMark = d.getElementById("pose").options[0].textContent;

    var csv = await w.HP.bridge.call("exportRows", {});
    out.csvCols = csv.columns;
    out.csvScored = csv.rows.filter(function (r) {
      return r.predicted !== ""; }).length;
    out.csvGraded = csv.rows.filter(function (r) {
      return r.your_grade !== ""; }).length;
    out.csvAnalysed = csv.rows.filter(function (r) {
      return r.summary !== ""; }).length;

    // --- save and load --------------------------------------------------
    var party = await w.HP.bridge.call("saveParty", {});
    out.savedGrades = Object.keys(party.human_grades || {}).length;
    out.savedHasModel = !!party.committee;
    var back = await w.HP.bridge.call("loadParty",
                                      {text: JSON.stringify(party)});
    out.restored = back.restored;
    out.unmatched = back.unmatched;

    // Loading a party brings back grades and the model but NOT the
    // feature vectors - they are reproducible in seconds and would make
    // the file enormous. So predictions are unavailable until the poses
    // are analysed again, and the export has to say so by leaving the
    // column empty rather than inventing a score.
    var afterLoad = await w.HP.bridge.call("exportRows", {});
    out.scoredAfterLoad = afterLoad.rows.filter(function (r) {
      return r.predicted !== ""; }).length;

    // a party from a different file must not pretend to fit
    var alien = JSON.parse(JSON.stringify(party));
    alien.human_grades = {"999|NOT_IN_THIS_FILE": "good"};
    alien.grades = {"999|NOT_IN_THIS_FILE": 2};
    var bad = await w.HP.bridge.call("loadParty",
                                     {text: JSON.stringify(alien)});
    out.alienRestored = bad.restored;
    out.alienUnmatched = bad.unmatched;
    out.ok = true;
  } catch (e) {
    out.error = String((e && e.stack) || e);
  }
  done(out);
})();
</script></body></html>"""


def main():
    chrome = find_chrome()
    if chrome is None or not os.path.exists(REC):
        print("need headless Chrome and the sample data; skipping")
        return 0

    with open(os.path.join(WEB, DRIVER), "w", encoding="utf-8") as fh:
        fh.write(PAGE)
    with open(os.path.join(WEB, "__rec"), "w", encoding="utf-8") as fh:
        fh.write(open(REC, encoding="utf-8", errors="replace").read())
    with open(os.path.join(WEB, "__lig"), "w", encoding="utf-8") as fh:
        fh.write(trim_mol2(LIG, POSES))

    inbox = []
    httpd = serve(WEB, PORT, inbox)
    try:
        t0 = time.time()
        got = run_browser(chrome, PORT, inbox, timeout=900, page=DRIVER)
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
    print("the loop ran in %.0f s" % secs)
    check(not got.get("errors"),
          "no JavaScript error (%s)" % ("; ".join(got["errors"]) or "none"), f)

    check(got["gradedBeforeFeaturise"] == 1,
          "a pose can be graded with no model and nothing analysed "
          "(%d recorded)" % got["gradedBeforeFeaturise"], f)
    check("you graded this" in (got.get("gradeHintEarly") or ""),
          "and the page says so: %r" % (got.get("gradeHintEarly") or "")[:60],
          f)
    check(got["earlyRows"] == got["poses"]
          and got["earlyGradedRow"] == "good",
          "the results export works with no model: %d row(s), first is %r"
          % (got["earlyRows"], got["earlyGradedRow"]), f)
    check(got["earlySavedGrades"] == 1 and not got["earlySavedModel"],
          "and a party with opinions and no model still saves", f)

    check(got["featurised"] == got["poses"],
          "every pose featurised (%d of %d) in %.1f s"
          % (got["featurised"], got["poses"], got["featuriseMs"] / 1000.0), f)

    check(got["graded"] == 10,
          "ten grades recorded (%d)" % got["graded"], f)
    counts = got.get("gradeCounts") or {}
    check(sorted(counts.values()) == [3, 3, 4],
          "spread across all three (%s)" % counts, f)
    check(got.get("gradedFeaturised") == 10,
          "and all of them reached the learner once analysed (%s) - "
          "including the one given before anything was"
          % got.get("gradedFeaturised"), f)
    check(got["canTrain"] is True, "which is enough to train", f)
    check(got["afterHotkey"] == 11 and got["hotkeyGrade"] == "good",
          "the b/m/g hotkey grades too (%s on pose 12)"
          % got.get("hotkeyGrade"), f)

    check(got["trained"] is True,
          "the model trained in %.1f s" % (got["trainMs"] / 1000.0), f)
    check(got["suggestions"] > 0,
          "and suggested %d pose(s) to look at next" % got["suggestions"], f)
    overlap = sorted(set(got["suggestedIndices"]) & set(got["gradedIndices"]))
    check(not overlap,
          "none of which you have already graded (%s)" % overlap, f)
    check(got["suggestBoxShown"],
          "the suggestion list is on screen", f)
    check("[" in (got.get("poseOptionHasMark") or ""),
          "and the pose list marks what you graded: %r"
          % got.get("poseOptionHasMark"), f)

    check(got["scoredAfterLoad"] == 0,
          "after loading a party the score column is empty until the poses "
          "are analysed again, rather than carrying an invented number "
          "(%d scored)" % got["scoredAfterLoad"], f)
    check(got["csvGraded"] == 11 and got["csvScored"] == got["poses"]
          and got["csvAnalysed"] == got["poses"],
          "the results CSV carries %d grade(s), %d prediction(s) and %d "
          "analysed pose(s) over %s columns"
          % (got["csvGraded"], got["csvScored"], got["csvAnalysed"],
             len(got["csvCols"])), f)
    check(got["savedGrades"] == 11 and got["savedHasModel"],
          "a saved party carries %d grade(s) and the model"
          % got["savedGrades"], f)
    check(got["restored"] == 11 and got["unmatched"] == 0,
          "and loads back whole (%d restored, %d unmatched)"
          % (got["restored"], got["unmatched"]), f)
    check(got["alienRestored"] == 0 and got["alienUnmatched"] == 1,
          "a grade for a pose not in this file is reported, not applied "
          "to whatever is at that index", f)

    print()
    if f:
        print("WEB AL FAILURES (%d):" % len(f))
        for x in f:
            print("   -", x)
        return 1
    print("WEB AL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
