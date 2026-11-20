#!/usr/bin/env python
"""Run the panel's JavaScript against a real snapshot and fail on any error.

Parsing the file is not enough. A missing helper - `n2(...)` used with no
`function n2` anywhere - parses perfectly and then throws the moment an atom
is clicked, which blanks the whole "What you clicked" column while every
other check stays green. The only way to catch that is to execute it.

Headless Chrome renders the page with a snapshot built from the real
handbook, and this asserts that nothing threw and that each section put
something in the DOM. Skipped, not failed, where Chrome is absent.

Usage:  python tools/check_panel_js.py [--keep]
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
PANEL = os.path.join(ROOT, "chimerax", "src", "ui", "panel.html")
LEARN = os.path.join(ROOT, "chimerax", "src", "ui", "learn.html")
CHROME = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
          "/Applications/Chromium.app/Contents/MacOS/Chromium",
          "/usr/bin/google-chrome", "/usr/bin/chromium")

CATCH = """<script>
window.__jsErrors = [];
window.onerror = function (m, u, l, c, e) {
  window.__jsErrors.push(m + " @line " + l);
};
window.addEventListener("unhandledrejection", function (ev) {
  window.__jsErrors.push("unhandled rejection: " + ev.reason);
});
</script></head>"""

REPORT = """<script>
window.addEventListener("load", function () {
  var d = document.createElement("pre");
  d.id = "__report";
  d.textContent = JSON.stringify({
    errors: window.__jsErrors,
    hit: (document.getElementById("hit") || {}).innerHTML || "",
    scrollers: [[document.scrollingElement, "page"],
                [document.querySelector(".col.right"), "right column"],
                [document.querySelector(".col:not(.right)"), "left column"]]
      .filter(function (p) {
        return p[0] && p[0].scrollHeight - p[0].clientHeight > 2;
      }).map(function (p) { return p[1]; }),
    pose: (document.getElementById("pose") || {}).innerHTML || "",
    toggles: (document.getElementById("toggles") || {}).innerHTML || "",
    sumNci: (document.getElementById("sum-nci") || {}).textContent || "",
    sumDisplay: (document.getElementById("sum-display") || {}).textContent || "",
    foldsOpen: ["fold-nci", "fold-display"].map(function (i) {
      var el = document.getElementById(i);
      return el ? el.open : null;
    })
  });
  document.body.appendChild(d);
});
</script></body>"""


LEARN_REPORT = """<script>
window.addEventListener("load", function () {
  var d = document.createElement("pre");
  d.id = "__report";
  var el = document.getElementById("terms");
  var fold = document.getElementById("termfold");
  d.textContent = JSON.stringify({
    errors: window.__jsErrors,
    terms: el ? el.innerHTML : null,
    open: fold ? !!fold.open : null,
    overflow: document.documentElement.scrollWidth
             - document.documentElement.clientWidth,
    marker: fold ? getComputedStyle(fold.querySelector("summary"),
                                    "::before").content : null,
    boxes: [].slice.call(document.querySelectorAll("input[data-term]"))
      .map(function (b) {
        return [b.getAttribute("data-term"), b.checked]; }),
    model: (document.getElementById("modelinfo") || {}).innerHTML || "",
    status: (document.getElementById("status") || {}).innerHTML || ""
  });
  document.body.appendChild(d);
});
</script></body>"""


def learn_snapshot(terms):
    """The Learn panel's snapshot, with a score-term block to render."""
    return {
        "stats": {"poses": 24, "graded": 12, "grades": {"bad": 4, "maybe": 4,
                  "good": 4}, "human_grades": {"bad": 4, "maybe": 4,
                  "good": 4}, "assumed_grades": {}, "n_assumed": 0,
                  "assumed_grade": None, "trained_on": 12, "ready": True,
                  "can_train": True, "why_not": "", "feature_version": 2,
                  "waiting": 0, "score_terms": terms["chosen"]},
        "running": None, "worklist": None,
        "options": ["bad", "maybe", "good"],
        "current": None, "strategy": "uncertainty", "auto_advance": True,
        "suggestions": [], "curve": None, "terms": terms,
    }


def _row(name, usable=True, verdict="varies", mean=-40.0, sd=3.0):
    return {"name": name, "n": 24, "coverage": 1.0, "distinct": 20,
            "mean": mean, "sd": sd, "sentinels": 0,
            "verdict": verdict, "why": "" if usable else "a row number",
            "usable": usable}


def check_learn(chrome, work):
    """The score-term chooser, executed rather than parsed."""
    src = open(LEARN, encoding="utf-8").read()
    assert "</head>" in src and "</body>" in src, "learn.html lost its shell"

    dock = ["Total Energy", "Van der Waals", "Electrostatic"]
    cases = [
        ("dock", {"chosen": dock, "suggested": dock, "source": "dock",
                  "why": "DOCK 3 was recognised by name",
                  "slots": 7, "program": "DOCK 3", "verified": True,
                  "live": 3,
                  "available": [_row(t) for t in dock]
                               + [_row("Number", False, "identifier")]}),
        ("deaf", {"chosen": ["r_i_docking_score"], "suggested": dock,
                  "source": "dock", "why": "DOCK 3 was recognised by name",
                  "slots": 7, "program": "DOCK 3", "verified": True,
                  "live": 0,
                  "available": [_row(t) for t in dock]}),
        ("none", {"chosen": [], "suggested": dock, "source": "generic",
                  "why": "no docking program was recognised by name",
                  "slots": 7, "program": None, "verified": False,
                  "live": 0, "available": [_row(t) for t in dock]}),
        ("nodata", {"chosen": list(dock), "suggested": [], "source": "generic",
                    "why": "no pose carries a usable score term",
                    "slots": 7, "program": None, "verified": False,
                    "live": 0, "available": []}),
    ]
    failures = []
    for name, terms in cases:
        page = src.replace("</head>", CATCH, 1).replace(
            "</body>", LEARN_REPORT, 1)
        page = page.replace('hp("ready");',
                            "window.hpLearnState(%s);"
                            % json.dumps(learn_snapshot(terms)), 1)
        r = render(chrome, page, work, "learn-" + name, (460, 900))

        def check(cond, msg):
            print(("  OK   " if cond else "  FAIL ") + "learn/%s: %s"
                  % (name, msg))
            if not cond:
                failures.append("learn/%s: %s" % (name, msg))

        check(not r["errors"], "no JavaScript error (%s)"
              % ("; ".join(r["errors"]) or "none"))
        # The panel docks at about 460px. A table that overflows it puts a
        # horizontal scrollbar under a list you are meant to read across.
        check(r["overflow"] <= 0,
              "nothing overflows the docked width (%dpx over)"
              % r["overflow"])
        ticked = [n for n, on in r["boxes"] if on]
        if name == "nodata":
            check("learns from geometry and interactions alone"
                  in (r["terms"] or ""),
                  "says plainly that there is no score to learn from")
            check(not r["boxes"], "and offers no checkboxes")
            continue
        check(len(r["boxes"]) == len(terms["available"]),
              "every term the poses carry is offered (%d)" % len(r["boxes"]))
        check(ticked == [t for t in terms["chosen"]
                         if any(x["name"] == t for x in terms["available"])],
              "the ticked boxes are the current pick: %s" % ticked)
        check("Number" in (r["terms"] or "") or name != "dock",
              "the junk is shown, named as junk, rather than hidden")
        if name == "deaf":
            # the whole reason this feature exists: a pick that matches
            # nothing in the file must not look like a working one
            check("None of the 1 selected term" in r["terms"],
                  "a pick that matches nothing in the file says so")
            check(r["open"] is True,
                  "and the section opens itself rather than waiting to be "
                  "found")
        if name == "dock":
            check(r["open"] is False,
                  "a working pick stays folded away")
            check("DOCK 3" in r["terms"], "and names the program")
        if name == "none":
            check("No score terms" in r["terms"],
                  "choosing none is reported as a choice, not an error")
    return failures


def find_chrome():
    for c in CHROME:
        if os.path.exists(c):
            return c
    return None


def snapshot(hb, mids, title):
    anns = hb.annotations_for(mids)
    wanted = {"pKBHX"} | {a["quantity"] for a in anns
                          if a.get("changes_the_answer")}
    scales = []
    for q in ("pKBHX", "pKBH+"):
        if q in wanted:
            sc = hb.scale(q, mids, same_element_only=False)
            if sc:
                scales.append(sc)
    return {
        "ready": True, "handbook": hb.describe(), "viewdock": True,
        "options": {"enabled_nci": ["hbond", "salt_bridge", "pi_stack"],
                    "show_lengths": True, "show_angles": False,
                    "show_clashes": True, "show_limits": False,
                    "show_scale": True, "hover_detail": True},
        "pose": {"name": "ZINC000123456789.1",
                 "counts": {"hbond": 3, "salt_bridge": 1},
                 "warnings": [], "interactions": [],
                 "strain": {"clash": 1}, "worst_overlap": 0.62},
        "hit": {"atom": 17, "title": title, "moiety_ids": list(mids),
                "highlight": [], "annotations": anns, "warnings": [],
                "entries": hb.for_moiety(mids[0])[:3]},
        "scales": scales, "torsion": None,
    }


def render(chrome, html, workdir, name, size=(1100, 800)):
    path = os.path.join(workdir, name + ".html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    out = subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--dump-dom",
         "--window-size=%d,%d" % size,
         "--virtual-time-budget=4000", "file://" + path],
        capture_output=True, text=True, timeout=120).stdout
    m = re.search(r'<pre id="__report">(.*?)</pre>', out, re.S)
    if not m:
        raise AssertionError(
            "%s: the page never finished loading, so nothing could be "
            "checked" % name)
    import html as _h
    return json.loads(_h.unescape(m.group(1)))


def main():
    chrome = find_chrome()
    if chrome is None:
        print("no headless Chrome found; skipping the panel JS check")
        return 0

    from hitpicking.handbook import Handbook
    hb = Handbook.load_or_none()
    if hb is None:
        print("handbook.json not built; skipping the panel JS check")
        return 0

    src = open(PANEL, encoding="utf-8").read()
    assert "</head>" in src and "</body>" in src, "panel.html lost its shell"

    # 900 and 1200 are two-column; 620 is the narrow docked panel, where
    # the media query collapses to one column.
    cases = [
        ("neutral", ["pyridine_n"], "UNL1 N1 - pyridine-type aromatic N",
         (900, 700)),
        ("cation", ["amine_n_tertiary"], "UNL1 N2 - tertiary amine",
         (1200, 900)),
        ("no_data", ["hydrophobic"], "UNL1 C4 - hydrophobic carbon",
         (620, 800)),
    ]
    failures = []
    work = tempfile.mkdtemp(prefix="hp-panel-")
    for name, mids, title, size in cases:
        page = src.replace("</head>", CATCH, 1).replace("</body>", REPORT, 1)
        page = page.replace('hp("ready");',
                            "window.hpSetState(%s);"
                            % json.dumps(snapshot(hb, mids, title)), 1)
        r = render(chrome, page, work, name, size)

        def check(cond, msg):
            tag = "%s@%dx%d" % (name, size[0], size[1])
            print(("  OK   " if cond else "  FAIL ") + "%s: %s" % (tag, msg))
            if not cond:
                failures.append("%s: %s" % (tag, msg))

        check(not r["errors"], "no JavaScript error (%s)"
              % ("; ".join(r["errors"]) or "none"))
        check("Ctrl-click" not in r["hit"] and len(r["hit"]) > 80,
              "the clicked atom rendered rather than the empty state")
        check(title.split(" - ")[1][:12] in r["hit"],
              "and it is the atom that was clicked")
        check("checkbox" in r["toggles"], "the interaction toggles rendered")
        # Nested scrollers put two bars side by side on the right, which is
        # what a column capped at 100vh inside a taller page produces.
        check(r["scrollers"] in ([], ["page"]),
              "exactly one thing scrolls: %s" % (r["scrollers"] or ["nothing"]))
        check(r["sumNci"] and r["sumDisplay"],
              "both folds summarise what is on while collapsed (%r / %r)"
              % (r["sumNci"], r["sumDisplay"]))
        check(r["foldsOpen"] == [False, False],
              "and both start folded away: %s" % r["foldsOpen"])
        # The verdict card is the only place that writes "<b>NN%</b>
        # protonated". The word on its own also appears in entry prose -
        # "a basic pyridine ... may be protonated" - which is the handbook
        # doing its job and must not be mistaken for the card.
        verdict = re.search(r"</b>\s*protonated", r["hit"])
        # `class="tabs"` is the container; the buttons are `tab` and `tab on`
        tabs = re.findall(r'class="tab(?: on)?"', r["hit"])
        if name == "cation":
            check(bool(verdict),
                  "a cationic atom says so where it will be read")
            check(len(tabs) == 2, "and offers both scales: %s" % tabs)
        if name == "neutral":
            check(not verdict,
                  "a neutral atom does not spend a card saying it is neutral")
            check(not tabs, "and shows no tab bar for a single scale")

    failures += check_learn(chrome, work)

    print()
    if failures:
        print("PANEL JS FAILURES (%d):" % len(failures))
        for f in failures:
            print("   -", f)
        return 1
    print("PANEL JS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
