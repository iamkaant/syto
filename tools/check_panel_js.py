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


def render(chrome, html, workdir, name):
    path = os.path.join(workdir, name + ".html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    out = subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--dump-dom",
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

    cases = [
        ("neutral", ["pyridine_n"], "UNL1 N1 - pyridine-type aromatic N"),
        ("cation", ["amine_n_tertiary"], "UNL1 N2 - tertiary amine"),
        ("no_data", ["hydrophobic"], "UNL1 C4 - hydrophobic carbon"),
    ]
    failures = []
    work = tempfile.mkdtemp(prefix="hp-panel-")
    for name, mids, title in cases:
        page = src.replace("</head>", CATCH, 1).replace("</body>", REPORT, 1)
        page = page.replace('hp("ready");',
                            "window.hpSetState(%s);"
                            % json.dumps(snapshot(hb, mids, title)), 1)
        r = render(chrome, page, work, name)

        def check(cond, msg):
            print(("  OK   " if cond else "  FAIL ") + "%s: %s" % (name, msg))
            if not cond:
                failures.append("%s: %s" % (name, msg))

        check(not r["errors"], "no JavaScript error (%s)"
              % ("; ".join(r["errors"]) or "none"))
        check("Ctrl-click" not in r["hit"] and len(r["hit"]) > 80,
              "the clicked atom rendered rather than the empty state")
        check(title.split(" - ")[1][:12] in r["hit"],
              "and it is the atom that was clicked")
        check("checkbox" in r["toggles"], "the interaction toggles rendered")
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
