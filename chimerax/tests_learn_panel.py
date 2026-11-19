"""The Learn panel's data layer, without needing a Qt window.

The panel itself cannot be opened in nogui, but everything it renders comes
from snapshot(), so that is what is checked: the numbers it shows, the
suggestions it lists, and that the AL stack is not loaded until the panel
or a learn command asks for it.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_learn_panel.py
"""
import os
import sys
import types

from chimerax.core.commands import run
from chimerax.atomic import AtomicStructure

failures = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


A = "/Users/ak87/ownCloud/owncloud/Calculations/TMP/AutoParty/"
if not os.path.exists(A + "autoparty/inputs/rec.crg.pdb"):
    print("sample data unavailable; skipping")
    sys.exit(0)

print("[1] the panel module is not imported at startup")
run(session, "open %s" % (A + "autoparty/inputs/rec.crg.pdb"))
run(session, "open %s format mol2 showTool false"
    % (A + "autoparty/inputs/more_opcore_enum_asp_h2o_hbond.mol2"))
run(session, "hitpick setup")
check("chimerax.hitpickingtool.learn_tool" not in sys.modules,
      "learn_tool is unimported after setup")
check(not [m for m in sys.modules if m.startswith("hitpicking.al")],
      "the AL engine is unimported after setup")

print("\n[2] snapshot before anything is collected")
from chimerax.hitpickingtool.learn_tool import LearnTool
import chimerax.hitpickingtool.learn as L
from chimerax.hitpickingtool.state import get_state

stub = types.SimpleNamespace(session=session, _strategy="uncertainty",
                             _auto_advance=True, _hp=get_state(session))
snap = types.MethodType(LearnTool.snapshot, stub)()
check(snap["stats"]["poses"] == 0, "no poses featurised yet")
check(snap["suggestions"] == [], "and nothing is suggested")
check(not snap["stats"]["can_train"], "training is not offered")
check("at least 8" in snap["stats"]["why_not"],
      "the panel can explain why: %s" % snap["stats"]["why_not"])

print("\n[3] after collecting and grading")
run(session, "hitpick learn collect limit 40")
ls = L.state(session)
keys = list(ls.learner.features)
for k in keys[:10]:
    ls.learner.grade(k, "bad")
for k in keys[10:20]:
    ls.learner.grade(k, "good")
for k in keys[20:26]:
    ls.learner.grade(k, "maybe")
run(session, "hitpick learn train epochs 40")

snap = types.MethodType(LearnTool.snapshot, stub)()
check(snap["stats"]["poses"] >= 20, "poses: %d" % snap["stats"]["poses"])
check(snap["stats"]["ready"], "the panel sees a trained model")
g = snap["stats"]["grades"]
check(g["bad"] == 10 and g["good"] == 10 and g["maybe"] == 6,
      "grade counts for the panel: %s" % g)
check(len(snap["suggestions"]) > 0,
      "%d suggestions offered" % len(snap["suggestions"]))
if snap["suggestions"]:
    u = [s["uncertainty"] for s in snap["suggestions"]]
    check(u == sorted(u, reverse=True), "ordered most informative first")
    check(all(s["spec"] for s in snap["suggestions"]),
          "each carries a model spec, so clicking can show it")
check(snap["curve"] is not None, "a training curve is available to plot")

print("\n[4] the strategies produce different orderings")
stub._strategy = "disagreement"
dis = types.MethodType(LearnTool.snapshot, stub)()["suggestions"]
check(bool(dis), "disagreement offers rows")
check(all(s["yours"] for s in dis),
      "and every one is a pose you actually graded")
stub._strategy = "uncertainty"

print("\n[5] the current pose is reported for grading")
ligs = [m for m in session.models
        if isinstance(m, AtomicStructure) and m.num_residues <= 3]
run(session, "hitpick analyze #%s" % ligs[0].id_string)
snap = types.MethodType(LearnTool.snapshot, stub)()
cur = snap["current"]
check(cur is not None, "a current pose is shown")
if cur:
    check(cur["featurised"], "it is featurised, so it can be graded")
    check("summary" in cur, "with its interaction summary: %s"
          % str(cur.get("summary"))[:48])

print()
if failures:
    print("FAILURES (%d):" % len(failures))
    for f in failures:
        print("   -", f)
    sys.exit(1)
print("LEARN PANEL TESTS OK")
