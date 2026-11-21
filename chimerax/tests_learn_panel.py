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


# The sample data is not in this repository and its location is not this
# laptop's home directory. tools/samples.py resolves it; a test that
# cannot find it skips.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools"))
from samples import AUTOPARTY          # noqa: E402

A = AUTOPARTY + "/"
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

print("\n[6] grade in ViewDock, then analyse in the Learn panel")
# The workflow a party actually runs: ratings get given in ViewDock's table,
# sometimes before the poses have been analysed. Those grades used to be
# dropped in silence - iter_rated() returned nothing unless the ViewDock TOOL
# object was found, even though the ratings live on the structures.
import chimerax.hitpickingtool.viewdock_link as VL

L.clear(session)
poses = [m for m in session.models
         if isinstance(m, AtomicStructure) and m.num_residues <= 3]
check(len(poses) >= 8, "at least 8 poses to work with (%d)" % len(poses))

# A) grade BEFORE analysing
for k, s_ in enumerate(poses[:8]):
    if not isinstance(getattr(s_, "viewdock_data", None), dict):
        s_.viewdock_data = {}
    s_.viewdock_data[VL.RATING_KEY] = 1 + (k % 3)
snap = types.MethodType(LearnTool.snapshot, stub)()["stats"]
check(snap["graded"] == 0, "nothing is counted as graded before analysis")
check(snap["waiting"] == 8,
      "but the panel says 8 are rated and waiting (got %s)" % snap["waiting"])

# B) analyse: the earlier grades must now count, with no regrading
run(session, "hitpick learn collect")
snap = types.MethodType(LearnTool.snapshot, stub)()["stats"]
check(snap["graded"] == 8,
      "the 8 grades given before analysis are picked up (got %d)"
      % snap["graded"])
check(snap["waiting"] == 0, "and none are left waiting")
counts = snap["grades"]
check(counts["bad"] == 3 and counts["maybe"] == 3 and counts["good"] == 2,
      "the ordinal mapping survives: %s" % counts)

# C) grading afterwards shows up without pressing Train
spare = [s_ for s_ in poses[8:]][:1]
if spare:
    spare[0].viewdock_data = dict(
        getattr(spare[0], "viewdock_data", None) or {})
    spare[0].viewdock_data[VL.RATING_KEY] = 3
    snap = types.MethodType(LearnTool.snapshot, stub)()["stats"]
    check(snap["graded"] == 9,
          "a grade given after analysis appears at once (got %d)"
          % snap["graded"])

# D) changing a grade in ViewDock overrides the earlier one
poses[0].viewdock_data[VL.RATING_KEY] = 3
snap = types.MethodType(LearnTool.snapshot, stub)()["stats"]
check(snap["grades"]["bad"] == 2 and snap["grades"]["good"] >= 3,
      "changing a rating moves the pose between grades: %s" % snap["grades"])

# E) and the model can be trained from grades that were only ever given
#    in ViewDock - the point of the whole exercise
check(snap["can_train"], "the model can train: %s" % snap.get("why_not"))

print()
if failures:
    print("FAILURES (%d):" % len(failures))
    for f in failures:
        print("   -", f)
    sys.exit(1)
print("LEARN PANEL TESTS OK")
