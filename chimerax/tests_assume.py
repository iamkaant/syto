"""`hitpick learn assume`: the grade a list-by-exception never records.

Some people walk a hit-picking list end to end and press b or m only on what
they object to. ViewDock has one Rating column and no way to say "unrated
here means I looked and had no objection", so a model trained on that list
learns rejection and nothing else. This is the command that says it out loud
- and the thing it must never do is let an assumption pass for a click.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_assume.py
"""
import os
import sys
import types

from chimerax.core.commands import run

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

run(session, "open %s" % (A + "autoparty/inputs/rec.crg.pdb"))
run(session, "open %s format mol2 showTool false"
    % (A + "autoparty/inputs/more_opcore_enum_asp_h2o_hbond.mol2"))
run(session, "hitpick setup")
run(session, "hitpick learn collect limit 24 wait true")

import chimerax.syto.learn as L
from chimerax.syto import viewdock_link as VL
from chimerax.syto.cmd import hitpick_learn_assume
from chimerax.syto.learn_tool import LearnTool
from chimerax.syto.state import get_state

ls = L.state(session)
poses = [ls.keys[k] for k in sorted(ls.keys)]
n_poses = len(poses)
check(n_poses >= 12, "%d poses featurised to work with" % n_poses)

# rate a handful by exception: objections only, nothing marked good
for i, s in enumerate(poses[:6]):
    s.viewdock_data = dict(getattr(s, "viewdock_data", None) or {})
    s.viewdock_data[VL.RATING_KEY] = 1 if i < 4 else 2
ls.pull_viewdock_ratings()

print("\n[1] before assuming, the model has nothing good to aim at")
d = L.describe(session)
check(d["grades"]["good"] == 0, "no pose is graded good")
check(d["graded"] == 6, "6 graded by hand (got %d)" % d["graded"])
check(d["n_assumed"] == 0, "and nothing is assumed yet")

print("\n[2] assuming good fills in exactly the unrated poses")
res = hitpick_learn_assume(session, "good")
want = n_poses - 6
check(res["assumed"] == want,
      "%d unrated pose(s) assumed good (got %d)" % (want, res["assumed"]))
d = L.describe(session)
check(d["graded"] == n_poses, "every featurised pose now has a grade")
check(d["grades"]["good"] == want, "all of them good (got %d)"
      % d["grades"]["good"])

print("\n[3] an assumption never passes for a click")
check(d["human_grades"]["bad"] == 4 and d["human_grades"]["maybe"] == 2,
      "the clicked grades are reported apart: %s" % d["human_grades"])
check(d["human_grades"]["good"] == 0,
      "and no 'good' is attributed to the human")
check(d["assumed_grades"]["good"] == want,
      "the assumed ones are counted as assumed: %s" % d["assumed_grades"])
check(d["assumed_grade"] == "good", "and the panel can name the grade used")

print("\n[4] a real rating overrides the assumption, and stops counting as one")
victim = None
for k in sorted(ls.assumed):
    victim = k
    break
ls.keys[victim].viewdock_data[VL.RATING_KEY] = 1      # bad
ls.pull_viewdock_ratings()
check(victim not in ls.assumed,
      "the pose that was clicked is no longer assumed")
check(ls.learner.grades.label(victim) == "bad",
      "and it carries the grade that was actually given")
d = L.describe(session)
check(d["human_grades"]["bad"] == 5, "the clicked count went up: %s"
      % d["human_grades"])
check(d["n_assumed"] == want - 1, "the assumed count went down (got %d)"
      % d["n_assumed"])

print("\n[5] the panel offers it only when it is the right answer")
stub = types.SimpleNamespace(session=session, _strategy="uncertainty",
                             _auto_advance=True, _hp=get_state(session))
snap = types.MethodType(LearnTool.snapshot, stub)()["stats"]
check(snap["n_assumed"] == want - 1, "the panel sees the assumed count")
check(snap["grades"]["good"] == want - 1,
      "and the totals the model trains on")

print("\n[6] forget takes back every assumption and no clicked grade")
res = hitpick_learn_assume(session, forget=True)
check(res["forgotten"] == want - 1, "dropped %d assumed grade(s)"
      % res["forgotten"])
d = L.describe(session)
check(d["graded"] == 7, "the 7 clicked grades remain (got %d)" % d["graded"])
check(d["grades"]["good"] == 0, "and nothing is graded good again")
check(d["n_assumed"] == 0, "nothing is marked assumed")
check(ls.learner.grades.label(victim) == "bad",
      "the pose clicked during the assumption kept its real grade")

print("\n[7] the premise is tested against the model, not just asserted")
# rebuild a worked list and train on it, so there is a model to ask
for i, st_ in enumerate(poses):
    st_.viewdock_data = dict(getattr(st_, "viewdock_data", None) or {})
    st_.viewdock_data[VL.RATING_KEY] = 0
for i, st_ in enumerate(poses[:20]):
    st_.viewdock_data[VL.RATING_KEY] = 1 if i % 2 else 2
ls.forget_assumed()
for k in list(ls.learner.grades.keys()):
    ls.learner.grades.set(k, None)
ls.pull_viewdock_ratings()

check(L.assumption_evidence(session) is None,
      "no verdict is offered before there is a model to ask")

if ls.learner.can_train()[0]:
    ls.learner.train(epochs=12)
    ev = L.assumption_evidence(session)
    if ev is None:
        check(True, "too few poses here to judge, and it says so by "
                    "returning nothing rather than a number")
    else:
        check(ev["n_unrated"] > 0 and ev["n_human"] > 0,
              "the check compares %d unrated against %d graded"
              % (ev["n_unrated"], ev["n_human"]))
        check(all(0.0 <= v["auc"] <= 1.0 for v in ev["versus"].values()),
              "every comparison is a real AUC: %s"
              % {k: round(v["auc"], 2) for k, v in ev["versus"].items()})
        check("good" not in ev["versus"] or ev["versus"]["good"]["n"] >= 10,
              "and a group too small to judge is left out rather than "
              "reported as noise")

print("\n[8] assuming on an empty collection is refused, not silently useless")
from chimerax.core.errors import UserError
L.clear(session)
try:
    hitpick_learn_assume(session, "good")
    check(False, "should have refused with nothing featurised")
except UserError as e:
    check("collect" in str(e), "says what to do instead: %s" % e)

print()
if failures:
    print("FAILURES (%d):" % len(failures))
    for f in failures:
        print("   -", f)
    sys.exit(1)
print("ASSUME TESTS OK")
