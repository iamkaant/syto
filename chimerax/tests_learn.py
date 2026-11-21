"""The `hitpick learn` namespace, end to end in ChimeraX.

Checks that nothing AL-related loads until asked, that grades flow in from
ViewDock's own rating scale, that training refuses politely when it cannot
learn anything, and that a saved model will not load under a different
feature version.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_learn.py
"""
import os
import sys

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

print("[1] nothing AL-related is loaded before it is asked for")
run(session, "open %s" % (A + "autoparty/inputs/rec.crg.pdb"))
run(session, "open %s format mol2 showTool false"
    % (A + "autoparty/inputs/more_opcore_enum_asp_h2o_hbond.mol2"))
run(session, "hitpick setup")
loaded = [m for m in sys.modules if m.startswith("hitpicking.al")]
check(not loaded, "hitpicking.al is not imported by setup (%s)" % loaded)

print("\n[2] collect analyses and featurises")
run(session, "hitpick learn collect limit 40")
import chimerax.syto.learn as L
d = L.describe(session)
print("    %s" % d)
check(d["poses"] >= 20, "poses featurised: %d" % d["poses"])
check(d["feature_version"] is not None, "the feature version is recorded")

print("\n[3] training refuses when it cannot learn anything")
ls = L.state(session)
keys = list(ls.learner.features)
for k in keys[:10]:
    ls.learner.grade(k, "bad")          # all one grade
ok, why = ls.learner.can_train()
check(not ok, "refused with: %s" % why)
check("different grades" in why, "and the reason is the useful one")

print("\n[4] with a spread of grades it trains and ranks")
for k in keys[10:20]:
    ls.learner.grade(k, "good")
for k in keys[20:26]:
    ls.learner.grade(k, "maybe")
run(session, "hitpick learn train epochs 40")
d = L.describe(session)
check(d["ready"], "a model exists")
check(d["trained_on"] >= 20, "trained on %d poses" % d["trained_on"])

rows = run(session, "hitpick learn next count 5")
check(bool(rows), "it suggests poses to grade next")
if rows:
    unc = [r["uncertainty"] for r in rows]
    check(unc == sorted(unc, reverse=True),
          "suggestions are ordered by uncertainty, most informative first")
    check(all(r["key"] not in ls.learner.grades for r in rows),
          "and it does not suggest poses already graded")

print("\n[5] ViewDock's rating scale maps straight onto the grades")
from chimerax.syto import viewdock_link as vl
check(vl.RATING_LABELS == ("(unrated)", "bad", "maybe", "good"),
      "ViewDock: %s" % (vl.RATING_LABELS,))
check(ls.learner.options == ("bad", "maybe", "good"),
      "learner:   %s" % (ls.learner.options,))

print("\n[6] a model from a different feature version is refused")
import json
import tempfile
path = os.path.join(tempfile.gettempdir(), "_hp_model.json")
run(session, "hitpick learn save %s" % path)
check(os.path.exists(path), "the model saves")
with open(path) as fh:
    blob = json.load(fh)
blob["feature_version"] = 999
with open(path, "w") as fh:
    json.dump(blob, fh)
try:
    L.load(session, path)
    check(False, "loading a stale feature version should raise")
except ValueError as e:
    check("feature version" in str(e),
          "refused clearly: %s" % str(e)[:70])

print()
if failures:
    print("FAILURES (%d):" % len(failures))
    for f in failures:
        print("   -", f)
    sys.exit(1)
print("LEARN TESTS OK")
