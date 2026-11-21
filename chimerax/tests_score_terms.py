"""Choosing the docking score terms the model learns from.

The seven score slots in the descriptor block were wired to DOCK 3's term
names. A mol2 from Glide, Vina, GOLD or rDock carries different names, so
every one of those slots held zero - and nothing crashed, nothing warned.
The model trained, predicted and was simply deaf to the number its user
trusts most.

What is pinned here is the ChimeraX side of the fix: the command, the
panel's view of it, that changing the pick does not force a re-analysis,
and - separately and just as important - that our OWN numeric columns
never reach the chooser. HP Pred Rank is the model's output; offering it
as an input would be target leakage in one click.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_score_terms.py
"""
import json
import os
import sys
import types

import numpy as np

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

import chimerax.hitpickingtool.learn as L
import chimerax.hitpickingtool.viewdock_link as VL
from chimerax.hitpickingtool import adapter
from chimerax.hitpickingtool.learn_tool import LearnTool
from chimerax.hitpickingtool.state import get_state
from hitpicking.al import scoreterms as S
from hitpicking.al.features import SCORE_SLICE

print("[1] a DOCK file is recognised, and its bookkeeping is not offered")
ls = L.state(session)
info = ls.suggest_score_terms()
check(info["source"] == "dock", "recognised as %s: %s"
      % (info["source"], info["why"]))
check(tuple(info["terms"]) == S.DOCK_TERMS,
      "picks DOCK's own seven terms")
rejected = {r["name"] for r in info["available"] if not r["usable"]}
for junk in ("Number", "Rank", "Setnum", "Matchnum"):
    check(junk in rejected, "%s is not offered as a feature" % junk)
check(ls.learner.score_terms == S.DOCK_TERMS,
      "and that is what a fresh learner starts with")

print("\n[2] our own columns never reach the chooser")
# Write the full set of HP columns onto a pose, exactly as a trained
# session would, then ask what score terms the poses carry.
poses = VL.rated_structures(session)
check(len(poses) > 10, "%d pose(s) loaded to work with" % len(poses))
p0 = poses[0]
data = getattr(p0, "viewdock_data", None) or {}
data.update({VL.COL_PRANK: 1.73, VL.COL_UNC: 0.21, VL.COL_HBONDS: 3,
             VL.COL_CLASH: 0.4, VL.RATING_KEY: 3})
p0.viewdock_data = data
got = adapter.pose_scores(p0)
leaked = sorted(k for k in got if k.startswith("HP ") or k == "rating")
check(not leaked, "no HP column leaked into the score terms: %s" % leaked)
check("Total Energy" in got, "while the docking program's own terms do")
offered = {r["name"] for r in ls.suggest_score_terms(refresh=True)["available"]}
check(VL.COL_PRANK not in offered,
      "the model's own prediction is not offered as an input to itself")

print("\n[3] the command reports without changing anything")
was = list(ls.learner.score_terms)
r = run(session, "hitpick learn terms")
json.dumps(r)
check(list(r["terms"]) == was, "reporting leaves the pick alone")
check(len(r["available"]) > len(r["terms"]),
      "and lists everything the poses carry (%d term(s))"
      % len(r["available"]))

print("\n[4] changing the pick rewrites the collected poses in place")
run(session, "hitpick learn collect limit 24")
ls = L.state(session)
keys = list(ls.learner.features)
check(len(keys) >= 8, "%d pose(s) featurised" % len(keys))
check(len(ls.learner.scores) == len(keys),
      "each one kept its score dictionary, so no re-analysis is needed")
check(any(np.any(ls.learner.features[k][SCORE_SLICE]) for k in keys),
      "the score slots are actually carrying numbers")

before = {k: ls.learner.features[k].copy() for k in keys}
r = run(session, "hitpick learn terms Electrostatic, Max Strain")
json.dumps(r)
check(tuple(ls.learner.score_terms) == ("Electrostatic", "Max Strain"),
      "the pick changed to %s" % (list(ls.learner.score_terms),))
check(r["changed"] == len(keys),
      "%d pose(s) rewritten without re-analysing" % r["changed"])
same_outside = all(
    np.array_equal(np.delete(before[k], np.s_[SCORE_SLICE]),
                   np.delete(ls.learner.features[k], np.s_[SCORE_SLICE]))
    for k in keys)
check(same_outside, "and nothing outside the score slots moved")
moved = any(not np.array_equal(before[k][SCORE_SLICE],
                               ls.learner.features[k][SCORE_SLICE])
            for k in keys)
check(moved, "while the score slots did")

print("\n[5] a trained model is discarded, and said to be")
for i, k in enumerate(keys):
    ls.learner.grade(k, i % 3)
run(session, "hitpick learn train epochs 8 wait true")
check(ls.learner.committee is not None, "trained")
r = run(session, "hitpick learn terms auto")
check(ls.learner.committee is None,
      "the committee was fitted to different columns, so it is gone")
check(r["retrain"] is True, "and the command says so: retrain=%r"
      % r["retrain"])
check(len(ls.learner.grades) == len(keys), "the grades survived")
check(tuple(ls.learner.score_terms) == S.DOCK_TERMS,
      "'auto' put DOCK's terms back")

print("\n[6] 'none' is a real choice, not an error")
r = run(session, "hitpick learn terms none")
check(ls.learner.score_terms == (), "no terms selected")
check(all(not np.any(ls.learner.features[k][SCORE_SLICE]) for k in keys),
      "every score slot is zero")
run(session, "hitpick learn terms auto")

print("\n[7] a term no pose carries is refused, with a way forward")
from chimerax.core.errors import UserError
try:
    run(session, "hitpick learn terms r_i_glide_gscore")
    check(False, "a term from another program was accepted in silence")
except UserError as e:
    check("no loaded pose carries" in str(e),
          "refused and explained: %s" % str(e).split(".")[0])

print("\n[8] the panel sees the pick, and warns when it is deaf")
stub = types.SimpleNamespace(session=session, _strategy="uncertainty",
                             _auto_advance=True, _hp=get_state(session))
snap = types.MethodType(LearnTool.snapshot, stub)()
json.dumps(snap)
t = snap["terms"]
check(t is not None, "the snapshot carries a terms block")
check(t["chosen"] == list(S.DOCK_TERMS), "showing the current pick")
check(t["live"] == len(S.DOCK_TERMS),
      "and that all %d are present in these poses" % t["live"])
check(t["program"] == "DOCK 3" and t["verified"],
      "named, and marked as verified rather than guessed")

# Now the case the whole feature exists for: terms that match nothing.
ls.learner.set_score_terms(["Electrostatic"])
ls.learner.score_terms = ("r_i_docking_score",)   # as a Glide model would be
snap = types.MethodType(LearnTool.snapshot, stub)()
check(snap["terms"]["live"] == 0,
      "the panel can tell that no selected term is in these poses")
check(snap["terms"]["suggested"] == list(S.DOCK_TERMS),
      "and has something to suggest instead")

print("\n[9] a non-DOCK table is handled by name, not by luck")
# A pose set as smina would leave it. No table entry is needed for the
# generic path, but this one is known, so it should be named.
samples = [{"minimizedAffinity": -8.0 - i * 0.1,
            "minimizedRMSD": 0.4 + i * 0.01,
            "pose": float(i)} for i in range(20)]
info = S.suggest(samples)
check(info["source"] == "vina", "recognised as %s" % info["source"])
check(info["terms"][0] == "minimizedAffinity",
      "the primary score leads: %s" % (list(info["terms"]),))
check("pose" not in info["terms"], "and the index is left out")

print()
if failures:
    print("FAILURES (%d):" % len(failures))
    for f in failures:
        print("   -", f)
    sys.exit(1)
print("SCORE TERM TESTS OK")
