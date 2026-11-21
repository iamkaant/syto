"""Clashes must be DRAWN, not just counted.

Detecting a clash and reporting it in the log is not much use when you are
looking at the pose; the first version of this module had no visual output
at all. This checks that clashes become pseudobonds, that severe ones are
separated from ordinary ones, and that the toggle really removes them.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_clash_draw.py
"""
import sys

failures = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)

import os
from chimerax.core.commands import run
from chimerax.atomic import AtomicStructure, PseudobondGroup

# The sample data is not in this repository and its location is not this
# laptop's home directory. tools/samples.py resolves it; a test that
# cannot find it skips.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools"))
from samples import AUTOPARTY          # noqa: E402

A = AUTOPARTY + "/"
run(session, "open %s" % (A + "autoparty/inputs/rec.crg.pdb"))
run(session, "open %s format mol2 showTool false"
    % (A + "autoparty/inputs/more_opcore_enum_asp_h2o_hbond.mol2"))

models = [m for m in session.models if isinstance(m, AtomicStructure)]
receptor = max(models, key=lambda m: m.num_residues)
# SD_295648_3521 is the pose ChimeraX and we both found 5 clashes in
target = [m for m in models if "3521" in m.name]
lig = target[0] if target else [m for m in models if m is not receptor][0]

run(session, "hitpick setup")
run(session, "hitpick analyze #%s" % lig.id_string)

def clash_groups():
    return {m.name: m.num_pseudobonds for m in session.models
            if isinstance(m, PseudobondGroup) and "clash" in m.name}


import chimerax.hitpickingtool.state as hpstate
st = hpstate.get_state(session)

print("\n[1] clashes are drawn")
g = clash_groups()
print("    groups:", g)
check(bool(g), "clashes appear as pseudobond groups")
check(sum(g.values()) == len(st.result.strain.clashes),
      "every detected clash is drawn (%d drawn, %d detected)"
      % (sum(g.values()), len(st.result.strain.clashes)))
check(any("severe" in k for k in g),
      "severe clashes get their own group, separable in the Model Panel")

print("\n[2] the pose summary mentions them")
check("clash" in st.result.summary(), "summary mentions clashes")

print("\n[3] toggling off removes them")
run(session, "hitpick clashes false")
check(not clash_groups(), "no clash groups remain when switched off")

print("\n[4] toggling back on restores them")
run(session, "hitpick clashes true")
g2 = clash_groups()
check(g2 == g, "the same clashes come back (%s)" % g2)

print("\n[5] favourable interactions are untouched by the clash toggle")
nci = [m.name for m in session.models
       if isinstance(m, PseudobondGroup) and m.name.startswith("hitpick")
       and "clash" not in m.name]
check(bool(nci), "interaction groups still present: %s" % nci)

print()
if failures:
    print("FAILURES (%d):" % len(failures))
    for f in failures:
        print("   -", f)
    sys.exit(1)
print("CLASH DRAW TESTS OK")
