"""The model's predictions, where a party can actually use them.

A 25-row list in a panel is the wrong shape for picking from a few
thousand poses. The predictions therefore go into ViewDock's own table,
which means obeying its contract: every key in viewdock_data must have a
column titled exactly display_key(key), or the next row click raises
KeyError. That failure has bitten this bundle before, so it is pinned.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_predict_columns.py
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


A = "/Users/ak87/ownCloud/owncloud/Calculations/TMP/AutoParty/"
if not os.path.exists(A + "autoparty/inputs/rec.crg.pdb"):
    print("sample data unavailable; skipping")
    sys.exit(0)

run(session, "open %s" % (A + "autoparty/inputs/rec.crg.pdb"))
run(session, "open %s format mol2" % (A + "autoparty/inputs/more_opcore_enum_asp_h2o_hbond.mol2"))
run(session, "hitpick setup")

import chimerax.hitpickingtool.viewdock_link as VL
from chimerax.hitpickingtool import learn

print("[1] every prediction column obeys ViewDock's key/title contract")
for title in VL.PREDICTION_COLUMNS:
    check(VL.display_key_identity(title),
          "%r survives display_key unchanged" % title)
    check(title in VL.COLUMNS, "%r is declared in COLUMNS" % title)

print("\n[2] nothing is written before the model is trained")
learn.clear(session)
n, why = learn.apply_predictions(session)
check(n == 0, "no rows written")
check("not been trained" in why, "and it says why: %s" % why)

# ViewDock's tool cannot start without a GUI, so the table itself is
# stubbed here, exactly as tests_viewdock_contract stubs the structures.
# What is under test is our side of the contract: that we create a column
# before writing its key, and write nothing we have not created.
class StubColumn:
    def __init__(self, title):
        self.title = title


class StubTable:
    def __init__(self, refuse=()):
        self.columns = []
        self.refuse = set(refuse)
        self.updated = []

    def add_column(self, title, func, **kw):
        if title in self.refuse:
            raise RuntimeError("this table will not take %r" % title)
        self.columns.append(StubColumn(title))

    def update_column(self, col, **kw):
        self.updated.append(col.title)


class StubTool:
    def __init__(self, structures, refuse=()):
        self.struct_table = StubTable(refuse)
        self.vd_structures = list(structures)

    display_key = staticmethod(lambda k: k)


print("\n[3] after training, every featurised pose gets a row value")
run(session, "hitpick learn collect limit 24")
st = learn.state(session)
keys = list(st.learner.features)
check(len(keys) >= 8, "%d poses featurised to work with" % len(keys))
for i, k in enumerate(keys):
    st.learner.grade(k, i % 3)
run(session, "hitpick learn train epochs 10 wait true")
check(st.learner.committee is not None, "the model trained")

poses = [st.keys[k] for k in keys]
stub = StubTool(poses)
real_find = VL.find_viewdock
VL.find_viewdock = lambda s: stub
try:
    written, why = learn.apply_predictions(session)
    check(written == len(keys),
          "all %d featurised poses written (got %d) %s"
          % (len(keys), written, why))

    scored = [p for p in poses
              if VL.COL_PRED in (getattr(p, "viewdock_data", None) or {})]
    check(len(scored) == written, "the count matches the poses carrying a value")

    print("\n[4] the values are the kind a table can sort")
    vals = [p.viewdock_data for p in scored]
    check(len(vals) > 0, "there are values to check (not a vacuous pass)")
    check(all(isinstance(d[VL.COL_PRANK], int) for d in vals),
          "HP Pred Rank is an int, so the column orders numerically")
    check(all(d[VL.COL_PRED] in ("bad", "maybe", "good") for d in vals),
          "HP Pred is a readable grade: %s"
          % sorted({d[VL.COL_PRED] for d in vals}))
    check(all(isinstance(d[VL.COL_UNC], float) for d in vals),
          "HP Uncert is a float")
    check(all(d[VL.COL_PRED] == ("bad", "maybe", "good")[d[VL.COL_PRANK]]
              for d in vals),
          "the word and the rank agree with each other")
    check(set(stub.struct_table.updated) >= set(VL.PREDICTION_COLUMNS),
          "the table was told to redraw each column: %s"
          % sorted(set(stub.struct_table.updated)))

    print("\n[5] no key is written that has no column - the KeyError contract")
    titles = {c.title for c in stub.struct_table.columns}
    orphan = sorted({k for d in vals for k in d
                     if k not in titles and k != VL.RATING_KEY
                     and k not in ("Name",)})
    # keys ViewDock itself put there are its own business; ours must resolve
    ours = [k for k in orphan if k.startswith("HP ")]
    check(not ours, "no HP key lacks a column: %s" % ours)

    print("\n[6] a table that refuses a column gets no value for it")
    for p in poses:
        p.viewdock_data.pop(VL.COL_UNC, None)
    stub2 = StubTool(poses, refuse=[VL.COL_UNC])
    VL.find_viewdock = lambda s: stub2
    learn.apply_predictions(session)
    leaked = [p for p in poses
              if VL.COL_UNC in (getattr(p, "viewdock_data", None) or {})]
    check(not leaked,
          "%d pose(s) carry HP Uncert though the column was refused"
          % len(leaked))
    check(all(VL.COL_PRED in p.viewdock_data for p in poses),
          "the columns that WERE accepted still got their values")

    print("\n[7] predictions can be cleared again")
    VL.find_viewdock = lambda s: stub2
    removed = VL.clear_predictions(session)
    check(removed > 0, "%d value(s) cleared" % removed)
    left = [p for p in poses
            if VL.COL_PRED in (getattr(p, "viewdock_data", None) or {})]
    check(not left, "no pose still carries a prediction")
finally:
    VL.find_viewdock = real_find

print("\n[8] with no ViewDock it says so rather than failing")
import json
r = run(session, "hitpick learn apply")
json.dumps(r)
check(isinstance(r, dict) and r.get("written") == 0,
      "returns %r" % r)
check("not running" in (r.get("why") or ""), "and explains why")

print()
if failures:
    print("FAILURES (%d):" % len(failures))
    for f in failures:
        print("   -", f)
    sys.exit(1)
print("PREDICT COLUMN TESTS OK")
