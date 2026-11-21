"""Verify labels are actually created and carry the right text.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_labels.py

These two bugs were both invisible to the existing tests because nothing
read back what was drawn:
  * "Label pKBHX" deleted every label and added none (per-atom `label`
    COMMANDS built from spec strings failed silently for ligands with no
    chain id);
  * switching angles on changed nothing (interaction lines were labelled
    per GROUP with a fixed "{0.length}" format, so per-line angles could
    never appear).
"""
import os
import re
import sys

from chimerax.core.commands import run
from chimerax.label.label3d import labels_model

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# The sample data is not in this repository and its location is not this
# laptop's home directory. tools/samples.py resolves it; a test that
# cannot find it skips.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools"))
from samples import AUTOPARTY          # noqa: E402

A = AUTOPARTY + "/"
REC = A + "autoparty/inputs/rec.crg.pdb"
LIG = os.path.join(ROOT, "..", "AutoParty", "autoparty", "inputs",
                   "more_opcore_enum_asp_h2o_hbond.mol2")

failures = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def all_label_texts(session):
    """Every label text currently in the session, with its parent model.

    ObjectLabels exposes labels() - there is no `all_labels` attribute, and
    getattr(..., "all_labels", []) silently returns nothing, which is how the
    first version of this test "passed" against broken code.
    """
    out = []
    for m in list(session.models):
        lm = labels_model(m, create=False)
        if lm is None:
            continue
        try:
            labs = lm.labels()
        except Exception:
            continue
        for lbl in labs:
            out.append((m.name, lbl.text))
    return out


if not os.path.exists(REC):
    print("sample data unavailable; skipping")
    sys.exit(0)

# a small multi-pose file is built by the e2e helper; fall back to the big one
pose_file = os.path.join(HERE, "..", "..", "poses3.mol2")
if not os.path.exists(pose_file):
    pose_file = LIG

run(session, "open %s" % REC)
run(session, "open %s format mol2 showTool false" % pose_file)

from chimerax.atomic import AtomicStructure
ligs = [m for m in session.models
        if isinstance(m, AtomicStructure) and m.num_residues <= 3]
run(session, "hitpick setup")
run(session, "hitpick analyze #%s" % ligs[0].id_string)

import chimerax.hitpickingtool.state as hpstate
st = hpstate.get_state(session)

print("\n[1] interaction labels show lengths by default")
texts = [t for _, t in all_label_texts(session)]
print("    sample:", texts[:4])
DIST = re.compile(r"\d+\.\d{2}")
ANGLE = re.compile(r"(DHA|HAX|CXA|XAY|tilt|off)\d")

check(len(texts) > 0, "some interaction labels exist")
check(any(DIST.search(t) for t in texts), "a length appears in the label text")
check(not any(ANGLE.search(t) for t in texts), "no angles while angles are off")

print("\n[2] switching angles ON changes the label text")
run(session, "hitpick angles true")
texts_ang = [t for _, t in all_label_texts(session)]
print("    sample:", texts_ang[:4])
check(any(ANGLE.search(t) for t in texts_ang), "angles now appear in the labels")
check(texts_ang != texts, "label text actually changed")

print("\n[3] switching angles OFF removes them again")
run(session, "hitpick angles false")
texts_off = [t for _, t in all_label_texts(session)]
check(not any(ANGLE.search(t) for t in texts_off), "angles gone again")

print("\n[4] pKBHX atom labels appear and do not wipe the others")
before = len(all_label_texts(session))
run(session, "hitpick labels pKBHX")
after = all_label_texts(session)
pk = [t for _, t in after if "pKBHX" in t]
print("    pKBHX labels:", pk[:4])
check(len(pk) > 0, "pKBHX labels were created")
check(len(after) > before, "total label count grew (nothing was wiped)")
check(any(DIST.search(t) and "pKBHX" not in t for _, t in after),
      "interaction length labels survived the atom labelling")

print("\n[5] 'labels off' removes only our atom labels")
run(session, "hitpick labels off")
after_off = all_label_texts(session)
check(not any("pKBHX" in t for _, t in after_off), "pKBHX labels removed")
check(any(DIST.search(t) and "pKBHX" not in t for _, t in after_off),
      "interaction labels still present after clearing atom labels")

print("\n[6] lengths toggle off clears interaction labels")
run(session, "hitpick lengths false")
check(not any(DIST.search(t) and "pKBHX" not in t
              for _, t in all_label_texts(session)),
      "interaction labels gone when lengths are off")

print()
if failures:
    print("FAILURES (%d):" % len(failures))
    for f in failures:
        print("   -", f)
    sys.exit(1)
print("LABEL TESTS OK")
