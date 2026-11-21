"""Atom labels come from three places; an atom can carry only one label.

All four of these were one bug report:

  * a bond marked strained whose own popup said it was not
  * clicking that bond ERASED the word "strained"
  * clicking the next bond did not bring it back
  * the "1 2 3 4" digits from every bond ever clicked stayed on screen

The first is in the torsion judgement; the rest are what happens when two
features write the same per-atom attribute over each other.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_label_layers.py
"""
import os
import sys

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
run(session, "hitpick analyze")

from chimerax.hitpickingtool import render
from chimerax.hitpickingtool.state import get_state

st = get_state(session)
atoms = st.all_atoms
L = render.LABEL_ATTR


def text(i):
    return getattr(atoms[i], L, "") or ""


print("[1] strain notes are drawn on the atoms that earned them")
render.clear_atom_labels(session, atoms)
n = render.label_strain(session, st.result.strain, atoms)
strained = [i for i in range(len(atoms)) if text(i)]
check(n > 0 and strained, "%d strain note(s) drawn" % n)
anchor = strained[0]
before = text(anchor)
check(bool(before), "an atom carries one: %r" % before)

print("\n[2] a torsion's numbers do not erase the strain note under them")
torsions = [t for t in st.result.strain.torsions if anchor in tuple(t.atoms)]
if not torsions:
    # fall back to any torsion, and check the digits land somewhere
    torsions = list(st.result.strain.torsions[:1])
    overlap = False
else:
    overlap = True
t = torsions[0]
render.show_torsion(session, t, atoms)
digits = {i: text(i) for i in (int(x) for x in t.atoms)}
check(all(d for d in digits.values()), "all four atoms numbered: %s"
      % list(digits.values()))
if overlap:
    check(before in text(anchor),
          "and the strain note is still there beside the number: %r"
          % text(anchor))
else:
    check(text(anchor) == before,
          "and an untouched strain note is unchanged: %r" % text(anchor))

print("\n[3] clicking the next bond takes the old numbers away")
others = [x for x in st.result.strain.torsions
          if tuple(x.atoms) != tuple(t.atoms)]
check(bool(others), "there is a second torsion to click")
if others:
    t2 = others[0]
    render.show_torsion(session, t2, atoms)
    old_only = [i for i in (int(x) for x in t.atoms)
                if i not in {int(x) for x in t2.atoms}]
    left = {i: text(i) for i in old_only if text(i).split(" · ")[0] in
            ("1", "2", "3", "4")}
    check(not left, "no digits left on the bond clicked before: %s" % left)
    check(all(text(i) for i in (int(x) for x in t2.atoms)),
          "and the new bond is numbered")

print("\n[4] and the strain note survives all of it")
if overlap:
    check(before in text(anchor),
          "the strain note is still readable: %r" % text(anchor))
else:
    check(text(anchor) == before, "the strain note is untouched")
render.clear_torsion(session, atoms)
check(text(anchor) == before,
      "clearing the torsion leaves the strain note alone: %r" % text(anchor))

print("\n[5] handbook values are a third layer, not a replacement")
render.annotate_atoms(session, {anchor: "pKBHX 1.86"}, atoms)
check("pKBHX" in text(anchor) and before in text(anchor),
      "value and strain note share the atom: %r" % text(anchor))
render.clear_label_layer(session, atoms, render.LAYER_VALUE)
check("pKBHX" not in text(anchor) and before in text(anchor),
      "dropping the values keeps the strain note: %r" % text(anchor))

print("\n[6] clear_atom_labels still clears everything")
render.clear_atom_labels(session, atoms)
check(not any(text(i) for i in range(len(atoms))),
      "nothing is labelled after a full clear")

print("\n[7] the 3D flag and the popup agree about what is strained")
# An exact QM class is a measurement; the geometric rule is a proxy for it.
# A torsion whose own QM scan says it costs little must not be left flagged.
from hitpicking.strain import torsions as T
bad = []
checked = 0
for x in st.result.strain.torsions:
    view = T.profile_view(x.bond_type, x.subtype, x.ortho_substituted,
                          x.angle)
    if view is None or not view.get("exact"):
        continue
    # the energy lives on the marker, not at the top level - reading
    # view["energy"] here made this check pass without testing anything
    cost = (view.get("marker") or {}).get("energy")
    if cost is None:
        continue
    checked += 1
    if x.flagged and cost < T.QM_NOTE_KCAL:
        bad.append((x.subtype or x.bond_type, x.assessment, round(cost, 2)))
check(checked > 0, "%d torsion(s) had an exact QM energy to check against "
      "(not a vacuous pass)" % checked)
check(not bad,
      "none is flagged while its own exact QM scan says it is cheap: %s"
      % bad[:4])

print("\n[8] a flagged torsion's note never says it is not strain")
contradictions = []
for x in st.result.strain.torsions:
    n = (x.note or "").lower()
    if not x.flagged:
        continue
    # "not strain" is only allowed where it is explicitly framed as the
    # reading being overruled
    if "not strain" in n and "would pass this" not in n:
        contradictions.append((x.assessment, (x.note or "")[:90]))
check(not contradictions,
      "no note contradicts the flag above it: %s" % contradictions[:2])

print()
if failures:
    print("FAILURES (%d):" % len(failures))
    for f in failures:
        print("   -", f)
    sys.exit(1)
print("LABEL LAYER TESTS OK")
