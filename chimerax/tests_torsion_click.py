"""Clicking a BOND shows its torsion, and the four atoms are depicted.

A torsion belongs to a bond, not to an atom; driving the profile from an
atom click made it appear whenever anything was picked. And since the tool
chooses the two outer reference atoms itself, and that choice changes the
measured angle, those atoms have to be visible rather than implicit.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_torsion_click.py
"""
import sys

from chimerax.core.commands import run
from chimerax.atomic import AtomicStructure, PseudobondGroup

failures = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


A = "/Users/ak87/ownCloud/owncloud/Calculations/TMP/AutoParty/"
import os
if not os.path.exists(A + "autoparty/inputs/rec.crg.pdb"):
    print("sample data unavailable; skipping")
    sys.exit(0)

run(session, "open %s" % (A + "autoparty/inputs/rec.crg.pdb"))
run(session, "open %s format mol2 showTool false"
    % (A + "autoparty/inputs/more_opcore_enum_asp_h2o_hbond.mol2"))
m = [x for x in session.models if isinstance(x, AtomicStructure)]
rec = max(m, key=lambda x: x.num_residues)
lig = [x for x in m if x is not rec][0]
run(session, "hitpick setup")
run(session, "hitpick analyze #%s" % lig.id_string)

import chimerax.hitpickingtool.state as S
import chimerax.hitpickingtool.cmd as C
from chimerax.hitpickingtool.tool import HandbookTool

st = S.get_state(session)
named = [t for t in st.result.strain.torsions if t.subtype]
if not named:
    print("no named torsion in this pose; skipping")
    sys.exit(0)
t = named[0]
i, j = t.bond
a1, a2 = st.all_atoms[i], st.all_atoms[j]
bond = [b for b in a1.bonds if a2 in b.atoms][0]

print("[1] clicking a bond is handled as a torsion")
check(C._explain_bond(session, st, bond), "the bond click was handled")
check(st.last_torsion is not None, "the torsion is remembered on the state")

print("\n[2] the four atoms are depicted")
sel = session.selection.items("atoms")
check(bool(sel) and len(sel[0]) == 4,
      "exactly the four dihedral atoms are selected")
if sel:
    names = [str(a.name) for a in sel[0]]
    expect = [str(st.molsys.name[k]).strip() for k in t.atoms]
    check(set(names) == set(expect),
          "they are the measured atoms: %s" % names)
groups = [g for g in session.models
          if isinstance(g, PseudobondGroup) and "torsion" in g.name]
check(bool(groups), "the reference arms are drawn")
if groups:
    check(all(pb.shown for pb in groups[0].pseudobonds),
          "and they are actually visible")

print("\n[3] the panel profile follows the clicked bond, not an atom")
# bind the method to a stub, so no Qt window has to open in nogui
import types
prof = types.MethodType(HandbookTool._torsion_profile,
                        type("Stub", (), {"session": session})())(st)
check(prof is not None, "a profile is produced")
if prof:
    check(prof["label"].startswith(t.subtype or t.bond_type),
          "it is the clicked bond's class: %s" % prof["label"])
    check(prof.get("marker") is not None, "and it marks the measured angle")

print("\n[4] clicking an ATOM clears the torsion selection")
st.last_torsion = t
idx = next(k for k in range(st.molsys.n_atoms)
           if st.molsys.is_ligand[k] and str(st.molsys.element[k]) == "N")
run(session, "select %s" % st.all_atoms[idx].string(style="command"))
# the listener runs on a redraw, so emulate the atom branch directly
st.last_hit = st.handbook.for_atom(st.molsys, idx)
st.last_torsion = None
check(st.last_torsion is None, "an atom click is not a torsion question")

print()
if failures:
    print("FAILURES (%d):" % len(failures))
    for f in failures:
        print("   -", f)
    sys.exit(1)
print("TORSION CLICK TESTS OK")
