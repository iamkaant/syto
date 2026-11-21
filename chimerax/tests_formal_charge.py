"""IDATM's '+' is not a formal charge.

ChimeraX types neutral trimethylamine as N3+ exactly as it types a
protonated amine. Reading that '+' as +1 marked every tertiary amine
cationic, which became a phantom ammonium group and, through it, a phantom
salt bridge. Reported on ZINCmn0000027Q3g, whose ring nitrogen is neutral.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_formal_charge.py
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


def build(atoms, name):
    lines = []
    for i, (el, an, x, y, z) in enumerate(atoms, 1):
        lines.append("HETATM%5d %-4s LIG A   1    %8.3f%8.3f%8.3f"
                     "  1.00  0.00          %2s" % (i, an, x, y, z, el))
    lines.append("END")
    p = "/tmp/_fc_%s.pdb" % name
    open(p, "w").write("\n".join(lines) + "\n")
    return run(session, "open %s" % p)[0]


from chimerax.hitpickingtool.adapter import from_structures
from hitpicking.perceive.features import perceive_all

print("[1] a neutral three-coordinate amine is not charged")
# The molecule carries a hydrogen on purpose. With NO hydrogens anywhere,
# perception deliberately assumes amines are protonated and says so - that
# rule is correct, and a hydrogen-free test molecule would be testing it
# rather than the formal-charge logic.
m = build([("N", "N1", 0, 0, 0), ("C", "C1", 1.47, 0, 0),
           ("C", "C2", -.7, 1.27, 0), ("C", "C3", -.7, -1.27, 0),
           ("H", "H1", 1.84, 0.93, 0.26)], "neutral")
n = [a for a in m.atoms if a.element.name == "N"][0]
ms, _ = from_structures([m], ligand_structures=[m])
print("    ChimeraX idatm:", n.idatm_type, "| connections:", len(n.neighbors))
check(n.idatm_type.endswith("+"), "ChimeraX really does type it N3+")
check(int(ms.formal_charge[0]) == 0, "we assign it formal charge 0")
feats = perceive_all(ms)
check(not [x for x in feats["charged"] if x.kind == "cation"],
      "no cationic group is perceived")
run(session, "close #%s" % m.id_string)

print("\n[2] a four-coordinate nitrogen IS charged")
m = build([("N", "N1", 0, 0, 0), ("C", "C1", 1.47, 0, 0),
           ("C", "C2", -.7, 1.27, 0), ("C", "C3", -.7, -1.27, 0),
           ("H", "H1", 0, 0, 1.01)], "charged")
ms, _ = from_structures([m], ligand_structures=[m])
check(int(ms.formal_charge[0]) == 1, "we assign it formal charge +1")
check([x for x in perceive_all(ms)["charged"] if x.kind == "cation"],
      "a cationic group is perceived")
run(session, "close #%s" % m.id_string)

print("\n[3] the reported ligand: one charge, on the right nitrogen")
# A saved session kept beside the repository rather than in it: it is
# research data, not source. Absent from a fresh clone, and skipped then.
CXS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "charge_plus1_H20_H25-SV.cxs")
if not os.path.exists(CXS):
    print("    session file unavailable; skipping")
else:
    run(session, "open %s" % CXS)
    models = [x for x in session.models if isinstance(x, AtomicStructure)]
    rec = max(models, key=lambda x: x.num_residues)
    lig = [x for x in models if "ZINCmn0000027Q3g" in x.name]
    if not lig:
        print("    ligand not in the session; skipping")
    else:
        lig = lig[0]
        ms, _ = from_structures([rec, lig], ligand_structures=[lig])
        f = perceive_all(ms)
        cats = [x for x in f["charged"] if x.kind == "cation"
                and all(ms.is_ligand[a] for a in x.atoms)]
        stated = (getattr(lig, "viewdock_data", {}) or {}).get("Ligand Charge")
        print("    DOCK states Ligand Charge =", stated)
        check(len(cats) == 1,
              "exactly one cationic group, matching DOCK (got %d)" % len(cats))
        if cats:
            names = {str(ms.name[a]).strip() for a in cats[0].atoms}
            check("N2" in names, "the charge is on N2, the 4-connection N")
            check("N1" not in names,
                  "the neutral ring nitrogen N1 is NOT called cationic")

print()
if failures:
    print("FAILURES (%d):" % len(failures))
    for x in failures:
        print("   -", x)
    sys.exit(1)
print("FORMAL CHARGE TESTS OK")
