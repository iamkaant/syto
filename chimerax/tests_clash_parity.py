"""Our clash detector vs ChimeraX's, on real docked poses.

The core cannot call chimerax.clashes at runtime (the web front end has no
ChimeraX), so the two implementations must be checked against each other
here. Thresholds are deliberately identical: clash_threshold 0.6,
hbond_allowance 0.4, bond_separation 4.

Perfect agreement is not expected. ChimeraX uses its own per-element vdW
radii while the core uses Bondi, and the two differ for hydrogen in
particular, so a contact sitting exactly on the threshold can fall either
way. What must hold is that the two broadly agree and that neither finds
severe clashes the other misses entirely.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_clash_parity.py
"""
import os
import sys

from chimerax.core.commands import run

A = "/Users/ak87/ownCloud/owncloud/Calculations/TMP/AutoParty/"
REC = A + "autoparty/inputs/rec.crg.pdb"
HERE = os.path.dirname(os.path.abspath(__file__))
POSES = os.path.join(HERE, "..", "..", "poses3.mol2")
if not os.path.exists(POSES):
    POSES = A + "autoparty/inputs/more_opcore_enum_asp_h2o_hbond.mol2"

if not os.path.exists(REC):
    print("sample data unavailable; skipping")
    sys.exit(0)

run(session, "open %s" % REC)
run(session, "open %s format mol2 showTool false" % POSES)

from chimerax.atomic import AtomicStructure, Atoms
from chimerax.clashes.clashes import find_clashes as cx_find_clashes
from chimerax.clashes.settings import defaults as cx_defaults

# Import the bundle FIRST: it puts its vendored copy of the core on sys.path,
# so `hitpicking` is only importable afterwards.
from chimerax.hitpickingtool.adapter import from_structures
from hitpicking.strain import (find_clashes, DEFAULT_CLASH_OVERLAP,
                               DEFAULT_HBOND_ALLOWANCE,
                               DEFAULT_BOND_SEPARATION)

print("thresholds  ours: %.2f / %.2f / %d" % (
    DEFAULT_CLASH_OVERLAP, DEFAULT_HBOND_ALLOWANCE, DEFAULT_BOND_SEPARATION))
print("        chimerax: %.2f / %.2f / %d" % (
    cx_defaults["clash_threshold"], cx_defaults["clash_hbond_allowance"],
    cx_defaults["bond_separation"]))
same_defaults = (
    abs(cx_defaults["clash_threshold"] - DEFAULT_CLASH_OVERLAP) < 1e-9
    and abs(cx_defaults["clash_hbond_allowance"] - DEFAULT_HBOND_ALLOWANCE) < 1e-9
    and cx_defaults["bond_separation"] == DEFAULT_BOND_SEPARATION)
print("defaults match:", same_defaults)

models = [m for m in session.models if isinstance(m, AtomicStructure)]
receptor = max(models, key=lambda m: m.num_residues)
ligands = [m for m in models if m is not receptor][:10]

tot_ours = tot_cx = tot_shared = 0
rows = []
for lig in ligands:
    ms, atoms = from_structures([receptor, lig], ligand_structures=[lig])
    ours = find_clashes(ms)
    ours_pairs = set()
    for c in ours:
        a, b = atoms[c.i], atoms[c.j]
        ours_pairs.add(frozenset((a, b)))

    cx = cx_find_clashes(session, lig.atoms, inter_model=True,
                         intra_model=False, inter_submodel=False)
    cx_pairs = set()
    for a, partners in cx.items():
        for b in partners:
            if a.structure is lig and b.structure is receptor:
                cx_pairs.add(frozenset((a, b)))

    shared = ours_pairs & cx_pairs
    tot_ours += len(ours_pairs)
    tot_cx += len(cx_pairs)
    tot_shared += len(shared)
    rows.append((lig.name[:22], len(ours_pairs), len(cx_pairs), len(shared)))

print()
print("%-24s %6s %6s %6s" % ("pose", "ours", "cx", "both"))
for r in rows:
    print("%-24s %6d %6d %6d" % r)
print("%-24s %6d %6d %6d" % ("TOTAL", tot_ours, tot_cx, tot_shared))

failures = []
if not same_defaults:
    failures.append("thresholds differ from ChimeraX")
if tot_cx == 0 and tot_ours == 0:
    print("\nneither tool found a clash; nothing to compare")
else:
    union = tot_ours + tot_cx - tot_shared
    agreement = tot_shared / float(union) if union else 1.0
    print("\nagreement (intersection / union): %.0f%%" % (100 * agreement))
    if agreement < 0.6:
        failures.append("agreement with ChimeraX only %.0f%%" % (100 * agreement))

print()
if failures:
    print("FAILURES:", failures)
    sys.exit(1)
print("CLASH PARITY OK")
