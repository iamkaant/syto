"""End-to-end test of the installed bundle inside ChimeraX."""
import os
from chimerax.core.commands import run

SCRATCH = os.path.dirname(os.path.abspath(__file__))
A = "/Users/ak87/ownCloud/owncloud/Calculations/TMP/AutoParty/"
REC = A + "autoparty/inputs/rec.crg.pdb"
LIG = os.path.join(SCRATCH, "poses3.mol2")

print("=" * 72)
run(session, "open %s" % REC)
run(session, "open %s format mol2 showTool false" % LIG)
print("models:", [(m.id_string, m.name, getattr(m, 'num_atoms', '?'))
                 for m in session.models])

print("\n--- hitpick setup ---")
run(session, "hitpick setup")

from chimerax.atomic import AtomicStructure
ligs = [m for m in session.models
        if isinstance(m, AtomicStructure) and m.num_residues <= 3]
print("ligand models:", [(m.id_string, m.name) for m in ligs])

for lig in ligs[:3]:
    print("\n--- hitpick analyze #%s ---" % lig.id_string)
    run(session, "hitpick analyze #%s" % lig.id_string)
    run(session, "hitpick report")

print("\n--- pseudobond groups drawn ---")
from chimerax.atomic import PseudobondGroup
for m in session.models:
    if isinstance(m, PseudobondGroup) and m.name.startswith("hitpick"):
        print("   %-34s %d bonds" % (m.name, m.num_pseudobonds))

print("\n--- toggles: hide stacking ---")
run(session, "hitpick hide pi_stack_parallel pi_stack_tshaped")
for m in session.models:
    if isinstance(m, PseudobondGroup) and m.name.startswith("hitpick"):
        print("   %-34s %d bonds" % (m.name, m.num_pseudobonds))

print("\n--- toggles: show all again ---")
run(session, "hitpick show all")

print("\n--- hitpick explain (a ligand nitrogen) ---")

import chimerax.hitpickingtool.state as hpstate
st = hpstate.get_state(session)
target = None
for i, a in enumerate(st.all_atoms):
    if a.structure in ligs and a.element.name == "N":
        target = (i, a)
        break
if target:
    i, a = target
    print("clicking:", a, "index", i)
    hit = st.handbook.for_atom(st.molsys, i)
    print("  title:", hit.title)
    print("  highlight:", hit.highlight)
    print("  entries:", [e["id"] for e in hit.entries[:4]])
    for ann in hit.annotations[:3]:
        print("  %s median %.2f (%.2f..%.2f, n=%d, %d flagged)"
              % (ann["quantity"], ann["median"], ann["min"], ann["max"],
                 ann["n"], ann["n_flagged"]))
    print("  3D label:", st.handbook.label_for_atom(st.molsys, i))

print("\n--- hitpick labels pKBHX ---")
run(session, "hitpick labels pKBHX")

print("\n--- hitpick clear ---")
run(session, "hitpick clear")
print("=" * 72)
print("E2E OK")
