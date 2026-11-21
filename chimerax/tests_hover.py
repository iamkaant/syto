"""The hover balloon must describe what is under the cursor.

ChimeraX's own balloon shows an atom spec. This replaces it with the
handbook's answer: an atom's role and basicity, an interaction's geometry
against its limits, a clash's overlap, a bond's torsion.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_hover.py
"""
import os
import sys

failures = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)

import re, sys
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
rec = max(models, key=lambda m: m.num_residues)
lig = [m for m in models if "3521" in m.name] or [m for m in models if m is not rec]
lig = lig[0]
run(session, "hitpick setup"); run(session, "hitpick analyze #%s" % lig.id_string)

from chimerax.hitpickingtool import hover
import chimerax.hitpickingtool.state as S
st = S.get_state(session)

def plain(html):
    return re.sub(r"<[^>]+>", " ", html or "").replace("&Aring;", "A") \
             .replace("&deg;", "deg").replace("&mdash;", "-").replace("&nbsp;", " ")

print("=== hovering a ligand nitrogen ===")
for a in lig.atoms:
    if a.element.name == "N":
        print(" ", " ".join(plain(hover._describe_atom(st, a)).split())[:200]); break

print("\n=== hovering each interaction line ===")
for m in session.models:
    if not isinstance(m, PseudobondGroup) or not m.name.startswith("hitpick"):
        continue
    for pb in list(m.pseudobonds)[:1]:
        txt = hover._describe_pseudobond(st, pb)
        print("  [%s]" % m.name)
        print("   ", " ".join(plain(txt).split())[:260] if txt else "NO TEXT")

print("\n=== hovering a rotatable bond ===")
shown = 0
for t in st.result.strain.torsions:
    i, j = t.bond
    a1, a2 = st.all_atoms[i], st.all_atoms[j]
    for b in a1.bonds:
        if a2 in b.atoms:
            txt = hover._describe_bond(st, b)
            if txt:
                print("   ", " ".join(plain(txt).split())[:220]); shown += 1
            break
    if shown >= 2: break

print("\n=== assertions ===")
n_atom = hover._describe_atom(st, [a for a in lig.atoms
                                   if a.element.name == "N"][0])
check(n_atom and "pKBHX" in n_atom, "an atom balloon carries its basicity")
check(n_atom and ("role" in plain(n_atom)), "an atom balloon states its role")

texts = {}
for m in session.models:
    if isinstance(m, PseudobondGroup) and m.name.startswith("hitpick"):
        for pb in list(m.pseudobonds)[:1]:
            texts[m.name] = hover._describe_pseudobond(st, pb)
check(all(v for v in texts.values()),
      "every drawn line has balloon text (%d groups)" % len(texts))
inter = [v for k, v in texts.items() if "clash" not in k]
check(all("limits" in plain(v) for v in inter),
      "interaction balloons quote the limits")
clash = [v for k, v in texts.items() if "clash" in k]
check(all("overlap" in plain(v) for v in clash),
      "clash balloons quote the overlap")

stack = [v for k, v in texts.items() if "stack" in k]
if stack:
    t = plain(stack[0])
    # the lateral offset is a LENGTH; printing it in degrees was a real bug
    check("offset" in t and "offset 0" not in t.replace("offset ", "offset 0", 0),
          "stacking balloon names the lateral offset")
    check("deg" not in t.split("lateral offset")[-1][:14],
          "the lateral offset is not shown in degrees")
    check("plane separation" in t,
          "stacking balloon says which distance it measured")

print()
if failures:
    print("FAILURES (%d):" % len(failures))
    for f in failures:
        print("   -", f)
    sys.exit(1)
print("HOVER TESTS OK")
