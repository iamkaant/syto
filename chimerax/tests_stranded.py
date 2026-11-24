"""Stranded H-bond donors and acceptors actually reach ChimeraX.

The detection lived in the core's ReceptorContext.analyze. The ChimeraX
command builds its PoseResult itself and never called it, so the two
checkboxes, the `hitpick stranded` command, the panel card and the 3D
words were all wired to a list that was always empty. Nothing failed:
an empty list is a legitimate answer. So this pins that the list is NOT
empty where the core says it should not be, and that the answer is the
core's own.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_stranded.py
"""
import os
import sys
import types

from chimerax.core.commands import run

failures = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools"))
from samples import AUTOPARTY          # noqa: E402

A = AUTOPARTY + "/"
if not os.path.exists(A + "autoparty/inputs/rec.crg.pdb"):
    print("sample data unavailable; skipping")
    print("STRANDED TESTS OK")
    sys.exit(0)

run(session, "open %s" % (A + "autoparty/inputs/rec.crg.pdb"))
run(session, "open %s format mol2 showTool false"
    % (A + "autoparty/inputs/more_opcore_enum_asp_h2o_hbond.mol2"))
run(session, "hitpick setup")

from chimerax.syto import render
from chimerax.syto.state import get_state
from chimerax.syto.tool import HandbookTool
from hitpicking.nci.unsatisfied import find_unsatisfied
from hitpicking.perceive.features import perceive_all

ligs = [m for m in session.models if getattr(m, "viewdock_data", None)]


def roles():
    r = get_state(session).result
    return None if r.unsatisfied is None else [u.role for u in r.unsatisfied]


def words(st):
    """The 'stranded ...' words written on atoms in 3D."""
    return sorted(i for i, a in enumerate(st.all_atoms)
                  if "stranded" in (getattr(a, render.LABEL_ATTR, "") or ""))


print("[1] with both settings on, the command gives the core's own answer")
run(session, "hitpick stranded donors true acceptors true")
found = None
n_any = 0
for lig in ligs[:40]:
    run(session, "hitpick analyze #%s" % lig.id_string, log=False)
    st = get_state(session)
    un = st.result.unsatisfied
    if un is None:
        check(False, "result.unsatisfied is None for %s" % lig.name)
        break
    core = find_unsatisfied(st.molsys, perceive_all(st.molsys),
                            st.result.interactions)
    if sorted(u.atom for u in un) != sorted(u.atom for u in core):
        check(False, "%s: command and core disagree" % lig.name)
        break
    if un:
        n_any += 1
        if found is None or (any("donor" in u.role for u in un)
                             and any("acceptor" in u.role for u in un)):
            found = lig
else:
    check(True, "command and core agree on the first %d poses"
          % min(40, len(ligs)))
check(n_any > 0, "%d pose(s) have stranded atoms at all" % n_any)

if found is not None:
    run(session, "hitpick analyze #%s" % found.id_string, log=False)
    st = get_state(session)
    both = roles()
    print("     %s: %s" % (found.name, both))

    print("[2] the atoms are marked in 3D and listed for the panel")
    check(words(st) == sorted(set(u.atom for u in st.result.unsatisfied)),
          "every stranded atom carries the word in 3D, and no other atom")
    stub = types.SimpleNamespace(session=session, _state=st)
    for _n in ("_snapshot", "_torsion_list", "_torsion_profile",
               "_pick_torsion"):
        setattr(stub, _n, types.MethodType(getattr(HandbookTool, _n), stub))
    s = stub._snapshot(st)
    check(len(s["pose"]["unsatisfied"]) == len(both),
          "the snapshot carries %d row(s) for the panel card" % len(both))
    run(session, "hitpick report", log=False)
    check(True, "report runs with stranded atoms present")

    print("[3] each setting filters its own kind")
    run(session, "hitpick stranded donors true acceptors false", log=False)
    d_only = roles()
    check(all("donor" in r for r in d_only),
          "acceptors off: only donors remain (%d)" % len(d_only))
    run(session, "hitpick stranded donors false acceptors true", log=False)
    a_only = roles()
    check(all("acceptor" in r for r in a_only),
          "donors off: only acceptors remain (%d)" % len(a_only))
    check(len(set(both)) and len(d_only) + len(a_only) >= len(both),
          "the two halves cover the whole list")
    run(session, "hitpick stranded donors false acceptors false", log=False)
    st = get_state(session)
    check(roles() == [] and words(st) == [],
          "both off: no rows and no words left in 3D")

if failures:
    print("FAILURES (%d):" % len(failures))
    for f in failures:
        print("   -", f)
    raise SystemExit(1)
print("STRANDED TESTS OK")
