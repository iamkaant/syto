"""The controls that existed only as typed commands.

Four things the panel could not do - show the residues being contacted,
turn explain-on-select off, set the line width, and read the flagged
torsions as a list - are now on screen. What is pinned here is the
ChimeraX side: that each path does what the command does, that the
snapshot carries the state back so the control can reflect it, and that
'reveal' is genuinely reversible.

That last one is the reason this file exists. 'hitpick reveal' took a
`show` argument and ignored it: there was no way back, so a checkbox
would have been a lie. It now hides exactly the atoms it turned on, and
nothing the user displayed themselves.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_panel_controls.py
"""
import json
import os
import sys
import types

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

from chimerax.hitpickingtool import render
from chimerax.hitpickingtool.state import get_state
from chimerax.hitpickingtool.tool import HandbookTool

st = get_state(session)
lig = [m for m in session.models
       if getattr(m, "viewdock_data", None)][0]
run(session, "hitpick analyze #%s" % lig.id_string)
st = get_state(session)
check(st.result is not None, "a pose is analysed to work with")

# The panel cannot be opened without a GUI, but every control reads and
# writes through _snapshot and handle_scheme, so those are bound to a bare
# object and exercised directly.
stub = types.SimpleNamespace(session=session, _state=st)
for _n in ("_snapshot", "_torsion_list", "_torsion_profile",
           "_pick_torsion"):
    setattr(stub, _n, types.MethodType(getattr(HandbookTool, _n), stub))


def snap():
    return stub._snapshot(get_state(session))

print("[1] the snapshot carries every option the controls bind to")
s0 = snap()
json.dumps(s0)
for k in ("reveal_partners", "handbook_on_click", "line_width"):
    check(k in s0["options"], "options.%s is sent to the panel" % k)
check(isinstance(s0["options"]["line_width"], float),
      "line_width is a number the slider can take: %r"
      % s0["options"]["line_width"])

print("\n[2] reveal is reversible, and only over what it turned on")
# Hide the whole receptor, then display ONE residue by hand. Revealing
# must not claim that one, and un-revealing must leave it alone.
rec = st.receptor_model
rec.atoms.displays = False
mine = rec.residues[0].atoms
mine.displays = True
before = int(rec.atoms.displays.sum())

n = run(session, "hitpick reveal")
after = int(rec.atoms.displays.sum())
check(n > 0, "revealed %d atom(s)" % n)
check(after > before, "more is displayed than before (%d -> %d)"
      % (before, after))
check(st.options.reveal_partners is True, "and the option records it")
check(snap()["options"]["reveal_partners"] is True,
      "so the checkbox can show it ticked")

run(session, "hitpick reveal false")
back = int(rec.atoms.displays.sum())
check(back == before, "unticking puts it back exactly (%d -> %d, was %d)"
      % (after, back, before))
check(all(mine.displays),
      "and the residue the USER displayed is still displayed")
check(st.options.reveal_partners is False, "the option records that too")

run(session, "hitpick reveal false")
check(int(rec.atoms.displays.sum()) == before,
      "hiding twice is not an error and takes nothing extra away")

print("\n[3] autoexplain and width go through the same path as the command")
run(session, "hitpick autoexplain false")
check(snap()["options"]["handbook_on_click"] is False,
      "autoexplain off reaches the panel")
run(session, "hitpick autoexplain true")
check(snap()["options"]["handbook_on_click"] is True, "and back on")

w = run(session, "hitpick width 0.12")
check(abs(w - 0.12) < 1e-6, "width set to %.3f" % w)
check(abs(snap()["options"]["line_width"] - 0.12) < 1e-6,
      "and the slider would read it back")
run(session, "hitpick width 0.075")

print("\n[4] the torsion list is the flagged set, worst first")
rows = snap()["torsions"]
check(isinstance(rows, list) and rows, "%d flagged torsion(s) listed"
      % len(rows))
flagged = [t for t in st.result.strain.torsions if t.flagged]
check(len(rows) == len(flagged),
      "one row per flagged torsion (%d)" % len(flagged))
measured = [r for r in rows if r["energy"] is not None]
check(all(measured[i]["energy"] >= measured[i + 1]["energy"]
          for i in range(len(measured) - 1)),
      "the measured ones are worst first: %s"
      % [r["energy"] for r in measured])
check(all(r["energy"] is not None for r in rows[:len(measured)]),
      "and the unmeasured ones come after, not interleaved")
for r in rows:
    check(r["severity"] in ("warning", "note"),
          "every row has a severity (%s)" % r["severity"])
    break
check(all("," in r["bond"] for r in rows),
      "each row carries the bond the panel clicks through to")

print("\n[5] clicking a row selects that bond, as a bond click would")
target = rows[0]
stub._pick_torsion(target["bond"])
check(st.last_torsion is not None, "a torsion was chosen")
check("%d,%d" % st.last_torsion.bond == target["bond"],
      "and it is the one clicked: %s" % (st.last_torsion.bond,))
sel = session.selection.items("atoms")
n_sel = sum(len(a) for a in sel) if sel else 0
check(n_sel == 2, "the bond's two atoms are selected (%d)" % n_sel)
after = snap()
check(any(r["selected"] for r in after["torsions"]),
      "and the list marks the chosen row")
check(sum(1 for r in after["torsions"] if r["selected"]) == 1,
      "exactly one row")

print("\n[6] a bond that is not a torsion is ignored, not an exception")
stub._pick_torsion("99998,99999")
check(True, "no exception")
stub._pick_torsion("nonsense")
check(True, "and neither does a malformed one")

print()
if failures:
    print("FAILURES (%d):" % len(failures))
    for f in failures:
        print("   -", f)
    sys.exit(1)
print("PANEL CONTROL TESTS OK")
