"""Every hitpick command's return value must survive JSON encoding.

ChimeraX JSON-encodes a command's return value when the REST bridge runs
with `json true`. Returning a rich Python object raises inside the encoder,
the response is never queued, and the REST handler thread waits forever
while holding the port: the whole bridge wedges and the command looks like
it silently did nothing.

That is exactly what `hitpick analyze` did by returning a PoseResult. This
test runs each command the way the bridge does and encodes the result.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_json_returns.py
"""
import json
import os
import sys

from chimerax.core.commands import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The sample data is not in this repository and its location is not this
# laptop's home directory. tools/samples.py resolves it; a test that
# cannot find it skips.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools"))
from samples import AUTOPARTY          # noqa: E402

A = AUTOPARTY + "/"
REC = A + "autoparty/inputs/rec.crg.pdb"
POSES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "poses3.mol2")
if not os.path.exists(POSES):
    POSES = A + "autoparty/inputs/more_opcore_enum_asp_h2o_hbond.mol2"

failures = []


def check_json(label, value):
    """Encode exactly as chimerax/rest_server/server.py does."""
    try:
        json.JSONEncoder().encode(value)
    except TypeError as e:
        print("  FAIL %-28s %s" % (label, e))
        failures.append(label)
        return
    kind = type(value).__name__
    print("  OK   %-28s -> %s" % (label, kind))


if not os.path.exists(REC):
    print("sample data unavailable; skipping")
    sys.exit(0)

run(session, "open %s" % REC)
run(session, "open %s format mol2 showTool false" % POSES)
from chimerax.atomic import AtomicStructure
ligs = [m for m in session.models
        if isinstance(m, AtomicStructure) and m.num_residues <= 3]
lig = "#%s" % ligs[0].id_string

print("encoding each command's return value:")
cases = [
    ("hitpick setup", "hitpick setup"),
    ("hitpick analyze", "hitpick analyze %s" % lig),
    ("hitpick report", "hitpick report"),
    ("hitpick show", "hitpick show all"),
    ("hitpick hide", "hitpick hide hydrophobic"),
    ("hitpick lengths", "hitpick lengths true"),
    ("hitpick angles", "hitpick angles true"),
    ("hitpick labels", "hitpick labels pKBHX"),
    ("hitpick labels off", "hitpick labels off"),
    ("hitpick autoexplain", "hitpick autoexplain true"),
    ("hitpick clear", "hitpick clear"),
]
for label, cmd in cases:
    try:
        value = run(session, cmd)
    except Exception as e:
        print("  FAIL %-28s raised %s: %s" % (label, type(e).__name__, e))
        failures.append(label)
        continue
    check_json(label, value)

# hitpick explain needs a selection
run(session, "hitpick analyze %s" % lig)
import chimerax.hitpickingtool.state as hpstate
st = hpstate.get_state(session)
target = None
for i, a in enumerate(st.all_atoms):
    if a.structure in ligs and a.element.name == "N":
        target = a
        break
if target is not None:
    run(session, "select %s" % target.string(style="command"))
    try:
        check_json("hitpick explain", run(session, "hitpick explain"))
    except Exception as e:
        print("  FAIL hitpick explain raised", e)
        failures.append("hitpick explain")

print()
if failures:
    print("FAILURES (%d): %s" % (len(failures), failures))
    sys.exit(1)
print("JSON RETURN TESTS OK")
