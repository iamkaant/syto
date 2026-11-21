"""Check our ViewDock columns against ViewDock's REAL display_key().

ViewDock's update_model_description does:
    column_map = {col.title: col for col in struct_table.columns}
    all_titles = [display_key(key) for key in structure.viewdock_data]
    column_map[title]            # KeyError if we broke the contract

so every key we write must satisfy display_key(key) == some column title.
"""
import sys

from chimerax.viewdock.tool import ViewDockTool
from chimerax.syto import viewdock_link as vl

failures = []

# ViewDock's display_key does not touch self, so call it unbound.
real = ViewDockTool.display_key

print("our columns:", vl.COLUMNS)
for title in vl.COLUMNS:
    got = real(None, title)
    ok = (got == title)
    print("  %-14r -> display_key %-14r  %s" % (title, got, "OK" if ok else "MISMATCH"))
    if not ok:
        failures.append((title, got))

# our mirror must agree with the real thing, for every key we might emit
for title in vl.COLUMNS + (vl.RATING_KEY, "H-Bonds", "Clashes",
                           "Total Energy", "some_key.with.dots"):
    mine = vl.display_key_identity(title)
    theirs = real(None, title)
    if mine != theirs:
        failures.append(("mirror", title, mine, theirs))
        print("  MIRROR DIFFERS for %r: ours=%r theirs=%r" % (title, mine, theirs))

# a pose carrying our keys must resolve against a table holding our columns
class FakeStructure:
    pass


class FakeResult:
    counts = {"hbond": 3, "hbond_charged": 1, "salt_bridge": 2,
              "pi_stack_parallel": 1, "pi_stack_tshaped": 1}

    class _Good:
        quality = 0.9

    class _Poor:
        quality = 0.3
    # one well formed, one not: "HP Good" must count 1, where the old
    # summed "HP Quality" would have returned 1.2
    interactions = [_Good(), _Poor()]
    strain = None


s = FakeStructure()
s.viewdock_data = {"Name": "x", "rating": 2}
vl.annotate_structure(None, s, FakeResult(), allowed=set(vl.COLUMNS))
print("\nviewdock_data after annotate:", s.viewdock_data)
good = s.viewdock_data.get(vl.COL_GOOD)
if good != 1:
    failures.append(("HP Good should count well-formed interactions, got", good))
    print("  FAIL HP Good = %r, expected 1" % (good,))
else:
    print("  OK   HP Good counts only the well-formed interaction")
if "HP Quality" in s.viewdock_data:
    failures.append("the summed HP Quality is gone")
    print("  FAIL HP Quality is still written")
column_titles = set(vl.COLUMNS) | {"Name", "Rating"}
for key in s.viewdock_data:
    if key == vl.RATING_KEY:
        continue
    t = real(None, key)
    if t not in column_titles:
        failures.append(("unresolvable", key, t))
        print("  UNRESOLVABLE key %r -> title %r" % (key, t))

# and with no columns available, nothing of ours may be written
s2 = FakeStructure()
s2.viewdock_data = {"Name": "y", "HP Salt": 99}
vl.annotate_structure(None, s2, FakeResult(), allowed=set())
leftover = [k for k in s2.viewdock_data if k in vl.COLUMNS]
print("keys written when no column exists:", leftover)
if leftover:
    failures.append(("wrote without column", leftover))

print()
if failures:
    print("FAILURES:", failures)
    sys.exit(1)
print("VIEWDOCK CONTRACT OK")
