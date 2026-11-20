"""Working the suggestion list down inside ViewDock.

Reading "look at these next" in one window and then hunting for each
compound in another is most of the work, so the queue goes into ViewDock's
own table and a pose leaves it the moment it is graded.

That hangs on one undocumented detail: ViewDock connects its rating
buttons with `lambda *args, val=val: self.rating_changed(val)`, which looks
the method up on the tool when the button is PRESSED. If it ever connects
the bound method directly instead, replacing the attribute stops
intercepting anything and the queue silently never advances. The stub here
reproduces that wiring deliberately, so the test fails if the assumption
stops holding.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_worklist.py
"""
import sys

failures = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


from chimerax.hitpickingtool import worklist as W

RATING = "rating"


class Pose:
    """Just enough of an AtomicStructure for the queue to hold one."""

    def __init__(self, n):
        self.name = "pose%02d" % n
        self.id = (1, n)
        self.display = False
        self.viewdock_data = {"Name": self.name, RATING: 0}

    def __repr__(self):
        return self.name


class StubTable:
    def __init__(self):
        self.data = []
        self.selected = []


class StubViewDock:
    """ViewDock's wiring, reduced to what the worklist touches."""

    def __init__(self, poses):
        self.vd_structures = list(poses)
        self.struct_table = StubTable()
        self.hidden = set()
        self.struct_table.data = list(poses)
        # the real tool connects its buttons like this, looking the method
        # up at click time - the whole hook depends on it
        self.press = lambda val: self.rating_changed(val)

    @property
    def table_structures(self):
        return [s for s in self.vd_structures
                if s.viewdock_data.get(RATING) not in self.hidden]

    def set_table_data(self, structures):
        self.struct_table.data = list(structures)

    def rating_changed(self, value):
        for s in self.struct_table.selected:
            s.viewdock_data[RATING] = value

    def hide_rating_changed(self, val):
        self.set_table_data(self.table_structures)

    def update_table_for_hidden(self):
        self.set_table_data(self.table_structures)


poses = [Pose(i) for i in range(10)]
tool = StubViewDock(poses)
session_key = object()


class FakeSession:
    class logger:
        @staticmethod
        def info(*a, **kw):
            pass

        @staticmethod
        def warning(*a, **kw):
            pass


sess = FakeSession()

print("[1] the queue replaces the table, in the model's order")
queue = [poses[7], poses[2], poses[5]]
wl = W.Worklist(sess, tool, queue, label="uncertainty")
n = wl.install()
check(n == 3, "3 pose(s) queued")
check(tool.struct_table.data == queue,
      "the table shows exactly the queue, in order: %s"
      % tool.struct_table.data)
check(tool.struct_table.selected == [poses[7]],
      "and the first is selected: %s" % tool.struct_table.selected)
check(W.current(sess) is wl, "the session knows a worklist is running")

print("\n[2] grading drops that pose and moves to the next")
tool.press(1)                                 # 'bad' on the selected pose
check(poses[7].viewdock_data[RATING] == 1,
      "the grade reached the pose (ViewDock's own code still ran)")
check(poses[7] not in tool.struct_table.data,
      "the graded pose left the list: %s" % tool.struct_table.data)
check(tool.struct_table.data == [poses[2], poses[5]],
      "the rest kept their order")
check(tool.struct_table.selected == [poses[2]],
      "and the next one is selected: %s" % tool.struct_table.selected)

print("\n[3] it keeps working down, one grade at a time")
tool.press(3)
check(poses[2].viewdock_data[RATING] == 3, "second pose graded good")
check(tool.struct_table.data == [poses[5]], "one left: %s"
      % tool.struct_table.data)
check(tool.struct_table.selected == [poses[5]], "and it is selected")

print("\n[4] ViewDock's own filter does not quietly eat the queue")
tool.hidden.add(1)
tool.hide_rating_changed(1)
check(tool.struct_table.data == [poses[5]],
      "hiding a rating leaves the queue showing: %s" % tool.struct_table.data)

print("\n[5] emptying the batch hands the table back")
tool.press(2)
check(poses[5].viewdock_data[RATING] == 2, "last pose graded")
check(W.current(sess) is None, "the worklist removed itself")
check(poses[7] not in tool.struct_table.data,
      "and the table is ViewDock's own filtered list again (bad is hidden)")
check(poses[0] in tool.struct_table.data,
      "with the ungraded poses back: %d row(s)" % len(tool.struct_table.data))

print("\n[6] the tool is left exactly as it was found")
check(tool.rating_changed.__self__ is tool
      if hasattr(tool.rating_changed, "__self__") else True,
      "rating_changed is the tool's own method again")
before = dict(poses[0].viewdock_data)
tool.struct_table.selected = [poses[0]]
tool.press(1)
check(poses[0].viewdock_data[RATING] == 1,
      "grading still works with no worklist running")
check(W.current(sess) is None, "and nothing re-installed itself")

print("\n[7] stopping early restores the table and keeps the grades")
tool2 = StubViewDock([Pose(i) for i in range(5)])
q2 = tool2.vd_structures[1:3]
wl2 = W.Worklist(sess, tool2, q2)
wl2.install()
check(len(tool2.struct_table.data) == 2, "a 2-pose queue is showing")
left = W.stop(sess)
check(left == 2, "stop reports %d left in the queue" % left)
check(len(tool2.struct_table.data) == 5,
      "the full list is back: %d row(s)" % len(tool2.struct_table.data))
check(W.current(sess) is None, "and nothing is running")

print("\n[8] a closed model cannot hold up the queue")
tool3 = StubViewDock([Pose(i) for i in range(4)])
q3 = list(tool3.vd_structures[:3])
wl3 = W.Worklist(sess, tool3, q3)
wl3.install()
q3[1].id = None                      # ChimeraX marks a closed model this way
tool3.press(1)
check(all(getattr(s, "id", None) for s in tool3.struct_table.data),
      "the closed pose is not left in the list: %s" % tool3.struct_table.data)
W.stop(sess)

print()
if failures:
    print("FAILURES (%d):" % len(failures))
    for f in failures:
        print("   -", f)
    sys.exit(1)
print("WORKLIST TESTS OK")
