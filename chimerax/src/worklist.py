"""A queue of poses to grade, driven from inside ViewDock's own table.

The Learn panel can list what to look at next, but reading a list in one
window and then hunting for each compound in another is most of the work.
This puts the list where the grading happens: ViewDock's table is set to
the queue, in the model's order, and a pose leaves the moment it is graded
so the next one is already in front of you.

How it hooks in, and why that is safe-ish:

  ViewDock connects its rating buttons with
      but.clicked.connect(lambda *args, val=val: self.rating_changed(val))
  The lambda looks `rating_changed` up on the tool when the button is
  pressed, not when it is connected, so replacing the attribute intercepts
  every rating - including the b/m/g hotkeys, which `process_hotkey` routes
  through the same buttons. There is no public API for any of this, so
  every call is wrapped and the queue removes itself cleanly if ViewDock's
  internals move.

The queue is deliberately NOT re-ranked while you work it. Re-sorting the
list under someone part way through is disorienting, and the point of a
batch is that it is a batch: when it empties, ask for another.
"""

from __future__ import annotations

_ACTIVE = {}

# the tool methods that rebuild the table from ViewDock's own filter, and
# would otherwise silently drop the queue
_REBUILDERS = ("hide_rating_changed", "update_table_for_hidden")


class Worklist:
    def __init__(self, session, tool, structures, label=""):
        self.session = session
        self.tool = tool
        self.queue = list(structures)
        self.label = label
        self.done = 0
        # What each pose was rated when it joined the queue. A pose leaves
        # when its rating CHANGES, not when it merely has one: the
        # "disagreement" strategy deliberately queues poses you have
        # already graded, and testing for "has a rating" emptied that whole
        # batch the instant it was shown.
        self._was = {}
        for st in self.queue:
            try:
                self._was[id(st)] = (st.viewdock_data or {}).get("rating", 0)
            except Exception:
                self._was[id(st)] = 0
        self._saved = None
        self._orig = {}

    # ------------------------------------------------------------ install

    def install(self):
        tool = self.tool
        try:
            self._saved = list(tool.struct_table.data)
        except Exception:
            self._saved = None

        def wrap(name, after):
            orig = getattr(tool, name)
            self._orig[name] = orig

            def patched(*a, **kw):
                try:
                    result = orig(*a, **kw)
                finally:
                    try:
                        after()
                    except Exception:
                        self.remove(restore=False)
                return result
            setattr(tool, name, patched)

        wrap("rating_changed", self._after_rating)
        for name in _REBUILDERS:
            try:
                wrap(name, self._show)
            except Exception:
                pass
        _ACTIVE[self.session] = self
        self._show(select_first=True)
        return len(self.queue)

    def remove(self, restore=True):
        for name, orig in self._orig.items():
            try:
                setattr(self.tool, name, orig)
            except Exception:
                pass
        self._orig.clear()
        if restore:
            try:
                # back to whatever ViewDock's own filter says, not the
                # snapshot, in case ratings changed while we held the table
                self.tool.set_table_data(self.tool.table_structures)
            except Exception:
                if self._saved is not None:
                    try:
                        self.tool.set_table_data(self._saved)
                    except Exception:
                        pass
        if _ACTIVE.get(self.session) is self:
            del _ACTIVE[self.session]

    # ------------------------------------------------------------- behave

    def _after_rating(self):
        """A grade was just given: drop those poses and move on."""
        graded = []
        for st in self.queue:
            try:
                now = (getattr(st, "viewdock_data", None) or {}).get(
                    "rating", 0)
            except Exception:
                continue
            if now != self._was.get(id(st), 0):
                graded.append(st)
        if graded:
            self.done += len(graded)
            self.queue = [s for s in self.queue if s not in graded]
        self._show(select_first=True)
        if not self.queue:
            self._finish()

    def _show(self, select_first=False):
        tool = self.tool
        # a pose whose model was closed is no longer ours to show
        self.queue = [s for s in self.queue if getattr(s, "id", None)]
        if not self.queue:
            return
        try:
            if list(tool.struct_table.data) != self.queue:
                tool.set_table_data(self.queue)
        except Exception:
            return
        if not select_first:
            return
        # Keep the selection if it is still in the queue; otherwise take the
        # head. Only checking "is anything selected" left the pose that had
        # just been graded selected although it was no longer in the table,
        # so the next b/m/g re-graded it and the queue never moved.
        try:
            sel = [s for s in (tool.struct_table.selected or [])
                   if s in self.queue]
            if not sel:
                tool.struct_table.selected = self.queue[0:1]
        except Exception:
            pass

    def _finish(self):
        n = self.done
        self.remove()
        self.session.logger.info(
            "hitpick: worklist finished - %d pose(s) graded%s. "
            "Train, then ask for another batch: hitpick learn worklist"
            % (n, (" from " + self.label) if self.label else ""))

    # -------------------------------------------------------------- state

    def describe(self):
        return {"left": len(self.queue), "done": self.done,
                "label": self.label}


def current(session):
    return _ACTIVE.get(session)


def stop(session):
    wl = _ACTIVE.get(session)
    if wl is None:
        return 0
    n = len(wl.queue)
    wl.remove()
    return n
