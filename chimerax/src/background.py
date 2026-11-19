"""Running the slow parts of active learning without freezing the window.

Analysing a pose costs about 0.14 s, so a party with a few thousand poses
loaded is a six-minute freeze if it runs on the GUI thread. Training is
another twenty seconds. Neither is acceptable while people are sitting
round the screen waiting to grade the next compound.

The split that makes this work: of that 0.14 s only about 0.035 s touches
ChimeraX at all - reading the atoms out into a MolSystem. The rest is
perception, detection, strain and featurisation, which run on plain numpy
arrays that no longer refer to anything ChimeraX owns. So the worker
thread does the long part, and hands the short part back to the GUI
thread through `on_main`.

That handing back is not optional. ChimeraX's atoms, models, pseudobonds
and tables are C++ objects with no locking, and reading one while the
main thread redraws is a crash, not a race you get away with.

Without a GUI - tests, nogui, scripts - everything runs inline, so the
behaviour under test is the behaviour in the window, minus the threading.
"""

from __future__ import annotations

import threading


class Cancelled(Exception):
    """Raised inside the worker when the user presses Cancel."""


def _has_gui(session):
    ui = getattr(session, "ui", None)
    return ui is not None and getattr(ui, "is_gui", False) and hasattr(
        ui, "thread_safe")


class Job:
    """One cancellable piece of background work."""

    def __init__(self, session, title="working"):
        self.session = session
        self.title = title
        self.result = None
        self.error = None
        self.done = False
        self.progress = (0, 0, "")
        self._cancel = threading.Event()
        self._thread = None
        self._on_progress = None
        self._on_done = None

    # ------------------------------------------------------------ control

    def cancel(self):
        self._cancel.set()

    @property
    def cancelled(self):
        return self._cancel.is_set()

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def checkpoint(self):
        """Raise if the user has cancelled. Call between units of work."""
        if self._cancel.is_set():
            raise Cancelled()

    # ------------------------------------------------- crossing the threads

    def on_main(self, fn, *args, **kw):
        """Run `fn` on the GUI thread and wait for its result.

        The wait is deliberate. The worker needs what `fn` read before it
        can go on, and letting it run ahead would mean a second call
        arriving while the first is still touching the same structure.
        `fn` is always short - tens of milliseconds - so the GUI thread is
        never held long enough to feel stuck.
        """
        self.checkpoint()
        if not _has_gui(self.session) or (
                threading.current_thread() is threading.main_thread()):
            return fn(*args, **kw)

        box = {}
        finished = threading.Event()

        def wrapper():
            try:
                box["value"] = fn(*args, **kw)
            except BaseException as exc:         # carried to the worker
                box["error"] = exc
            finally:
                finished.set()

        self.session.ui.thread_safe(wrapper, post_event=True)
        # No timeout and no early exit: once the event is posted the call
        # WILL happen, and returning before it does would let the worker
        # touch the same structure from two threads.
        finished.wait()
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def report(self, done, total, message=""):
        self.progress = (done, total, message)
        cb = self._on_progress
        if cb is None:
            return
        if _has_gui(self.session):
            self.session.ui.thread_safe(cb, done, total, message,
                                        post_event=True)
        else:
            cb(done, total, message)

    # -------------------------------------------------------------- running

    def start(self, work, on_progress=None, on_done=None, threaded=None):
        """Run `work(self)`, in a thread where there is a window to keep alive.

        `threaded=False` forces it inline. A caller that means to wait for
        the result MUST do that rather than join the thread: joining blocks
        the GUI thread, and `on_main` posts its work to the GUI thread and
        waits for it, so the two deadlock permanently.

        Returns True if it went to a thread, False if it ran inline.
        """
        self._on_progress = on_progress
        self._on_done = on_done
        if threaded is False or not _has_gui(self.session):
            self._run(work)
            return False
        self._thread = threading.Thread(
            target=self._run, args=(work,), daemon=True,
            name="hitpick-%s" % self.title)
        self._thread.start()
        return True

    def _run(self, work):
        try:
            self.result = work(self)
        except Cancelled:
            self.error = None
            self.result = None
        except BaseException as exc:
            self.error = exc
        finally:
            self.done = True
            cb = self._on_done
            if cb is not None:
                if _has_gui(self.session):
                    self.session.ui.thread_safe(cb, self, post_event=True)
                else:
                    cb(self)


# One job at a time per session: two collects at once would analyse the
# same poses twice and fight over the learner's dictionaries.
_CURRENT = {}


def current(session):
    job = _CURRENT.get(session)
    if job is not None and job.done and not job.running:
        return None
    return job


def claim(session, title):
    """Start a job, or raise if one is already going."""
    running = current(session)
    if running is not None:
        raise RuntimeError(
            "hitpick is already %s. Wait for it, or cancel it first."
            % running.title)
    job = Job(session, title)
    _CURRENT[session] = job
    return job


def release(session, job):
    if _CURRENT.get(session) is job:
        _CURRENT.pop(session, None)


def cancel(session):
    job = current(session)
    if job is None:
        return False
    job.cancel()
    return True
