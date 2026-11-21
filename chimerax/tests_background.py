"""Collecting and training without freezing the window.

Analysing a pose costs ~0.14 s, so a few thousand poses is minutes of
work. It therefore runs on a worker thread, with the ChimeraX-touching
part handed back to the GUI thread. Without a GUI it all runs inline,
which is what makes it testable here.

Run:  ChimeraX --nogui --silent --exit --script chimerax/tests_background.py
"""
import os
import sys
import threading
import time

from chimerax.core.commands import run
from chimerax.atomic import AtomicStructure

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

from chimerax.hitpickingtool import background, learn

print("[1] with no GUI the work runs inline, so scripts still behave")
job = background.Job(session, "test")
threaded = job.start(lambda j: 21 * 2)
check(threaded is False, "start() reports it did not use a thread")
check(job.done and job.result == 42, "it ran and returned: %r" % job.result)
check(job.error is None, "with no error")

print("\n[2] on_main runs the callable and passes its value back")
job = background.Job(session, "test")
check(job.on_main(lambda a, b: a + b, 2, 3) == 5, "on_main returns the result")
try:
    job.on_main(lambda: 1 / 0)
    check(False, "an exception inside on_main reaches the caller")
except ZeroDivisionError:
    check(True, "an exception inside on_main reaches the caller")

print("\n[3] cancelling stops the work and keeps what was done")
learn.clear(session)
job = background.Job(session, "analysing")
seen = []


def work(j):
    for i in range(50):
        j.checkpoint()
        seen.append(i)
        if i == 4:
            j.cancel()
    return "finished"


job.start(work)
check(job.cancelled, "the job knows it was cancelled")
check(job.result is None, "it did not return a result")
check(job.error is None, "a cancel is not an error")
check(len(seen) == 5, "it stopped at the checkpoint after cancel (%d)"
      % len(seen))

print("\n[4] collect reports progress and can be stopped part-way")
learn.clear(session)
progress = []
job = background.Job(session, "analysing")


def collect_some(j):
    return learn.collect(session, limit=12, quiet=True, job=j)


job.start(collect_some,
          on_progress=lambda d, t, m: progress.append((d, t)))
check(job.error is None, "collect finished without error: %s" % job.error)
check(job.result and job.result > 0, "it featurised %s pose(s)" % job.result)
check(bool(progress), "progress was reported: %s" % (progress[:3],))
if progress:
    check(progress[-1][0] == job.result,
          "the last report matches what was done (%s vs %s)"
          % (progress[-1][0], job.result))

print("\n[5] a cancelled collect keeps the poses it already analysed")
learn.clear(session)
job = background.Job(session, "analysing")
done_before = {"n": 0}


def stop_early(j):
    import chimerax.hitpickingtool.learn as L
    real = L._features_for

    def counting(hp, lig, jj):
        done_before["n"] += 1
        if done_before["n"] > 3:
            jj.cancel()
        return real(hp, lig, jj)

    L._features_for = counting
    try:
        return L.collect(session, limit=20, quiet=True, job=j)
    finally:
        L._features_for = real


job.start(stop_early)
st = learn.state(session)
check(job.cancelled, "the collect was cancelled")
check(len(st.learner.features) > 0,
      "the %d poses analysed before the cancel were kept"
      % len(st.learner.features))
check(len(st.learner.features) < 20, "and it did not finish all 20")

print("\n[6] only one job at a time")
learn.clear(session)
j1 = background.claim(session, "analysing")
try:
    background.claim(session, "training")
    check(False, "a second job is refused while one is running")
except RuntimeError as e:
    check("already" in str(e), "a second job is refused: %s" % e)
background.release(session, j1)
j2 = background.claim(session, "training")
check(j2 is not None, "and allowed once the first is released")
background.release(session, j2)

print("\n[7] the collect command works and returns something JSON-safe")
learn.clear(session)
import json
d = run(session, "hitpick learn collect limit 8")
check(isinstance(d, dict), "the command returns a dict")
json.dumps(d)
check(True, "which is JSON-serialisable, so the REST bridge survives it")
check(d.get("poses", 0) > 0, "and it featurised poses: %s" % d.get("poses"))

print("\n[8] cancel command with nothing running says so")
r = run(session, "hitpick learn cancel")
check(r.get("cancelled") is False, "reports there was nothing to cancel")
json.dumps(r)

print()
if failures:
    print("FAILURES (%d):" % len(failures))
    for f in failures:
        print("   -", f)
    sys.exit(1)
print("BACKGROUND TESTS OK")
