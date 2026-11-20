#!/usr/bin/env python
"""How fast can the browser featurise poses, against native Python?

Featurisation is the slow step of a hit-picking party: every pose goes
through the full interaction and strain analysis before the model has
anything to learn from. On the ChimeraX side that is about a tenth of a
second a pose, which is a minute for a few hundred. Whether the web page
can host a party rather than only review one comes down to what that
number is in Pyodide, and that is not something to guess at.

Both runs use the same receptor, the same poses and the same core.

Usage:  python tools/bench_web_al.py [--poses 40]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from check_web_parity import (LIG, PORT, REC, find_chrome,  # noqa: E402
                              run_browser, serve)
from check_web_viewer import trim_mol2  # noqa: E402

WEB = os.path.join(ROOT, "web")
DRIVER = "__bench.html"


def native(n):
    """The same work, in the Python this machine runs natively."""
    from hitpicking.analysis import Options, ReceptorContext
    from hitpicking.al.features import pose_features
    from hitpicking.io.mol2 import read_mol2
    from hitpicking.io.pdb import read_pdb

    t0 = time.perf_counter()
    rec = read_pdb(REC)
    ctx = ReceptorContext(rec, Options())
    setup = time.perf_counter() - t0

    poses = read_mol2(LIG, limit=n)
    opts = Options()
    t0 = time.perf_counter()
    for ms in poses:
        res = ctx.analyze(ms, options=opts)
        pose_features(res)
    total = time.perf_counter() - t0
    return {"setup": setup, "poses": len(poses), "seconds": total,
            "per_pose": total / len(poses)}


PAGE = """<!doctype html><html><head><meta charset="utf-8"></head><body>
<script type="module">
import * as B from "./ui/bridge.js";
</script>
<script src="ui/bridge.js"></script>
<script>
var LOG = [];
function done(o) {
  o.log = LOG.join("\\n");
  navigator.sendBeacon("/__result", JSON.stringify(o));
}
(async function () {
  var out = {ok: false};
  try {
    var t0 = performance.now();
    var bridge = new HPBridge.WorkerBridge("js/worker.js");
    await bridge.call("version");
    out.bootMs = Math.round(performance.now() - t0);
    LOG.push("engine up in " + out.bootMs + " ms");

    var rec = await (await fetch("__rec")).text();
    var lig = await (await fetch("__lig")).text();

    t0 = performance.now();
    await bridge.call("setup", {name: "rec.pdb", text: rec});
    out.setupMs = Math.round(performance.now() - t0);
    var p = await bridge.call("loadPoses",
                              {name: "poses.mol2", text: lig, limit: 5000});
    out.poses = p.count;
    LOG.push("receptor perceived in " + out.setupMs + " ms, "
             + p.count + " poses");

    // A warm-up pose first: the very first call pays for numpy import
    // and for whatever the JIT has not seen yet, and counting that in a
    // per-pose rate would understate every pose after it.
    var warm = await bridge.call("featurise", {start: 0, count: 1});
    out.warmupPerPose = warm.per_pose;
    LOG.push("warm-up pose: " + warm.per_pose + " s");

    t0 = performance.now();
    var r = await bridge.call("featurise", {start: 1, count: COUNT - 1});
    out.wallMs = Math.round(performance.now() - t0);
    out.featurised = r.featurised;
    out.enginePerPose = r.per_pose;
    out.analyseShare = r.analyse_share;
    out.known = r.total_known;
    LOG.push("featurised " + r.featurised + " in " + r.seconds + " s");

    var st = await bridge.call("alStatus", {});
    out.status = st;
    out.ok = true;
  } catch (e) {
    out.error = String((e && e.stack) || e);
  }
  done(out);
})();
</script></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=int, default=40)
    args = ap.parse_args(argv)

    chrome = find_chrome()
    if chrome is None or not os.path.exists(REC):
        print("need headless Chrome and the sample data; skipping")
        return 0

    print("native Python (%s)" % sys.version.split()[0])
    nat = native(args.poses)
    print("   receptor perceived in %.2f s" % nat["setup"])
    print("   %d pose(s) featurised in %.2f s  =  %.3f s a pose"
          % (nat["poses"], nat["seconds"], nat["per_pose"]))

    with open(os.path.join(WEB, DRIVER), "w", encoding="utf-8") as fh:
        fh.write(PAGE.replace("COUNT", str(args.poses)))
    with open(os.path.join(WEB, "__rec"), "w", encoding="utf-8") as fh:
        fh.write(open(REC, encoding="utf-8", errors="replace").read())
    with open(os.path.join(WEB, "__lig"), "w", encoding="utf-8") as fh:
        fh.write(trim_mol2(LIG, args.poses))

    inbox = []
    httpd = serve(WEB, PORT, inbox)
    try:
        got = run_browser(chrome, PORT, inbox, timeout=900, page=DRIVER)
    finally:
        httpd.shutdown()
        for n in (DRIVER, "__rec", "__lig"):
            try:
                os.remove(os.path.join(WEB, n))
            except OSError:
                pass

    print()
    for line in (got.get("log") or "").strip().splitlines():
        print("   | " + line)
    if not got.get("ok"):
        print("the browser run failed: %s" % got.get("error"))
        return 1

    print("\nPyodide, in a browser tab")
    print("   engine boot %.1f s, receptor perceived in %.2f s"
          % (got["bootMs"] / 1000.0, got["setupMs"] / 1000.0))
    print("   first pose %.3f s (cold), then %.3f s a pose over %d"
          % (got["warmupPerPose"], got["enginePerPose"], got["featurised"]))

    ratio = got["enginePerPose"] / nat["per_pose"]
    print("\nPyodide is %.1fx slower a pose than native here." % ratio)
    print("What that means for a party:")
    for n in (100, 500, 1000, 2000):
        print("   %5d poses   native %5.0f s   browser %5.0f s  (%.1f min)"
              % (n, n * nat["per_pose"], n * got["enginePerPose"],
                 n * got["enginePerPose"] / 60.0))
    print("\n   (%.0f%% of each pose is the interaction and strain "
          "analysis; the rest is building the vector.)"
          % (100 * (got.get("analyseShare") or 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
