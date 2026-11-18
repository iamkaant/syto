#!/usr/bin/env python3
"""How much of a real pose set does the QM torsion reference actually cover?

The point of the QM data is to replace guesswork with energies. Whether it
succeeds is not a question about the reference in the abstract - it is a
question about THESE ligands: of every rotatable torsion the tool meets in
them, what fraction gets an exact class, what fraction falls back, and what
fraction gets nothing at all.

Run it on whatever you are picking from:

    python tools/torsion_coverage.py \\
        --receptor rec.crg.pdb --ligands poses.mol2 --limit 200
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--receptor", required=True)
    ap.add_argument("--ligands", required=True)
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()

    from hitpicking.analysis import ReceptorContext
    from hitpicking.strain import torsions as T

    qm = T.qm_reference()
    if qm is None:
        sys.exit("no QM reference built; run tools/fetch_qm_torsions.py "
                 "then tools/build_torsion_profiles.py")
    print("QM reference: %d classes from %d scans"
          % (len(qm["classes"]), qm.get("n_scans_total", 0)))

    ctx = ReceptorContext.from_pdb(a.receptor)
    if a.ligands.endswith((".mol2", ".mol2.gz")):
        from hitpicking.io.mol2 import read_mol2_iter as reader
    else:
        from hitpicking.io.sdf import read_sdf_iter as reader

    seen = collections.Counter()
    exact = collections.Counter()
    fallback = collections.Counter()
    missing = collections.Counter()
    n_poses = 0

    for pose in reader(a.ligands):
        res = ctx.analyze(pose)
        for t in (res.strain.torsions if res.strain else []):
            key = "%s|%s|%s" % (t.bond_type, t.subtype or "-",
                                "ortho" if t.ortho_substituted else "no-ortho")
            seen[key] += 1
            cost = T.qm_cost(t.bond_type, t.subtype, t.ortho_substituted,
                             t.angle)
            if cost is None:
                missing[key] += 1
            elif cost[5]:
                exact[key] += 1
            else:
                fallback[key] += 1
        n_poses += 1
        if n_poses >= a.limit:
            break

    total = sum(seen.values())
    if not total:
        sys.exit("no torsions found")

    print("\n%d poses, %d rotatable torsions\n" % (n_poses, total))
    print("%-44s %7s %7s %7s %7s" % ("class", "seen", "exact", "fallbk", "none"))
    for key, n in seen.most_common():
        print("%-44s %7d %7d %7d %7d"
              % (key[:44], n, exact[key], fallback[key], missing[key]))

    e, f, m = sum(exact.values()), sum(fallback.values()), sum(missing.values())
    print("\nexact QM class   %6d  %5.1f%%" % (e, 100.0 * e / total))
    print("fallback (shown but never used to flag)"
          "   %6d  %5.1f%%" % (f, 100.0 * f / total))
    print("no reference     %6d  %5.1f%%" % (m, 100.0 * m / total))

    if m:
        print("\nuncovered classes, commonest first:")
        for key, n in missing.most_common(12):
            print("   %-44s %d" % (key[:44], n))
        print("\nThese are what more QM data would buy. Fetch the datasets "
              "whose chemistry matches them, or accept that the tool reports "
              "them as unjudged.")


if __name__ == "__main__":
    main()
