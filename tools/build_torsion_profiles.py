#!/usr/bin/env python3
"""Aggregate the fetched QM torsion scans into per-class energy curves.

Each scan is folded into the coordinate the tool actually measures, so a
lookup at analysis time is direct:

  * conjugated, biaryl and amide bonds -> TWIST, min(|t|, 180-|t|), 0 to 90
    degrees, because coplanar is coplanar whether the angle reads 0 or 180;
  * sp3-sp3 bonds -> the 3-fold coordinate, |t| mod 120 folded to 0 to 60,
    so staggered sits at 60 and eclipsed at 0.

Energies are re-referenced to the minimum of the FOLDED curve, then binned
and summarised by median with an interquartile range. The median is used
rather than the mean because a handful of scans in any class have an
intramolecular hydrogen bond or a steric clash that dominates their profile,
and those should not drag the whole class.

Output: data-src/external/torsion_qm.json, carrying the curves, the sample
sizes and the CC-BY-4.0 attribution.

    python tools/build_torsion_profiles.py
    python tools/build_torsion_profiles.py --min-scans 8
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data-src", "external", "qm_torsions", "raw")
ATTR = os.path.join(ROOT, "data-src", "external", "qm_torsions",
                    "ATTRIBUTION.json")
OUT = os.path.join(ROOT, "data-src", "external", "torsion_qm.json")

TWIST_BIN = 15.0          # degrees
SP3_PERIOD = 120.0
CONJ_PERIOD = 180.0


# The one specification in these datasets that is quantum mechanics:
# OpenFF's "default", B3LYP-D3(BJ)/DZVP in Psi4, as the QCArchive server
# reports it for all four datasets. Everything else is a force field.
QM_SPEC = "default"
LEVEL_OF_THEORY = "B3LYP-D3(BJ)/DZVP (Psi4), relaxed torsion drives"


def fold_angle(angle, bond_type):
    """Map a scan angle onto the coordinate the tool measures."""
    a = abs(float(angle)) % 360.0
    if a > 180.0:
        a = 360.0 - a
    if bond_type == "sp3-sp3":
        f = a % SP3_PERIOD
        return min(f, SP3_PERIOD - f)        # 0 (eclipsed) .. 60 (staggered)
    return min(a, CONJ_PERIOD - a)           # 0 (planar) .. 90 (perpendicular)


def fold_profile(profile, bond_type):
    """[(folded_angle, energy)] re-referenced to the folded minimum."""
    pts = [(fold_angle(a, bond_type), float(e)) for a, e in profile]
    if not pts:
        return []
    lo = min(e for _, e in pts)
    return [(a, e - lo) for a, e in pts]


def bin_index(angle):
    return int(round(angle / TWIST_BIN))


def summarise(values):
    values = sorted(values)
    n = len(values)
    med = statistics.median(values)
    if n >= 4:
        q1 = statistics.median(values[:n // 2])
        q3 = statistics.median(values[(n + 1) // 2:])
    else:
        q1 = q3 = med
    return {"median": round(med, 2), "q1": round(q1, 2), "q3": round(q3, 2),
            "n": n}


def build(min_scans=5, split_ortho=True):
    files = sorted(glob.glob(os.path.join(RAW, "*.json")))
    if not files:
        sys.exit("no scans in %s - run tools/fetch_qm_torsions.py first" % RAW)

    classes = {}
    datasets = set()
    skipped_mm = 0
    for path in files:
        try:
            with open(path) as fh:
                rec = json.load(fh)
        except Exception:
            continue
        # Only the QM specification. The Substituted Phenyl sets also
        # carry torsion drives run with force fields (OpenFF 1.x, GAFF
        # 2.11, in OpenMM) under other specification names; the fetch
        # takes every specification, and those 935 molecular-mechanics
        # scans were being pooled into what the tool reports as QM. A
        # force field's torsion profile is the thing these QM scans
        # exist to correct - it has no business in the reference.
        if rec.get("spec", QM_SPEC) != QM_SPEC:
            skipped_mm += 1
            continue
        datasets.add(rec.get("dataset", "?"))
        bond_type = rec.get("bond_type") or "other"
        subtype = rec.get("subtype") or ""
        ortho = bool(rec.get("ortho_substituted"))
        key = (bond_type, subtype, ortho if split_ortho else False)
        folded = fold_profile(rec.get("profile") or [], bond_type)
        if not folded:
            continue
        slot = classes.setdefault(key, {"bins": {}, "scans": 0,
                                        "barriers": [], "minima": []})
        slot["scans"] += 1
        slot["barriers"].append(max(e for _, e in folded))
        slot["minima"].append(min(folded, key=lambda t: t[1])[0])
        for a, e in folded:
            slot["bins"].setdefault(bin_index(a), []).append(e)

    out_classes = {}
    for (bond_type, subtype, ortho), slot in sorted(classes.items()):
        if slot["scans"] < min_scans:
            continue
        curve = {}
        for idx, vals in sorted(slot["bins"].items()):
            curve[str(int(idx * TWIST_BIN))] = summarise(vals)
        name = "%s|%s|%s" % (bond_type, subtype or "-",
                             "ortho" if ortho else "no-ortho")
        # A catch-all class has a number but no chemistry behind it: every
        # sp3-sp3 bond in one bucket, or everything perception could not
        # name. Its median is an average over unlike things, so it is marked
        # here and the tool reports it as indicative rather than quoting it
        # as this bond's cost.
        coarse = (bond_type in ("other", "sp3-sp3")
                  or subtype in ("", "-", "conjugated"))
        out_classes[name] = {
            "coarse": coarse,
            "bond_type": bond_type,
            "subtype": subtype,
            "ortho_substituted": ortho,
            "coordinate": ("eclipsed_0_staggered_60" if bond_type == "sp3-sp3"
                           else "planar_0_perpendicular_90"),
            "n_scans": slot["scans"],
            "curve": curve,
            "barrier": summarise(slot["barriers"]),
            "preferred_angle": summarise(slot["minima"]),
        }

    attribution = {}
    if os.path.exists(ATTR):
        with open(ATTR) as fh:
            attribution = json.load(fh)
    attribution["datasets_used"] = sorted(datasets)
    attribution["level_of_theory"] = LEVEL_OF_THEORY
    attribution["specification"] = QM_SPEC
    attribution["non_qm_scans_excluded"] = skipped_mm

    payload = {
        "schema_version": 1,
        "units": "kcal/mol, relative to the minimum of each folded scan",
        "bin_width_deg": TWIST_BIN,
        "n_scans_total": sum(c["n_scans"] for c in out_classes.values()),
        "min_scans_per_class": min_scans,
        "classes": out_classes,
        "attribution": attribution,
        "caveats": [
            "Relaxed gas-phase QM scans of isolated molecules. They give the "
            "INTRINSIC cost of a torsion, not what a bound conformation can "
            "afford: a protein may well pay 2 kcal/mol for a better contact.",
            "Each curve pools chemically related bonds, so substituents move "
            "individual cases away from the median. The interquartile range "
            "is reported for that reason.",
            "Energies are relative within a scan; they are not comparable "
            "between different molecules in absolute terms.",
        ],
    }

    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    # also place it where the core package loads it from, so it ships with
    # the bundle and the web build like the handbook does
    pkg = os.path.join(ROOT, "core", "hitpicking", "data", "torsion_qm.json")
    os.makedirs(os.path.dirname(pkg), exist_ok=True)
    with open(pkg, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)

    print("%d scans -> %d classes" % (len(files), len(out_classes)))
    for name, c in sorted(out_classes.items(),
                          key=lambda kv: -kv[1]["n_scans"]):
        pref = c["preferred_angle"]["median"]
        bar = c["barrier"]["median"]
        print("  %-42s n=%-4d prefers %3.0f deg, barrier %.1f kcal/mol"
              % (name, c["n_scans"], pref, bar))
    print("\nwrote %s" % os.path.relpath(OUT, ROOT))
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-scans", type=int, default=5,
                    help="classes with fewer scans are dropped")
    ap.add_argument("--merge-ortho", action="store_true",
                    help="do not separate ortho-substituted bonds")
    a = ap.parse_args()
    build(min_scans=a.min_scans, split_ortho=not a.merge_ortho)


if __name__ == "__main__":
    main()
