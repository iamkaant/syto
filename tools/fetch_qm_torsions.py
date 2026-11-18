#!/usr/bin/env python3
"""Pull relaxed QM torsion scans from QCArchive and classify them our way.

Why
---
The tool had no torsion reference at all: no redistributable CSD library
exists, so torsions were judged only where the chemistry is unambiguous and
everything else was marked `unjudged`. QCArchive holds tens of thousands of
RELAXED QM torsion drives, which give something the CSD cannot: the energy
cost in kcal/mol of sitting at a given angle, not merely how often that
angle is observed.

Licence
-------
The public QCArchive server states no site-wide data licence - checked, and
none found. The OpenFF `qca-dataset-submission` repository DOES state, in its
LICENSE file, "All datasets in this repository are licensed under the
CC-BY-4.0 license." So by default this pulls only datasets submitted by
OpenFF, and records that provenance with every number. Third-party sets
(XtalPi, SiliconTX and others) are included only with --all-datasets, which
prints a warning: their terms were not verified.

CC-BY-4.0 requires attribution and a statement of changes. The compiled
output carries both.

Usage
-----
    python tools/fetch_qm_torsions.py --list
    python tools/fetch_qm_torsions.py --datasets "OpenFF Substituted Phenyl Set 1"
    python tools/fetch_qm_torsions.py --openff-all        # every OpenFF set
    python tools/fetch_qm_torsions.py --openff-all -j 8   # on a cluster

Writes one JSON per record into --out (default data-src/external/qm_torsions/
raw/), so it is resumable: re-running skips what is already there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HARTREE_TO_KCAL = 627.509474
SERVER = "https://api.qcarchive.molssi.org"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_OUT = os.path.join(ROOT, "data-src", "external", "qm_torsions")

sys.path.insert(0, os.path.join(ROOT, "core"))

# Datasets whose licence we can point at: the OpenFF repo states CC-BY-4.0.
OPENFF_PREFIXES = ("OpenFF", "SMIRNOFF")

ATTRIBUTION = {
    "source": "MolSSI QCArchive (public server)",
    "server": SERVER,
    "submitter": "Open Force Field Initiative",
    "dataset_licence": "CC-BY-4.0",
    "licence_statement": (
        "All datasets in this repository are licensed under the CC-BY-4.0 "
        "license. (openforcefield/qca-dataset-submission, LICENSE)"),
    "licence_url": "https://github.com/openforcefield/qca-dataset-submission"
                   "/blob/master/LICENSE",
    "changes": (
        "Relaxed torsion-drive energies were converted from hartree to "
        "kcal/mol relative to each scan's own minimum, and each scan was "
        "classified by the rotatable-bond type used by this tool. No "
        "energies were recomputed."),
    "server_licence_note": (
        "The public QCArchive server itself states no site-wide data "
        "licence; only OpenFF-submitted datasets are used by default."),
}


def client():
    from qcportal import PortalClient
    return PortalClient(SERVER)


def list_datasets(c, openff_only=True):
    out = []
    for row in c.list_datasets():
        if row.get("dataset_type") != "torsiondrive":
            continue
        name = row.get("dataset_name", "")
        if openff_only and not name.startswith(OPENFF_PREFIXES):
            continue
        out.append((name, row.get("record_count", 0) or 0))
    out.sort(key=lambda t: -t[1])
    return out


# --------------------------------------------------------------- chemistry

def molsys_from_qcmol(qcmol):
    """A MolSystem from a QCElemental molecule, so OUR perception applies.

    Classifying with the same code the tool runs at analysis time is the
    whole point: a profile filed under "anilide N" must mean what the tool
    means by that when it looks at a pose.
    """
    import numpy as np
    from hitpicking.molsys import MolSystem, PART_LIGAND
    from hitpicking.perceive.conjugation import mark_conjugated

    BOHR = 0.529177210903
    xyz = np.asarray(qcmol.geometry, dtype=float).reshape(-1, 3) * BOHR
    symbols = [str(s).capitalize() for s in qcmol.symbols]
    bonds = []
    orders = []
    for a, b, o in (qcmol.connectivity or []):
        bonds.append((int(a), int(b)))
        orders.append(float(o))
    if not bonds:
        return None

    n = len(symbols)
    import numpy as _np
    max_order = _np.zeros(n)
    for (a, b), o in zip(bonds, orders):
        max_order[a] = max(max_order[a], o)
        max_order[b] = max(max_order[b], o)

    from hitpicking.molsys import HYB_SP, HYB_SP2, HYB_SP3
    hyb = _np.full(n, HYB_SP3, dtype=_np.int8)
    hyb[max_order >= 2] = HYB_SP2
    hyb[max_order >= 3] = HYB_SP
    deg = _np.bincount(_np.asarray(bonds).ravel(), minlength=n)
    hyb[deg >= 4] = HYB_SP3
    hyb[_np.isin(symbols, ("S", "P")) & (deg >= 3)] = HYB_SP3

    ms = MolSystem(xyz, symbols, bonds, hyb=hyb,
                   part=_np.full(n, PART_LIGAND, dtype=_np.int8),
                   res_name=_np.array(["LIG"] * n, dtype="<U4"))

    # aromaticity by the same rule the SDF reader uses
    from hitpicking.io.sdf import _ring_aromatic_kekule
    arom = _np.zeros(n, dtype=bool)
    for ring in ms.rings():
        if _ring_aromatic_kekule(ms, ring, max_order.astype(int)):
            for a in ring:
                arom[a] = True
    ms.aromatic = arom
    ms._rings = None
    mark_conjugated(ms, set_hyb=True, own_multiple_bond=(max_order >= 2))
    return ms


def classify_scan(ms, dihedral):
    """Our bond type and subtype for the scanned dihedral."""
    from hitpicking.strain import torsions as T
    p, a, b, q = [int(x) for x in dihedral]
    btype = T._classify_bond(ms, a, b)
    subtype = ""
    if btype in (T.CONJUGATED, T.BIARYL):
        subtype = (T.conjugated_subtype(ms, a, b) if btype == T.CONJUGATED
                   else "biaryl")
    elif btype == T.AMIDE:
        subtype = "amide C-N"
    ortho = False
    for ring_side, other in ((a, b), (b, a)):
        if ms.aromatic[ring_side]:
            ortho = T._has_ortho_substituent(ms, ring_side, other)
            break
    return btype, subtype, ortho


# ------------------------------------------------------------------ fetch

def profile_from_record(rec):
    """[(angle_deg, relative_energy_kcal)] sorted by angle, or None."""
    energies = rec.final_energies
    if not energies:
        return None
    pts = []
    for key, val in energies.items():
        try:
            angle = float(key[0])
        except Exception:
            continue
        pts.append((angle, float(val)))
    if len(pts) < 6:
        return None
    pts.sort()
    lo = min(v for _, v in pts)
    return [(a, (v - lo) * HARTREE_TO_KCAL) for a, v in pts]


def fetch_dataset(c, name, out_dir, limit=None, verbose=True):
    ds = c.get_dataset("torsiondrive", name)
    os.makedirs(out_dir, exist_ok=True)
    done = skipped = failed = 0
    t0 = time.time()
    for entry_name, spec_name, rec in ds.iterate_records(status="complete"):
        if limit and done >= limit:
            break
        if rec is None:
            continue
        path = os.path.join(out_dir, "%d.json" % rec.id)
        if os.path.exists(path):
            skipped += 1
            continue
        try:
            prof = profile_from_record(rec)
            if prof is None:
                failed += 1
                continue
            dihedrals = rec.specification.keywords.dihedrals
            if not dihedrals or len(dihedrals[0]) != 4:
                failed += 1
                continue
            qcmol = rec.initial_molecules[0]
            ms = molsys_from_qcmol(qcmol)
            if ms is None:
                failed += 1
                continue
            btype, subtype, ortho = classify_scan(ms, dihedrals[0])
            payload = {
                "record_id": rec.id,
                "dataset": name,
                "entry": entry_name,
                "spec": spec_name,
                "dihedral": [int(x) for x in dihedrals[0]],
                "bond_type": btype,
                "subtype": subtype,
                "ortho_substituted": bool(ortho),
                "n_atoms": ms.n_atoms,
                "profile": [[round(a, 1), round(e, 4)] for a, e in prof],
            }
            with open(path, "w") as fh:
                json.dump(payload, fh)
            done += 1
            if verbose and done % 50 == 0:
                print("    %s: %d fetched (%.0fs)"
                      % (name[:40], done, time.time() - t0), flush=True)
        except Exception as e:
            failed += 1
            if verbose and failed <= 3:
                print("    skip %s: %s" % (str(entry_name)[:40], e))
    return done, skipped, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list datasets and exit")
    ap.add_argument("--datasets", nargs="*", help="dataset names to fetch")
    ap.add_argument("--openff-all", action="store_true",
                    help="every OpenFF-submitted torsiondrive dataset")
    ap.add_argument("--all-datasets", action="store_true",
                    help="include third-party sets whose licence is unverified")
    ap.add_argument("--limit", type=int, help="max records per dataset")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    c = client()
    openff_only = not a.all_datasets
    available = list_datasets(c, openff_only=openff_only)

    if a.list:
        total = sum(n for _, n in available)
        print("%d torsiondrive datasets, %d scans%s:"
              % (len(available), total,
                 "" if a.all_datasets else " (OpenFF-submitted only)"))
        for name, n in available:
            print("  %6d  %s" % (n, name))
        if a.all_datasets:
            print("\nWARNING: third-party datasets are included. Their terms "
                  "were not verified; only the OpenFF sets have a stated "
                  "CC-BY-4.0 licence.")
        return

    if a.openff_all:
        names = [n for n, _ in available]
    elif a.datasets:
        names = a.datasets
    else:
        ap.error("give --datasets, --openff-all or --list")

    raw = os.path.join(a.out, "raw")
    os.makedirs(raw, exist_ok=True)
    with open(os.path.join(a.out, "ATTRIBUTION.json"), "w") as fh:
        json.dump(ATTRIBUTION, fh, indent=1)

    grand = 0
    for name in names:
        print("==> %s" % name, flush=True)
        try:
            done, skipped, failed = fetch_dataset(c, name, raw, a.limit)
        except Exception as e:
            print("    FAILED: %s" % e)
            continue
        grand += done
        print("    %d new, %d already present, %d skipped"
              % (done, skipped, failed), flush=True)
    print("\n%d new scans in %s" % (grand, raw))
    print("next:  python tools/build_torsion_profiles.py")


if __name__ == "__main__":
    main()
