#!/usr/bin/env python3
"""Derive torsion-angle FREQUENCY distributions from the COD.

This is the open analogue of a Mogul search, and it answers a different
question from the QM scans: not "what does this twist cost" but "what angles
do real crystal structures actually show". Both are worth having. A torsion
can be cheap and still rare, or common and strained by a packing effect.

Licence
-------
The Crystallography Open Database states, verbatim:

    "All data in the COD and the database itself are dedicated to the public
     domain and licensed under the CC0 License. Users of the data should
     acknowledge the original authors of the structural data."

CC0, so derived histograms are freely redistributable. The acknowledgement
is a request rather than a condition, and the compiled output carries it.

Getting the data (tens of GB; see tools/fetch_cod.sh)
-----------------------------------------------------
    rsync -av --delete rsync://www.crystallography.net/cif/ cif/

Running it (designed for a cluster array job)
---------------------------------------------
    python tools/cod_torsions.py --cif-dir cif/ --shard 0/64 --out shards/
    ...
    python tools/cod_torsions.py --merge shards/ --out data-src/external/

Each shard writes its own counts, so N jobs can run independently and the
merge is a sum. Re-running a shard overwrites only its own file.

Deliberate limits
-----------------
* Only organic structures are used: carbon present, no metals, and a
  reasonable size. COD is mostly minerals and inorganics by count.
* Bonds are inferred from distance, exactly as the PDB reader does, because
  CIFs carry no connectivity.
* Disordered and partially occupied sites are skipped: a torsion measured
  across a disordered atom is meaningless.
* Hydrogens are frequently absent or calculated in COD entries, so torsions
  needing a hydrogen reference (an -NH2 or -OH) are counted separately and
  can be excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "core"))

BIN_DEG = 10.0
MAX_ATOMS = 300
METALS_EXCLUDED = True

ATTRIBUTION = {
    "source": "Crystallography Open Database (COD)",
    "url": "https://www.crystallography.net/cod/",
    "licence": "CC0 1.0 (public domain dedication)",
    "licence_statement": (
        "All data in the COD and the database itself are dedicated to the "
        "public domain and licensed under the CC0 License. Users of the data "
        "should acknowledge the original authors of the structural data."),
    "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
    "changes": (
        "Torsion angles were measured from deposited coordinates after "
        "inferring connectivity from interatomic distance, classified by the "
        "rotatable-bond types used by this tool, and binned into histograms. "
        "No coordinates are redistributed."),
    "acknowledgement": (
        "Structural data were deposited in the COD by their original authors, "
        "who are acknowledged collectively here; per-structure attribution is "
        "retained in the COD entries themselves."),
}


def read_cif(path):
    """(symbols, xyz) for one CIF, or None if it is unusable.

    Uses gemmi when available - it handles the dialect properly - and falls
    back to a deliberately strict minimal parser so a cluster node without
    gemmi still does something sane rather than something wrong.
    """
    try:
        import gemmi
    except ImportError:
        return _read_cif_minimal(path)
    try:
        doc = gemmi.cif.read(path)
        block = doc.sole_block()
        st = gemmi.make_small_structure_from_block(block)
        if st is None or not st.sites:
            return None
        symbols, xyz = [], []
        for site in st.sites:
            if site.occ and site.occ < 0.99:
                return None          # disorder: skip the whole structure
            el = site.element.name
            pos = st.cell.orthogonalize(site.fract)
            symbols.append(el)
            xyz.append([pos.x, pos.y, pos.z])
        return symbols, xyz
    except Exception:
        return None


def _read_cif_minimal(path):
    """Fallback: read _atom_site_ loops with fractional coordinates."""
    import math
    try:
        text = open(path, "r", errors="replace").read()
    except Exception:
        return None
    cell = {}
    for key in ("length_a", "length_b", "length_c",
                "angle_alpha", "angle_beta", "angle_gamma"):
        for line in text.splitlines():
            if line.strip().startswith("_cell_" + key):
                try:
                    cell[key] = float(line.split()[1].split("(")[0])
                except Exception:
                    pass
                break
    if len(cell) != 6:
        return None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        j, tags = i + 1, []
        while j < len(lines) and lines[j].strip().startswith("_"):
            tags.append(lines[j].strip())
            j += 1
        if not any(t.startswith("_atom_site_fract_x") for t in tags):
            i = j
            continue
        idx = {t: k for k, t in enumerate(tags)}
        need = ["_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
        if not all(t in idx for t in need):
            return None
        sym_tag = ("_atom_site_type_symbol" if "_atom_site_type_symbol" in idx
                   else "_atom_site_label")
        symbols, frac = [], []
        while j < len(lines):
            row = lines[j].strip()
            if not row or row.startswith(("_", "loop_", "#", "data_")):
                break
            parts = row.split()
            if len(parts) < len(tags):
                break
            try:
                f = [float(parts[idx[t]].split("(")[0]) for t in need]
            except Exception:
                break
            raw = parts[idx[sym_tag]]
            symbols.append("".join(ch for ch in raw if ch.isalpha())[:2])
            frac.append(f)
            j += 1
        if not symbols:
            return None
        a, b, c = cell["length_a"], cell["length_b"], cell["length_c"]
        al, be, ga = (math.radians(cell["angle_" + k])
                      for k in ("alpha", "beta", "gamma"))
        v = math.sqrt(max(1e-9, 1 - math.cos(al) ** 2 - math.cos(be) ** 2
                          - math.cos(ga) ** 2
                          + 2 * math.cos(al) * math.cos(be) * math.cos(ga)))
        M = [[a, b * math.cos(ga), c * math.cos(be)],
             [0, b * math.sin(ga),
              c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)],
             [0, 0, c * v / math.sin(ga)]]
        xyz = [[sum(M[r][k] * f[k] for k in range(3)) for r in range(3)]
               for f in frac]
        return symbols, xyz
    return None


def is_organic(symbols):
    from hitpicking.elements import METALS
    s = set(symbols)
    if "C" not in s:
        return False
    if METALS_EXCLUDED and (s & METALS):
        return False
    return len(symbols) <= MAX_ATOMS


def torsions_for_structure(symbols, xyz):
    """Classified torsions for one crystal structure."""
    import numpy as np
    from hitpicking.molsys import MolSystem, PART_LIGAND
    from hitpicking.io.bonding import infer_bonds
    from hitpicking.io.sdf import _ring_aromatic_kekule
    from hitpicking.perceive.conjugation import mark_conjugated
    from hitpicking.molsys import HYB_SP2, HYB_SP3
    from hitpicking.strain import torsions as T

    n = len(symbols)
    ms = MolSystem(np.asarray(xyz, dtype=float),
                   np.asarray(symbols, dtype="<U2"), None,
                   part=np.full(n, PART_LIGAND, dtype=np.int8),
                   res_name=np.array(["LIG"] * n, dtype="<U4"))
    ms.bonds = infer_bonds(ms)
    ms._adj = None
    if len(ms.bonds) < 3:
        return []

    # geometric typing: a crystal structure carries no bond orders
    hyb = np.full(n, HYB_SP3, dtype=np.int8)
    for i in range(n):
        if str(ms.element[i]) == "H":
            continue
        nb = ms.neighbors(i)
        if len(nb) == 3:
            p = ms.xyz[nb[:3]]
            nrm = np.cross(p[1] - p[0], p[2] - p[0])
            nn = np.linalg.norm(nrm)
            if nn > 1e-8 and abs(float(np.dot(ms.xyz[i] - p[0], nrm / nn))) < 0.2:
                hyb[i] = HYB_SP2
        elif len(nb) == 2 and str(ms.element[i]) in ("C", "N"):
            ang = ms.angle(int(nb[0]), i, int(nb[1]))
            if ang > 155:
                hyb[i] = 1
            elif ang > 115:
                hyb[i] = HYB_SP2
    ms.hyb = hyb
    ms._rings = None
    arom = np.zeros(n, dtype=bool)
    pseudo_order = np.where(hyb == HYB_SP2, 2, 1)
    for ring in ms.rings():
        if _ring_aromatic_kekule(ms, ring, pseudo_order):
            for a in ring:
                arom[a] = True
    ms.aromatic = arom
    ms._rings = None
    mark_conjugated(ms, set_hyb=True, own_multiple_bond=(hyb == HYB_SP2))

    out = []
    has_h = bool((ms.element == "H").any())
    for t in T.analyze_torsions(ms, ligand_only=False):
        p, a, b, q = t.atoms
        needs_h = (str(ms.element[p]) == "H" or str(ms.element[q]) == "H")
        out.append((t.bond_type, t.subtype, t.ortho_substituted,
                    t.angle, needs_h, has_h))
    return out


def fold(angle, bond_type):
    a = abs(float(angle)) % 360.0
    if a > 180.0:
        a = 360.0 - a
    if bond_type == "sp3-sp3":
        f = a % 120.0
        return min(f, 120.0 - f)
    return min(a, 180.0 - a)


def run_shard(cif_dir, shard, nshards, out_dir, limit=None, verbose=True):
    import collections
    counts = collections.defaultdict(collections.Counter)
    stats = collections.Counter()
    t0 = time.time()

    files = []
    for dirpath, _, names in os.walk(cif_dir):
        for nm in names:
            if nm.endswith(".cif"):
                files.append(os.path.join(dirpath, nm))
    files.sort()
    mine = [f for k, f in enumerate(files) if k % nshards == shard]
    if limit:
        mine = mine[:limit]
    if verbose:
        print("shard %d/%d: %d of %d CIFs" % (shard, nshards, len(mine),
                                              len(files)), flush=True)

    for k, path in enumerate(mine):
        stats["seen"] += 1
        parsed = read_cif(path)
        if parsed is None:
            stats["unparsed"] += 1
            continue
        symbols, xyz = parsed
        if not is_organic(symbols):
            stats["not_organic"] += 1
            continue
        try:
            found = torsions_for_structure(symbols, xyz)
        except Exception:
            stats["failed"] += 1
            continue
        stats["used"] += 1
        for bond_type, subtype, ortho, angle, needs_h, has_h in found:
            if needs_h and not has_h:
                continue
            key = "%s|%s|%s%s" % (bond_type, subtype or "-",
                                  "ortho" if ortho else "no-ortho",
                                  "|H" if needs_h else "")
            counts[key][int(round(fold(angle, bond_type) / BIN_DEG))] += 1
            stats["torsions"] += 1
        if verbose and (k + 1) % 2000 == 0:
            print("  %d/%d  (%.0fs, %d torsions)"
                  % (k + 1, len(mine), time.time() - t0, stats["torsions"]),
                  flush=True)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "shard_%04d_of_%04d.json" % (shard, nshards))
    with open(out, "w") as fh:
        json.dump({"shard": shard, "nshards": nshards,
                   "bin_deg": BIN_DEG, "stats": dict(stats),
                   "counts": {k: dict(v) for k, v in counts.items()}}, fh)
    if verbose:
        print("wrote %s  (%s)" % (out, dict(stats)), flush=True)
    return out


def merge(shard_dir, out_dir, min_count=50):
    import collections
    import glob
    counts = collections.defaultdict(collections.Counter)
    stats = collections.Counter()
    files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.json")))
    if not files:
        sys.exit("no shard files in %s" % shard_dir)
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        stats.update(d.get("stats", {}))
        for key, hist in d.get("counts", {}).items():
            for b, c in hist.items():
                counts[key][int(b)] += int(c)

    classes = {}
    for key, hist in counts.items():
        total = sum(hist.values())
        if total < min_count:
            continue
        dist = {str(int(b * BIN_DEG)): {"count": c,
                                        "fraction": round(c / total, 4)}
                for b, c in sorted(hist.items())}
        mode = max(hist.items(), key=lambda kv: kv[1])[0]
        classes[key] = {"n_torsions": total,
                        "modal_angle": int(mode * BIN_DEG),
                        "distribution": dist}

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "torsion_cod.json")
    with open(out, "w") as fh:
        json.dump({"schema_version": 1,
                   "bin_width_deg": BIN_DEG,
                   "units": "counts and fractions of observed torsions",
                   "structures_used": stats.get("used", 0),
                   "torsions_total": stats.get("torsions", 0),
                   "min_count_per_class": min_count,
                   "classes": classes,
                   "attribution": ATTRIBUTION,
                   "caveats": [
                       "Frequencies, not energies: a rare angle is not "
                       "necessarily a costly one, and crystal packing can "
                       "favour geometries an isolated molecule would not.",
                       "Connectivity is inferred from distance, and "
                       "hybridisation and aromaticity from geometry, since "
                       "CIFs carry neither.",
                       "COD accepts open deposition and its quality is "
                       "uneven; disordered sites are skipped entirely.",
                   ]}, fh, indent=1, sort_keys=True)
    print("%d shards -> %d classes, %d torsions from %d structures"
          % (len(files), len(classes), stats.get("torsions", 0),
             stats.get("used", 0)))
    for k, c in sorted(classes.items(), key=lambda kv: -kv[1]["n_torsions"])[:15]:
        print("  %-44s n=%-7d mode %3d deg" % (k, c["n_torsions"],
                                               c["modal_angle"]))
    print("\nwrote %s" % out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cif-dir", help="directory of COD CIFs (walked recursively)")
    ap.add_argument("--shard", help="i/N, e.g. 3/64")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="max CIFs for this shard")
    ap.add_argument("--merge", help="directory of shard outputs to merge")
    ap.add_argument("--min-count", type=int, default=50)
    a = ap.parse_args()

    if a.merge:
        merge(a.merge, a.out, a.min_count)
        return
    if not a.cif_dir:
        ap.error("--cif-dir is required unless --merge is given")
    shard, nshards = 0, 1
    if a.shard:
        shard, nshards = (int(x) for x in a.shard.split("/"))
    run_shard(a.cif_dir, shard, nshards, a.out, a.limit)


if __name__ == "__main__":
    main()
