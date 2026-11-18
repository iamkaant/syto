"""MDL SDF/MOL (V2000) reader.

Unlike the ChimeraX path, SDF *does* carry bond orders. They are used here to
derive hybridisation and aromaticity and are then discarded: nothing
downstream may depend on them, because ChimeraX cannot supply them.
"""

from __future__ import annotations

import gzip
import io as _io
import numpy as np

from ..elements import normalize_element
from ..molsys import (MolSystem, HYB_SP, HYB_SP2, HYB_SP3, HYB_UNKNOWN,
                      PART_LIGAND)
from ..perceive.conjugation import mark_conjugated

# MDL charge codes (atom block column 7): 1=+3, 2=+2, 3=+1, 4=doublet,
# 5=-1, 6=-2, 7=-3. Superseded by 'M  CHG' but still written by old tools.
_MDL_CHG = {0: 0, 1: 3, 2: 2, 3: 1, 4: 0, 5: -1, 6: -2, 7: -3}

RECORD_SEPARATOR = "$" * 4

# Max out-of-plane deviation (A) still accepted as an aromatic ring.
# Deliberately loose; see _ring_aromatic_kekule.
RING_PLANARITY_TOL = 0.30


def _open(path):
    if str(path).endswith(".gz"):
        return _io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8",
                                 errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_sdf_records(path):
    """Yield the lines of each record (split on the SDF record separator)."""
    buf = []
    with _open(path) as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if ln.strip() == RECORD_SEPARATOR:
                if buf:
                    yield buf
                buf = []
            else:
                buf.append(ln)
    if buf and any(s.strip() for s in buf):
        yield buf


def _parse_record(lines, part, source):
    if len(lines) < 4:
        return None
    title = lines[0].strip()
    counts = lines[3]
    try:
        na = int(counts[0:3])
        nb = int(counts[3:6])
    except ValueError:
        return None
    if na <= 0:
        return None

    xyz = np.empty((na, 3), dtype=np.float64)
    element = np.empty(na, dtype="<U2")
    fchg = np.zeros(na, dtype=np.int8)

    for k in range(na):
        ln = lines[4 + k]
        xyz[k] = (float(ln[0:10]), float(ln[10:20]), float(ln[20:30]))
        element[k] = normalize_element(ln[31:34])
        if len(ln) >= 39:
            try:
                fchg[k] = _MDL_CHG.get(int(ln[36:39]), 0)
            except ValueError:
                pass

    bonds = np.zeros((nb, 2), dtype=np.int32)
    order = np.zeros(nb, dtype=np.int8)
    for k in range(nb):
        ln = lines[4 + na + k]
        a = int(ln[0:3]) - 1
        b = int(ln[3:6]) - 1
        try:
            o = int(ln[6:9])
        except ValueError:
            o = 1
        bonds[k] = (a, b)
        order[k] = o

    props = {}
    i = 4 + na + nb
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("M  END"):
            i += 1
            break
        if ln.startswith("M  CHG"):
            f = ln.split()
            cnt = int(f[2])
            for c in range(cnt):
                ai = int(f[3 + 2 * c]) - 1
                cv = int(f[4 + 2 * c])
                if 0 <= ai < na:
                    fchg[ai] = cv
        i += 1
    while i < len(lines):
        ln = lines[i]
        if ln.startswith(">"):
            s, e = ln.find("<"), ln.rfind(">")
            key = ln[s + 1:e] if s != -1 and e > s else ln.strip("> ")
            vals = []
            i += 1
            while i < len(lines) and lines[i].strip() != "":
                vals.append(lines[i])
                i += 1
            raw = "\n".join(vals).strip()
            try:
                props[key] = float(raw)
            except ValueError:
                props[key] = raw
        i += 1

    hyb, arom, conj = _perceive(xyz, element, bonds, order)

    meta = {"source": str(source), "mol_name": title, "format": "sdf",
            "props": props, "bond_order": order, "conjugated": conj}
    return MolSystem(xyz, element, bonds, hyb=hyb, aromatic=arom,
                     formal_charge=fchg,
                     part=np.full(na, part, dtype=np.int8), meta=meta)


def _perceive(xyz, element, bonds, order):
    """Derive hybridisation and aromaticity from bond orders + geometry."""
    n = len(element)
    hyb = np.full(n, HYB_SP3, dtype=np.int8)
    arom = np.zeros(n, dtype=bool)

    max_order = np.zeros(n, dtype=np.int8)
    for (a, b), o in zip(bonds, order):
        oo = 2 if o == 4 else o          # 4 = aromatic query bond
        if oo > max_order[a]:
            max_order[a] = oo
        if oo > max_order[b]:
            max_order[b] = oo
        if o == 4:
            arom[a] = True
            arom[b] = True

    hyb[max_order == 2] = HYB_SP2
    hyb[max_order == 3] = HYB_SP
    hyb[element == "H"] = HYB_SP3

    # cumulated double bonds (allene, ketene) -> sp
    dcount = np.zeros(n, dtype=np.int8)
    for (a, b), o in zip(bonds, order):
        if o == 2:
            dcount[a] += 1
            dcount[b] += 1
    hyb[(dcount >= 2) & (element == "C")] = HYB_SP

    # Steric number overrides bond order. A sulfonyl S carries two S=O bonds
    # but has four sigma substituents and is tetrahedral, not planar; the same
    # holds for sulfoxides, phosphine oxides and phosphates. Typing these sp2
    # would also drag the attached sulfonamide N along with them.
    deg = (np.bincount(bonds.ravel(), minlength=n).astype(np.int8)
           if len(bonds) else np.zeros(n, dtype=np.int8))
    hyb[deg >= 4] = HYB_SP3
    hyb[np.isin(element, ("S", "P")) & (deg >= 3)] = HYB_SP3

    tmp = MolSystem(xyz, element, bonds, hyb=hyb, aromatic=arom)
    for ring in tmp.rings():
        if _ring_aromatic_kekule(tmp, ring, max_order):
            for a in ring:
                arom[a] = True
                if element[a] != "H":
                    hyb[a] = HYB_SP2

    # Lone-pair conjugation (amide N, ester O, aniline N ...). Shared with
    # every other input path so all front ends type these atoms identically.
    tmp2 = MolSystem(xyz, element, bonds, hyb=hyb, aromatic=arom)
    conj = mark_conjugated(tmp2, set_hyb=True,
                           own_multiple_bond=(max_order >= 2))
    return tmp2.hyb, arom, conj


def _ring_aromatic_kekule(ms, ring, max_order) -> bool:
    """Aromatic if planar and every member is sp2 or a lone-pair heteroatom.

    Covers benzene/pyridine (all sp2) and pyrrole/furan/thiophene/imidazole
    (one sp3-looking heteroatom contributing its lone pair). Deliberately
    conservative: a ring that fails these tests is reported non-aromatic
    rather than guessed.
    """
    if len(ring) not in (5, 6):
        return False
    idx = list(ring)
    el = ms.element
    for a in idx:
        if max_order[a] >= 2:
            continue
        if str(el[a]) in ("N", "O", "S") and len(ms.heavy_neighbors(a)) <= 3:
            continue
        return False
    if not any(max_order[a] >= 2 for a in idx):
        return False                      # fully saturated ring
    # Planarity is a loose sanity check only. A *docked pose* can be
    # measurably distorted while the molecule is unambiguously aromatic, so
    # the tolerance is generous here and genuine ring puckering is reported
    # by hitpicking.strain instead of silently altering the chemistry.
    c, nrm = ms.ring_plane(ring)
    dev = np.abs((ms.xyz[idx] - c) @ nrm)
    return bool(dev.max() < RING_PLANARITY_TOL)


def read_sdf(path, part=PART_LIGAND, limit=None):
    out = []
    for i, rec in enumerate(iter_sdf_records(path)):
        if limit is not None and i >= limit:
            break
        ms = _parse_record(rec, part, path)
        if ms is not None:
            out.append(ms)
    return out


def read_sdf_iter(path, part=PART_LIGAND):
    for rec in iter_sdf_records(path):
        ms = _parse_record(rec, part, path)
        if ms is not None:
            yield ms
