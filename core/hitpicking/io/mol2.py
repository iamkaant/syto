"""Tripos mol2 reader, with UCSF DOCK's comment block.

Handles the multi-record mol2 that DOCK 3.7/3.8 emits (one @<TRIPOS>MOLECULE
per pose, each preceded by a '##########  Key: value' comment block carrying
the score terms). Sybyl atom types give hybridisation and aromaticity
directly, so no bond-order perception is needed on this path.
"""

from __future__ import annotations

import gzip
import io as _io
import numpy as np

from ..molsys import (MolSystem, HYB_SP, HYB_SP2, HYB_SP3, HYB_UNKNOWN,
                      PART_LIGAND, PART_RECEPTOR, PART_WATER)
from ..perceive.conjugation import mark_conjugated

# Sybyl atom type -> (element, hybridisation, aromatic, formal charge)
# Formal charge is set only where the type itself determines it. Charged
# *groups* (carboxylate, guanidinium, amidinium) are perceived separately in
# perceive.charges, because they span several atoms.
SYBYL = {
    "C.3":   ("C", HYB_SP3, False, 0),
    "C.2":   ("C", HYB_SP2, False, 0),
    "C.1":   ("C", HYB_SP,  False, 0),
    "C.ar":  ("C", HYB_SP2, True,  0),
    "C.cat": ("C", HYB_SP2, False, 0),   # guanidinium central C
    "N.3":   ("N", HYB_SP3, False, 0),
    "N.2":   ("N", HYB_SP2, False, 0),
    "N.1":   ("N", HYB_SP,  False, 0),
    "N.ar":  ("N", HYB_SP2, True,  0),
    "N.am":  ("N", HYB_SP2, False, 0),   # amide N: planar, poor acceptor
    "N.pl3": ("N", HYB_SP2, False, 0),   # planar N (aniline, amidine)
    "N.4":   ("N", HYB_SP3, False, 1),   # quaternary / protonated amine
    "O.3":   ("O", HYB_SP3, False, 0),
    "O.2":   ("O", HYB_SP2, False, 0),
    "O.co2": ("O", HYB_SP2, False, 0),   # carboxylate O (group charge -1)
    "O.spc": ("O", HYB_SP3, False, 0),
    "O.t3p": ("O", HYB_SP3, False, 0),
    "S.3":   ("S", HYB_SP3, False, 0),
    "S.2":   ("S", HYB_SP2, False, 0),
    "S.o":   ("S", HYB_SP3, False, 0),   # sulfoxide
    "S.o2":  ("S", HYB_SP3, False, 0),   # sulfone
    "S.O":   ("S", HYB_SP3, False, 0),
    "S.O2":  ("S", HYB_SP3, False, 0),
    "P.3":   ("P", HYB_SP3, False, 0),
    "H":     ("H", HYB_UNKNOWN, False, 0),
    "H.spc": ("H", HYB_UNKNOWN, False, 0),
    "H.t3p": ("H", HYB_UNKNOWN, False, 0),
    "F":     ("F", HYB_SP3, False, 0),
    "Cl":    ("Cl", HYB_SP3, False, 0),
    "Br":    ("Br", HYB_SP3, False, 0),
    "I":     ("I", HYB_SP3, False, 0),
    "Du":    ("",  HYB_UNKNOWN, False, 0),
}

WATER_RESNAMES = frozenset({"HOH", "WAT", "H2O", "TIP", "TIP3", "SOL", "DOD"})


def _open(path):
    if str(path).endswith(".gz"):
        return _io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _sybyl_lookup(t):
    """Resolve a Sybyl type, tolerating case variants and unknown types."""
    if t in SYBYL:
        return SYBYL[t]
    # case-insensitive retry (S.O2 vs S.o2 etc.)
    for k, v in SYBYL.items():
        if k.lower() == t.lower():
            return v
    from ..elements import normalize_element
    el = normalize_element(t)
    return (el, HYB_UNKNOWN, False, 0)


def parse_dock_comments(lines):
    """Parse DOCK's '##########  Key:  value' block into a dict.

    Repeated keys (OXR/OXS matching-sphere lines) collect into a list.
    Numeric-looking values are converted to float.
    """
    out = {}
    for ln in lines:
        if not ln.startswith("#"):
            continue
        body = ln.lstrip("#").strip()
        if ":" not in body:
            continue
        key, _, val = body.partition(":")
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        try:
            conv = float(val)
        except ValueError:
            conv = val
        if key in out:
            if not isinstance(out[key], list):
                out[key] = [out[key]]
            out[key].append(conv)
        else:
            out[key] = conv
    return out


def iter_mol2_records(path):
    """Yield (comment_lines, record_lines) for each @<TRIPOS>MOLECULE block.

    DOCK writes each pose's '##########' block *before* its MOLECULE line, so
    comments seen while a record is open belong to the NEXT record, not the
    one currently being read.
    """
    cur_comments, pending, record = [], [], []
    in_record = False
    with _open(path) as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if ln.startswith("@<TRIPOS>MOLECULE"):
                if in_record:
                    yield cur_comments, record
                    cur_comments, pending = pending, []
                record = [ln]
                in_record = True
                continue
            if ln.startswith("#"):
                (pending if in_record else cur_comments).append(ln)
                continue
            if in_record:
                record.append(ln)
    if in_record:
        yield cur_comments, record


def _parse_record(comment_lines, lines, part_default, source):
    section = None
    atoms, bonds = [], []
    mol_name = ""
    header_i = 0
    for ln in lines:
        s = ln.strip()
        if s.startswith("@<TRIPOS>"):
            section = s[len("@<TRIPOS>"):].strip().upper()
            header_i = 0
            continue
        if section == "MOLECULE":
            if header_i == 0 and s:
                mol_name = s
            if s:
                header_i += 1
            continue
        if section == "ATOM":
            if not s:
                continue
            f = s.split()
            if len(f) < 6:
                continue
            atoms.append(f)
            continue
        if section == "BOND":
            if not s:
                continue
            f = s.split()
            if len(f) < 4:
                continue
            bonds.append((int(f[1]), int(f[2]), f[3]))
            continue

    n = len(atoms)
    if n == 0:
        return None

    xyz = np.empty((n, 3), dtype=np.float64)
    element = np.empty(n, dtype="<U2")
    name = np.empty(n, dtype="<U6")
    hyb = np.zeros(n, dtype=np.int8)
    arom = np.zeros(n, dtype=bool)
    fchg = np.zeros(n, dtype=np.int8)
    res_name = np.empty(n, dtype="<U4")
    res_seq = np.zeros(n, dtype=np.int32)
    part = np.full(n, part_default, dtype=np.int8)
    sybyl = np.empty(n, dtype="<U6")
    pcharge = np.zeros(n, dtype=np.float64)
    serial_to_idx = {}

    for k, f in enumerate(atoms):
        serial_to_idx[int(f[0])] = k
        name[k] = f[1][:6]
        xyz[k] = (float(f[2]), float(f[3]), float(f[4]))
        t = f[5]
        sybyl[k] = t[:6]
        el, h, ar, fc = _sybyl_lookup(t)
        element[k], hyb[k], arom[k], fchg[k] = el, h, ar, fc
        rn = f[7][:4] if len(f) > 7 else ""
        res_name[k] = rn
        res_seq[k] = int(f[6]) if len(f) > 6 and f[6].lstrip("-").isdigit() else 0
        if len(f) > 8:
            try:
                pcharge[k] = float(f[8])
            except ValueError:
                pass
        if rn.upper()[:3] in WATER_RESNAMES or rn.upper() in WATER_RESNAMES:
            part[k] = PART_WATER

    bidx = []
    for a, b, bt in bonds:
        ia, ib = serial_to_idx.get(a), serial_to_idx.get(b)
        if ia is None or ib is None:
            continue
        bidx.append((ia, ib))
        if bt.lower() == "ar":
            arom[ia] = True
            arom[ib] = True
            if hyb[ia] == HYB_UNKNOWN:
                hyb[ia] = HYB_SP2
            if hyb[ib] == HYB_UNKNOWN:
                hyb[ib] = HYB_SP2

    meta = {
        "source": str(source),
        "mol_name": mol_name,
        "dock": parse_dock_comments(comment_lines),
        "sybyl": sybyl,
        "partial_charge": pcharge,
        "format": "mol2",
    }
    if not meta["dock"].get("Name") and mol_name:
        meta["dock"]["Name"] = mol_name

    ms = MolSystem(
        xyz, element, np.array(bidx, dtype=np.int32).reshape(-1, 2),
        name=name, hyb=hyb, aromatic=arom, formal_charge=fchg,
        part=part, res_name=res_name, res_seq=res_seq, meta=meta,
    )
    # Sybyl types already distinguish N.am from N.3, but the conjugation flag
    # is computed the same way on every path so the handbook sees one answer.
    mark_conjugated(ms, set_hyb=True)
    return ms


def read_mol2(path, part=PART_LIGAND, limit=None):
    """Read every record of a mol2 file. Returns a list of MolSystem."""
    out = []
    for i, (comments, rec) in enumerate(iter_mol2_records(path)):
        if limit is not None and i >= limit:
            break
        ms = _parse_record(comments, rec, part, path)
        if ms is not None:
            out.append(ms)
    return out


def read_mol2_iter(path, part=PART_LIGAND):
    """Streaming variant: yields one MolSystem per record."""
    for comments, rec in iter_mol2_records(path):
        ms = _parse_record(comments, rec, part, path)
        if ms is not None:
            yield ms


def pose_name(ms) -> str:
    d = ms.meta.get("dock", {})
    v = d.get("Name") or ms.meta.get("mol_name") or ""
    return str(v).strip()


def dock_scores(ms) -> dict:
    """The numeric DOCK score terms from the comment block."""
    return {k: v for k, v in ms.meta.get("dock", {}).items()
            if isinstance(v, float)}
