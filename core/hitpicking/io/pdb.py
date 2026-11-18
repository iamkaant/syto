"""PDB reader for receptors.

Chemistry comes from residue templates (see aa_templates); bonds are inferred
from distance. Non-standard residues fall back to geometric perception.
"""

from __future__ import annotations

import gzip
import io as _io
import numpy as np

from .. import aa_templates as T
from ..elements import normalize_element
from ..molsys import (MolSystem, PART_RECEPTOR, PART_LIGAND, PART_WATER,
                      PART_ION, PART_COFACTOR, HYB_SP2, HYB_SP3)
from .bonding import infer_bonds
from ..perceive.conjugation import mark_conjugated


def _open(path):
    if str(path).endswith(".gz"):
        return _io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def read_pdb(path, keep_waters: bool = True, keep_altloc: str = "A",
             hetatm_as_ligand=False):
    """Read a PDB file into one MolSystem.

    keep_altloc: which alternate-location indicator to keep ('' and this
    letter are kept, others dropped). Pass None to keep everything.
    """
    xs, els, names, res_names, res_seqs, chains, parts = [], [], [], [], [], [], []
    bfac, occ = [], []
    conect = []
    serial_to_idx = {}
    n_altloc_dropped = 0

    with _open(path) as fh:
        for ln in fh:
            rec = ln[:6]
            if rec in ("ATOM  ", "HETATM"):
                altloc = ln[16:17].strip()
                if keep_altloc is not None and altloc and altloc != keep_altloc:
                    n_altloc_dropped += 1
                    continue
                try:
                    x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
                except ValueError:
                    continue
                name = ln[12:16].strip()
                rn = ln[17:20].strip().upper()
                ch = ln[21:22].strip()
                try:
                    rs = int(ln[22:26])
                except ValueError:
                    rs = 0
                el_field = ln[76:78].strip()
                el = normalize_element(el_field) if el_field else _element_from_name(name)

                if rn in T.WATER_RESNAMES:
                    part = PART_WATER
                elif rn in T.ION_RESNAMES and len(name) <= 2:
                    part = PART_ION
                elif rec == "HETATM" and rn not in T.STANDARD_AA:
                    part = PART_LIGAND if hetatm_as_ligand else PART_COFACTOR
                else:
                    part = PART_RECEPTOR

                try:
                    serial = int(ln[6:11])
                    serial_to_idx[serial] = len(xs)
                except ValueError:
                    pass

                xs.append((x, y, z)); els.append(el); names.append(name[:6])
                res_names.append(rn[:4]); res_seqs.append(rs); chains.append(ch[:2])
                parts.append(part)
                try:
                    occ.append(float(ln[54:60]))
                except ValueError:
                    occ.append(1.0)
                try:
                    bfac.append(float(ln[60:66]))
                except ValueError:
                    bfac.append(0.0)
            elif rec == "CONECT":
                try:
                    a = int(ln[6:11])
                except ValueError:
                    continue
                for s in (ln[11:16], ln[16:21], ln[21:26], ln[26:31]):
                    try:
                        conect.append((a, int(s)))
                    except ValueError:
                        pass

    if not xs:
        raise ValueError("no ATOM/HETATM records in %s" % path)

    n = len(xs)
    hyb = np.zeros(n, dtype=np.int8)
    arom = np.zeros(n, dtype=bool)
    for k in range(n):
        h, a = T.classify_atom(res_names[k], names[k], els[k])
        hyb[k], arom[k] = h, a

    ms = MolSystem(
        np.array(xs), np.array(els, dtype="<U2"), None,
        name=np.array(names, dtype="<U6"), hyb=hyb, aromatic=arom,
        part=np.array(parts, dtype=np.int8),
        res_name=np.array(res_names, dtype="<U4"),
        res_seq=np.array(res_seqs, dtype=np.int32),
        chain_id=np.array(chains, dtype="<U2"),
        meta={"source": str(path), "format": "pdb",
              "bfactor": np.array(bfac), "occupancy": np.array(occ),
              "altloc_dropped": n_altloc_dropped},
    )

    if not keep_waters:
        ms = ms.subset(~ms.is_water)

    ms.bonds = infer_bonds(ms)
    ms._adj = None
    # Honour CONECT for HETATM groups where it exists and distances disagree
    if conect:
        _merge_conect(ms, conect, serial_to_idx)

    _perceive_unknown_residues(ms)
    mark_conjugated(ms, set_hyb=True)
    return ms


def _merge_conect(ms, conect, serial_to_idx):
    extra = set()
    have = set(map(tuple, np.sort(ms.bonds, axis=1).tolist())) if len(ms.bonds) else set()
    for a, b in conect:
        ia, ib = serial_to_idx.get(a), serial_to_idx.get(b)
        if ia is None or ib is None or ia == ib:
            continue
        key = (min(ia, ib), max(ia, ib))
        if key not in have:
            extra.add(key)
    if extra:
        ms.bonds = np.concatenate(
            [ms.bonds, np.array(sorted(extra), dtype=np.int32)]).astype(np.int32)
        ms._adj = None


def _element_from_name(name: str) -> str:
    """Fallback when columns 77-78 are missing (older/edited PDB files)."""
    s = name.strip()
    if not s:
        return "C"
    if s[0].isdigit():
        s = s[1:]
    two = s[:2].capitalize()
    if two in ("Cl", "Br", "Zn", "Mg", "Ca", "Na", "Fe", "Mn", "Cu", "Se"):
        return two
    return s[0].upper() if s else "C"


def _perceive_unknown_residues(ms):
    """Geometric hyb/aromatic perception for atoms templates did not cover."""
    unknown = np.flatnonzero(ms.hyb == 0)
    if len(unknown) == 0:
        return
    for i in unknown:
        i = int(i)
        el = str(ms.element[i])
        if el == "H":
            ms.hyb[i] = HYB_SP3
            continue
        hv = ms.heavy_neighbors(i)
        nh = len(ms.attached_hydrogens(i))
        deg = len(hv) + nh
        if deg >= 4:
            ms.hyb[i] = HYB_SP3
        elif deg == 3:
            ms.hyb[i] = HYB_SP2 if _is_planar(ms, i) else HYB_SP3
        elif deg == 2:
            ang = ms.angle(int(ms.neighbors(i)[0]), i, int(ms.neighbors(i)[1]))
            ms.hyb[i] = 1 if ang > 155 else (HYB_SP2 if ang > 115 else HYB_SP3)
        else:
            ms.hyb[i] = HYB_SP3
    # aromatic rings among the newly typed atoms
    for ring, is_ar in zip(ms.rings(), ms.ring_aromatic()):
        if is_ar:
            for a in ring:
                ms.aromatic[a] = True


def _is_planar(ms, i, tol=0.20):
    nb = ms.neighbors(i)
    if len(nb) < 3:
        return False
    p = ms.xyz[nb[:3]]
    c = ms.xyz[i]
    nrm = np.cross(p[1] - p[0], p[2] - p[0])
    norm = np.linalg.norm(nrm)
    if norm < 1e-8:
        return True
    return abs(float(np.dot(c - p[0], nrm / norm))) < tol
