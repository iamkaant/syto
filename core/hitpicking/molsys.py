"""MolSystem: the one in-memory representation shared by every front end.

Deliberate design constraints
-----------------------------
* numpy only. No RDKit, no ChimeraX, no scipy. This is what lets the exact
  same code run inside ChimeraX's Python 3.11 and inside Pyodide.
* **No bond orders.** ChimeraX does not store them, so nothing downstream may
  depend on them. Chemistry is expressed through element + hybridisation +
  aromaticity + formal charge + attached-hydrogen count, which both the
  ChimeraX adapter and the file parsers can supply faithfully.
* Atom order is stable and is the public identifier. Front ends map their own
  atom handles to these indices and back.
"""

from __future__ import annotations

import numpy as np

from . import elements as E

# Hybridisation codes
HYB_UNKNOWN = 0
HYB_SP = 1
HYB_SP2 = 2
HYB_SP3 = 3

# Which molecule an atom belongs to
PART_RECEPTOR = 0
PART_LIGAND = 1
PART_WATER = 2
PART_ION = 3
PART_COFACTOR = 4


class MolSystem:
    """A receptor + ligand(s) + waters in one coordinate frame.

    All per-atom arrays have length ``n_atoms`` and share an index space.
    """

    __slots__ = (
        "xyz", "element", "name", "hyb", "aromatic", "formal_charge",
        "part", "res_name", "res_seq", "chain_id", "bonds", "bond_aromatic",
        "_rings", "_ring_aromatic", "_adj", "_adj_start", "_grid", "meta",
    )

    def __init__(
        self,
        xyz,
        element,
        bonds=None,
        name=None,
        hyb=None,
        aromatic=None,
        formal_charge=None,
        part=None,
        res_name=None,
        res_seq=None,
        chain_id=None,
        meta=None,
    ):
        self.xyz = np.ascontiguousarray(xyz, dtype=np.float64).reshape(-1, 3)
        n = len(self.xyz)
        self.element = np.asarray(element, dtype="<U2")
        if len(self.element) != n:
            raise ValueError("element length %d != n_atoms %d" % (len(self.element), n))

        self.bonds = (
            np.zeros((0, 2), dtype=np.int32)
            if bonds is None
            else np.ascontiguousarray(bonds, dtype=np.int32).reshape(-1, 2)
        )
        if len(self.bonds) and (self.bonds.max() >= n or self.bonds.min() < 0):
            raise ValueError("bond index out of range")

        self.name = _col(name, n, "<U6", "")
        self.hyb = _col(hyb, n, np.int8, HYB_UNKNOWN)
        self.aromatic = _col(aromatic, n, bool, False)
        self.formal_charge = _col(formal_charge, n, np.int8, 0)
        self.part = _col(part, n, np.int8, PART_RECEPTOR)
        self.res_name = _col(res_name, n, "<U4", "")
        self.res_seq = _col(res_seq, n, np.int32, 0)
        self.chain_id = _col(chain_id, n, "<U2", "")
        self.meta = dict(meta or {})

        self._rings = None
        self._ring_aromatic = None
        self._adj = None
        self._adj_start = None
        self._grid = None

    # ---------------------------------------------------------------- basics

    @property
    def n_atoms(self) -> int:
        return len(self.xyz)

    @property
    def is_hydrogen(self) -> np.ndarray:
        return self.element == "H"

    @property
    def heavy(self) -> np.ndarray:
        """Boolean mask of non-hydrogen atoms."""
        return self.element != "H"

    @property
    def is_ligand(self) -> np.ndarray:
        return self.part == PART_LIGAND

    @property
    def is_water(self) -> np.ndarray:
        return self.part == PART_WATER

    def __repr__(self):
        return "<MolSystem %d atoms, %d bonds, %d ligand>" % (
            self.n_atoms, len(self.bonds), int(self.is_ligand.sum()))

    # ------------------------------------------------------------ adjacency

    def _build_adj(self):
        """CSR-style adjacency so neighbours() is O(degree) with no Python dicts."""
        n = self.n_atoms
        if len(self.bonds) == 0:
            self._adj_start = np.zeros(n + 1, dtype=np.int32)
            self._adj = np.zeros(0, dtype=np.int32)
            return
        both = np.concatenate([self.bonds, self.bonds[:, ::-1]])
        order = np.argsort(both[:, 0], kind="stable")
        both = both[order]
        counts = np.bincount(both[:, 0], minlength=n)
        start = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(counts, out=start[1:])
        self._adj_start = start
        self._adj = np.ascontiguousarray(both[:, 1], dtype=np.int32)

    def neighbors(self, i: int) -> np.ndarray:
        if self._adj is None:
            self._build_adj()
        return self._adj[self._adj_start[i]:self._adj_start[i + 1]]

    def degree(self) -> np.ndarray:
        """Total bonded degree (hydrogens included)."""
        if self._adj is None:
            self._build_adj()
        return np.diff(self._adj_start).astype(np.int32)

    def heavy_degree(self) -> np.ndarray:
        if self._adj is None:
            self._build_adj()
        isH = self.is_hydrogen
        out = np.zeros(self.n_atoms, dtype=np.int32)
        for i in range(self.n_atoms):
            nb = self._adj[self._adj_start[i]:self._adj_start[i + 1]]
            out[i] = int((~isH[nb]).sum())
        return out

    def n_hydrogens(self) -> np.ndarray:
        """Count of *explicit* attached hydrogens.

        There is no implicit-H model here. Inputs are expected to carry
        explicit hydrogens; `hitpicking.io` warns when they do not, because
        H-bond geometry and pKBHX relevance both depend on them.
        """
        if self._adj is None:
            self._build_adj()
        isH = self.is_hydrogen
        out = np.zeros(self.n_atoms, dtype=np.int32)
        for i in range(self.n_atoms):
            nb = self._adj[self._adj_start[i]:self._adj_start[i + 1]]
            out[i] = int(isH[nb].sum())
        return out

    def attached_hydrogens(self, i: int) -> np.ndarray:
        nb = self.neighbors(i)
        return nb[self.is_hydrogen[nb]]

    def heavy_neighbors(self, i: int) -> np.ndarray:
        nb = self.neighbors(i)
        return nb[~self.is_hydrogen[nb]]

    # ------------------------------------------------------- neighbour grid

    def _build_grid(self, cell: float = 4.0):
        """Uniform spatial hash. Replaces scipy.cKDTree to keep deps at numpy."""
        xyz = self.xyz
        origin = xyz.min(axis=0) - 1e-6
        idx = np.floor((xyz - origin) / cell).astype(np.int64)
        dims = idx.max(axis=0) + 1
        flat = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
        order = np.argsort(flat, kind="stable")
        self._grid = {
            "cell": cell, "origin": origin, "dims": dims,
            "flat_sorted": flat[order], "order": order,
        }

    def pairs_within(self, cutoff: float, mask_a=None, mask_b=None):
        """All atom pairs (i, j) closer than `cutoff`, with distances.

        Returns (i_idx, j_idx, dist). If both masks are given, i comes from
        mask_a and j from mask_b (the usual ligand-vs-receptor query).
        Self-pairs and duplicates are removed.
        """
        if self._grid is None or self._grid["cell"] < cutoff:
            self._build_grid(max(4.0, cutoff))
        g = self._grid
        cell, origin, dims = g["cell"], g["origin"], g["dims"]
        flat_sorted, order = g["flat_sorted"], g["order"]

        a_idx = np.arange(self.n_atoms) if mask_a is None else np.flatnonzero(mask_a)
        if len(a_idx) == 0:
            return _empty_pairs()
        b_allowed = None if mask_b is None else np.asarray(mask_b, dtype=bool)

        cidx = np.floor((self.xyz - origin) / cell).astype(np.int64)
        out_i, out_j = [], []
        # 27-cell stencil
        offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        for i in a_idx:
            cx, cy, cz = cidx[i]
            cand = []
            for dx, dy, dz in offs:
                x, y, z = cx + dx, cy + dy, cz + dz
                if x < 0 or y < 0 or z < 0 or x >= dims[0] or y >= dims[1] or z >= dims[2]:
                    continue
                key = (x * dims[1] + y) * dims[2] + z
                lo = np.searchsorted(flat_sorted, key, "left")
                hi = np.searchsorted(flat_sorted, key, "right")
                if hi > lo:
                    cand.append(order[lo:hi])
            if not cand:
                continue
            c = np.concatenate(cand)
            if b_allowed is not None:
                c = c[b_allowed[c]]
            c = c[c != i]
            if len(c) == 0:
                continue
            out_i.append(np.full(len(c), i, dtype=np.int64))
            out_j.append(c)
        if not out_i:
            return _empty_pairs()
        ii = np.concatenate(out_i)
        jj = np.concatenate(out_j)
        d = np.linalg.norm(self.xyz[ii] - self.xyz[jj], axis=1)
        keep = d <= cutoff
        ii, jj, d = ii[keep], jj[keep], d[keep]
        if mask_b is None:
            # symmetric query: keep each unordered pair once
            keep = ii < jj
            ii, jj, d = ii[keep], jj[keep], d[keep]
        return ii, jj, d

    # ----------------------------------------------------------- ring search

    def rings(self):
        """Smallest set of smallest rings, as tuples of atom indices.

        Uses a bounded BFS per bond, which is adequate for drug-sized ligands
        and protein residues (ring sizes 3-8).
        """
        if self._rings is None:
            self._rings, self._ring_aromatic = _find_rings(self)
        return self._rings

    def ring_aromatic(self):
        if self._rings is None:
            self.rings()
        return self._ring_aromatic

    def ring_info(self):
        """Per-atom (n_rings, min_ring_size) arrays."""
        rings = self.rings()
        n_r = np.zeros(self.n_atoms, dtype=np.int32)
        min_sz = np.zeros(self.n_atoms, dtype=np.int32)
        for r in rings:
            for a in r:
                n_r[a] += 1
                if min_sz[a] == 0 or len(r) < min_sz[a]:
                    min_sz[a] = len(r)
        return n_r, min_sz

    def ring_plane(self, ring):
        """(centroid, unit normal) by SVD. Used by stacking / cation-pi."""
        pts = self.xyz[list(ring)]
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c)
        return c, vt[2]

    # ---------------------------------------------------------------- geometry

    def distance(self, i, j):
        return float(np.linalg.norm(self.xyz[i] - self.xyz[j]))

    def angle(self, i, j, k):
        """Angle i-j-k in degrees (j is the vertex)."""
        v1 = self.xyz[i] - self.xyz[j]
        v2 = self.xyz[k] - self.xyz[j]
        return _angle_between(v1, v2)

    def dihedral(self, i, j, k, l):
        """Signed torsion i-j-k-l in degrees, range (-180, 180]."""
        p = self.xyz[[i, j, k, l]]
        b0 = p[0] - p[1]
        b1 = p[2] - p[1]
        b2 = p[3] - p[2]
        b1n = b1 / (np.linalg.norm(b1) or 1.0)
        v = b0 - np.dot(b0, b1n) * b1n
        w = b2 - np.dot(b2, b1n) * b1n
        x = np.dot(v, w)
        y = np.dot(np.cross(b1n, v), w)
        return float(np.degrees(np.arctan2(y, x)))

    # ------------------------------------------------------------- selection

    def subset(self, mask):
        """A new MolSystem with only the masked atoms. Bonds are remapped."""
        mask = np.asarray(mask, dtype=bool)
        keep = np.flatnonzero(mask)
        remap = np.full(self.n_atoms, -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        if len(self.bonds):
            bm = mask[self.bonds[:, 0]] & mask[self.bonds[:, 1]]
            nb = remap[self.bonds[bm]]
        else:
            nb = np.zeros((0, 2), dtype=np.int32)
        return MolSystem(
            self.xyz[keep], self.element[keep], nb,
            name=self.name[keep], hyb=self.hyb[keep], aromatic=self.aromatic[keep],
            formal_charge=self.formal_charge[keep], part=self.part[keep],
            res_name=self.res_name[keep], res_seq=self.res_seq[keep],
            chain_id=self.chain_id[keep], meta=dict(self.meta),
        )

    def atom_label(self, i: int) -> str:
        """Human-readable atom id, e.g. 'ASP155.A OD1' or 'LIG N3'."""
        rn = str(self.res_name[i]).strip()
        ch = str(self.chain_id[i]).strip()
        nm = str(self.name[i]).strip() or str(self.element[i])
        if rn:
            res = "%s%d" % (rn, int(self.res_seq[i]))
            if ch:
                res += "." + ch
            return "%s %s" % (res, nm)
        return nm

    def residue_key(self, i: int):
        return (str(self.chain_id[i]), int(self.res_seq[i]), str(self.res_name[i]))


# ------------------------------------------------------------------ helpers

def _col(value, n, dtype, default):
    if value is None:
        return np.full(n, default, dtype=dtype)
    a = np.asarray(value, dtype=dtype)
    if len(a) != n:
        raise ValueError("column length %d != n_atoms %d" % (len(a), n))
    return a


def _empty_pairs():
    z = np.zeros(0, dtype=np.int64)
    return z, z, np.zeros(0, dtype=np.float64)


def _angle_between(v1, v2) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    c = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, c)))))


def _find_rings(ms, max_size: int = 8):
    """Find rings up to `max_size`, keeping the smallest set.

    For each bond, the shortest alternative path between its two atoms (with
    that bond removed) closes a ring. Collecting these and de-duplicating gives
    the SSSR in practice for the fused systems that occur in drug-like ligands.
    """
    if ms._adj is None:
        ms._build_adj()
    heavy = ms.heavy
    seen = set()
    rings = []
    for b in range(len(ms.bonds)):
        a0, a1 = int(ms.bonds[b, 0]), int(ms.bonds[b, 1])
        if not (heavy[a0] and heavy[a1]):
            continue
        path = _shortest_path_excluding_bond(ms, a0, a1, heavy, max_size)
        if path is None:
            continue
        key = frozenset(path)
        if key in seen:
            continue
        seen.add(key)
        rings.append(tuple(path))
    rings.sort(key=len)
    arom = [_ring_is_aromatic(ms, r) for r in rings]
    return rings, arom


def _shortest_path_excluding_bond(ms, start, goal, heavy, max_size):
    """BFS from start to goal without using the direct start-goal bond."""
    from collections import deque
    prev = {start: -1}
    q = deque([(start, 0)])
    while q:
        cur, depth = q.popleft()
        if depth >= max_size - 1:
            continue
        for nb in ms.neighbors(cur):
            nb = int(nb)
            if not heavy[nb]:
                continue
            if cur == start and nb == goal:
                continue          # the bond we are excluding
            if nb == goal:
                path = [goal, cur]
                p = prev[cur]
                while p != -1:
                    path.append(p)
                    p = prev[p]
                return path if len(path) >= 3 else None
            if nb not in prev:
                prev[nb] = cur
                q.append((nb, depth + 1))
    return None


def _ring_is_aromatic(ms, ring) -> bool:
    """A ring is aromatic if the parser/adapter flagged all its atoms so.

    We do not re-derive aromaticity from bond orders: we do not have them.
    mol2 Sybyl types, SDF aromatic flags and ChimeraX's own ring.aromatic all
    supply this directly. Falls back to a planarity + sp2 test when the flags
    are absent entirely.
    """
    idx = list(ring)
    if ms.aromatic[idx].all():
        return True
    if ms.aromatic[idx].any():
        return False
    if len(ring) not in (5, 6):
        return False
    if not (ms.hyb[idx] == HYB_SP2).all():
        return False
    # planarity: max deviation from the best-fit plane
    c, nrm = ms.ring_plane(ring)
    dev = np.abs((ms.xyz[idx] - c) @ nrm)
    return bool(dev.max() < 0.30)
